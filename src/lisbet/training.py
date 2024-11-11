"""Training and fitting functions for LISBET.

Notes
-----
[a] The dictionary of RNG seed could be refactored as a dataclass in the future.

[b] The train/dev split is performed here and not in the input_pipeline module to
    emphasize that the test set is frozen and won't be used for hyper-parameters tuning.

[c] When mixing datasets of different lengths, the training and evaluation loops will
    stop after exhausting the shortest one. Please consider using random sampling.

"""

import logging
import re
from collections import defaultdict
from datetime import datetime
from functools import partial
from itertools import repeat
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torchinfo import summary
from torchvision import transforms
from tqdm.auto import tqdm

from . import input_pipeline, modeling
from .datasets import load_records


def binary_loss(inputs, targets):
    inputs = inputs.view(-1)
    targets = targets.float()
    return torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets)


class RandomXYSwap:
    """Random transformation spapping x and y coordinates"""

    def __init__(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample):
        transformed_sample = (
            np.stack((sample[:, 1::2], sample[:, ::2]), axis=2).reshape(sample.shape)
            if self.rng.random() < 0.5
            else sample
        )
        return transformed_sample


class Trainer:
    """LISBET trainer."""

    def __init__(self, config):
        self.config = config

        # Configure base runtime arguments
        self.task_ids = self.config["task_ids"].split(",")
        self.run_id = (
            self.config["run_id"]
            if self.config["run_id"] is not None
            else datetime.now().strftime("%Y%m%d%H%M%S")
        )

        # Configure accelerator
        if torch.cuda.is_available():
            self.device_type = "cuda"
        elif torch.mps.is_available():
            self.device_type = "mps"
        else:
            self.device_type = "cpu"
        self.device = torch.device(self.device_type)

        # Configure RNGs
        self.run_seeds = self.generate_seeds()
        torch.manual_seed(self.run_seeds["torch"])

        # Load records
        self.train_rec, self.test_rec, self.dev_rec = self.load_records()

        # Determine data shape from first record
        self.bp_dim = self.train_rec[self.task_ids[0]][0][1]["keypoints"].shape[1]

        # Determine max sequence length
        # TODO: Find a better way to compute max_len or fix in the embedder exporter
        self.max_len = (
            2 * self.config["window_size"]
            if "nwp" in self.task_ids or "load_backbone_weights" in self.config
            else self.config["window_size"]
        )

        # Compute backbone output token idx
        if self.config["window_size"] > self.config["window_offset"] >= 0:
            self.output_token_idx = -(self.config["window_offset"] + 1)
            logging.debug("Output token IDX = %d", self.output_token_idx)
        else:
            raise RuntimeError(
                "Window offset must be a positive integer smaller than the window size"
                f" or zero, got {self.config['window_offset']}."
            )

        # Configure tasks
        self.tasks = self.configure_tasks()

        # Configure model
        self.model = self.configure_model()
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.config["mixed_precision"]
        )
        summary(self.model)

        # Configure optimizer
        self.optimizer = torch.optim.Adamax(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config["learning_rate"],
        )

    def generate_seeds(self):
        # Generate multiple seeds from the base one
        rng = np.random.default_rng(self.config["seed"])
        seed_keys = (
            ["torch", "dev_split", "test_split"]
            + [
                f"{group}_shuffle_{task_id}"
                for task_id in self.task_ids
                for group in ("train", "dev", "test")
            ]
            + [f"transform_{task_id}" for task_id in self.task_ids]
            + [f"dataset_{task_id}" for task_id in self.task_ids if task_id != "cfc"]
        )
        run_seeds = {sk: rng.integers(low=0, high=2**32) for sk in seed_keys}

        # Override test_split seed if needed
        # NOTE: This prevents test set spillover during HP tuning
        if self.config["seed_test_split"] is not None:
            run_seeds["test_split"] = self.config["seed_test_split"]
            logging.debug("Overriding test set split seed")

        logging.debug("Generated seeds: %s", run_seeds)

        return run_seeds

    def load_records(self):
        # Identify all data sources
        datasets = self.config["data_format"].split(",")
        datapaths = self.config["data_path"].split(",")
        if len(datasets) == len(datapaths):
            datasources = list(zip(datasets, datapaths))
        elif len(datapaths) == 1:
            datasources = list(zip(datasets, repeat(datapaths[0])))
        else:
            raise ValueError(
                "Input arguments datasets and datapaths must have the same length, or"
                " datapath must be a single element."
            )
        logging.debug(datasources)

        # Build task to data mapping, by default use all data for every task
        task_data = {
            task_id: list(range(len(datasources))) for task_id in self.task_ids
        }

        # Update task to data mapping, if requested
        if self.config["task_data"] is not None:
            logging.debug("Updating task to data mapping")
            pattern = r"(\b(?:" + r"|".join(self.task_ids) + r")\b):(\[(.*?)\])"
            matches = re.findall(pattern, self.config["task_data"])
            task_data |= {
                key: [int(x) for x in vals.split(",")] for key, _, vals in matches
            }
        logging.debug(task_data)

        # Load records
        records = [
            load_records(
                dataset,
                datapath,
                self.config["data_filter"],
                self.config["dev_ratio"],
                self.config["test_ratio"],
                self.run_seeds["dev_split"],
                self.run_seeds["test_split"],
            )
            for dataset, datapath in datasources
        ]

        # Create the lists of records for each task
        train_rec = defaultdict(list)
        test_rec = defaultdict(list)
        dev_rec = defaultdict(list)

        # Assign records
        for task_id, dataidx_lst in task_data.items():
            for dataidx in dataidx_lst:
                train_rec[task_id].extend(records[dataidx][0])
                if (rec := records[dataidx][1]) is not None:
                    test_rec[task_id].extend(rec)
                if (rec := records[dataidx][2]) is not None:
                    dev_rec[task_id].extend(rec)
                logging.info(
                    "Assigning records from dataset no. %d to task %s", dataidx, task_id
                )
            logging.info("Final training set size = %d", len(train_rec[task_id]))
            logging.debug([key for key, _ in train_rec[task_id]])

            logging.info("Final test set size = %d", len(test_rec[task_id]))
            logging.debug([key for key, _ in test_rec[task_id]])

            logging.info("Final dev set size = %d", len(dev_rec[task_id]))
            logging.debug([key for key, _ in dev_rec[task_id]])

        return train_rec, test_rec, dev_rec

    def _configure_frame_classification(self):
        if "annotations" not in self.train_rec["cfc"][0][1]:
            raise RuntimeError("The provided dataset does not contain annotations.")

        # Find number of behaviors in the training set
        labels = np.concatenate(
            [data["annotations"] for _, data in self.train_rec["cfc"]]
        )
        classes = np.unique(labels)
        num_classes = len(classes)
        np.testing.assert_array_equal(classes, np.array(range(num_classes)))

        # Create classification head
        head = modeling.ClassificationHead(
            output_token_idx=self.output_token_idx,
            emb_dim=self.config["emb_dim"],
            out_dim=num_classes,
            hidden_dim=self.config["hidden_dim"],
        )

        # Compute class weight
        class_weight = torch.Tensor(
            compute_class_weight("balanced", classes=classes, y=labels)
        )
        logging.debug("Class weights: %s", class_weight)

        # Create loss
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight.to(self.device))

        # Create data transformers
        train_transform = (
            transforms.Compose(
                [RandomXYSwap(self.run_seeds["transform_cfc"]), torch.Tensor]
            )
            if self.config["data_augmentation"]
            else transforms.Compose([torch.Tensor])
        )
        eval_transform = transforms.Compose([torch.Tensor])

        # Create dataloaders
        datasets = {
            key: input_pipeline.FrameClassificationDataset(
                records=rec,
                window_size=self.config["window_size"],
                window_offset=self.config["window_offset"],
                transform=transform,
                num_classes=num_classes,
            )
            for key, rec, transform in [
                ("train", self.train_rec["cfc"], train_transform),
                ("dev", self.dev_rec["cfc"], eval_transform),
                ("test", self.test_rec["cfc"], eval_transform),
            ]
            if rec
        }

        # Metric
        include_labels = (
            (0, 1, 2) if "CalMS21_Task1" in self.config["data_format"] else None
        )
        metric = partial(f1_score, labels=include_labels, average="macro")
        metric.__name__ = "macro_f1_score"
        logging.debug("Using labels %s to compute F1 score in CFC task", include_labels)

        # Create task
        # NOTE: This could become a dataclass
        task = {
            "task_id": "cfc",
            "head": head,
            "out_dim": num_classes,
            "criterion": criterion,
            "datasets": datasets,
            "resample": False,
            "predictor": lambda output: torch.max(output, 1)[1],
            "metric": metric,
        }

        return task

    def _configure_selfsupervised_classification(self, task_id):
        # Create classification head
        head = modeling.ClassificationHead(
            output_token_idx=None,
            emb_dim=self.config["emb_dim"],
            out_dim=1,
            hidden_dim=self.config["hidden_dim"],
        )

        # Create loss
        criterion = binary_loss

        # Create data transformers
        train_transform = (
            transforms.Compose(
                [RandomXYSwap(self.run_seeds[f"transform_{task_id}"]), torch.Tensor]
            )
            if self.config["data_augmentation"]
            else transforms.Compose([torch.Tensor])
        )
        eval_transform = transforms.Compose([torch.Tensor])

        # Create dataloaders
        task_map = {
            "nwp": input_pipeline.NextWindowPredictionDataset,
            "smp": input_pipeline.SwapMousePredictionDataset,
            "vsp": input_pipeline.VideoSpeedPredictionDataset,
            "dmp": input_pipeline.DelayMousePredictionDataset,
        }
        datasets = {
            key: task_map[task_id](
                records=rec,
                window_size=self.config["window_size"],
                window_offset=self.config["window_offset"],
                transform=transform,
                seed=self.run_seeds[f"dataset_{task_id}"],
            )
            for key, rec, transform in [
                ("train", self.train_rec[task_id], train_transform),
                ("dev", self.dev_rec[task_id], eval_transform),
                ("test", self.test_rec[task_id], eval_transform),
            ]
            if rec
        }

        # Create task
        # NOTE: This could become a dataclass
        task = {
            "task_id": task_id,
            "head": head,
            "out_dim": 1,
            "criterion": criterion,
            "datasets": datasets,
            "resample": True,
            "predictor": lambda output: output > 0.0,
            "metric": accuracy_score,
        }

        return task

    def configure_tasks(self):
        tasks = []
        for task_id in self.task_ids:
            if task_id == "cfc":
                tasks.append(self._configure_frame_classification())
            elif task_id in ("nwp", "smp", "vsp", "dmp"):
                tasks.append(self._configure_selfsupervised_classification(task_id))
            elif task_id in ("vsr", "dmr"):
                raise NotImplementedError(f"Task {task_id} pending")
            else:
                raise ValueError(f"Unknown task {task_id}")

        return tasks

    def configure_model(self):
        # Create model
        model = modeling.MultiTaskModel(
            modeling.Backbone(
                bp_dim=self.bp_dim,
                emb_dim=self.config["emb_dim"],
                hidden_dim=self.config["hidden_dim"],
                num_heads=self.config["num_heads"],
                num_layers=self.config["num_layers"],
                max_len=self.max_len,
            ),
            {task["task_id"]: task["head"] for task in self.tasks},
        ).to(self.device)

        if self.config["compile_model"]:
            model = torch.compile(model)

        if self.config["load_backbone_weights"]:
            # Load pretrained weights
            incompatible_layers = model.load_state_dict(
                torch.load(self.config["load_backbone_weights"], weights_only=True),
                strict=False,
            )
            logging.info(
                "Loaded weights from file.\nMissing keys: %s\nUnexpected keys: %s",
                incompatible_layers.missing_keys,
                incompatible_layers.unexpected_keys,
            )

        if self.config["freeze_backbone_weights"]:
            for param in model.backbone.parameters():
                param.requires_grad = False

        return model

    def configure_dataloaders(self, group):
        # Estimate number of samples
        num_samples = min(len(task["datasets"][group]) for task in self.tasks)
        if self.config[f"{group}_sample"] is not None:
            num_samples = int(num_samples * self.config[f"{group}_sample"])
        logging.info("Using %d samples from the %s group", num_samples, group)

        # Create a dataloader for each task
        dataloaders = []
        for task in self.tasks:
            # Create new sample, if requested
            # NOTE: This has a regularization effect in self-supervised training
            if task["resample"]:
                task["datasets"][group].resample_dataset()

            sampler = torch.utils.data.RandomSampler(
                task["datasets"][group], num_samples=num_samples
            )
            dataloader = torch.utils.data.DataLoader(
                task["datasets"][group],
                batch_size=self.config["batch_size"],
                sampler=sampler,
                num_workers=1,
                pin_memory=True,
            )
            dataloaders.append(dataloader)

        return dataloaders

    def train_epoch(self, dataloaders):
        self.model.train()

        # Logging
        losses = defaultdict(list)
        labels = defaultdict(list)
        predictions = defaultdict(list)

        # Iterate over all batches
        for batch_data in tqdm(zip(*dataloaders)):
            self.optimizer.zero_grad()
            batch_losses = []

            # Iterate over all tasks
            for task, (data, target) in zip(self.tasks, batch_data):
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                with torch.autocast(
                    device_type=self.device_type,
                    dtype=torch.float16,
                    enabled=self.config["mixed_precision"],
                ):
                    output = self.model(data, task["task_id"])
                    loss = task["criterion"](output, target)
                    predicted = task["predictor"](output)

                batch_losses.append(loss)

                # Store loss value for stats
                losses[task["task_id"]].append(loss.item())
                labels[task["task_id"]].append(target.detach().cpu().numpy())
                predictions[task["task_id"]].append(predicted.detach().cpu().numpy())

            total_loss = sum(self.scaler.scale(loss) for loss in batch_losses)
            total_loss.backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return losses, labels, predictions

    def evaluate(self, dataloaders):
        self.model.eval()

        # Logging
        losses = defaultdict(list)
        labels = defaultdict(list)
        predictions = defaultdict(list)

        with torch.no_grad():
            # Iterate over all batches
            for batch_data in tqdm(zip(*dataloaders)):
                # Iterate over all tasks
                for task, (data, target) in zip(self.tasks, batch_data):
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)

                    with torch.autocast(
                        device_type=self.device_type,
                        dtype=torch.float16,
                        enabled=self.config["mixed_precision"],
                    ):
                        output = self.model(data, task["task_id"])
                        loss = task["criterion"](output, target)
                        predicted = task["predictor"](output)

                    # Store loss value for stats
                    losses[task["task_id"]].append(loss.item())
                    labels[task["task_id"]].append(target.cpu().numpy())
                    predictions[task["task_id"]].append(predicted.cpu().numpy())

        return losses, labels, predictions

    def compute_epoch_logs(self, group_id, losses, labels, predictions):
        epoch_log = {}
        for task in self.tasks:
            # Compute metrics
            metric_name = f"{task['task_id']}_{group_id}_{task['metric'].__name__}"
            epoch_log[metric_name] = task["metric"](
                np.concatenate(labels[task["task_id"]]),
                np.concatenate(predictions[task["task_id"]]),
            )
            # Compute mean losses
            loss_name = f"{task['task_id']}_{group_id}_loss"
            epoch_log[loss_name] = np.mean(losses[task["task_id"]])

        return epoch_log

    def save_weights(self, filename):
        weights_path = (
            Path(self.config["output_path"])
            / "models"
            / self.run_id
            / "weights"
            / filename
        )
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), weights_path)

    def save_model_config(self):
        model_config = {
            "window_size": self.config["window_size"],
            "window_offset": self.config["window_offset"],
            "output_token_idx": self.output_token_idx,
            "bp_dim": self.bp_dim,
            "emb_dim": self.config["emb_dim"],
            "hidden_dim": self.config["hidden_dim"],
            "num_heads": self.config["num_heads"],
            "num_layers": self.config["num_layers"],
            "max_len": self.max_len,
            "out_dim": {task["task_id"]: task["out_dim"] for task in self.tasks},
        }
        model_path = (
            Path(self.config["output_path"])
            / "models"
            / self.run_id
            / "model_config.yml"
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "w", encoding="utf-8") as f_yaml:
            yaml.safe_dump(model_config, f_yaml)

    def save_history(self, history):
        history_path = (
            Path(self.config["output_path"])
            / "models"
            / self.run_id
            / "training_history.log"
        )
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history = pd.DataFrame.from_dict(history)
        history.to_csv(history_path)

    def train(self):
        """Train model."""
        # Save model config
        self.save_model_config()

        history = []

        for epoch in range(self.config["epochs"]):
            history_entry = {"epoch": epoch}
            print(f"Epoch {epoch}")

            # Get dataloaders
            train_dataloaders = self.configure_dataloaders("train")

            # Run training epoch
            losses, labels, predictions = self.train_epoch(train_dataloaders)

            # Get epoch logs
            train_log = self.compute_epoch_logs("train", losses, labels, predictions)
            logging.info(train_log)

            # Update history entry for current epoch
            history_entry.update(train_log)

            # Save weights, if requested
            if self.config["save_weights"] == "all":
                self.save_weights(f"weights_epoch{epoch}.pt")

            if self.config["dev_ratio"] is not None:
                # Get dataloaders
                dev_dataloaders = self.configure_dataloaders("dev")

                # Run dev epoch
                losses, labels, predictions = self.evaluate(dev_dataloaders)

                # Get epoch logs
                dev_log = self.compute_epoch_logs("dev", losses, labels, predictions)
                logging.info(dev_log)

                # Update history entry for current epoch
                history_entry.update(dev_log)

            # Update history
            history.append(history_entry)

        # Save final weights, if requested
        if self.config["save_weights"] == "last":
            self.save_weights("weights_last.pt")

        # Save history, if requested
        if self.config["save_history"]:
            self.save_history(history)


def train(
    # Basic params
    data_format: str = "CalMS21_Task1",
    data_path: str = "datasets/CalMS21",
    data_filter: Optional[str] = None,
    window_size: int = 200,
    window_offset: int = 0,
    fps_scaling: float = 1.0,
    test_ratio: Optional[float] = None,
    # Training params
    epochs: int = 10,
    batch_size: int = 32,
    task_ids: str = "cfc",
    task_data: Optional[str] = None,
    seed: int = 1991,
    seed_test_split: Optional[int] = None,
    run_id: Optional[str] = None,
    data_augmentation: bool = False,
    train_sample: Optional[float] = None,
    dev_sample: Optional[float] = None,
    dev_ratio: Optional[float] = None,
    # Model architecture
    num_layers: int = 4,
    emb_dim: int = 32,
    num_heads: int = 4,
    hidden_dim: int = 128,
    learning_rate: float = 1e-4,
    # Model weights and saving
    load_backbone_weights: Optional[Path] = None,
    freeze_backbone_weights: bool = False,
    save_weights: Optional[str] = None,
    save_history: bool = False,
    output_path: Path = Path("."),
    # Performance options
    mixed_precision: bool = False,
    compile_model: bool = False,
) -> None:
    """Train a LISBET model.

    All parameters match the CLI arguments exactly.
    """
    # TODO: Drop the Trainer, it lost most of its usefulness after refactoring the CLI
    trainer = Trainer(locals())
    trainer.train()
