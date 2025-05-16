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
    """Random transformation swapping x and y coordinates"""

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


def _generate_seeds(seed, task_ids, seed_test_split):
    """Internal helper. Generates multiple seeds from the base one."""
    rng = np.random.default_rng(seed)
    seed_keys = (
        ["torch", "dev_split", "test_split"]
        + [
            f"{group}_shuffle_{task_id}"
            for task_id in task_ids
            for group in ("train", "dev", "test")
        ]
        + [f"transform_{task_id}" for task_id in task_ids]
        + [f"dataset_{task_id}" for task_id in task_ids if task_id != "cfc"]
    )
    run_seeds = {sk: rng.integers(low=0, high=2**32) for sk in seed_keys}

    # Override test_split seed if needed
    # NOTE: This prevents test set spillover during HP tuning
    if seed_test_split is not None:
        run_seeds["test_split"] = seed_test_split
        logging.debug("Overriding test set split seed")

    logging.debug("Generated seeds: %s", run_seeds)

    return run_seeds


def _load_records(
    data_format,
    data_path,
    data_scale,
    data_filter,
    dev_ratio,
    test_ratio,
    dev_seed,
    test_seed,
    keypoints_subset,
    task_ids,
    task_data,
):
    """Internal helper. Loads and splits records for all tasks."""
    datasets = data_format.split(",")
    datapaths = data_path.split(",")
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
    task_data_map = {task_id: list(range(len(datasources))) for task_id in task_ids}

    # Update task to data mapping, if requested
    if task_data is not None:
        logging.debug("Updating task to data mapping")
        pattern = r"(\b(?:" + r"|".join(task_ids) + r")\b):(\[(.*?)\])"
        matches = re.findall(pattern, task_data)
        task_data_map |= {
            key: [int(x) for x in vals.split(",")] for key, _, vals in matches
        }
    logging.debug(task_data_map)

    # Load records
    records = [
        load_records(
            dataset,
            datapath,
            data_scale=data_scale,
            data_filter=data_filter,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
            dev_seed=dev_seed,
            test_seed=test_seed,
            keypoints_subset=keypoints_subset,
        )
        for dataset, datapath in datasources
    ]

    # Sanity check: All posetracks must have the same 'features' coordinate across
    #               datasets. As consistency within a dataset is already checked, we
    #               only need to check the first record of each dataset against the
    #               others.
    main_features = [
        recs["main_records"][0][1]["posetracks"].coords["features"].values.tolist()
        for recs in records
    ]
    ref_features = main_features[0]
    for i, features in enumerate(main_features):
        if features != ref_features:
            raise ValueError(
                f"Inconsistent posetracks coordinates in loaded records, dataset {i}:\n"
                f"Reference features:\n{ref_features}\n"
                f"Record features:\n{features}"
            )

    # Create the lists of records for each task
    train_rec = defaultdict(list)
    test_rec = defaultdict(list)
    dev_rec = defaultdict(list)

    # Assign records
    for task_id, dataidx_lst in task_data_map.items():
        for dataidx in dataidx_lst:
            train_rec[task_id].extend(records[dataidx]["main_records"])
            if "test_records" in records[dataidx]:
                test_rec[task_id].extend(records[dataidx]["test_records"])
            if "dev_records" in records[dataidx]:
                dev_rec[task_id].extend(records[dataidx]["dev_records"])
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


def _configure_classification_task(
    train_rec,
    dev_rec,
    test_rec,
    window_size,
    window_offset,
    emb_dim,
    hidden_dim,
    data_augmentation,
    run_seeds,
    device,
    data_format,
):
    """Internal helper. Configures the classification task."""
    if "annotations" not in train_rec["cfc"][0][1]:
        raise RuntimeError("The provided dataset does not contain annotations.")

    # Find number of behaviors in the training set
    labels = np.concatenate(
        [data["annotations"].label_cat for _, data in train_rec["cfc"]]
    )
    classes = np.unique(labels)
    num_classes = len(classes)
    np.testing.assert_array_equal(classes, np.array(range(num_classes)))

    # Create classification head
    head = modeling.ClassificationHead(
        output_token_idx=-(window_offset + 1),
        emb_dim=emb_dim,
        out_dim=num_classes,
        hidden_dim=hidden_dim,
    )

    # Compute class weight
    class_weight = torch.Tensor(
        compute_class_weight("balanced", classes=classes, y=labels)
    )
    logging.debug("Class weights: %s", class_weight)

    # Create loss
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight.to(device))

    # Create data transformers
    train_transform = (
        transforms.Compose([RandomXYSwap(run_seeds["transform_cfc"]), torch.Tensor])
        if data_augmentation
        else transforms.Compose([torch.Tensor])
    )
    eval_transform = transforms.Compose([torch.Tensor])

    # Create dataloaders
    datasets = {
        key: input_pipeline.FrameClassificationDataset(
            records=rec,
            window_size=window_size,
            window_offset=window_offset,
            transform=transform,
            num_classes=num_classes,
        )
        for key, rec, transform in [
            ("train", train_rec["cfc"], train_transform),
            ("dev", dev_rec["cfc"], eval_transform),
            ("test", test_rec["cfc"], eval_transform),
        ]
        if rec
    }

    # Metric
    metric = partial(f1_score, average="macro")
    metric.__name__ = "macro_f1_score"

    # Create task
    # NOTE: This could become a dataclass
    task = {
        "task_id": "cfc",
        "head": head,
        "out_dim": num_classes,
        "criterion": criterion,
        "datasets": datasets,
        "resample": False,
        "predictor": lambda output: torch.argmax(output, dim=1),
        "metric": metric,
    }

    return task


def _configure_selfsupervised_task(
    task_id,
    train_rec,
    dev_rec,
    test_rec,
    window_size,
    window_offset,
    emb_dim,
    hidden_dim,
    data_augmentation,
    run_seeds,
):
    """Internal helper. Configures a self-supervised task."""
    # Create classification head
    head = modeling.ClassificationHead(
        output_token_idx=None,
        emb_dim=emb_dim,
        out_dim=1,
        hidden_dim=hidden_dim,
    )

    # Create loss
    criterion = binary_loss

    # Create data transformers
    train_transform = (
        transforms.Compose(
            [RandomXYSwap(run_seeds[f"transform_{task_id}"]), torch.Tensor]
        )
        if data_augmentation
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
            window_size=window_size,
            window_offset=window_offset,
            transform=transform,
            seed=run_seeds[f"dataset_{task_id}"],
        )
        for key, rec, transform in [
            ("train", train_rec[task_id], train_transform),
            ("dev", dev_rec[task_id], eval_transform),
            ("test", test_rec[task_id], eval_transform),
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


def _configure_tasks(
    train_rec,
    dev_rec,
    test_rec,
    task_ids,
    window_size,
    window_offset,
    emb_dim,
    hidden_dim,
    data_augmentation,
    run_seeds,
    device,
    data_format,
):
    """Internal helper. Configures all tasks."""
    tasks = []
    for task_id in task_ids:
        if task_id == "cfc":
            tasks.append(
                _configure_classification_task(
                    train_rec,
                    dev_rec,
                    test_rec,
                    window_size,
                    window_offset,
                    emb_dim,
                    hidden_dim,
                    data_augmentation,
                    run_seeds,
                    device,
                    data_format,
                )
            )
        elif task_id in ("nwp", "smp", "vsp", "dmp"):
            tasks.append(
                _configure_selfsupervised_task(
                    task_id,
                    train_rec,
                    dev_rec,
                    test_rec,
                    window_size,
                    window_offset,
                    emb_dim,
                    hidden_dim,
                    data_augmentation,
                    run_seeds,
                )
            )
        else:
            raise ValueError(f"Unknown task {task_id}")

    return tasks


def _build_model(
    bp_dim,
    emb_dim,
    hidden_dim,
    num_heads,
    num_layers,
    max_len,
    tasks,
    compile_model,
    load_backbone_weights,
    freeze_backbone_weights,
    device,
):
    """Internal helper. Builds the LISBET model."""
    model = modeling.MultiTaskModel(
        modeling.Backbone(
            bp_dim=bp_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_len=max_len,
        ),
        {task["task_id"]: task["head"] for task in tasks},
    ).to(device)

    if compile_model:
        model = torch.compile(model)

    if load_backbone_weights:
        incompatible_layers = model.load_state_dict(
            torch.load(load_backbone_weights, weights_only=True),
            strict=False,
        )
        logging.info(
            "Loaded weights from file.\nMissing keys: %s\nUnexpected keys: %s",
            incompatible_layers.missing_keys,
            incompatible_layers.unexpected_keys,
        )

    if freeze_backbone_weights:
        for param in model.backbone.parameters():
            param.requires_grad = False
    return model


def _configure_optimizer_and_scheduler(model, learning_rate, mixed_precision):
    """Internal helper. Configures optimizer, scheduler, and scaler."""
    # Configure optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    # Configure LR (warmup scheduler)
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-2,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    # Configure LR (main scheduler)
    T_0 = 10
    T_mult = 2
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult
    )

    # Configure final LR scheduler
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    # Configure mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    return optimizer, scheduler, scaler


def _configure_dataloaders(tasks, group, batch_size, group_sample):
    """Internal helper. Configures dataloaders for a group."""
    # Estimate number of samples
    num_samples = min(len(task["datasets"][group]) for task in tasks)
    if group_sample is not None:
        num_samples = int(num_samples * group_sample)
    logging.info("Using %d samples from the %s group", num_samples, group)

    # Create a dataloader for each task
    dataloaders = []
    for task in tasks:
        # Create new sample, if requested
        # NOTE: This has a regularization effect in self-supervised training
        if task["resample"]:
            task["datasets"][group].resample_dataset()

        sampler = torch.utils.data.RandomSampler(
            task["datasets"][group], num_samples=num_samples
        )
        dataloader = torch.utils.data.DataLoader(
            task["datasets"][group],
            batch_size=batch_size,
            sampler=sampler,
            num_workers=1,
            pin_memory=True,
        )
        dataloaders.append(dataloader)

    return dataloaders


def _train_one_epoch(
    model, dataloaders, optimizer, scheduler, scaler, tasks, device, mixed_precision
):
    """Internal helper. Runs one training epoch."""
    model.train()

    # Logging
    losses = defaultdict(list)
    labels = defaultdict(list)
    predictions = defaultdict(list)

    # Iterate over all batches
    for batch_data in tqdm(zip(*dataloaders)):
        optimizer.zero_grad()
        batch_losses = []

        # Iterate over all tasks
        for task, (data, target) in zip(tasks, batch_data):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision,
            ):
                output = model(data, task["task_id"])
                loss = task["criterion"](output, target)
                predicted = task["predictor"](output)

            batch_losses.append(loss)

            # Store loss value for stats
            losses[task["task_id"]].append(loss.item())
            labels[task["task_id"]].append(target.detach().cpu().numpy())
            predictions[task["task_id"]].append(predicted.detach().cpu().numpy())

        total_loss = sum(scaler.scale(loss) for loss in batch_losses)
        total_loss.backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()

    return losses, labels, predictions


def _evaluate(model, dataloaders, tasks, device, mixed_precision):
    """Internal helper. Evaluates model on a group."""
    model.eval()

    # Logging
    losses = defaultdict(list)
    labels = defaultdict(list)
    predictions = defaultdict(list)

    with torch.no_grad():
        # Iterate over all batches
        for batch_data in tqdm(zip(*dataloaders)):
            # Iterate over all tasks
            for task, (data, target) in zip(tasks, batch_data):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=mixed_precision,
                ):
                    output = model(data, task["task_id"])
                    loss = task["criterion"](output, target)
                    predicted = task["predictor"](output)

                # Store loss value for stats
                losses[task["task_id"]].append(loss.item())
                labels[task["task_id"]].append(target.cpu().numpy())
                predictions[task["task_id"]].append(predicted.cpu().numpy())

    return losses, labels, predictions


def _compute_epoch_logs(group_id, tasks, losses, labels, predictions):
    """Internal helper. Computes metrics and mean losses for an epoch."""
    epoch_log = {}
    for task in tasks:
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


def _save_weights(model, output_path, run_id, filename):
    """Internal helper. Saves model weights."""
    weights_path = Path(output_path) / "models" / run_id / "weights" / filename
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)


def _save_model_config(
    output_path,
    run_id,
    window_size,
    window_offset,
    output_token_idx,
    bp_dim,
    emb_dim,
    hidden_dim,
    num_heads,
    num_layers,
    max_len,
    tasks,
    input_features,
):
    """Internal helper. Saves model config."""
    model_config = {
        "window_size": window_size,
        "window_offset": window_offset,
        "output_token_idx": output_token_idx,
        "bp_dim": bp_dim,
        "emb_dim": emb_dim,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "max_len": max_len,
        "out_dim": {task["task_id"]: task["out_dim"] for task in tasks},
        "input_features": input_features,
    }
    model_path = Path(output_path) / "models" / run_id / "model_config.yml"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f_yaml:
        yaml.safe_dump(model_config, f_yaml)


def _save_history(output_path, run_id, history):
    """Internal helper. Saves training history."""
    history_path = Path(output_path) / "models" / run_id / "training_history.log"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame.from_dict(history)
    history.to_csv(history_path)


def train(
    # Data parameters
    data_format: str = "CalMS21_Task1",
    data_path: str = "datasets/CalMS21",
    data_scale: Optional[str] = None,
    data_filter: Optional[str] = None,
    keypoints_subset: Optional[str] = None,
    window_size: int = 200,
    window_offset: int = 0,
    fps_scaling: float = 1.0,
    test_ratio: Optional[float] = None,
    dev_ratio: Optional[float] = None,
    train_sample: Optional[float] = None,
    dev_sample: Optional[float] = None,
    # Training parameters
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 1991,
    seed_test_split: Optional[int] = None,
    run_id: Optional[str] = None,
    data_augmentation: bool = False,
    # Task parameters
    task_ids: str = "cfc",
    task_data: Optional[str] = None,
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
) -> torch.nn.Module:
    """
    Train a LISBET model.

    This function orchestrates the full training pipeline for LISBET, including
    data loading, model construction, training, evaluation, and saving artifacts.
    All parameters match the CLI arguments exactly.

    Parameters
    ----------
    data_format : str, default="CalMS21_Task1"
        Dataset format or identifier.
    data_path : str, default="datasets/CalMS21"
        Path to the root directory of the dataset.
    data_scale : str or None, optional
        If supplied as WIDTHxHEIGHT or WIDTHxHEIGHTxDEPTH, every input coordinate is
        assumed to be in data units and is divided by the given scale to obtain
        normalized coordinates in the range [0, 1]. Otherwise, the algorithm infers the
        active extent directly from the data.
    data_filter : str or None, optional
        Comma-separated substrings; a record is kept if any substring occurs in its
        relative path. By default, all records are kept.
    keypoints_subset : str or None, optional
        Optional subset string in the format 'INDIVS;COORDS;PARTS', where each field is
        a comma-separated list or '*' for all. If None, all data is loaded.
    window_size : int, default=200
        Number of frames to consider at each time.
    window_offset : int, default=0
        Window offset for classification tasks.
    fps_scaling : float, default=1.0
        FPS scaling factor.
    test_ratio : float or None, optional
        Fraction of all records to devote to the test set. If None, no test split is
        performed.
    dev_ratio : float or None, optional
        Fraction of the training set to hold out for hyper-parameter tuning. If None,
        no dev split is performed.
    train_sample : float or None, optional
        Fraction of samples from the training set to use.
    dev_sample : float or None, optional
        Fraction of samples from the dev set to use.
    epochs : int, default=10
        Number of training epochs.
    batch_size : int, default=32
        Batch size for training.
    seed : int, default=1991
        Base random seed.
    seed_test_split : int or None, optional
        Random seed for test set split.
    run_id : str or None, optional
        ID of the run. If None, a timestamp is used.
    data_augmentation : bool, default=False
        Enable data augmentation.
    task_ids : str, default="cfc"
        Task ID or comma-separated list of task IDs.
    task_data : str or None, optional
        Task-to-data mapping, e.g., "cfc:[0],nwp:[0,1]".
    num_layers : int, default=4
        Number of transformer layers.
    emb_dim : int, default=32
        Dimension of embedding.
    num_heads : int, default=4
        Number of attention heads.
    hidden_dim : int, default=128
        Units in dense layers.
    learning_rate : float, default=1e-4
        Learning rate.
    load_backbone_weights : Path or None, optional
        Path to backbone weights from pretrained model.
    freeze_backbone_weights : bool, default=False
        Freeze the backbone weights.
    save_weights : str or None, optional
        Save 'all', 'last', or None model weights.
    save_history : bool, default=False
        Save model's training history.
    output_path : Path, default=Path(".")
        Output directory for models and logs.
    mixed_precision : bool, default=False
        Run training in mixed precision mode.
    compile_model : bool, default=False
        Compile training with XLA.

    Returns
    -------
    model : torch.nn.Module
        The trained LISBET model instance.

    Notes
    -----
    All arguments are exposed for CLI and documentation. For advanced usage,
    see the LISBET documentation.
    """
    # Configure base runtime arguments
    task_ids_list = task_ids.split(",")
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d%H%M%S")

    # Configure accelerator
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    device = torch.device(device_type)
    logging.info("Using %s for training model %s.", device_type, run_id)

    # Configure RNGs
    run_seeds = _generate_seeds(seed, task_ids_list, seed_test_split)
    torch.manual_seed(run_seeds["torch"])

    # Load records
    train_rec, test_rec, dev_rec = _load_records(
        data_format,
        data_path,
        data_scale,
        data_filter,
        dev_ratio,
        test_ratio,
        run_seeds.get("dev_split"),
        run_seeds.get("test_split"),
        keypoints_subset,
        task_ids_list,
        task_data,
    )

    # Determine data shape from first record
    bp_dim = train_rec[task_ids_list[0]][0][1]["posetracks"].sizes["features"]

    # Determine input_features list for config consistency
    input_features = (
        train_rec[task_ids_list[0]][0][1]["posetracks"]
        .coords["features"]
        .values.tolist()
    )

    # Determine max sequence length
    # TODO: Find a better way to compute max_len or fix in the embedder exporter
    max_len = (
        2 * window_size
        if "nwp" in task_ids_list or load_backbone_weights
        else window_size
    )

    # Compute backbone output token idx
    output_token_idx = -(window_offset + 1)
    if not (window_size > window_offset >= 0):
        raise RuntimeError(
            "Window offset must be a positive integer smaller than the window size"
            f" or zero, got {window_offset}."
        )
    logging.debug("Output token IDX = %d", output_token_idx)

    # Configure tasks
    tasks = _configure_tasks(
        train_rec,
        dev_rec,
        test_rec,
        task_ids_list,
        window_size,
        window_offset,
        emb_dim,
        hidden_dim,
        data_augmentation,
        run_seeds,
        device,
        data_format,
    )

    # Build model
    model = _build_model(
        bp_dim,
        emb_dim,
        hidden_dim,
        num_heads,
        num_layers,
        max_len,
        tasks,
        compile_model,
        load_backbone_weights,
        freeze_backbone_weights,
        device,
    )
    model_stats = summary(model, verbose=0)
    logging.info("Model summary\n" + str(model_stats))

    # Optimizer and scheduler
    optimizer, scheduler, scaler = _configure_optimizer_and_scheduler(
        model, learning_rate, mixed_precision
    )

    # Save model config
    _save_model_config(
        output_path,
        run_id,
        window_size,
        window_offset,
        output_token_idx,
        bp_dim,
        emb_dim,
        hidden_dim,
        num_heads,
        num_layers,
        max_len,
        tasks,
        input_features,
    )

    # Training loop
    history = []
    for epoch in range(epochs):
        history_entry = {"epoch": epoch}
        print(f"Epoch {epoch}")
        logging.info("Current LR = %f", scheduler.get_last_lr()[0])

        # Get dataloaders
        train_dataloaders = _configure_dataloaders(
            tasks, "train", batch_size, train_sample
        )

        # Run training epoch
        losses, labels, predictions = _train_one_epoch(
            model,
            train_dataloaders,
            optimizer,
            scheduler,
            scaler,
            tasks,
            device,
            mixed_precision,
        )

        # Get epoch logs
        train_log = _compute_epoch_logs("train", tasks, losses, labels, predictions)
        logging.info(", ".join(f"{k}: {v:.3f}" for k, v in train_log.items()))

        # Update history entry for current epoch
        history_entry.update(train_log)

        # Save weights, if requested
        if save_weights == "all":
            _save_weights(model, output_path, run_id, f"weights_epoch{epoch}.pt")

        if dev_ratio is not None:
            # Get dataloaders
            dev_dataloaders = _configure_dataloaders(
                tasks, "dev", batch_size, dev_sample
            )

            # Run dev epoch
            losses, labels, predictions = _evaluate(
                model, dev_dataloaders, tasks, device, mixed_precision
            )

            # Get epoch logs
            dev_log = _compute_epoch_logs("dev", tasks, losses, labels, predictions)
            logging.info(", ".join(f"{k}: {v:.3f}" for k, v in dev_log.items()))

            # Update history entry for current epoch
            history_entry.update(dev_log)

        # Update history
        history.append(history_entry)

        # Save history, if requested
        if save_history:
            _save_history(output_path, run_id, history)

    # Save final weights, if requested
    if save_weights == "last":
        _save_weights(model, output_path, run_id, "weights_last.pt")

    return model
