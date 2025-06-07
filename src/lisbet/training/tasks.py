"""Task configuration module."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from torchmetrics import Metric
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryAccuracy, MulticlassF1Score
from torchvision import transforms

from lisbet import datasets, modeling
from lisbet.transforms_extra import RandomXYSwap, Record2Tensor


@dataclass
class Task:
    task_id: str
    head: torch.nn.Module
    out_dim: int
    loss_function: torch.nn.Module
    train_dataset: Dataset
    train_loss: Metric
    train_score: Metric
    dev_dataset: Optional[Dataset] = None
    dev_loss: Optional[Metric] = None
    dev_score: Optional[Metric] = None


def _configure_classification_task(
    train_rec,
    dev_rec,
    window_size,
    window_offset,
    emb_dim,
    hidden_dim,
    data_augmentation,
    run_seeds,
    device,
):
    """Internal helper. Configures the classification task."""
    if train_rec["cfc"][0].annotations is None:
        raise RuntimeError("The provided dataset does not contain annotations.")

    # Find number of behaviors in the training set
    labels = np.concatenate(
        [
            rec.annotations.target_cls.argmax("behaviors").squeeze().values
            for rec in train_rec["cfc"]
        ]
    )
    classes = np.unique(labels)
    num_classes = len(classes)
    np.testing.assert_array_equal(classes, np.array(range(num_classes)))

    # Create classification head
    head = modeling.FrameClassificationHead(
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

    # Create data transformers
    train_transform = (
        transforms.Compose([RandomXYSwap(run_seeds["transform_cfc"]), Record2Tensor()])
        if data_augmentation
        else transforms.Compose([Record2Tensor()])
    )

    # Create dataloaders
    train_dataset = datasets.RandomWindowDataset(
        records=train_rec["cfc"],
        window_size=window_size,
        window_offset=window_offset,
        transform=train_transform,
        base_seed=run_seeds["dataset_cfc"],
    )

    # Create task as dataclass with default dev attributes
    task = Task(
        task_id="cfc",
        head=head,
        out_dim=num_classes,
        loss_function=torch.nn.CrossEntropyLoss(weight=class_weight.to(device)),
        train_dataset=train_dataset,
        train_loss=MeanMetric().to(device),
        train_score=MulticlassF1Score(num_classes, average="macro").to(device),
    )

    # Update dev attributes if dev records are provided
    if dev_rec["cfc"]:
        dev_transform = transforms.Compose([Record2Tensor()])
        task.dev_dataset = datasets.RandomWindowDataset(
            records=dev_rec["cfc"],
            window_size=window_size,
            window_offset=window_offset,
            transform=dev_transform,
            base_seed=run_seeds["dataset_cfc"],
        )
        task.dev_loss = MeanMetric().to(device)
        task.dev_score = MulticlassF1Score(num_classes, average="macro").to(device)

    return task


def _configure_selfsupervised_task(
    task_id,
    train_rec,
    dev_rec,
    window_size,
    window_offset,
    emb_dim,
    hidden_dim,
    data_augmentation,
    run_seeds,
    device,
):
    """Internal helper. Configures a self-supervised task."""
    # Create classification head
    head = modeling.WindowClassificationHead(
        emb_dim=emb_dim, out_dim=1, hidden_dim=hidden_dim
    )

    # Create data transformers
    train_transform = (
        transforms.Compose(
            [RandomXYSwap(run_seeds[f"transform_{task_id}"]), Record2Tensor()]
        )
        if data_augmentation
        else transforms.Compose([Record2Tensor()])
    )

    # Create dataloaders
    task_map = {
        "nwp": datasets.NWPDataset,
        "smp": datasets.SMPDataset,
        "dmp": datasets.DMPDataset,
        "vsp": datasets.VSPDataset,
    }
    train_dataset = task_map[task_id](
        records=train_rec[task_id],
        window_size=window_size,
        window_offset=window_offset,
        transform=train_transform,
        base_seed=run_seeds[f"dataset_{task_id}"],
    )

    # Create task as dataclass with default dev attributes
    task = Task(
        task_id=task_id,
        head=head,
        out_dim=1,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        train_dataset=train_dataset,
        train_loss=MeanMetric().to(device),
        train_score=BinaryAccuracy().to(device),
    )

    # Update dev attributes if dev records are provided
    if dev_rec[task_id]:
        dev_transform = transforms.Compose([Record2Tensor()])
        task.dev_dataset = task_map[task_id](
            records=dev_rec[task_id],
            window_size=window_size,
            window_offset=window_offset,
            transform=dev_transform,
            base_seed=run_seeds[f"dataset_{task_id}"],
        )
        task.dev_loss = MeanMetric().to(device)
        task.dev_score = BinaryAccuracy().to(device)

    return task


def configure_tasks(
    train_rec,
    dev_rec,
    task_ids,
    window_size,
    window_offset,
    emb_dim,
    hidden_dim,
    data_augmentation,
    run_seeds,
    device,
):
    """Internal helper. Configures all tasks."""
    tasks = []
    for task_id in task_ids:
        if task_id == "cfc":
            tasks.append(
                _configure_classification_task(
                    train_rec,
                    dev_rec,
                    window_size,
                    window_offset,
                    emb_dim,
                    hidden_dim,
                    data_augmentation,
                    run_seeds,
                    device,
                )
            )
        elif task_id == "lfc":
            raise NotImplementedError(
                "Multi-Label Frame Classification task is not implemented yet."
            )
        elif task_id in ("nwp", "smp", "vsp", "dmp"):
            tasks.append(
                _configure_selfsupervised_task(
                    task_id,
                    train_rec,
                    dev_rec,
                    window_size,
                    window_offset,
                    emb_dim,
                    hidden_dim,
                    data_augmentation,
                    run_seeds,
                    device,
                )
            )
        else:
            raise ValueError(f"Unknown task {task_id}")

    return tasks
