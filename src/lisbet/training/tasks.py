"""Task configuration module."""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from torchmetrics import Metric
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import (
    BinaryAccuracy,
    MulticlassF1Score,
    MultilabelF1Score,
)
from torchvision import transforms

from lisbet import datasets, modeling
from lisbet.transforms_extra import (
    PoseToTensor,
    RandomBlockPermutation,
    RandomPermutation,
    GaussianJitter,
    GaussianWindowJitter,
)


@dataclass
class Task:
    task_id: str
    head: torch.nn.Module
    out_dim: int
    loss_function: torch.nn.Module
    train_dataset: Dataset
    train_loss: Metric
    train_score: Metric
    dev_dataset: Dataset | None = None
    dev_loss: Metric | None = None
    dev_score: Metric | None = None


def _build_augmentation_transforms(data_augmentation, seed):
    """Build transformation pipeline from data augmentation configuration.

    Parameters
    ----------
    data_augmentation : list[DataAugmentationConfig] or None
        Data augmentation configuration. If None or empty list, returns only
        PoseToTensor. If list of DataAugmentationConfig, builds transforms
        according to specifications.
    seed : int
        Random seed for the transformations.

    Returns
    -------
    transforms.Compose
        Composed transformation pipeline.
    """
    transform_list = []

    # Build transforms from DataAugmentationConfig objects
    if data_augmentation:
        for idx, aug_config in enumerate(data_augmentation):
            # Create a unique seed for each augmentation
            aug_seed = seed + idx

            # Build the transform based on augmentation name
            if aug_config.name == "all_perm_id":
                transform = RandomPermutation(aug_seed, coordinate="individuals")
                if aug_config.p < 1.0:
                    transform = transforms.RandomApply([transform], p=aug_config.p)
            elif aug_config.name == "all_perm_ax":
                transform = RandomPermutation(aug_seed, coordinate="space")
                if aug_config.p < 1.0:
                    transform = transforms.RandomApply([transform], p=aug_config.p)
            elif aug_config.name == "blk_perm_id":
                transform = RandomBlockPermutation(
                    aug_seed,
                    coordinate="individuals",
                    permute_fraction=aug_config.frac,
                )
                if aug_config.p < 1.0:
                    transform = transforms.RandomApply([transform], p=aug_config.p)
            elif aug_config.name == "gauss_jitter":
                transform = GaussianJitter(
                    seed=aug_seed,
                    p=aug_config.p,
                    sigma=aug_config.sigma,
                )
            elif aug_config.name == "gauss_window_jitter":
                transform = GaussianWindowJitter(
                    seed=aug_seed,
                    p=aug_config.p,
                    sigma=aug_config.sigma,
                    window=aug_config.window,
                )
            else:
                raise ValueError(f"Unknown augmentation type: {aug_config.name}")

            transform_list.append(transform)

    # Always add PoseToTensor at the end
    transform_list.append(PoseToTensor())

    return transforms.Compose(transform_list)


def _configure_supervised_multilabel_task(
    train_rec,
    dev_rec,
    window_size,
    window_offset,
    embedding_dim,
    hidden_dim,
    data_augmentation,
    run_seeds,
    device,
):
    """Internal helper. Configures the multi-label classification task."""
    if train_rec["multilabel"][0].annotations is None:
        raise RuntimeError("The provided dataset does not contain annotations.")

    # Find number of behaviors in the training set
    labels = np.concatenate(
        [
            rec.annotations.target_cls.mean(dim="annotators").values
            for rec in train_rec["multilabel"]
        ]
    )
    n_samples, num_labels = labels.shape

    # Create classification head
    head = modeling.FrameClassificationHead(
        output_token_idx=-(window_offset + 1),
        input_dim=embedding_dim,
        num_classes=num_labels,
        hidden_dim=hidden_dim,
    )

    # Compute label weight
    n_positive = np.sum(labels, axis=0)
    label_weight = torch.from_numpy(n_samples / (2.0 * n_positive + 1e-6))
    logging.debug("Label weights: %s", label_weight)

    # Create data transformers
    train_transform = _build_augmentation_transforms(
        data_augmentation, run_seeds["transform_multilabel"]
    )

    # Create dataloaders
    train_dataset = datasets.SocialBehaviorDataset(
        records=train_rec["multilabel"],
        window_size=window_size,
        window_offset=window_offset,
        transform=train_transform,
        annot_format="multilabel",
        base_seed=run_seeds["dataset_multilabel"],
    )

    # Create task as dataclass with default dev attributes
    task = Task(
        task_id="multilabel",
        head=head,
        out_dim=num_labels,
        loss_function=torch.nn.BCEWithLogitsLoss(weight=label_weight.to(device)),
        train_dataset=train_dataset,
        train_loss=MeanMetric().to(device),
        train_score=MultilabelF1Score(num_labels, average="macro").to(device),
    )

    # Update dev attributes if dev records are provided
    if dev_rec["multilabel"]:
        dev_transform = transforms.Compose([PoseToTensor()])
        task.dev_dataset = datasets.SocialBehaviorDataset(
            records=dev_rec["multilabel"],
            window_size=window_size,
            window_offset=window_offset,
            transform=dev_transform,
            annot_format="multilabel",
            base_seed=run_seeds["dataset_multilabel"],
        )
        task.dev_loss = MeanMetric().to(device)
        task.dev_score = MultilabelF1Score(num_labels, average="macro").to(device)

    return task


def _configure_supervised_multiclass_task(
    train_rec,
    dev_rec,
    window_size,
    window_offset,
    embedding_dim,
    hidden_dim,
    data_augmentation,
    run_seeds,
    device,
):
    """Internal helper. Configures the multi-class classification task."""
    if train_rec["multiclass"][0].annotations is None:
        raise RuntimeError("The provided dataset does not contain annotations.")

    # Find number of behaviors in the training set
    labels = np.concatenate(
        [
            rec.annotations.target_cls.argmax("behaviors").squeeze().values
            for rec in train_rec["multiclass"]
        ]
    )
    classes = np.unique(labels)
    num_classes = len(classes)
    np.testing.assert_array_equal(classes, np.array(range(num_classes)))

    # Create classification head
    head = modeling.FrameClassificationHead(
        output_token_idx=-(window_offset + 1),
        input_dim=embedding_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
    )

    # Compute class weight
    class_weight = torch.Tensor(
        compute_class_weight("balanced", classes=classes, y=labels)
    )
    logging.debug("Class weights: %s", class_weight)

    # Create data transformers
    train_transform = _build_augmentation_transforms(
        data_augmentation, run_seeds["transform_multiclass"]
    )

    # Create dataloaders
    train_dataset = datasets.SocialBehaviorDataset(
        records=train_rec["multiclass"],
        window_size=window_size,
        window_offset=window_offset,
        transform=train_transform,
        base_seed=run_seeds["dataset_multiclass"],
    )

    # Create task as dataclass with default dev attributes
    task = Task(
        task_id="multiclass",
        head=head,
        out_dim=num_classes,
        loss_function=torch.nn.CrossEntropyLoss(weight=class_weight.to(device)),
        train_dataset=train_dataset,
        train_loss=MeanMetric().to(device),
        train_score=MulticlassF1Score(num_classes, average="macro").to(device),
    )

    # Update dev attributes if dev records are provided
    if dev_rec["multiclass"]:
        dev_transform = transforms.Compose([PoseToTensor()])
        task.dev_dataset = datasets.SocialBehaviorDataset(
            records=dev_rec["multiclass"],
            window_size=window_size,
            window_offset=window_offset,
            transform=dev_transform,
            base_seed=run_seeds["dataset_multiclass"],
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
    embedding_dim,
    hidden_dim,
    data_augmentation,
    run_seeds,
    device,
):
    """Internal helper. Configures a self-supervised task."""
    # Create classification head
    head = modeling.WindowClassificationHead(
        input_dim=embedding_dim, num_classes=1, hidden_dim=hidden_dim
    )

    # Create data transformers
    train_transform = _build_augmentation_transforms(
        data_augmentation, run_seeds[f"transform_{task_id}"]
    )

    # Create dataloaders
    task_map = {
        "cons": datasets.GroupConsistencyDataset,
        "order": datasets.TemporalOrderDataset,
        "shift": datasets.TemporalShiftDataset,
        "warp": datasets.TemporalWarpDataset,
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
        dev_transform = transforms.Compose([PoseToTensor()])
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
    embedding_dim,
    hidden_dim,
    data_augmentation,
    run_seeds,
    device,
):
    """Internal helper. Configures all tasks."""
    tasks = []
    for task_id in task_ids:
        if task_id == "multiclass":
            tasks.append(
                _configure_supervised_multiclass_task(
                    train_rec,
                    dev_rec,
                    window_size,
                    window_offset,
                    embedding_dim,
                    hidden_dim,
                    data_augmentation,
                    run_seeds,
                    device,
                )
            )
        elif task_id == "multilabel":
            tasks.append(
                _configure_supervised_multilabel_task(
                    train_rec,
                    dev_rec,
                    window_size,
                    window_offset,
                    embedding_dim,
                    hidden_dim,
                    data_augmentation,
                    run_seeds,
                    device,
                )
            )
        elif task_id in ("cons", "order", "shift", "warp"):
            tasks.append(
                _configure_selfsupervised_task(
                    task_id,
                    train_rec,
                    dev_rec,
                    window_size,
                    window_offset,
                    embedding_dim,
                    hidden_dim,
                    data_augmentation,
                    run_seeds,
                    device,
                )
            )
        else:
            raise ValueError(f"Unknown task {task_id}")

    return tasks
