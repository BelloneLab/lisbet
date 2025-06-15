"""Model evaluation utilities for LISBET.

This module provides functions to evaluate classification models on labeled datasets,
using the new LISBET inference API, torchmetrics, and improved output handling.
"""

from typing import Optional

import torch
from lightning.fabric.utilities.data import suggested_max_num_workers
from rich import print as rprint
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from tqdm.auto import tqdm

from lisbet.datasets import AnnotatedWindowDataset
from lisbet.inference.common import (
    check_feature_compatibility,
    load_model_and_config,
    select_device,
)
from lisbet.io import load_records
from lisbet.io.core import dump_evaluation_results
from lisbet.transforms_extra import PoseToTensor


def evaluate(
    model_path: str,
    weights_path: str,
    data_format: str,
    data_path: str,
    data_scale: Optional[str] = None,
    data_filter: Optional[str] = None,
    window_size: int = 200,
    window_offset: int = 0,
    fps_scaling: float = 1.0,
    batch_size: int = 128,
    select_coords: Optional[str] = None,
    rename_coords: Optional[str] = None,
    ignore_index: Optional[int] = None,
    mode: str = "multiclass",
    threshold: float = 0.5,
    output_path: Optional[str] = None,
) -> dict:
    """
    Evaluate a classification model on a labeled dataset and print/save metrics.

    Parameters
    ----------
    model_path : str
        Path to the model config (YAML).
    weights_path : str
        Path to the model weights.
    data_format : str
        Format of the dataset to analyze.
    data_path : str
        Path to the directory containing the dataset files.
    data_scale : str or None, optional
        Scaling string or None for auto-scaling.
    data_filter : str, optional
        Filter to apply when loading records.
    window_size : int, default=200
        Size of the sliding window to apply on the input sequences.
    window_offset : int, default=0
        Sliding window offset.
    fps_scaling : float, default=1.0
        FPS scaling factor.
    batch_size : int, default=128
        Batch size for inference.
    select_coords : str, optional
        Optional subset string in the format 'INDIVIDUALS;AXES;KEYPOINTS'.
    rename_coords : str, optional
        Optional coordinate names remapping in the format 'INDIVIDUALS;AXES;KEYPOINTS'.
    mode : str, default='multiclass'
        Evaluation mode: 'multiclass' or 'multilabel'.
    output_path : str, optional
        If given, the evaluation report will be saved as a YAML file in this directory.
    ignore_index : int, optional
        Index to ignore in the evaluation metrics (e.g., background class).
    threshold : float, default=0.5
        Threshold for multilabel binarization.

    Returns
    -------
    dict
        Evaluation report with metrics.
    """
    device = select_device()
    model, config = load_model_and_config(model_path, weights_path, device)

    # Load records and check features
    records = load_records(
        data_format=data_format,
        data_path=data_path,
        data_scale=data_scale,
        data_filter=data_filter,
        select_coords=select_coords,
        rename_coords=rename_coords,
    )
    check_feature_compatibility(config, records)

    # Prepare dataset for evaluation
    dataset = AnnotatedWindowDataset(
        records=records,
        window_size=window_size,
        window_offset=window_offset,
        fps_scaling=fps_scaling,
        transform=PoseToTensor(),
        annot_format=mode,
    )
    num_workers = min(suggested_max_num_workers(1), batch_size // 8)
    prefetch_factor = 4 if num_workers > 0 else None
    pin_memory = device.type == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )

    # Initialize metrics
    n_categories = records[0].annotations.sizes["behaviors"]
    metrics_kwargs = {
        "average": "macro",
        "ignore_index": ignore_index,
    }
    if mode == "multiclass":
        metrics_kwargs["num_classes"] = n_categories
    elif mode == "multilabel":
        metrics_kwargs["num_labels"] = n_categories
        metrics_kwargs["threshold"] = threshold
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Per-class metrics
    per_class_metrics_kwargs = metrics_kwargs.copy()
    per_class_metrics_kwargs["average"] = "none"
    per_class_metrics_kwargs["ignore_index"] = None

    f1_metric = F1Score(task=mode, **metrics_kwargs).to(device)
    acc_metric = Accuracy(task=mode, **metrics_kwargs).to(device)
    f1_per_class = F1Score(task=mode, **per_class_metrics_kwargs).to(device)
    precision_per_class = Precision(task=mode, **per_class_metrics_kwargs).to(device)
    recall_per_class = Recall(task=mode, **per_class_metrics_kwargs).to(device)

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = model(x, mode)

            # Udpate metrics
            f1_metric.update(logits, y)
            acc_metric.update(logits, y)
            f1_per_class.update(logits, y)
            precision_per_class.update(logits, y)
            recall_per_class.update(logits, y)

    # Compute metrics
    report = {
        "mode": mode,
        "f1_macro": float(f1_metric.compute()),
        "accuracy_macro": float(acc_metric.compute()),
        "per_class": {
            "f1": f1_per_class.compute().cpu().numpy().tolist(),
            "precision": precision_per_class.compute().cpu().numpy().tolist(),
            "recall": recall_per_class.compute().cpu().numpy().tolist(),
        },
    }

    # Print summary
    if ignore_index is not None:
        rprint(
            f"\n[bold red]WARNING: Ignoring index {ignore_index} in macro metrics.\n"
            "A bug in torchmetrics may cause incorrect macro F1 and accuracy "
            "(see https://github.com/Lightning-AI/torchmetrics/issues/2441).\n"
            "Please consider validating your results agaist the per-class metrics.",
        )
    rprint("\n[bold green]Evaluation Summary")
    rprint(f"Mode: {mode}")
    rprint(f"Macro F1: {report['f1_macro']:.3f}")
    rprint(f"Macro Accuracy: {report['accuracy_macro']:.3f}")
    rprint("Per-class metrics:")
    for i, (f1, precision, recall) in enumerate(
        zip(
            report["per_class"]["f1"],
            report["per_class"]["precision"],
            report["per_class"]["recall"],
        )
    ):
        rprint(
            f"  Class {i}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}"
        )

    # Save results if requested
    if output_path is not None:
        dump_evaluation_results(report, output_path, model_path)

    return report
