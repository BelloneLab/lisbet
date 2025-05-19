"""Model evaluation utilities for LISBET.

This module provides functions to evaluate classification models on labeled datasets.
"""

from typing import Optional

import numpy as np
from sklearn.metrics import classification_report, f1_score

from . import inference
from .datasets import load_records


def evaluate_model(
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
    labels: Optional[str] = None,
):
    """
    Evaluate a classification model on a labeled dataset and print F1 score.

    Parameters
    ----------
    (same as inference.annotate_behavior)
    Returns
    -------
    None
        Prints F1 score and classification report.
    """
    # Run inference to get predictions
    results = inference._process_inference_dataset(
        model_path=model_path,
        weights_path=weights_path,
        forward_fn=inference._classification_forward,
        data_format=data_format,
        data_path=data_path,
        data_scale=data_scale,
        window_size=window_size,
        window_offset=window_offset,
        fps_scaling=fps_scaling,
        batch_size=batch_size,
        data_filter=data_filter,
        select_coords=select_coords,
        rename_coords=rename_coords,
    )

    # Load ground-truth labels
    group_records = load_records(
        data_format,
        data_path,
        data_filter=data_filter,
        data_scale=data_scale,
        select_coords=select_coords,
        rename_coords=rename_coords,
    )

    # Flatten all records
    y_true = []
    y_pred = []
    for key, pred_arr in results:
        # Find corresponding record
        rec = next(rec for rec in group_records["main_records"] if rec[0] == key)
        true_labels = rec[1]["annotations"].label_cat.values

        # pred_arr is one-hot, take argmax
        pred_labels = np.argmax(pred_arr, axis=1)

        y_true.append(true_labels)
        y_pred.append(pred_labels)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Process user-specified labels if provided
    label_list = None
    if labels is not None:
        label_list = [int(x) for x in labels.split(",") if x.strip() != ""]

    # Compute and print F1 score
    print(
        "Macro F1 score:", f1_score(y_true, y_pred, average="macro", labels=label_list)
    )
    print(classification_report(y_true, y_pred, digits=3, labels=label_list))
