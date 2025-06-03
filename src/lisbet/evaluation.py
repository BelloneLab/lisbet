"""Model evaluation utilities for LISBET.

This module provides functions to evaluate classification models on labeled datasets.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from sklearn.metrics import classification_report

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
    output_path: Optional[str] = None,
):
    """
    Evaluate a classification model on a labeled dataset and print F1 score.

    This function loads a classification model, runs inference on a labeled dataset,
    and computes classification metrics (F1 score and others). Optionally, it saves
    the classification report to a YAML file.

    Parameters
    ----------
    model_path : str
        Path to the model config (YAML format).
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
        Optional subset string in the format 'INDIVIDUALS;AXES;KEYPOINTS', where each
        field is a comma-separated list or '*' for all. If None, all data is loaded.
    rename_coords : str, optional
        Optional coordinate names remapping in the format 'INDIVIDUALS;AXES;KEYPOINTS',
        where each field is a comma-separated list of maps 'old_id:new_id' or '*' for
        no remapping at that level. If None, original dataset names are used.
    labels : str, optional
        Comma-separated list of integer class labels to include in the report. If None,
        all labels present in the data are used.
    output_path : str, optional
        If given, the classification report will be saved as a YAML file in this
        directory.

    Returns
    -------
    dict
        Classification report as returned by `sklearn.metrics.classification_report`.

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
    # NOTE: We load the records twice, but at least we don't have to re-implement the
    #       forward pass. In the future, we could consider decomposing the inference
    #       function into smaller components to avoid this duplication.
    records = load_records(
        data_format=data_format,
        data_path=data_path,
        data_filter=data_filter,
        data_scale=data_scale,
        select_coords=select_coords,
        rename_coords=rename_coords,
    )

    # Flatten all records
    y_true = []
    y_pred = []
    for (key, pred_arr), rec in zip(results, records):
        assert rec.id == key

        true_labels = rec.annotations.target_cls.argmax("behaviors").squeeze().values

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
    # NOTE: This repetition is not ideal, but we need to print the report in a
    #       human-readable format and also save it to a file. We could consider
    #       refactoring, but the current approach is simple and works well.
    report_dict = classification_report(
        y_true, y_pred, digits=3, labels=label_list, output_dict=True
    )
    report_str = classification_report(y_true, y_pred, digits=3, labels=label_list)
    print(report_str)

    # Save classification report to file if output_path is provided
    if output_path is not None:
        # Find model ID
        with open(model_path, encoding="utf-8") as f_yaml:
            model_config = yaml.safe_load(f_yaml)
        model_id = model_config["model_id"]

        # Create output directory if it doesn't exist
        report_path = (
            Path(output_path) / "evaluations" / model_id / "classification_report.yml"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Save report
        with open(report_path, "w", encoding="utf-8") as f_yaml:
            yaml.safe_dump(report_dict, f_yaml)

    return report_dict
