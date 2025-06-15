"""
Behavior annotation (classification and multi-label) for LISBET.
"""

from functools import partial
from typing import Optional

import numpy as np
import torch
from torch.nn.functional import one_hot

from lisbet.inference.common import predict
from lisbet.io import dump_annotations


def _multiclass_forward(model: torch.nn.Module, data: torch.Tensor) -> torch.Tensor:
    """Forward function for multiclass classification."""
    output = model(data, "multiclass")
    labels = one_hot(torch.argmax(output, dim=1), num_classes=output.shape[1])

    return labels


def _multilabel_forward(
    model: torch.nn.Module, data: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Forward function for multilabel classification."""
    output = model(data, "multilabel")
    labels = (torch.sigmoid(output) > threshold).int()

    return labels


def annotate_behavior(
    model_path: str,
    weights_path: str,
    data_format: str,
    data_path: str,
    data_scale: Optional[str] = None,
    data_filter: Optional[str] = None,
    mode: str = "multiclass",
    threshold: float = 0.5,
    window_size: int = 200,
    window_offset: int = 0,
    fps_scaling: float = 1.0,
    batch_size: int = 128,
    output_path: Optional[str] = None,
    select_coords: Optional[str] = None,
    rename_coords: Optional[str] = None,
) -> list[tuple[str, np.ndarray]]:
    """
    Run LISBET behavior classification for every record in a dataset.

    This function loads a classification model and processes an entire dataset,
    producing behavior annotations for each sequence.

    Parameters
    ----------
    model_path : str
        Path to the model config (JSON format).
    weights_path : str
        Path to the HDF5 file containing the model weights.
    data_format : str
        Format of the dataset to analyze.
    data_path : str
        Path to the directory containing the dataset files.
    data_scale : str or None
        Scaling string or None for auto-scaling.
    data_filter : str, optional
        Filter to apply when loading records.
    mode : str, default='multiclass'
        Classification mode, either 'multiclass' or 'multilabel'.
    threshold : float, default=0.5
        Probability threshold for multilabel classification.
    window_size : int, default=200
        Size of the sliding window to apply on the input sequences.
    window_offset : int, default=0
        Sliding window offset.
    fps_scaling : float, default=1.0
        FPS scaling factor.
    batch_size : int, default=128
        Batch size for inference.
    output_path : str or None, optional
        If given, predictions will be saved as CSV files in this directory.
    select_coords : str, optional
        Optional subset string in the format 'INDIVIDUALS;AXES;KEYPOINTS', where each
        field is a comma-separated list or '*' for all. If None, all data is loaded.
    rename_coords : str, optional
        Optional coordinate names remapping in the format 'INDIVIDUALS;AXES;KEYPOINTS',
        where each field is a comma-separated list of maps 'old_id:new_id' or '*' for
        no remapping at that level. If None, original dataset names are used.

    Returns
    -------
    list of tuple of (str, ndarray)
        A list of (sequence ID, predicted behavior) tuples for each sequence.

    Raises
    ------
    ValueError
        If the loaded model is not a classification model.
    """
    if mode == "multiclass":
        forward_fn = _multiclass_forward
    elif mode == "multilabel":
        forward_fn = partial(_multilabel_forward, threshold=threshold)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    results = predict(
        model_path=model_path,
        weights_path=weights_path,
        forward_fn=forward_fn,
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

    if output_path is not None:
        # Save predictions to output path
        dump_annotations(results, output_path)

    return results
