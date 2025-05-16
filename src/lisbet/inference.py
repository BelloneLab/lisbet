"""Inference functions for LISBET.

This module provides functions for running inference with LISBET models, including
behavior annotation and embedding computation. It supports both single-sequence
and dataset-wide inference.
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torchvision import transforms
from tqdm.auto import tqdm

from . import modeling
from .datasets import load_records
from .input_pipeline import FrameClassificationDataset


def run_inference_for_sequence(
    model: torch.nn.Module,
    sequence: tuple,
    forward_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    window_size: int,
    window_offset: int,
    fps_scaling: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run model inference on a single sequence.

    This function handles the core inference logic for a single sequence,
    including data loading, batching, and model forward pass.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for inference. Must be in eval mode.
    sequence : tuple
        Body-pose sequence to analyze, as (sequence_id, sequence_data).
    forward_fn : callable
        Function that takes (model, data) and returns predictions.
        This defines how to process the model output.
    window_size : int
        Size of the sliding window to apply on the input sequences.
        Larger windows provide more temporal context.
    window_offset : int
        Sliding window offset. Controls the output token.
    fps_scaling : float
        FPS scaling factor to adjust for different video frame rates.
    batch_size : int
        Number of windows to process in parallel.
    device : str
        Device to use for inference (e.g., 'cuda' or 'cpu').

    Returns
    -------
    ndarray
        Model predictions or embeddings for the sequence.
        Shape depends on the forward_fn implementation.
    """
    model.eval()

    dataset = FrameClassificationDataset(
        records=[sequence],
        window_size=window_size,
        window_offset=window_offset,
        fps_scaling=fps_scaling,
        transform=transforms.Compose([torch.Tensor]),
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    predictions = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            pred = forward_fn(model, data)
            predictions.append(pred.cpu().numpy())

    return np.concatenate(predictions)


def _process_inference_dataset(
    model_path: str,
    weights_path: str,
    forward_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    data_format: str,
    data_path: str,
    data_scale: str,
    window_size: int,
    window_offset: int,
    fps_scaling: float,
    batch_size: int,
    data_filter: Optional[str] = None,
    keypoints_subset: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> list[tuple[str, np.ndarray]]:
    """
    Load model, records, check input_features compatibility, and process the dataset.

    Parameters
    ----------
    model_path : str
        Path to the model config (YAML format).
    weights_path : str
        Path to the model weights.
    forward_fn : callable
        Function defining how to process model outputs.
    data_format : str
        Format of the dataset to analyze.
    data_path : str
        Path to the directory containing the dataset files.
    data_scale : str or None
        Scaling string or None for auto-scaling.
    window_size : int
        Size of the sliding window.
    window_offset : int
        Sliding window offset.
    fps_scaling : float
        FPS scaling factor.
    batch_size : int
        Batch size for inference.
    data_filter : str, optional
        Filter to apply when loading records.
    keypoints_subset : str, optional
        Optional subset string in the format 'INDIVS;COORDS;PARTS', where each field is
        a comma-separated list or '*' for all. If None, all data is loaded.
    device : str, optional
        Device to use. Defaults to CUDA if available.

    Returns
    -------
    list of tuple of (str, ndarray)
        List of (sequence ID, predictions) pairs.

    Raises
    ------
    RuntimeError
        If duplicate sequence IDs are found across dataset splits.
    ValueError
        If input features are incompatible.
    """
    if device is None:
        # Configure accelerator
        if torch.cuda.is_available():
            device_type = "cuda"
        elif torch.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
        device = torch.device(device_type)

    # Load model config and model
    with open(model_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model = modeling.load_model(model_path, weights_path)
    model.to(device)

    # Load records
    group_records = load_records(
        data_format,
        data_path,
        data_filter=data_filter,
        data_scale=data_scale,
        keypoints_subset=keypoints_subset,
    )

    # Input features compatibility check
    model_features = [tuple(x) for x in config.get("input_features")]
    dataset_features = (
        group_records["main_records"][0][1]["posetracks"]
        .coords["features"]
        .values.tolist()
    )
    if dataset_features != model_features:
        raise ValueError(
            f"Incompatible input features!\n"
            f"Model expects: {model_features}\n"
            f"Dataset provides: {dataset_features}"
        )

    # Analyze records
    # NOTE: We assume no overlapping record IDs. That is, records could be stored in a
    #       single list with no ambiguity. To ensure that, we keep a set of observed
    #       keys and check for duplicates. This safety check is not strictly necessary
    #       and should be removed in the future or moved to the dataset loading.
    results = []
    seen_keys = set()
    for group_name, group_data in group_records.items():
        for seq in tqdm(
            group_data, desc=f"Analyzing {data_format} dataset, {group_name} group"
        ):
            # Extract sequence ID
            key = seq[0]

            # Check for duplicated keys
            if key in seen_keys:
                raise RuntimeError(f"Duplicated key {key}")
            seen_keys.add(key)

            # Run inference
            model_output = run_inference_for_sequence(
                model,
                seq,
                forward_fn,
                window_size,
                window_offset,
                fps_scaling,
                batch_size,
                device,
            )
            results.append((key, model_output))

    return results


def _classification_forward(model: torch.nn.Module, data: torch.Tensor) -> torch.Tensor:
    """Process classification model outputs.

    Parameters
    ----------
    model : torch.nn.Module
        The classification model.
    data : torch.Tensor
        Input data batch.

    Returns
    -------
    torch.Tensor
        One-hot encoded class predictions.
    """
    output = model(data, "cfc")
    return torch.nn.functional.one_hot(
        torch.argmax(output, dim=1), num_classes=output.shape[1]
    )


def _embedding_forward(model: torch.nn.Module, data: torch.Tensor) -> torch.Tensor:
    """Process embedding model outputs.

    Parameters
    ----------
    model : torch.nn.Module
        The embedding model.
    data : torch.Tensor
        Input data batch.

    Returns
    -------
    torch.Tensor
        Computed embeddings.
    """
    return model(data).squeeze(dim=1)


def annotate_behavior(
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
    output_path: Optional[str] = None,
    keypoints_subset: Optional[str] = None,
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
    keypoints_subset : str, optional
        Optional subset string in the format 'INDIVS;COORDS;PARTS', where each field is
        a comma-separated list or '*' for all. If None, all data is loaded.

    Returns
    -------
    list of tuple of (str, ndarray)
        A list of (sequence ID, predicted behavior) tuples for each sequence.

    Raises
    ------
    ValueError
        If the loaded model is not a classification model.
    """
    results = _process_inference_dataset(
        model_path=model_path,
        weights_path=weights_path,
        forward_fn=_classification_forward,
        data_format=data_format,
        data_path=data_path,
        data_scale=data_scale,
        window_size=window_size,
        window_offset=window_offset,
        fps_scaling=fps_scaling,
        batch_size=batch_size,
        data_filter=data_filter,
        keypoints_subset=keypoints_subset,
    )

    # Store results on file, if requested
    if output_path is not None:
        for key, model_output in results:
            dst_path = (
                Path(output_path) / "annotations" / key / "machineAnnotation_lisbet.csv"
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(model_output).to_csv(dst_path)

    return results


def compute_embeddings(
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
    output_path: Optional[str] = None,
    keypoints_subset: Optional[str] = None,
) -> list[tuple[str, np.ndarray]]:
    """
    Compute LISBET embeddings for every record in a dataset.

    This function loads an embedding model and processes an entire dataset,
    computing embeddings for each sequence.

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
    window_size : int, default=200
        Size of the sliding window to apply on the input sequences.
    window_offset : int, default=0
        Sliding window offset.
    fps_scaling : float, default=1.0
        FPS scaling factor.
    batch_size : int, default=128
        Batch size for inference.
    output_path : str or None, optional
        If given, embeddings will be saved as CSV files in this directory.
    keypoints_subset : str, optional
        Optional subset string in the format 'INDIVS;COORDS;PARTS', where each field is
        a comma-separated list or '*' for all. If None, all data is loaded.

    Returns
    -------
    list of tuple of (str, ndarray)
        A list of (sequence ID, embedding) tuples for each sequence.

    Raises
    ------
    ValueError
        If the loaded model is not an embedding model.
    """
    results = _process_inference_dataset(
        model_path=model_path,
        weights_path=weights_path,
        forward_fn=_embedding_forward,
        data_format=data_format,
        data_path=data_path,
        data_scale=data_scale,
        window_size=window_size,
        window_offset=window_offset,
        fps_scaling=fps_scaling,
        batch_size=batch_size,
        data_filter=data_filter,
        keypoints_subset=keypoints_subset,
    )

    # Store results on file, if requested
    if output_path is not None:
        for key, model_output in results:
            dst_path = (
                Path(output_path) / "embeddings" / key / "features_lisbet_embedding.csv"
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(model_output).to_csv(dst_path)

    return results
