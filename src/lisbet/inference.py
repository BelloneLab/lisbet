"""Inference functions for LISBET.

This module provides functions for running inference with LISBET models, including
behavior annotation and embedding computation. It supports both single-sequence
and dataset-wide inference.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

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


def _process_dataset(
    model: torch.nn.Module,
    forward_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    data_format: str,
    data_path: str,
    window_size: int,
    window_offset: int,
    fps_scaling: float,
    batch_size: int,
    data_filter: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> List[Tuple[str, np.ndarray]]:
    """Process an entire dataset with the given model and forward function.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for inference.
    forward_fn : callable
        Function defining how to process model outputs.
    data_format : str
        Format of the dataset to analyze.
    data_path : str
        Path to the directory containing the dataset files.
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

    # Transfer model to device
    model.to(device)

    # Load records
    # NOTE: If the dataset has an explicit test set, the corresponding records will be
    #       allocated to the 'test' group. Otherwise, all records will be in the 'train'
    #       group. However, no training is performed here. Data are simply analyzed and
    #       returned, regardless of the group.
    group_records = load_records(data_format, data_path, data_filter=data_filter)
    seen_keys = set()
    results = []

    # Analyze every group (i.e. train, test and dev)
    # NOTE: We assume no overlapping record IDs in the groups. That is, records could be
    #       stored in a single list with no ambiguity. To ensure that, we keep a set of
    #       observed keys and check for duplicates.
    for group, records in zip(("train", "test", "dev"), group_records):
        if records is None:
            logging.debug("Empty %s group, skipping", group)
            continue

        for seq in tqdm(
            records, desc=f"Analyzing {data_format} dataset, {group} group"
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
        torch.max(output, 1)[1], num_classes=output.shape[1]
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
    data_filter: Optional[str] = None,
    window_size: int = 200,
    window_offset: int = 0,
    fps_scaling: float = 1.0,
    batch_size: int = 128,
    output_path: Optional[str] = None,
) -> List[Tuple[str, np.ndarray]]:
    """Run LISBET behavior classification for every record in a dataset.

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

    Returns
    -------
    list of tuple of (str, ndarray)
        A list of (sequence ID, predicted behavior) tuples for each sequence.

    Raises
    ------
    ValueError
        If the loaded model is not a classification model.
    """
    model = modeling.load_model(model_path, weights_path)
    if not isinstance(model, modeling.MultiTaskModel) or "cfc" not in model.task_heads:
        raise ValueError("Model must be a classification model for behavior annotation")

    results = _process_dataset(
        model=model,
        forward_fn=_classification_forward,
        data_format=data_format,
        data_path=data_path,
        window_size=window_size,
        window_offset=window_offset,
        fps_scaling=fps_scaling,
        batch_size=batch_size,
        data_filter=data_filter,
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
    data_filter: Optional[str] = None,
    window_size: int = 200,
    window_offset: int = 0,
    fps_scaling: float = 1.0,
    batch_size: int = 128,
    output_path: Optional[str] = None,
) -> List[Tuple[str, np.ndarray]]:
    """Compute LISBET embeddings for every record in a dataset.

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

    Returns
    -------
    list of tuple of (str, ndarray)
        A list of (sequence ID, embedding) tuples for each sequence.

    Raises
    ------
    ValueError
        If the loaded model is not an embedding model.
    """
    model = modeling.load_model(model_path, weights_path)
    if not isinstance(model, modeling.EmbeddingModel):
        raise ValueError("Model must be an embedding model for computing embeddings")

    results = _process_dataset(
        model=model,
        forward_fn=_embedding_forward,
        data_format=data_format,
        data_path=data_path,
        window_size=window_size,
        window_offset=window_offset,
        fps_scaling=fps_scaling,
        batch_size=batch_size,
        data_filter=data_filter,
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
