"""
Embedding extraction for LISBET.
"""


import numpy as np
import torch

from lisbet.inference.common import predict
from lisbet.io import dump_embeddings


def _embedding_forward(model: torch.nn.Module, data: torch.Tensor) -> torch.Tensor:
    """Forward function for extracting embeddings from the model."""
    return model(data, "embedding").squeeze(dim=1)


def compute_embeddings(
    model_path: str,
    weights_path: str,
    data_format: str,
    data_path: str,
    data_scale: str | None = None,
    data_filter: str | None = None,
    window_size: int = 200,
    window_offset: int = 0,
    fps_scaling: float = 1.0,
    batch_size: int = 128,
    output_path: str | None = None,
    select_coords: str | None = None,
    rename_coords: str | None = None,
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
        A list of (sequence ID, embedding) tuples for each sequence.

    Raises
    ------
    ValueError
        If the loaded model is not an embedding model.
    """
    results = predict(
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
        select_coords=select_coords,
        rename_coords=rename_coords,
    )

    # Store results on file, if requested
    if output_path is not None:
        dump_embeddings(results, output_path)

    return results
