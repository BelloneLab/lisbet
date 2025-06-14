"""
Core prediction logic for LISBET inference.
"""

from itertools import zip_longest

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.table import Table
from torchvision import transforms
from tqdm.auto import tqdm

from lisbet import modeling
from lisbet.datasets import WindowDataset
from lisbet.io import load_records
from lisbet.transforms_extra import PoseToTensor


def select_device(device: str | None = None) -> torch.device:
    """
    Select the appropriate torch device.

    Parameters
    ----------
    device : str or None, optional
        Device string (e.g., 'cuda', 'cpu', 'mps'). If None, automatically selects
        the best available device.

    Returns
    -------
    torch.device
        The selected torch device.
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def load_model_and_config(model_path: str, weights_path: str, device: torch.device):
    """
    Load model configuration and model, move model to device and set to eval mode.

    Parameters
    ----------
    model_path : str
        Path to the model configuration YAML file.
    weights_path : str
        Path to the model weights file.
    device : torch.device
        Device to move the model to.

    Returns
    -------
    model : torch.nn.Module
        The loaded model.
    config : dict
        The loaded model configuration.
    """
    with open(model_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model = modeling.load_model(model_path, weights_path)
    model.to(device)
    model.eval()
    return model, config


def check_feature_compatibility(config, records):
    """
    Check if the input features of the model and dataset match.

    Parameters
    ----------
    config : dict
        Model configuration dictionary containing input features.
    records : list
        List of dataset records to check for feature compatibility.

    Raises
    ------
    ValueError
        If the input features of the model and dataset do not match.
    """
    model_features = config.get("input_features", {})
    dataset_coords = records[0].posetracks.coords

    # Extract dataset features
    dataset_individuals = list(dataset_coords["individuals"].values)
    dataset_keypoints = list(dataset_coords["keypoints"].values)
    dataset_space = list(dataset_coords["space"].values)

    # Extract model features
    model_individuals = list(model_features.get("individuals", []))
    model_keypoints = list(model_features.get("keypoints", []))
    model_space = list(model_features.get("space", []))

    # Compare features
    features_match = (
        dataset_individuals == model_individuals
        and dataset_keypoints == model_keypoints
        and dataset_space == model_space
    )

    if not features_match:
        console = Console()
        table = Table(title="Input Features Compatibility Check")

        columns = [
            ("Model Individuals", model_individuals, "cyan"),
            ("Dataset Individuals", dataset_individuals, "magenta"),
            ("Model Keypoints", model_keypoints, "cyan"),
            ("Dataset Keypoints", dataset_keypoints, "magenta"),
            ("Model Space", model_space, "cyan"),
            ("Dataset Space", dataset_space, "magenta"),
        ]
        for name, _, style in columns:
            table.add_column(name, style=style)

        for row in zip_longest(
            model_individuals,
            dataset_individuals,
            model_keypoints,
            dataset_keypoints,
            model_space,
            dataset_space,
            fillvalue="",
        ):
            table.add_row(*(str(item) for item in row))

        # Print the table to string
        with console.capture() as capture:
            console.print(
                "[bold red]ERROR: Incompatible input features between model and "
                "dataset!\nPlease use 'select_coords' and 'rename_coords' to "
                "align model and dataset input features.[/bold red]"
            )
            console.print(table)
        table_str = capture.get()

        raise ValueError(f"Incompatible input features.\n{table_str}")


def predict_record(
    record,
    model,
    device,
    window_size,
    window_offset,
    fps_scaling,
    batch_size,
    forward_fn,
):
    """
    Run prediction on a single record and return the output.

    Parameters
    ----------
    record : object
        The dataset record to predict on.
    model : torch.nn.Module
        The trained model for inference.
    device : torch.device
        Device to run inference on.
    window_size : int
        Size of the sliding window for the dataset.
    window_offset : int
        Offset for the sliding window.
    fps_scaling : float
        Scaling factor for frames per second.
    batch_size : int
        Batch size for inference.
    forward_fn : callable
        Function to perform the forward pass (model, data) -> prediction.

    Returns
    -------
    output : np.ndarray
        The concatenated prediction output for the record.
    """
    dataset = WindowDataset(
        records=[record],
        window_size=window_size,
        window_offset=window_offset,
        fps_scaling=fps_scaling,
        transform=transforms.Compose([PoseToTensor()]),
    )

    # WARNING: Do not use `num_workers` in DataLoader for inference. The behavior of
    #          an iterable-style dataset is different from a map-style dataset, and will
    #          cause `num_workers` * `batch_size` batches to be generated before
    #          exhausting the dataset.
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    predictions = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            pred = forward_fn(model, data)
            predictions.append(pred.cpu().numpy())

    output = np.concatenate(predictions)

    return output


def predict(
    model_path: str,
    weights_path: str,
    data_format: str,
    data_path: str,
    *,
    data_scale: str | None = None,
    data_filter: str | None = None,
    window_size: int = 200,
    window_offset: int = 0,
    fps_scaling: float = 1.0,
    batch_size: int = 128,
    select_coords: str | None = None,
    rename_coords: str | None = None,
    device: str | None = None,
    forward_fn: callable = None,
) -> list[tuple[str, np.ndarray]]:
    """
    Run model prediction on all records in a dataset.

    Handles model loading, feature compatibility, batching, and device management.

    Parameters
    ----------
    model_path : str
        Path to the model configuration YAML file.
    weights_path : str
        Path to the model weights file.
    data_format : str
        Format of the input data (e.g., 'h5', 'csv').
    data_path : str
        Path to the input data.
    data_scale : str or None, optional
        Scaling method for the data.
    data_filter : str or None, optional
        Filter to apply to the data.
    window_size : int, optional
        Size of the sliding window for the dataset. Default is 200.
    window_offset : int, optional
        Offset for the sliding window. Default is 0.
    fps_scaling : float, optional
        Scaling factor for frames per second. Default is 1.0.
    batch_size : int, optional
        Batch size for inference. Default is 128.
    select_coords : str or None, optional
        Coordinate selection string for filtering input features.
    rename_coords : str or None, optional
        Coordinate renaming string for aligning input features.
    device : str or None, optional
        Device string (e.g., 'cuda', 'cpu', 'mps'). If None, automatically selects
        device.
    forward_fn : callable, optional
        Function to perform the forward pass (model, data) -> prediction.

    Returns
    -------
    results : list of tuple
        List of (record_id, prediction) tuples, where prediction is a numpy array.
    """

    # Device selection
    device = select_device(device)

    # Load model config and model
    model, config = load_model_and_config(model_path, weights_path, device)

    # Load records
    records = load_records(
        data_format=data_format,
        data_path=data_path,
        data_filter=data_filter,
        data_scale=data_scale,
        select_coords=select_coords,
        rename_coords=rename_coords,
    )

    # Input features compatibility check
    check_feature_compatibility(config, records)

    results = []
    for record in tqdm(records, desc=f"Predicting {data_format} dataset"):
        output = predict_record(
            record,
            model,
            device,
            window_size,
            window_offset,
            fps_scaling,
            batch_size,
            forward_fn,
        )
        results.append((record.id, output))

    return results
