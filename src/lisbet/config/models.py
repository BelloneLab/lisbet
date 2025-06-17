"""Configuration dataclasses for LISBET Enhanced Backbone System.

This module provides structured configuration classes for defining model
architectures, training parameters, and experiment settings.
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing.

    Parameters
    ----------
    train_paths : list[str]
        List of paths to training data files.
    val_paths : list[str]
        List of paths to validation data files.
    test_paths : list[str], optional
        List of paths to test data files.
    window_size : int
        Size of the input window for sequence modeling.
    window_offset : int
        Offset for the output token relative to the window.
    input_features : list[str]
        List of feature names to use as input.
    preprocessing : dict[str, Any]
        Dictionary of preprocessing parameters.
    batch_size : int
        Batch size for data loading.
    num_workers : int
        Number of workers for data loading.
    pin_memory : bool
        Whether to pin memory for faster GPU transfer.
    """

    train_paths: list[str]
    val_paths: list[str]
    window_size: int
    window_offset: int
    input_features: list[str]
    test_paths: list[str] = field(default_factory=list)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class BackboneConfig:
    """Configuration for backbone architectures.

    Currently supports transformer-based backbones.

    Parameters
    ----------
    type : str
        Type of backbone architecture (e.g., "transformer").
    feature_dim : int
        Dimension of input features.
    embedding_dim : int
        Dimension of output embeddings.
    hidden_dim : int
        Dimension of feedforward network in transformer layers.
    num_heads : int
        Number of attention heads in multi-head attention.
    num_layers : int
        Number of transformer encoder layers.
    max_length : int
        Maximum sequence length for positional embeddings.
    dropout : float
        Dropout rate for regularization.
    activation : str
        Activation function to use in feedforward networks.
    """

    type: str
    feature_dim: int
    embedding_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    max_length: int
    dropout: float = 0.0
    activation: str = "gelu"


@dataclass
class TaskConfig:
    """Configuration for a specific task.

    Parameters
    ----------
    task_id : str
        Unique identifier for the task.
    task_type : str
        Type of task (e.g., "frame_classification", "window_classification",
        "embedding").
    num_classes : int, optional
        Number of output classes for classification tasks.
    output_token_idx : int, optional
        Index of the token to use for output (-1 for last token).
        Not used for window_classification tasks which use max pooling.
    hidden_dim : int, optional
        Hidden dimension for task head MLP. If None, uses single linear layer.
    loss_weight : float
        Weight for this task's loss in multi-task training.
    metrics : list[str]
        List of metrics to compute for this task.
    """

    task_id: str
    task_type: str
    output_token_idx: int | None = -1
    num_classes: int | None = None
    hidden_dim: int | None = None
    loss_weight: float = 1.0
    metrics: list[str] = field(default_factory=lambda: ["accuracy"])


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    learning_rate : float
        Learning rate for optimizer.
    batch_size : int
        Training batch size.
    seed : int
        Random seed for reproducibility.
    mixed_precision : bool
        Whether to use mixed precision training.
    """

    epochs: int
    learning_rate: float
    batch_size: int = 32
    seed: int = 1991
    mixed_precision: bool = False


@dataclass
class ExperimentConfig:
    """Top-level configuration combining all experiment settings.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment for logging and saving.
    run_id : str, optional
        Unique identifier for this run. If None, will be auto-generated.
    data : DataConfig
        Data loading and preprocessing configuration.
    backbone : BackboneConfig
        Backbone architecture configuration.
    tasks : list[TaskConfig]
        List of task configurations for multi-task learning.
    training : TrainingConfig
        Training hyperparameters configuration.
    output_dir : str
        Directory to save experiment outputs.
    seed : int, optional
        Random seed for reproducibility.
    tags : list[str]
        List of tags for experiment organization.
    notes : str
        Additional notes about the experiment.
    """

    experiment_name: str
    data: DataConfig
    backbone: BackboneConfig
    tasks: list[TaskConfig]
    training: TrainingConfig
    output_dir: str
    run_id: str | None = None
    seed: int | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def get_task_by_id(self, task_id: str) -> TaskConfig | None:
        """Get a task configuration by its ID.

        Parameters
        ----------
        task_id : str
            The task ID to search for.

        Returns
        -------
        TaskConfig or None
            The task configuration if found, None otherwise.
        """
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def get_task_ids(self) -> list[str]:
        """Get all task IDs in this experiment.

        Returns
        -------
        list[str]
            List of all task IDs.
        """
        return [task.task_id for task in self.tasks]

    def validate(self) -> None:
        """Validate the configuration for consistency.

        Raises
        ------
        ValueError
            If the configuration is invalid.
        """
        # Check for duplicate task IDs
        task_ids = self.get_task_ids()
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Duplicate task IDs found in configuration")

        # Validate backbone configuration
        if self.backbone.type not in ["transformer"]:
            raise ValueError(f"Unsupported backbone type: {self.backbone.type}")

        # Validate task configurations
        for task in self.tasks:
            if task.task_type in ["frame_classification", "window_classification"] and (
                task.num_classes is None or task.num_classes <= 0
            ):
                raise ValueError(f"Task {task.task_id} requires positive num_classes")

            # Validate output_token_idx for tasks that need it
            if (
                task.task_type in ["frame_classification", "embedding"]
                and task.output_token_idx is None
            ):
                raise ValueError(
                    f"Task {task.task_id} of type {task.task_type} "
                    f"requires output_token_idx"
                )

        # Validate data configuration
        if not self.data.train_paths:
            raise ValueError("At least one training path must be specified")

        if not self.data.val_paths:
            raise ValueError("At least one validation path must be specified")

        if self.data.window_size <= 0:
            raise ValueError("Window size must be positive")

        if self.data.batch_size <= 0:
            raise ValueError("Batch size must be positive")


def apply_overrides(
    config: ExperimentConfig, overrides: dict[str, Any]
) -> ExperimentConfig:
    """Apply parameter overrides to an experiment configuration.

    Parameters
    ----------
    config : ExperimentConfig
        Base experiment configuration.
    overrides : dict[str, Any]
        Dictionary of parameter overrides using dot notation for nested parameters.
        For example: {"backbone.num_layers": 8, "training.learning_rate": 1e-3}

    Returns
    -------
    ExperimentConfig
        New configuration with overrides applied.
    """
    # Create a deep copy to avoid modifying the original
    new_config = copy.deepcopy(config)

    for key, value in overrides.items():
        if hasattr(new_config, key):
            setattr(new_config, key, value)
        else:
            # Handle nested parameters with dot notation
            if "." in key:
                parts = key.split(".", 1)
                section = parts[0]
                nested_key = parts[1]

                if section == "data" and hasattr(new_config.data, nested_key):
                    setattr(new_config.data, nested_key, value)
                elif section == "backbone" and hasattr(new_config.backbone, nested_key):
                    setattr(new_config.backbone, nested_key, value)
                elif section == "training" and hasattr(new_config.training, nested_key):
                    setattr(new_config.training, nested_key, value)
                else:
                    logging.warning(f"Unknown override parameter: {key}")
            else:
                logging.warning(f"Unknown override parameter: {key}")

    # Validate the updated configuration
    new_config.validate()
    return new_config
