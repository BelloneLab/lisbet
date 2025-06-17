"""Configuration system for LISBET Enhanced Backbone System.

This package provides structured configuration classes for defining model
architectures, training parameters, and experiment settings.
"""

from lisbet.config.models import (
    BackboneConfig,
    DataConfig,
    ExperimentConfig,
    TaskConfig,
    TrainingConfig,
    apply_overrides,
)
from lisbet.config.presets import (
    create_experiment_config,
    get_classification_task_presets,
    get_default_data_config,
    get_default_training_config,
    get_transformer_backbone_presets,
    list_available_presets,
)

__all__ = [
    # Configuration dataclasses
    "BackboneConfig",
    "DataConfig",
    "ExperimentConfig",
    "TaskConfig",
    "TrainingConfig",
    # Configuration utilities
    "apply_overrides",
    # Preset functions
    "create_experiment_config",
    "get_classification_task_presets",
    "get_default_data_config",
    "get_default_training_config",
    "get_transformer_backbone_presets",
    "list_available_presets",
]

__doc__ = """
Configuration System
===================

The LISBET configuration system provides a clean, structured approach to
defining experiments through dataclasses and presets.

Basic Usage
-----------
Create an experiment configuration from presets:

>>> from lisbet.config import create_experiment_config
>>> config = create_experiment_config(
...     experiment_name="my_experiment",
...     backbone_preset="transformer-base",
...     task_presets=["frame_classification"]
... )

Available Presets
-----------------
List all available presets:

>>> from lisbet.config import list_available_presets
>>> presets = list_available_presets()
>>> print(presets)

Custom Configuration
--------------------
Create custom configurations using the dataclasses:

>>> from lisbet.config import ExperimentConfig, BackboneConfig, TaskConfig
>>> backbone = BackboneConfig(
...     type="transformer",
...     feature_dim=64,
...     embedding_dim=256,
...     hidden_dim=512,
...     num_heads=8,
...     num_layers=6,
...     max_length=1024
... )
"""
