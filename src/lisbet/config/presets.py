"""Built-in presets for transformer configurations.

This module provides predefined configurations for common transformer
architectures used in the LISBET Enhanced Backbone System.
"""

from lisbet.config.models import (
    BackboneConfig,
    DataConfig,
    ExperimentConfig,
    TaskConfig,
    TrainingConfig,
)


def get_transformer_backbone_presets() -> dict[str, BackboneConfig]:
    """Get predefined transformer backbone configurations.

    Returns
    -------
    dict[str, BackboneConfig]
        Dictionary mapping preset names to backbone configurations.
    """
    return {
        "transformer-small": BackboneConfig(
            type="transformer",
            feature_dim=32,
            embedding_dim=128,
            hidden_dim=256,
            num_heads=4,
            num_layers=4,
            max_length=512,
            dropout=0.1,
            activation="gelu",
        ),
        "transformer-base": BackboneConfig(
            type="transformer",
            feature_dim=64,
            embedding_dim=256,
            hidden_dim=512,
            num_heads=8,
            num_layers=6,
            max_length=1024,
            dropout=0.1,
            activation="gelu",
        ),
        "transformer-large": BackboneConfig(
            type="transformer",
            feature_dim=128,
            embedding_dim=512,
            hidden_dim=1024,
            num_heads=16,
            num_layers=12,
            max_length=2048,
            dropout=0.1,
            activation="gelu",
        ),
    }


def get_default_data_config() -> DataConfig:
    """Get a default data configuration template.

    Returns
    -------
    DataConfig
        Default data configuration that can be customized.
    """
    return DataConfig(
        train_paths=[],
        val_paths=[],
        test_paths=[],
        window_size=64,
        window_offset=0,
        input_features=[],
        preprocessing={},
        batch_size=32,
        num_workers=4,
        pin_memory=True,
    )


def get_default_training_config() -> TrainingConfig:
    """Get a default training configuration.

    Returns
    -------
    TrainingConfig
        Default training configuration suitable for most experiments.
    """
    return TrainingConfig(
        epochs=10,
        learning_rate=1e-4,
        batch_size=32,
        seed=1991,
        mixed_precision=False,
    )


def get_classification_task_presets() -> dict[str, TaskConfig]:
    """Get predefined task configurations for classification.

    Returns
    -------
    dict[str, TaskConfig]
        Dictionary mapping task names to task configurations.
    """
    return {
        "frame_classification": TaskConfig(
            task_id="frame_classification",
            task_type="frame_classification",
            output_token_idx=-1,
            num_classes=2,
            hidden_dim=None,
            loss_weight=1.0,
            metrics=["accuracy", "f1"],
        ),
        "window_classification": TaskConfig(
            task_id="window_classification",
            task_type="window_classification",
            output_token_idx=None,
            num_classes=2,
            hidden_dim=None,
            loss_weight=1.0,
            metrics=["accuracy", "f1"],
        ),
        "embedding": TaskConfig(
            task_id="embedding",
            task_type="embedding",
            output_token_idx=-1,
            num_classes=None,
            hidden_dim=None,
            loss_weight=1.0,
            metrics=[],
        ),
    }


def create_experiment_config(
    experiment_name: str,
    backbone_preset: str = "transformer-base",
    task_presets: list[str] | None = None,
    **overrides,
) -> ExperimentConfig:
    """Create a complete experiment configuration from presets.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    backbone_preset : str, optional
        Name of the backbone preset to use. Default is "transformer-base".
    task_presets : list[str] or None, optional
        List of task preset names to include. If None, uses frame_classification.
    **overrides
        Additional configuration overrides.

    Returns
    -------
    ExperimentConfig
        Complete experiment configuration.

    Raises
    ------
    ValueError
        If an unknown preset name is provided.
    """
    # Get backbone configuration
    backbone_presets = get_transformer_backbone_presets()
    if backbone_preset not in backbone_presets:
        raise ValueError(f"Unknown backbone preset: {backbone_preset}")
    backbone_config = backbone_presets[backbone_preset]

    # Get task configurations
    if task_presets is None:
        task_presets = ["frame_classification"]

    task_preset_configs = get_classification_task_presets()
    task_configs = []
    for task_preset in task_presets:
        if task_preset not in task_preset_configs:
            raise ValueError(f"Unknown task preset: {task_preset}")
        task_configs.append(task_preset_configs[task_preset])

    # Create experiment configuration
    config = ExperimentConfig(
        experiment_name=experiment_name,
        data=get_default_data_config(),
        backbone=backbone_config,
        tasks=task_configs,
        training=get_default_training_config(),
        output_dir="./experiments",
        run_id=None,
        seed=42,
        tags=[],
        notes="",
    )

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Try to apply to nested configs
            if key.startswith("data."):
                nested_key = key[5:]  # Remove "data." prefix
                if hasattr(config.data, nested_key):
                    setattr(config.data, nested_key, value)
            elif key.startswith("backbone."):
                nested_key = key[9:]  # Remove "backbone." prefix
                if hasattr(config.backbone, nested_key):
                    setattr(config.backbone, nested_key, value)
            elif key.startswith("training."):
                nested_key = key[9:]  # Remove "training." prefix
                if hasattr(config.training, nested_key):
                    setattr(config.training, nested_key, value)

    return config


def list_available_presets() -> dict[str, list[str]]:
    """List all available presets.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping preset categories to available preset names.
    """
    return {
        "backbone": list(get_transformer_backbone_presets().keys()),
        "tasks": list(get_classification_task_presets().keys()),
    }
