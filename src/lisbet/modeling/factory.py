"""Factory functions for creating models from configurations.

This module provides factory functions that create model components and complete
models from configuration objects, maintaining compatibility with direct parameter
passing while adding structured configuration support.
"""

from typing import Any

from torch import nn

from lisbet.config.models import BackboneConfig, ExperimentConfig, TaskConfig
from lisbet.modeling.backbones.transformer import TransformerBackbone
from lisbet.modeling.heads.classification import (
    FrameClassificationHead,
    WindowClassificationHead,
)
from lisbet.modeling.heads.embedding import EmbeddingHead
from lisbet.modeling.models import MultiTaskModel


def create_backbone_from_config(config: BackboneConfig) -> nn.Module:
    """Create a backbone model from configuration.

    Parameters
    ----------
    config : BackboneConfig
        Configuration specifying the backbone architecture and parameters.

    Returns
    -------
    nn.Module
        Backbone model instance.

    Raises
    ------
    ValueError
        If an unsupported backbone type is specified.
    """
    if config.type == "transformer":
        return TransformerBackbone(
            feature_dim=config.feature_dim,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_length=config.max_length,
        )
    else:
        raise ValueError(f"Unsupported backbone type: {config.type}")


def create_task_head_from_config(task_config: TaskConfig, input_dim: int) -> nn.Module:
    """Create a task head from configuration.

    Parameters
    ----------
    task_config : TaskConfig
        Configuration specifying the task head type and parameters.
    input_dim : int
        Input dimension from the backbone (embedding dimension).

    Returns
    -------
    nn.Module
        Task head model instance.

    Raises
    ------
    ValueError
        If an unsupported task type is specified or required parameters are missing.
    """
    if task_config.task_type == "frame_classification":
        if task_config.num_classes is None:
            raise ValueError("num_classes is required for frame_classification tasks")
        return FrameClassificationHead(
            output_token_idx=task_config.output_token_idx,
            input_dim=input_dim,
            num_classes=task_config.num_classes,
            hidden_dim=task_config.hidden_dim,
        )
    elif task_config.task_type == "window_classification":
        if task_config.num_classes is None:
            raise ValueError("num_classes is required for window_classification tasks")
        return WindowClassificationHead(
            input_dim=input_dim,
            num_classes=task_config.num_classes,
            hidden_dim=task_config.hidden_dim,
        )
    elif task_config.task_type == "embedding":
        return EmbeddingHead(
            output_token_idx=task_config.output_token_idx,
            input_dim=input_dim,
        )
    else:
        raise ValueError(f"Unsupported task type: {task_config.task_type}")


def create_model_from_config(config: ExperimentConfig) -> MultiTaskModel:
    """Create a complete multi-task model from experiment configuration.

    Parameters
    ----------
    config : ExperimentConfig
        Complete experiment configuration including backbone and task specifications.

    Returns
    -------
    MultiTaskModel
        Multi-task model with backbone and all specified task heads.

    Raises
    ------
    ValueError
        If the configuration is invalid or specifies unsupported components.
    """
    # Validate configuration first
    config.validate()

    # Create backbone
    backbone = create_backbone_from_config(config.backbone)

    # Create task heads
    task_heads = {}
    for task_config in config.tasks:
        head = create_task_head_from_config(task_config, config.backbone.embedding_dim)
        task_heads[task_config.task_id] = head

    return MultiTaskModel(backbone=backbone, task_heads=task_heads)


def create_transformer_model(
    feature_dim: int,
    embedding_dim: int,
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    max_length: int,
    task_heads: dict[str, nn.Module],
) -> MultiTaskModel:
    """Create a transformer-based multi-task model with direct parameters.

    This function maintains compatibility with the existing direct parameter
    passing approach while internally using the new configuration system.

    Parameters
    ----------
    feature_dim : int
        Dimension of input features.
    embedding_dim : int
        Dimension of output embeddings.
    hidden_dim : int
        Dimension of feedforward network in transformer layers.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    max_length : int
        Maximum sequence length for positional embeddings.
    task_heads : dict[str, nn.Module]
        Dictionary mapping task IDs to task head modules.

    Returns
    -------
    MultiTaskModel
        Multi-task model with transformer backbone and specified task heads.
    """
    # Create backbone using direct parameters
    backbone = TransformerBackbone(
        feature_dim=feature_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_length=max_length,
    )

    return MultiTaskModel(backbone=backbone, task_heads=task_heads)


def create_frame_classification_head(
    output_token_idx: int,
    input_dim: int,
    num_classes: int,
    hidden_dim: int | None = None,
) -> FrameClassificationHead:
    """Create a frame classification head with direct parameters.

    Parameters
    ----------
    output_token_idx : int
        Index of the token to use for classification.
    input_dim : int
        Input dimension from the backbone.
    num_classes : int
        Number of output classes.
    hidden_dim : int or None, optional
        Hidden dimension for MLP. If None, uses single linear layer.

    Returns
    -------
    FrameClassificationHead
        Frame classification head instance.
    """
    return FrameClassificationHead(
        output_token_idx=output_token_idx,
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
    )


def create_window_classification_head(
    input_dim: int,
    num_classes: int,
    hidden_dim: int | None = None,
) -> WindowClassificationHead:
    """Create a window classification head with direct parameters.

    Parameters
    ----------
    input_dim : int
        Input dimension from the backbone.
    num_classes : int
        Number of output classes.
    hidden_dim : int or None, optional
        Hidden dimension for MLP. If None, uses single linear layer.

    Returns
    -------
    WindowClassificationHead
        Window classification head instance.
    """
    return WindowClassificationHead(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
    )


def create_embedding_head(
    output_token_idx: int,
) -> EmbeddingHead:
    """Create an embedding head with direct parameters.

    Parameters
    ----------
    output_token_idx : int
        Index of the token to use for embedding extraction.

    Returns
    -------
    EmbeddingHead
        Embedding head instance.
    """
    return EmbeddingHead(
        output_token_idx=output_token_idx,
    )


def get_model_info(model: MultiTaskModel) -> dict[str, Any]:
    """Get detailed information about a model's architecture.

    Parameters
    ----------
    model : MultiTaskModel
        Model to analyze.

    Returns
    -------
    dict[str, Any]
        Dictionary containing model architecture information including
        parameter counts, layer details, and task configurations.
    """
    info = {
        "backbone": {
            "type": model.backbone.__class__.__name__,
            "config": model.backbone.get_config()
            if hasattr(model.backbone, "get_config")
            else {},
            "num_parameters": sum(p.numel() for p in model.backbone.parameters()),
        },
        "task_heads": {},
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }

    for task_id, head in model.task_heads.items():
        info["task_heads"][task_id] = {
            "type": head.__class__.__name__,
            "config": head.get_config() if hasattr(head, "get_config") else {},
            "num_parameters": sum(p.numel() for p in head.parameters()),
        }

    return info
