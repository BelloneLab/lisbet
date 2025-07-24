"""Model factory utilities for LISBET.

This module provides functions to create models from configuration Pydantic models,
including support for built-in transformer presets and parameter overrides.

All configuration objects must be Pydantic models with fields matching the expected
model constructor arguments.

Example
-------
from lisbet.config.presets import instantiate_backbone_preset
from lisbet.modeling.factory import create_model_from_config

cfg = instantiate_backbone_preset("transformer-base", feature_dim=32, embedding_dim=32)
model = create_model_from_config(cfg)
"""

from lisbet.config.schemas import ModelConfig
from lisbet.modeling import (
    EmbeddingHead,
    FrameClassificationHead,
    MultiTaskModel,
    TransformerBackbone,
    WindowClassificationHead,
)
from lisbet.modeling.backbones.lstm import LSTMBackbone

# Registry for backbone types (future extensibility)
BACKBONE_REGISTRY = {
    "transformer": TransformerBackbone,
    "lstm": LSTMBackbone,
}

# Registry for head types (future extensibility)
HEAD_REGISTRY = {
    "frame_classification": FrameClassificationHead,
    "window_classification": WindowClassificationHead,
    "embedding": EmbeddingHead,
}


def create_model_from_config(model_config: ModelConfig) -> MultiTaskModel:
    """
    Create a LISBET model from a configuration Pydantic model and head definitions.

    Parameters
    ----------
    model_config : ModelConfig
        Configuration Pydantic model instance

    Returns
    -------
    MultiTaskModel
        Instantiated LISBET model.

    Raises
    ------
    ValueError
        If the backbone type is unknown or a task_id is unrecognized.

    Notes
    -----
    - The config must be a Pydantic model with fields matching the backbone constructor.
    - The backbone config must have a 'type' field for discrimination.
    """
    backbone_type = model_config.backbone.type
    backbone_cls = BACKBONE_REGISTRY.get(backbone_type)
    if backbone_cls is None:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
    backbone_kwargs = model_config.backbone.model_dump(exclude={"type"})
    backbone = backbone_cls(**backbone_kwargs)

    # Build heads for each task
    heads = {}
    for task_id, head_cfg in model_config.out_heads.items():
        if task_id == "embedding":
            heads[task_id] = EmbeddingHead(
                output_token_idx=head_cfg.get("output_token_idx", -1)
            )
        elif task_id in ("multiclass", "multilabel"):
            heads[task_id] = FrameClassificationHead(
                output_token_idx=head_cfg.get("output_token_idx", -1),
                input_dim=backbone_kwargs["embedding_dim"],
                num_classes=head_cfg["num_classes"],
                hidden_dim=head_cfg.get("hidden_dim"),
            )
        elif task_id in ("cons", "order", "shift", "warp"):
            heads[task_id] = WindowClassificationHead(
                input_dim=backbone_kwargs["embedding_dim"],
                num_classes=head_cfg.get("num_classes", 1),
                hidden_dim=head_cfg.get("hidden_dim"),
            )
        else:
            raise ValueError(f"Unknown task_id: {task_id}")

    return MultiTaskModel(backbone, heads)
