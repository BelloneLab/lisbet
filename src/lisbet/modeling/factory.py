"""Model factory utilities for LISBET.

This module provides functions to create models from configuration dataclasses,
including support for built-in transformer presets and parameter overrides.

All configuration objects must be dataclasses with fields matching the expected
model constructor arguments.

Example
-------
from lisbet.config.presets import TransformerBaseConfig
from lisbet.modeling.factory import create_model_from_config

cfg = TransformerBaseConfig(feature_dim=32, embedding_dim=32)
model = create_model_from_config(cfg)
"""

from dataclasses import asdict, is_dataclass
from typing import Any

from lisbet.modeling import (
    EmbeddingHead,
    FrameClassificationHead,
    MultiTaskModel,
    TransformerBackbone,
    WindowClassificationHead,
)

# Registry for backbone types (future extensibility)
BACKBONE_REGISTRY: dict[str, type] = {
    "transformer": TransformerBackbone,
}

# Registry for head types (future extensibility)
HEAD_REGISTRY: dict[str, type] = {
    "frame_classification": FrameClassificationHead,
    "window_classification": WindowClassificationHead,
    "embedding": EmbeddingHead,
}


def create_model_from_config(
    config: Any,
    out_heads: dict[str, dict] = None,
    backbone_type: str = "transformer",
) -> MultiTaskModel:
    """
    Create a LISBET model from a configuration dataclass and head definitions.

    Parameters
    ----------
    config : dataclass
        Configuration dataclass instance (e.g., TransformerBaseConfig).
    out_heads : dict, optional
        Dictionary mapping task IDs to head configuration dicts.
        Example: {"multiclass": {"num_classes": 4}, "embedding": {}}
    backbone_type : str, default="transformer"
        Backbone type to use (currently only "transformer" is supported).

    Returns
    -------
    MultiTaskModel
        Instantiated LISBET model.

    Raises
    ------
    ValueError
        If the config is not a dataclass or backbone_type is unknown.

    Notes
    -----
    - The config must be a dataclass with fields matching the backbone constructor.
    - out_heads must be provided for all tasks; each value is passed as kwargs to the
      head.
    """
    if not is_dataclass(config):
        raise ValueError("Config must be a dataclass instance.")

    backbone_cls = BACKBONE_REGISTRY.get(backbone_type)
    if backbone_cls is None:
        raise ValueError(f"Unknown backbone type: {backbone_type}")

    backbone_kwargs = asdict(config)
    backbone = backbone_cls(**backbone_kwargs)

    if out_heads is None:
        raise ValueError("out_heads must be provided to define model outputs.")

    # Build heads for each task
    heads = {}
    for task_id, head_cfg in out_heads.items():
        if task_id == "embedding":
            head_cls = EmbeddingHead
            # output_token_idx is typically -1 (last token)
            heads[task_id] = head_cls(
                output_token_idx=head_cfg.get("output_token_idx", -1)
            )
        elif task_id in ("multiclass", "multilabel"):
            head_cls = FrameClassificationHead
            heads[task_id] = head_cls(
                output_token_idx=head_cfg.get("output_token_idx", -1),
                input_dim=backbone_kwargs["embedding_dim"],
                num_classes=head_cfg["num_classes"],
                hidden_dim=head_cfg.get("hidden_dim"),
            )
        elif task_id in ("cons", "order", "shift", "warp"):
            head_cls = WindowClassificationHead
            heads[task_id] = head_cls(
                input_dim=backbone_kwargs["embedding_dim"],
                num_classes=head_cfg.get("num_classes", 1),
                hidden_dim=head_cfg.get("hidden_dim"),
            )
        else:
            raise ValueError(f"Unknown task_id: {task_id}")

    return MultiTaskModel(backbone, heads)
