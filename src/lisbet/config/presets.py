"""Built-in configuration presets for LISBET transformer models.

This module provides standard transformer model configurations as Python dataclasses.
Presets are intended to simplify model creation and ensure consistency across
experiments. All presets use modern Python type hints and numpy-style docstrings.

Presets
-------
- transformer-small: Small transformer for quick tests.
- transformer-base: Default/base transformer (matches CLI defaults).
- transformer-large: Large transformer for high-capacity experiments.
"""

from lisbet.config.schemas import BackboneConfig

TRANSFORMER_PRESETS = {
    "transformer-small": BackboneConfig(
        model_type="transformer",
        embedding_dim=16,
        hidden_dim=32,
        num_heads=2,
        num_layers=1,
    ),
    "transformer-base": BackboneConfig(
        model_type="transformer",
        embedding_dim=32,
        hidden_dim=128,
        num_heads=4,
        num_layers=4,
    ),
    "transformer-large": BackboneConfig(
        model_type="transformer",
        embedding_dim=64,
        hidden_dim=256,
        num_heads=8,
        num_layers=8,
    ),
}
