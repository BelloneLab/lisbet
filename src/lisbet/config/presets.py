"""Built-in configuration presets for LISBET.

This module provides standard model configurations as Python dataclasses. Presets are
intended to simplify model creation and ensure consistency across experiments.

Presets
-------
- transformer-small: Small transformer for quick tests.
- transformer-base: Default/base transformer (matches CLI defaults).
- transformer-large: Large transformer for high-capacity experiments.
- lstm-base: Standard LSTM backbone.
"""

from lisbet.config.schemas import LSTMBackboneConfig, TransformerBackboneConfig

BACKBONE_PRESETS = {
    "transformer-small": TransformerBackboneConfig(
        feature_dim=None,
        embedding_dim=16,
        hidden_dim=32,
        num_heads=2,
        num_layers=1,
        max_length=None,
    ),
    "transformer-base": TransformerBackboneConfig(
        feature_dim=None,
        embedding_dim=32,
        hidden_dim=128,
        num_heads=4,
        num_layers=4,
        max_length=None,
    ),
    "transformer-large": TransformerBackboneConfig(
        feature_dim=None,
        embedding_dim=64,
        hidden_dim=256,
        num_heads=8,
        num_layers=8,
        max_length=None,
    ),
    "lstm-base": LSTMBackboneConfig(
        feature_dim=None,
        embedding_dim=32,
        hidden_dim=128,
        num_layers=2,
    ),
}
