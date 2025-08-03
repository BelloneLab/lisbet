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

BACKBONE_PRESETS = {
    "transformer-small": {
        "type": "transformer",
        "embedding_dim": 16,
        "hidden_dim": 32,
        "num_heads": 2,
        "num_layers": 1,
    },
    "transformer-base": {
        "type": "transformer",
        "embedding_dim": 32,
        "hidden_dim": 128,
        "num_heads": 4,
        "num_layers": 4,
    },
    "transformer-large": {
        "type": "transformer",
        "embedding_dim": 64,
        "hidden_dim": 256,
        "num_heads": 8,
        "num_layers": 8,
    },
    "lstm-base": {
        "type": "lstm",
        "embedding_dim": 32,
        "hidden_dim": 128,
        "num_layers": 2,
    },
    "tcn-base": {
        "type": "tcn",
        "embedding_dim": 32,
        "hidden_dim": 64,
        "num_layers": 4,
        "kernel_size": 16,
        "dilation_base": 2,
        "dropout": 0.1,
    },
}
