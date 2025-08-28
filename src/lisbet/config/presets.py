"""Built-in configuration presets for LISBET.

This module provides standard model configurations as Python dataclasses. Presets are
intended to simplify model creation and ensure consistency across experiments.

Presets
-------
- transformer-base: Default/base transformer (matches CLI defaults).
- lstm-base: Standard LSTM backbone with default parameters.
- tcn-base: Standard TCN backbone with default parameters.
"""

BACKBONE_PRESETS = {
    "transformer-base": {
        "type": "transformer",
        "embedding_dim": 32,
        "hidden_dim": 128,
        "num_heads": 4,
        "num_layers": 4,
    },
    "lstm-base": {
        "type": "lstm",
        "embedding_dim": 32,
        "hidden_dim": 64,
        "num_layers": 2,
    },
    "tcn-base": {
        "type": "tcn",
        "embedding_dim": 32,
        "hidden_dim": 64,
        "num_layers": 2,
        "kernel_size": 8,
        "dilation_base": 16,
        "dropout": 0.0,
    },
}
