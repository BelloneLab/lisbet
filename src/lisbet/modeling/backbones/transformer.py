"""Transformer Backbone for Lisbet."""

from typing import Any

import torch
from torch import nn

from lisbet.modeling.backbones.base import BackboneInterface
from lisbet.modeling.modules_extra import PosEmbedding


class TransformerBackbone(BackboneInterface):
    """Transformer backbone for sequence modeling.

    A transformer-based backbone that processes input sequences using
    self-attention mechanisms. The backbone includes frame embedding,
    positional embedding, transformer encoder layers, and layer normalization.

    Parameters
    ----------
    feature_dim : int
        Dimension of the input features.
    embedding_dim : int
        Dimension of the output embeddings.
    hidden_dim : int
        Dimension of the feedforward network inside transformer layers.
    num_heads : int
        Number of attention heads in the multi-head attention mechanism.
    num_layers : int
        Number of transformer encoder layers.
    max_length : int
        Maximum sequence length for positional embeddings.

    Attributes
    ----------
    frame_embedder : nn.Linear
        Linear layer for embedding input frames.
    pos_embedder : PosEmbedding
        Positional embedding module.
    transformer_encoder : nn.TransformerEncoder
        Stack of transformer encoder layers.
    layer_norm : nn.LayerNorm
        Layer normalization applied to the output.
    """

    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        max_length: int,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_length = max_length

        self.frame_embedder = nn.Linear(feature_dim, embedding_dim)
        self.pos_embedder = PosEmbedding(max_length, embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(
            embedding_dim,
            num_heads,
            hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers, enable_nested_tensor=False
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer backbone.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, feature_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        x = self.frame_embedder(x)
        x = self.pos_embedder(x)
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        return x

    def get_config(self) -> dict[str, Any]:
        """Get the configuration dictionary for this backbone.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary containing all parameters needed
            to recreate this backbone instance.
        """
        return {
            "feature_dim": self.feature_dim,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "max_length": self.max_length,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TransformerBackbone":
        """Create a TransformerBackbone instance from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary containing all parameters needed
            to create the backbone instance.

        Returns
        -------
        TransformerBackbone
            New TransformerBackbone instance created from the configuration.
        """
        return cls(
            feature_dim=config["feature_dim"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            max_length=config["max_length"],
        )
