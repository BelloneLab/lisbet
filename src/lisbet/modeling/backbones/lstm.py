"""LSTM Backbone for LISBET."""

from typing import Any

import torch
from torch import nn

from lisbet.modeling.backbones.base import BackboneInterface


class LSTMBackbone(BackboneInterface):
    """
    LSTM backbone for sequence modeling.

    Parameters
    ----------
    feature_dim : int
        Dimension of the input features.
    embedding_dim : int
        Dimension of the output embeddings.
    hidden_dim : int
        Dimension of the LSTM hidden state.
    num_layers : int
        Number of LSTM layers.
    """

    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(feature_dim, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        print("LSTMBackbone initialized")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM backbone.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, feature_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        x = self.input_proj(x)
        out, _ = self.lstm(x)
        out = self.output_proj(out)
        return out

    def get_config(self) -> dict[str, Any]:
        """Get the configuration dictionary for this backbone."""
        return {
            "feature_dim": self.feature_dim,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LSTMBackbone":
        """Create an LSTMBackbone instance from a configuration dictionary."""
        return cls(
            feature_dim=config["feature_dim"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
        )
