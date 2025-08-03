"""TCN Backbone for LISBET.

Implements a Temporal Convolutional Network (TCN) with dilated convolutions and
residual connections.

Reference:
    Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic
    Convolutional and Recurrent Networks for Sequence Modeling. arXiv:1803.01271.
"""

from typing import Any

import torch
from torch import nn

from lisbet.modeling.backbones.base import BackboneInterface


class Chomp1d(nn.Module):
    """Removes padding from the end of the sequence to ensure causality."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[..., : -self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """A single TCN block with dilated convolutions and residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = self.downsample(x)
        return self.final_relu(out + res)


class TCNBackbone(BackboneInterface):
    """
    Temporal Convolutional Network (TCN) backbone for sequence modeling.

    Parameters
    ----------
    feature_dim : int
        Dimension of the input features.
    embedding_dim : int
        Dimension of the output embeddings.
    hidden_dim : int
        Number of channels in the hidden layers.
    num_layers : int
        Number of temporal blocks (layers).
    kernel_size : int, optional
        Size of the convolutional kernel. Default: 3
    dilation_base : int, optional
        Base for the dilation factor. Default: 2
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.dropout = dropout

        layers = []
        in_channels = feature_dim
        for i in range(num_layers):
            out_channels = embedding_dim if i == num_layers - 1 else hidden_dim
            dilation = dilation_base**i
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TCN backbone.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, feature_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        # Convert to (batch, features, seq)
        x = x.transpose(1, 2)
        out = self.network(x)
        # Convert back to (batch, seq, features)
        out = out.transpose(1, 2)
        return out

    def get_config(self) -> dict[str, Any]:
        """Get the configuration dictionary for this backbone."""
        return {
            "feature_dim": self.feature_dim,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "kernel_size": self.kernel_size,
            "dilation_base": self.dilation_base,
            "dropout": self.dropout,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TCNBackbone":
        """Create a TCNBackbone instance from a configuration dictionary."""
        return cls(
            feature_dim=config["feature_dim"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            kernel_size=config.get("kernel_size", 3),
            dilation_base=config.get("dilation_base", 2),
            dropout=config.get("dropout", 0.0),
        )
