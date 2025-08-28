"""Classification heads for frame and window classification tasks."""

from typing import Any

import torch
from torch import nn

from lisbet.modeling.modules_extra import MLP


class FrameClassificationHead(nn.Module):
    """Frame-level classification head.

    This head selects a specific token from the sequence (typically the last one)
    and applies a classification layer to predict frame-level labels.

    Parameters
    ----------
    output_token_idx : int
        Index of the token to use for classification (e.g., -1 for last token).
    input_dim : int
        Dimension of the input embeddings (formerly emb_dim).
    num_classes : int
        Number of output classes (formerly out_dim).
    hidden_dim : int or None, optional
        Dimension of the hidden layer. If None, uses a single linear layer.
        If provided, uses an MLP with the specified hidden dimension.

    Attributes
    ----------
    output_token_idx : int
        Index of the token used for classification.
    logits : nn.Module
        Classification layer (either Linear or MLP).
    """

    def __init__(
        self,
        output_token_idx: int,
        input_dim: int,
        num_classes: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.output_token_idx = output_token_idx
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.logits = (
            nn.Linear(input_dim, num_classes)
            if hidden_dim is None
            else MLP(input_dim, num_classes, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the frame classification head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns
        -------
        torch.Tensor
            Classification logits of shape (batch_size, num_classes).
        """
        x = x[:, self.output_token_idx]
        x = self.logits(x)
        return x

    def get_config(self) -> dict[str, Any]:
        """Get the configuration dictionary for this head.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary containing all parameters needed
            to recreate this head instance.
        """
        return {
            "output_token_idx": self.output_token_idx,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FrameClassificationHead":
        """Create a FrameClassificationHead instance from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary containing all parameters needed
            to create the head instance.

        Returns
        -------
        FrameClassificationHead
            New FrameClassificationHead instance created from the configuration.
        """
        return cls(**config)


class WindowClassificationHead(nn.Module):
    """Window-level classification head.

    This head performs global max pooling over the sequence dimension and
    applies a classification layer to predict window-level labels.

    Parameters
    ----------
    input_dim : int
        Dimension of the input embeddings (formerly emb_dim).
    num_classes : int
        Number of output classes (formerly out_dim).
    hidden_dim : int or None, optional
        Dimension of the hidden layer. If None, uses a single linear layer.
        If provided, uses an MLP with the specified hidden dimension.

    Attributes
    ----------
    logits : nn.Module
        Classification layer (either Linear or MLP).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.logits = (
            nn.Linear(input_dim, num_classes)
            if hidden_dim is None
            else MLP(input_dim, num_classes, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the window classification head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns
        -------
        torch.Tensor
            Classification logits of shape (batch_size, num_classes).
        """
        x, _ = torch.max(x, dim=1)
        x = self.logits(x)
        return x

    def get_config(self) -> dict[str, Any]:
        """Get the configuration dictionary for this head.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary containing all parameters needed
            to recreate this head instance.
        """
        return {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WindowClassificationHead":
        """Create a WindowClassificationHead instance from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary containing all parameters needed
            to create the head instance.

        Returns
        -------
        WindowClassificationHead
            New WindowClassificationHead instance created from the configuration.
        """
        return cls(**config)
