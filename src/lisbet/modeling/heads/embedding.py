"""Embedding head for extracting behavior embeddings."""

from typing import Any

import torch
from torch import nn


class EmbeddingHead(nn.Module):
    """Embedding head for extracting behavior embeddings.

    This head selects a specific token from the sequence (typically the last one)
    and returns it as the behavior embedding without any additional transformation.

    Parameters
    ----------
    output_token_idx : int
        Index of the token to use for embedding extraction (e.g., -1 for last token).

    Attributes
    ----------
    output_token_idx : int
        Index of the token used for embedding extraction.
    """

    def __init__(self, output_token_idx: int) -> None:
        super().__init__()
        self.output_token_idx = output_token_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the embedding head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns
        -------
        torch.Tensor
            Embedding tensor of shape (batch_size, embedding_dim).
        """
        x = x[:, self.output_token_idx]
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
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EmbeddingHead":
        """Create an EmbeddingHead instance from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary containing all parameters needed
            to create the head instance.

        Returns
        -------
        EmbeddingHead
            New EmbeddingHead instance created from the configuration.
        """
        return cls(**config)
