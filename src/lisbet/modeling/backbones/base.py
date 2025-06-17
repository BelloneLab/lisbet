"""Abstract base class for backbone models."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class BackboneInterface(nn.Module, ABC):
    """Abstract base class for all backbone models.

    This interface defines the required methods that all backbone
    implementations must provide.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, output_dim).
        """
        pass

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Get the configuration dictionary for this backbone.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary containing all parameters needed
            to recreate this backbone instance.
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> "BackboneInterface":
        """Create a backbone instance from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary containing all parameters needed
            to create the backbone instance.

        Returns
        -------
        BackboneInterface
            New backbone instance created from the configuration.
        """
        pass
