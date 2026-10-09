"""Projection head for contrastive learning tasks."""

from typing import Any

import torch
from torch import nn


class ProjectionMLP(nn.Module):
    """MLP with batch normalization for projection head.
    
    Following SimCLR and MoCo v2 architecture:
    Linear → BatchNorm → ReLU → Linear → BatchNorm
    
    Parameters
    ----------
    in_features : int
        Input dimension.
    out_features : int
        Output dimension.
    hidden_dim : int
        Hidden layer dimension.
    """

    def __init__(self, in_features: int, out_features: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the projection MLP.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features).
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_features).
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)  # SimCLR includes BN after final layer
        return x


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning.
    
    Projects embeddings into a lower-dimensional space where contrastive
    loss is computed. Typically used with InfoNCE loss.
    
    This head performs global max pooling over the sequence dimension (consistent
    with WindowClassificationHead), projects through an MLP with batch normalization,
    and optionally normalizes the output for use with cosine similarity-based losses.
    
    Parameters
    ----------
    input_dim : int
        Dimension of the input embeddings.
    projection_dim : int
        Dimension of the projection space.
    hidden_dim : int or None, optional
        Dimension of the hidden layer. If None, uses a single linear layer.
        If provided, uses an MLP with batch normalization (recommended).
    normalize : bool, optional
        Whether to L2-normalize the output. Default is True for InfoNCE.
    
    Attributes
    ----------
    projection : nn.Module
        Projection layer (either Linear or ProjectionMLP with BatchNorm).
    normalize : bool
        Whether to normalize outputs.
    
    Notes
    -----
    Following SimCLR design:
    - Gradients flow through projection head to backbone during training
    - Projection head is discarded after training; only backbone used for inference
    - Batch normalization improves stability and prevents mode collapse
    - Uses global max pooling (same as WindowClassificationHead) for consistency
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        hidden_dim: int | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.normalize = normalize

        # Use ProjectionMLP with BatchNorm if hidden_dim is provided
        if hidden_dim is None:
            self.projection = nn.Linear(input_dim, projection_dim)
        else:
            self.projection = ProjectionMLP(input_dim, projection_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the projection head.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_dim).
        Returns
        -------
        torch.Tensor
            Projected embeddings of shape (batch_size, projection_dim).
        """
        # Global max pooling over sequence (consistent with classification heads)
        x, _ = torch.max(x, dim=1)

        # Project
        x = self.projection(x)

        # Normalize if required (for cosine similarity-based losses)
        if self.normalize:
            x = torch.nn.functional.normalize(x, p=2, dim=-1)

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
            "projection_dim": self.projection_dim,
            "hidden_dim": self.hidden_dim,
            "normalize": self.normalize,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ProjectionHead":
        """Create a ProjectionHead instance from a configuration dictionary.
        
        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary containing all parameters needed
            to create the head instance.
        
        Returns
        -------
        ProjectionHead
            New ProjectionHead instance created from the configuration.
        """
        return cls(**config)