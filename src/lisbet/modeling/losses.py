"""Loss functions for LISBET training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning.
    
    Computes the InfoNCE (Normalized Temperature-scaled Cross Entropy) loss
    for contrastive learning. Given embeddings from original and transformed
    views, maximizes agreement between positive pairs while distinguishing
    from negative pairs.
    
    Parameters
    ----------
    temperature : float, optional
        Temperature parameter for scaling. Default is 0.07.
    reduction : str, optional
        Reduction method ('mean', 'sum', or 'none'). Default is 'mean'.
    
    References
    ----------
    van den Oord et al., "Representation Learning with Contrastive Predictive
    Coding", 2018. https://arxiv.org/abs/1807.03748
    
    Chen et al., "A Simple Framework for Contrastive Learning of Visual
    Representations" (SimCLR), 2020. https://arxiv.org/abs/2002.05709
    
    Notes
    -----
    The loss is symmetric: both z_i→z_j and z_j→z_i contribute to the final loss.
    Gradients flow through the projection head to the backbone (following SimCLR).
    The projection head is used only during training and discarded for inference.
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self, z_i: torch.Tensor, z_j: torch.Tensor
    ) -> torch.Tensor:
        """Compute InfoNCE loss.
        Parameters
        ----------
        z_i : torch.Tensor
            Projected embeddings from original windows.
            Shape: (batch_size, projection_dim)
        z_j : torch.Tensor
            Projected embeddings from transformed windows.
            Shape: (batch_size, projection_dim)
        Returns
        -------
        torch.Tensor
            Scalar loss value (if reduction='mean' or 'sum'),
            or per-sample losses (if reduction='none').
        
        Notes
        -----
        The loss is computed as:
        1. Concatenate z_i and z_j to form a batch of 2N samples
        2. Compute pairwise cosine similarities
        3. For each sample, positive pair is its transformed counterpart,
           all other 2N-2 samples are negatives
        4. Apply temperature scaling and compute cross-entropy
        """
        batch_size = z_i.shape[0]

        # Concatenate embeddings: [z_i; z_j]
        # Shape: (2 * batch_size, projection_dim)
        z = torch.cat([z_i, z_j], dim=0)

        # Compute cosine similarity matrix
        # Shape: (2 * batch_size, 2 * batch_size)
        sim_matrix = F.cosine_similarity(
            z.unsqueeze(1), z.unsqueeze(0), dim=-1
        )

        # Scale by temperature
        sim_matrix = sim_matrix / self.temperature

        # Create labels: for each i, positive is i + batch_size (and vice versa)
        # First batch_size samples: positives are at indices [batch_size, 2*batch_size)
        # Second batch_size samples: positives are at indices [0, batch_size)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z.device)

        # Mask out self-similarities (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        # Compute cross-entropy loss
        # sim_matrix is logits, labels are positive indices
        loss = F.cross_entropy(
            sim_matrix, labels, reduction=self.reduction
        )

        return loss