"""Metrics for evaluating contrastive learning quality."""

import torch
from torchmetrics import Metric


class AlignmentMetric(Metric):
    """Measures alignment of positive pairs in contrastive learning.
    
    Alignment quantifies how close positive pairs are in the embedding space.
    Lower values indicate better alignment (positive pairs are closer).
    
    This metric computes the expected squared L2 distance between positive pairs:
    Alignment = E[||f(x) - f(x')||^2]
    where x and x' are augmented views of the same sample.
    
    References
    ----------
    Wang & Isola, "Understanding Contrastive Representation Learning through
    Alignment and Uniformity on the Hypersphere", ICML 2020.
    https://arxiv.org/abs/2005.10242
    """

    def __init__(self):
        super().__init__()
        self.add_state("total_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, z_i: torch.Tensor, z_j: torch.Tensor):
        """Update metric with a batch of positive pairs.
        
        Parameters
        ----------
        z_i : torch.Tensor
            First view embeddings of shape (batch_size, embedding_dim).
        z_j : torch.Tensor
            Second view embeddings of shape (batch_size, embedding_dim).
        """
        # Squared L2 distance between positive pairs
        dist = torch.sum((z_i - z_j) ** 2, dim=-1).mean()
        self.total_dist += dist
        self.count += 1

    def compute(self):
        """Compute the alignment metric.
        
        Returns
        -------
        torch.Tensor
            Average squared L2 distance across all positive pairs.
        """
        return self.total_dist / self.count


class UniformityMetric(Metric):
    """Measures uniformity of embeddings on the hypersphere.
    
    Uniformity quantifies how evenly embeddings are distributed on the unit
    hypersphere. More negative values indicate better uniformity (embeddings
    are more evenly spread out).
    
    This metric computes:
    Uniformity = log E[e^(-t * ||f(x) - f(y)||^2)]
    where x and y are different samples, and t is a temperature parameter.
    
    Parameters
    ----------
    t : float, optional
        Temperature parameter. Default is 2.
    
    References
    ----------
    Wang & Isola, "Understanding Contrastive Representation Learning through
    Alignment and Uniformity on the Hypersphere", ICML 2020.
    https://arxiv.org/abs/2005.10242
    """

    def __init__(self, t: float = 2.0):
        super().__init__()
        self.t = t
        self.add_state(
            "total_uniform", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, z: torch.Tensor):
        """Update metric with a batch of embeddings.
        
        Parameters
        ----------
        z : torch.Tensor
            Normalized embeddings of shape (batch_size, embedding_dim).
            Should be L2-normalized (on unit hypersphere).
        """
        # Compute pairwise squared L2 distances
        # pdist returns distances for all pairs (i, j) where i < j
        pdist = torch.pdist(z, p=2)
        
        # Compute uniformity: log of average exp(-t * distance^2)
        uniform = torch.log(torch.exp(-self.t * pdist**2).mean() + 1e-8)
        
        self.total_uniform += uniform
        self.count += 1

    def compute(self):
        """Compute the uniformity metric.
        
        Returns
        -------
        torch.Tensor
            Average uniformity score (more negative = better uniformity).
        """
        return self.total_uniform / self.count
