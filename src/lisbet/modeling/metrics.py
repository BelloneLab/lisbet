"""Metrics for evaluating contrastive learning quality."""

import torch
from torchmetrics import Metric
import torch.nn.functional as F

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

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        self.add_state("total_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, z_i: torch.Tensor, z_j: torch.Tensor):
        if self.normalize:
            z_i = F.normalize(z_i, p=2, dim=-1)
            z_j = F.normalize(z_j, p=2, dim=-1)

        # Sum of squared L2 distances across all pairs in the batch
        dist = torch.sum((z_i - z_j) ** 2, dim=-1).sum()
        
        self.total_dist += dist
        self.num_samples += z_i.shape[0]

    def compute(self):
        return self.total_dist / torch.clamp(self.num_samples, min=1)


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

    def __init__(self, t: float = 2.0, normalize: bool = True):
        super().__init__()
        self.t = t
        self.normalize = normalize
        
        # Accumulate total exponential distances and total pair count
        self.add_state(
            "total_exp_kernel", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("num_pairs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, z: torch.Tensor):
        if self.normalize:
            z = F.normalize(z, p=2, dim=-1)

        # Compute pairwise squared L2 distances for upper triangle (i < j)
        pdist = torch.pdist(z, p=2)

        # Sum of Gaussian kernels for this batch
        exp_kernel_sum = torch.exp(-self.t * (pdist ** 2)).sum()

        self.total_exp_kernel += exp_kernel_sum
        self.num_pairs += pdist.shape[0]

    def compute(self):
        # Average over all pairs, then take log once at epoch end
        avg_kernel = self.total_exp_kernel / torch.clamp(self.num_pairs, min=1)
        return torch.log(avg_kernel + 1e-8)