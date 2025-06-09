"""Augmentation module for transforming samples in a dataset."""

import numpy as np
import torch


class RandomXYSwap:
    """Random transformation swapping x and y coordinates"""

    def __init__(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample):
        transformed_sample = (
            np.stack((sample[:, 1::2], sample[:, ::2]), axis=2).reshape(sample.shape)
            if self.rng.random() < 0.5
            else sample
        )
        return transformed_sample


class PoseToTensor:
    """Extract the position variable from a record"""

    def __call__(self, sample):
        """Extract the position variable from a record."""
        return torch.Tensor(
            sample.stack(features=("individuals", "keypoints", "space")).position.values
        )
