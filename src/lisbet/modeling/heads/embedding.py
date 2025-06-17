"""Embedding head for extracting behavior embeddings."""

from torch import nn


class EmbeddingHead(nn.Module):
    """Embedding head for extracting behavior embeddings."""

    def __init__(self, output_token_idx):
        super().__init__()
        self.output_token_idx = output_token_idx

    def forward(self, x):
        x = x[:, self.output_token_idx]
        return x
