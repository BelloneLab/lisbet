"""Classification heads for frame and window classification tasks."""

import torch
from torch import nn

from lisbet.modeling.modules_extra import MLP


class FrameClassificationHead(nn.Module):
    """Frame classifier."""

    def __init__(self, output_token_idx, emb_dim, out_dim, hidden_dim):
        super().__init__()
        self.output_token_idx = output_token_idx
        self.logits = (
            nn.Linear(emb_dim, out_dim)
            if hidden_dim is None
            else MLP(emb_dim, out_dim, hidden_dim)
        )

    def forward(self, x):
        x = x[:, self.output_token_idx]
        x = self.logits(x)
        return x


class WindowClassificationHead(nn.Module):
    """Window classifier."""

    def __init__(self, emb_dim, out_dim, hidden_dim):
        super().__init__()
        self.logits = (
            nn.Linear(emb_dim, out_dim)
            if hidden_dim is None
            else MLP(emb_dim, out_dim, hidden_dim)
        )

    def forward(self, x):
        x, _ = torch.max(x, dim=1)
        x = self.logits(x)
        return x
