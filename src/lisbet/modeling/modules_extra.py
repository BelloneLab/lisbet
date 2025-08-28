"""Extra modules for various tasks, such as positional embedding and MLP."""

import math

import torch
from torch import nn


class PosEmbedding(nn.Module):
    """Positional embedding."""

    def __init__(self, max_len, emb_dim, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.pos_emb = nn.Parameter(torch.empty((max_len, emb_dim), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.pos_emb, a=math.sqrt(5))

    def forward(self, x):
        return x + self.pos_emb[: x.size(-2)]

    def extra_repr(self):
        return f"max_len={self.max_len}, emb_dim={self.emb_dim}"


class MLP(nn.Module):
    """Multilayer perceptron."""

    def __init__(self, in_features, out_features, hid_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hid_features)
        self.linear2 = nn.Linear(hid_features, out_features)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
