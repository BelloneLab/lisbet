"""PyTorch models and their extensions.
The transformer model is based on ViT [1] and its reference implementation in JAX/Flax,
available at https://github.com/google-research/vision_transformer.

Notes
-----
[a] Early versions of LISBET were using TensorFlow/Keras.

References
----------
[1] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X.,
    Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J.,
    & Houlsby, N. (2020). An Image is Worth 16x16 Words: Transformers for Image
    Recognition at Scale. ArXiv:2010.11929 [Cs]. http://arxiv.org/abs/2010.11929

"""

import math
import pprint

import torch
import yaml
from rich.console import Console
from rich.table import Table
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


class Backbone(nn.Module):
    """Backbone."""

    def __init__(self, bp_dim, emb_dim, hidden_dim, num_heads, num_layers, max_len):
        super().__init__()
        self.frame_embedder = nn.Linear(bp_dim, emb_dim)
        self.pos_embedder = PosEmbedding(max_len, emb_dim)
        encoder_layers = nn.TransformerEncoderLayer(
            emb_dim,
            num_heads,
            hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers, enable_nested_tensor=False
        )
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.frame_embedder(x)
        x = self.pos_embedder(x)
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        return x


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


class EmbeddingHead(nn.Module):
    """Embedding head for extracting behavior embeddings."""

    def __init__(self, output_token_idx):
        super().__init__()
        self.output_token_idx = output_token_idx

    def forward(self, x):
        x = x[:, self.output_token_idx]
        return x


class LISBETModel(nn.Module):
    """Base model for all tasks."""

    def __init__(self, backbone, task_heads):
        super().__init__()
        self.backbone = backbone
        self.task_heads = nn.ModuleDict(task_heads)

    def forward(self, x, task_id):
        x = self.backbone(x)
        x = self.task_heads[task_id](x)
        return x


def model_info(model_path):
    """Print information about a LISBET model config file."""

    with open(model_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    console = Console()
    table = Table(title="LISBET Model Configuration")

    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in config.items():
        # Pretty-print nested dicts/lists
        if isinstance(value, (dict, list)):
            value_str = pprint.pformat(value, compact=True, width=60)
        else:
            value_str = str(value)
        table.add_row(str(key), value_str)

    console.print(table)
