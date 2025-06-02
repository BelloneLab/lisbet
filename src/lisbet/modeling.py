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

import logging
import math
import pprint
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.table import Table
from torch import nn
from torchinfo import summary


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


class ClassificationHead(nn.Module):
    """Classifier."""

    def __init__(self, output_token_idx, emb_dim, out_dim, hidden_dim):
        super().__init__()
        # Select all tokens or a single one (in a list to keep dimensions consistent)
        self.output_token_idx = (
            slice(None) if output_token_idx is None else [output_token_idx]
        )
        self.logits = (
            nn.Linear(emb_dim, out_dim)
            if hidden_dim is None
            else MLP(emb_dim, out_dim, hidden_dim)
        )

    def forward(self, x):
        x = x[:, self.output_token_idx]
        x, _ = torch.max(x, dim=1)
        x = self.logits(x)
        x = torch.squeeze(x)
        return x


class EmbeddingModel(nn.Module):
    """Base model for embedding."""

    def __init__(self, output_token_idx, **backbone_kwargs):
        super().__init__()
        self.backbone = Backbone(**backbone_kwargs)
        # Select all tokens or a single one (in a list to keep dimensions consistent)
        self.output_token_idx = (
            slice(None) if output_token_idx is None else [output_token_idx]
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x[:, self.output_token_idx]
        return x


class MultiTaskModel(nn.Module):
    """Base model for all tasks."""

    def __init__(self, backbone, task_heads):
        super().__init__()
        self.backbone = backbone
        self.task_heads = nn.ModuleDict(task_heads)

    def forward(self, x, task_id):
        x = self.backbone(x)
        x = self.task_heads[task_id](x)
        return x


def load_model(config_path, weights_path):
    """Load a pretrained model.

    This function extends the behavior of the default model.load_model by wrapping the
    references to custom layers. Furthermore, it supports weights in the HDF5 format,
    which we prefer for sharing.

    Parameters
    ----------
    config_path : str or path-like
        Path to the model configuration file (JSON).
    weights_path : str or path-like
        Path to the model weights (HDF5).

    Returns
    -------
    torch.nn.Module : The loaded model.

    """
    with open(config_path, encoding="utf-8") as f_yaml:
        model_config = yaml.safe_load(f_yaml)

    if "out_dim" in model_config:
        # TODO: I should probably revise the way the MultiTaskModel is initialized.
        #       The old EmbeddingModel style (**model_config) is probably a better
        #       choice if we remove the extra keys.
        backbone = Backbone(
            bp_dim=model_config["bp_dim"],
            emb_dim=model_config["emb_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_heads=model_config["num_heads"],
            num_layers=model_config["num_layers"],
            max_len=model_config["max_len"],
        )

        model = MultiTaskModel(
            backbone,
            {
                task_id: ClassificationHead(
                    output_token_idx=model_config["output_token_idx"],
                    emb_dim=model_config["emb_dim"],
                    out_dim=out_dim,
                    hidden_dim=model_config["hidden_dim"],
                )
                for task_id, out_dim in model_config["out_dim"].items()
            },
        )
    else:
        model = EmbeddingModel(
            output_token_idx=model_config["output_token_idx"],
            bp_dim=model_config["bp_dim"],
            emb_dim=model_config["emb_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_heads=model_config["num_heads"],
            num_layers=model_config["num_layers"],
            max_len=model_config["max_len"],
        )

    # Load weights
    # NOTE: Setting strict=False allows for partial loading (i.e., dropping
    #       self-supervised training heads)
    incompatible_layers = model.load_state_dict(
        torch.load(weights_path, weights_only=True, map_location=torch.device("cpu")),
        strict=False,
    )
    logging.info(
        "Loaded weights from file.\nMissing keys: %s\nUnexpected keys: %s",
        incompatible_layers.missing_keys,
        incompatible_layers.unexpected_keys,
    )

    return model


def export_embedder(model_path, weights_path, output_path=Path(".")):
    # Get hyper-parameters
    with open(model_path, encoding="utf-8") as f_yaml:
        yaml_config = yaml.safe_load(f_yaml)
    model_id = yaml_config["model_id"] + "-embedder"

    # Create behavior embedding model
    # TODO: This should be improved.
    model_config = {
        "model_id": model_id,
        "bp_dim": yaml_config["bp_dim"],
        "emb_dim": yaml_config["emb_dim"],
        "hidden_dim": yaml_config["hidden_dim"],
        "num_heads": yaml_config["num_heads"],
        "num_layers": yaml_config["num_layers"],
        "max_len": yaml_config["max_len"],
        "input_features": yaml_config["input_features"],
    }
    embedding_model = EmbeddingModel(
        output_token_idx=-1,
        bp_dim=model_config["bp_dim"],
        emb_dim=model_config["emb_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        max_len=model_config["max_len"],
    )
    summary(embedding_model)

    # Load weights from pretrained model
    embedding_model.load_state_dict(
        torch.load(weights_path, weights_only=True, map_location=torch.device("cpu")),
        strict=False,
    )

    # Create output directory
    output_path = output_path / "models" / model_id

    # Store configuration
    model_config["output_token_idx"] = -1
    model_path = output_path / "model_config.yml"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f_yaml:
        yaml.safe_dump(model_config, f_yaml)

    # Store weights
    weights_path = output_path / "weights" / weights_path.name
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embedding_model.state_dict(), weights_path)


def fetch_model(model_id, download_path=Path(".")):
    """Fetch a model from the HF Hub."""
    valid_model_ids = [
        "lisbet32x4-calms21UftT1-classifier",
        "lisbet32x4-calms21U-embedder",
    ]
    assert model_id in valid_model_ids, (
        f"Model ID '{model_id}' not found in the list of available models."
    )

    model_path = download_path / model_id
    snapshot_download(
        repo_id=f"gchindemi/{model_id}", repo_type="model", local_dir=model_path
    )


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
