"""Transformer Backbone for Lisbet."""

from torch import nn

from lisbet.modeling.modules_extra import PosEmbedding


class TransformerBackbone(nn.Module):
    """Transformer Backbone."""

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
