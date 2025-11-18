from lisbet.modeling.backbones.lstm import LSTMBackbone
from lisbet.modeling.backbones.transformer import TransformerBackbone
from lisbet.modeling.heads.classification import (
    FrameClassificationHead,
    WindowClassificationHead,
)
from lisbet.modeling.heads.embedding import EmbeddingHead
from lisbet.modeling.heads.projection import ProjectionHead
from lisbet.modeling.info import model_info
from lisbet.modeling.losses import InfoNCELoss
from lisbet.modeling.metrics import AlignmentMetric, UniformityMetric
from lisbet.modeling.models import MultiTaskModel

__all__ = [
    "FrameClassificationHead",
    "WindowClassificationHead",
    "EmbeddingHead",
    "ProjectionHead",
    "InfoNCELoss",
    "AlignmentMetric",
    "UniformityMetric",
    "model_info",
    "MultiTaskModel",
    "LSTMBackbone",
    "TransformerBackbone",
]

__doc__ = """
PyTorch models and their extensions.
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
