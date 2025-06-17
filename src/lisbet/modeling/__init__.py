from lisbet.modeling.backbones.transformer import TransformerBackbone
from lisbet.modeling.factory import (
    create_backbone_from_config,
    create_embedding_head,
    create_frame_classification_head,
    create_model_from_config,
    create_task_head_from_config,
    create_transformer_model,
    create_window_classification_head,
    get_model_info,
)
from lisbet.modeling.heads.classification import (
    FrameClassificationHead,
    WindowClassificationHead,
)
from lisbet.modeling.heads.embedding import EmbeddingHead
from lisbet.modeling.info import model_info
from lisbet.modeling.models import MultiTaskModel

__all__ = [
    "FrameClassificationHead",
    "WindowClassificationHead",
    "EmbeddingHead",
    "model_info",
    "MultiTaskModel",
    "TransformerBackbone",
    # Factory functions
    "create_backbone_from_config",
    "create_embedding_head",
    "create_frame_classification_head",
    "create_model_from_config",
    "create_task_head_from_config",
    "create_transformer_model",
    "create_window_classification_head",
    "get_model_info",
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
