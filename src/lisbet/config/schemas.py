from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


class TransformerBackboneConfig(BaseModel):
    type: Literal["transformer"] = "transformer"
    feature_dim: int | None = None
    embedding_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    max_length: int


class TCNBackboneConfig(BaseModel):
    type: Literal["tcn"] = "tcn"
    feature_dim: int | None = None
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    kernel_size: int = 3
    dilation_base: int = 2
    dropout: float = 0.0


class LSTMBackboneConfig(BaseModel):
    type: Literal["lstm"] = "lstm"
    feature_dim: int | None = None
    embedding_dim: int
    hidden_dim: int
    num_layers: int


BackboneConfig = Annotated[
    TransformerBackboneConfig | LSTMBackboneConfig | TCNBackboneConfig,
    Field(discriminator="type"),
]


class DataConfig(BaseModel):
    data_path: str
    data_format: str = "DLC"
    data_scale: str | None = None
    data_filter: str | None = None
    select_coords: str | None = None
    rename_coords: str | None = None
    window_size: int = 200
    window_offset: int = 0
    fps_scaling: float = 1.0
    dev_ratio: float | None = None
    train_sample: float | None = None
    dev_sample: float | None = None


class DataAugmentationConfig(BaseModel):
    """Configuration for a single data augmentation technique.

    Augmentation families and parameter semantics:

        Permutation-based :
            - all_perm_id: Full-window permutation of individual identities.
            - all_perm_ax: Full-window permutation of spatial axes.
            - blk_perm_id: Block (contiguous frames) permutation of individual
                identities. Uses ``frac`` for relative block length.

        For these, ``p`` is the probability of applying the *entire* transform
        (implemented via ``RandomApply`` in the pipeline).

        Jitter-based :
            - gauss_jitter: For the full window  (time, keypoints,
                individuals), adds N(0, sigma) noise.

        Ablation-based :
            - kp_ablation: Per-element Bernoulli(pB) mask over (time, keypoints,
                individuals), sets selected elements to NaN (all space dims).
                Simulates missing or occluded keypoints.



    Attributes:
        name: Name of the augmentation technique (all_perm_id, all_perm_ax, blk_perm_id)
        p: Probability of applying this transformation (0.0 to 1.0)
        pB: When applicable, per-element Bernoulli probability (kp_ablation types only)
        frac: Fraction of frames to permute (only for blk_perm_id, 0.0 to 1.0 exclusive)
        sigma: Standard deviation of Gaussian noise (jitter types only).
        frac: Fraction-based block length for blk_gauss_jitter (also required).
    """
    model_config = {"extra": "forbid"}  # Reject unknown parameters!
    name: Literal[
        "all_perm_id",
        "all_perm_ax",
        "blk_perm_id",
        "gauss_jitter",
        "kp_ablation",
    ]
    p: float = 1.0
    pB: float | None = None
    frac: float | None = None
    sigma: float | None = None

    # Define which parameters are valid for each augmentation type
    VALID_PARAMS = {
        "all_perm_id": {"p"},
        "all_perm_ax": {"p"},
        "blk_perm_id": {"p", "frac"},
        "gauss_jitter": {"p", "sigma"},
        "kp_ablation": {"p", "pB"},
    }

    @field_validator("p")
    @classmethod
    def validate_probability(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Probability p must be between 0.0 and 1.0")
        return v

    @field_validator("frac")
    @classmethod
    def validate_fraction(cls, v, info):
        if v is not None and not 0.0 < v < 1.0:
            raise ValueError("Fraction frac must be between 0.0 and 1.0 (exclusive)")
        return v

    @field_validator("sigma")
    @classmethod
    def validate_sigma(cls, v):
        if v is not None and v <= 0.0:
            raise ValueError("sigma must be > 0.0")
        return v
    
    @field_validator("pB")
    @classmethod
    def validate_pB(cls, v):
        if v is not None and not 0.0 < v <= 1.0:
            raise ValueError("pB must be > 0.0 and <= 1.0")
        return v



class DataAugmentationPipeline(BaseModel):
    augmentations: list[DataAugmentationConfig]

class ModelConfig(BaseModel):
    model_id: str | None = None
    backbone: BackboneConfig
    out_heads: dict[str, dict]
    input_features: dict[str, list[str]]
    window_size: int
    window_offset: int


class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    data_augmentation: list[DataAugmentationConfig] | None = None
    save_weights: str | None = None
    save_history: bool = False
    mixed_precision: bool = False
    head_type: Literal["mlp", "linear"] = "mlp"
    freeze_backbone_weights: bool = False
    load_backbone_weights: str | Path | None = None


class ExperimentConfig(BaseModel):
    run_id: str | None = None
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    task_ids_list: list[str]
    task_data: str | None = None
    seed: int = 1991
    output_path: Path = Path(".")
