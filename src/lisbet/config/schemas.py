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

    Permutation-based (legacy):
        - all_perm_id: Full-window permutation of individual identities.
        - all_perm_ax: Full-window permutation of spatial axes.
        - blk_perm_id: Block (contiguous frames) permutation of individual identities.
          Uses ``frac`` for relative block length.
        For these, ``p`` is the probability of applying the *entire* transform
        (implemented via ``RandomApply`` in the pipeline).

        Jitter-based (new):
                - gauss_jitter: Per-element Bernoulli(p) mask over (time, keypoints, individuals),
                    adds N(0, sigma) noise to selected elements (broadcast over space dims).
                - gauss_window_jitter: Bernoulli(p) over (time, keypoints, individuals) selects
                    start elements. Each start activates a temporal window of length ``window``
                    for that (keypoint, individual) only (all space dims). Overlapping windows
                    merge; noise applied once per affected element-frame.
        For jitter transforms, ``p`` is *internal* (not wrapped by RandomApply) and
        drives the element/window sampling process.

    Attributes:
        name: Augmentation identifier.
        p: Probability parameter (semantics depend on family, see above).
        frac: Fraction of frames for block permutation (blk_perm_id only).
        sigma: Standard deviation of Gaussian noise (jitter types only).
        window: Window length (frames) for gauss_window_jitter only.
    """

    name: Literal[
        "all_perm_id",
        "all_perm_ax",
        "blk_perm_id",
        "gauss_jitter",
        "gauss_window_jitter",
    ]
    p: float = 1.0
    frac: float | None = None
    sigma: float | None = None
    window: int | None = None

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
    def validate_sigma(cls, v, info):
        if info.data.get("name") in ("gauss_jitter", "gauss_window_jitter"):
            # Required (will default later if None), must be positive
            if v is not None and v <= 0.0:
                raise ValueError("sigma must be > 0.0 for jitter augmentations")
        else:
            # Disallow sigma for non-jitter types to avoid silent misuse
            if v is not None:
                raise ValueError("sigma parameter only valid for jitter augmentations")
        return v

    @field_validator("window")
    @classmethod
    def validate_window(cls, v, info):
        name = info.data.get("name")
        if name == "gauss_window_jitter":
            if v is not None and v <= 0:
                raise ValueError("window must be a positive integer for gauss_window_jitter")
        else:
            if v is not None:
                raise ValueError("window parameter only valid for gauss_window_jitter")
        return v

    def model_post_init(self, __context):
        """Post-init normalization & cross-field validation."""
        # blk_perm_id: assign default frac if missing
        if self.name == "blk_perm_id":
            if self.frac is None:
                self.frac = 0.5
        else:
            if self.frac is not None:
                raise ValueError(
                    f"frac parameter is only valid for blk_perm_id, not {self.name}"
                )

        # Jitter defaults & requirements
        if self.name in ("gauss_jitter", "gauss_window_jitter"):
            if self.sigma is None:
                self.sigma = 0.01  # default sigma
            if self.name == "gauss_window_jitter":
                if self.window is None:
                    self.window = 10  # default window length
        else:
            # Non-jitter types must not have sigma/window
            if self.sigma is not None:
                raise ValueError("sigma parameter only valid for jitter augmentations")
            if self.window is not None:
                raise ValueError("window parameter only valid for gauss_window_jitter")


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
