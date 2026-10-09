from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    annot_format: str = "movement"
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

        For every family, ``p`` is the probability of applying the entire transform
        (implemented via ``RandomApply`` in the pipeline).

        Permutation-based :
            - all_perm_id: Full-window permutation of individual identities.
            - all_perm_ax: Full-window permutation of spatial axes.
            - blk_perm_id: Block (contiguous frames) permutation of individual
                identities. Uses ``frac`` for relative block length.

        Jitter-based :
            - gauss_jitter: Adds independent N(0, sigma) noise to every coordinate in
                the full window.

        Ablation-based :
            - kp_ablation: Samples a Bernoulli(pB) mask for each (keypoint,
                individual) pair and sets all of its spatial coordinates to 0.0 for
                the full window. Simulates sustained missing or occluded keypoints.

        Horizontal Transformations:
            - all_translate: Randomly translate all individuals together in x,y.
            - all_mirror_x: Randomly mirror horizontally around x=0.5.
            - all_zoom: Randomly zoom/dezoom around center (0.5, 0.5).

        Rotation-based :
            - rotation: Random rotation of all keypoint coordinates around the
                center of the normalized coordinate space. Supports 2D and 3D
                (auto-detected). Uses ``max_angle`` for the rotation range and
                ``mode`` for post-rotation normalization.


    Attributes:
        name: Name of the augmentation technique
        p: Probability of applying this transformation (0.0 to 1.0)
        pB: Per-(keypoint, individual) Bernoulli probability (kp_ablation only)
        frac: Nominal fraction of frames to permute (block augmentations only, (0, 1])
        sigma: Standard deviation of Gaussian noise (jitter types only)
        max_angle: Maximum rotation angle in degrees (rotation only, default 180.0)
        mode: Rotation normalization: "truncate", "rescale", or "none" (default).
    """

    model_config = {"extra": "forbid"}  # Reject unknown parameters!
    name: Literal[
        "all_perm_id",
        "all_perm_ax",
        "blk_perm_id",
        "gauss_jitter",
        "kp_ablation",
        "all_translate",
        "all_mirror_x",
        "all_zoom",
        "rotation",
    ]
    p: float = 1.0
    pB: float | None = None
    frac: float | None = None
    sigma: float | None = None
    max_angle: float | None = None
    mode: str | None = None

    # Define which parameters are valid for each augmentation type
    VALID_PARAMS: ClassVar[dict[str, set[str]]] = {
        "all_perm_id": {"p"},
        "all_perm_ax": {"p"},
        "blk_perm_id": {"p", "frac"},
        "gauss_jitter": {"p", "sigma"},
        "kp_ablation": {"p", "pB"},
        "all_translate": {"p"},
        "all_mirror_x": {"p"},
        "all_zoom": {"p"},
        "rotation": {"p", "max_angle", "mode"},
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
        if v is not None and not 0.0 < v <= 1.0:
            raise ValueError("Fraction frac must be in (0, 1]")
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

    @field_validator("max_angle")
    @classmethod
    def validate_max_angle(cls, v):
        if v is not None and not 0.0 <= v <= 360.0:
            raise ValueError("max_angle must be between 0.0 and 360.0")
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        if v is not None and v not in ("truncate", "rescale", "none"):
            raise ValueError("mode must be one of 'truncate', 'rescale', 'none'")
        return v

    @model_validator(mode="after")
    def validate_parameters_for_augmentation(self):
        """
        Ensure only valid parameters are set for each augmentation type and apply
        defaults.
        """
        valid_params = self.VALID_PARAMS[self.name]

        # Check which parameters are actually set (not None)
        set_params = set()
        if self.pB is not None:
            set_params.add("pB")
        if self.frac is not None:
            set_params.add("frac")
        if self.sigma is not None:
            set_params.add("sigma")
        if self.max_angle is not None:
            set_params.add("max_angle")
        if self.mode is not None:
            set_params.add("mode")

        # Find invalid parameters
        invalid_params = set_params - valid_params
        if invalid_params:
            raise ValueError(
                f"Invalid parameter(s) {invalid_params} for augmentation "
                f"'{self.name}'. Valid parameters are: {valid_params}"
            )

        # Check required parameters and set defaults
        block_types = ("blk_perm_id", "blk_translate", "blk_mirror_x", "blk_zoom")

        if self.name in block_types and self.frac is None:
            # Set defaults for block-based augmentations
            if self.name == "blk_perm_id":
                self.frac = 0.5
            else:  # blk_translate, blk_mirror_x, blk_zoom
                self.frac = 0.1

        if self.name == "kp_ablation" and self.pB is None:
            raise ValueError(
                f"Parameter 'pB' is required for augmentation '{self.name}'"
            )

        if self.name == "gauss_jitter" and self.sigma is None:
            # Set default sigma
            self.sigma = 0.01

        if self.name == "rotation":
            if self.max_angle is None:
                self.max_angle = 180.0
            if self.mode is None:
                self.mode = "truncate"

        return self


class DataAugmentationPipeline(BaseModel):
    augmentations: list[DataAugmentationConfig]


class ModelConfig(BaseModel):
    model_id: str | None = None
    backbone: BackboneConfig
    out_heads: dict[str, dict]
    input_features: dict[str, list[str]]
    window_size: int
    window_offset: int


class TaskConfig(BaseModel):
    """Per-task hyperparameter overrides (set via ``--set task.<id>.<param>=``)."""

    model_config = ConfigDict(extra="forbid")

    # InfoNCE temperature for the contrastive "geom" task. None means "use the
    # task default" (see lisbet.training.tasks.DEFAULT_GEOM_TEMPERATURE).
    temperature: float | None = Field(default=None, gt=0)


class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    data_augmentation: list[DataAugmentationConfig] | None = None
    task_configs: dict[str, TaskConfig] = Field(default_factory=dict)
    # Self-supervised window sampling: "any" = original (windows may extend past the record
    # edges, zero padded); "inside" = only windows entirely inside their record.
    window_sampling: Literal["any", "inside"] = "any"
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
