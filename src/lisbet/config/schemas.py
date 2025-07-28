from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class TransformerBackboneConfig(BaseModel):
    type: Literal["transformer"] = "transformer"
    feature_dim: Optional[int] = None
    embedding_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    max_length: int


class LSTMBackboneConfig(BaseModel):
    type: Literal["lstm"] = "lstm"
    feature_dim: Optional[int] = None
    embedding_dim: int
    hidden_dim: int
    num_layers: int


BackboneConfig = Annotated[
    Union[TransformerBackboneConfig, LSTMBackboneConfig], Field(discriminator="type")
]


class DataConfig(BaseModel):
    data_path: str
    data_format: str = "DLC"
    data_scale: Optional[str] = None
    data_filter: Optional[str] = None
    select_coords: Optional[str] = None
    rename_coords: Optional[str] = None
    window_size: int = 200
    window_offset: int = 0
    fps_scaling: float = 1.0
    dev_ratio: Optional[float] = None
    train_sample: Optional[float] = None
    dev_sample: Optional[float] = None


class ModelConfig(BaseModel):
    model_id: str
    backbone: BackboneConfig
    out_heads: dict[str, dict]
    input_features: dict[str, list[str]]
    window_size: int
    window_offset: int


class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    data_augmentation: bool = False
    save_weights: Optional[str] = None
    save_history: bool = False
    mixed_precision: bool = False
    freeze_backbone_weights: bool = False
    load_backbone_weights: Optional[Union[str, Path]] = None


class ExperimentConfig(BaseModel):
    run_id: str
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    task_ids_list: list[str]
    task_data: Optional[str] = None
    seed: int = 1991
    output_path: Path = Path(".")
