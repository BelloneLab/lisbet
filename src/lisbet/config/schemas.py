from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
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


@dataclass
class BackboneConfig:
    model_type: str
    embedding_dim: int
    num_layers: int
    num_heads: int
    hidden_dim: int


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    data_augmentation: bool = False
    save_weights: Optional[str] = None
    save_history: bool = False
    mixed_precision: bool = False
    freeze_backbone_weights: bool = False
    load_backbone_weights: Optional[str] = None


@dataclass
class ExperimentConfig:
    run_id: str
    backbone: BackboneConfig
    training: TrainingConfig
    data: DataConfig
    task_ids_list: list[str]
    task_data: Optional[str] = None
    seed: int = 1991
    output_path: Path = Path(".")
