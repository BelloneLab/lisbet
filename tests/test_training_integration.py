import pytest

from lisbet.config.schemas import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    TransformerBackboneConfig,
)
from lisbet.hub import fetch_dataset
from lisbet.training import train


@pytest.mark.integration
def test_train_integration(tmp_path):
    # Download a small sample dataset using the LISBET API
    fetch_dataset("SampleData", download_path=tmp_path)
    data_path = tmp_path / "datasets" / "sample_keypoints"

    # Configure experiment
    backbone_config = TransformerBackboneConfig(
        embedding_dim=4,
        hidden_dim=8,
        num_heads=1,
        num_layers=1,
        max_length=4,
    )

    data_config = DataConfig(
        data_path=str(data_path),
        data_format="DLC",
        window_size=4,
        window_offset=0,
        dev_ratio=None,
        train_sample=None,
        dev_sample=None,
    )

    model_config = ModelConfig(
        model_id="pytest_run",
        backbone=backbone_config,
        out_heads={},
        input_features={},
        window_size=4,
        window_offset=0,
    )

    training_config = TrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        data_augmentation=None,
        save_weights="last",
        save_history=True,
        mixed_precision=False,
    )

    experiment_config = ExperimentConfig(
        run_id="pytest_run",
        model=model_config,
        training=training_config,
        data=data_config,
        task_ids_list=["cons"],
        task_data=None,
        seed=1991,
        output_path=tmp_path,
    )

    # Run a minimal training (1 epoch, small batch, minimal model)
    model = train(experiment_config)
    # Check that model is returned and weights/history are saved
    assert hasattr(model, "state_dict")
    weights_path = tmp_path / "models" / "pytest_run" / "weights" / "weights_last.pt"
    assert weights_path.exists()
    hist_path = (
        tmp_path
        / "models"
        / "pytest_run"
        / "training_history"
        / "version_0"
        / "metrics.csv"
    )
    assert hist_path.exists()
