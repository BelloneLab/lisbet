"""Integration tests for data augmentation in training pipeline."""

import pytest

from lisbet.config.schemas import (
    DataAugmentationConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    TransformerBackboneConfig,
)
from lisbet.hub import fetch_dataset
from lisbet.training import train


@pytest.mark.integration
def test_train_with_augmentation_all_perm_id(tmp_path):
    """Test training with all_perm_id augmentation."""
    # Download a small sample dataset
    fetch_dataset("SampleData", download_path=tmp_path)
    data_path = tmp_path / "datasets" / "sample_keypoints"

    # Configure experiment with all_perm_id augmentation
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
    )

    model_config = ModelConfig(
        model_id="test_aug_perm_id",
        backbone=backbone_config,
        out_heads={},
        input_features={},
        window_size=4,
        window_offset=0,
    )

    # Use new augmentation configuration
    aug_configs = [DataAugmentationConfig(name="all_perm_id", p=1.0)]

    training_config = TrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        data_augmentation=aug_configs,
        save_weights="last",
        mixed_precision=False,
    )

    experiment_config = ExperimentConfig(
        run_id="test_aug_perm_id",
        model=model_config,
        training=training_config,
        data=data_config,
        task_ids_list=["cons"],
        task_data=None,
        seed=1991,
        output_path=tmp_path,
    )

    # Train model
    model = train(experiment_config)

    # Check that model is returned
    assert hasattr(model, "state_dict")


@pytest.mark.integration
def test_train_with_augmentation_probability(tmp_path):
    """Test training with augmentation probability < 1.0."""
    # Download a small sample dataset
    fetch_dataset("SampleData", download_path=tmp_path)
    data_path = tmp_path / "datasets" / "sample_keypoints"

    # Configure experiment with probability-based augmentation
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
    )

    model_config = ModelConfig(
        model_id="test_aug_prob",
        backbone=backbone_config,
        out_heads={},
        input_features={},
        window_size=4,
        window_offset=0,
    )

    # Use augmentation with p=0.5
    aug_configs = [DataAugmentationConfig(name="all_perm_id", p=0.5)]

    training_config = TrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        data_augmentation=aug_configs,
        save_weights="last",
        mixed_precision=False,
    )

    experiment_config = ExperimentConfig(
        run_id="test_aug_prob",
        model=model_config,
        training=training_config,
        data=data_config,
        task_ids_list=["cons"],
        task_data=None,
        seed=1991,
        output_path=tmp_path,
    )

    # Train model
    model = train(experiment_config)

    # Check that model is returned
    assert hasattr(model, "state_dict")


@pytest.mark.integration
def test_train_with_block_permutation(tmp_path):
    """Test training with blk_perm_id augmentation."""
    # Download a small sample dataset
    fetch_dataset("SampleData", download_path=tmp_path)
    data_path = tmp_path / "datasets" / "sample_keypoints"

    # Configure experiment with block permutation
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
    )

    model_config = ModelConfig(
        model_id="test_aug_blk",
        backbone=backbone_config,
        out_heads={},
        input_features={},
        window_size=4,
        window_offset=0,
    )

    # Use block permutation with fraction
    aug_configs = [DataAugmentationConfig(name="blk_perm_id", p=1.0, frac=0.3)]

    training_config = TrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        data_augmentation=aug_configs,
        save_weights="last",
        mixed_precision=False,
    )

    experiment_config = ExperimentConfig(
        run_id="test_aug_blk",
        model=model_config,
        training=training_config,
        data=data_config,
        task_ids_list=["cons"],
        task_data=None,
        seed=1991,
        output_path=tmp_path,
    )

    # Train model
    model = train(experiment_config)

    # Check that model is returned
    assert hasattr(model, "state_dict")


@pytest.mark.integration
def test_train_with_multiple_augmentations(tmp_path):
    """Test training with multiple augmentations."""
    # Download a small sample dataset
    fetch_dataset("SampleData", download_path=tmp_path)
    data_path = tmp_path / "datasets" / "sample_keypoints"

    # Configure experiment with multiple augmentations
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
    )

    model_config = ModelConfig(
        model_id="test_aug_multi",
        backbone=backbone_config,
        out_heads={},
        input_features={},
        window_size=4,
        window_offset=0,
    )

    # Use multiple augmentations with different parameters
    aug_configs = [
        DataAugmentationConfig(name="all_perm_id", p=0.5),
        DataAugmentationConfig(name="all_perm_ax", p=0.7),
        DataAugmentationConfig(name="blk_perm_id", p=0.3, frac=0.2),
        DataAugmentationConfig(name="gauss_jitter", p=0.4, sigma=0.01),
        DataAugmentationConfig(name="gauss_window_jitter", p=0.1, sigma=0.02, window=15),
    ]

    training_config = TrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        data_augmentation=aug_configs,
        save_weights="last",
        mixed_precision=False,
    )

    experiment_config = ExperimentConfig(
        run_id="test_aug_multi",
        model=model_config,
        training=training_config,
        data=data_config,
        task_ids_list=["cons"],
        task_data=None,
        seed=1991,
        output_path=tmp_path,
    )

    # Train model
    model = train(experiment_config)

    # Check that model is returned
    assert hasattr(model, "state_dict")


@pytest.mark.integration
def test_train_multiclass_with_augmentation(tmp_path):
    """Test training multiclass task with augmentation."""
    # Download a small sample dataset
    fetch_dataset("SampleData", download_path=tmp_path)
    data_path = tmp_path / "datasets" / "sample_keypoints"

    # Configure experiment with augmentation for supervised task
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
        dev_ratio=0.2,
    )

    model_config = ModelConfig(
        model_id="test_aug_multiclass",
        backbone=backbone_config,
        out_heads={},
        input_features={},
        window_size=4,
        window_offset=0,
    )

    # Use augmentation for multiclass task
    aug_configs = [
        DataAugmentationConfig(name="all_perm_id", p=0.5),
        DataAugmentationConfig(name="all_perm_ax", p=0.5),
    ]

    training_config = TrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        data_augmentation=aug_configs,
        save_weights="last",
        mixed_precision=False,
    )

    experiment_config = ExperimentConfig(
        run_id="test_aug_multiclass",
        model=model_config,
        training=training_config,
        data=data_config,
        task_ids_list=["multiclass"],
        task_data=None,
        seed=1991,
        output_path=tmp_path,
    )

    # Train model
    model = train(experiment_config)

    # Check that model is returned
    assert hasattr(model, "state_dict")
