from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

import lisbet.evaluation as evaluation
from lisbet.io import Record


@pytest.fixture
def mock_records():
    """Create mock records with annotations for testing."""
    # Create mock pose data
    positions = np.random.rand(20, 2, 3, 2) * 0.5 + 0.25  # 20 frames, 2 individuals
    posetracks = xr.Dataset(
        {"position": (["time", "space", "keypoints", "individuals"], positions)},
        coords={
            "time": np.arange(20),
            "space": ["x", "y"],
            "keypoints": ["nose", "ear", "tail"],
            "individuals": ["mouse1", "mouse2"],
        },
    )

    # Create mock annotations - simple pattern for testing
    target_cls = np.zeros((20, 2, 1), dtype=int)  # 2 behaviors
    for i in range(20):
        behavior_idx = i % 2  # Alternate between behaviors
        target_cls[i, behavior_idx, 0] = 1

    annotations = xr.Dataset(
        {"target_cls": (["time", "behaviors", "annotators"], target_cls)},
        coords={
            "time": np.arange(20),
            "behaviors": ["behavior_0", "behavior_1"],
            "annotators": ["annotator0"],
        },
    )

    record = Record(id="test_record", posetracks=posetracks, annotations=annotations)
    return [record]


@pytest.fixture
def mock_model():
    """Create a mock model that returns predictable outputs."""
    model = Mock()
    model.eval = Mock()

    def mock_forward(x, mode):
        batch_size = x.shape[0]
        if mode == "multiclass":
            # Return logits for 2 classes - alternate predictions
            logits = torch.zeros(batch_size, 2)
            for i in range(batch_size):
                logits[i, i % 2] = 1.0  # Predict class based on batch index
        else:
            # For multilabel, return sigmoid-like outputs
            logits = torch.rand(batch_size, 2)
        return logits

    model.side_effect = mock_forward
    model.__call__ = mock_forward
    return model


@pytest.fixture
def mock_config():
    """Create a mock model config."""
    return {"model_id": "test_model", "window_size": 10, "some_other_config": "value"}


def test_evaluate_basic(tmp_path, mock_records, mock_model, mock_config):
    """Test basic evaluation functionality."""
    # Create dummy model and weights files
    model_path = tmp_path / "model_config.yml"
    with open(model_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(mock_config, f)

    weights_path = tmp_path / "weights.pt"
    torch.save({"dummy": "weights"}, weights_path)

    # Mock the model loading and other dependencies
    with (
        patch("lisbet.evaluation.load_model_and_config") as mock_load,
        patch("lisbet.evaluation.load_records") as mock_load_records,
        patch("lisbet.evaluation.check_feature_compatibility"),
        patch("lisbet.evaluation.select_device") as mock_device,
    ):
        mock_load.return_value = (mock_model, mock_config)
        mock_load_records.return_value = mock_records
        mock_device.return_value = torch.device("cpu")

        # Run evaluation
        report = evaluation.evaluate(
            model_path=str(model_path),
            weights_path=str(weights_path),
            data_format="movement",
            data_path="dummy",
            window_size=10,
            mode="multiclass",
        )

        # Check return structure
        assert isinstance(report, dict)
        assert "mode" in report
        assert "f1_macro" in report
        assert "accuracy_macro" in report
        assert "per_class" in report
        assert report["mode"] == "multiclass"

        # Check per_class structure
        assert "f1" in report["per_class"]
        assert "precision" in report["per_class"]
        assert "recall" in report["per_class"]
        assert len(report["per_class"]["f1"]) == 2  # 2 classes


def test_evaluate_with_output_path(tmp_path, mock_records, mock_model, mock_config):
    """Test evaluation with output path (file saving)."""
    # Create dummy model and weights files
    model_path = tmp_path / "model_config.yml"
    with open(model_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(mock_config, f)

    weights_path = tmp_path / "weights.pt"
    torch.save({"dummy": "weights"}, weights_path)

    # Mock dependencies
    with (
        patch("lisbet.evaluation.load_model_and_config") as mock_load,
        patch("lisbet.evaluation.load_records") as mock_load_records,
        patch("lisbet.evaluation.check_feature_compatibility"),
        patch("lisbet.evaluation.select_device") as mock_device,
    ):
        mock_load.return_value = (mock_model, mock_config)
        mock_load_records.return_value = mock_records
        mock_device.return_value = torch.device("cpu")

        # Run evaluation with output path
        report = evaluation.evaluate(
            model_path=str(model_path),
            weights_path=str(weights_path),
            data_format="movement",
            data_path="dummy",
            window_size=10,
            mode="multiclass",
            output_path=str(tmp_path),
        )

        # Check that the report file was saved
        expected_path = (
            tmp_path / "evaluations" / "test_model" / "evaluation_report.yml"
        )
        assert expected_path.exists()

        # Check that the saved content matches the returned report
        with open(expected_path, encoding="utf-8") as f:
            saved_report = yaml.safe_load(f)

        assert saved_report["mode"] == report["mode"]
        assert saved_report["f1_macro"] == report["f1_macro"]
        assert saved_report["accuracy_macro"] == report["accuracy_macro"]


def test_evaluate_multilabel_mode(tmp_path, mock_records, mock_config):
    """Test evaluation in multilabel mode."""
    # Create a mock model for multilabel
    mock_model = Mock()
    mock_model.eval = Mock()

    def mock_forward_multilabel(x, mode):
        batch_size = x.shape[0]
        # Return sigmoid-like outputs for multilabel
        return torch.sigmoid(torch.randn(batch_size, 2))

    mock_model.side_effect = mock_forward_multilabel
    mock_model.__call__ = mock_forward_multilabel

    # Create dummy files
    model_path = tmp_path / "model_config.yml"
    with open(model_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(mock_config, f)

    weights_path = tmp_path / "weights.pt"
    torch.save({"dummy": "weights"}, weights_path)

    # Mock dependencies
    with (
        patch("lisbet.evaluation.load_model_and_config") as mock_load,
        patch("lisbet.evaluation.load_records") as mock_load_records,
        patch("lisbet.evaluation.check_feature_compatibility"),
        patch("lisbet.evaluation.select_device") as mock_device,
    ):
        mock_load.return_value = (mock_model, mock_config)
        mock_load_records.return_value = mock_records
        mock_device.return_value = torch.device("cpu")

        # Run evaluation in multilabel mode
        report = evaluation.evaluate(
            model_path=str(model_path),
            weights_path=str(weights_path),
            data_format="movement",
            data_path="dummy",
            window_size=10,
            mode="multilabel",
            threshold=0.5,
        )

        # Check mode is set correctly
        assert report["mode"] == "multilabel"
        assert "f1_macro" in report
        assert "accuracy_macro" in report


def test_evaluate_invalid_mode(tmp_path, mock_records, mock_model, mock_config):
    """Test that invalid mode raises an error."""
    # Create dummy files
    model_path = tmp_path / "model_config.yml"
    with open(model_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(mock_config, f)

    weights_path = tmp_path / "weights.pt"
    torch.save({"dummy": "weights"}, weights_path)

    # Mock dependencies
    with (
        patch("lisbet.evaluation.load_model_and_config") as mock_load,
        patch("lisbet.evaluation.load_records") as mock_load_records,
        patch("lisbet.evaluation.check_feature_compatibility"),
        patch("lisbet.evaluation.select_device") as mock_device,
    ):
        mock_load.return_value = (mock_model, mock_config)
        mock_load_records.return_value = mock_records
        mock_device.return_value = torch.device("cpu")

        # Test invalid mode
        with pytest.raises(ValueError, match="Invalid label format"):
            evaluation.evaluate(
                model_path=str(model_path),
                weights_path=str(weights_path),
                data_format="movement",
                data_path="dummy",
                window_size=10,
                mode="invalid_mode",
            )


def test_evaluate_with_ignore_index(tmp_path, mock_records, mock_model, mock_config):
    """Test evaluation with ignore_index parameter."""
    # Create dummy files
    model_path = tmp_path / "model_config.yml"
    with open(model_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(mock_config, f)

    weights_path = tmp_path / "weights.pt"
    torch.save({"dummy": "weights"}, weights_path)

    # Mock dependencies
    with (
        patch("lisbet.evaluation.load_model_and_config") as mock_load,
        patch("lisbet.evaluation.load_records") as mock_load_records,
        patch("lisbet.evaluation.check_feature_compatibility"),
        patch("lisbet.evaluation.select_device") as mock_device,
    ):
        mock_load.return_value = (mock_model, mock_config)
        mock_load_records.return_value = mock_records
        mock_device.return_value = torch.device("cpu")

        # Run evaluation with ignore_index
        report = evaluation.evaluate(
            model_path=str(model_path),
            weights_path=str(weights_path),
            data_format="movement",
            data_path="dummy",
            window_size=10,
            mode="multiclass",
            ignore_index=1,  # Ignore class 1
        )

        # Should still return a valid report
        assert isinstance(report, dict)
        assert "f1_macro" in report
        assert "accuracy_macro" in report


def test_evaluate_metrics_consistency(tmp_path, mock_records, mock_config):
    """Test that evaluation metrics are computed consistently."""
    # Create a deterministic mock model
    mock_model = Mock()
    mock_model.eval = Mock()

    def deterministic_forward(x, mode):
        batch_size = x.shape[0]
        # Always predict class 0 for consistent testing
        logits = torch.zeros(batch_size, 2)
        logits[:, 0] = 1.0  # Always predict class 0
        return logits

    mock_model.side_effect = deterministic_forward
    mock_model.__call__ = deterministic_forward

    # Create dummy files
    model_path = tmp_path / "model_config.yml"
    with open(model_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(mock_config, f)

    weights_path = tmp_path / "weights.pt"
    torch.save({"dummy": "weights"}, weights_path)

    # Mock dependencies
    with (
        patch("lisbet.evaluation.load_model_and_config") as mock_load,
        patch("lisbet.evaluation.load_records") as mock_load_records,
        patch("lisbet.evaluation.check_feature_compatibility"),
        patch("lisbet.evaluation.select_device") as mock_device,
    ):
        mock_load.return_value = (mock_model, mock_config)
        mock_load_records.return_value = mock_records
        mock_device.return_value = torch.device("cpu")

        # Run evaluation
        report = evaluation.evaluate(
            model_path=str(model_path),
            weights_path=str(weights_path),
            data_format="movement",
            data_path="dummy",
            window_size=10,
            mode="multiclass",
        )

        # Since model always predicts class 0 and ground truth alternates,
        # we expect specific performance characteristics
        assert 0.0 <= report["f1_macro"] <= 1.0
        assert 0.0 <= report["accuracy_macro"] <= 1.0
        assert len(report["per_class"]["f1"]) == 2
        assert len(report["per_class"]["precision"]) == 2
        assert len(report["per_class"]["recall"]) == 2
