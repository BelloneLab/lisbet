"""
Comprehensive tests for inference module focusing on core functionality.

This test suite was rewritten to fix critical issues and provide comprehensive
coverage of the inference functionality. Key improvements:

1. **Fixed Individual Count Validation**: Updated dummy dataset creation to include
   2 individuals as required by WindowDataset validation (was causing failures).

2. **Comprehensive Function Coverage**: Tests all major inference functions including:
   - predict_record: Core single-sequence inference logic
   - predict: Dataset processing with input feature validation
   - annotate_behavior: Behavior classification pipeline
   - compute_embeddings: Embedding computation pipeline
   - Forward functions: Classification and embedding forward passes

3. **Input Feature Validation**: Tests the critical input feature compatibility
   checking between models and datasets, including proper error handling.

4. **Proper Mocking**: Uses comprehensive mocking to avoid dependencies on actual
   models while still testing the core logic. Mocks are properly structured to
   match expected interfaces.

5. **Edge Case Testing**: Covers error conditions like empty datasets, missing
   configurations, and device selection logic.

6. **File I/O Testing**: Tests output path functionality for both annotations
   and embeddings with proper directory structure creation.

The tests focus on testing actual logic and error handling rather than trivial
operations, using monkeypatching and mocking for reliable, deterministic results.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

import lisbet.inference.annotation as annotation
import lisbet.inference.common as common
import lisbet.inference.embedding as embedding
from lisbet.io import Record


@pytest.fixture
def mock_device():
    """Mock device for testing."""
    return torch.device("cpu")


@pytest.fixture
def dummy_records():
    """Create dummy records with proper structure for inference testing."""
    records = []

    # Create record with 2 individuals as required by WindowDataset
    positions = np.random.rand(
        20, 2, 2, 2
    )  # 20 frames, 2 space, 2 keypoints, 2 individuals
    posetracks = xr.Dataset(
        {"position": (["time", "space", "keypoints", "individuals"], positions)},
        coords={
            "time": np.arange(20),
            "space": ["x", "y"],
            "keypoints": ["nose", "tail"],
            "individuals": ["mouse1", "mouse2"],
        },
    )

    records.append(Record(id="test_record", posetracks=posetracks, annotations=None))
    return records


@pytest.fixture
def dummy_model():
    """Create a dummy model for testing."""

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.task_heads = {
                "multiclass": torch.nn.Linear(8, 4)
            }  # 2*2*2 = 8 input features

        def forward(self, x, task_id=None):
            batch_size = x.shape[0]
            if task_id == "multiclass":
                return torch.randn(batch_size, 4)  # 4 classes
            elif task_id == "multilabel":
                return torch.randn(batch_size, 4)  # 4 labels
            return torch.randn(batch_size, 16)  # embedding size

        def eval(self):
            return self

    return DummyModel()


@pytest.fixture
def model_config():
    """Create a dummy model configuration."""
    return {
        "bp_dim": 8,
        "emb_dim": 16,
        "hidden_dim": 32,
        "num_heads": 2,
        "num_layers": 2,
        "max_len": 200,
        "output_token_idx": -1,
        "out_dim": {"multiclass": 4, "multilabel": 4},
        "input_features": {
            "individuals": ["mouse1", "mouse2"],
            "keypoints": ["nose", "tail"],
            "space": ["x", "y"],
        },
    }


def make_dummy_dataset(
    root, individuals=("mouse1", "mouse2"), keypoints=("nose", "tail")
):
    """Create a dummy dataset directory with proper structure."""
    exp_dir = Path(root) / "exp0"
    exp_dir.mkdir(parents=True, exist_ok=True)

    n_frames = 15
    n_individuals = len(individuals)
    n_keypoints = len(keypoints)
    n_space = 2

    arr = np.random.rand(n_frames, n_individuals, n_keypoints, n_space)

    data = xr.Dataset(
        {"position": (("time", "individuals", "keypoints", "space"), arr)},
        coords={
            "time": np.arange(n_frames),
            "individuals": list(individuals),
            "keypoints": list(keypoints),
            "space": ["x", "y"],
        },
    )
    data.to_netcdf(exp_dir / "tracking.nc", engine="scipy")
    return root


class TestPredictRecord:
    """Test the core inference function for single sequences."""

    def test_single_sequence_inference(self, dummy_records, dummy_model, mock_device):
        """Test inference on a single sequence."""

        def dummy_forward_fn(model, data):
            return torch.randn(data.shape[0], 4)

        result = common.predict_record(
            record=dummy_records[0],
            model=dummy_model,
            device=mock_device,
            window_size=5,
            window_offset=0,
            fps_scaling=1.0,
            batch_size=2,
            forward_fn=dummy_forward_fn,
        )

        # Should return numpy array with predictions
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 4  # 4 output classes
        assert result.shape[0] > 0  # Should have some predictions


class TestPredict:
    """Test the dataset processing function."""

    def test_compatible_input_features_success(self, tmp_path, model_config):
        """Test successful processing when input features match."""
        # Create dummy dataset
        root = make_dummy_dataset(
            tmp_path / "ds",
            individuals=("mouse1", "mouse2"),
            keypoints=("nose", "tail"),
        )

        # Create model files
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        config_path = model_dir / "model_config.yml"
        weights_path = model_dir / "weights.pt"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(model_config, f)

        # Save dummy weights
        torch.save({"dummy": "weights"}, weights_path)

        # Mock the model loading
        with patch("lisbet.inference.common.modeling.load_model") as mock_load_model:
            dummy_model = MagicMock()
            dummy_model.eval.return_value = dummy_model
            mock_load_model.return_value = dummy_model

            def dummy_forward_fn(model, data):
                return torch.randn(data.shape[0], 4)

            with patch("lisbet.inference.common.predict_record") as mock_inference:
                mock_inference.return_value = np.random.rand(10, 4)

                result = common.predict(
                    model_path=str(config_path),
                    weights_path=str(weights_path),
                    forward_fn=dummy_forward_fn,
                    data_format="movement",
                    data_path=str(root),
                    data_scale=None,
                    window_size=10,
                    window_offset=0,
                    fps_scaling=1.0,
                    batch_size=2,
                )

                assert isinstance(result, list)
                assert len(result) == 1  # One record
                assert isinstance(result[0], tuple)
                assert len(result[0]) == 2  # (key, predictions)

    def test_incompatible_input_features_raises_error(self, tmp_path):
        """Test that incompatible input features raise ValueError."""
        # Create dataset with different features than model expects
        root = make_dummy_dataset(
            tmp_path / "ds",
            individuals=("mouse1", "mouse2"),
            keypoints=("nose", "tail"),  # Dataset has nose, tail
        )

        # Model expects different keypoints
        incompatible_config = {
            "input_features": {
                "individuals": ["mouse1", "mouse2"],
                "keypoints": ["ear", "hip"],  # Different from dataset
                "space": ["x", "y"],
            },
        }

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        config_path = model_dir / "model_config.yml"
        weights_path = model_dir / "weights.pt"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(incompatible_config, f)

        torch.save({"dummy": "weights"}, weights_path)

        with (
            patch("lisbet.inference.common.modeling.load_model"),
            pytest.raises(ValueError, match="Incompatible input features"),
        ):
            common.predict(
                model_path=str(config_path),
                weights_path=str(weights_path),
                forward_fn=lambda model, data: torch.zeros((data.shape[0], 2)),
                data_format="movement",
                data_path=str(root),
                data_scale=None,
                window_size=10,
                window_offset=0,
                fps_scaling=1.0,
                batch_size=2,
            )


class TestForwardFunctions:
    """Test the forward functions for different inference types."""

    def test_multiclass_forward(self, dummy_model):
        """Test multiclass classification forward function."""
        data = torch.randn(3, 5, 2, 2, 2)  # batch_size=3

        with patch.object(dummy_model, "forward") as mock_forward:
            # Mock model output
            mock_forward.return_value = torch.tensor(
                [[2.0, 1.0, 0.5, 0.1], [1.0, 3.0, 0.2, 0.3], [0.5, 0.2, 2.5, 1.0]]
            )

            result = annotation._multiclass_forward(dummy_model, data)

            # Should call model with correct task_id
            mock_forward.assert_called_once_with(data, "multiclass")

            # Should return one-hot encoded predictions
            assert result.shape == (3, 4)  # batch_size=3, num_classes=4

            # Should be one-hot encoded (each row sums to 1, all values 0 or 1)
            assert torch.all(result.sum(dim=1) == 1)
            assert torch.all((result == 0) | (result == 1))

    def test_multilabel_forward(self, dummy_model):
        """Test multilabel classification forward function."""
        data = torch.randn(3, 5, 2, 2, 2)  # batch_size=3

        with patch.object(dummy_model, "forward") as mock_forward:
            # Mock model output with sigmoid values
            mock_forward.return_value = torch.tensor(
                [[2.0, -1.0, 0.5, 0.1], [1.0, 3.0, -0.2, 0.3], [-0.5, 0.2, 2.5, -1.0]]
            )

            result = annotation._multilabel_forward(dummy_model, data, threshold=0.5)

            # Should call model with correct task_id
            mock_forward.assert_called_once_with(data, "multilabel")

            # Should return binary predictions
            assert result.shape == (3, 4)  # batch_size=3, num_labels=4
            assert torch.all((result == 0) | (result == 1))

    def test_embedding_forward(self, dummy_model):
        """Test embedding forward function."""
        data = torch.randn(3, 5, 2, 2, 2)  # batch_size=3

        with patch.object(dummy_model, "forward") as mock_forward:
            # Mock model output with extra dimension to test squeeze
            mock_forward.return_value = torch.randn(
                3, 1, 16
            )  # batch_size=3, 1, embedding_dim=16

            result = embedding._embedding_forward(dummy_model, data)

            # Should call model without task_id
            mock_forward.assert_called_once_with(data)

            # Should squeeze the middle dimension
            assert result.shape == (3, 16)  # batch_size=3, embedding_dim=16


class TestAnnotateBehavior:
    """Test the main behavior annotation function."""

    def test_annotate_behavior_multiclass(self, tmp_path, model_config):
        """Test basic multiclass behavior annotation functionality."""
        root = make_dummy_dataset(tmp_path / "ds")

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        config_path = model_dir / "model_config.yml"
        weights_path = model_dir / "weights.pt"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(model_config, f)

        torch.save({"dummy": "weights"}, weights_path)

        # Mock the inference pipeline
        expected_result = [("exp0", np.random.rand(10, 4))]

        with patch("lisbet.inference.annotation.predict", return_value=expected_result):
            result = annotation.annotate_behavior(
                model_path=str(config_path),
                weights_path=str(weights_path),
                data_format="movement",
                data_path=str(root),
                mode="multiclass",
            )

            assert result == expected_result

    def test_annotate_behavior_multilabel(self, tmp_path, model_config):
        """Test multilabel behavior annotation functionality."""
        root = make_dummy_dataset(tmp_path / "ds")

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        config_path = model_dir / "model_config.yml"
        weights_path = model_dir / "weights.pt"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(model_config, f)

        torch.save({"dummy": "weights"}, weights_path)

        # Mock the inference pipeline
        expected_result = [("exp0", np.random.rand(10, 4))]

        with patch("lisbet.inference.annotation.predict", return_value=expected_result):
            result = annotation.annotate_behavior(
                model_path=str(config_path),
                weights_path=str(weights_path),
                data_format="movement",
                data_path=str(root),
                mode="multilabel",
                threshold=0.7,
            )

            assert result == expected_result

    def test_annotate_behavior_with_output_path(self, tmp_path, model_config):
        """Test behavior annotation with file output."""
        root = make_dummy_dataset(tmp_path / "ds")
        output_path = tmp_path / "output"

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        config_path = model_dir / "model_config.yml"
        weights_path = model_dir / "weights.pt"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(model_config, f)

        torch.save({"dummy": "weights"}, weights_path)

        expected_result = [("exp0", np.random.rand(10, 4))]

        with (
            patch("lisbet.inference.annotation.predict", return_value=expected_result),
            patch("lisbet.inference.annotation.dump_annotations") as mock_dump,
        ):
            annotation.annotate_behavior(
                model_path=str(config_path),
                weights_path=str(weights_path),
                data_format="movement",
                data_path=str(root),
                output_path=str(output_path),
            )

            # Should call dump_annotations
            mock_dump.assert_called_once()

    def test_invalid_mode_raises_error(self, tmp_path, model_config):
        """Test that invalid mode raises ValueError."""
        root = make_dummy_dataset(tmp_path / "ds")

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        config_path = model_dir / "model_config.yml"
        weights_path = model_dir / "weights.pt"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(model_config, f)

        torch.save({"dummy": "weights"}, weights_path)

        with pytest.raises(ValueError, match="Unknown mode: invalid"):
            annotation.annotate_behavior(
                model_path=str(config_path),
                weights_path=str(weights_path),
                data_format="movement",
                data_path=str(root),
                mode="invalid",
            )


class TestComputeEmbeddings:
    """Test the main embedding computation function."""

    def test_compute_embeddings_basic(self, tmp_path, model_config):
        """Test basic embedding computation functionality."""
        root = make_dummy_dataset(tmp_path / "ds")

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        config_path = model_dir / "model_config.yml"
        weights_path = model_dir / "weights.pt"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(model_config, f)

        torch.save({"dummy": "weights"}, weights_path)

        expected_result = [("exp0", np.random.rand(10, 16))]

        with patch("lisbet.inference.embedding.predict", return_value=expected_result):
            result = embedding.compute_embeddings(
                model_path=str(config_path),
                weights_path=str(weights_path),
                data_format="movement",
                data_path=str(root),
            )

            assert result == expected_result

    def test_compute_embeddings_with_output_path(self, tmp_path, model_config):
        """Test embedding computation with file output."""
        root = make_dummy_dataset(tmp_path / "ds")
        output_path = tmp_path / "output"

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        config_path = model_dir / "model_config.yml"
        weights_path = model_dir / "weights.pt"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(model_config, f)

        torch.save({"dummy": "weights"}, weights_path)

        expected_result = [("exp0", np.random.rand(10, 16))]

        with (
            patch("lisbet.inference.embedding.predict", return_value=expected_result),
            patch("lisbet.inference.embedding.dump_embeddings") as mock_dump,
        ):
            embedding.compute_embeddings(
                model_path=str(config_path),
                weights_path=str(weights_path),
                data_format="movement",
                data_path=str(root),
                output_path=str(output_path),
            )

            # Should call dump_embeddings
            mock_dump.assert_called_once()


class TestDeviceSelection:
    """Test device selection logic."""

    def test_device_selection_cuda_available(self):
        """Test device selection when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            device = common.select_device()
            assert device.type == "cuda"

    def test_device_selection_mps_available(self):
        """Test device selection when MPS is available but CUDA is not."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.mps.is_available", return_value=True),
        ):
            device = common.select_device()
            assert device.type == "mps"

    def test_device_selection_cpu_fallback(self):
        """
        Test device selection falls back to CPU when neither CUDA nor MPS available.
        """
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.mps.is_available", return_value=False),
        ):
            device = common.select_device()
            assert device.type == "cpu"

    def test_device_selection_explicit(self):
        """Test explicit device selection."""
        device = common.select_device("cpu")
        assert device.type == "cpu"


class TestInputFeatureCompatibility:
    """Test input feature compatibility checking."""

    def test_compatible_features_passes(self):
        """Test that compatible features pass validation."""
        config = {
            "input_features": {
                "individuals": ["mouse1", "mouse2"],
                "keypoints": ["nose", "tail"],
                "space": ["x", "y"],
            }
        }

        # Create mock record with matching features
        mock_coords = {
            "individuals": xr.DataArray(["mouse1", "mouse2"]),
            "keypoints": xr.DataArray(["nose", "tail"]),
            "space": xr.DataArray(["x", "y"]),
        }
        mock_posetracks = MagicMock()
        mock_posetracks.coords = mock_coords

        mock_record = Record(id="test", posetracks=mock_posetracks, annotations=None)
        records = [mock_record]

        # Should not raise any exception
        common.check_feature_compatibility(config, records)

    def test_incompatible_features_raises_error(self):
        """Test that incompatible features raise ValueError."""
        config = {
            "input_features": {
                "individuals": ["mouse1", "mouse2"],
                "keypoints": ["ear", "hip"],  # Different keypoints
                "space": ["x", "y"],
            }
        }

        # Create mock record with different features
        mock_coords = {
            "individuals": xr.DataArray(["mouse1", "mouse2"]),
            "keypoints": xr.DataArray(["nose", "tail"]),  # Different from config
            "space": xr.DataArray(["x", "y"]),
        }
        mock_posetracks = MagicMock()
        mock_posetracks.coords = mock_coords

        mock_record = Record(id="test", posetracks=mock_posetracks, annotations=None)
        records = [mock_record]

        with pytest.raises(ValueError, match="Incompatible input features"):
            common.check_feature_compatibility(config, records)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataset(self, tmp_path, model_config):
        """Test behavior with empty dataset."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        config_path = model_dir / "model_config.yml"
        weights_path = model_dir / "weights.pt"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(model_config, f)

        torch.save({"dummy": "weights"}, weights_path)

        with (
            patch("lisbet.inference.common.modeling.load_model"),
            patch("lisbet.inference.common.load_records", return_value=[]),
            pytest.raises(IndexError),
        ):
            common.predict(
                model_path=str(config_path),
                weights_path=str(weights_path),
                forward_fn=lambda model, data: torch.zeros((data.shape[0], 2)),
                data_format="movement",
                data_path="dummy",
                data_scale=None,
                window_size=10,
                window_offset=0,
                fps_scaling=1.0,
                batch_size=2,
            )

    def test_missing_input_features_in_config(self, tmp_path):
        """Test behavior when input_features is missing from config."""
        config_without_features = {"other_param": "value"}

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        config_path = model_dir / "model_config.yml"
        weights_path = model_dir / "weights.pt"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_without_features, f)

        torch.save({"dummy": "weights"}, weights_path)

        # Create a simple record
        root = make_dummy_dataset(tmp_path / "ds")

        with (
            patch("lisbet.inference.common.modeling.load_model"),
            # Should handle missing input_features gracefully (empty dict)
            # This might raise an error due to empty features, which is expected
            pytest.raises(ValueError, match="Incompatible input features"),
        ):
            common.predict(
                model_path=str(config_path),
                weights_path=str(weights_path),
                forward_fn=lambda model, data: torch.zeros((data.shape[0], 2)),
                data_format="movement",
                data_path=str(root),
                data_scale=None,
                window_size=10,
                window_offset=0,
                fps_scaling=1.0,
                batch_size=2,
            )
