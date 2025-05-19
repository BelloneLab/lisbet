from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
import yaml

import lisbet.training as training


def make_dummy_dataset(root: Path, keypoints=("nose", "tail")):
    """
    Create a dummy dataset directory with one experiment subdirectory containing a
    tracking.nc file.
    """
    exp_dir = root / "exp0"
    exp_dir.mkdir(parents=True)
    arr = np.arange(10 * 1 * len(keypoints) * 2).reshape((10, 1, len(keypoints), 2))
    data = xr.Dataset(
        {
            "position": (
                ("time", "individuals", "keypoints", "space"),
                arr,
            )
        },
        coords={
            "time": np.arange(10),
            "individuals": ["mouse"],
            "keypoints": list(keypoints),
            "space": ["x", "y"],
        },
    )
    data.to_netcdf(exp_dir / "tracking.nc", engine="scipy")
    return root


def test_load_records_success(tmp_path):
    """Test _load_records succeeds with consistent features across datasets."""
    root1 = make_dummy_dataset(tmp_path / "ds1", keypoints=("nose", "tail"))
    root2 = make_dummy_dataset(tmp_path / "ds2", keypoints=("nose", "tail"))
    train_rec, test_rec, dev_rec = training._load_records(
        data_format="movement,movement",
        data_path=f"{root1},{root2}",
        data_scale=None,
        data_filter=None,
        dev_ratio=None,
        test_ratio=None,
        dev_seed=None,
        test_seed=None,
        select_coords=None,
        rename_coords=None,
        task_ids=["task"],
        task_data=None,
    )
    assert "task" in train_rec
    assert len(train_rec["task"]) == 2


def test_load_records_inconsistent_features_raises(tmp_path):
    """
    Test _load_records raises ValueError if features are inconsistent across datasets.
    """
    root1 = make_dummy_dataset(tmp_path / "ds1", keypoints=("nose", "tail"))
    root2 = make_dummy_dataset(tmp_path / "ds2", keypoints=("nose",))
    with pytest.raises(
        ValueError, match="Inconsistent posetracks coordinates in loaded records"
    ):
        training._load_records(
            data_format="movement,movement",
            data_path=f"{root1},{root2}",
            data_scale=None,
            data_filter=None,
            dev_ratio=None,
            test_ratio=None,
            dev_seed=None,
            test_seed=None,
            select_coords=None,
            rename_coords=None,
            task_ids=["task"],
            task_data=None,
        )


def test_generate_seeds_deterministic_and_override():
    # Only seed_test_split is supported as override
    seeds1 = training._generate_seeds(123, ["cfc", "nwp"], 42)
    seeds2 = training._generate_seeds(123, ["cfc", "nwp"], 42)
    seeds3 = training._generate_seeds(123, ["cfc", "nwp"], None)
    assert seeds1 == seeds2
    assert "torch" in seeds1
    assert seeds1["test_split"] == 42
    assert seeds3["test_split"] != 42  # Should be different if not overridden


def test_configure_dataloaders_min_samples(monkeypatch):
    # Mock a minimal dataset and dataloader
    class DummyDataset:
        def __len__(self):
            return 10

        def resample_dataset(self):
            self.resampled = True

    class DummyDataLoader:
        def __init__(self, dataset, batch_size, sampler, num_workers, pin_memory):
            self.dataset = dataset
            self.batch_size = batch_size

    # Patch torch.utils.data.RandomSampler and DataLoader
    class DummySampler:
        def __init__(self, dataset, num_samples):
            self.dataset = dataset
            self.num_samples = num_samples

    monkeypatch.setattr(training.torch.utils.data, "RandomSampler", DummySampler)
    monkeypatch.setattr(training.torch.utils.data, "DataLoader", DummyDataLoader)

    tasks = [
        {
            "datasets": {"train": DummyDataset()},
            "resample": False,
        },
        {
            "datasets": {"train": DummyDataset()},
            "resample": True,
        },
    ]
    dataloaders = training._configure_dataloaders(
        tasks, "train", batch_size=4, group_sample=None
    )
    assert len(dataloaders) == 2
    assert isinstance(dataloaders[0], DummyDataLoader)
    assert hasattr(tasks[1]["datasets"]["train"], "resampled")


def test_compute_epoch_logs_basic():
    # Simulate two tasks
    def dummy_metric(y_true, y_pred):
        return float(np.sum(y_true == y_pred)) / len(y_true)

    tasks = [
        {"task_id": "cfc", "metric": dummy_metric},
        {"task_id": "nwp", "metric": dummy_metric},
    ]
    losses = defaultdict(list)
    labels = defaultdict(list)
    predictions = defaultdict(list)
    # Simulate 2 batches per task
    for tid in ["cfc", "nwp"]:
        labels[tid] = [np.array([0, 1, 1]), np.array([1, 0, 1])]
        predictions[tid] = [np.array([0, 1, 0]), np.array([1, 0, 1])]
        losses[tid] = [0.5, 0.3]
    logs = training._compute_epoch_logs("train", tasks, losses, labels, predictions)
    assert "cfc_train_dummy_metric" in logs
    assert "nwp_train_dummy_metric" in logs
    assert "cfc_train_loss" in logs
    assert "nwp_train_loss" in logs
    assert np.isclose(logs["cfc_train_loss"], 0.4)
    assert np.isclose(logs["nwp_train_loss"], 0.4)
    assert 0.0 <= logs["cfc_train_dummy_metric"] <= 1.0


def test_save_and_load_weights(tmp_path):
    # Minimal model
    model = torch.nn.Linear(2, 2)
    run_id = "testrun"
    filename = "weights_test.pt"
    training._save_weights(model, tmp_path, run_id, filename)
    weights_path = tmp_path / "models" / run_id / "weights" / filename
    assert weights_path.exists()
    # Load and check state dict
    state = torch.load(weights_path)
    assert isinstance(state, dict)


def test_save_model_config_and_history(tmp_path):
    run_id = "testrun"
    tasks = [{"task_id": "cfc", "out_dim": 3}, {"task_id": "nwp", "out_dim": 1}]
    input_features = [["mouse", "nose", "x"], ["mouse", "nose", "y"]]
    training._save_model_config(
        tmp_path, run_id, 200, 0, -1, 8, 32, 128, 4, 4, 200, tasks, input_features
    )
    config_path = tmp_path / "models" / run_id / "model_config.yml"
    assert config_path.exists()

    # Check input_features in config
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert "input_features" in config
    assert config["input_features"] == input_features

    # Save history
    history = [
        {"epoch": 0, "cfc_train_loss": 0.1},
        {"epoch": 1, "cfc_train_loss": 0.05},
    ]
    training._save_history(tmp_path, run_id, history)
    hist_path = tmp_path / "models" / run_id / "training_history.log"
    assert hist_path.exists()

    df = pd.read_csv(hist_path)
    assert "epoch" in df.columns
    assert "cfc_train_loss" in df.columns
