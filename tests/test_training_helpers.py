from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
import yaml
from sklearn.utils._param_validation import InvalidParameterError

from lisbet.datasets.core import Record
from lisbet.training.core import _compute_epoch_logs, _configure_dataloaders
from lisbet.training.io import dump_model_config, dump_weights, load_multi_records
from lisbet.training.preprocessing import split_multi_records
from lisbet.training.tasks import Task
from lisbet.training.utils import generate_seeds


@pytest.fixture
def dummy_dataset(tmp_path):
    """
    Create a dummy dataset directory with two experiments, each containing a
    tracking.nc file.
    Returns
    -------
    Path
        Path to the root of the dummy dataset directory.
    """
    root = tmp_path / "dataset"
    root.mkdir()
    for i in range(2):
        exp_dir = root / f"exp{i}"
        exp_dir.mkdir()
        arr = np.arange(40).reshape((10, 1, 2, 2)) + i * 10
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
                "keypoints": ["nose", "tail"],
                "space": ["x", "y"],
            },
        )
        data.to_netcdf(exp_dir / "tracking.nc", engine="scipy")
    return root


@pytest.fixture
def large_dummy_dataset(tmp_path):
    """
    Create a dummy dataset directory with five experiments for valid split testing.
    Returns
    -------
    Path
        Path to the root of the dummy dataset directory.
    """
    root = tmp_path / "dataset_large"
    root.mkdir()
    for i in range(5):
        exp_dir = root / f"exp{i}"
        exp_dir.mkdir()
        arr = np.arange(40).reshape((10, 1, 2, 2)) + i * 10
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
                "keypoints": ["nose", "tail"],
                "space": ["x", "y"],
            },
        )
        data.to_netcdf(exp_dir / "tracking.nc", engine="scipy")
    return root


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


def test_load_multi_records_success(tmp_path):
    """Test _load_multi_records succeeds with consistent features across datasets."""
    root1 = make_dummy_dataset(tmp_path / "ds1", keypoints=("nose", "tail"))
    root2 = make_dummy_dataset(tmp_path / "ds2", keypoints=("nose", "tail"))
    records = load_multi_records(
        data_format="movement,movement",
        data_path=f"{root1},{root2}",
        data_scale=None,
        data_filter=None,
        select_coords=None,
        rename_coords=None,
    )
    assert len(records) == 2


def test_load_multi_records_inconsistent_features_raises(tmp_path):
    """
    Test _load_multi_records raises ValueError if features are inconsistent across
    datasets.
    """
    root1 = make_dummy_dataset(tmp_path / "ds1", keypoints=("nose", "tail"))
    root2 = make_dummy_dataset(tmp_path / "ds2", keypoints=("nose",))
    with pytest.raises(
        ValueError, match="Inconsistent posetracks coordinates in loaded records"
    ):
        load_multi_records(
            data_format="movement,movement",
            data_path=f"{root1},{root2}",
            data_scale=None,
            data_filter=None,
            select_coords=None,
            rename_coords=None,
        )


def test_splits_raises(dummy_dataset):
    """
    Test that requesting a split with too large dev_ratio raises sklearn's
    InvalidParameterError.

    Parameters
    ----------
    dummy_dataset : Path
        Path to the dummy dataset fixture.
    """
    # Create dummy multi_records: one dataset with two records
    multi_records = [
        [Record(id="exp0", posetracks=None), Record(id="exp1", posetracks=None)]
    ]
    # dev_ratio=1.0 would leave no training data
    with pytest.raises(InvalidParameterError) as excinfo:
        split_multi_records(
            multi_records=multi_records,
            dev_ratio=1.0,
            dev_seed=42,
            task_ids=["cfc"],
            task_data=None,
        )
    assert "test_size" in str(excinfo.value)
    assert "Got 1.0 instead" in str(excinfo.value)


def test_valid_two_way_split(large_dummy_dataset):
    """
    Test that a valid two-way split (train/dev) works and all splits are non-empty.

    Parameters
    ----------
    large_dummy_dataset : Path
        Path to the large dummy dataset fixture.
    """
    # Create multi_records as a single dataset with 5 records
    multi_records = [[Record(id=f"exp{i}", posetracks=None) for i in range(5)]]
    # Use dev_ratio=0.2 (should give 4 train, 1 dev)
    train_rec, dev_rec = split_multi_records(
        multi_records=multi_records,
        dev_ratio=0.2,
        dev_seed=42,
        task_ids=["cfc"],
        task_data=None,
    )
    # Should have 4 train, 1 dev for task 'cfc'
    assert "cfc" in train_rec
    assert "cfc" in dev_rec
    assert len(train_rec["cfc"]) == 4
    assert len(dev_rec["cfc"]) == 1
    # All record IDs should be unique across splits
    train_ids = set(rec.id for rec in train_rec["cfc"])
    dev_ids = set(rec.id for rec in dev_rec["cfc"])
    assert train_ids.isdisjoint(dev_ids)
    assert len(train_ids | dev_ids) == 5


def test_generate_seeds_deterministic_and_override():
    seeds1 = generate_seeds(123, ["cfc", "nwp"])
    seeds2 = generate_seeds(123, ["cfc", "nwp"])
    assert seeds1 == seeds2
    assert "torch" in seeds1


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

    monkeypatch.setattr("torch.utils.data.RandomSampler", DummySampler)
    monkeypatch.setattr("torch.utils.data.DataLoader", DummyDataLoader)

    # Use Task dataclass with train_dataset attribute
    task1 = Task(
        task_id="dummy1",
        head=None,
        out_dim=1,
        loss_function=None,
        train_dataset=DummyDataset(),
        train_loss=None,
        train_score=None,
        resample=False,
    )
    task2 = Task(
        task_id="dummy2",
        head=None,
        out_dim=1,
        loss_function=None,
        train_dataset=DummyDataset(),
        train_loss=None,
        train_score=None,
        resample=True,
    )
    tasks = [task1, task2]
    dataloaders = _configure_dataloaders(
        tasks, "train", batch_size=4, group_sample=None, pin_memory=False
    )
    assert len(dataloaders) == 2
    assert isinstance(dataloaders[0], DummyDataLoader)
    assert hasattr(tasks[1].train_dataset, "resampled")


def test_compute_epoch_logs_basic():
    # Simulate two tasks with train_loss and train_score metrics
    class DummyMetric:
        def __init__(self):
            self._value = 0.5

        def update(self, *args, **kwargs):
            pass

        def compute(self):
            return self._value

        def reset(self):
            pass

    task1 = Task(
        task_id="cfc",
        head=None,
        out_dim=1,
        loss_function=None,
        train_dataset=None,
        train_loss=DummyMetric(),
        train_score=DummyMetric(),
        resample=False,
    )
    task2 = Task(
        task_id="nwp",
        head=None,
        out_dim=1,
        loss_function=None,
        train_dataset=None,
        train_loss=DummyMetric(),
        train_score=DummyMetric(),
        resample=False,
    )
    tasks = [task1, task2]
    logs = _compute_epoch_logs("train", tasks)
    assert "cfc_train_score" in logs
    assert "nwp_train_score" in logs
    assert "cfc_train_loss" in logs
    assert "nwp_train_loss" in logs
    assert np.isclose(logs["cfc_train_loss"], 0.5)
    assert np.isclose(logs["nwp_train_loss"], 0.5)
    assert 0.0 <= logs["cfc_train_score"] <= 1.0


def test_save_and_load_weights(tmp_path):
    # Minimal model
    model = torch.nn.Linear(2, 2)
    run_id = "testrun"
    filename = "weights_test.pt"
    dump_weights(model, tmp_path, run_id, filename)
    weights_path = tmp_path / "models" / run_id / "weights" / filename
    assert weights_path.exists()
    # Load and check state dict
    state = torch.load(weights_path)
    assert isinstance(state, dict)


def test_save_model_config(tmp_path):
    run_id = "testrun"
    # Use Task dataclass for tasks
    task1 = Task(
        task_id="cfc",
        head=None,
        out_dim=3,
        loss_function=None,
        train_dataset=None,
        train_loss=None,
        train_score=None,
        resample=False,
    )
    task2 = Task(
        task_id="nwp",
        head=None,
        out_dim=1,
        loss_function=None,
        train_dataset=None,
        train_loss=None,
        train_score=None,
        resample=False,
    )
    tasks = [task1, task2]
    input_features = [["mouse", "nose", "x"], ["mouse", "nose", "y"]]
    dump_model_config(
        tmp_path, run_id, 200, 0, -1, 8, 32, 128, 4, 4, 200, tasks, input_features
    )
    config_path = tmp_path / "models" / run_id / "model_config.yml"
    assert config_path.exists()

    # Check input_features in config
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert "input_features" in config
    assert config["input_features"] == input_features
