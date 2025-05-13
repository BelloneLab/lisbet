import numpy as np
import pytest
import xarray as xr

from lisbet.datasets.core import load_records


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
    # exp0: values 0..39, exp1: values 10..49
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


def test_data_scaling_default(dummy_dataset):
    """
    Test that data is scaled to [0, 1] per record when data_scale is None.

    Parameters
    ----------
    dummy_dataset : Path
        Path to the dummy dataset fixture.
    """
    groups = load_records(data_format="movement", data_path=dummy_dataset)
    for _, rec in groups["main_records"]:
        ds = rec["posetracks"]
        arr = ds["position"].values
        assert np.all(arr >= 0.0)
        assert np.all(arr <= 1.0)
        np.testing.assert_almost_equal(arr.min(), 0.0)
        np.testing.assert_almost_equal(arr.max(), 1.0)


def test_data_scaling_explicit_raises(dummy_dataset):
    """
    Test that explicit scaling raises ValueError if data is not in [0, 1].

    Parameters
    ----------
    dummy_dataset : Path
        Path to the dummy dataset fixture.
    """
    with pytest.raises(ValueError, match="coordinates are not in \\[0, 1\\]"):
        load_records(
            data_format="movement", data_path=dummy_dataset, data_scale="20x20"
        )


def test_data_scaling_explicit_valid(tmp_path):
    """
    Test that explicit scaling works if data is in [0, 1] after scaling.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory fixture.
    """
    root = tmp_path / "dataset_valid"
    root.mkdir()
    exp_dir = root / "exp0"
    exp_dir.mkdir()
    arr = np.linspace(0, 1, 40).reshape((10, 1, 2, 2))
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
    groups = load_records(data_format="movement", data_path=root, data_scale="1x1")
    ds = groups["main_records"][0][1]["posetracks"]
    arr = ds["position"].values
    assert np.all(arr >= 0.0)
    assert np.all(arr <= 1.0)


def test_splits_raises(dummy_dataset):
    """
    Test that requesting three splits on a tiny dataset raises ValueError.

    Parameters
    ----------
    dummy_dataset : Path
        Path to the dummy dataset fixture.
    """
    with pytest.raises(ValueError, match="the resulting train set will be empty"):
        load_records(
            data_format="movement",
            data_path=dummy_dataset,
            test_ratio=0.5,
            dev_ratio=0.5,
            test_seed=42,
            dev_seed=43,
        )


def test_valid_three_way_split(large_dummy_dataset):
    """
    Test that a valid three-way split works and all splits are non-empty.

    Parameters
    ----------
    large_dummy_dataset : Path
        Path to the large dummy dataset fixture.
    """
    groups = load_records(
        data_format="movement",
        data_path=large_dummy_dataset,
        test_ratio=0.2,
        dev_ratio=0.25,
        test_seed=42,
        dev_seed=43,
    )
    # Should have 1 test, 1 dev, 3 main
    assert len(groups["test_records"]) == 1
    assert len(groups["dev_records"]) == 1
    assert len(groups["main_records"]) == 3
    # All record IDs should be unique across splits
    all_ids = [rec[0] for group in groups.values() for rec in group]
    assert len(set(all_ids)) == 5
    splits = [set(rec[0] for rec in group) for group in groups.values()]
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            assert splits[i].isdisjoint(splits[j])


def test_data_filter(dummy_dataset):
    """
    Test that data_filter argument filters records as expected.

    Parameters
    ----------
    dummy_dataset : Path
        Path to the dummy dataset fixture.
    """
    groups = load_records(
        data_format="movement", data_path=dummy_dataset, data_filter="exp0"
    )
    assert len(groups["main_records"]) == 1
    assert groups["main_records"][0][0] == "exp0"


def test_keypoints_subset_selection(dummy_dataset):
    """
    Test that keypoints_subset argument selects only the requested individuals, coords, and keypoints.

    Parameters
    ----------
    dummy_dataset : Path
        Path to the dummy dataset fixture.
    """
    # Only select 'mouse', 'x', and 'nose'
    groups = load_records(
        data_format="movement",
        data_path=dummy_dataset,
        keypoints_subset="mouse;x;nose"
    )
    ds = groups["main_records"][0][1]["posetracks"]
    assert list(ds["individuals"].values) == ["mouse"]
    assert list(ds["space"].values) == ["x"]
    assert list(ds["keypoints"].values) == ["nose"]
