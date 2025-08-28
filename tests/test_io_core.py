import numpy as np
import pytest
import xarray as xr
import tempfile
from pathlib import Path

from lisbet.io import load_records
from lisbet.io.core import dump_embeddings, dump_annotations


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
    records = load_records(data_format="movement", data_path=dummy_dataset)
    for rec in records:
        ds = rec.posetracks
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
    records = load_records(data_format="movement", data_path=root, data_scale="1x1")
    ds = records[0].posetracks
    arr = ds["position"].values
    assert np.all(arr >= 0.0)
    assert np.all(arr <= 1.0)


def test_data_filter(dummy_dataset):
    """
    Test that data_filter argument filters records as expected.

    Parameters
    ----------
    dummy_dataset : Path
        Path to the dummy dataset fixture.
    """
    records = load_records(
        data_format="movement", data_path=dummy_dataset, data_filter="exp0"
    )
    assert len(records) == 1
    assert records[0].id == "exp0"


def test_select_coords_selection(dummy_dataset):
    """
    Test that select_coords argument selects only the requested individuals, axes, and
    keypoints.
    """
    # Only select 'mouse', 'x', and 'nose'
    records = load_records(
        data_format="movement",
        data_path=dummy_dataset,
        select_coords="mouse;x;nose",
    )
    ds = records[0].posetracks
    assert list(ds["individuals"].values) == ["mouse"]
    assert list(ds["space"].values) == ["x"]
    assert list(ds["keypoints"].values) == ["nose"]


def test_select_coords_wildcard(dummy_dataset):
    """
    Test that select_coords with '*' includes all items at that level.
    """
    records = load_records(
        data_format="movement",
        data_path=dummy_dataset,
        select_coords="*;*;*",
    )
    ds = records[0].posetracks
    assert set(str(x) for x in ds["individuals"].values) == {"mouse"}
    assert set(str(x) for x in ds["space"].values) == {"x", "y"}
    assert set(str(x) for x in ds["keypoints"].values) == {"nose", "tail"}


def test_rename_coords_basic(dummy_dataset):
    """
    Test that rename_coords argument renames individuals, axes, and keypoints as
    expected.
    """
    records = load_records(
        data_format="movement",
        data_path=dummy_dataset,
        rename_coords="mouse:rat;x:horizontal;nose:snout,tail:tailbase",
    )
    ds = records[0].posetracks
    assert set(str(x) for x in ds["individuals"].values) == {"rat"}
    assert set(str(x) for x in ds["space"].values) == {"horizontal", "y"}
    assert set(str(x) for x in ds["keypoints"].values) == {"snout", "tailbase"}


def test_rename_coords_wildcard(dummy_dataset):
    """
    Test that rename_coords with '*' leaves coordinates unchanged.
    """
    records = load_records(
        data_format="movement",
        data_path=dummy_dataset,
        rename_coords="*;*;*",
    )
    ds = records[0].posetracks
    assert set(ds["individuals"].values) == {"mouse"}
    assert set(ds["space"].values) == {"x", "y"}
    assert set(ds["keypoints"].values) == {"nose", "tail"}


def test_rename_coords_invalid_format(dummy_dataset):
    """
    Test that rename_coords with invalid format raises ValueError.
    """
    with pytest.raises(ValueError, match="rename_coords must have format"):
        load_records(
            data_format="movement",
            data_path=dummy_dataset,
            rename_coords="mouse:rat;x:horizontal",  # Missing keypoints field
        )
    with pytest.raises(ValueError, match="rename_coords must have format"):
        load_records(
            data_format="movement",
            data_path=dummy_dataset,
            rename_coords="mouse:rat;;nose:snout",  # Empty field not allowed
        )


def test_inconsistent_features_raises(tmp_path):
    """
    Test that load_records raises ValueError if 'features' coordinate is inconsistent.
    """
    root = tmp_path / "dataset_feat"
    root.mkdir()
    # exp0: features = nose, tail
    exp0 = root / "exp0"
    exp0.mkdir()
    arr0 = np.arange(40).reshape((10, 1, 2, 2))
    data0 = xr.Dataset(
        {
            "position": (
                ("time", "individuals", "keypoints", "space"),
                arr0,
            )
        },
        coords={
            "time": np.arange(10),
            "individuals": ["mouse"],
            "keypoints": ["nose", "tail"],
            "space": ["x", "y"],
        },
    )
    data0.to_netcdf(exp0 / "tracking.nc", engine="scipy")
    # exp1: features = nose only
    exp1 = root / "exp1"
    exp1.mkdir()
    arr1 = np.arange(20).reshape((10, 1, 1, 2))
    data1 = xr.Dataset(
        {
            "position": (
                ("time", "individuals", "keypoints", "space"),
                arr1,
            )
        },
        coords={
            "time": np.arange(10),
            "individuals": ["mouse"],
            "keypoints": ["nose"],
            "space": ["x", "y"],
        },
    )
    data1.to_netcdf(exp1 / "tracking.nc", engine="scipy")
    with pytest.raises(
        ValueError, match="Inconsistent posetracks coordinates in record"
    ):
        load_records(data_format="movement", data_path=root)


def test_explicit_scaling_raises_on_out_of_range(tmp_path):
    """
    Test that explicit scaling raises ValueError if data is not in [0, 1] after scaling.
    """
    coords = {
        "time": np.arange(10),
        "individuals": ["mouse"],
        "keypoints": ["nose", "tail"],
        "space": ["x", "y"],
    }
    arr = np.arange(40).reshape((10, 1, 2, 2)).astype(np.float32)
    exp1_dir = tmp_path / "exp1" / "exp0"
    exp1_dir.mkdir(parents=True)
    ds1 = xr.Dataset(
        {"position": (("time", "individuals", "keypoints", "space"), arr)},
        coords=coords,
    )
    ds1.to_netcdf(exp1_dir / "tracking.nc", engine="scipy")

    with pytest.raises(ValueError, match="coordinates are not in \\[0, 1\\]"):
        load_records(
            data_format="movement", data_path=tmp_path / "exp1", data_scale="10x10"
        )


def test_explicit_and_image_size_px_scaling_identical_in_range(tmp_path):
    """
    Test that explicit scaling and image_size_px attribute scaling produce identical
    results when data is in the expected range.
    """
    coords = {
        "time": np.arange(10),
        "individuals": ["mouse"],
        "keypoints": ["nose", "tail"],
        "space": ["x", "y"],
    }
    arr_ok = np.linspace(0, 10, 40).reshape((10, 1, 2, 2)).astype(np.float32)

    exp1_dir = tmp_path / "exp1" / "exp0"
    exp1_dir.mkdir(parents=True)
    ds1 = xr.Dataset(
        {"position": (("time", "individuals", "keypoints", "space"), arr_ok)},
        coords=coords,
    )
    ds1.to_netcdf(exp1_dir / "tracking.nc", engine="scipy")

    exp2_dir = tmp_path / "exp2" / "exp0"
    exp2_dir.mkdir(parents=True)
    ds2 = xr.Dataset(
        {"position": (("time", "individuals", "keypoints", "space"), arr_ok)},
        coords=coords,
    )
    ds2.attrs["image_size_px"] = [10, 10]
    ds2.to_netcdf(exp2_dir / "tracking.nc", engine="scipy")

    records_explicit = load_records(
        data_format="movement", data_path=tmp_path / "exp1", data_scale="10x10"
    )
    records_image_size = load_records(
        data_format="movement", data_path=tmp_path / "exp2", data_scale=None
    )

    arr1 = records_explicit[0].posetracks["position"].values
    arr2 = records_image_size[0].posetracks["position"].values

    np.testing.assert_array_equal(arr1, arr2)


def test_dump_embeddings_includes_frame_index(tmp_path):
    """Test that dump_embeddings includes frame index in CSV output."""
    # Create test data: 3 frames, 4 features
    test_data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    results = [("test_record", test_data)]
    
    # Save embeddings
    dump_embeddings(results, tmp_path)
    
    # Check that CSV file was created
    csv_path = tmp_path / "embeddings" / "test_record" / "features_lisbet_embedding.csv"
    assert csv_path.exists()
    
    # Read and verify CSV content
    with open(csv_path) as f:
        lines = f.readlines()
    
    # Should have header + 3 data lines
    assert len(lines) == 4
    
    # Check header includes index column (empty first column name)
    header = lines[0].strip().split(',')
    assert header[0] == ""  # Index column has no name
    assert len(header) == 5  # index + 4 feature columns
    
    # Check first data line has frame index 0
    first_data = lines[1].strip().split(',')
    assert first_data[0] == "0"
    assert len(first_data) == 5  # index + 4 values
    
    # Check second data line has frame index 1
    second_data = lines[2].strip().split(',')
    assert second_data[0] == "1"
    
    # Check third data line has frame index 2
    third_data = lines[3].strip().split(',')
    assert third_data[0] == "2"


def test_dump_annotations_includes_frame_index(tmp_path):
    """Test that dump_annotations includes frame index in CSV output."""
    # Create test data: 3 frames, 2 behavior classes  
    test_data = np.array([[1, 0], [0, 1], [1, 1]])
    results = [("test_record", test_data)]
    
    # Save annotations
    dump_annotations(results, tmp_path)
    
    # Check that CSV file was created
    csv_path = tmp_path / "annotations" / "test_record" / "machineAnnotation_lisbet.csv"
    assert csv_path.exists()
    
    # Read and verify CSV content
    with open(csv_path) as f:
        lines = f.readlines()
    
    # Should have header + 3 data lines
    assert len(lines) == 4
    
    # Check header includes index column (empty first column name)
    header = lines[0].strip().split(',')
    assert header[0] == ""  # Index column has no name
    assert len(header) == 3  # index + 2 behavior columns
    
    # Check first data line has frame index 0
    first_data = lines[1].strip().split(',')
    assert first_data[0] == "0"
    assert len(first_data) == 3  # index + 2 values
    
    # Check second data line has frame index 1  
    second_data = lines[2].strip().split(',')
    assert second_data[0] == "1"
    
    # Check third data line has frame index 2
    third_data = lines[3].strip().split(',')
    assert third_data[0] == "2"


def test_dump_functions_create_directories(tmp_path):
    """Test that dump functions create necessary directory structure."""
    test_data = np.array([[1.0, 2.0]])
    results = [("nested/path/record", test_data)]
    
    # Test embeddings
    dump_embeddings(results, tmp_path)
    embeddings_path = tmp_path / "embeddings" / "nested" / "path" / "record" / "features_lisbet_embedding.csv"
    assert embeddings_path.exists()
    
    # Test annotations
    dump_annotations(results, tmp_path)
    annotations_path = tmp_path / "annotations" / "nested" / "path" / "record" / "machineAnnotation_lisbet.csv"
    assert annotations_path.exists()
