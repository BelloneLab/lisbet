import copy

import numpy as np
import pytest
import torch
import xarray as xr

from lisbet.datasets import (
    GroupConsistencyDataset,
    RandomWindowDataset,
    TemporalOrderDataset,
    TemporalShiftDataset,
    TemporalWarpDataset,
    WindowDataset,
)

# --- Fixtures and Helpers ---


@pytest.fixture
def simple_record():
    """Create a simple record with deterministic values for interpolation checks."""
    arr = np.arange(5 * 2 * 1 * 1).reshape((5, 2, 1, 1)).astype(np.float32)
    ds = xr.Dataset(
        {"position": (("time", "individuals", "keypoints", "space"), arr)},
        coords={
            "time": np.arange(5),
            "individuals": ["A", "B"],
            "keypoints": ["nose"],
            "space": ["x"],
        },
    )

    class DummyAnn:
        def __init__(self):
            # Alternate labels 0, 1, 0, 1, ...
            self.target_cls = xr.DataArray(
                np.eye(2)[np.arange(5) % 2].reshape(5, 2, 1),
                dims=("time", "behaviors", "annotators"),
            )

        def __getitem__(self, item):
            return getattr(self, item)

    rec = type("Rec", (), {})()
    rec.posetracks = ds
    rec.annotations = DummyAnn()
    rec.id = "rec"
    return rec


@pytest.fixture
def two_records(simple_record):
    """Return two simple records with offset values for swap/consistency tests."""
    rec1 = copy.deepcopy(simple_record)
    rec2 = copy.deepcopy(simple_record)
    rec2.posetracks["position"].values += 1000
    rec2.id = "rec2"
    return [rec1, rec2]


# --- WindowDataset ---


def test_windowdataset_window_and_label_correctness(simple_record):
    """Test that WindowDataset yields correct window slices and labels."""
    ds = WindowDataset([simple_record], window_size=3)
    it = iter(ds)
    # First window (should be padded at start)
    x, y = next(it)
    expected = np.array([0, 0, 0])  # All zeros due to padding
    np.testing.assert_array_equal(x["position"].values[:, 0, 0, 0], expected)
    assert y == 0  # Label from annotations at frame 0

    # Second window (should have 2 zeros, 1 real)
    x, y = next(it)
    expected = np.array([0, 0, 2])
    np.testing.assert_array_equal(x["position"].values[:, 0, 0, 0], expected)
    assert y == 1

    # Third window (should have 1 zero, 2 real)
    x, y = next(it)
    expected = np.array([0, 2, 4])
    np.testing.assert_array_equal(x["position"].values[:, 0, 0, 0], expected)
    assert y == 0

    # Middle window (no padding)
    for _ in range(2):
        x, y = next(it)
    expected = np.array([4, 6, 8])
    np.testing.assert_array_equal(x["position"].values[:, 0, 0, 0], expected)
    assert y == 0


def test_windowdataset_interpolation_values(simple_record):
    """Test that interpolation produces correct values for non-integer fps_scaling."""
    ds = WindowDataset([simple_record], window_size=3, fps_scaling=1.5)
    # For frame 2, scaled_window_size = 4, so indices: -1, 0, 1, 2
    # interp_time_coords = linspace(-1,2,3) = [-1,0.5,2]
    # Should interpolate between padded 0 and real data
    x = ds._select_and_pad(0, 2)
    # At time=-1: padded 0
    # At time=0.5: interpolate between 0 and 2 -> 1
    # At time=2: value is 4
    vals = x["position"].values[:, 0, 0, 0]
    assert np.allclose(vals, [0, 1, 4])


def test_windowdataset_transform_applied(simple_record):
    """Test that a custom transform is applied to the window."""

    class AddOne:
        def __call__(self, x):
            x = x.copy(deep=True)
            x["position"].values += 1
            return x

    ds = WindowDataset([simple_record], window_size=3, transform=AddOne())
    x, _ = next(iter(ds))
    ds_no = WindowDataset([simple_record], window_size=3)
    x_no, _ = next(iter(ds_no))
    np.testing.assert_array_equal(x["position"].values, x_no["position"].values + 1)


def test_windowdataset_errors():
    """Test error handling for WindowDataset."""
    rec = None
    with pytest.raises(ValueError):
        WindowDataset([], window_size=3)
    arr = np.zeros((2, 1, 1, 1))
    ds = xr.Dataset(
        {"position": (("time", "individuals", "keypoints", "space"), arr)},
        coords={
            "time": [0, 1],
            "individuals": ["A"],
            "keypoints": ["nose"],
            "space": ["x"],
        },
    )
    rec = type("Rec", (), {})()
    rec.posetracks = ds
    rec.annotations = None
    with pytest.raises(ValueError):
        WindowDataset([rec], window_size=1)


# --- RandomWindowDataset ---


def test_randomwindowdataset_reproducibility(simple_record):
    """Test that RandomWindowDataset yields reproducible samples with the same seed."""
    ds1 = RandomWindowDataset([simple_record], window_size=3, base_seed=42)
    ds2 = RandomWindowDataset([simple_record], window_size=3, base_seed=42)
    vals1 = [next(iter(ds1))[0]["position"].values.copy() for _ in range(5)]
    vals2 = [next(iter(ds2))[0]["position"].values.copy() for _ in range(5)]
    for a, b in zip(vals1, vals2):
        np.testing.assert_array_equal(a, b)


def test_randomwindowdataset_label_and_window(simple_record):
    """Test that RandomWindowDataset yields correct label and window for known seed."""
    ds = RandomWindowDataset([simple_record], window_size=3, base_seed=0)
    x, y = next(iter(ds))
    # Should be a valid window and label from the record
    assert x["position"].shape[0] == 3
    assert y in (0, 1)


def test_randomwindowdataset_error():
    """Test error handling for RandomWindowDataset."""
    with pytest.raises(ValueError):
        RandomWindowDataset([], window_size=3)


# --- GroupConsistencyDataset ---


def test_groupconsistencydataset_consistent(two_records):
    """
    Test that non-swapped samples are consistent (label 0, all individuals from same
    record).
    """
    ds = GroupConsistencyDataset(two_records, window_size=3, base_seed=123)
    # Force non-swap by monkeypatching torch.rand to always return >= 0.5
    orig_rand = torch.rand
    torch.rand = lambda *a, **k: torch.tensor([0.7])
    x, y = next(iter(ds))
    torch.rand = orig_rand
    assert y == 0
    # All individuals should match one of the records
    vals = x["position"].values
    for rec in two_records:
        if np.allclose(vals, rec.posetracks["position"].values[:3]):
            break
    else:
        raise AssertionError("Individuals do not match any record")


def test_groupconsistencydataset_inconsistent(two_records):
    """
    Test that swapped samples are inconsistent (label 1, individuals split between
    records).
    """
    ds = GroupConsistencyDataset(two_records, window_size=3, base_seed=123)
    # Force swap by monkeypatching torch.rand to always return < 0.5
    orig_rand = torch.rand
    torch.rand = lambda *a, **k: torch.tensor([0.2])
    x, y = next(iter(ds))
    torch.rand = orig_rand
    assert y == 1
    vals = x["position"].values
    # The two individuals should not have identical values (should come from different
    # records or windows)
    assert not np.allclose(vals[:, 0], vals[:, 1]), (
        "Individuals come from the same record/frames"
    )


def test_groupconsistencydataset_error():
    """Test error handling for GroupConsistencyDataset with <2 individuals."""
    arr = np.zeros((5, 1, 1, 1))
    ds = xr.Dataset(
        {"position": (("time", "individuals", "keypoints", "space"), arr)},
        coords={
            "time": np.arange(5),
            "individuals": ["A"],
            "keypoints": ["nose"],
            "space": ["x"],
        },
    )
    rec = type("Rec", (), {})()
    rec.posetracks = ds
    rec.annotations = None
    with pytest.raises(ValueError):
        ds = GroupConsistencyDataset([rec], window_size=3)
        next(iter(ds))


# --- TemporalOrderDataset ---


def test_temporalorderdataset_positive_and_negative(two_records):
    """
    Test that TemporalOrderDataset yields both positive and negative samples with
    correct concatenation.
    """
    ds = TemporalOrderDataset(two_records, window_size=4, method="simple", base_seed=42)
    found_pos = found_neg = False
    for _ in range(20):
        x, y = next(iter(ds))
        assert x["position"].shape[0] == 4
        if y == 1:
            found_pos = True
        elif y == 0:
            found_neg = True
        # Check concatenation: first half from pre, second half from post
        split = 2
        pre = x["position"].values[:split]
        post = x["position"].values[split:]
        # Can't check exact values due to randomness, but shape should match
        assert pre.shape[0] == 2 and post.shape[0] == 2
    assert found_pos and found_neg


def test_temporalorderdataset_strict_method(two_records):
    """Test that TemporalOrderDataset with 'strict' method yields valid samples."""
    ds = TemporalOrderDataset(two_records, window_size=4, method="strict", base_seed=42)
    for _ in range(10):
        x, y = next(iter(ds))
        assert y in (0, 1)
        assert x["position"].shape[0] == 4


def test_temporalorderdataset_error():
    """Test error handling for TemporalOrderDataset."""
    with pytest.raises(ValueError):
        TemporalOrderDataset([], window_size=3)


# --- TemporalShiftDataset ---


def test_temporalshiftdataset_classification_and_regression(two_records):
    """
    Test that TemporalShiftDataset yields correct classification and regression labels.
    """
    ds = TemporalShiftDataset(two_records, window_size=4, regression=False)
    for _ in range(5):
        x, y = next(iter(ds))
        assert y in (0, 1)
    ds_reg = TemporalShiftDataset(two_records, window_size=4, regression=True)
    for _ in range(5):
        x, y = next(iter(ds_reg))
        assert 0.0 <= y <= 1.0


def test_temporalshiftdataset_error():
    """Test error handling for TemporalShiftDataset with <2 individuals."""
    arr = np.zeros((5, 1, 1, 1))
    ds = xr.Dataset(
        {"position": (("time", "individuals", "keypoints", "space"), arr)},
        coords={
            "time": np.arange(5),
            "individuals": ["A"],
            "keypoints": ["nose"],
            "space": ["x"],
        },
    )
    rec = type("Rec", (), {})()
    rec.posetracks = ds
    rec.annotations = None
    with pytest.raises(ValueError):
        ds = TemporalShiftDataset([rec], window_size=3)
        next(iter(ds))


# --- TemporalWarpDataset ---


def test_temporalwarpdataset_classification_and_regression(two_records):
    """
    Test that TemporalWarpDataset yields correct classification and regression labels.
    """
    ds = TemporalWarpDataset(two_records, window_size=4, regression=False)
    for _ in range(5):
        x, y = next(iter(ds))
        assert y in (0, 1)
    ds_reg = TemporalWarpDataset(two_records, window_size=4, regression=True)
    for _ in range(5):
        x, y = next(iter(ds_reg))
        assert 0.0 <= y <= 1.0


def test_temporalwarpdataset_speed_and_interpolation(simple_record):
    ds = TemporalWarpDataset(
        [simple_record], window_size=3, min_speed=1.0, max_speed=1.0, base_seed=42
    )
    x, y = next(iter(ds))
    # Check shape
    assert x["position"].shape[0] == 3
    # Check values are from the correct record (should be in [0, 8])
    vals = x["position"].values[:, 0, 0, 0]
    assert np.all((vals >= 0) & (vals <= 8))


def test_temporalwarpdataset_error():
    """Test error handling for TemporalWarpDataset."""
    with pytest.raises(ValueError):
        TemporalWarpDataset([], window_size=3)
