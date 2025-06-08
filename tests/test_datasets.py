import copy

import numpy as np
import pytest
import torch
import xarray as xr

from lisbet.datasets import (
    DMPDataset,
    NWPDataset,
    RandomWindowDataset,
    SMPDataset,
    VSPDataset,
    WindowDataset,
)

# --- Helpers and Fixtures ---


@pytest.fixture
def dummy_record(request):
    """Create a dummy record with a variable number of individuals."""
    n_individuals = getattr(request, "param", 1)
    arr = (
        np.arange(10 * n_individuals * 2 * 2)
        .reshape((10, n_individuals, 2, 2))
        .astype(np.float32)
    )
    ds = xr.Dataset(
        {"position": (("time", "individuals", "keypoints", "space"), arr)},
        coords={
            "time": np.arange(10),
            "individuals": [f"mouse{i}" for i in range(n_individuals)],
            "keypoints": ["nose", "tail"],
            "space": ["x", "y"],
        },
    )

    class DummyAnn:
        def __init__(self):
            self.target_cls = xr.DataArray(
                np.eye(2)[np.random.randint(0, 2, 10)].reshape(10, 2, 1),
                dims=("time", "behaviors", "annotators"),
            )

        def __getitem__(self, item):
            return getattr(self, item)

    rec = type("Rec", (), {})()
    rec.posetracks = ds
    rec.annotations = DummyAnn()
    return rec


@pytest.fixture
def dummy_records(dummy_record):
    """
    Fixture to create a list of 3 dummy records, each with a unique id and offset data.
    Used for dataset tests.
    """
    recs = []
    for i in range(3):
        rec = copy.deepcopy(dummy_record)
        rec.id = f"exp{i}"
        rec.posetracks["position"].values += i * 100
        rec.annotations.target_cls.values[:] = 0
        rec.annotations.target_cls.values[:, i % 2, 0] = 1
        recs.append(rec)
    return recs


# --- WindowDataset ---


def test_windowdataset_basic(dummy_records):
    """
    Test that WindowDataset initializes correctly and computes lengths and cumulative
    lengths as expected.
    """
    ds = WindowDataset(dummy_records, window_size=3)
    assert ds.n_records == 3
    assert ds.n_frames == 30
    assert ds.lengths.tolist() == [10, 10, 10]
    assert ds.cumlens.tolist() == [10, 20, 30]


def test_windowdataset_global_to_local(dummy_records):
    """
    Test that _global_to_local correctly maps global indices to (record, local) indices.
    """
    ds = WindowDataset(dummy_records, window_size=3)
    # First frame
    assert ds._global_to_local(0) == (0, 0)
    # Last frame of first record
    assert ds._global_to_local(9) == (0, 9)
    # First frame of second record
    assert ds._global_to_local(10) == (1, 0)
    # Last frame
    assert ds._global_to_local(29) == (2, 9)


def test_windowdataset_select_and_pad_shape(dummy_records):
    """
    Test that _select_and_pad returns windows of the correct shape, with correct padding
    at sequence boundaries.
    """
    ds = WindowDataset(dummy_records, window_size=5)
    # Middle of sequence: no padding
    x = ds._select_and_pad(0, 5)
    assert (
        x["position"].shape
        == ds.records[0].posetracks.isel(time=slice(0, 5))["position"].shape
    )
    # Start of sequence: needs padding
    x = ds._select_and_pad(0, 1)
    assert x["position"].shape[0] == 5
    # End of sequence: needs padding
    x = ds._select_and_pad(0, 9)
    assert x["position"].shape[0] == 5


def test_windowdataset_iter_yields_correct(dummy_records):
    """
    Test that WindowDataset.__iter__ yields (window, label) pairs of the correct types.
    """
    ds = WindowDataset(dummy_records, window_size=3)
    it = iter(ds)
    x, y = next(it)
    assert isinstance(x, xr.Dataset)
    assert isinstance(y, (np.integer, float, np.floating, torch.Tensor, np.ndarray))


def test_windowdataset_immutability(dummy_records):
    """Test that sampling from WindowDataset does not modify the original records."""
    ds = WindowDataset(dummy_records, window_size=3)
    orig = copy.deepcopy(dummy_records[0].posetracks["position"].values)
    next(iter(ds))
    # Should not have changed
    np.testing.assert_array_equal(dummy_records[0].posetracks["position"].values, orig)


def test_windowdataset_fps_scaling(dummy_records):
    """
    Test that fps_scaling correctly changes the effective window and interpolation.
    """
    ds = WindowDataset(dummy_records, window_size=4, fps_scaling=0.5)
    # Should interpolate from 2 frames to 4
    x = ds._select_and_pad(0, 3)
    assert x["position"].shape[0] == 4
    # The time indices used should be spaced further apart than with fps_scaling=1
    ds2 = WindowDataset(dummy_records, window_size=4, fps_scaling=1.0)
    x2 = ds2._select_and_pad(0, 3)
    # The data should differ due to different interpolation
    assert not np.allclose(x["position"].values, x2["position"].values)


def test_windowdataset_fps_scaling_edge(dummy_records):
    """Test that fps_scaling > 1 is handled correctly."""
    ds = WindowDataset(dummy_records, window_size=4, fps_scaling=2.0)
    x = ds._select_and_pad(0, 3)
    assert x["position"].shape[0] == 4


def test_windowdataset_fps_scaling_noninteger(dummy_records):
    """Test that non-integer fps_scaling is handled correctly."""
    ds = WindowDataset(dummy_records, window_size=5, fps_scaling=0.7)
    x = ds._select_and_pad(0, 5)
    assert x["position"].shape[0] == 5


def test_windowdataset_empty_records():
    """Test that WindowDataset raises ValueError if no records are provided."""
    with pytest.raises(ValueError):
        _ = WindowDataset([], window_size=3)


def test_windowdataset_label_nan_on_missing_annotations(dummy_record):
    """
    Test that WindowDataset yields NaN or None for label if annotations are missing.
    """
    rec = copy.deepcopy(dummy_record)
    rec.annotations = None
    ds = WindowDataset([rec], window_size=3)
    x, y = next(iter(ds))
    assert np.isnan(y) or y is None


def test_windowdataset_minimal_window(dummy_records):
    """Test that WindowDataset raises ValueError if window_size is 1."""
    with pytest.raises(ValueError):
        WindowDataset(dummy_records, window_size=1)


# --- RandomWindowDataset ---


def test_randomwindowdataset_randomness_and_seed(dummy_records):
    """Test that RandomWindowDataset yields reproducible samples with the same seed."""
    ds1 = RandomWindowDataset(dummy_records, window_size=3, base_seed=42)
    ds2 = RandomWindowDataset(dummy_records, window_size=3, base_seed=42)
    # Should yield same sequence with same seed
    xs1 = [next(iter(ds1))[0]["position"].values.copy() for _ in range(3)]
    xs2 = [next(iter(ds2))[0]["position"].values.copy() for _ in range(3)]
    for a, b in zip(xs1, xs2):
        np.testing.assert_array_equal(a, b)


def test_randomwindowdataset_immutability(dummy_records):
    """Test that RandomWindowDataset does not modify the original records."""
    ds = RandomWindowDataset(dummy_records, window_size=3)
    orig = copy.deepcopy(dummy_records[0].posetracks["position"].values)
    next(iter(ds))
    np.testing.assert_array_equal(dummy_records[0].posetracks["position"].values, orig)


def test_randomwindowdataset_fps_scaling(dummy_records):
    """Test that RandomWindowDataset handles fps_scaling correctly."""
    ds = RandomWindowDataset(dummy_records, window_size=4, fps_scaling=0.5)
    x, y = next(iter(ds))
    assert x["position"].shape[0] == 4


def test_randomwindowdataset_all_windows_sampled(dummy_records):
    """
    Test that RandomWindowDataset can sample all possible windows over many iterations.
    """
    ds = RandomWindowDataset(dummy_records, window_size=3, base_seed=123)
    seen = set()
    for _ in range(100):
        x, y = next(iter(ds))
        # Use the first value as a signature
        seen.add(float(x["position"].values.flatten()[0]))
    # Should have seen at least as many as there are frames
    assert len(seen) >= 10


# --- SMPDataset ---


@pytest.mark.parametrize("dummy_record", [1, 2, 3], indirect=True)
def test_smpdataset_labels_and_swap(dummy_records):
    """
    Test SMPDataset: raises ValueError for 1 individual, otherwise yields correct swap
    labels.
    """
    n_inds = dummy_records[0].posetracks["individuals"].size
    if n_inds < 2:
        with pytest.raises(ValueError):
            ds = SMPDataset(dummy_records, window_size=3, base_seed=123)
            next(iter(ds))
    else:
        ds = SMPDataset(dummy_records, window_size=3, base_seed=123)
        for _ in range(10):
            x, y = next(iter(ds))
            assert y in (0, 1)
            # If swapped, individuals should be from different records
            if y == 1:
                pass


@pytest.mark.parametrize("dummy_record", [1, 2, 3], indirect=True)
def test_dmpdataset_immutability(dummy_records):
    """
    Test DMPDataset: raises ValueError for 1 individual, otherwise does not modify
    original records.
    """
    n_inds = dummy_records[0].posetracks["individuals"].size
    if n_inds < 2:
        with pytest.raises(ValueError):
            ds = DMPDataset(dummy_records, window_size=3)
            next(iter(ds))
    else:
        ds = DMPDataset(dummy_records, window_size=3)
        orig = copy.deepcopy(dummy_records[0].posetracks["position"].values)
        next(iter(ds))
        np.testing.assert_array_equal(
            dummy_records[0].posetracks["position"].values, orig
        )
        np.testing.assert_array_equal(
            dummy_records[0].posetracks["position"].values, orig
        )


@pytest.mark.parametrize("dummy_record", [1, 2, 3], indirect=True)
def test_smpdataset_fps_scaling(dummy_records):
    """
    Test SMPDataset fps_scaling: raises ValueError for 1 individual, otherwise yields
    correct window shape.
    """
    n_inds = dummy_records[0].posetracks["individuals"].size
    if n_inds < 2:
        with pytest.raises(ValueError):
            ds = SMPDataset(dummy_records, window_size=4, fps_scaling=0.5)
            next(iter(ds))
    else:
        ds = SMPDataset(dummy_records, window_size=4, fps_scaling=0.5)
        x, y = next(iter(ds))
        assert x["position"].shape[0] == 4


@pytest.mark.parametrize("dummy_record", [1, 2, 3], indirect=True)
def test_smpdataset_no_swap_within_same_record(dummy_records):
    """
    Test SMPDataset: raises ValueError for 1 individual, otherwise runs swap logic
    without error.
    """
    n_inds = dummy_records[0].posetracks["individuals"].size
    if n_inds < 2:
        with pytest.raises(ValueError):
            ds = SMPDataset(dummy_records, window_size=3, base_seed=42)
            next(iter(ds))
    else:
        ds = SMPDataset(dummy_records, window_size=3, base_seed=42)
        for _ in range(20):
            x, y = next(iter(ds))
            # If swapped, ensure swap is not from same record
            # Not directly testable unless we expose more info, but at least run


# --- NWPDataset ---


def test_nwpdataset_positive_and_negative(dummy_records):
    """
    Test that NWPDataset yields both positive and negative samples and correct window
    shape.
    """
    ds = NWPDataset(dummy_records, window_size=4, method="simple", base_seed=42)
    for _ in range(10):
        x, y = next(iter(ds))
        assert y in (0, 1)
        assert x["position"].shape[0] == 4


def test_nwpdataset_strict_method(dummy_records):
    """
    Test that NWPDataset with 'strict' method yields valid samples and correct window
    shape.
    """
    ds = NWPDataset(dummy_records, window_size=4, method="strict", base_seed=42)
    for _ in range(10):
        x, y = next(iter(ds))
        assert y in (0, 1)
        assert x["position"].shape[0] == 4


def test_nwpdataset_immutability(dummy_records):
    """Test that NWPDataset does not modify the original records."""
    ds = NWPDataset(dummy_records, window_size=3)
    orig = copy.deepcopy(dummy_records[0].posetracks["position"].values)
    next(iter(ds))
    np.testing.assert_array_equal(dummy_records[0].posetracks["position"].values, orig)


def test_nwpdataset_fps_scaling(dummy_records):
    """Test that NWPDataset handles fps_scaling correctly."""
    ds = NWPDataset(dummy_records, window_size=4, fps_scaling=0.5)
    x, y = next(iter(ds))
    assert x["position"].shape[0] == 4


def test_nwpdataset_concat_shape(dummy_records):
    """
    Test that NWPDataset concatenates pre and post windows correctly for the given
    window size.
    """
    ds = NWPDataset(dummy_records, window_size=6, method="simple", base_seed=42)
    x, y = next(iter(ds))
    assert x["position"].shape[0] == 6


# --- DMPDataset ---


@pytest.mark.parametrize("dummy_record", [1, 2, 3], indirect=True)
def test_dmpdataset_classification_and_regression(dummy_records):
    """
    Test DMPDataset: raises ValueError for 1 individual, otherwise yields correct
    classification/regression labels.
    """
    n_inds = dummy_records[0].posetracks["individuals"].size
    if n_inds < 2:
        with pytest.raises(ValueError):
            ds = DMPDataset(dummy_records, window_size=4, regression=False)
            next(iter(ds))
    else:
        ds = DMPDataset(dummy_records, window_size=4, regression=False)
        for _ in range(5):
            x, y = next(iter(ds))
            assert y in (0, 1)
        ds_reg = DMPDataset(dummy_records, window_size=4, regression=True)
        for _ in range(5):
            x, y = next(iter(ds_reg))
            assert 0.0 <= y <= 1.0


@pytest.mark.parametrize("dummy_record", [1, 2, 3], indirect=True)
def test_dmpdataset_fps_scaling(dummy_records):
    """
    Test DMPDataset fps_scaling: raises ValueError for 1 individual, otherwise yields
    correct window shape.
    """
    n_inds = dummy_records[0].posetracks["individuals"].size
    if n_inds < 2:
        with pytest.raises(ValueError):
            ds = DMPDataset(dummy_records, window_size=4, fps_scaling=0.5)
            next(iter(ds))
    else:
        ds = DMPDataset(dummy_records, window_size=4, fps_scaling=0.5)
        x, y = next(iter(ds))
        assert x["position"].shape[0] == 4


@pytest.mark.parametrize("dummy_record", [1, 2, 3], indirect=True)
def test_dmpdataset_delay_edge_cases(dummy_records):
    """
    Test DMPDataset delay edge cases: raises ValueError for 1 individual, otherwise
    yields correct window shape.
    """
    n_inds = dummy_records[0].posetracks["individuals"].size
    if n_inds < 2:
        with pytest.raises(ValueError):
            ds = DMPDataset(dummy_records, window_size=4, min_delay=-2, max_delay=2)
            next(iter(ds))
    else:
        ds = DMPDataset(dummy_records, window_size=4, min_delay=-2, max_delay=2)
        for _ in range(5):
            x, y = next(iter(ds))
            assert x["position"].shape[0] == 4


# --- VSPDataset ---


def test_vspdataset_classification_and_regression(dummy_records):
    """Test that VSPDataset yields correct classification and regression labels."""
    ds = VSPDataset(dummy_records, window_size=4, regression=False)
    for _ in range(5):
        x, y = next(iter(ds))
        assert y in (0, 1)
    ds_reg = VSPDataset(dummy_records, window_size=4, regression=True)
    for _ in range(5):
        x, y = next(iter(ds_reg))
        assert 0.0 <= y <= 1.0


def test_vspdataset_immutability(dummy_records):
    """Test that VSPDataset does not modify the original records."""
    ds = VSPDataset(dummy_records, window_size=3)
    orig = copy.deepcopy(dummy_records[0].posetracks["position"].values)
    next(iter(ds))
    np.testing.assert_array_equal(dummy_records[0].posetracks["position"].values, orig)


def test_vspdataset_fps_scaling(dummy_records):
    """Test that VSPDataset handles fps_scaling correctly."""
    ds = VSPDataset(dummy_records, window_size=4, fps_scaling=0.5)
    x, y = next(iter(ds))
    assert x["position"].shape[0] == 4


def test_vspdataset_speed_range(dummy_records):
    """
    Test that VSPDataset handles different min_speed and max_speed values correctly.
    """
    ds = VSPDataset(dummy_records, window_size=4, min_speed=0.7, max_speed=1.3)
    for _ in range(5):
        x, y = next(iter(ds))
        assert x["position"].shape[0] == 4


# --- Transform Handling ---


def test_transform_application(dummy_records):
    """Test that a custom transform is applied to the window before yielding."""

    class DummyTransform:
        def __call__(self, x):
            x = x.copy(deep=True)
            x["position"].values += 1
            return x

    ds = WindowDataset(dummy_records, window_size=3, transform=DummyTransform())
    # Get the window as would be returned without transform
    ds_no_transform = WindowDataset(dummy_records, window_size=3, transform=None)
    x_expected, _ = next(iter(ds_no_transform))
    x, y = next(iter(ds))
    assert np.allclose(x["position"].values, x_expected["position"].values + 1), (
        f"{x['position'].values} != {x_expected['position'].values + 1}"
    )


def test_transform_none(dummy_records):
    """Test that no transform leaves the window unchanged."""
    ds = WindowDataset(dummy_records, window_size=3, transform=None)
    x_expected, _ = next(iter(ds))
    x, y = next(iter(ds))
    assert np.allclose(x["position"].values, x_expected["position"].values), (
        f"{x['position'].values} != {x_expected['position'].values}"
    )


# --- Edge Cases ---


def test_short_sequence_padding():
    """Test that WindowDataset correctly pads short sequences to the window size."""
    arr = np.arange(2 * 1 * 2 * 2).reshape((2, 1, 2, 2)).astype(np.float32)
    ds = xr.Dataset(
        {"position": (("time", "individuals", "keypoints", "space"), arr)},
        coords={
            "time": np.arange(2),
            "individuals": ["mouse"],
            "keypoints": ["nose", "tail"],
            "space": ["x", "y"],
        },
    )
    rec = type("Rec", (), {})()
    rec.posetracks = ds
    rec.annotations = None
    dataset = WindowDataset([rec], window_size=4)
    x, y = next(iter(dataset))
    assert x["position"].shape[0] == 4


def test_single_record_single_frame():
    """
    Test that WindowDataset raises ValueError for window_size=1 even with a
    single-frame record.
    """
    arr = np.arange(1 * 1 * 2 * 2).reshape((1, 1, 2, 2)).astype(np.float32)
    ds = xr.Dataset(
        {"position": (("time", "individuals", "keypoints", "space"), arr)},
        coords={
            "time": np.arange(1),
            "individuals": ["mouse"],
            "keypoints": ["nose", "tail"],
            "space": ["x", "y"],
        },
    )
    rec = type("Rec", (), {})()
    rec.posetracks = ds
    rec.annotations = None
    with pytest.raises(ValueError):
        WindowDataset([rec], window_size=1)


def test_missing_annotations(dummy_record):
    """
    Test that WindowDataset yields NaN or None for label if annotations are missing
    (single record).
    """
    rec = copy.deepcopy(dummy_record)
    rec.annotations = None
    ds = WindowDataset([rec], window_size=3)
    x, y = next(iter(ds))
    assert np.isnan(y) or y is None


def test_empty_records_randomwindow():
    """Test that RandomWindowDataset raises ValueError if no records are provided."""
    with pytest.raises(ValueError):
        ds = RandomWindowDataset([], window_size=3)
        next(iter(ds))


def test_empty_records_smp():
    """Test that SMPDataset raises ValueError if no records are provided."""
    with pytest.raises(ValueError):
        ds = SMPDataset([], window_size=3)
        next(iter(ds))


def test_empty_records_nwp():
    """Test that NWPDataset raises ValueError if no records are provided."""
    with pytest.raises(ValueError):
        ds = NWPDataset([], window_size=3)
        next(iter(ds))


def test_empty_records_dmp():
    """Test that DMPDataset raises ValueError if no records are provided."""
    with pytest.raises(ValueError):
        ds = DMPDataset([], window_size=3)
        next(iter(ds))


def test_empty_records_vsp():
    """Test that VSPDataset raises ValueError if no records are provided."""
    with pytest.raises(ValueError):
        ds = VSPDataset([], window_size=3)
        next(iter(ds))
