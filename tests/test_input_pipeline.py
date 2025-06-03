"""Tests for the input_pipeline module."""

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from lisbet import input_pipeline
from lisbet.datasets.core import Record


@pytest.fixture
def records():
    """Dummy records with xarray.Dataset for posetracks."""

    # Create two fake records with 10 and 5 frames, 4 features
    def make_posetracks(arr):
        # arr shape: (time, features)
        # We'll reshape to (time, individuals, keypoints, space)
        # For simplicity: 1 individual, 2 keypoints, 2 space (x, y)
        arr = arr.reshape((-1, 1, 2, 2))
        ds = xr.Dataset(
            {
                "position": (
                    ("time", "individuals", "keypoints", "space"),
                    arr,
                )
            },
            coords={
                "time": np.arange(arr.shape[0]),
                "individuals": ["mouse"],
                "keypoints": ["nose", "tail"],
                "space": ["x", "y"],
            },
        )
        # Stack features as in the LISBET pipeline
        ds = ds.stack(features=("individuals", "keypoints", "space"))

        return ds

    rec = [
        Record(id="a", posetracks=make_posetracks(np.arange(40))),
        Record(id="b", posetracks=make_posetracks(np.arange(40, 60))),
    ]

    return rec


class TestSwapMousePredictionDataset:
    def test_noop(self, records):
        dataset = input_pipeline.SwapMousePredictionDataset(
            records,
            window_size=4,
            window_offset=2,
            transform=None,
            seed=1234,
        )
        # Must call resample_dataset before using the dataset
        dataset.resample_dataset()

        for i, (actual_data, label) in enumerate(dataset):
            if label == 0:
                # Find original window in the list
                key, loc = dataset.window_catalog[dataset.main_indices[i]]
                target_data = dataset._select_and_pad(key, loc)
                npt.assert_array_equal(actual_data, target_data)


def make_multi_individual_record(n_frames=5, n_inds=3, n_keypoints=2, n_space=2):
    # Shape: (time, individuals, keypoints, space)
    arr = np.arange(n_frames * n_inds * n_keypoints * n_space).reshape(
        (n_frames, n_inds, n_keypoints, n_space)
    )
    ds = xr.Dataset(
        {
            "position": (
                ("time", "individuals", "keypoints", "space"),
                arr,
            )
        },
        coords={
            "time": np.arange(n_frames),
            "individuals": [f"mouse_{i}" for i in range(n_inds)],
            "keypoints": [f"kpt{i}" for i in range(n_keypoints)],
            "space": ["x", "y"],
        },
    )
    ds = ds.stack(features=("individuals", "keypoints", "space"))
    return ds


def test_base_dataset_individual_indices():
    rec = [
        Record(id="rec1", posetracks=make_multi_individual_record(n_inds=3)),
    ]
    ds = input_pipeline.BaseDataset(
        rec, window_size=3, window_offset=0, fps_scaling=1.0, transform=None
    )
    assert ds.individuals == ["mouse_0", "mouse_1", "mouse_2"]
    for ind in ds.individuals:
        idx = ds.individual_feature_indices[ind]
        # Each individual should have the same number of features
        assert len(idx) == len(ds.individual_feature_indices["mouse_0"])


def test_swap_mouse_prediction_dataset_swaps_only_second_individual():
    rec = [
        Record(id="rec1", posetracks=make_multi_individual_record(n_inds=3)),
        Record(id="rec2", posetracks=make_multi_individual_record(n_inds=3) + 1000),
    ]
    ds = input_pipeline.SwapMousePredictionDataset(
        rec, window_size=3, window_offset=0, fps_scaling=1.0, transform=None, seed=42
    )
    ds.resample_dataset()
    for i, (actual_data, label) in enumerate(ds):
        # Get indices for each individual
        idxs = ds.individual_feature_indices
        # Get the original and swapped data
        curr_idx = ds.main_indices[i]
        curr_key, curr_loc = ds._global_to_local(curr_idx)
        orig_data = ds._select_and_pad(curr_key, curr_loc)
        swap_idx = ds.extras[i]
        swap_key, swap_loc = ds._global_to_local(swap_idx)
        swap_data = ds._select_and_pad(swap_key, swap_loc)
        # Only the second individual's features should be swapped if label==1
        if label == 1:
            # First and third individuals unchanged
            npt.assert_array_equal(
                actual_data[..., idxs["mouse_0"]], orig_data[..., idxs["mouse_0"]]
            )
            npt.assert_array_equal(
                actual_data[..., idxs["mouse_2"]], orig_data[..., idxs["mouse_2"]]
            )
            # Second individual swapped
            npt.assert_array_equal(
                actual_data[..., idxs["mouse_1"]], swap_data[..., idxs["mouse_1"]]
            )
        else:
            # All individuals unchanged
            npt.assert_array_equal(actual_data, orig_data)


def test_delay_mouse_prediction_dataset_shifts_only_second_individual():
    rec = [
        Record(id="rec1", posetracks=make_multi_individual_record(n_inds=3)),
    ]
    ds = input_pipeline.DelayMousePredictionDataset(
        rec, window_size=3, window_offset=0, fps_scaling=1.0, transform=None, seed=123
    )
    ds.resample_dataset()
    for i, (actual_data, _) in enumerate(ds):
        idxs = ds.individual_feature_indices
        curr_idx = ds.main_indices[i]
        curr_key, curr_loc = ds._global_to_local(curr_idx)
        orig_data = ds._select_and_pad(curr_key, curr_loc)
        sft_loc = ds.extras[i]
        sft_data = ds._select_and_pad(curr_key, sft_loc)
        # Only the second individual's features should be shifted
        npt.assert_array_equal(
            actual_data[..., idxs["mouse_0"]], orig_data[..., idxs["mouse_0"]]
        )
        npt.assert_array_equal(
            actual_data[..., idxs["mouse_2"]], orig_data[..., idxs["mouse_2"]]
        )
        npt.assert_array_equal(
            actual_data[..., idxs["mouse_1"]], sft_data[..., idxs["mouse_1"]]
        )
