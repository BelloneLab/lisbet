"""Tests for the input_pipeline module."""

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from lisbet import input_pipeline


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
        (
            "a",
            {
                "posetracks": make_posetracks(np.arange(40)),
            },
        ),
        (
            "b",
            {
                "posetracks": make_posetracks(np.arange(40, 60)),
            },
        ),
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
