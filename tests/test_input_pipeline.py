"""Tests for the input_pipeline module."""

import numpy as np
import numpy.testing as npt
import pytest

from lisbet import input_pipeline


@pytest.fixture
def records():
    """Dummy records."""
    # Create fake records
    rec = [
        (
            "a",
            {
                "keypoints": np.arange(40).reshape((10, 4)),
                "annotations": list(range(4)) * 10,
            },
        ),
        (
            "b",
            {
                "keypoints": np.arange(40, 60).reshape((5, 4)),
                "annotations": list(range(4)) * 4,
            },
        ),
    ]

    return rec


# class TestFrameClassificationGenerator:
#     @pytest.mark.parametrize("epoch", [3, 5])
#     @pytest.mark.parametrize("shuffle", [False, True])
#     def test_resume_from_epoch(self, records, epoch, shuffle):
#         """Test whether the random stream is properly managed when resuming from a given
#         epoch.

#         """
#         dataset = input_pipeline.frame_classification_generator(
#             records=records,
#             window_size=6,
#             n_classes=4,
#             shuffle=shuffle,
#             seed=123,
#             annotated=False,
#             epoch=0,
#         )

#         actual_data = [list(dataset) for _ in range(epoch)][-1]

#         dataset = input_pipeline.frame_classification_generator(
#             records=records,
#             window_size=6,
#             n_classes=4,
#             shuffle=shuffle,
#             seed=123,
#             annotated=False,
#             epoch=epoch - 1,
#         )

#         target_data = list(dataset)

#         for (act, _), (trg, _) in zip(actual_data, target_data):
#             npt.assert_array_equal(act, trg)


class TestSwapMousePredictionDataset:
    def test_noop(self, records):
        dataset = input_pipeline.SwapMousePredictionDataset(
            records,
            window_size=4,
            window_offset=2,
            transform=None,
            seed=1234,
        )

        for i, (actual_data, label) in enumerate(dataset):
            if label == 0:
                # Find original window in the list
                key, loc = dataset.window_catalog[dataset.main_indices[i]]
                target_data = dataset._select_and_pad(key, loc)

                npt.assert_array_equal(actual_data, target_data)
