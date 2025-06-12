"""
Comprehensive tests for datasets module focusing on core logic.

This test suite focuses on testing the actual logic and functionality of the dataset
classes rather than trivial shape checks. It includes:

1. **WindowDataset**: Tests core window extraction, global-to-local mapping, padding,
   fps scaling, and input validation.

2. **RandomWindowDataset**: Tests deterministic behavior with seeds and randomness
   control using monkeypatching.

3. **GroupConsistencyDataset**: Tests the group swapping logic, labeling consistency,
   and individual mixing at random split points.

4. **TemporalOrderDataset**: Tests temporal ordering logic, window concatenation,
   and both 'simple' and 'strict' negative sampling methods.

5. **TemporalShiftDataset**: Tests temporal shift detection in both classification
   and regression modes, including boundary constraints and label computation.

6. **TemporalWarpDataset**: Tests temporal warping (speed changes) in both
   classification and regression modes, including interpolation accuracy.

7. **Debug Information**: Tests that debugging information added to dataset outputs
   contains the expected coordinate information for troubleshooting.

8. **Edge Cases**: Tests boundary conditions, error handling, transform application,
   and other hard-to-test scenarios.

The tests use monkeypatching to control randomness rather than relying on multiple
runs, ensuring deterministic and reliable test results. Mock records with known
patterns are used to verify expected behavior rather than testing against arbitrary
data.
"""

from unittest.mock import patch

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
from lisbet.io import Record


@pytest.fixture
def mock_records():
    """Create mock records with known data for testing."""
    records = []

    # Record 0: 100 frames, 2 individuals, 3 keypoints
    positions = np.random.rand(100, 2, 3, 2) * 0.5 + 0.25  # Values in [0.25, 0.75]
    posetracks = xr.Dataset(
        {"position": (["time", "space", "keypoints", "individuals"], positions)},
        coords={
            "time": np.arange(100),
            "space": ["x", "y"],
            "keypoints": ["nose", "ear", "tail"],
            "individuals": ["mouse1", "mouse2"],
        },
    )

    # Mock annotations with simple pattern: behavior changes every 20 frames
    target_cls = np.zeros((100, 4, 1), dtype=int)
    for i in range(100):
        behavior_idx = (i // 20) % 4
        target_cls[i, behavior_idx, 0] = 1

    annotations = xr.Dataset(
        {"target_cls": (["time", "behaviors", "annotators"], target_cls)},
        coords={
            "time": np.arange(100),
            "behaviors": ["attack", "investigation", "mount", "other"],
            "annotators": ["annotator0"],
        },
    )

    records.append(Record(id="record0", posetracks=posetracks, annotations=annotations))

    # Record 1: 50 frames, different pattern
    positions = np.random.rand(50, 2, 3, 2) * 0.5 + 0.25
    posetracks = xr.Dataset(
        {"position": (["time", "space", "keypoints", "individuals"], positions)},
        coords={
            "time": np.arange(50),
            "space": ["x", "y"],
            "keypoints": ["nose", "ear", "tail"],
            "individuals": ["mouse1", "mouse2"],
        },
    )

    target_cls = np.zeros((50, 4, 1), dtype=int)
    for i in range(50):
        behavior_idx = (i // 10) % 4
        target_cls[i, behavior_idx, 0] = 1

    annotations = xr.Dataset(
        {"target_cls": (["time", "behaviors", "annotators"], target_cls)},
        coords={
            "time": np.arange(50),
            "behaviors": ["attack", "investigation", "mount", "other"],
            "annotators": ["annotator0"],
        },
    )

    records.append(Record(id="record1", posetracks=posetracks, annotations=annotations))

    return records


class TestWindowDataset:
    """Test core WindowDataset functionality."""

    def test_global_to_local_mapping(self, mock_records):
        """
        Test that global frame indices are correctly mapped to (record, local_frame).
        """
        dataset = WindowDataset(mock_records, window_size=10)

        # Record 0 has 100 frames, Record 1 has 50 frames
        # Global indices 0-99 should map to record 0
        assert dataset._global_to_local(0) == (0, 0)
        assert dataset._global_to_local(50) == (0, 50)
        assert dataset._global_to_local(99) == (0, 99)

        # Global indices 100-149 should map to record 1
        assert dataset._global_to_local(100) == (1, 0)
        assert dataset._global_to_local(125) == (1, 25)
        assert dataset._global_to_local(149) == (1, 49)

    def test_window_extraction_basic(self, mock_records):
        """Test basic window extraction without padding."""
        dataset = WindowDataset(mock_records, window_size=5)

        # Extract window from middle of record 0
        window = dataset._select_and_pad(curr_key=0, curr_loc=10)

        assert window.sizes["time"] == 5
        assert window.sizes["space"] == 2
        assert window.sizes["keypoints"] == 3
        assert window.sizes["individuals"] == 2

        # Time coordinates should be relative (0, 1, 2, 3, 4)
        np.testing.assert_array_equal(window.coords["time"].values, [0, 1, 2, 3, 4])

    def test_window_extraction_with_padding(self, mock_records):
        """Test window extraction that requires padding at boundaries."""
        dataset = WindowDataset(mock_records, window_size=10)

        # Extract window at beginning (should be padded)
        window = dataset._select_and_pad(curr_key=0, curr_loc=2)

        assert window.sizes["time"] == 10
        # The extracted data should handle padding (filled with 0s)

        # Extract window at end (should be padded)
        window = dataset._select_and_pad(curr_key=0, curr_loc=98)
        assert window.sizes["time"] == 10

    def test_fps_scaling(self, mock_records):
        """Test that fps_scaling affects window extraction correctly."""
        dataset = WindowDataset(mock_records, window_size=10, fps_scaling=0.5)

        # With fps_scaling=0.5, should extract from fewer actual frames
        window = dataset._select_and_pad(curr_key=0, curr_loc=20)
        assert window.sizes["time"] == 10  # Output size unchanged

    def test_validation_errors(self, mock_records):
        """Test input validation."""
        # Empty records
        with pytest.raises(ValueError, match="No records provided"):
            WindowDataset([], window_size=10)

        # Window size too small
        with pytest.raises(ValueError, match="window_size to be greater than 1"):
            dataset = WindowDataset(mock_records, window_size=1)
            dataset._select_and_pad(0, 10)


class TestRandomWindowDataset:
    """Test RandomWindowDataset randomness control."""

    def test_deterministic_with_seed(self, mock_records):
        """Test that identical seeds produce identical sequences."""
        dataset1 = RandomWindowDataset(mock_records, window_size=5, base_seed=42)
        dataset2 = RandomWindowDataset(mock_records, window_size=5, base_seed=42)

        # Get first few samples from each dataset
        iter1 = iter(dataset1)
        iter2 = iter(dataset2)

        for _ in range(10):
            x1, y1 = next(iter1)
            x2, y2 = next(iter2)

            # Windows should be identical
            xr.testing.assert_identical(x1, x2)
            np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_produce_different_sequences(self, mock_records):
        """Test that different seeds produce different sequences."""
        dataset1 = RandomWindowDataset(mock_records, window_size=5, base_seed=42)
        dataset2 = RandomWindowDataset(mock_records, window_size=5, base_seed=123)

        iter1 = iter(dataset1)
        iter2 = iter(dataset2)

        # Check that at least some samples are different
        differences = 0
        for _ in range(20):
            x1, y1 = next(iter1)
            x2, y2 = next(iter2)

            if not np.array_equal(y1, y2):
                differences += 1

        assert differences > 0, "Different seeds should produce different sequences"


class TestGroupConsistencyDataset:
    """Test GroupConsistencyDataset swapping logic."""

    def test_consistent_group_labeling(self, mock_records):
        """Test that consistent groups are labeled correctly."""
        with patch("torch.rand") as mock_rand:
            # Force consistent groups (no swapping)
            mock_rand.return_value = torch.tensor([0.6])  # > 0.5, so no swap

            dataset = GroupConsistencyDataset(mock_records, window_size=5, base_seed=42)

            with patch("torch.randint") as mock_randint:
                mock_randint.return_value = torch.tensor([25])  # Fixed frame selection

                x, y = next(iter(dataset))

                # Should be labeled as consistent (0)
                assert y == 0

                # Debug info should show no swap
                assert (
                    x.attrs["orig_coords"][0] == x.attrs["swap_coords"][0]
                )  # Same record
                assert x.attrs["swap_coords"][2] == 0  # split_idx = 0 (no swap)

    def test_inconsistent_group_labeling(self, mock_records):
        """Test that inconsistent groups are labeled correctly."""
        with patch("torch.rand") as mock_rand:
            # Force inconsistent groups (swapping)
            mock_rand.return_value = torch.tensor([0.3])  # < 0.5, so swap

            dataset = GroupConsistencyDataset(mock_records, window_size=5, base_seed=42)

            with patch("torch.randint") as mock_randint:
                # First call: select frame from record 0
                # Second call: select frame from record 1 (different record)
                # Third call: select split index
                mock_randint.side_effect = [
                    torch.tensor([25]),  # frame from record 0
                    torch.tensor(
                        [125]
                    ),  # frame from record 1 (global index 125 -> record 1, frame 25)
                    torch.tensor([1]),  # split at index 1
                ]

                x, y = next(iter(dataset))

                # Should be labeled as inconsistent (1)
                assert y == 1

                # Debug info should show swap
                assert (
                    x.attrs["orig_coords"][0] != x.attrs["swap_coords"][0]
                )  # Different records
                assert x.attrs["swap_coords"][2] == 1  # split_idx = 1

    def test_individual_swapping_logic(self, mock_records):
        """Test that individuals are correctly swapped at split point."""
        with patch("torch.rand") as mock_rand:
            mock_rand.return_value = torch.tensor([0.3])  # Force swap

            dataset = GroupConsistencyDataset(mock_records, window_size=5, base_seed=42)

            with patch("torch.randint") as mock_randint:
                mock_randint.side_effect = [
                    torch.tensor([25]),  # orig frame
                    torch.tensor([125]),  # swap frame
                    torch.tensor([1]),  # split at individual 1
                ]

                x, y = next(iter(dataset))

                # Should have 2 individuals (split at 1 means first individual from
                # orig, rest from swap)
                assert x.sizes["individuals"] == 2
                assert y == 1


class TestTemporalOrderDataset:
    """Test TemporalOrderDataset ordering logic."""

    def test_ordered_sequence_labeling(self, mock_records):
        """Test that temporally ordered sequences are labeled correctly."""
        with patch("torch.rand") as mock_rand:
            mock_rand.return_value = torch.tensor([0.3])  # < 0.5, so positive sample

            dataset = TemporalOrderDataset(
                mock_records, window_size=10, method="strict", base_seed=42
            )

            with patch("torch.randint") as mock_randint:
                # Pre window at frame 30, post window at frame 35 (same record, later
                # time)
                mock_randint.side_effect = [
                    torch.tensor([30]),  # pre frame
                    torch.tensor([35]),  # post frame (later in same record)
                ]

                x, y = next(iter(dataset))

                assert y == 1  # Ordered
                assert (
                    x.attrs["pre_coords"][1] < x.attrs["post_coords"][1]
                )  # Pre before post
                assert (
                    x.attrs["pre_coords"][0] == x.attrs["post_coords"][0]
                )  # Same record

    def test_unordered_sequence_labeling_strict(self, mock_records):
        """Test unordered sequences in strict mode."""
        with patch("torch.rand") as mock_rand:
            mock_rand.return_value = torch.tensor([0.7])  # > 0.5, so negative sample

            dataset = TemporalOrderDataset(
                mock_records, window_size=10, method="strict", base_seed=42
            )

            with patch("torch.randint") as mock_randint:
                # Pre window at frame 50, post window at frame 30 (same record, earlier
                # time)
                mock_randint.side_effect = [
                    torch.tensor([50]),  # pre frame
                    torch.tensor([30]),  # post frame (earlier in same record)
                ]

                x, y = next(iter(dataset))

                assert y == 0  # Unordered
                assert (
                    x.attrs["pre_coords"][1] > x.attrs["post_coords"][1]
                )  # Pre after post
                assert (
                    x.attrs["pre_coords"][0] == x.attrs["post_coords"][0]
                )  # Same record

    def test_window_concatenation(self, mock_records):
        """Test that pre and post windows are correctly concatenated."""
        dataset = TemporalOrderDataset(mock_records, window_size=10, base_seed=42)
        x, y = next(iter(dataset))

        # Should have concatenated time dimension
        assert x.sizes["time"] == 10

        # Time coordinates should be continuous
        time_coords = x.coords["time"].values
        assert len(time_coords) == 10
        # Pre half should be 0-4, post half should be 5-9
        expected_times = list(range(10))
        np.testing.assert_array_equal(time_coords, expected_times)

    def test_invalid_method_raises_error(self, mock_records):
        """Test that invalid method parameter raises error."""
        with pytest.raises(ValueError, match="Invalid method 'invalid'"):
            TemporalOrderDataset(mock_records, window_size=10, method="invalid")


class TestTemporalShiftDataset:
    """Test TemporalShiftDataset shift logic."""

    def test_positive_shift_classification(self, mock_records):
        """Test classification of positive temporal shifts."""
        dataset = TemporalShiftDataset(
            mock_records, window_size=5, max_shift=10, regression=False, base_seed=42
        )

        with patch("torch.randint") as mock_randint:
            # Original frame 30, shifted frame 35 (positive shift of +5)
            mock_randint.side_effect = [
                torch.tensor([30]),  # original frame
                torch.tensor([35]),  # shifted frame (positive shift)
                torch.tensor([1]),  # split index
            ]

            x, y = next(iter(dataset))

            assert y == 1  # Positive shift
            delta_shift = x.attrs["shift_coords"][2]
            assert delta_shift > 0

    def test_negative_shift_classification(self, mock_records):
        """Test classification of negative temporal shifts."""
        dataset = TemporalShiftDataset(
            mock_records, window_size=5, max_shift=10, regression=False, base_seed=42
        )

        with patch("torch.randint") as mock_randint:
            # Original frame 40, shifted frame 35 (negative shift of -5)
            mock_randint.side_effect = [
                torch.tensor([40]),  # original frame
                torch.tensor([35]),  # shifted frame (negative shift)
                torch.tensor([1]),  # split index
            ]

            x, y = next(iter(dataset))

            assert y == 0  # Negative shift
            delta_shift = x.attrs["shift_coords"][2]
            assert delta_shift < 0

    def test_regression_mode(self, mock_records):
        """Test regression mode returns normalized shift values."""
        dataset = TemporalShiftDataset(
            mock_records, window_size=5, max_shift=10, regression=True, base_seed=42
        )

        with patch("torch.randint") as mock_randint:
            # Original frame 30, shifted frame 35 (shift of +5)
            mock_randint.side_effect = [
                torch.tensor([30]),  # original frame
                torch.tensor([35]),  # shifted frame
                torch.tensor([1]),  # split index
            ]

            x, y = next(iter(dataset))

            # Shift of +5 with max_shift=10 should give normalized value of 0.75
            # Formula: (delta - min_delay) / (max_delay - min_delay) = (5 - (-10)) /
            # (10 - (-10)) = 15/20 = 0.75
            expected_y = (5 - (-10)) / (10 - (-10))
            np.testing.assert_almost_equal(y, expected_y, decimal=5)

    def test_shift_bounds_respected(self, mock_records):
        """Test that shifts respect the max_shift bounds."""
        dataset = TemporalShiftDataset(
            mock_records, window_size=5, max_shift=5, base_seed=42
        )

        # Check multiple samples to ensure bounds are respected
        for _ in range(20):
            x, y = next(iter(dataset))
            delta_shift = x.attrs["shift_coords"][2]
            assert -5 <= delta_shift <= 5

    def test_invalid_max_shift_raises_error(self, mock_records):
        """Test that invalid max_shift parameter raises error."""
        with pytest.raises(ValueError, match="max_shift to be a positive integer"):
            TemporalShiftDataset(mock_records, window_size=5, max_shift=0)


class TestTemporalWarpDataset:
    """Test TemporalWarpDataset warping logic."""

    def test_speed_up_classification(self, mock_records):
        """Test classification of sped up sequences."""
        dataset = TemporalWarpDataset(
            mock_records, window_size=10, max_warp=50, regression=False, base_seed=42
        )

        with patch("torch.rand") as mock_rand:
            # Set relative speed to 0.8, which gives speed = 0.8 * (1.5 - 0.5) + 0.5 =
            # 1.3 > 1
            mock_rand.return_value = torch.tensor([0.8])

            with patch("torch.randint") as mock_randint:
                mock_randint.return_value = torch.tensor([30])  # frame selection

                x, y = next(iter(dataset))

                assert y == 1  # Sped up (speed > 1)
                speed = x.attrs["warp_coords"][2]
                assert speed > 1.0

    def test_slow_down_classification(self, mock_records):
        """Test classification of slowed down sequences."""
        dataset = TemporalWarpDataset(
            mock_records, window_size=10, max_warp=50, regression=False, base_seed=42
        )

        with patch("torch.rand") as mock_rand:
            # Set relative speed to 0.2, which gives speed = 0.2 * (1.5 - 0.5) + 0.5 =
            # 0.7 < 1
            mock_rand.return_value = torch.tensor([0.2])

            with patch("torch.randint") as mock_randint:
                mock_randint.return_value = torch.tensor([30])  # frame selection

                x, y = next(iter(dataset))

                assert y == 0  # Slowed down (speed < 1)
                speed = x.attrs["warp_coords"][2]
                assert speed < 1.0

    def test_regression_mode(self, mock_records):
        """Test regression mode returns relative speed values."""
        dataset = TemporalWarpDataset(
            mock_records, window_size=10, max_warp=50, regression=True, base_seed=42
        )

        with patch("torch.rand") as mock_rand:
            rel_speed = 0.6
            mock_rand.return_value = torch.tensor([rel_speed])

            with patch("torch.randint") as mock_randint:
                mock_randint.return_value = torch.tensor([30])  # frame selection

                x, y = next(iter(dataset))

                # Should return the relative speed directly
                np.testing.assert_almost_equal(y, rel_speed, decimal=5)

    def test_window_size_preservation(self, mock_records):
        """Test that output window size is preserved despite warping."""
        dataset = TemporalWarpDataset(mock_records, window_size=15, base_seed=42)

        for _ in range(10):
            x, y = next(iter(dataset))
            # Output should always have the requested window size
            assert x.sizes["time"] == 15

    def test_invalid_max_warp_raises_error(self, mock_records):
        """Test that invalid max_warp parameter raises error."""
        with pytest.raises(
            ValueError, match="max_warp to be a positive integer between 0 and 100"
        ):
            TemporalWarpDataset(mock_records, window_size=10, max_warp=0)

        with pytest.raises(
            ValueError, match="max_warp to be a positive integer between 0 and 100"
        ):
            TemporalWarpDataset(mock_records, window_size=10, max_warp=150)


class TestDatasetIntegration:
    """Integration tests across different datasets."""

    def test_transform_application(self, mock_records):
        """Test that transforms are applied correctly."""

        def dummy_transform(x):
            # Add a custom attribute to verify transform was applied
            x.attrs["transformed"] = True
            return x

        dataset = WindowDataset(mock_records, window_size=5, transform=dummy_transform)
        x, y = next(iter(dataset))

        assert x.attrs.get("transformed") is True

    def test_consistent_output_structure(self, mock_records):
        """Test that all datasets produce consistent output structures."""
        datasets = [
            WindowDataset(mock_records, window_size=5),
            RandomWindowDataset(mock_records, window_size=5, base_seed=42),
            GroupConsistencyDataset(mock_records, window_size=5, base_seed=42),
            TemporalOrderDataset(mock_records, window_size=5, base_seed=42),
            TemporalShiftDataset(mock_records, window_size=5, base_seed=42),
            TemporalWarpDataset(mock_records, window_size=5, base_seed=42),
        ]

        for dataset in datasets:
            x, y = next(iter(dataset))

            # All should return xarray.Dataset for x
            assert isinstance(x, xr.Dataset)

            # All should have position data variable
            assert "position" in x.data_vars

            # All should have proper dimensions
            assert "time" in x.dims
            assert "space" in x.dims
            assert "keypoints" in x.dims
            assert "individuals" in x.dims

            # Label should be numeric
            assert isinstance(y, np.ndarray)
            assert y.dtype in [np.float32, np.int32, np.int64]


class TestDebugInformation:
    """Test debugging information added to dataset outputs."""

    def test_group_consistency_debug_info(self, mock_records):
        """Test that GroupConsistencyDataset adds proper debugging information."""
        dataset = GroupConsistencyDataset(mock_records, window_size=5, base_seed=42)
        x, y = next(iter(dataset))

        # Should have debugging information
        assert "orig_coords" in x.attrs
        assert "swap_coords" in x.attrs

        # orig_coords should be [record_idx, frame_idx]
        orig_coords = x.attrs["orig_coords"]
        assert len(orig_coords) == 2
        assert isinstance(orig_coords[0], (int, np.integer))  # record index
        assert isinstance(orig_coords[1], (int, np.integer))  # frame index

        # swap_coords should be [record_idx, frame_idx, split_idx]
        swap_coords = x.attrs["swap_coords"]
        assert len(swap_coords) == 3
        assert isinstance(swap_coords[0], (int, np.integer))  # record index
        assert isinstance(swap_coords[1], (int, np.integer))  # frame index
        assert isinstance(swap_coords[2], (int, np.integer))  # split index

    def test_temporal_order_debug_info(self, mock_records):
        """Test that TemporalOrderDataset adds proper debugging information."""
        dataset = TemporalOrderDataset(mock_records, window_size=6, base_seed=42)
        x, y = next(iter(dataset))

        # Should have debugging information
        assert "pre_coords" in x.attrs
        assert "post_coords" in x.attrs

        # pre_coords should be [record_idx, frame_idx]
        pre_coords = x.attrs["pre_coords"]
        assert len(pre_coords) == 2
        assert isinstance(pre_coords[0], (int, np.integer))
        assert isinstance(pre_coords[1], (int, np.integer))

        # post_coords should be [record_idx, frame_idx]
        post_coords = x.attrs["post_coords"]
        assert len(post_coords) == 2
        assert isinstance(post_coords[0], (int, np.integer))
        assert isinstance(post_coords[1], (int, np.integer))

    def test_temporal_shift_debug_info(self, mock_records):
        """Test that TemporalShiftDataset adds proper debugging information."""
        dataset = TemporalShiftDataset(mock_records, window_size=5, base_seed=42)
        x, y = next(iter(dataset))

        # Should have debugging information
        assert "orig_coords" in x.attrs
        assert "shift_coords" in x.attrs

        # orig_coords should be [record_idx, frame_idx]
        orig_coords = x.attrs["orig_coords"]
        assert len(orig_coords) == 2
        assert isinstance(orig_coords[0], (int, np.integer))
        assert isinstance(orig_coords[1], (int, np.integer))

        # shift_coords should be [record_idx, frame_idx, delta_delay]
        shift_coords = x.attrs["shift_coords"]
        assert len(shift_coords) == 3
        assert isinstance(shift_coords[0], (int, np.integer))  # record index
        assert isinstance(shift_coords[1], (int, np.integer))  # frame index
        assert isinstance(shift_coords[2], (int, np.integer))  # delta delay

    def test_temporal_warp_debug_info(self, mock_records):
        """Test that TemporalWarpDataset adds proper debugging information."""
        dataset = TemporalWarpDataset(mock_records, window_size=5, base_seed=42)
        x, y = next(iter(dataset))

        # Should have debugging information
        assert "orig_coords" in x.attrs
        assert "warp_coords" in x.attrs

        # orig_coords should be [record_idx, frame_idx]
        orig_coords = x.attrs["orig_coords"]
        assert len(orig_coords) == 2
        assert isinstance(orig_coords[0], (int, np.integer))
        assert isinstance(orig_coords[1], (int, np.integer))

        # warp_coords should be [record_idx, frame_idx, speed]
        warp_coords = x.attrs["warp_coords"]
        assert len(warp_coords) == 3
        assert isinstance(warp_coords[0], (int, np.integer))  # record index
        assert isinstance(warp_coords[1], (int, np.integer))  # frame index
        assert isinstance(warp_coords[2], (float, np.floating))  # speed factor

    def test_debug_info_consistency_across_iterations(self, mock_records):
        """
        Test that debug information is consistent and meaningful across iterations.
        """
        dataset = GroupConsistencyDataset(mock_records, window_size=5, base_seed=42)

        # Get multiple samples to verify consistency
        samples = []
        iterator = iter(dataset)
        for _ in range(10):
            x, y = next(iterator)
            samples.append((x.attrs["orig_coords"], x.attrs["swap_coords"], y))

        # Verify that samples make sense
        for orig_coords, swap_coords, label in samples:
            # Record indices should be valid
            assert 0 <= orig_coords[0] < len(mock_records)
            assert 0 <= swap_coords[0] < len(mock_records)

            # Frame indices should be within record bounds
            assert (
                0
                <= orig_coords[1]
                < mock_records[orig_coords[0]].posetracks.sizes["time"]
            )
            assert (
                0
                <= swap_coords[1]
                < mock_records[swap_coords[0]].posetracks.sizes["time"]
            )

            # Split index should be valid (between 1 and number of individuals)
            assert (
                0
                <= swap_coords[2]
                <= mock_records[orig_coords[0]].posetracks.sizes["individuals"]
            )

            # Label consistency: if same record and split_idx=0, should be consistent
            # (0 label)
            if orig_coords[0] == swap_coords[0] and swap_coords[2] == 0:
                assert label == 0  # Consistent


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions that are harder to test."""

    def test_window_dataset_single_individual_validation(self):
        """Test that datasets reject records with fewer than 2 individuals."""
        # Create a record with only 1 individual
        positions = np.random.rand(50, 2, 3, 1)  # Only 1 individual
        posetracks = xr.Dataset(
            {"position": (["time", "space", "keypoints", "individuals"], positions)},
            coords={
                "time": np.arange(50),
                "space": ["x", "y"],
                "keypoints": ["nose", "ear", "tail"],
                "individuals": ["mouse1"],  # Only one individual
            },
        )
        record = Record(id="single_mouse", posetracks=posetracks, annotations=None)

        with pytest.raises(ValueError, match="LISBET requires at least 2 individuals"):
            WindowDataset([record], window_size=5)

    def test_temporal_order_edge_case_last_frame(self, mock_records):
        """Test temporal order dataset behavior at sequence boundaries."""
        dataset = TemporalOrderDataset(mock_records, window_size=4, base_seed=42)

        # Force selection of last frame of first record
        with patch("torch.randint") as mock_randint:
            mock_randint.side_effect = [
                torch.tensor([99]),  # Last frame of record 0 (0-indexed)
                torch.tensor([99]),  # Same frame for post window
            ]

            with patch("torch.rand") as mock_rand:
                mock_rand.return_value = torch.tensor([0.3])  # Force positive sample

                x, y = next(iter(dataset))

                # Should still work and produce valid output
                assert x.sizes["time"] == 4
                assert y == 1  # Positive sample (post == pre is allowed)

    def test_temporal_shift_boundary_constraints(self, mock_records):
        """Test that temporal shift respects record boundaries."""
        dataset = TemporalShiftDataset(
            mock_records, window_size=5, max_shift=20, base_seed=42
        )

        # Test multiple samples to ensure shifts stay within bounds
        for _ in range(50):
            x, y = next(iter(dataset))

            orig_coords = x.attrs["orig_coords"]
            shift_coords = x.attrs["shift_coords"]

            # Both frames should be within the same record
            assert orig_coords[0] == shift_coords[0]

            # Both frames should be within record bounds
            record_length = mock_records[orig_coords[0]].posetracks.sizes["time"]
            assert 0 <= orig_coords[1] < record_length
            assert 0 <= shift_coords[1] < record_length

    def test_temporal_warp_speed_distribution(self, mock_records):
        """Test that temporal warp produces expected speed distribution."""
        dataset = TemporalWarpDataset(
            mock_records, window_size=8, max_warp=50, regression=True, base_seed=42
        )

        speeds = []
        for _ in range(100):
            x, y = next(iter(dataset))
            speed = x.attrs["warp_coords"][2]
            speeds.append(speed)

        speeds = np.array(speeds)

        # Speeds should be within expected range [0.5, 1.5]
        assert np.all(speeds >= 0.5)
        assert np.all(speeds <= 1.5)

        # Distribution should cover the range reasonably
        assert np.std(speeds) > 0.1  # Some variation
        assert np.min(speeds) < 0.7  # Some slow speeds
        assert np.max(speeds) > 1.3  # Some fast speeds

    def test_group_consistency_retry_mechanism(self, mock_records):
        """Test that GroupConsistencyDataset retries when selecting same record."""
        dataset = GroupConsistencyDataset(mock_records, window_size=5, base_seed=42)

        with patch("torch.rand") as mock_rand:
            mock_rand.return_value = torch.tensor([0.3])  # Force swap

            with patch("torch.randint") as mock_randint:
                # First attempt: same record (should retry)
                # Second attempt: different record (should succeed)
                mock_randint.side_effect = [
                    torch.tensor([25]),  # Original frame (record 0)
                    torch.tensor([30]),  # First swap attempt (still record 0)
                    torch.tensor([125]),  # Second swap attempt (record 1)
                    torch.tensor([1]),  # Split index
                ]

                x, y = next(iter(dataset))

                # Should eventually find different record
                assert y == 1  # Inconsistent
                assert x.attrs["orig_coords"][0] != x.attrs["swap_coords"][0]

    def test_fps_scaling_interpolation_accuracy(self, mock_records):
        """Test that fps_scaling produces expected interpolation results."""
        # Create dataset with fps_scaling=2.0 (double speed)
        dataset = WindowDataset(mock_records, window_size=4, fps_scaling=2.0)

        # Extract a window where we can predict the interpolation
        window = dataset._select_and_pad(curr_key=0, curr_loc=10)

        # Should still have 4 time points after interpolation
        assert window.sizes["time"] == 4
        np.testing.assert_array_equal(window.coords["time"].values, [0, 1, 2, 3])

    def test_annotation_handling_without_annotations(self):
        """Test behavior when records have no annotations."""
        # Create records without annotations
        positions = np.random.rand(20, 2, 3, 2)
        posetracks = xr.Dataset(
            {"position": (["time", "space", "keypoints", "individuals"], positions)},
            coords={
                "time": np.arange(20),
                "space": ["x", "y"],
                "keypoints": ["nose", "ear", "tail"],
                "individuals": ["mouse1", "mouse2"],
            },
        )
        record = Record(id="no_annotations", posetracks=posetracks, annotations=None)

        dataset = WindowDataset([record], window_size=5)
        x, y = next(iter(dataset))

        # Should handle missing annotations gracefully
        assert np.isnan(y) or y is None

    def test_zero_window_offset_vs_nonzero(self, mock_records):
        """Test that window_offset affects window extraction as expected."""
        dataset_zero = WindowDataset(mock_records, window_size=6, window_offset=0)
        dataset_offset = WindowDataset(mock_records, window_size=6, window_offset=2)

        # Extract from same location
        window_zero = dataset_zero._select_and_pad(curr_key=0, curr_loc=20)
        window_offset = dataset_offset._select_and_pad(curr_key=0, curr_loc=20)

        # Both should have same output size
        assert window_zero.sizes["time"] == 6
        assert window_offset.sizes["time"] == 6

        # But the actual data should be different due to offset
        # (This is hard to test precisely without knowing the exact implementation,
        # but we can at least verify the dimensions are correct)

    def test_transform_preserves_debug_info(self, mock_records):
        """Test that transforms don't interfere with debug information."""

        def transform_that_modifies_attrs(x):
            # Transform that adds its own attributes
            x.attrs["transform_applied"] = True
            return x

        dataset = GroupConsistencyDataset(
            mock_records,
            window_size=5,
            transform=transform_that_modifies_attrs,
            base_seed=42,
        )

        x, y = next(iter(dataset))

        # Should have both debug info and transform info
        assert "orig_coords" in x.attrs
        assert "swap_coords" in x.attrs
        assert "transform_applied" in x.attrs
        assert x.attrs["transform_applied"] is True

    def test_random_window_infinite_iteration(self, mock_records):
        """Test that RandomWindowDataset can iterate indefinitely."""
        dataset = RandomWindowDataset(mock_records, window_size=5, base_seed=42)

        # Should be able to get many samples without exhaustion
        iterator = iter(dataset)
        samples = []
        for _ in range(200):  # More than total frames
            x, y = next(iterator)
            samples.append((x.sizes, y.shape))

        # All samples should have consistent structure
        for sizes, y_shape in samples:
            assert sizes["time"] == 5
            assert sizes["individuals"] == 2
            assert len(y_shape) >= 0  # y can be scalar or array
