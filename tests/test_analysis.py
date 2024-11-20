"""Tests for the analysis module."""

import pandas as pd
import pytest

from lisbet import analysis


class TestBoutStats:

    def test_bout_stats_basic(self):
        sequences = [1, 1, 2, 2, 2, 1, 1, 1, 3, 3, 1, 1, 2]
        lengths = [8, 5]
        fps = 30

        result = analysis.bout_stats(sequences, lengths, fps)

        expected_data = {
            "Motif ID": [1, 2, 1, 2, 3],
            "Group label": ["default"] * 5,
            "Mean bout duration (s)": [
                (2 + 3) / (2 * fps),
                3 / fps,
                2 / fps,
                1 / fps,
                2 / fps,
            ],
            "Rate (events / min)": [
                (2 * 60 * fps) / 8,  # 2 events : 8 frames = n events : 60*FPS frames
                (1 * 60 * fps) / 8,
                (1 * 60 * fps) / 5,
                (1 * 60 * fps) / 5,
                (1 * 60 * fps) / 5,
            ],
        }

        expected_result = pd.DataFrame(expected_data)

        pd.testing.assert_frame_equal(result, expected_result)
