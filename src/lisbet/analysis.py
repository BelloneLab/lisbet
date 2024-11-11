"""LISBET analysis.
Module for analyzing sequences of states (motifs) in a dataset.

This module provides functions to compute various statistics for sequences of states,
including bout statistics, transition probabilities, and F1 score matrices. The
functions are designed to work with sequences of states, where each state represents a
motif in the sequence.

Functions
---------
bout_stats(sequences, lengths, fps, groups=None)
    Compute statistics for sequences of states (motifs) in a dataset.

transition_probability(sequences, lengths, dummy_state_id=None, groups=None)
    Compute transition probabilities between states in a sequence.

f1_score_matrix(labels, predictions)
    Compute an F1 score matrix comparing predicted states to true labels.

"""

from itertools import groupby

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm.auto import trange


def bout_stats(sequences, lengths, fps, groups=None):
    """
    Compute statistics for sequences of states (motifs) in a dataset.

    This function calculates the mean bout duration and event rate for each motif in
    the sequence.

    Parameters
    ----------
    sequences : list or array-like
        The sequence of states, where each state represents a motif.
    lengths : list or array-like
        The length (duration) of each sequence. Must be the same length as `sequences`.
    fps : int or float
        The frame rate at which the sequences were recorded. Used to convert bout
        durations and event rates to seconds and minutes respectively.
    groups : list or array-like, optional
        A list of group labels corresponding to each sequence. If None, all sequences
        are assumed to belong to a default group. Default is None.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns: "Motif ID", "Group label", "Mean bout
        duration (s)", "Rate (events / min)". Each row corresponds to a unique motif
        in the sequences, and the DataFrame is grouped by the group label if provided.

    """
    analysis_results = []
    for i, seq_duration in enumerate(lengths):
        # Select sequence data
        start = sum(lengths[:i])
        stop = start + seq_duration
        seq_pred = sequences[start:stop]

        # Set group label, if available
        group_label = groups[i] if groups is not None else "default"

        # Compute sequence of states
        events = pd.DataFrame(
            [(k, sum(1 for i in g)) for k, g in groupby(seq_pred)],
            columns=["motif_id", "bout_duration"],
        )

        # Compute statistics
        events_stats = events.groupby("motif_id").agg(["mean", "std", "count", "sum"])

        for motif_id in events_stats.index:
            analysis_results.append(
                {
                    "Motif ID": motif_id,
                    "Group label": group_label,
                    "Mean bout duration (s)": (
                        events_stats.loc[motif_id, "bout_duration"]["mean"] / fps
                    ),
                    "Rate (events / min)": events_stats.loc[motif_id, "bout_duration"][
                        "count"
                    ]
                    / (events_stats["bout_duration"]["sum"].sum() / fps / 60),
                }
            )

    analysis_results = pd.DataFrame(
        analysis_results,
        columns=[
            "Motif ID",
            "Group label",
            "Mean bout duration (s)",
            "Rate (events / min)",
        ],
    )

    return analysis_results


def transition_probability(sequences, lengths, dummy_state_id=None, groups=None):
    """
    Compute transition probabilities between states in a sequence.

    This function calculates the probability of transitioning from one state to another
    in a sequence of states, optionally ignoring a specified dummy state.

    Parameters
    ----------
    sequences : list or array-like
        The sequence of states, where each state represents a motif.
    lengths : list or array-like
        The length (duration) of each sequence. Must be the same length as `sequences`.
    dummy_state_id : int, optional
        The state ID to be ignored in the analysis (e.g., a dummy state). If None, no
        state is ignored. Default is None.
    groups : list or array-like, optional
        A list of group labels corresponding to each sequence. If None, all sequences
        are assumed to belong to a default group. Default is None.

    Returns
    -------
    dict
        A dictionary where each key is a group label and each value is a 2D numpy array
        representing the transition probability matrix for that group. The matrix has
        dimensions (number of states) x (number of states).

    """
    # Find number of states
    assert np.min(sequences) == 0
    num_states = int(np.max(sequences)) + 1

    if dummy_state_id is not None:
        num_states = num_states - 1

    # Count transitions
    if groups is None:
        occurrences = {"default": np.zeros((num_states, num_states))}
        groups = ["default"] * len(lengths)
    else:
        occurrences = {k: np.zeros((num_states, num_states)) for k in np.unique(groups)}

    for j, seq_duration in enumerate(lengths):
        # Select sequence data
        start = sum(lengths[:j])
        stop = start + seq_duration
        seq_pred = sequences[start:stop]
        group = groups[j]

        # Compute sequence of states, ignoring the dummy state
        events = [
            (k, sum(1 for i in g)) for k, g in groupby(seq_pred) if k != dummy_state_id
        ]

        for i in range(len(events) - 1):
            src = events[i][0]
            dst = events[i + 1][0]

            # Skip self-events, introduced by ignoring the dummy state
            if src != dst:
                occurrences[group][src][dst] += 1

    # Make probability
    trans_prob = {
        group: occurrences[group] / np.sum(occurrences[group], axis=1)[:, np.newaxis]
        for group in occurrences.keys()
    }

    return trans_prob


def f1_score_matrix(labels, predictions):
    """
    Compute an F1 score matrix comparing predicted states to true labels.

    This function calculates the F1 score for each pair of predicted and true states,
    resulting in a matrix of F1 scores where each row corresponds to a predicted state
    and each column corresponds to a true state.

    Parameters
    ----------
    labels : list or array-like
        The true state labels.
    predictions : list or array-like
        The predicted state labels.

    Returns
    -------
    numpy.ndarray
        A 2D numpy array where each element [i, j] represents the F1 score for
        predicting state i as state j.

    """
    n_states = np.max(predictions) + 1
    n_classes = np.max(labels) + 1

    f1_matrix = []
    for s in trange(n_states, desc="Computing F1 score"):
        bin_pred = np.array(predictions == s, dtype=int)
        score = []
        for l in range(n_classes):
            bin_lab = np.array(labels == l, dtype=int)
            score.append(f1_score(bin_lab, bin_pred, zero_division=0.0))
        f1_matrix.append(score)
    f1_matrix = np.array(f1_matrix)

    return f1_matrix
