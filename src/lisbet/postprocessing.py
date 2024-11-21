"""LISBET"""

import logging
from itertools import combinations, groupby
from math import comb
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.signal import savgol_filter
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import f1_score, silhouette_samples, silhouette_score
from tqdm.auto import tqdm


def load_annotations(annot_root, hmm_list):
    """
    Load machine annotations from the given root directory.

    Parameters
    ----------
    annot_root : str
        The root directory containing the annotation files.
    hmm_list : list of int
        List of numbers of states for Hidden Markov Models (HMMs).

    Returns
    -------
    session_data : dict
        Dictionary where keys are session paths and values are DataFrames with
        concatenated HMM annotations.

    """
    # Find all sessions
    session_paths = set(
        fname.parent for fname in Path(annot_root).glob("**/machineAnnotation*.csv")
    )
    logging.debug("Available sessions: %d", len(session_paths))

    # Load data
    session_data = {}

    for session_path in tqdm(session_paths, desc="Loading data"):
        key = str(session_path.relative_to(annot_root))

        # Load annotation files
        annot_data = [
            pd.read_csv(
                session_path / f"machineAnnotation_hmm{hmm_id}.csv", index_col=0
            )
            for hmm_id in hmm_list
        ]
        annot_data = pd.concat(annot_data, axis=1, keys=hmm_list)
        annot_data.columns = [
            f"HMM_{x}_{y}" for x, y in annot_data.columns.to_flat_index()
        ]

        session_data[key] = annot_data

    return session_data


def _filter_by_frame(concat_data, frame_threshold):
    """
    Filter motifs based on frame threshold.

    Parameters
    ----------
    concat_data : DataFrame
        Concatenated DataFrame containing motif data.
    frame_threshold : float
        Minimum mean value for frames to be kept.

    Returns
    -------
    concat_data : DataFrame
        Filtered DataFrame with columns having mean values above the threshold.

    """
    # Calculate the mean of each column
    mean_values = concat_data.mean(axis=0)

    # Identify columns where the mean is greater than or equal to the threshold
    valid_columns = mean_values[mean_values >= frame_threshold].index
    logging.debug("Filter by frame valid columns: %s", valid_columns.values)

    # Filter the DataFrame to keep only the desired columns
    concat_data = concat_data[valid_columns]

    return concat_data


def _filter_by_bout(concat_data, bout_threshold, fps):
    """
    Filter motifs based on bout threshold.

    Parameters
    ----------
    concat_data : DataFrame
        Concatenated DataFrame containing motif data.
    bout_threshold : float
        Minimum mean bout duration for motifs to be kept.
    fps : int
        Frames per second, used to compute bout duration.

    Returns
    -------
    concat_data : DataFrame
        Filtered DataFrame with columns having mean bout durations above the threshold.

    """
    events = []

    for column_name in tqdm(concat_data.columns, desc="Computing bouts duration"):
        column_data = concat_data[column_name]

        # Compute bout duration
        events.extend(
            [
                (column_name, sum(1 for i in g) / fps)
                for k, g in groupby(column_data)
                if k == 1
            ]
        )

    events = pd.DataFrame(
        events,
        columns=["motif_id", "bout_duration (s)"],
    )

    events_stats = (
        events.groupby("motif_id")["bout_duration (s)"]
        .agg(["mean", "std", "count", "sum"])
        .reset_index()
    )

    valid_columns = events_stats[events_stats["mean"] >= bout_threshold]["motif_id"]
    logging.debug("Filter by bout valid columns: %s", valid_columns.values)

    concat_data = concat_data[valid_columns]

    return concat_data


def _filter_by_distance(concat_data, distance_threshold):
    cond_dist_matrix = squareform(pdist(concat_data.T, metric="jaccard"))

    valid_columns = np.where(np.sum(cond_dist_matrix < distance_threshold, axis=0) - 1)
    logging.debug(
        "Filter by distance valid columns: %s",
        concat_data.columns[valid_columns].values,
    )

    concat_data = concat_data.iloc[:, valid_columns[0]]

    return concat_data


def select_prototypes(
    data_path: str,
    hmm_list: Optional[List[int]] = None,
    hmm_range: Optional[Tuple[int, int]] = None,
    method: str = "best",
    frame_threshold: Optional[float] = None,
    bout_threshold: Optional[float] = None,
    distance_threshold: Optional[float] = None,
    fps: Optional[int] = None,
    output_path: Optional[str] = None,
) -> Tuple[Dict, List[Tuple[str, pd.DataFrame]]]:
    """
    Select motifs from a set of Hidden Markov Models using a posteriori linkage.

    Parameters
    ----------
    data_path : str
        The root directory containing the annotation files.
    hmm_list : list of int, optional
        A sorted list of unique Hidden Markov Model sizes. If `None`, `hmm_range` must be provided.
    hmm_range : tuple of int, optional
        A tuple specifying the range of Hidden Markov Model sizes (low, high). Used if `hmm_list` is `None`.
    method : str, default='best'
        Method for selecting prototypes. Valid options are 'min' and 'best'.
    frame_threshold : float, optional
        Minimum fraction of allocated frames for motifs to be kept.
    bout_threshold : float, optional
        Minimum mean bout duration in seconds for motifs to be kept.
    distance_threshold : float, optional
        Maximum Jaccard distance from the closest motif (pairs only).
    fps : int, optional
        Frames per second, used to compute bout duration.
    output_path : str, optional
        Path to store the output predictions. If `None`, results are not saved.

    Returns
    -------
    hmm_info : dict
        Dictionary containing supporting information useful for plotting the results.
    predictions : list of tuples
        List of tuples, where each tuple contains a session key and the corresponding
        motifs DataFrame.

    Notes
    -----
    [a] This method could be easily generalized to other clustering algorithms.

    """
    if hmm_list is None:
        low, high = hmm_range
        hmm_list = list(range(low, high + 1))

    # List of states must be sorted
    assert all(a < b for a, b in zip(hmm_list, hmm_list[1:]))

    # Load session data
    session_data = load_annotations(data_path, hmm_list)

    # Concatenate all sessions in a single dataset
    concat_data = pd.concat(session_data.values(), ignore_index=True)
    logging.debug("Annotation size: %s", concat_data.shape)

    # Filter motifs, if requested
    if frame_threshold is not None:
        concat_data = _filter_by_frame(concat_data, frame_threshold)
        logging.debug("Annotation size after frame threshold: %s", concat_data.shape)

    if bout_threshold is not None:
        concat_data = _filter_by_bout(concat_data, bout_threshold, fps)
        logging.debug("Annotation size after bout threshold: %s", concat_data.shape)

    if distance_threshold is not None:
        concat_data = _filter_by_distance(concat_data, distance_threshold)
        logging.debug("Annotation size after distance threshold: %s", concat_data.shape)

    # Compute distance between motifs
    cond_dist_matrix = pdist(concat_data.T, metric="jaccard")
    # cond_dist_matrix = pdist(
    #     concat_data.T,
    #     metric=lambda u, v: 1 - f1_score(u, v, average="binary"),
    # )
    # n, k = concat_data.shape[1], 2
    # cond_dist_matrix = [
    #     1 - f1_score(concat_data.iloc[:, u], concat_data.iloc[:, v], average="binary")
    #     for (u, v)
    #     in tqdm(
    #         combinations(range(n), k),
    #         desc="Computing motifs similarity",
    #         total=comb(n, k),
    #     )
    # ]

    # Compute linkage
    link_matrix = hierarchy.linkage(
        cond_dist_matrix, method="average", metric=None, optimal_ordering=True
    )

    # Scan candidate clusters
    min_clusters = 2
    max_clusters = concat_data.shape[1]
    candidates = []
    for n_clusters in range(min_clusters, max_clusters):
        # NOTE: We convert cluster labels to zero-based indexing
        labels = hierarchy.fcluster(link_matrix, n_clusters, criterion="maxclust") - 1
        score = silhouette_score(
            squareform(cond_dist_matrix), labels, metric="precomputed"
        )
        candidates.append((score, n_clusters, labels))

    # Find peak of smoothed silhouette score vector
    y_data = np.array([c[0] for c in candidates])
    y_sg = savgol_filter(y_data, window_length=max(3, max_clusters // 10), polyorder=2)

    # Get best candidate
    best_score, best_n_clusters, best_labels = candidates[np.argmax(y_sg)]
    logging.info("Best number of clusters: %d", best_n_clusters)

    # Identify prototypes
    if method == "min":
        prototypes = [
            np.where(best_labels == cid)[0][0] for cid in range(best_n_clusters)
        ]

    elif method == "best":
        samples = silhouette_samples(
            squareform(cond_dist_matrix), best_labels, metric="precomputed"
        )
        prototypes = [
            np.argmax(np.where(best_labels == cid, samples, -1))
            for cid in range(best_n_clusters)
        ]
    else:
        raise NotImplementedError(
            f"Unknown method {method}. Valid options are 'min' and 'best'"
        )
    logging.debug("Prototypes %s: %s", method, concat_data.columns[prototypes].values)

    # Assign predictions to match the corresponding sequences
    predictions = []
    for key, data in session_data.items():
        filtered_data = data[concat_data.columns]

        motifs = pd.DataFrame(filtered_data.iloc[:, prototypes])
        logging.debug(
            "Session %s prototypes %s: %s", key, method, motifs.columns.values
        )

        predictions.append((key, motifs))

        # Store predictions on file, if requested
        if output_path is not None:
            dst_path = Path(output_path) / key
            dst_path.mkdir(parents=True, exist_ok=True)

            motifs.to_csv(
                dst_path
                / f"machineAnnotation_hmm{method}_{min(hmm_list)}_{max(hmm_list)}.csv"
            )

    # Collect supporting information, useful for plotting the results
    hmm_info = {
        "cond_dist_matrix": cond_dist_matrix,
        "link_matrix": link_matrix,
        "all_score": np.array([c[0] for c in candidates]),
        "all_n_clusters": np.array([c[1] for c in candidates]),
        "all_labels": np.array([c[2] for c in candidates]),
        "best_n_clusters": best_n_clusters,
        "best_score": best_score,
        "best_labels": best_labels,
        "prototypes": prototypes,
    }

    # Store supporting information on file, if requested
    if output_path is not None:
        dst_path = Path(output_path)
        dst_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            dst_path / f"info_hmm{method}_{min(hmm_list)}_{max(hmm_list)}.npz",
            **hmm_info,
        )

    return hmm_info, predictions
