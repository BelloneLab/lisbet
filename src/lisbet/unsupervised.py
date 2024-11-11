"""LISBET module for sequence segmentation and dimensionality reduction."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from hmmlearn import hmm
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
from umap import UMAP


def _one_hot(targets, num_classes):
    """One-hot encoding."""
    return np.eye(num_classes, dtype=int)[np.array(targets)]


def _get_embeddings(features_path, datafilter=None):
    """
    Retrieve embeddings for a given dataset.

    Parameters
    ----------
    features_path : str or None, optional
        Path to the directory containing precomputed features. If None, features are
        computed using the provided raw data.
    datafilter : str or None, optional
        Comma-separated string specifying filters for loading precomputed features.

    Returns
    -------
    embeddings : list
        A list of tuples, where each tuple contains the file name and corresponding
        embeddings as a NumPy array.

    Notes
    -----
    If `features_path` is None, the function computes features using the specified
    embedding model. If `features_path` is provided, precomputed features are loaded
    and filtered based on the specified criteria in `datafilter`.
    """
    # Load precomputed features and filter data, if requested
    featpaths = list(Path(features_path).rglob("features_*_embedding.csv"))
    embeddings = [
        (
            str(fpath.relative_to(features_path).parent),  # key
            pd.read_csv(fpath, index_col=0, dtype=np.float32).values,  # data
        )
        for fpath in tqdm(featpaths, desc="Loading embeddings")
        if datafilter is None
        or any(
            flt in str(fpath.relative_to(features_path))
            for flt in datafilter.split(",")
        )
    ]
    logging.info("Loaded %d sequences", len(embeddings))

    logging.debug([emb[0] for emb in embeddings])

    return embeddings


def segment_hmm(
    data_path,
    num_states=4,
    num_iter=10,
    data_filter=None,
    hmm_seed=None,
    output_path=None,
):
    """Segment"""
    # Get LISBET embeddings for the dataset
    embeddings = _get_embeddings(data_path, data_filter)

    # Merge sequences
    lengths = [emb[1].shape[0] for emb in embeddings]
    all_embeddings = np.concatenate([emb[1] for emb in embeddings])

    # # NEW!!! Smoothing
    # all_embeddings = median_filter(all_embeddings, size=(30, 1), origin=(-15, 0))

    # Fit model
    hmm_model = hmm.GaussianHMM(
        n_components=num_states,
        covariance_type="full",
        random_state=hmm_seed,
        n_iter=num_iter,
        tol=1e-3,
        verbose=False,
    )
    hmm_model.fit(all_embeddings, lengths=lengths)
    hmm_history = list(hmm_model.monitor_.history)
    hmm_metrics = {
        "ll": hmm_model.score(all_embeddings, lengths=lengths),
        "aic": hmm_model.aic(all_embeddings, lengths=lengths),
        "bic": float(hmm_model.bic(all_embeddings, lengths=lengths)),
    }

    # Store fitting results on file, if requested
    if output_path is not None:
        dst_path = Path(output_path)
        dst_path.mkdir(parents=True, exist_ok=True)

        # History
        history = pd.DataFrame(hmm_history, columns=["History"])
        history.to_csv(dst_path / f"history_hmm{num_states}.csv")

        # Metrics
        with open(
            dst_path / f"metrics_hmm{num_states}.yaml", "w", encoding="utf-8"
        ) as f_yaml:
            yaml.safe_dump(hmm_metrics, f_yaml)

    # HMM predictions
    all_predictions = hmm_model.predict(all_embeddings, lengths=lengths)

    # Assign predictions to match the corresponding sequences
    predictions = []
    for seq_idx, (key, seq) in enumerate(embeddings):
        # Find prediction boundaries for the current sequence
        seq_start = sum(lengths[:seq_idx])
        seq_stop = seq_start + lengths[seq_idx]
        assert seq.shape[0] == lengths[seq_idx]
        logging.debug("Sequence start = %d, Sequence stop = %d", seq_start, seq_stop)

        # Extract prediction
        pred = all_predictions[seq_start:seq_stop]
        predictions.append((key, pred))

        # Store predictions on file, if requested
        if output_path is not None:
            dst_path = Path(output_path) / key
            dst_path.mkdir(parents=True, exist_ok=True)

            # HMM motifs
            motifs = pd.DataFrame(
                _one_hot(pred, num_states),
                columns=[f"Motif_{i}" for i in range(num_states)],
            )
            motifs.to_csv(dst_path / f"machineAnnotation_hmm{num_states}.csv")

    return hmm_history, predictions


def reduce_umap(
    data_path,
    num_dims=2,
    num_neighbors=60,
    data_filter=None,
    sample_size=None,
    sample_seed=None,
    umap_seed=None,
    output_path=None,
):
    """Dimensionality reduction using UMAP."""
    # Get LISBET embeddings for the dataset
    embeddings = _get_embeddings(data_path, data_filter)

    # Random sample
    # NOTE: Sampling the embeddings after computing/reading them is inefficient.
    #       Furthermore, the sampling logic should probably be part of the data reader.
    #       For the moment we keep it here for simplicity, but we might rethink that.
    if sample_size is not None:
        rng = np.random.default_rng(seed=sample_seed)
        indices = rng.choice(len(embeddings), size=sample_size, replace=False)
        embeddings = [embeddings[idx] for idx in indices]
        logging.info("Sampled %d sequences", len(embeddings))

    # Merge sequences
    lengths = [emb[1].shape[0] for emb in embeddings]
    all_embeddings = np.concatenate([emb[1] for emb in embeddings])

    # Scale features for dataset
    all_embeddings = MinMaxScaler().fit_transform(all_embeddings)

    # Reduce embedding size using UMAP
    all_predictions = UMAP(
        n_neighbors=num_neighbors,
        n_components=num_dims,
        random_state=umap_seed,
        verbose=True,
    ).fit_transform(all_embeddings)

    # Store fitting results on file, if requested
    if output_path is not None:
        dst_path = Path(output_path)
        dst_path.mkdir(parents=True, exist_ok=True)

    # UMAP predictions
    predictions = []
    for seq_idx, (key, seq) in enumerate(embeddings):
        # Find prediction boundaries for the current sequence
        seq_start = sum(lengths[:seq_idx])
        seq_stop = seq_start + lengths[seq_idx]
        assert seq.shape[0] == lengths[seq_idx]
        logging.debug("Sequence start = %d, Sequence stop = %d", seq_start, seq_stop)

        # Extract prediction
        pred = all_predictions[seq_start:seq_stop]
        predictions.append((key, pred))

        # Store results on file, if requested
        if output_path is not None:
            dst_path = (
                Path(output_path)
                / "embeddings"
                / key
                / f"features_umap{num_dims}d{num_neighbors}_dimred.csv"
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            umap_embeddings = pd.DataFrame(pred)
            umap_embeddings.to_csv(dst_path)

    return predictions
