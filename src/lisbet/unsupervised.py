"""LISBET module for sequence segmentation and dimensionality reduction."""

import logging
import random
from pathlib import Path

import joblib
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


def _fit_hmm_func(n_components, num_iter, data, lengths, seed):
    """Fit HMM model, target function."""
    hmm_model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        random_state=seed,
        n_iter=num_iter,
        tol=1e-3,
        verbose=False,
    )
    hmm_model.fit(data, lengths=lengths)
    hmm_history = list(hmm_model.monitor_.history)
    hmm_metrics = {
        "ll": hmm_model.score(data, lengths=lengths),
        "aic": hmm_model.aic(data, lengths=lengths),
        "bic": float(hmm_model.bic(data, lengths=lengths)),
    }

    return hmm_model, hmm_history, hmm_metrics


def _fit_hmm(
    min_n_components,
    max_n_components,
    num_iter,
    embeddings,
    frac,
    n_jobs,
    seed,
    output_path=None,
):
    """Fit HMM model."""
    # Random sample
    if frac is not None:
        assert 0 < frac <= 1, "frac must be in the (0, 1] range"

        rng = np.random.default_rng(seed=seed)
        indices = rng.choice(len(embeddings), size=int(np.ceil(frac * len(embeddings))))
        fit_embeddings = [embeddings[idx] for idx in indices]
        logging.info("Sampled %d sequences for model fitting", len(fit_embeddings))
    else:
        fit_embeddings = embeddings

    # Merge sequences
    fit_lengths = [emb[1].shape[0] for emb in fit_embeddings]
    fit_embeddings = np.concatenate([emb[1] for emb in fit_embeddings])

    # Make n_components range
    # NOTE: Shuffling is used mostly to improve user experience by making the ETA more
    #       accurate. An LPT scheduler would be more efficient, but it could severely
    #       degrade user experience as the ETA would be initially overestimated.
    n_components_range = list(range(min_n_components, max_n_components + 1))
    n_models = len(n_components_range)
    random.shuffle(n_components_range)

    # Fit model
    # NOTE: We are only parallelizing HMM fitting, not the prediction step or the disk
    #       I/O. This is because the latter two steps are not CPU-bound and we wanted to
    #       keep the parallelized code as simple as possible.
    # NOTE: Using the default loky backend raises an Exception due to a bug in joblib
    #       (see https://github.com/joblib/joblib/issues/1707). The issue has been
    #       fixed, but the patch will not be available until the next release of joblib,
    #       currently at version 1.4.2. In the meantime, we can use the threading
    #       backend via prefer="threads".
    parallel = joblib.Parallel(
        n_jobs=n_jobs, return_as="generator_unordered", prefer="threads"
    )
    fitting_results = parallel(
        joblib.delayed(_fit_hmm_func)(
            n_components, num_iter, fit_embeddings, fit_lengths, seed
        )
        for n_components in n_components_range
    )

    fitted_models = []
    for hmm_model, hmm_history, hmm_metrics in tqdm(
        fitting_results, total=n_models, desc="Fitting HMM models"
    ):
        logging.debug(
            "HMM with %d states: log-likelihood = %.2f, AIC = %.2f, BIC = %.2f",
            hmm_model.n_components,
            hmm_metrics["ll"],
            hmm_metrics["aic"],
            hmm_metrics["bic"],
        )

        fitted_models.append(hmm_model)

        # Store fitting results on file, if requested
        if output_path is not None:
            dst_path = Path(output_path) / "annotations"
            dst_path.mkdir(parents=True, exist_ok=True)

            # HMM model
            joblib.dump(
                hmm_model, dst_path / f"model_hmm{hmm_model.n_components}.joblib"
            )

            # History
            hmm_history_df = pd.DataFrame(hmm_history, columns=["History"])
            hmm_history_df.to_csv(dst_path / f"history_hmm{hmm_model.n_components}.csv")

            # Metrics
            with open(
                dst_path / f"metrics_hmm{hmm_model.n_components}.yaml",
                "w",
                encoding="utf-8",
            ) as f_yaml:
                yaml.safe_dump(hmm_metrics, f_yaml)

    return fitted_models


def segment_hmm(
    data_path,
    min_n_components=2,
    max_n_components=32,
    num_iter=10,
    data_filter=None,
    fit_frac=None,
    hmm_seed=None,
    n_jobs=-1,
    pretrained_path=None,
    output_path=None,
):
    """
    Segment time series data using Hidden Markov Models.

    This function fits one or more HMM models to the embeddings and uses the models to
    segment the data into discrete states.

    Parameters
    ----------
    data_path : str or Path
        Path to the directory containing LISBET embeddings.
    min_n_components : int, optional
        Minimum number of states to use in the HMM.
    max_n_components : int, optional
        Maximum number of states to use in the HMM.
    num_iter : int, default=10
        Maximum number of iterations for the Baum-Welch algorithm.
    data_filter : callable, optional
        Function to filter the data before fitting.
    fit_frac : float, optional
        Fraction of data to use for fitting. If None, use all data.
    hmm_seed : int, optional
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs to run, -1 means using all processors.
    pretrained_path : str or Path, optional
        Path to the directory containing pretrained HMM models. If None, models are
        trained from scratch.
    output_path : str or Path, optional
        Path to save the results. If None, results are not saved to disk.

    Returns
    -------
    predictions : dict
        Dictionary mapping the number of states to the predicted segments for each
        sequence.

    Raises
    ------
    ValueError
        If min_n_components or max_n_components are smaller than 2, or max_n_components
        is smaller than min_n_components.

    """
    # Calculate the number of models to fit
    if not (2 <= min_n_components <= max_n_components):
        raise ValueError("Must satisfy: 2 <= min_n_components <= max_n_components")

    # Get LISBET embeddings for the dataset
    embeddings = _get_embeddings(data_path, data_filter)

    if pretrained_path is None:
        # Fit HMM models
        hmm_models = _fit_hmm(
            min_n_components=min_n_components,
            max_n_components=max_n_components,
            num_iter=num_iter,
            embeddings=embeddings,
            frac=fit_frac,
            n_jobs=n_jobs,
            seed=hmm_seed,
            output_path=output_path,
        )
    else:
        # Load pretrained HMM models
        hmm_models = [
            joblib.load(pretrained_path / f"model_hmm{n}.joblib")
            for n in range(min_n_components, max_n_components + 1)
        ]

    # Segment all sequences and store predictions to disk, if requested
    predictions = {}
    for hmm_model in tqdm(hmm_models, desc="Segmenting sequences"):
        # HMM predictions
        all_lengths = [emb[1].shape[0] for emb in embeddings]
        all_embeddings = np.concatenate([emb[1] for emb in embeddings])
        all_predictions = hmm_model.predict(all_embeddings, lengths=all_lengths)

        # Assign predictions to match the corresponding sequences
        hmm_predictions = []
        for seq_idx, (key, seq) in enumerate(embeddings):
            # Find prediction boundaries for the current sequence
            seq_start = sum(all_lengths[:seq_idx])
            seq_stop = seq_start + all_lengths[seq_idx]
            assert seq.shape[0] == all_lengths[seq_idx]
            logging.debug(
                "Sequence start = %d, Sequence stop = %d", seq_start, seq_stop
            )

            # Extract prediction
            pred = all_predictions[seq_start:seq_stop]
            hmm_predictions.append((key, pred))

            # Store predictions on file, if requested
            if output_path is not None:
                dst_path = Path(output_path) / "annotations" / key
                dst_path.mkdir(parents=True, exist_ok=True)

                # HMM motifs
                motifs = pd.DataFrame(
                    _one_hot(pred, hmm_model.n_components),
                    columns=[f"Motif_{i}" for i in range(hmm_model.n_components)],
                )
                motifs.to_csv(
                    dst_path / f"machineAnnotation_hmm{hmm_model.n_components}.csv"
                )

        predictions[hmm_model.n_components] = hmm_predictions

    return predictions


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
