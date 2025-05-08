"""Core dataset functionalities."""

import logging
import re
from functools import partial
from pathlib import Path

import pandas as pd
import pooch
import xarray as xr
from huggingface_hub import snapshot_download
from movement.io import load_poses
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from . import calms21


def _load_posetracks(seq_path, data_format, rescale):
    """Load pose-tracking data from a file."""
    # Valid filenames and their corresponding loading functions
    # TODO: Test re matching for all supported formats.
    data_readers = {
        "DLC": (r"(?i)(DLC.*?shuffle\d+|tracking).*\.csv$", load_poses.from_dlc_file),
        "SLEAP": (r"(?i)SLEAP.*\.h5$", load_poses.from_sleap_file),
        "movement": (r"(?i)tracking.*\.nc$", partial(xr.open_dataset, engine="scipy")),
    }

    if data_format in data_readers:
        # Find all files matching the regex and load them
        pattern, loader = data_readers[data_format]
        rx = re.compile(pattern)
        dss = [loader(pth) for pth in seq_path.iterdir() if rx.search(pth.name)]

    else:
        raise ValueError(f"Unknown data format {data_format}")

    # Check if any datasets were found
    if len(dss) == 0:
        return None

    # Merge all datasets into a single one
    # NOTE: There should be only one dataset per sequence, but we keep this for
    #       compatibility with multiple single-individual datasets
    ds = xr.concat(dss, dim="individuals")

    logging.debug("Individuals: %s", ds["individuals"].values)

    # Rescale coordinates in the (0, 1) range, if requested
    if rescale:
        reduce_dims = ("time", "keypoints", "individuals")

        pos = ds["position"]
        min_val = pos.min(dim=reduce_dims, skipna=True)
        max_val = pos.max(dim=reduce_dims, skipna=True)

        ds = ds.assign(position=(pos - min_val) / (max_val - min_val))

        logging.debug(
            "Rescaled coordinates between min values %s and max values %s",
            min_val.values,
            max_val.values,
        )

    return ds


def _load_annotations(seq_path):
    """Load annotations from a file."""
    # Find all files matching the annotations regex and load them
    rx = re.compile(r"(?i)(manual_scoring|annotations).*\.csv$")
    annotations = [
        pd.read_csv(pth, header=[0, 1])
        for pth in seq_path.iterdir()
        if rx.search(pth.name)
    ]

    # Check if any annotations were found
    if len(annotations) == 0:
        return None

    # Merge all annotations into a single one
    annotations = pd.concat(annotations, axis=0, ignore_index=True)

    logging.debug("Annotations: %s", annotations.columns.values)

    # Convert annotations to label encoding format
    # NOTE: This is a temporary solution to maintain compatibility with the current
    #       implementation of training and inference pipelines.
    # TODO: Add full support for one-hot and multi-label annotations
    annotations = annotations.values.argmax(axis=1)

    return annotations


def load_records(
    data_format,
    data_path,
    data_filter=None,
    dev_ratio=None,
    test_ratio=None,
    dev_seed=None,
    test_seed=None,
    rescale=True,
):
    """
    Load pose‑tracking records, (optionally) filter them, and (optionally) split
    them into *main*, *test*, and *dev* subsets.

    Parameters
    ----------
    data_format : {'movement', 'DLC', 'SLEAP'}
        Dataset format to load. Only ``'movement'`` is implemented at present.
    data_path : str or Path
        Root directory containing the sequence sub‑directories.
    data_filter : str, optional
        Comma‑separated substrings; a record is kept if **any** substring
        occurs in its relative path.  By default, all records are kept.
    dev_ratio : float, optional
        Fraction of the **post‑test** remainder to devote to the dev set.
        If ``None`` (default) no dev split is performed.
    test_ratio : float, optional
        Fraction of *all* records to devote to the test set.  If ``None``
        (default) no test split is performed unless the dataset already
        contains an explicit test split (not yet supported here).
    dev_seed, test_seed : int, optional
        Random seeds forwarded to :pyfunc:`sklearn.model_selection.train_test_split`
        for reproducibility of the dev and test splits, respectively.
    rescale : bool, optional
        If ``True`` (default), rescale the coordinates in the (0, 1) range.

    Returns
    -------
    dict[str, list[tuple[str, dict[str, xarray.Dataset]]]]
        A mapping whose keys are

        * ``'main_records'`` – always present
        * ``'test_records'`` – present only if *test_ratio* was supplied
        * ``'dev_records'`` – present only if *dev_ratio* was supplied

        Each value is a list of ``(record_id, data_dict)`` pairs; every
        ``data_dict`` currently contains a single entry ``'posetracks'`` holding
        an *xarray* dataset of shape `(individuals, frames, points, coords)`.

    Raises
    ------
    ValueError
        If *data_format* is unsupported.
    NotImplementedError
        For recognized but unimplemented formats.

    Notes
    -----
    When *dev_ratio* is supplied it applies **after** any test split; i.e.
    the dev set is carved out of what remains.

    Examples
    --------
    >>> groups = load_records(
    ...     data_format="movement",
    ...     data_path="~/datasets/mice",
    ...     test_ratio=0.2,
    ...     dev_ratio=0.1,
    ...     test_seed=0,
    ...     dev_seed=0,
    ... )
    >>> groups.keys()
    dict_keys(['main_records', 'test_records', 'dev_records'])

    """
    # Find all potential record paths
    seq_paths = [f for f in Path(data_path).rglob("*") if f.is_dir()]

    # Load and preprocess raw data
    all_records = []
    for seq_path in tqdm(seq_paths, desc="Loading dataset"):
        # Load pose-tracking data
        if (ds := _load_posetracks(seq_path, data_format, rescale)) is not None:
            rec_data = {"posetracks": ds}
        else:
            logging.debug("Skipping %s, no tracking data found", str(seq_path))
            continue

        # Load annotations
        if (annotations := _load_annotations(seq_path)) is not None:
            rec_data["annotations"] = annotations

        # Create record id
        rec_id = str(seq_path.relative_to(data_path))

        # Add record to the list
        all_records.append((rec_id, rec_data))

    # TODO: It would be important to run a few sanity checks on the data. For example,
    #       all record ids should be unique, and the data should be consistent across
    #       recordings (i.e., same individuals/space/keypoints in the same order).

    # Filter data, if requested
    if data_filter is not None:
        filters = data_filter.split(",")

        all_records = [
            rec for rec in all_records if any(flt in rec[0] for flt in filters)
        ]

        logging.info("Filtered dataset size =  %d videos", len(all_records))
        logging.debug([key for key, val in all_records])

    # Split dataset into train/test/dev sets
    groups = {}

    # Optional test‑set split
    if test_ratio is not None:
        all_records, groups["test_records"] = train_test_split(
            all_records, test_size=test_ratio, random_state=test_seed
        )

    # Optional dev‑set split (performed on what’s left after the test split)
    # NOTE: We assume that dev_ratio refers to the dataset size after the test split.
    if dev_ratio is not None:
        all_records, groups["dev_records"] = train_test_split(
            all_records, test_size=dev_ratio, random_state=dev_seed
        )

    # Remaining records become the “main” set
    # NOTE: We call this "main_records", rather than "train_records" to avoid
    #       confusion in inference mode. That is, "main_records" refers to all data #       during inference or an unsplit dataset during training.
    groups["main_records"] = all_records

    for group_id, group_data in groups.items():
        logging.info("Placing %d sequences in the %s group", len(group_data), group_id)
        logging.debug([key for key, val in group_data])

    return groups


def dump_records(data_path, records):
    """
    Dump a list of records to a file.

    Pose tracks are saved in a NetCDF format, and annotations are saved in CSV format.

    Parameters
    ----------
    data_path : str or Path
        Directory where the records will be saved.

    records : list of tuples
        List of records to be saved. Each record is a tuple containing

    """
    for key, data in tqdm(records, desc="Dumping records to disk"):
        rec_path = data_path / key
        rec_path.mkdir(parents=True, exist_ok=True)

        # Save posetracks
        data["posetracks"].to_netcdf(rec_path / "tracking.nc", engine="scipy")

        # Save annotations
        if "annotations" in data:
            data["annotations"].to_csv(rec_path / "annotations.csv", index=False)


def fetch_dataset(dataset_id, download_path):
    """
    Download and preprocess keypoints datasets from remote repositories.

    Downloads the specified dataset, processes raw data (e.g., keypoints, annotations),
    and stores them in a standardized format for analysis.

    Parameters
    ----------
    dataset_id : str
        Identifier for the dataset to fetch. Currently supported datasets:
        - "CalMS21_Task1": Mouse behavior classification dataset
        - "CalMS21_Unlabeled": Unlabeled mouse behavior videos
        - "SampleData": Sample dataset for testing
        Additional datasets may be supported in future versions.

    download_path : str
        Base directory path where the dataset will be stored. The function creates
        subdirectories for cache and processed data.

    Returns
    -------
    None
        Data is saved to disk in standardized format.

    Raises
    ------
    ValueError
        If dataset_id is not one of the supported options.

    Notes
    -----
    The function handles downloads with checksums and caching using pooch.
    Downloaded data is temporarily stored in a cache directory before being
    processed into the final standardized format.
    """
    if dataset_id == "CalMS21_Task1":
        # Get data from Caltech repo
        fnames = pooch.retrieve(
            url="https://data.caltech.edu/records/s0vdx-0k302/files/task1_classic_classification.zip?download=1",
            known_hash="md5:8a02654fddae28614ee24a6a082261b8",
            path=Path(download_path) / ".cache" / "lisbet",
            processor=pooch.Unzip(
                members=[
                    "task1_classic_classification/calms21_task1_train.json",
                    "task1_classic_classification/calms21_task1_test.json",
                ],
            ),
            progressbar=True,
        )

        # Preprocess keypoints
        rawdata_path = Path(fnames[0]).parents[1]
        train_records, test_records = calms21.load_taskx(rawdata_path, taskid=1)

        # Store data in LISBET-compatible format
        data_path = Path(download_path) / "CalMS21" / "task1_classic_classification"
        dump_records(data_path, train_records)
        dump_records(data_path, test_records)

    elif dataset_id == "CalMS21_Unlabeled":
        # Get data from Caltech repo
        fnames = pooch.retrieve(
            url="https://data.caltech.edu/records/s0vdx-0k302/files/unlabeled_videos.zip?download=1",
            known_hash="md5:35ab3acdeb231a3fe1536e38ad223b2e",
            path=Path(download_path) / ".cache" / "lisbet",
            processor=pooch.Unzip(
                members=[
                    "unlabeled_videos/calms21_unlabeled_videos_part1.json",
                    "unlabeled_videos/calms21_unlabeled_videos_part2.json",
                    "unlabeled_videos/calms21_unlabeled_videos_part3.json",
                    "unlabeled_videos/calms21_unlabeled_videos_part4.json",
                ],
            ),
            progressbar=True,
        )

        # Preprocess keypoints
        rawdata_path = Path(fnames[0]).parents[1]
        all_records = calms21.load_unlabeled(rawdata_path)

        # Store data in LISBET-compatible format
        data_path = Path(download_path) / "CalMS21" / "unlabeled_videos"
        dump_records(data_path, all_records)

    elif dataset_id == "SampleData":
        # Fetch data from HuggingFace repo
        # NOTE: This is a small sample dataset for testing purposes
        data_path = snapshot_download(
            repo_id="gchindemi/lisbet-examples",
            allow_patterns="sample_keypoints/",
            local_dir=download_path,
            repo_type="dataset",
        )

    else:
        raise ValueError(f"Unknown dataset {dataset_id}")
