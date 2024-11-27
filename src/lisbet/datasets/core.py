"""Core dataset functionalities."""

import logging
from pathlib import Path

import pooch
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split

from . import calms21, dirtree, h5archive


def load_records(
    data_format,
    data_path,
    data_filter=None,
    dev_ratio=None,
    test_ratio=None,
    dev_seed=None,
    test_seed=None,
):
    """
    Load records from a dataset and split them into train, test, and optionally dev
    sets.

    Parameters
    ----------
    data_format : str
        Format of the dataset to load. Currently supported datasets are "CalMS21_Task1",
        "CalMS21_Unlabeled", and "GenericDLC".
    data_path : str
        Path to the directory containing the dataset records.
    dev_ratio : float, optional
        Fraction of the training set to use as the dev set. If None (default), no dev
        set is created.
    test_ratio : float, optional
        Fraction of the dataset to use as the test set. If None (default), no test is
        created unless it already exists as an explicit test split in the dataset (i.e
        CalMS21_Task1 dataset).
    dev_seed : int, optional
        Seed for the random number generator used to split the training set into train
        and dev sets.
    test_seed : int, optional
        Seed for the random number generator used to split the dataset into train and
        test sets.

    Returns
    -------
    list : Training set records.
    list : Test set records.
    list : Dev set records.

    Raises
    ------
    ValueError
        If the dataset name is unknown or unsupported.

    """
    if data_format == "maDLC":
        train_rec, test_rec = dirtree.load(
            data_path, seed=test_seed, test_ratio=test_ratio, multi_animal=True
        )

    elif data_format == "saDLC":
        train_rec, test_rec = dirtree.load(
            data_path, seed=test_seed, test_ratio=test_ratio, multi_animal=False
        )

    elif data_format == "h5archive":
        train_rec, test_rec = h5archive.load(
            data_path, seed=test_seed, test_ratio=test_ratio
        )

    else:
        raise ValueError(f"Unknown dataset {data_format}")

    # Filter data, if requested
    if data_filter is not None:
        filters = data_filter.split(",")

        if train_rec is not None:
            train_rec = [
                rec for rec in train_rec if any(flt in rec[0] for flt in filters)
            ]
            logging.info("Filtered training set size =  %d videos", len(train_rec))
            logging.debug([key for key, val in train_rec])

        if test_rec is not None:
            test_rec = [
                rec for rec in test_rec if any(flt in rec[0] for flt in filters)
            ]
            logging.info("Filtered test set size =  %d videos", len(test_rec))
            logging.debug([key for key, val in test_rec])

    # Make devset if requested
    if dev_ratio is not None:
        train_rec, dev_rec = train_test_split(
            train_rec, test_size=dev_ratio, random_state=dev_seed
        )

        logging.info(
            "Holding out %f%% of the training set for HP tuning.", dev_ratio * 100
        )
        logging.debug(
            "Dev set size = %d videos, new Train set size = %d videos",
            len(dev_rec),
            len(train_rec),
        )
        logging.debug([key for key, val in dev_rec])
        logging.debug([key for key, val in train_rec])
    else:
        dev_rec = None

    return train_rec, test_rec, dev_rec


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
        data_path.mkdir(parents=True, exist_ok=True)
        h5archive.dump(data_path / "train_records.h5", train_records)
        h5archive.dump(data_path / "test_records.h5", test_records)

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
        all_records, _ = calms21.load_unlabeled(rawdata_path)

        # Store data in LISBET-compatible format
        data_path = Path(download_path) / "CalMS21" / "unlabeled_videos"
        data_path.mkdir(parents=True, exist_ok=True)
        h5archive.dump(data_path / "all_records.h5", all_records)

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
