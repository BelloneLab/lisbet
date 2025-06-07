from pathlib import Path

import pooch
from huggingface_hub import snapshot_download

from lisbet.datasets import calms21, dump_records, mabe22


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
            url=(
                "https://data.caltech.edu/records/s0vdx-0k302/files/"
                "task1_classic_classification.zip?download=1"
            ),
            known_hash="md5:8a02654fddae28614ee24a6a082261b8",
            path=Path(download_path) / "datasets" / ".cache" / "lisbet",
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
        data_path = (
            Path(download_path)
            / "datasets"
            / "CalMS21"
            / "task1_classic_classification"
        )
        dump_records(data_path, train_records)
        dump_records(data_path, test_records)

    elif dataset_id == "CalMS21_Unlabeled":
        # Get data from Caltech repo
        fnames = pooch.retrieve(
            url=(
                "https://data.caltech.edu/records/s0vdx-0k302/files/"
                "unlabeled_videos.zip?download=1"
            ),
            known_hash="md5:35ab3acdeb231a3fe1536e38ad223b2e",
            path=Path(download_path) / "datasets" / ".cache" / "lisbet",
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
        records = calms21.load_unlabeled(rawdata_path)

        # Store data in LISBET-compatible format
        data_path = Path(download_path) / "datasets" / "CalMS21" / "unlabeled_videos"
        dump_records(data_path, records)

    elif dataset_id == "MABe22_MouseTriplets":
        # Get data from Caltech repo
        train_path = pooch.retrieve(
            url=(
                "https://data.caltech.edu/records/rdsa8-rde65/files/"
                "mouse_triplet_train.npy?download=1"
            ),
            known_hash="md5:76a48f3a1679a219a0e7e8a87871cc74",
            path=Path(download_path) / "datasets" / ".cache" / "lisbet",
            progressbar=True,
        )
        test_seq_path = pooch.retrieve(
            url=(
                # TMP, bug in default Caltech repo
                "https://data.caltech.edu/records/8kdn3-95j37/files/"
                "mouse_triplet_test.npy?download=1"
            ),
            known_hash="md5:20dc132300118a64aac665dd68153b20",
            path=Path(download_path) / "datasets" / ".cache" / "lisbet",
            progressbar=True,
        )
        test_labels_path = pooch.retrieve(
            url=(
                "https://data.caltech.edu/records/rdsa8-rde65/files/"
                "mouse_triplets_test_labels.npy?download=1"
            ),
            known_hash="md5:5a54f2d29a13a256aabbefc61a633176",
            path=Path(download_path) / "datasets" / ".cache" / "lisbet",
            progressbar=True,
        )

        # Preprocess keypoints
        train_records, test_records = mabe22.load_mouse_triplets(
            train_path, test_seq_path, test_labels_path
        )

        # Store records in LISBET-compatible format
        data_path = Path(download_path) / "datasets" / "MABe22" / "mouse_triplets"
        dump_records(data_path / "train", train_records)
        dump_records(data_path / "test", test_records)

    elif dataset_id == "SampleData":
        # Fetch data from HuggingFace repo
        # NOTE: This is a small sample dataset for testing purposes
        data_path = snapshot_download(
            repo_id="gchindemi/lisbet-examples",
            allow_patterns="sample_keypoints/",
            local_dir=Path(download_path) / "datasets",
            repo_type="dataset",
        )

    else:
        raise ValueError(f"Unknown dataset {dataset_id}")


def fetch_model(model_id, download_path=Path(".")):
    """Fetch a model from the HF Hub."""
    valid_model_ids = [
        "lisbet32x4-calms21UftT1-classifier",
        "lisbet32x4-calms21U-embedder",
    ]
    assert model_id in valid_model_ids, (
        f"Model ID '{model_id}' not found in the list of available models."
    )

    model_path = download_path / model_id
    snapshot_download(
        repo_id=f"gchindemi/{model_id}", repo_type="model", local_dir=model_path
    )
