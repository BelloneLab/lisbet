"""Core dataset functionalities."""

import logging
import re
from functools import partial
from pathlib import Path

import pooch
import xarray as xr
from huggingface_hub import snapshot_download
from movement.io import load_poses
from movement.transforms import scale
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from . import calms21, mabe22


def _load_posetracks(seq_path, data_format, data_scale, select_coords, rename_coords):
    """
    Load and preprocess a pose-tracking dataset from a sequence directory.

    Applies optional coordinate selection and renaming, and rescales coordinates
    to [0, 1] if requested.

    Parameters
    ----------
    seq_path : Path
        Path to the sequence directory.
    data_format : str
        Format of the dataset ('DLC', 'SLEAP', 'movement').
    data_scale : str or None
        Scaling string or None for auto-scaling.
    select_coords : str or None
        Optional subset string in the format 'INDIVIDUALS;AXES;KEYPOINTS', where each
        field is a comma-separated list or '*' for all.
    rename_coords : str or None
        Optional coordinate names remapping in the format 'INDIVIDUALS;AXES;KEYPOINTS',
        where each field is a comma-separated list of maps 'old_id:new_id' or '*' for
        no remapping at that level.

    Returns
    -------
    xarray.Dataset
        The loaded and preprocessed dataset.

    """
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

    # Drop confidence variable, if present
    # NOTE: This variable is currently not needed in LISBET, but it may become useful in
    #       the future, especially if we decide to provide a measure of tracking
    #       quality to the model.
    if "confidence" in ds:
        ds = ds.drop_vars("confidence")

    # Apply coordinates selection if requested
    # TODO: Move string parsing to CLI during refactoring and make 'sel_dict' an
    #       argument.
    if select_coords is not None:
        # Parse the subset string: 'INDIVIDUALS;AXES;KEYPOINTS'
        fields = select_coords.split(";")
        if len(fields) != 3:
            raise ValueError(
                "select_coords must have format 'INDIVIDUALS;AXES;KEYPOINTS', "
                "e.g. 'ind1,ind2;x,y;nose,neck,tail'"
            )
        # Use a compact dict comprehension for selection
        sel_keys = ["individuals", "space", "keypoints"]
        sel_dict = {
            key: [item.strip() for item in field.split(",") if item.strip()]
            for key, field in zip(sel_keys, fields)
            if field.strip() and field.strip() != "*"
        }
        if sel_dict:
            ds = ds.sel(**sel_dict)

            logging.debug("Subset selection: %s", sel_dict)

    # Apply coordinates renaming if requested
    # TODO: Move string parsing to CLI during refactoring and make 'remap_dict' an
    #       argument.
    if rename_coords is not None:
        # Parse the remapping string: 'INDIVIDUALS;AXES;KEYPOINTS'
        fields = rename_coords.split(";")
        if len(fields) != 3 or any(f.strip() == "" for f in fields):
            raise ValueError(
                "rename_coords must have format 'INDIVIDUALS;AXES;KEYPOINTS', "
                "using '*' for no remapping at a level, e.g. "
                "'mouse1:resident,mouse2:intruder;*;nose:snout,tail:tailbase'"
            )
        rename_keys = ["individuals", "space", "keypoints"]
        remap_dict = {}
        for key, field in zip(rename_keys, fields):
            if field.strip() != "*":
                mapping = {}
                for pair in field.split(","):
                    old, new = pair.split(":")
                    mapping[old.strip()] = new.strip()
                remap_dict[key] = (
                    key,
                    [mapping.get(val, val) for val in ds.coords[key].values],
                )
        if remap_dict:
            ds = ds.assign_coords(**remap_dict)

    # Rescale coordinates in the (0, 1) range
    if data_scale is not None:
        # Explicit scaling
        factor = [1 / float(val) for val in data_scale.split("x")]
        ds = ds.assign(position=scale(ds["position"], factor=factor))

        logging.debug("Rescaled coordinates by factor %s", factor)

    elif "image_size_px" in ds.attrs:
        # Rescale using image size
        factor = [1 / float(val) for val in ds.attrs["image_size_px"]]
        ds = ds.assign(position=scale(ds["position"], factor=factor))

        logging.debug(
            "Rescaled coordinates by image size %s", ds.attrs["image_size_px"]
        )

    else:
        # Auto-scaling
        reduce_dims = ("time", "keypoints", "individuals")

        pos = ds["position"]
        min_val = pos.min(dim=reduce_dims, skipna=True)
        max_val = pos.max(dim=reduce_dims, skipna=True)

        ds = ds.assign(position=(pos - min_val) / (max_val - min_val))

        # Validate scaling
        assert ds["position"].min() >= 0.0, "Coordinates should be in the [0, 1] range"
        assert ds["position"].max() <= 1.0, "Coordinates should be in the [0, 1] range"

        logging.debug(
            "Rescaled coordinates between min values %s and max values %s",
            min_val.values,
            max_val.values,
        )

    # After scaling, enforce [0, 1] range and raise if not satisfied
    min_val = ds["position"].min()
    max_val = ds["position"].max()
    if min_val < 0.0 or max_val > 1.0:
        raise ValueError(
            f"After applying data_scale={data_scale}, coordinates are not in [0, 1] "
            f"(min={min_val.values}, max={max_val.values}). Explicit scaling assumes "
            "that the video has already been cropped to the region of interest during "
            "pose estimation, its origin is at (0,0), and the maximum dimensions match "
            "the scale provided. If this is not the case, use auto mode "
            "(data_scale=None) for normalization."
        )

    # Stack variables into a single dimension
    # NOTE: This is done already here for performance reasons, as stacking in the
    #       `input_pipeline._select_and_pad` is very inefficient.
    ds = ds.stack(features=("individuals", "keypoints", "space"))

    # NOTE: We keep the whole Dataset object, rather than selecting the "position"
    #       variable, to allow for future extensions (e.g., adding more variables) and
    #       to keep the FPS information.
    return ds


def _load_annotations(seq_path):
    """
    Load annotations from a sequence directory, if present.

    Returns
    -------
    xarray.Dataset or None
        The loaded annotations, or None if not found.

    """
    # Find all files matching the annotations regex and load them
    rx = re.compile(r"(?i)(manual_scoring|annotations).*\.nc$")
    annotations = [
        xr.open_dataset(pth, engine="scipy")
        for pth in seq_path.iterdir()
        if rx.search(pth.name)
    ]

    # Check if any annotations were found
    if len(annotations) == 0:
        return None

    # Merge all annotations into a single one
    annotations = xr.concat(annotations, dim="annotators")

    logging.debug("Annotations: %s", annotations.coords["behaviors"].values)

    # Convert annotations to label encoding format
    # NOTE: This is a temporary solution to maintain compatibility with the current
    #       implementation of training and inference pipelines.
    # TODO: Add full support for one-hot and multi-label annotations
    annotations = annotations.assign(
        label_cat=annotations.label.argmax("behaviors").squeeze()
    )

    return annotations


def load_records(
    data_format,
    data_path,
    data_scale=None,
    data_filter=None,
    dev_ratio=None,
    test_ratio=None,
    dev_seed=None,
    test_seed=None,
    select_coords=None,
    rename_coords=None,
):
    """
    Load pose-tracking records from a directory, with optional filtering, coordinate
    selection, coordinate renaming, and train/dev/test splitting.

    Parameters
    ----------
    data_format : {'movement', 'DLC', 'SLEAP'}
        Dataset format to load.
    data_path : str or Path
        Root directory containing the sequence sub-directories.
    data_scale : str, optional
        If supplied as WIDTHxHEIGHT or WIDTHxHEIGHTxDEPTH, every input coordinate is
        assumed to be in data units and is divided by the given scale to obtain
        normalized coordinates in the range [0, 1]. Otherwise, the algorithm infers the
        active extent directly from the data.
    data_filter : str, optional
        Comma-separated substrings; a record is kept if any substring occurs in its
        relative path. By default, all records are kept.
    dev_ratio : float, optional
        Fraction of the post-test remainder to devote to the dev set.
        If None, no dev split is performed.
    test_ratio : float, optional
        Fraction of all records to devote to the test set. If None, no test split is
        performed.
    dev_seed : int, optional
        Random seeds for reproducibility of the dev split.
    test_seed : int, optional
        Random seeds for reproducibility of the test split.
    select_coords : str or None
        Optional subset string in the format 'INDIVIDUALS;AXES;KEYPOINTS', where each
        field is a comma-separated list or '*' for all. If None, all data is loaded.
        Example: 'mouse1,mouse2;x,y;nose,tail'.
    rename_coords : str or None
        Optional coordinate names remapping in the format 'INDIVIDUALS;AXES;KEYPOINTS',
        where each field is a comma-separated list of maps 'old_id:new_id' or '*' for
        no remapping at that level. If None, original dataset names are used.
        Example: 'mouse1:resident,mouse2:intruder;*;nose:snout,tail:tailbase'.

    Returns
    -------
    dict[str, list[tuple[str, dict[str, xarray.Dataset]]]]
        A mapping whose keys are: 'main_records', always present; 'test_records',
        present only if test_ratio was supplied; 'dev_records', present only if
        dev_ratio was supplied. Each value is a list of (record_id, data_dict) pairs;
        every data_dict contains at least 'posetracks' (xarray.Dataset), and optionally
        'annotations'.

    Raises
    ------
    ValueError
        If data_format is unsupported, or if select_coords/rename_coords are invalid.
    NotImplementedError
        For recognized but unimplemented formats.

    Notes
    -----
    When dev_ratio is supplied it applies after any test split; i.e. the dev set is
    carved out of what remains.

    Examples
    --------
    >>> groups = load_records(
    ...     data_format="movement",
    ...     data_path="~/datasets/mice",
    ...     test_ratio=0.2,
    ...     dev_ratio=0.1,
    ...     test_seed=0,
    ...     dev_seed=0,
    ...     select_coords="mouse1,mouse2;x,y;nose,tail",
    ...     rename_coords="mouse1:resident,mouse2:intruder;*;nose:snout,tail:tailbase",
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
        ds = _load_posetracks(
            seq_path, data_format, data_scale, select_coords, rename_coords
        )
        if ds is not None:
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

    # Sanity check: All posetracks must have the same 'features' coordinate (summary of
    #               individuals/keypoints/space)
    if all_records:
        ref_features = (
            all_records[0][1]["posetracks"].coords["features"].values.tolist()
        )
        for rec_id, rec_data in all_records:
            ds_features = rec_data["posetracks"].coords["features"].values.tolist()
            if ds_features != ref_features:
                raise ValueError(
                    f"Inconsistent posetracks coordinates in record '{rec_id}':\n"
                    f"Reference features:\n{ref_features}\n"
                    f"Record features:\n{ds_features}"
                )

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
    #       confusion in inference mode. That is, "main_records" refers to all data
    #       during inference or an unsplit dataset during training.
    groups["main_records"] = all_records

    for group_id, group_data in groups.items():
        logging.info("Placing %d sequences in the %s group", len(group_data), group_id)
        logging.debug([key for key, val in group_data])

    return groups


def dump_records(data_path, records):
    """
    Dump a list of records to a file.

    Pose tracks and annotations are saved in a NetCDF format.

    Parameters
    ----------
    data_path : str or Path
        Directory where the records will be saved.

    records : list of tuples
        List of records to be saved. Each record is a tuple containing

    """
    for key, data in tqdm(records, desc="Dumping records to disk"):
        rec_path = Path(data_path) / key
        rec_path.mkdir(parents=True, exist_ok=True)

        # Save posetracks
        data["posetracks"].to_netcdf(rec_path / "tracking.nc", engine="scipy")

        # Save annotations
        if "annotations" in data:
            data["annotations"].to_netcdf(rec_path / "annotations.nc", engine="scipy")


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
        all_records = calms21.load_unlabeled(rawdata_path)

        # Store data in LISBET-compatible format
        data_path = Path(download_path) / "datasets" / "CalMS21" / "unlabeled_videos"
        dump_records(data_path, all_records)

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
                "https://data.caltech.edu/records/rdsa8-rde65/files/"
                "mouse_triplet_test.npy?download=1"
            ),
            known_hash="md5:f43f0f8824ffe6a4496eaf3ba7559d5c",
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
