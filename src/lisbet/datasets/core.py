"""Core dataset functionalities."""

import logging
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import pooch
import xarray as xr
from huggingface_hub import snapshot_download
from movement.io import load_poses
from movement.transforms import scale
from tqdm.auto import tqdm

from . import calms21, mabe22


@dataclass
class Record:
    """
    Data structure representing a single pose-tracking record.

    Parameters
    ----------
    id : str
        Unique identifier for the record, typically derived from the relative path.
    posetracks : xarray.Dataset
        Pose-tracking data for the record.
    annotations : xarray.Dataset or None, optional
        Annotations associated with the record, if available.
    """

    id: str
    posetracks: xr.Dataset
    annotations: Optional[xr.Dataset] = None


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

    return annotations


def load_records(
    data_format,
    data_path,
    data_scale=None,
    data_filter=None,
    select_coords=None,
    rename_coords=None,
):
    """
    Load pose-tracking records from a directory, with optional filtering, coordinate
    selection and renaming.

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
    list[Record]
        A list of Record objects, each containing id, posetracks, and optionally
        annotations.

    Raises
    ------
    ValueError
        If data_format is unsupported, or if select_coords/rename_coords are invalid.
    NotImplementedError
        For recognized but unimplemented formats.

    Examples
    --------
    >>> records = load_records(
    ...     data_format="movement",
    ...     data_path="~/datasets/mice",
    ...     select_coords="mouse1,mouse2;x,y;nose,tail",
    ...     rename_coords="mouse1:resident,mouse2:intruder;*;nose:snout,tail:tailbase",
    ... )
    >>> print(len(records))
    42
    >>> print(records[0].id)
    'session1/seq001'
    >>> print(records[0].posetracks)
    <xarray.Dataset ...>
    >>> print(records[0].annotations)
    <xarray.Dataset ...> or None

    """
    # Find all potential record paths
    seq_paths = [f for f in Path(data_path).rglob("*") if f.is_dir()]

    # Filter data, if requested
    if data_filter is not None:
        filters = data_filter.split(",")

        seq_paths = [
            seq_path
            for seq_path in seq_paths
            if any(flt in str(seq_path.relative_to(data_path)) for flt in filters)
        ]

        logging.info("%d potential paths after filtering", len(seq_paths))
        logging.debug(seq_paths)

    # Load and preprocess raw data
    records = []
    for seq_path in tqdm(seq_paths, desc="Loading dataset"):
        # Load pose-tracking data
        posetracks = _load_posetracks(
            seq_path, data_format, data_scale, select_coords, rename_coords
        )
        if posetracks is None:
            logging.debug("Skipping %s, no tracking data found", str(seq_path))
            continue

        # Load annotations
        annotations = _load_annotations(seq_path)

        # Create record id
        rec_id = str(seq_path.relative_to(data_path))

        # Add Record object to the list
        records.append(
            Record(id=rec_id, posetracks=posetracks, annotations=annotations)
        )

    # Sanity check: All posetracks must have the same 'features' coordinate (summary of
    #               individuals/keypoints/space)
    if records:
        ref_features = records[0].posetracks.coords["features"].values.tolist()
        for rec in records:
            ds_features = rec.posetracks.coords["features"].values.tolist()
            if ds_features != ref_features:
                raise ValueError(
                    f"Inconsistent posetracks coordinates in record '{rec.id}':\n"
                    f"Reference features:\n{ref_features}\n"
                    f"Record features:\n{ds_features}"
                )
    else:
        raise ValueError(
            "No valid records found in the specified directory. Please check the data "
            "path, format and filters to ensure they match the dataset structure.\n"
            f"Current values are: \n data_path = {data_path}\n "
            f"data_format = {data_format}\n data_filter = {data_filter}\n"
        )

    return records


def dump_records(data_path, records):
    """
    Dump a list of records to a file.

    Pose tracks and annotations are saved in a NetCDF format.

    Parameters
    ----------
    data_path : str or Path
        Directory where the records will be saved.

    records : list of Record
        List of Record objects to be saved.

    """
    for rec in tqdm(records, desc="Dumping records to disk"):
        rec_path = Path(data_path) / rec.id
        rec_path.mkdir(parents=True, exist_ok=True)

        # Save posetracks
        rec.posetracks.to_netcdf(rec_path / "tracking.nc", engine="scipy")

        # Save annotations
        if rec.annotations is not None:
            rec.annotations.to_netcdf(rec_path / "annotations.nc", engine="scipy")


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
