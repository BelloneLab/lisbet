"""CalMS21 dataset."""

import json
import logging
import os

import numpy as np
import xarray as xr
from movement.io import load_poses
from tqdm.auto import trange


def _preprocess_calms21(raw_data):
    """Preprocess body pose in the CalMS21 records."""
    records = []
    for rec_id, val in raw_data.items():
        logging.debug("Processing %s data...", rec_id)

        # Invert coordinates and body parts dims
        posetracks = np.array(val["keypoints"]).transpose((0, 2, 3, 1))

        scores = np.array(val["scores"]).transpose((0, 2, 1))

        posetracks = load_poses.from_numpy(
            position_array=posetracks,
            confidence_array=scores,
            individual_names=["resident", "intruder"],
            keypoint_names=[
                "nose",
                "left_ear",
                "right_ear",
                "neck",
                "left_hip",
                "right_hip",
                "tail",
            ],
            fps=30,
            source_software="MARS",
        )
        posetracks.attrs["image_size_px"] = [1024, 570]

        # Create record data structure
        rec_data = {"posetracks": posetracks}

        # Load annotations
        # NOTE: For the moment we keep annotations as a separate field, but this could
        #       be added to the posetracks data structure in the future.
        if "annotations" in val:
            annotator_id = f"annotator{val['metadata']['annotator-id']}"

            # Build the column labels in *exact* column order
            behaviors = [
                behavior
                for behavior, _ in sorted(
                    val["metadata"]["vocab"].items(), key=lambda item: item[1]
                )
            ]

            # Convert annotations to one-hot encoding
            one_hot_annotations = np.eye(len(behaviors), dtype=int)[val["annotations"]]

            # Convert to xarray Dataset
            rec_data["annotations"] = xr.Dataset(
                data_vars=dict(
                    label=(
                        ["time", "behaviors", "annotators"],
                        one_hot_annotations[..., np.newaxis],
                    )
                ),
                coords=dict(
                    time=posetracks.time,
                    behaviors=behaviors,
                    annotators=[annotator_id],
                ),
                attrs=dict(
                    source_software=posetracks.source_software,
                    ds_type="annotations",
                    fps=posetracks.fps,
                    time_unit=posetracks.time_unit,
                ),
            )

        # Store preprocessed sequence
        records.append((rec_id, rec_data))

    return records


def load_unlabeled(datapath):
    """
    Load body pose records from the unlabeled videos in the CalMS21 dataset.

    Records are organized in a list of tuples (video_id, data), where data is a
    dictionary {"posetracks": xr.Dataset, "annotations": np.array}. This format has
    been chosen to simplify splitting the records into sets.

    Parameters
    ----------
    datapath : string or pathlib.Path
        Root directory of the CalMS21 dataset.

    Returns
    -------
    list : Records.

    Examples
    --------
    >>> records = load_calms21_unlabeled("datasets/CalMS21")

    """
    # Load and preprocess raw data
    records = []
    for part in trange(1, 5, desc="Loading CalMS21 unlabeled dataset"):
        logging.debug("Loading part%d data...", part)

        raw_path = os.path.join(
            datapath, "unlabeled_videos", f"calms21_unlabeled_videos_part{part}"
        )

        # Load from source
        with open(raw_path + ".json", encoding="utf-8") as f_json:
            raw_data = json.load(f_json)

        # Preprocess data
        # NOTE: We use a list to simplify splitting into train/dev/test sets
        records.extend(_preprocess_calms21(raw_data["unlabeled_videos"]))

    return records


def load_taskx(datapath, taskid):
    """Load body pose records from the task 1/2/3 videos in the CalMS21 dataset.

    Data is split into training and testing is prescribed in the dataset. Records in
    are organized in a list of tuples (video_id, data), where data is a dictionary
    {"posetracks": xr.Dataset, "annotations": np.array}. This format has been chosen to
    simplify splitting the records into sets.

    Parameters
    ----------
    datapath : string or pathlib.Path
        Root directory of the CalMS21 dataset.
    taskid : int
        Name of the dataset to load. Valid options are 1, 2 and 3.

    Returns
    -------
    list : Training set records.
    list : Test set records.

    Examples
    --------
    >>> rec_train, rec_test = load_taskx("datasets/CalMS21", taskid=1)

    """
    # Map task to dataset name
    dataset_names = {
        1: "task1_classic_classification",
        2: "task2_annotation_styles",
        3: "task3_new_behaviors",
    }

    # Validate arguments
    assert taskid in (1, 2, 3)

    # Load and preprocess raw data
    records = {}
    for key in ["train", "test"]:
        logging.debug("Loading %s data...", key)

        raw_path = os.path.join(
            datapath, dataset_names[taskid], f"calms21_task{taskid}_{key}"
        )

        # Load from source
        with open(raw_path + ".json", encoding="utf-8") as f_json:
            raw_data = json.load(f_json)

        # Preprocess data
        # NOTE 1: We use a list to simplify splitting into train/dev/test sets
        # NOTE 2: The record name is sufficient to disambiguate conditions (i.e.
        #         annotator ID)
        records[key] = [
            rec
            for cond_data in raw_data.values()
            for rec in _preprocess_calms21(cond_data)
        ]

    logging.info("Training set size =  %d videos", len(records["train"]))
    logging.debug([key for key, val in records["train"]])
    logging.info("Test set size = %d videos", len(records["test"]))
    logging.debug([key for key, val in records["test"]])

    return records["train"], records["test"]
