"""CalMS21 dataset."""

import json
import logging
import os
from pathlib import Path

import numpy as np
import xarray as xr
from movement.io import load_poses
from tqdm.auto import trange

from .core import Record


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
        record = Record(id=rec_id, posetracks=posetracks)

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
            record.annotations = xr.Dataset(
                data_vars=dict(
                    target_cls=(
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
        records.append(record)

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
    taskpath = Path(datapath) / dataset_names[taskid]

    # Load and preprocess train data
    logging.debug("Loading train data...")
    with open(taskpath / f"calms21_task{taskid}_train.json", encoding="utf-8") as f:
        train_data = json.load(f)

    # NOTE 1: We use a list to simplify splitting into train/dev/test sets
    # NOTE 2: The record name is sufficient to disambiguate conditions (i.e.
    #         annotator ID)
    train_records = [
        rec
        for annot_data in train_data.values()
        for rec in _preprocess_calms21(annot_data)
    ]

    # Load and preprocess test data
    logging.debug("Loading test data...")
    with open(taskpath / f"calms21_task{taskid}_test.json", encoding="utf-8") as f:
        test_data = json.load(f)

    test_records = [
        rec
        for annot_data in test_data.values()
        for rec in _preprocess_calms21(annot_data)
    ]

    logging.info("Train set size: %d videos", len(train_records))
    logging.debug("Train seq IDs: %s", ", ".join(str(rec.id) for rec in train_records))
    logging.info("Test set size: %d videos", len(test_records))
    logging.debug("Test seq IDs: %s", ", ".join(str(rec.id) for rec in test_records))

    return train_records, test_records
