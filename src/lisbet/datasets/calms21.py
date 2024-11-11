"""CalMS21 dataset."""

import json
import logging
import os

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import trange


def preprocess_calms21(raw_data, rescale):
    """Preprocess body pose in the CalMS21 records.

    Body pose data is reshaped into a numpy array of dims = (n frames, 28), after
    inverting the body parts and coordinates dimensions. Optionally, the keypoints are
    scaled in the (0, 1) range. Annotations data are converted into a numpy array and
    copyed as is in the returned records. Records in are organized in a list of tuples
    (video_id, data), where data is a dictionary {"keypoints": np.array, "annotations":
    np.array}. This format has been chosen to simplify splitting the records into sets.
    as well.

    Parameters
    ----------
    raw_data : dictionary
        Records as read from the the CalMS21 dataset files.
    rescale : bool
        Rescale body pose data in the (0, 1) range.

    Returns
    -------
    list : Preprocessed records.

    """
    # Reshape data
    records = []
    for key, val in raw_data.items():
        logging.debug("Reshaping %s data...", key)

        # Invert coordinates and body parts dims
        keypoints = np.array(val["keypoints"])
        keypoints = keypoints.transpose((0, 1, 3, 2))
        nframes = keypoints.shape[0]

        # Rescale in (0, 1) range if requested
        if rescale:
            keypoints[..., 0] = keypoints[..., 0] / 1024
            keypoints[..., 1] = keypoints[..., 1] / 570

        # Flatten all dims, except number of frames
        keypoints = keypoints.reshape(nframes, 28)

        # Store preprocessed sequence
        # NOTE: We use a list to simplify splitting into train/dev/test sets
        if "annotations" in val:
            records.append(
                (
                    key,
                    {
                        "keypoints": keypoints,
                        "annotations": np.array(val["annotations"]),
                    },
                )
            )
        else:
            records.append((key, {"keypoints": keypoints}))

    return records


def load_unlabeled(datapath, test_ratio=None, rescale=True, seed=None):
    """Load body pose records from the unlabeled videos in the CalMS21 dataset.

    Data is split into training and testing set according to the test_ratio. Records in
    the two sets are organized in a list of tuples (video_id, data), where data is a
    dictionary {"keypoints": np.array}. This format has been chosen to simplify further
    splitting of the training set (i.e. cross-validation) and to support annotated data
    as well.

    Parameters
    ----------
    datapath : string or pathlib.Path
        Root directory of the CalMS21 dataset.
    test_ratio : float, optional
        Fraction of the data to be used to create the test set.
    rescale : bool, optional
        Rescale body pose data in the (0, 1) range.
    seed : int, optional
        Seed of the random number generator, used to split the records.

    Returns
    -------
    list : Training set records.
    list : Test set records.

    Examples
    --------
    >>> rec_train, rec_test = load_calms21_unlabeled("datasets/CalMS21")

    """
    # Load and preprocess raw data
    records = []
    for part in trange(1, 5, desc="Loading CalMS21 unlabeled dataset"):
        logging.debug("Loading part%d data...", part)

        raw_path = os.path.join(
            datapath, "unlabeled_videos", f"calms21_unlabeled_videos_part{part}"
        )

        # Load from source
        with open(raw_path + ".json", "r", encoding="utf-8") as f_json:
            raw_data = json.load(f_json)

        # Preprocess data
        # NOTE: We use a list to simplify splitting into train/dev/test sets
        records.extend(preprocess_calms21(raw_data["unlabeled_videos"], rescale))

    if test_ratio is not None:
        # Split safety check
        assert test_ratio < 1

        # Split sets randomly
        rec_train, rec_test = train_test_split(
            records, test_size=test_ratio, random_state=seed
        )

        logging.info("Test set size = %d videos", len(rec_test))
        logging.debug([key for key, val in rec_test])
    else:
        rec_train = records
        rec_test = None

    logging.info("Training set size =  %d videos", len(rec_train))
    logging.debug([key for key, val in rec_train])

    return rec_train, rec_test


def load_taskx(datapath, taskid, rescale=True):
    """Load body pose records from the task x videos in the CalMS21 dataset.

    Data is split into training and testing is prescribed in the dataset. Records in
    the two sets are organized in a list of tuples (video_id, data), where data is a
    dictionary {"keypoints": np.array}. This format has been chosen to simplify further
    splitting of the training set (i.e. cross-validation) and to support annotated data
    as well.

    Parameters
    ----------
    datapath : string or pathlib.Path
        Root directory of the CalMS21 dataset.
    taskid : int
        Name of the dataset to load. Valid options are 1, 2 and 3.
    rescale : bool, optional
        Rescale body pose data in the (0, 1) range.

    Returns
    -------
    list : Training set records.
    list : Test set records.

    Examples
    --------
    >>> rec_train, rec_test = load_calms21_task1("datasets/CalMS21")

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
        with open(raw_path + ".json", "r", encoding="utf-8") as f_json:
            raw_data = json.load(f_json)

        # Preprocess data
        # NOTE 1: We use a list to simplify splitting into train/dev/test sets
        # NOTE 2: The record name is sufficient to disambiguate conditions (i.e.
        #         annotator ID)
        records[key] = [
            rec
            for cond_data in raw_data.values()
            for rec in preprocess_calms21(cond_data, rescale)
        ]

    logging.info("Training set size =  %d videos", len(records["train"]))
    logging.debug([key for key, val in records["train"]])
    logging.info("Test set size = %d videos", len(records["test"]))
    logging.debug([key for key, val in records["test"]])

    return records["train"], records["test"]
