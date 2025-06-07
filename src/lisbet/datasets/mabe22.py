"""
MABe22 dataset.

References
----------
1. Sun, J. J. et al. MABe22: A Multi-Species Multi-Task Benchmark for Learned
   Representations of Behavior. Preprint at https://doi.org/10.48550/arXiv.2207.10553
   (2023).

2. Sun, J. et al. Dataset for MABe22: A Multi-Species Multi-Task Benchmark for Learned
   Representations of Behavior. CaltechDATA https://doi.org/10.22002/rdsa8-rde65 (2023).

3. AIcrowd | MABe 2022: Mouse-Triplets - Video Data | Challenges. AIcrowd | MABe 2022:
   Mouse-Triplets - Video Data | Challenges
   https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022/problems/mabe-2022-mouse-triplets-video-data.

4. AIcrowd | MABe 2022: Mouse Triplets | Challenges. AIcrowd | MABe 2022: Mouse
   Triplets | Challenges
   https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022/problems/mabe-2022-mouse-triplets.

"""

import logging

import numpy as np
import xarray as xr
from movement.io import load_poses
from tqdm.auto import tqdm

from .core import Record


def _preprocess_mabe22_sequence(raw_positions):
    """Preprocess a sequence in the MABe22 Mouse Triplets dataset."""
    # Extract dims for convenience
    n_frames, n_individuals, n_body_parts, n_space = raw_positions.shape

    # Invert coordinates and body parts dims
    # NOTE: The original data is in the format (frames, individuals, body parts, space),
    #       but we need to convert it to (frames, space, body parts, individuals) as
    #       required by the movement library.
    position_array = np.array(raw_positions).transpose((0, 3, 2, 1))

    # Missing confidence values
    confidence_array = np.full(
        (n_frames, n_body_parts, n_individuals), np.nan, dtype=float
    )

    # Convert to xarray Dataset
    posetracks = load_poses.from_numpy(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=[f"mouse_{i}" for i in range(n_individuals)],
        keypoint_names=[
            "nose",
            "left_ear",
            "right_ear",
            "neck",
            "left_forepaw",
            "right_forepaw",
            "center_back",
            "left_hindpaw",
            "right_hindpaw",
            "tail_base",
            "tail_middle",
            "tail_tip",
        ],
        fps=30,
        source_software="HRnetKumarLab",
    )
    posetracks.attrs["image_size_px"] = [850, 850]

    return posetracks


def _load_train(train_path):
    """Load the train data."""
    # Load raw train data
    train_raw = np.load(train_path, allow_pickle=True).item()

    # Annotations vocabulary
    behaviors = train_raw["vocabulary"]

    # Load and preprocess train sequences
    train_records = []
    for rec_id, rec_seq in tqdm(
        train_raw["sequences"].items(), desc="Processing train data"
    ):
        logging.debug("Processing %s data...", rec_id)

        # Keypoints
        posetracks = _preprocess_mabe22_sequence(rec_seq["keypoints"])

        # Annotations
        annotations = xr.Dataset(
            data_vars=dict(
                target_cls=(
                    ["time", "behaviors", "annotators"],
                    np.expand_dims(rec_seq["annotations"], axis=-1).transpose(1, 0, 2),
                )
            ),
            coords=dict(
                time=posetracks.time,
                behaviors=behaviors,
                annotators=["annotator0"],
            ),
            attrs=dict(
                source_software="VIA",
                ds_type="annotations",
                fps=posetracks.fps,
                time_unit=posetracks.time_unit,
            ),
        )

        # Create record data structure
        record = Record(id=rec_id, posetracks=posetracks, annotations=annotations)

        train_records.append(record)

    return train_records


def _load_test(test_seq_path, test_labels_path):
    """Load the test data."""
    # Load raw test data
    test_seq_raw = np.load(test_seq_path, allow_pickle=True).item()
    test_lab_raw = np.load(test_labels_path, allow_pickle=True).item()

    # Locate regression and classification labels
    # NOTE: There is a spelling mistake in the original dataset, where "Continuous" is
    #       misspelled as "Continious".
    cls_indices = [ttype == "Discrete" for ttype in test_lab_raw["task_type"]]
    reg_indices = [ttype == "Continious" for ttype in test_lab_raw["task_type"]]

    # Annotations vocabulary
    behaviors = np.array(test_lab_raw["vocabulary"])[cls_indices]
    quantities = np.array(test_lab_raw["vocabulary"])[reg_indices]

    logging.debug("Test vocabulary for classification: %s", behaviors)
    logging.debug("Test vocabulary for regression: %s", quantities)

    # Load and preprocess test sequences
    test_records = []
    for rec_id, rec_seq in tqdm(
        test_seq_raw["sequences"].items(), desc="Processing test data"
    ):
        logging.debug("Processing %s data...", rec_id)

        # Keypoints
        posetracks = _preprocess_mabe22_sequence(rec_seq["keypoints"])

        # Select and split annotations
        start_idx, stop_idx = test_lab_raw["frame_number_map"][rec_id]
        raw_annot_cls = test_lab_raw["label_array"][cls_indices, start_idx:stop_idx]
        raw_annot_reg = test_lab_raw["label_array"][reg_indices, start_idx:stop_idx]

        # Create and merge Datasets
        annot_cls = xr.Dataset(
            data_vars=dict(
                target_cls=(
                    ["time", "behaviors", "annotators"],
                    np.expand_dims(raw_annot_cls, axis=-1).transpose(1, 0, 2),
                )
            ),
            coords=dict(
                time=posetracks.time,
                behaviors=behaviors,
                annotators=["annotator0"],
            ),
            attrs=dict(
                source_software="VIA",
                ds_type="annotations",
                fps=posetracks.fps,
                time_unit=posetracks.time_unit,
            ),
        )

        annot_reg = xr.Dataset(
            data_vars=dict(
                target_reg=(
                    ["time", "quantities", "annotators"],
                    np.expand_dims(raw_annot_reg, axis=-1).transpose(1, 0, 2),
                )
            ),
            coords=dict(
                time=posetracks.time,
                quantities=quantities,
                annotators=["annotator0"],
            ),
            attrs=dict(
                source_software="VIA",
                ds_type="annotations",
                fps=posetracks.fps,
                time_unit=posetracks.time_unit,
            ),
        )

        annotations = xr.merge([annot_cls, annot_reg])

        # Create record data structure
        record = Record(id=rec_id, posetracks=posetracks, annotations=annotations)

        test_records.append(record)

    return test_records


def load_mouse_triplets(train_path, test_seq_path, test_labels_path):
    """
    Load the MABe22 Mouse Triplets dataset.

    Parameters
    ----------
    train_path : str
        Path to the training data file (numpy .npz format).
    test_seq_path : str
        Path to the test sequences file (numpy .npz format).
    test_labels_path : str
        Path to the test labels file (numpy .npz format).

    Returns
    -------
    list
        List of tuples (record_id, record_data) for training data.

    """
    # Load and process train records
    train_records = _load_train(train_path)

    # Load and process test records
    test_records = _load_test(test_seq_path, test_labels_path)

    return train_records, test_records
