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
from movement.io import load_poses
from tqdm.auto import tqdm


def _preprocess_mabe22_sequence(positions):
    """Preprocess a sequence in the MABe22 Mouse Triplets datset."""
    # Extract dims for convenience
    n_frames, n_individuals, n_body_parts, n_space = positions.shape

    # Invert coordinates and body parts dims
    # NOTE: The original data is in the format (frames, individuals, body parts, space),
    #       but we need to convert it to (frames, space, body parts, individuals) as
    #       required by the movement library.
    position_array = np.array(positions).transpose((0, 3, 2, 1))

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
    # train_vocab = train_raw["vocabulary"]

    # Load and preprocess train sequences
    train_records = []
    for rec_id, rec_seq in tqdm(
        train_raw["sequences"].items(), desc="Processing train data"
    ):
        logging.debug("Processing %s data...", rec_id)

        # Keypoints
        posetracks = _preprocess_mabe22_sequence(rec_seq["keypoints"])

        # Annotations
        # TODO

        # Create record data structure
        rec_data = {"posetracks": posetracks}

        train_records.append((rec_id, rec_data))

    return train_records


def _load_test(test_seq_path, test_labels_path):
    """Load the test data."""
    # Load raw test data
    test_raw = np.load(test_seq_path, allow_pickle=True).item()

    # Annotations vocabulary
    # test_vocab = test_labels_path["vocabulary"]

    # Load and preprocess test sequences
    test_records = []
    for rec_id, rec_seq in tqdm(
        test_raw["sequences"].items(), desc="Processing test data"
    ):
        logging.debug("Processing %s data...", rec_id)

        # Keypoints
        posetracks = _preprocess_mabe22_sequence(rec_seq["keypoints"])

        # Annotations
        # TODO

        # Create record data structure
        rec_data = {"posetracks": posetracks}

        test_records.append((rec_id, rec_data))

    return test_records


def load_mouse_triplets(train_path, test_seq_path, test_labels_path):
    """"""
    # Load and process train records
    train_records = _load_train(train_path)

    # Load and process test records
    test_records = _load_test(test_seq_path, test_labels_path)

    return train_records, test_records
