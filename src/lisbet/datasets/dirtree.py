"""A generic DLC dataset.
Wa assume each record is located in its own directory (i.e. the 'record directory'). In
the record directory must contain two files:
- The body pose data of the experimental mouse, named 'tracking_exp_2D_8KP.csv',
- The body pose data of the experimental mouse, named 'tracking_stim_2D_8KP.csv.
If annotations are available, an annotation file could be placed in the record directory
and must be named 'Manual_Scoring_annotator1.csv'.

"""

import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


def _find_tracking_files(dspath: Path):
    """
    Scan a directory for CSV files and return a filename based on matching rules.

        This function searches for CSV files in the specified directory and applies
        three matching strategies in order:
        1. Return if exactly one CSV file exists
        2. Return if exactly one file matches pattern1
        3. Return if exactly one file matches pattern2

        Parameters
        ----------
        directory_path : str
            Path to the directory to scan for CSV files.

        Returns
        -------
        Optional[str]
            The name of the matching file if found, None otherwise.

        Raises
        ------
        ValueError
            If multiple matching files are found where only one is expected,
            or if no matching files are found after trying all strategies.
    """
    # Get all CSV files
    csv_files = list(dspath.rglob("*.csv"))

    # Case 1: Exactly one CSV
    if len(csv_files) == 1:
        return csv_files[0]

    # Case 2: Try to look for DLC-like file names
    dlc_pattern = r".*?DLC.*?shuffle\d+.*?\.csv"
    dlc_matches = [f for f in csv_files if re.search(dlc_pattern, f.name)]
    if len(dlc_matches) == 1:
        return dlc_matches[0]
    if len(dlc_matches) > 1:
        raise ValueError(
            f"Multiple files match a DLC-like pattern '{dlc_pattern}': {dlc_matches}"
        )

    # Case 3: Try to look for file names containing the 'tracking' tag
    tag_matches = [f for f in csv_files if re.search(r"tracking", f.name)]
    if len(tag_matches) == 1:
        return tag_matches[0]
    if len(tag_matches) > 1:
        raise ValueError(
            f"Multiple files contain 'tracking' in their name: {tag_matches}"
        )

    raise ValueError(f"No tracking files found in {dspath}")


def _dlc2calms(dspath, rescale, ll_threshold=0, multi_animal=False):
    """Load and preprocess body pose records."""
    body_parts = ["nose", "earL", "earR", "neck", "hipsL", "hipsR", "tail"]
    coords = ["x", "y"]
    idx = pd.IndexSlice
    logging.debug(dspath)

    if multi_animal:
        # TODO: Remove
        body_parts = [
            "Nose",
            "Left ear",
            "Right ear",
            "Neck",
            "Left hip",
            "Right hip",
            "Tail",
        ]

        # Load tracking data
        trackingpath = _find_tracking_files(dspath)
        df = pd.read_csv(trackingpath, header=list(range(4)), index_col=0)

        # Ignore estimates with low likelihood
        for bp in body_parts:
            ignore = np.where(df.loc[:, idx[:, :, bp, "likelihood"]] < ll_threshold)[0]
            logging.debug(
                "Dropping %.1f %% of %s estimates",
                100 * ignore.shape[0] / df.shape[0],
                bp,
            )
            df.loc[ignore, idx[:, :, bp, coords]] = np.nan

        # Join and reorganize datasets
        df = (
            df.reindex(body_parts, axis="columns", level=2)
            .drop(columns="likelihood", level=3)
            .reindex(coords, axis="columns", level=3)
            .interpolate(limit_direction="both")
        )

        # Rescale coordinates in the (0, 1) range
        if rescale:
            # Rescale x
            x_min = df.loc[:, idx[:, :, :, "x"]].min(axis=None)
            x_max = df.loc[:, idx[:, :, :, "x"]].max(axis=None)
            df.loc[:, idx[:, :, :, "x"]] = (df.loc[:, idx[:, :, :, "x"]] - x_min) / (
                x_max - x_min
            )

            # Rescale y
            y_min = df.loc[:, idx[:, :, :, "y"]].min(axis=None)
            y_max = df.loc[:, idx[:, :, :, "y"]].max(axis=None)
            df.loc[:, idx[:, :, :, "y"]] = (df.loc[:, idx[:, :, :, "y"]] - y_min) / (
                y_max - y_min
            )

    else:
        # Load individual datasets
        dfs = {}
        for key in ["exp", "stim"]:
            dfs[key] = pd.read_csv(
                os.path.join(dspath, f"tracking_{key}_2D_8KP.csv"),
                header=list(range(4)),
                index_col=0,
            )

            # Ignore estimates with low likelihood
            for bp in body_parts:
                ignore = np.where(
                    dfs[key].loc[:, idx[:, :, bp, "likelihood"]] < ll_threshold
                )[0]
                logging.debug(
                    "Dropping %.1f %% of %s estimates for %s mouse",
                    100 * ignore.shape[0] / dfs[key].shape[0],
                    bp,
                    key,
                )
                dfs[key].loc[ignore, idx[:, :, bp, coords]] = np.nan

        # Join and reorganize datasets
        df = (
            pd.merge(
                dfs["exp"],
                dfs["stim"],
                left_index=True,
                right_index=True,
                validate="one_to_one",
            )
            .reindex(body_parts, axis="columns", level=2)
            .drop(columns="likelihood", level=3)
            .reindex(coords, axis="columns", level=3)
            .interpolate(limit_direction="both")
        )

        # Rescale coordinates in the (0, 1) range
        # NOTE: We use both datasets to compute the min/max for x and y, rather than the
        #       merged dataset, to improve the rescaling of short social sessions. That is,
        #       as we are only keeping frames where both mice are present, if the intruder
        #       is quickly removed from the cage there is little to no chance to explore the
        #       whole space, hence the rescaling could be incorrect.
        # NOTE: I think the remark above is not true anymore, we should revise this part.
        if rescale:
            # Rescale x
            x_min = min(
                tdf.loc[:, idx[:, :, :, "x"]].min().min() for tdf in dfs.values()
            )
            x_max = min(
                tdf.loc[:, idx[:, :, :, "x"]].max().max() for tdf in dfs.values()
            )
            df.loc[:, idx[:, :, :, "x"]] = (df.loc[:, idx[:, :, :, "x"]] - x_min) / (
                x_max - x_min
            )

            # Rescale y
            y_min = min(
                tdf.loc[:, idx[:, :, :, "y"]].min().min() for tdf in dfs.values()
            )
            y_max = min(
                tdf.loc[:, idx[:, :, :, "y"]].max().max() for tdf in dfs.values()
            )
            df.loc[:, idx[:, :, :, "y"]] = (df.loc[:, idx[:, :, :, "y"]] - y_min) / (
                y_max - y_min
            )

    # Check for missing frames
    times_gaps = np.diff(df.index)
    assert np.max(times_gaps) == 1

    # Output data
    keypoints = df.to_numpy()
    valid_indices = df.index

    return keypoints, valid_indices


def _read_annotations(dspath, kp_index):
    """Read human annotations from 'Manual_Scoring_annotator1.csv'."""
    expected_columns = [
        "approach",
        "attack",
        "copulation",
        "chase",
        "circle",
        "drink",
        "eat",
        "clean",
        "human",
        "sniff",
        "up",
        "walk_away",
        "other",
    ]

    # Find annotation file
    # ann_files = list(Path(dspath).glob("Manual_Scoring_annotator*.csv"))
    # assert len(ann_files) == 1
    # ann_file = ann_files[0]
    ann_file = Path(dspath) / "Manual_Scoring_annotator1.csv"

    # Extract annotator ID
    # pattern = r"Manual_Scoring_annotator(\d+)\.csv"
    # match = re.match(pattern, str(ann_file.name))
    # ann_id = int(match.group(1))
    ann_id = 1

    # Read annotations
    ann_df = pd.read_csv(ann_file, header=[0, 1])

    # Verify one-hot encoding
    assert np.all(ann_df.sum(axis=1) == 1)

    # Verify annotator id
    assert np.all(
        [annotator == f"annotator{ann_id}" for _, annotator in ann_df.columns]
    )

    # Verify columns order
    assert np.all(
        [
            c0 == c1
            for c0, c1 in zip(ann_df.columns.get_level_values(0), expected_columns)
        ]
    )

    # Drop level 1 columns (redundant, all corresponding to "annotator1")
    ann_df = ann_df.droplevel(1, axis=1)

    # Filter using keypoints dataframe index
    ann_df = ann_df.loc[kp_index]

    # Convert to categories
    categories = ann_df.values.argmax(axis=1)

    return categories


def load(datapath, test_ratio=None, rescale=True, seed=1923, multi_animal=False):
    """Load body pose records from a generic DLC dataset.

    Data is split into training and testing set according to the test_ratio. Records in
    the two sets are organized in a list of tuples (video_id, data), where data is a
    dictionary {"keypoints": np.array}. This format has been chosen to simplify further
    splitting of the training set (i.e. cross-validation) and to support annotated data
    as well.

    Parameters
    ----------
    datapath : string or pathlib.Path
        Root directory of the dataset.
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
    >>> rec_train, rec_test = load_calms21_unlabeled("datasets/CRIM13")

    """
    # Get list of all candidate paths
    seqpaths = [f for f in Path(datapath).glob("*/**") if f.is_dir()]

    # Load and preprocess raw data
    records = []
    for seqpath in tqdm(seqpaths, desc="Loading generic DLC dataset"):
        try:
            # Load and preprocess keypoints
            keypoints, valid_indices = _dlc2calms(
                seqpath, rescale, multi_animal=multi_animal
            )
        except (FileNotFoundError, ZeroDivisionError) as err:
            logging.debug("%s does not contain body pose data, skipping", seqpath)
            continue
        else:
            data = {"keypoints": keypoints}

        try:
            # Load annotations
            annotations = _read_annotations(seqpath, valid_indices)
        except FileNotFoundError:
            logging.debug("Missing annotations in %s", seqpath)
        else:
            data["annotations"] = annotations

        # Define video ID
        video_id = str(seqpath.relative_to(datapath))

        # Assemble record
        record = (video_id, data)

        records.append(record)

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
