"""
Iterable-style datasets for social behavior classification and self-supervised tasks.
"""

import numpy as np
import torch
import xarray as xr
from torch.utils.data import IterableDataset

from lisbet.datasets.common import AnnotatedWindowSelector, WindowSelector


class SocialBehaviorDataset(IterableDataset):
    """
    Iterable dataset for social behavior classification.

    Generates windows of pose data and corresponding labels for supervised
    classification of social behaviors. Each sample consists of a window of frames
    selected from a record, with the label extracted according to the specified
    annotation format. Supports binary, multiclass, and multilabel classification tasks.
    """

    def __init__(
        self,
        records,
        window_size,
        window_offset=0,
        fps_scaling=1.0,
        transform=None,
        annot_format="multiclass",
        base_seed=None,
    ):
        """
        Initialize the SocialBehaviorDataset.

        Parameters
        ----------
        records : list
            List of records containing the data.
        window_size : int
            Size of the window in frames.
        window_offset : int, optional
            Offset for the window in frames (default is 0).
        fps_scaling : float, optional
            Scaling factor for the frames per second (default is 1.0).
        transform : callable, optional
            A function/transform to apply to the data (default is None).
        annot_format : str, optional
            Format of the labels. Valid options are 'binary', 'multiclass', or
            'multilabel' for the respective classification tasks (default is
            'multiclass').
        base_seed : int, optional
            Base seed for random number generation (default is None, which generates a
            random seed).
        """
        super().__init__()

        self.window_selector = AnnotatedWindowSelector(
            records, window_size, window_offset, fps_scaling, annot_format
        )
        self.n_frames = self.window_selector.n_frames
        self.transform = transform

        self.base_seed = (
            base_seed
            if base_seed is not None
            else torch.randint(0, 2**31 - 1, (1,)).item()
        )

        # Set random generator for reproducibility
        # NOTE: This could be overridden by the worker_init_fn to ensure each worker
        #       has a different seed for data shuffling.
        self.g = torch.Generator().manual_seed(self.base_seed)

    def __iter__(self):
        while True:
            # Select a random window (global frame index)
            global_idx = torch.randint(0, self.n_frames, (1,), generator=self.g).item()

            # Map global index to (record_index, frame_index)
            rec_idx, frame_idx = self.window_selector.global_to_local(global_idx)

            # Extract corresponding window
            x, y = self.window_selector.select(rec_idx, frame_idx)

            if self.transform:
                x = self.transform(x)

            yield x, y


class GroupConsistencyDataset(IterableDataset):
    """
    Iterable dataset for the Group Consistency self-supervised task.

    Generates windows for training models to determine whether a group of tracked
    individuals in a window of frames originates from the same recording ("consistent")
    or is artificially constructed by combining individuals from different records
    ("inconsistent").

    Each sample consists of a window of frames, with 50% probability of being
    consistent and 50% probability of being inconsistent (via swapping individuals from
    another record).

    Notes
    -----
    1. The swap is performed by splitting the group of individuals at a random index,
       concatenating individuals from the original and swap windows. This allows for
       arbitrary group sizes and compositions.
    2. Padding may not be consistent for swapped windows, especially near the sequence
       boundaries. However, since the number of padded windows is small compared to the
       total, this edge case is not explicitly handled.
    3. This dataset requires that each record contains at least two individuals.
    4. The label is 0 for consistent (all individuals from the same record) and 1 for
       inconsistent (group contains individuals from different records).
    """

    def __init__(
        self,
        records,
        window_size,
        window_offset=0,
        fps_scaling=1.0,
        transform=None,
        base_seed=None,
    ):
        """
        Initialize the GroupConsistencyDataset.

        Parameters
        ----------
        records : list
            List of records containing the data.
        window_size : int
            Size of the window in frames.
        window_offset : int, optional
            Offset for the window in frames (default is 0).
        fps_scaling : float, optional
            Scaling factor for the frames per second (default is 1.0).
        transform : callable, optional
            A function/transform to apply to the data (default is None).
        base_seed : int, optional
            Base seed for random number generation (default is None, which generates a
            random seed).
        """
        super().__init__()

        self.window_selector = WindowSelector(
            records, window_size, window_offset, fps_scaling
        )
        self.n_frames = self.window_selector.n_frames
        self.transform = transform

        self.base_seed = (
            base_seed
            if base_seed is not None
            else torch.randint(0, 2**31 - 1, (1,)).item()
        )

        # Set random generator for reproducibility
        # NOTE: This could be overridden by the worker_init_fn to ensure each worker
        #       has a different seed for data shuffling.
        self.g = torch.Generator().manual_seed(self.base_seed)

    def __iter__(self):
        while True:
            # Select a random window (global frame index)
            global_idx = torch.randint(0, self.n_frames, (1,), generator=self.g).item()

            # Map global index to (record_index, frame_index)
            rec_idx, frame_idx = self.window_selector.global_to_local(global_idx)

            # Extract corresponding window
            x_orig = self.window_selector.select(rec_idx, frame_idx)

            if torch.rand((1,), generator=self.g).item() < 0.5:
                # Swap group, retry if a window from the same sequence was chosen
                while True:
                    global_idx_swap = torch.randint(
                        0, self.n_frames, (1,), generator=self.g
                    ).item()
                    rec_idx_swap, frame_idx_swap = self.window_selector.global_to_local(
                        global_idx_swap
                    )

                    if rec_idx_swap != rec_idx:
                        break

                # Extract swap window
                x_swap = self.window_selector.select(rec_idx_swap, frame_idx_swap)

                # Swap individuals splitting the group at a random index
                split_idx = torch.randint(
                    1, x_orig.coords["individuals"].size, (1,), generator=self.g
                ).item()
                x = xr.concat(
                    [
                        x_orig.isel(individuals=slice(0, split_idx)),
                        x_swap.isel(individuals=slice(split_idx, None)),
                    ],
                    dim="individuals",
                )

                y = np.array(1, ndmin=1, dtype=np.float32)

            else:
                # Don't swap
                rec_idx_swap, frame_idx_swap, split_idx = rec_idx, frame_idx, 0  # debug
                x = x_orig
                y = np.array(0, ndmin=1, dtype=np.float32)

            # Add debugging information
            x.attrs["orig_coords"] = [rec_idx, frame_idx]
            x.attrs["swap_coords"] = [rec_idx_swap, frame_idx_swap, split_idx]

            if self.transform:
                x = self.transform(x)

            yield x, y


class TemporalOrderDataset(IterableDataset):
    """
    Iterable dataset for the temporal order prediction self-supervised task.

    Generates samples for predicting whether a 'post' half-window follows a 'pre'
    half-window in the same recording (ordered) or not (unordered).

    Each sample consists of a window created by concatenating the first half of one
    window ('pre') and the second half of another window ('post'). Positive samples
    have the post window following the pre window in the same record; negative samples
    have the post window from a different record or from an earlier time in the same
    record, depending on the chosen 'method'.

    Notes
    -----
    1. Padding may differ between pre and post windows, especially near sequence
       boundaries. This is not explicitly handled, as the number of such cases is small
       relative to the dataset size.
    2. The last window_size frames of each record may produce overlapping pre and post
       windows in the positive ('ordered') case. These are included for simplicity and
       may help the model learn temporal relationships.
    3. The concatenation of pre and post windows along the time dimension is used for
       simplicity and compatibility with multi-task training. This approach encourages
       the model to learn both positional and sequence embeddings jointly. In the
       future, embeddings could be computed separately and concatenated before the
       classifier.
    4. In rare cases, pre and post windows may overlap perfectly and be labeled as both
       positive and negative, depending on random sampling. This ambiguity is tolerated
       for simplicity, as it is infrequent and unlikely to significantly impact model
       performance.
    5. The 'method' parameter controls how negative samples are selected: 'simple',
       post windows can come from any record or from earlier times in the same record;
       'strict', post windows are always from the same record but must precede the pre
       window in time.
    """

    def __init__(
        self,
        records,
        window_size,
        window_offset=0,
        fps_scaling=1.0,
        transform=None,
        method="strict",
        base_seed=None,
    ):
        """
        Initialize the TemporalOrderDataset.

        Parameters
        ----------
        records : list
            List of records containing the data.
        window_size : int
            Size of the window in frames.
        window_offset : int, optional
            Offset for the window in frames (default is 0).
        fps_scaling : float, optional
            Scaling factor for the frames per second (default is 1.0).
        transform : callable, optional
            A function/transform to apply to the data (default is None).
        method : str, optional
            Selection method for negative class examples. Options are 'simple'
            (post window can be from any record or earlier in the same record) and
            'strict' (post window is always from the same record but must precede the
            pre window).
        base_seed : int, optional
            Base seed for random number generation (default is None, which generates a
            random seed).
        """
        # Validate input parameters
        if method not in ("simple", "strict"):
            raise ValueError(
                f"Invalid method '{method}'. Choose either 'simple' or 'strict'."
            )

        super().__init__()

        self.window_selector = WindowSelector(
            records, window_size, window_offset, fps_scaling
        )
        self.n_frames = self.window_selector.n_frames
        self.transform = transform

        self.method = method

        self.base_seed = (
            base_seed
            if base_seed is not None
            else torch.randint(0, 2**31 - 1, (1,)).item()
        )

        # Set random generator for reproducibility
        # NOTE: This could be overridden by the worker_init_fn to ensure each worker
        #       has a different seed for data shuffling.
        self.g = torch.Generator().manual_seed(self.base_seed)

    def __iter__(self):
        while True:
            # Select a random window (global frame index)
            global_idx_pre = torch.randint(
                0, self.n_frames, (1,), generator=self.g
            ).item()

            # Map global index to (record_index, frame_index)
            rec_idx_pre, frame_idx_pre = self.window_selector.global_to_local(
                global_idx_pre
            )

            if torch.rand((1,), generator=self.g).item() < 0.5:
                # Positive sample: post window follows pre window in the same record
                # Allow zero-distance windows for every frame, including the last
                rec_idx_post = rec_idx_pre
                frame_idx_post = torch.randint(
                    frame_idx_pre,
                    self.window_selector.lengths[rec_idx_pre],
                    (1,),
                    generator=self.g,
                ).item()

                y = np.array(1, ndmin=1, dtype=np.float32)

            else:
                if self.method == "simple":
                    # Negative sample: post window from any record or earlier in same
                    # record
                    while True:
                        global_idx_post = torch.randint(
                            0, self.n_frames, (1,), generator=self.g
                        ).item()
                        rec_idx_post, frame_idx_post = (
                            self.window_selector.global_to_local(global_idx_post)
                        )

                        if (
                            rec_idx_post != rec_idx_pre
                            or frame_idx_post < frame_idx_pre
                        ):
                            # Valid negative: different record or earlier in same record
                            break

                elif self.method == "strict":
                    # Negative sample: post window from same record, but before pre
                    # window
                    rec_idx_post = rec_idx_pre
                    frame_idx_post = torch.randint(
                        0, frame_idx_pre + 1, (1,), generator=self.g
                    ).item()

                else:
                    raise ValueError(
                        f"Invalid method '{self.method}'. Choose either 'simple' or "
                        "'strict'."
                    )

                y = np.array(0, ndmin=1, dtype=np.float32)

            # Extract corresponding window
            x_pre = self.window_selector.select(rec_idx_pre, frame_idx_pre)

            # Extract next window
            x_post = self.window_selector.select(rec_idx_post, frame_idx_post)

            # Concatenate pre and post half-windows
            split_idx = np.ceil(self.window_selector.window_size / 2).astype(int)
            x = xr.concat(
                (
                    x_pre.isel(time=slice(0, split_idx)),
                    x_post.isel(time=slice(split_idx, None)),
                ),
                dim="time",
            )

            # Add debugging information
            x.attrs["pre_coords"] = [rec_idx_pre, frame_idx_pre]
            x.attrs["post_coords"] = [rec_idx_post, frame_idx_post]

            if self.transform:
                x = self.transform(x)

            yield x, y


class TemporalShiftDataset(IterableDataset):
    """
    Iterable dataset for the temporal shift prediction or regression task.

    Generates samples in which the trajectory of the second individual in a group is
    shifted in time by a random delay within a specified interval (default: -60 to +60
    frames).

    For each sample, a window of frames is selected from a record. The first
    individual's data is taken from this window, while the second individual's data is
    taken from a window at the same location but shifted by a random delay (positive or
    negative) within the allowed range. The two individuals' data are then concatenated
    along the 'individuals' dimension, forming a group window where one individual's
    trajectory is temporally shifted relative to the other.

    The task can be formulated as either: binary classification, predict whether the
    shift is positive (future) or negative (past); regression, estimate the normalized
    value of the temporal shift.

    Notes
    -----
    1. This dataset requires that each record contains at least two individuals.
    2. The shift is always performed within the same record; the shifted window is
       sampled such that it stays within the valid frame range.
    3. The split between individuals is randomized for each sample, so the shifted
       trajectory may correspond to any individual in the group except the first.
    4. The label is either: the normalized shift value in [0, 1], where 0 corresponds to
       neg. max_shift and 1 to pos. max shift (regression); or 1 if the shift is
       positive (delta_delay > 0), 0 otherwise (classification).
    5. Padding may occur if the shifted window extends beyond the sequence boundaries,
       but this is handled by the window extraction logic and is rare for typical
       settings.
    6. The window_offset parameter determines the temporal alignment of the window
       relative to the reference frame.
    """

    def __init__(
        self,
        records,
        window_size,
        window_offset=0,
        fps_scaling=1.0,
        transform=None,
        max_shift=60,
        regression=False,
        base_seed=None,
    ):
        """
        Initialize the TemporalShiftDataset.

        Parameters
        ----------
        records : list
            List of records containing the data.
        window_size : int
            Size of the window in frames.
        window_offset : int, optional
            Offset for the window in frames (default is 0).
        fps_scaling : float, optional
            Scaling factor for the frames per second (default is 1.0).
        transform : callable, optional
            A function/transform to apply to the data (default is None).
        max_shift : int, optional
            Maximum time shift to apply, expressed in number of frames (default is 60).
        regression : bool, optional
            Whether to perform regression (default is False, which performs binary
            classification).
        base_seed : int, optional
            Base seed for random number generation (default is None, which generates a
            random seed).
        """
        # Validate input parameters
        if max_shift <= 0:
            raise ValueError("LISBET requires max_shift to be a positive integer.")

        super().__init__()

        self.window_selector = WindowSelector(
            records, window_size, window_offset, fps_scaling
        )
        self.n_frames = self.window_selector.n_frames
        self.transform = transform

        self.min_delay = -max_shift
        self.max_delay = max_shift
        self.regression = regression

        self.base_seed = (
            base_seed
            if base_seed is not None
            else torch.randint(0, 2**31 - 1, (1,)).item()
        )

        # Set random generator for reproducibility
        # NOTE: This could be overridden by the worker_init_fn to ensure each worker
        #       has a different seed for data shuffling.
        self.g = torch.Generator().manual_seed(self.base_seed)

    def __iter__(self):
        while True:
            # Select a random window (global frame index)
            global_idx = torch.randint(0, self.n_frames, (1,), generator=self.g).item()

            # Map global index to (record_index, frame_index)
            rec_idx, frame_idx = self.window_selector.global_to_local(global_idx)

            # Extract corresponding window
            x_orig = self.window_selector.select(rec_idx, frame_idx)

            # Compute shift bounds
            lower_bound = max(frame_idx + self.min_delay, 0)
            upper_bound = min(
                frame_idx + self.max_delay, self.window_selector.lengths[rec_idx]
            )

            # Select random window from same sequence for the shift
            frame_idx_delay = torch.randint(
                lower_bound, upper_bound, (1,), generator=self.g
            ).item()

            # Get shift data
            x_shft = self.window_selector.select(rec_idx, frame_idx_delay)

            # Apply shifting by swapping individuals in the group at a random index
            split_idx = torch.randint(
                1, x_orig.coords["individuals"].size, (1,), generator=self.g
            ).item()
            x = xr.concat(
                [
                    x_orig.isel(individuals=slice(0, split_idx)),
                    x_shft.isel(individuals=slice(split_idx, None)),
                ],
                dim="individuals",
            )

            # Compute label
            delta_delay = frame_idx_delay - frame_idx

            if self.regression:
                # Set rescaled shift distance as label
                y = (delta_delay - self.min_delay) / (self.max_delay - self.min_delay)
                y = np.array(y, ndmin=1, dtype=np.float32)

            else:
                y = np.array(delta_delay > 0, ndmin=1, dtype=np.float32)

            # Add debugging information
            x.attrs["orig_coords"] = [rec_idx, frame_idx]
            x.attrs["shift_coords"] = [rec_idx, frame_idx_delay, delta_delay]

            if self.transform:
                x = self.transform(x)

            yield x, y


class TemporalWarpDataset(IterableDataset):
    """
    Iterable dataset for the temporal warp prediction or regression task.

    Generates windows in which the temporal pace (speed) of the window is artificially
    warped by resampling the frames at a random speed factor within a specified range
    (default: 50% to 150%).

    For each sample, a window is extracted from a random location in a record, and the
    time axis is rescaled by a randomly chosen speed factor. The resulting window is
    then interpolated back to the original window size, so the model always receives a
    fixed number of frames, but the underlying motion is either sped up or slowed down.

    The task can be formulated as either: binary classification, predict whether the
    window was sped up (speed > 100%) or slowed down (speed < 100%); regression,
    estimate the normalized speed factor used to warp the window.

    Notes
    -----
    1. The speed factor is sampled uniformly at random from [max_warp, 100 + max_warp]
       for each sample.
    2. The actual window is extracted by resampling the original frames at the chosen
       speed, then interpolated to the fixed window size.
    3. For regression, the label is the normalized speed factor in [0, 1], where 0
       corresponds to max_warp and 1 to 100 + max_warp.
    4. For classification, the label is 1 if the speed is above 100, and 0 otherwise.
    5. Padding may occur if the resampled window extends beyond the sequence
       boundaries, but this is handled by the window extraction logic and is rare for
       typical settings.
    6. The window_offset parameter determines the temporal alignment of the window
       relative to the reference frame.
    """

    def __init__(
        self,
        records,
        window_size,
        window_offset=0,
        fps_scaling=1.0,
        transform=None,
        max_warp=50.0,
        regression=False,
        base_seed=None,
    ):
        """
        Initialize the TemporalWarpDataset.

        Parameters
        ----------
        records : list
            List of records containing the data.
        window_size : int
            Size of the window in frames.
        window_offset : int, optional
            Offset for the window in frames (default is 0).
        fps_scaling : float, optional
            Scaling factor for the frames per second (default is 1.0).
        transform : callable, optional
            A function/transform to apply to the data (default is None).
        max_warp : float, optional
            Maximum time warp to apply, expressed as a percentage (default is 50).
        regression : bool, optional
            Whether to perform regression (default is False, which performs binary
            classification).
        base_seed : int, optional
            Base seed for random number generation (default is None, which generates a
            random seed).
        """
        # Validate input parameters
        if not (0 <= max_warp < 100):
            raise ValueError(
                "LISBET requires max_warp to be a positive value between 0 and 99."
            )

        super().__init__()

        self.window_selector = WindowSelector(
            records, window_size, window_offset, fps_scaling
        )
        self.n_frames = self.window_selector.n_frames
        self.transform = transform

        self.min_speed = 1 - max_warp / 100.0
        self.max_speed = 1.0 + max_warp / 100.0
        self.regression = regression

        self.base_seed = (
            base_seed
            if base_seed is not None
            else torch.randint(0, 2**31 - 1, (1,)).item()
        )

        # Set random generator for reproducibility
        # NOTE: This could be overridden by the worker_init_fn to ensure each worker
        #       has a different seed for data shuffling.
        self.g = torch.Generator().manual_seed(self.base_seed)

    def __iter__(self):
        while True:
            # Select a random window (global frame index)
            global_idx = torch.randint(0, self.n_frames, (1,), generator=self.g).item()

            # Map global index to (record_index, frame_index)
            rec_idx, frame_idx = self.window_selector.global_to_local(global_idx)

            # Draw playback speed at random
            rel_speed = torch.rand((1,), generator=self.g).item()
            speed = rel_speed * (self.max_speed - self.min_speed) + self.min_speed

            # Extract corresponding window, resampling at the specified speed
            x = self.window_selector.select(rec_idx, frame_idx, speed)

            if self.regression:
                # Set relative speed as label
                y = np.array(rel_speed, ndmin=1, dtype=np.float32)

            else:
                # Set speed threshold as label
                y = np.array(speed > 1, ndmin=1, dtype=np.float32)

            # Add debugging information
            x.attrs["orig_coords"] = [rec_idx, frame_idx]
            x.attrs["warp_coords"] = [rec_idx, frame_idx, speed]

            if self.transform:
                x = self.transform(x)

            yield x, y
