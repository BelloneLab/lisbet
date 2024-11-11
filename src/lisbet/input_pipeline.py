"""Input data managment."""

import logging
from abc import ABC
from itertools import product

import numpy as np
from scipy.interpolate import interp1d
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    def __init__(self, records, window_size, window_offset, fps_scaling, transform):
        # Store raw data
        self.records = dict(records)

        # Store other params
        self.window_size = window_size
        self.window_offset = window_offset
        self.fps_scaling = fps_scaling
        self.transform = transform

        # Create a catalog of all possible windows
        # NOTE: This is needed to allow shuffling
        self.window_catalog = [
            idx
            for key, data in records
            for idx in product([key], range(data["keypoints"].shape[0]))
        ]

    def __len__(self):
        return len(self.window_catalog)

    def _select_and_pad(self, curr_key, curr_loc, window_size=None):
        """Select a window from the catalog, applying padding if needed.

        The selected window is returned as a new numpy array to avoid unintentional
        changes to the records in the window dictionary (i.e. by the swap mouse
        prediction task).

        """
        if window_size is None:
            window_size = self.window_size

        x_data = self.records[curr_key]["keypoints"]
        seq_len = x_data.shape[0]

        # Compute actual window size
        act_window_size = int(np.rint(self.fps_scaling * window_size))
        act_window_offset = int(np.rint(self.fps_scaling * self.window_offset))
        # logging.debug(
        #     "Actual window size and offset: (%d, %d)",
        #     act_window_size,
        #     act_window_offset,
        # )

        # Calculate padding
        past_n = max(act_window_size - curr_loc - act_window_offset - 1, 0)
        future_n = max(curr_loc + act_window_offset + 1 - seq_len, 0)
        # logging.debug("Window padding: (%d, %d)", past_n, future_n)

        # Calculate data bounds
        start_idx = max(curr_loc - act_window_size + act_window_offset + 1, 0)
        stop_idx = min(curr_loc + act_window_offset + 1, seq_len)
        # logging.debug("Data bounds: (%d, %d, %d)", start_idx, curr_loc, stop_idx)

        # Pad data with zeros
        past_pad = np.zeros((past_n, *x_data.shape[1:]))
        future_pad = np.zeros((future_n, *x_data.shape[1:]))
        x_data = np.concatenate(
            [past_pad, x_data[start_idx:stop_idx], future_pad], axis=0
        )

        # assert (
        #     x_data.shape[0] == window_size
        # ), f"{seq_len}, {start_idx}, {curr_loc}, {stop_idx}, {past_n}, {future_n}"

        # Interpolate frames to get exactly window_size frames
        f1d = interp1d(np.linspace(0, 1, act_window_size), x_data, axis=0)
        x_data = f1d(np.linspace(0, 1, window_size))

        return x_data


class FrameClassificationDataset(BaseDataset):
    """Dataset generator for the frame classification task.

    In the frame classification task, each frame is assumed to belong to exactly one of
    n classes (behaviors) and the classifier has access to a window of frames in the
    past.

    Parameters
    ----------
    records : list[(video_id, data)]
        Body pose dataset.
    window_size : int
        Number of frames to consider (last frame is the classification target).
    window_offset : int
        Number of frames to shift to the right.
    num_classes : int, optional
        Number of behaviors in the dataset. If specified, records must be annotated.

    Returns
    -------
    torch.utils.data.Dataset : The windows dataset from the provided records.

    """

    def __init__(
        self,
        records,
        window_size,
        window_offset,
        fps_scaling=1.0,
        transform=None,
        num_classes=None,
    ):
        super().__init__(records, window_size, window_offset, fps_scaling, transform)

        # Store other params
        self.num_classes = num_classes

    def __getitem__(self, idx):
        # Select keypoints data
        curr_key, curr_loc = self.window_catalog[idx]
        x_data = self._select_and_pad(curr_key, curr_loc)

        if self.transform:
            x_data = self.transform(x_data)

        # Select annotation data, if requested, return x_data only otherwise
        if self.num_classes is not None:
            y_data = self.records[curr_key]["annotations"][curr_loc]

            return x_data, y_data

        return x_data


class SwapMousePredictionDataset(BaseDataset):
    """Dataset generator for the swap mouse prediction task.

    In the swap mouse prediction task, the tracking data of the second animal might be
    replaced with those from another record (i.e. swapping the mouse). The classifier
    has to predict whether the swap happened or not. To solve the task, the classifier
    has access to a window of frames in the past. Swapping happens on a per window basis
    (i.e. at each generation step a new record is chosen for the swap).

    Parameters
    ----------
    records : list[(video_id, data)]
        Body pose dataset.
    window_size : int
        Number of frames to consider.
    window_offset : int
        Number of frames to shift to the right.
    seed : int
        RNG seed for shuffling.

    Returns
    -------
    torch.utils.data.Dataset : The windows dataset from the provided records.

    Notes
    -----
    1) Padding is not necessarily consistent for swapped windows. However, considering
       the small number of padded windows compared to the total number of windows, we
       decided to leave this edge case unmanaged for the time being.

    """

    def __init__(
        self,
        records,
        window_size,
        window_offset,
        fps_scaling=1.0,
        transform=None,
        seed=None,
    ):
        super().__init__(records, window_size, window_offset, fps_scaling, transform)

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.main_indices = self.rng.permutation(len(self))
        self.resample_dataset()

    def resample_dataset(self):
        labels = []
        extras = []

        logging.info("Resampling twin windows for SMP")
        for curr_idx in self.main_indices:
            if self.rng.random() < 0.5:
                # Select keypoints data
                curr_key, _ = self.window_catalog[curr_idx]

                # Swap intruder mouse, retry if a window from the same sequence was
                # chosen
                while True:
                    swap_idx = self.rng.integers(len(self))
                    swap_key, _ = self.window_catalog[swap_idx]

                    if swap_key != curr_key:
                        break

                # Swap mice
                labels.append(1)
                extras.append(swap_idx)
            else:
                # Don't swap
                labels.append(0)
                extras.append(curr_idx)

        self.labels = labels
        self.extras = extras

    def __getitem__(self, idx):
        # Select label
        label = self.labels[idx]

        # Extract current index (shuffled)
        curr_idx = self.main_indices[idx]

        # Select keypoints data
        curr_key, curr_loc = self.window_catalog[curr_idx]
        curr_data = self._select_and_pad(curr_key, curr_loc)

        # Select positive or negative sample for the swap
        swap_idx = self.extras[idx]

        # Select keypoints data for swapping
        swap_key, swap_loc = self.window_catalog[swap_idx]
        swap_data = self._select_and_pad(swap_key, swap_loc)

        # Apply swapping
        half = curr_data.shape[1] // 2
        curr_data[..., half:] = swap_data[..., half:]

        if self.transform:
            curr_data = self.transform(curr_data)

        return curr_data, label


class NextWindowPredictionDataset(BaseDataset):
    """Dataset generator for the next window prediction task.

    In the next window prediction task, two windows are presented to the classifier. The
    goal of the task is to identify whether the second window comes from the same record
    of the first one or not. In the former case, the second window is randomly chosen to
    follow the first with a random delay (i.e. the first window "causes" the second). To
    solve the task, the classifier has access to a window of frames in the past.
    Swapping happens on a per window basis (i.e. at each generation step a new record is
    chosen for the swap).

    Parameters
    ----------
    records : list[(video_id, data)]
        Body pose dataset.
    window_size : int
        Number of frames to consider.
    window_offset : int
        Number of frames to shift to the right.
    seed : int
        RNG seed for shuffling.


    Returns
    -------
    torch.utils.data.Dataset : The windows dataset from the provided records.

    Notes
    -----
    1) Padding is not necessarily consistent for swapped windows (i.e. a padded window
       cannot follow an unpadded one if the padded window actually belongs to the same
       record. However, considering the small number of padded windows compared to the
       total number of windows, we decided to leave this edge case unmanaged for the
       time being.
    2) The last window_size windows of each record will have overlapping "second
       windows" in the "same record" case. We could have skipped these windows, but we
       decided to allow them as they are not many, compared to the total number of
       windows, and they could even be beneficial to learn cause-effect relationships.
    3) For simplicity we concatenate the current and next window. This choice helps
       managing multiple tasks during training, but it leads to learning a joint
       positional and sequence embedding. In the future we might decide to decouple
       them by adding a max_seq to the backbone model and concatenating x_data and
       next_data along a new axis.

    """

    def __init__(
        self,
        records,
        window_size,
        window_offset,
        fps_scaling=1.0,
        transform=None,
        seed=None,
    ):
        super().__init__(records, window_size, window_offset, fps_scaling, transform)

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.main_indices = self.rng.permutation(len(self))
        self.resample_dataset()

    def resample_dataset(self):
        labels = []
        extras = []

        logging.info("Resampling twin windows for NWP")
        for curr_idx in self.main_indices:
            # Select keypoints data
            curr_key, curr_loc = self.window_catalog[curr_idx]

            if self.rng.random() < 0.5:
                # Select a valid next window from same sequence (past current idx)
                curr_len = self.records[curr_key]["keypoints"].shape[0]
                next_idx = curr_idx + self.rng.integers(curr_len - curr_loc)

                # OBS: In order to generate a valid next window for every frame,
                #      including the last one of the sequence, we have to allow
                #      zero-distance windows
                assert (
                    self.window_catalog[next_idx][0] == curr_key
                    and self.window_catalog[next_idx][1] >= curr_loc
                )

                # Valid next window
                labels.append(1)
                extras.append(next_idx)
            else:
                # Select a random window, retry if a true next window was chosen
                while True:
                    next_idx = self.rng.integers(len(self))
                    next_key, next_loc = self.window_catalog[next_idx]

                    if next_key != curr_key or next_loc < curr_loc:
                        break

                # Random next window
                labels.append(0)
                extras.append(next_idx)

        self.labels = labels
        self.extras = extras

    def __getitem__(self, idx):
        # Select label
        label = self.labels[idx]

        curr_idx = self.main_indices[idx]

        # Select keypoints data
        curr_key, curr_loc = self.window_catalog[curr_idx]
        curr_data = self._select_and_pad(curr_key, curr_loc, self.window_size // 2)

        # Select positive or negative sample
        next_idx = self.extras[idx]

        # Select next window
        next_key, next_loc = self.window_catalog[next_idx]
        next_data = self._select_and_pad(next_key, next_loc, self.window_size // 2)

        # Concatenate windows
        curr_data = np.vstack((curr_data, next_data))

        if self.transform:
            curr_data = self.transform(curr_data)

        return curr_data, label


class DelayMousePredictionDataset(BaseDataset):
    """Dataset generator for the delay mouse prediction and regression tasks.

    In the delay mouse prediction/regression task, the intruder mouse trajectory is
    shifted in time by a random delay in the interval (-60, +60) frames. The goal of the
    prediction task is to assess whether the delay was negative or positive, while the
    goal of the regression task is to estimate by how many frames the trajectory was
    shifted. To solve the tasks, the network has access to a window of frames in the
    past.

    Parameters
    ----------
    records : list[(video_id, data)]
        Body pose dataset.
    window_size : int
        Number of frames to consider.
    seed : int
        RNG seed for shuffling.
    epoch : int
        Starting epoch, used together with seed to allow a reproducible random stream of
        windows if training is restarted from a specific epoch.
    min_delay : int, optional
        Lower bound for the delay.
    max_delay : int, optional
        Upper bound for the delay.
    regression : bool, optional
        Use regression label.

    Returns
    -------
    torch.utils.data.Dataset : The windows dataset from the provided records.

    Notes
    -----
    1) Padding is not necessarily consistent for delayed windows. However, considering
       the small number of padded windows compared to the total number of windows, we
       decided to leave this edge case unmanaged for the time being.
    2) The regression output (i.e. the delay prediction) is normalized in the (0, 1)
       range.

    """

    def __init__(
        self,
        records,
        window_size,
        window_offset,
        fps_scaling=1.0,
        transform=None,
        seed=None,
        min_delay=-60,
        max_delay=60,
        regression=False,
    ):
        super().__init__(records, window_size, window_offset, fps_scaling, transform)

        # Setup RNG
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Store other params
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.regression = regression

        self.main_indices = self.rng.permutation(len(self))
        self.resample_dataset()

    def resample_dataset(self):
        labels = []
        extras = []

        logging.info("Resampling twin windows for DMP")
        for curr_idx in self.main_indices:
            # Select keypoints data
            curr_key, curr_loc = self.window_catalog[curr_idx]
            curr_len = self.records[curr_key]["keypoints"].shape[0]

            # Compute shift bounds
            lower_bound = max(curr_loc + self.min_delay, 0)
            upper_bound = min(curr_loc + self.max_delay, curr_len)

            # Select random window from same sequence for the shift
            sft_loc = self.rng.integers(lower_bound, upper_bound)
            extras.append(sft_loc)

            # Compute shift
            delta_sft = sft_loc - curr_loc

            if self.regression:
                # Set rescaled shift distance as label
                label = (delta_sft - self.min_delay) / (self.max_delay - self.min_delay)
            else:
                label = int(delta_sft > 0)

            labels.append(label)

        self.labels = labels
        self.extras = extras

    def __getitem__(self, idx):
        # Select label
        label = self.labels[idx]

        curr_idx = self.main_indices[idx]

        # Select keypoints data
        curr_key, curr_loc = self.window_catalog[curr_idx]
        curr_data = self._select_and_pad(curr_key, curr_loc)

        # Select positive or negative sample for the shift
        sft_loc = self.extras[idx]

        # Get shift data
        sft_data = self._select_and_pad(curr_key, sft_loc)

        # Apply swapping
        half = curr_data.shape[1] // 2
        curr_data[..., half:] = sft_data[..., half:]

        if self.transform:
            curr_data = self.transform(curr_data)

        return curr_data, label


class VideoSpeedPredictionDataset(BaseDataset):
    """Dataset generator for the video speed prediction and regression tasks.

    In the video speed prediction/regression task, the pace of the window is multiplied
    by a random factor in the interval (0.5, 1.5). The goal of the prediction task is to
    assess whether the video speed was reduced or increased , while the  goal of the
    regression task is to estimate the pace factor. To solve the tasks, the network has
    access to a window of frames in the past.

    Parameters
    ----------
    records : list[(video_id, data)]
        Body pose dataset.
    window_size : int
        Number of frames to consider.
    seed : int
        RNG seed for shuffling.
    epoch : int
        Starting epoch, used together with seed to allow a reproducible random stream of
        windows if training is restarted from a specific epoch.
    min_speed : int, optional
        Lower bound for the speed.
    max_speed : int, optional
        Upper bound for the speed.
    regression : bool, optional
        Use regression label.

    Returns
    -------
    torch.utils.data.Dataset : The windows dataset from the provided records.

    """

    def __init__(
        self,
        records,
        window_size,
        window_offset,
        fps_scaling=1.0,
        transform=None,
        seed=None,
        min_speed=0.5,
        max_speed=1.5,
        regression=False,
    ):
        super().__init__(records, window_size, window_offset, fps_scaling, transform)

        # Setup RNG
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Store other params
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.regression = regression

        self.main_indices = self.rng.permutation(len(self))
        self.resample_dataset()

    def resample_dataset(self):
        logging.info("Resampling twin windows for VSP")

        # Draw playback speed at random
        speeds = self.rng.uniform(self.min_speed, self.max_speed, len(self))

        # Compute actual window size (i.e. the actual frames required to generate
        # the window)
        extras = np.round(speeds * self.window_size).astype(int)

        # Solve a regression or classification problem
        labels = (
            (speeds - self.min_speed) / (self.max_speed - self.min_speed)
            if self.regression
            else np.array(speeds > 1).astype(int)
        )

        self.labels = labels
        self.extras = extras

    def __getitem__(self, idx):
        # Select label
        label = self.labels[idx]

        curr_idx = self.main_indices[idx]

        # Select keypoints data
        curr_key, curr_loc = self.window_catalog[curr_idx]

        # Get actual window size
        act_window_size = self.extras[idx]

        # Get frames
        act_data = self._select_and_pad(curr_key, curr_loc, act_window_size)

        # Interpolate frames to get exactly window_size frames
        f1d = interp1d(np.linspace(0, 1, act_window_size), act_data, axis=0)
        curr_data = f1d(np.linspace(0, 1, self.window_size))

        if self.transform:
            curr_data = self.transform(curr_data)

        return curr_data, label
