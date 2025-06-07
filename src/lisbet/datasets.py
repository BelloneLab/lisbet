"""Input data management."""

import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import IterableDataset


class WindowDataset(IterableDataset):
    """Base class for datasets that generate windows of frames from records."""

    def __init__(
        self,
        records,
        window_size,
        window_offset=0,
        fps_scaling=1.0,
        transform=None,
    ):
        """
        Initialize the dataset.

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

        Returns
        -------
        torch.utils.data.IterableDataset
            The windows dataset from the provided records.
        """
        super().__init__()

        self.records = records
        self.n_records = len(records)

        self.window_size = window_size
        self.window_offset = window_offset
        self.fps_scaling = fps_scaling
        self.transform = transform

        self.lengths = np.array(
            [rec.posetracks.sizes["time"] for rec in self.records], dtype=int
        )
        self.cumlens = self.lengths.cumsum()
        self.n_frames = self.cumlens[-1]

    def __iter__(self):
        for global_idx in range(self.n_frames):
            # Map global index to (record_index, frame_index)
            rec_idx, frame_idx = self._global_to_local(global_idx)

            # Extract corresponding window
            x = self._select_and_pad(rec_idx, frame_idx, window_size=self.window_size)

            if self.transform:
                x = self.transform(x)

            y = (
                self.records[rec_idx]
                .annotations.target_cls.isel(time=frame_idx)
                .argmax("behaviors")
                .squeeze()
                .values
                if hasattr(self.records[rec_idx], "annotations")
                else torch.nan
            )

            yield x, y

    def _global_to_local(self, global_idx):
        """
        Map a global frame index (0 ≤ global_idx < total_n_frames) to a local pair
        (record_index, local_frame_index).
        """
        rec_idx = np.searchsorted(self.cumlens, global_idx, "right")
        prev_sum = 0 if rec_idx == 0 else self.cumlens[rec_idx - 1]
        local_idx = global_idx - prev_sum

        return rec_idx, local_idx

    def _select_and_pad(self, curr_key, curr_loc, window_size=None):
        """Select a window from the catalog, applying padding if needed.

        The selected window is returned as a new numpy array to avoid unintentional
        changes to the records in the window dictionary (i.e. by the swap mouse
        prediction task).

        Notes
        -----
        1. The interpolation is done here, and not directly on the records, to avoid
           resampling at the original fps before retuning the output during inference.
           Furthermore, even during training, it is useful to only consider the original
           frames, rather than artificially inflating or deflating the dataset.
        """
        if window_size is None:
            window_size = self.window_size

        x_data = self.records[curr_key]["posetracks"]
        seq_len = x_data.sizes["time"]

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
        past_pad = np.zeros((past_n, x_data.sizes["features"]))
        future_pad = np.zeros((future_n, x_data.sizes["features"]))
        x_data = np.concatenate(
            [
                past_pad,
                x_data["position"].isel(time=slice(start_idx, stop_idx)),
                future_pad,
            ],
            axis=0,
        )

        # assert (
        #     x_data.shape[0] == window_size
        # ), f"{seq_len}, {start_idx}, {curr_loc}, {stop_idx}, {past_n}, {future_n}"

        # Interpolate frames to get exactly window_size frames
        f1d = interp1d(np.linspace(0, 1, act_window_size), x_data, axis=0)
        x_data = f1d(np.linspace(0, 1, window_size))

        return x_data


class RandomWindowDataset(WindowDataset):
    """Base class for datasets that generate random windows of frames from records."""

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
        Initialize the dataset.

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

        Returns
        -------
        torch.utils.data.IterableDataset
            The windows dataset from the provided records.
        """
        super().__init__(records, window_size, window_offset, fps_scaling, transform)

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
            rec_idx, frame_idx = self._global_to_local(global_idx)

            # Extract corresponding window
            x = self._select_and_pad(rec_idx, frame_idx)

            if self.transform:
                x = self.transform(x)

            # Select annotation data
            y = (
                self.records[rec_idx]
                .annotations.target_cls.isel(time=frame_idx)
                .argmax("behaviors")
                .squeeze()
                .values
            )

            yield x, y


class SMPDataset(RandomWindowDataset):
    """
    Dataset generator for the swap mouse prediction task.

    In the swap mouse prediction task, the tracking data of the second animal might be
    replaced with those from another record (i.e. swapping the mouse). The classifier
    has to predict whether the swap happened or not. To solve the task, the classifier
    has access to a window of frames in the past, future or both, depending on the
    window_offset parameter.

    Notes
    -----
    1. Padding is not necessarily consistent for swapped windows. However, considering
       the small number of padded windows compared to the total number of windows, we
       decided to leave this edge case unmanaged for the time being.
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
        Initialize the dataset.

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

        Returns
        -------
        torch.utils.data.IterableDataset
            The windows dataset from the provided records.
        """
        super().__init__(records, window_size, window_offset, fps_scaling, transform)

        # Extract individuals and their feature indices from the first record
        features = self.records[0].posetracks.coords["features"].to_index()
        self.individuals = features.get_level_values("individuals").unique().tolist()
        self.individual_feature_indices = {
            ind: np.where(features.get_level_values("individuals") == ind)[0]
            for ind in self.individuals
        }

    def __iter__(self):
        while True:
            # Select a random window (global frame index)
            global_idx = torch.randint(0, self.n_frames, (1,), generator=self.g).item()

            # Map global index to (record_index, frame_index)
            rec_idx, frame_idx = self._global_to_local(global_idx)

            # Extract corresponding window
            x = self._select_and_pad(rec_idx, frame_idx)

            if torch.rand((1,), generator=self.g).item() < 0.5:
                # Swap intruder mouse, retry if a window from the same sequence was
                # chosen
                while True:
                    global_idx_swap = torch.randint(
                        0, self.n_frames, (1,), generator=self.g
                    ).item()
                    rec_idx_swap, frame_idx_swap = self._global_to_local(
                        global_idx_swap
                    )

                    if rec_idx_swap != rec_idx:
                        break

                # Extract swap window
                x_swap = self._select_and_pad(rec_idx_swap, frame_idx_swap)

                # Swap mice: always swap the second individual's features
                feature_idx = self.individual_feature_indices[self.individuals[1]]
                x[..., feature_idx] = x_swap[..., feature_idx]

                y = np.array(1, ndmin=1, dtype=np.float32)

            else:
                # Don't swap
                y = np.array(0, ndmin=1, dtype=np.float32)

            if self.transform:
                x = self.transform(x)

            yield x, y


class NWPDataset(RandomWindowDataset):
    """
    Dataset generator for the next window prediction task.

    In the next window prediction task, two windows are presented to the classifier. The
    goal of the task is to identify whether the second window comes from the same record
    of the first one or not. In the former case, the second window is randomly chosen to
    follow the first with a random delay (i.e. the first window "causes" the second). To
    solve the task, the classifier has access to a window of frames in the past, future
    or both, depending on the window_offset parameter.

    Notes
    -----
    1. Padding is not necessarily consistent for pre and post windows. However,
       considering the small number of padded windows compared to the total number of
       windows, we decided to leave this edge case unmanaged for the time being.
    2. The last window_size windows of each record will have overlapping post windows in
       the "same record" case. We could have skipped these windows, but we decided to
       allow them as they are not many, compared to the total number of windows, and
       they could even be beneficial to learn cause-effect relationships.
    3. For simplicity we concatenate the current and next window. This choice helps
       managing multiple tasks during training, but it leads to learning a joint
       positional and sequence embedding. In the future we might decide to decouple
       them by running backbone model twice and concatenating x embeddings before the
       classifier.
    4. In the current implementation, perfectly overlapping pre and post windows could
       be labeled as both positive and negative examples, depending on the random seed.
       While this ambiguity is not ideal, it is not expected to significantly affect the
       model performance, as the number of such cases is very small compared to the
       total. Furthermore, it simplifies the implementation by allowing us to handle the
       first and last windows of each record without special cases.
    """

    def __init__(
        self,
        records,
        window_size,
        window_offset=0,
        fps_scaling=1.0,
        method="strict",
        transform=None,
        base_seed=None,
    ):
        """
        Initialize the dataset.

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
        method : str, optional
            Selection method for examples of the negative class. Options are 'simple'
            (default) to allow negative examples to be selected from any sequence in
            the dataset, and 'strict' to force data from the same sequence.
        transform : callable, optional
            A function/transform to apply to the data (default is None).
        base_seed : int, optional
            Base seed for random number generation (default is None, which generates a
            random seed).

        Returns
        -------
        torch.utils.data.IterableDataset
            The windows dataset from the provided records.
        """
        super().__init__(
            records, window_size, window_offset, fps_scaling, transform, base_seed
        )

        if method not in ("simple", "strict"):
            raise ValueError(
                f"Invalid method '{method}'. Choose either 'simple' or 'strict'."
            )
        self.method = method

    def __iter__(self):
        while True:
            # Select a random window (global frame index)
            global_idx_pre = torch.randint(
                0, self.n_frames, (1,), generator=self.g
            ).item()

            # Map global index to (record_index, frame_index)
            rec_idx_pre, frame_idx_pre = self._global_to_local(global_idx_pre)

            if torch.rand((1,), generator=self.g).item() < 0.5:
                # Select a valid next window from same sequence (past current idx)
                # OBS: In order to generate a valid next window for every frame,
                #      including the last one of the sequence, we have to allow
                #      zero-distance windows
                global_idx_post = torch.randint(
                    frame_idx_pre, self.lengths[rec_idx_pre], (1,), generator=self.g
                ).item()

                # Map global index to (record_index, frame_index)
                rec_idx_post, frame_idx_post = self._global_to_local(global_idx_post)

                y = np.array(1, ndmin=1, dtype=np.float32)

            else:
                if self.method == "simple":
                    # Select a random window, retry if a true next window was chosen
                    while True:
                        global_idx_post = torch.randint(
                            0, self.n_frames, (1,), generator=self.g
                        ).item()
                        rec_idx_post, frame_idx_post = self._global_to_local(
                            global_idx_post
                        )

                        if (
                            rec_idx_post != rec_idx_pre
                            or frame_idx_post < frame_idx_pre
                        ):
                            # Found a valid next window (not from the same sequence or
                            # preceding the current one)
                            break

                elif self.method == "strict":
                    # Select the next window from the same sequence (past current idx)
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
            x_pre = self._select_and_pad(
                rec_idx_pre, frame_idx_pre, np.ceil(self.window_size / 2).astype(int)
            )

            # Extract next window
            x_post = self._select_and_pad(
                rec_idx_post, frame_idx_post, np.floor(self.window_size / 2).astype(int)
            )

            # Concatenate windows
            x = np.vstack((x_pre, x_post))

            if self.transform:
                x = self.transform(x)

            yield x, y


class DMPDataset(RandomWindowDataset):
    """
    Dataset generator for the delay mouse prediction task.

    In the delay mouse prediction/regression task, the intruder mouse trajectory is
    shifted in time by a random delay in the interval (-60, +60) frames. The goal of the
    prediction task is to assess whether the delay was negative or positive, while the
    goal of the regression task is to estimate by how many frames the trajectory was
    shifted. To solve the task, the classifier has access to a window of frames in the
    past, future or both, depending on the window_offset parameter.
    """

    def __init__(
        self,
        records,
        window_size,
        window_offset=0,
        fps_scaling=1.0,
        min_delay=-60,
        max_delay=60,
        regression=False,
        transform=None,
        base_seed=None,
    ):
        """
        Initialize the dataset.

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
        min_delay : int, optional
            Minimum delay in frames (default is -60).
        max_delay : int, optional
            Maximum delay in frames (default is 60).
        regression : bool, optional
            Whether to perform regression (default is False, which performs binary
            classification).
        transform : callable, optional
            A function/transform to apply to the data (default is None).
        base_seed : int, optional
            Base seed for random number generation (default is None, which generates a
            random seed).

        Returns
        -------
        torch.utils.data.IterableDataset
            The windows dataset from the provided records.
        """

        super().__init__(records, window_size, window_offset, fps_scaling, transform)

        # Extract individuals and their feature indices from the first record
        features = self.records[0].posetracks.coords["features"].to_index()
        self.individuals = features.get_level_values("individuals").unique().tolist()
        self.individual_feature_indices = {
            ind: np.where(features.get_level_values("individuals") == ind)[0]
            for ind in self.individuals
        }

        self.min_delay = min_delay
        self.max_delay = max_delay
        self.regression = regression

    def __iter__(self):
        while True:
            # Select a random window (global frame index)
            global_idx = torch.randint(0, self.n_frames, (1,), generator=self.g).item()

            # Map global index to (record_index, frame_index)
            rec_idx, frame_idx = self._global_to_local(global_idx)

            # Extract corresponding window
            x = self._select_and_pad(rec_idx, frame_idx)

            # Compute shift bounds
            lower_bound = max(frame_idx + self.min_delay, 0)
            upper_bound = min(frame_idx + self.max_delay, self.lengths[rec_idx])

            # Select random window from same sequence for the shift
            frame_idx_delay = torch.randint(
                lower_bound, upper_bound, (1,), generator=self.g
            ).item()

            # Get shift data
            x_delay = self._select_and_pad(rec_idx, frame_idx_delay)

            # Apply shifting: only shift the second individual's features
            feature_idx = self.individual_feature_indices[self.individuals[1]]
            x[..., feature_idx] = x_delay[..., feature_idx]

            # Compute label
            delta_delay = frame_idx_delay - frame_idx

            if self.regression:
                # Set rescaled shift distance as label
                y = (delta_delay - self.min_delay) / (self.max_delay - self.min_delay)
                y = np.array(y, ndmin=1, dtype=np.float32)
            else:
                y = np.array(delta_delay > 0, ndmin=1, dtype=np.float32)

            if self.transform:
                x = self.transform(x)

            yield x, y


class VSPDataset(RandomWindowDataset):
    """
    Dataset generator for the variable speed prediction task.

    In the video speed prediction/regression task, the pace of the window is multiplied
    by a random factor in the interval (0.5, 1.5). The goal of the prediction task is to
    assess whether the video speed was reduced or increased , while the  goal of the
    regression task is to estimate the pace factor. To solve the task, the classifier
    has access to a window of frames in the past, future or both, depending on the
    window_offset parameter.
    """

    def __init__(
        self,
        records,
        window_size,
        window_offset=0,
        fps_scaling=1.0,
        min_speed=0.5,
        max_speed=1.5,
        regression=False,
        transform=None,
        base_seed=None,
    ):
        """
        Initialize the dataset.

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
        min_speed : float, optional
            Minimum playback speed (default is 0.5).
        max_speed : float, optional
            Maximum playback speed (default is 1.5).
        regression : bool, optional
            Whether to perform regression (default is False, which performs binary
            classification).
        transform : callable, optional
            A function/transform to apply to the data (default is None).
        base_seed : int, optional
            Base seed for random number generation (default is None, which generates a
            random seed).

        Returns
        -------
        torch.utils.data.IterableDataset
            The windows dataset from the provided records.
        """
        super().__init__(records, window_size, window_offset, fps_scaling, transform)

        self.min_speed = min_speed
        self.max_speed = max_speed
        self.regression = regression

    def __iter__(self):
        while True:
            # Select a random window (global frame index)
            global_idx = torch.randint(0, self.n_frames, (1,), generator=self.g).item()

            # Map global index to (record_index, frame_index)
            rec_idx, frame_idx = self._global_to_local(global_idx)

            # Draw playback speed at random
            rel_speed = torch.rand((1,), generator=self.g).item()
            speed = rel_speed * (self.max_speed - self.min_speed) + self.min_speed

            # Compute actual window size (i.e. the actual frames required to generate
            # the window)
            window_size_act = np.round(speed * self.window_size).astype(int)

            # Extract corresponding window
            x_act = self._select_and_pad(rec_idx, frame_idx, window_size_act)

            # Interpolate frames to get exactly window_size frames
            f1d = interp1d(np.linspace(0, 1, window_size_act), x_act, axis=0)
            x = f1d(np.linspace(0, 1, self.window_size))

            if self.regression:
                # Set relative speed as label
                y = np.array(rel_speed, ndmin=1, dtype=np.float32)
            else:
                # Set speed threshold as label
                speed_threshold = (self.min_speed + self.max_speed) / 2.0
                y = np.array(speed > speed_threshold, ndmin=1, dtype=np.float32)

            if self.transform:
                x = self.transform(x)

            yield x, y
