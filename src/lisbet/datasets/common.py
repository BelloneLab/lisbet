"""Common code for selecting windows from a dataset of records."""

import numpy as np
import torch


class WindowSelector:
    """
    Selects windows from a dataset of records.

    This class provides methods to extract temporal windows from a list of records,
    handling padding and interpolation as needed. It supports mapping between global
    and local frame indices and can scale windows according to a frames-per-second
    (fps) scaling factor.
    """

    def __init__(self, records, window_size, window_offset=0, fps_scaling=1.0):
        """
        Initialize the WindowSelector.

        Parameters
        ----------
        records : list
            List of records containing pose tracking data.
        window_size : int
            Size of the window in frames.
        window_offset : int, optional
            Offset for the window in frames (default is 0).
        fps_scaling : float, optional
            Scaling factor for the frames per second (default is 1.0).

        Raises
        ------
        ValueError
            If no records are provided or if any record contains fewer than 2
            individuals.
        """
        # Validate input parameters
        if not records:
            raise ValueError("No records provided to the dataset.")
        if any([rec.posetracks["individuals"].size < 2 for rec in records]):
            raise ValueError("LISBET requires at least 2 individuals in each record.")

        self.records = records
        self.n_records = len(records)

        self.window_size = window_size
        self.window_offset = window_offset
        self.fps_scaling = fps_scaling

        # NOTE: Using torch tensors to trigger proper initialization of multiprocessing
        #       workers on MacOS. This is a workaround, but it does not affect the
        #       functionality of the class.
        self.lengths = torch.tensor(
            [rec.posetracks.sizes["time"] for rec in self.records], dtype=int
        )
        self.cumlens = torch.cumsum(self.lengths, dim=0)
        self.n_frames = int(self.cumlens[-1])

    def global_to_local(self, global_idx):
        """
        Map a global frame index to a local (record_index, local_frame_index) pair.

        Parameters
        ----------
        global_idx : int
            Global frame index (0 ≤ global_idx < total_n_frames).

        Returns
        -------
        rec_idx : int
            Index of the record containing the frame.
        local_idx : int
            Local frame index within the selected record.
        """
        rec_idx = torch.searchsorted(self.cumlens, global_idx, side="right").item()
        prev_sum = 0 if rec_idx == 0 else self.cumlens[rec_idx - 1].item()
        local_idx = global_idx - prev_sum

        return rec_idx, local_idx

    def select(self, rec_idx, frame_idx, window_size=None):
        """
        Select a window from the dataset, applying padding and interpolation as needed.

        The selected window is returned as a new xarray.Dataset to avoid unintentional
        changes to the records in the window dictionary (e.g., by the self-supervised
        tasks).

        Parameters
        ----------
        rec_idx : int
            Index of the record from which to select the window.
        frame_idx : int
            Index of the central frame within the record.
        window_size : int, optional
            Size of the window in frames. If None, uses the default window_size.

        Returns
        -------
        x : xarray.Dataset
            The selected and interpolated window.

        Notes
        -----
        1. The interpolation is done here, and not directly on the records, to avoid
           resampling at the original fps before returning the output during inference.
           Furthermore, even during training, it is useful to only iterate over the
           original frames, rather than artificially inflating or deflating the dataset.
        """
        if window_size is None:
            window_size = self.window_size
        if window_size <= 1:
            raise ValueError(
                "LISBET requires window_size to be greater than 1 to avoid degenerate "
                "interpolation."
            )

        x = self.records[rec_idx].posetracks

        # Compute scaled time coordinates
        scaled_window_size = int(np.rint(self.fps_scaling * window_size))
        scaled_window_offset = int(np.rint(self.fps_scaling * self.window_offset))
        scaled_start_idx = frame_idx - scaled_window_size + scaled_window_offset + 1
        scaled_stop_idx = frame_idx + scaled_window_offset
        scaled_time_coords = np.linspace(
            scaled_start_idx, scaled_stop_idx, scaled_window_size, dtype=int
        )

        # Compute interpolation time coordinates
        interp_time_coords = np.linspace(scaled_start_idx, scaled_stop_idx, window_size)

        # Compute relative time coordinates
        rel_time_coords = np.linspace(0, window_size - 1, window_size, dtype=int)

        # Select, pad (reindex) and interpolate data
        x = (
            x.reindex(time=scaled_time_coords, fill_value=0)
            .interp(time=interp_time_coords)
            .assign_coords(time=rel_time_coords)
        )

        return x


class AnnotatedWindowSelector(WindowSelector):
    """
    WindowSelector with annotation extraction.

    Extends WindowSelector to also extract annotation targets for each selected window,
    supporting binary, multiclass, and multilabel annotation formats.
    """

    def __init__(
        self,
        records,
        window_size,
        window_offset=0,
        fps_scaling=1.0,
        annot_format="multiclass",
    ):
        """
        Initialize the AnnotatedWindowSelector.

        Parameters
        ----------
        records : list
            List of records containing the data and annotations.
        window_size : int
            Size of the window in frames.
        window_offset : int, optional
            Offset for the window in frames (default is 0).
        fps_scaling : float, optional
            Scaling factor for the frames per second (default is 1.0).
        annot_format : str, optional
            Format of the annotations ('binary', 'multiclass', or 'multilabel').

        Raises
        ------
        ValueError
            If annot_format is not one of 'binary', 'multiclass', or 'multilabel'.
        """
        # Validate input parameters
        if annot_format not in ("binary", "multiclass", "multilabel"):
            raise ValueError(
                f"Invalid label format '{annot_format}'. "
                "Choose either 'binary', 'multiclass', or 'multilabel'."
            )

        super().__init__(records, window_size, window_offset, fps_scaling)

        self.annot_format = annot_format

    def select(self, rec_idx, frame_idx, window_size=None):
        """
        Select a window and its corresponding annotation target.

        Parameters
        ----------
        rec_idx : int
            Index of the record from which to select the window.
        frame_idx : int
            Index of the central frame within the record.
        window_size : int, optional
            Size of the window in frames. If None, uses the default window_size.

        Returns
        -------
        x : xarray.Dataset
            The selected and interpolated window.
        y : numpy.ndarray
            The annotation target(s) for the selected window, format depends on
            annot_format.
        """
        x = super().select(rec_idx, frame_idx, window_size)

        if self.annot_format == "binary":
            y = self.records[rec_idx].annotations.target_cls.isel(time=frame_idx).values

        elif self.annot_format == "multiclass":
            y = (
                self.records[rec_idx]
                .annotations.target_cls.isel(time=frame_idx)
                .argmax("behaviors")
                .squeeze()
                .values
            )

        elif self.annot_format == "multilabel":
            y = (
                self.records[rec_idx]
                .annotations.target_cls.isel(time=frame_idx)
                .squeeze()
                .values
            )

        return x, y
