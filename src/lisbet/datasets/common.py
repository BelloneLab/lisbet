"""Common code for selecting windows from a dataset of records."""

from typing import Literal

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

    def __init__(
        self,
        records,
        window_size,
        window_offset=0,
        fps_scaling=1.0,
        engine: Literal["xarray", "numpy"] = "xarray",
    ):
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
        engine : {"xarray", "numpy"}, optional
            Output engine. The default ``"xarray"`` returns an xarray Dataset with
            coordinates and data variables intact. ``"numpy"`` returns an owned,
            writable array with shape ``(time, individuals, keypoints, space)``.

        Raises
        ------
        ValueError
            If no records are provided or if any record contains fewer than 2
            individuals, or if ``engine`` is unsupported.
        """
        # Validate input parameters
        if not records:
            raise ValueError("No records provided to the dataset.")
        if any([rec.posetracks["individuals"].size < 2 for rec in records]):
            raise ValueError("LISBET requires at least 2 individuals in each record.")
        if engine not in ("xarray", "numpy"):
            raise ValueError(
                f"Invalid engine '{engine}'. Choose either 'xarray' or 'numpy'."
            )

        self.records = records
        self.n_records = len(records)
        self.engine = engine

        self.window_size = window_size
        self.window_offset = window_offset
        self.fps_scaling = fps_scaling
        self.rel_time_coords = np.linspace(
            0, self.window_size - 1, self.window_size, dtype=int
        )

        # NOTE: Using torch tensors to trigger proper initialization of multiprocessing
        #       workers on MacOS. This is a workaround, but it does not affect the
        #       functionality of the class.
        self.lengths = torch.tensor(
            [rec.posetracks.sizes["time"] for rec in self.records], dtype=int
        )
        self.cumlens = torch.cumsum(self.lengths, dim=0)
        self.n_frames = int(self.cumlens[-1])

        # NumPy selection only needs the position variable. Transpose and obtain its
        # backing array once so every sample can be sliced on canonical axes without
        # repeatedly asking xarray to index or materialize the record.
        self._position_arrays = None
        if self.engine == "numpy":
            self._position_arrays = tuple(
                rec.posetracks["position"]
                .transpose("time", "individuals", "keypoints", "space")
                .values
                for rec in self.records
            )

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

    def select(self, rec_idx, frame_idx, fps_scaling=None):
        """
        Select a window from the dataset, applying padding and interpolation as needed.

        The selected window is returned as an independent xarray Dataset or NumPy
        array to avoid unintentional changes to source records (for example, by a
        self-supervised task or augmentation).

        Parameters
        ----------
        rec_idx : int
            Index of the record from which to select the window.
        frame_idx : int
            Index of the central frame within the record.
        fps_scaling : float, optional
            Override the default fps scaling factor for this selection. If None, uses
            the default fps_scaling set during initialization.

        Returns
        -------
        x : xarray.Dataset or numpy.ndarray
            The selected and interpolated window. NumPy output has shape ``(time,
            individuals, keypoints, space)``.

        Notes
        -----
        1. The interpolation is done here, and not directly on the records, to avoid
           resampling at the original fps before returning the output during inference.
           Furthermore, even during training, it is useful to only iterate over the
           original frames, rather than artificially inflating or deflating the dataset.
        """
        if fps_scaling is None:
            fps_scaling = self.fps_scaling

        if self.engine == "numpy":
            return self._select_numpy(rec_idx, frame_idx, fps_scaling)

        return self._select_xarray(rec_idx, frame_idx, fps_scaling)

    def _select_xarray(self, rec_idx, frame_idx, fps_scaling):
        """Select a window using the original xarray implementation."""

        x = self.records[rec_idx].posetracks

        if fps_scaling == 1.0:
            # NOTE: If no fps scaling is applied, we can directly select the window
            #       from the posetrack data without (expensive) interpolation
            # Compute scaled time coordinates
            start_idx = frame_idx - self.window_size + self.window_offset + 1
            stop_idx = frame_idx + self.window_offset
            time_coords = np.linspace(start_idx, stop_idx, self.window_size, dtype=int)

            x = x.reindex(time=time_coords, fill_value=0).assign_coords(
                time=self.rel_time_coords
            )

        else:
            # NOTE: If fps scaling is applied, we need to interpolate the posetrack and
            #       then select the window from the interpolated data
            # Compute scaled time coordinates
            scaled_window_size = int(np.rint(fps_scaling * self.window_size))
            scaled_window_offset = int(np.rint(fps_scaling * self.window_offset))
            scaled_start_idx = frame_idx - scaled_window_size + scaled_window_offset + 1
            scaled_stop_idx = frame_idx + scaled_window_offset
            scaled_time_coords = np.linspace(
                scaled_start_idx, scaled_stop_idx, scaled_window_size, dtype=int
            )

            # Compute interpolation time coordinates
            interp_time_coords = np.linspace(
                scaled_start_idx, scaled_stop_idx, self.window_size
            )

            # Select, pad (reindex) and interpolate data
            x = (
                x.reindex(time=scaled_time_coords, fill_value=0)
                .interp(time=interp_time_coords)
                .assign_coords(time=self.rel_time_coords)
            )

        return x

    def _select_numpy(self, rec_idx, frame_idx, fps_scaling):
        """Select a canonical NumPy window without constructing xarray objects."""
        source = self._position_arrays[rec_idx]

        if fps_scaling == 1.0:
            start_idx = frame_idx - self.window_size + self.window_offset + 1
            return self._copy_padded_interval(source, start_idx, self.window_size)

        scaled_window_size = int(np.rint(fps_scaling * self.window_size))
        scaled_window_offset = int(np.rint(fps_scaling * self.window_offset))
        scaled_start_idx = frame_idx - scaled_window_size + scaled_window_offset + 1

        if scaled_window_size <= 0:
            raise ValueError("fps_scaling produces an empty source window.")

        scaled = self._copy_padded_interval(
            source, scaled_start_idx, scaled_window_size
        )

        # xarray's linear interpolation produces floating-point values even when the
        # source is integral or float32. A single source point is an underdetermined
        # linear interpolation and xarray returns NaNs for it.
        # TODO: In a future release, prefer an "at least float32" policy over xarray
        # parity: promote integer and float16 inputs to float32 while preserving
        # floating-point dtypes that are already float32 or higher.
        output_shape = (self.window_size, *source.shape[1:])
        if scaled_window_size == 1:
            return np.full(output_shape, np.nan, dtype=np.float64)

        interp_coords = np.linspace(
            0.0, scaled_window_size - 1, self.window_size, dtype=np.float64
        )
        lower = np.floor(interp_coords).astype(int)
        upper = np.ceil(interp_coords).astype(int)
        weights = interp_coords - lower
        weights = weights.reshape((-1,) + (1,) * (scaled.ndim - 1))

        scaled = scaled.astype(np.float64, copy=False)
        return scaled[lower] * (1.0 - weights) + scaled[upper] * weights

    @staticmethod
    def _copy_padded_interval(source, start_idx, size):
        """Copy an interval into a zero-padded, writable output buffer."""
        output = np.zeros((size, *source.shape[1:]), dtype=source.dtype)
        source_start = max(start_idx, 0)
        source_stop = min(start_idx + size, source.shape[0])

        if source_start < source_stop:
            output_start = source_start - start_idx
            output_stop = output_start + source_stop - source_start
            output[output_start:output_stop] = source[source_start:source_stop]

        return output


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
        engine: Literal["xarray", "numpy"] = "xarray",
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
        engine : {"xarray", "numpy"}, optional
            Output engine (default is ``"xarray"``).

        Raises
        ------
        ValueError
            If ``annot_format`` is not one of ``"binary"``, ``"multiclass"``, or
            ``"multilabel"``, or if ``engine`` is unsupported.
        """
        # Validate input parameters
        if annot_format not in ("binary", "multiclass", "multilabel"):
            raise ValueError(
                f"Invalid label format '{annot_format}'. "
                "Choose either 'binary', 'multiclass', or 'multilabel'."
            )

        super().__init__(records, window_size, window_offset, fps_scaling, engine)

        self.annot_format = annot_format
        self._annotation_arrays = None
        # NOTE: Keep annotation caching specific to the NumPy engine. Annotation files
        # opened by xarray may be lazy, so caching their full ``.values`` arrays would
        # change the xarray engine's initialization cost and memory use. Test lazy I/O
        # behavior explicitly before considering a shared eager cache in the future.
        if self.engine == "numpy":
            self._annotation_arrays = tuple(
                rec.annotations["target_cls"]
                .transpose("time", "behaviors", "annotators")
                .values
                for rec in self.records
            )

    def select(self, rec_idx, frame_idx, fps_scaling=None):
        """
        Select a window and its corresponding annotation target.

        Parameters
        ----------
        rec_idx : int
            Index of the record from which to select the window.
        frame_idx : int
            Index of the central frame within the record.
        fps_scaling : float, optional
            Override the default fps scaling factor for this selection. If None, uses
            the default fps_scaling set during initialization.

        Returns
        -------
        x : xarray.Dataset or numpy.ndarray
            The selected and interpolated window. NumPy output has shape ``(time,
            individuals, keypoints, space)``.
        y : numpy.ndarray
            The annotation target(s) for the selected window, format depends on
            annot_format.
        """
        x = super().select(rec_idx, frame_idx, fps_scaling)

        if self.engine == "numpy":
            target = self._annotation_arrays[rec_idx][frame_idx]
            if self.annot_format == "binary":
                y = target.copy()
            elif self.annot_format == "multiclass":
                # xarray skips NaNs by default when reducing floating-point arrays.
                y = np.asarray(np.nanargmax(target, axis=0)).squeeze().copy()
            else:
                y = target.squeeze().copy()

        elif self.annot_format == "binary":
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
