"""Common code for selecting windows from a dataset of records."""

import logging
import os
from typing import Literal

import numpy as np
import torch


def leakfree_config(window_sampling="any"):
    """Sampling switches of the self-supervised datasets, resolved when a dataset is built.

    ``window_sampling`` is the public option (``--window_sampling``): "any" keeps the original
    behaviour (windows may extend past the record edges and are zero padded); "inside" redraws every
    sample whose windows would touch an edge (see ``LISBET_PAD_MODE=reject`` below), draws the shift
    sign 50/50 before its size and applies the same rule to the geom views. Zero padding at record
    edges carries the label of the order, shift and cons tasks (the padded span of a window tells where
    in the record it sits), which a network can exploit instead of the content.

    The environment variables below are experimental overrides (they win over ``window_sampling``
    when set); all of them default to the original lisbet behaviour.

    LISBET_PAD_MODE   none | reject | equalise
        reject: every window a sample uses must lie inside its record (redraw
        otherwise); equalise (shift, numpy engine): both individuals are zeroed wherever
        either window was padded, so the two zero-runs are identical.
    LISBET_SHIFT_MODE signed | magnitude   (magnitude: label 1 if the clinician window
        is displaced by min_shift..max_shift frames, 0 if synchronous; needs reject)
    LISBET_SHIFT_SIGN original | balanced  (balanced: sign drawn 50/50 before the
        size, so P(d > 0 | position in record) = 0.5; d = 0 is never drawn)
    LISBET_MAX_SHIFT / LISBET_MIN_SHIFT    frames (defaults: dataset value / 20)
    LISBET_GEOM_PAD   keep | reject        (geom: reject padded anchor / target windows)
    LISBET_CONS_NEG   other | within        (cons negatives: clinician from another
        record (original) or from the same record at least LISBET_CONS_MIN_GAP frames
        away; implies reject, anchors restricted to records long enough for a partner)
    """
    if window_sampling not in ("any", "inside"):
        raise ValueError(f"window_sampling={window_sampling!r}, expected 'any' or 'inside'")
    base = {
        "pad_mode": "none",
        "shift_sign": "original",
        "geom_pad": "keep",
    }
    if window_sampling == "inside":
        base = {"pad_mode": "reject", "shift_sign": "balanced", "geom_pad": "reject"}
    env = os.environ.get
    cfg = {
        "pad_mode": env("LISBET_PAD_MODE", base["pad_mode"]),
        "shift_mode": env("LISBET_SHIFT_MODE", "signed"),
        "shift_sign": env("LISBET_SHIFT_SIGN", base["shift_sign"]),
        "max_shift": int(env("LISBET_MAX_SHIFT", "0")) or None,
        "min_shift": int(env("LISBET_MIN_SHIFT", "20")),
        "geom_pad": env("LISBET_GEOM_PAD", base["geom_pad"]),
        "cons_neg": env("LISBET_CONS_NEG", "other"),
        "cons_gap": int(env("LISBET_CONS_MIN_GAP", "600")),
    }
    allowed = {
        "pad_mode": ("none", "reject", "equalise"),
        "shift_mode": ("signed", "magnitude"),
        "shift_sign": ("original", "balanced"),
        "geom_pad": ("keep", "reject"),
        "cons_neg": ("other", "within"),
    }
    for k, v in allowed.items():
        if cfg[k] not in v:
            raise ValueError(f"LISBET switch {k}={cfg[k]!r}, expected one of {v}")
    if cfg["shift_mode"] == "magnitude" and cfg["pad_mode"] != "reject":
        raise ValueError("LISBET_SHIFT_MODE=magnitude requires LISBET_PAD_MODE=reject")
    return cfg


def log_leakfree_once(name, cfg):
    """Leave a trace in the training log when any switch is active."""
    if (
        cfg["pad_mode"] != "none"
        or cfg["shift_mode"] != "signed"
        or cfg["shift_sign"] != "original"
        or cfg["max_shift"]
        or cfg["geom_pad"] != "keep"
        or cfg["cons_neg"] != "other"
    ):
        logging.info("leak-free sampling switches active in %s: %s", name, cfg)


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

    def _scaled(self, fps_scaling=None):
        fs = self.fps_scaling if fps_scaling is None else fps_scaling
        if fs == 1.0:
            return self.window_size, self.window_offset
        return (
            int(np.rint(fs * self.window_size)),
            int(np.rint(fs * self.window_offset)),
        )

    def inside_range(self, rec_idx, fps_scaling=None):
        """(lo, hi): frames whose window lies entirely inside the record."""
        W, off = self._scaled(fps_scaling)
        return W - off - 1, int(self.lengths[rec_idx]) - 1 - off

    def inside(self, rec_idx, frame_idx, fps_scaling=None):
        lo, hi = self.inside_range(rec_idx, fps_scaling)
        return lo <= frame_idx <= hi

    def padding_mask(self, rec_idx, frame_idx):
        """(window_size,) bool, True where the window position is outside the record."""
        start = frame_idx - self.window_size + self.window_offset + 1
        pos = start + np.arange(self.window_size)
        return (pos < 0) | (pos > int(self.lengths[rec_idx]) - 1)

    def draw_inside(self, g, fps_scaling=None, max_tries=1_000_000):
        """Random frame (uniform over all frames, as the original draw) with an inside window."""
        for _ in range(max_tries):
            global_idx = torch.randint(0, self.n_frames, (1,), generator=g).item()
            rec_idx, frame_idx = self.global_to_local(global_idx)
            if self.inside(rec_idx, frame_idx, fps_scaling):
                return rec_idx, frame_idx
        raise RuntimeError("no record is long enough for a window fully inside it")

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
