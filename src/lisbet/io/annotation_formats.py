"""Annotation-format loaders for LISBET supervised fine-tuning.

This module loads annotations from different external formats and converts them
to the internal LISBET annotation representation:

    xarray.Dataset
        dims: time, behaviors, annotators
        data variable: target_cls(time, behaviors, annotators)

Supported formats
-----------------
movement
    Current/default LISBET annotation format. Loads NetCDF files such as
    annotations.nc or manual_scoring.nc.

csv-events
    Generic interval-based CSV event annotation format with at least:
    behavior,start_time,end_time

boris
    BORIS tabular CSV export with START/STOP rows. The converter pairs START
    and STOP rows and creates one behavioral interval per bout.

Notes
-----
The csv-events and boris loaders expect annotation times in seconds and convert them
to frame indices using the provided fps value, the BORIS FPS column, or a
default value.
"""

from __future__ import annotations

import logging
import os
import re
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr


BORIS_EVENT_HEADER = (
    "Time,Media file path,Total length,FPS,Subject,Behavior,"
    "Behavioral category,Comment,Status"
)


def _infer_n_frames(
    n_frames: Optional[int] = None,
    intervals_df: Optional[pd.DataFrame] = None,
    fps: float = 30,
) -> int:
    """Infer the number of frames from metadata or annotation intervals."""
    if n_frames is not None:
        return int(n_frames)

    if intervals_df is not None and len(intervals_df) > 0:
        return int(np.ceil(intervals_df["end_time"].max() * fps))

    raise ValueError(
        "Could not infer the number of frames. Provide n_frames or valid intervals."
    )


def _infer_fps(
    fps: Optional[float] = None,
    boris_events: Optional[pd.DataFrame] = None,
    default_fps: float = 30,
) -> float:
    """Infer FPS from explicit metadata, BORIS table, or fallback."""
    if fps is not None:
        return float(fps)

    if boris_events is not None and "FPS" in boris_events.columns:
        fps_values = pd.to_numeric(boris_events["FPS"], errors="coerce").dropna()
        if len(fps_values) > 0:
            return float(fps_values.iloc[0])

    return float(default_fps)


def intervals_to_lisbet_xarray(
    intervals_df: pd.DataFrame,
    fps: float = 30,
    n_frames: Optional[int] = None,
    annotator: str = "annotator0",
    behavior_col: str = "behavior",
    start_col: str = "start_time",
    end_col: str = "end_time",
    other_label: str = "other",
    add_other: bool = True,
    source_software: str = "CSV",
) -> xr.Dataset:
    """Convert interval annotations to the internal LISBET xarray format.

    Parameters
    ----------
    intervals_df
        Interval annotation table. Required columns are behavior, start_time,
        and end_time. Times are expected in seconds.
    fps
        Frames per second used to convert seconds to frame indices.
    n_frames
        Number of frames in the corresponding tracking sequence. If None, it is
        inferred from the maximum end time.
    annotator
        Name assigned to the annotator coordinate.
    behavior_col, start_col, end_col
        Column names in the interval table.
    other_label
        Label assigned to frames without any annotated behavior.
    add_other
        Whether to add an "other" behavior for unannotated frames.
    source_software
        Stored in the output xarray Dataset attributes.

    Returns
    -------
    xarray.Dataset
        LISBET-compatible annotation dataset.
    """
    df = intervals_df.copy()

    required_cols = [behavior_col, start_col, end_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required interval columns: {missing}")

    df[start_col] = pd.to_numeric(df[start_col], errors="raise")
    df[end_col] = pd.to_numeric(df[end_col], errors="raise")

    if (df[end_col] <= df[start_col]).any():
        bad = df.loc[df[end_col] <= df[start_col]]
        raise ValueError(f"Invalid intervals with end_time <= start_time:\n{bad}")

    df["start_frame"] = np.floor(df[start_col] * fps).astype(int)
    df["end_frame"] = np.ceil(df[end_col] * fps).astype(int)

    if n_frames is None:
        n_frames = _infer_n_frames(intervals_df=df, fps=fps)

    behaviors = sorted(df[behavior_col].astype(str).unique().tolist())

    if add_other and other_label not in behaviors:
        behaviors.append(other_label)

    behavior_to_idx = {behavior: idx for idx, behavior in enumerate(behaviors)}
    target_cls = np.zeros((n_frames, len(behaviors), 1), dtype=np.int64)

    for _, row in df.iterrows():
        behavior = str(row[behavior_col])
        start_frame = max(int(row["start_frame"]), 0)
        end_frame = min(int(row["end_frame"]), n_frames)

        if end_frame <= start_frame:
            continue

        b_idx = behavior_to_idx[behavior]
        target_cls[start_frame:end_frame, b_idx, 0] = 1

    if add_other:
        other_idx = behavior_to_idx[other_label]
        active_any = target_cls[:, :, 0].sum(axis=1) > 0
        target_cls[~active_any, other_idx, 0] = 1

    return xr.Dataset(
        data_vars={
            "target_cls": (
                ("time", "behaviors", "annotators"),
                target_cls,
            )
        },
        coords={
            "time": np.arange(n_frames, dtype=np.int64),
            "behaviors": np.array(behaviors, dtype=str),
            "annotators": np.array([annotator], dtype=str),
        },
        attrs={
            "source_software": source_software,
            "ds_type": "annotations",
            "fps": fps,
            "time_unit": "frames",
        },
    )


def load_movement_annotations(seq_path: Path) -> Optional[xr.Dataset]:
    """Load the current/default LISBET movement annotation format.

    This preserves the existing LISBET behavior: annotations are read from
    NetCDF files whose names contain "manual_scoring" or "annotations".
    """
    rx = re.compile(r"(?i)(manual_scoring|annotations).*\.nc$")

    annotations = [
        xr.open_dataset(pth, engine="scipy")
        for pth in seq_path.iterdir()
        if rx.search(pth.name)
    ]

    if len(annotations) == 0:
        return None

    annotations = xr.concat(annotations, dim="annotators")

    if "behaviors" in annotations.coords:
        logging.debug("Annotations: %s", annotations.coords["behaviors"].values)

    return annotations


def load_csv_event_annotations(
    seq_path: Path,
    n_frames: Optional[int] = None,
    fps: Optional[float] = None,
) -> Optional[xr.Dataset]:
    """Load a generic interval-based CSV annotation format.

    Expected CSV columns
    --------------------
    behavior,start_time,end_time

    Optional columns
    ----------------
    annotator,subject,video_id,media_file

    Times are expected in seconds.
    """
    rx = re.compile(r"(?i)(manual_scoring|annotations).*\.csv$")
    csv_files = [pth for pth in seq_path.iterdir() if rx.search(pth.name)]

    if len(csv_files) == 0:
        return None

    annotations = []

    for idx, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)

        annotator = (
            str(df["annotator"].iloc[0])
            if "annotator" in df.columns and len(df) > 0
            else f"annotator{idx}"
        )

        ann_fps = _infer_fps(fps=fps, default_fps=30)
        ann_n_frames = _infer_n_frames(
            n_frames=n_frames,
            intervals_df=df,
            fps=ann_fps,
        )

        ann = intervals_to_lisbet_xarray(
            df,
            fps=ann_fps,
            n_frames=ann_n_frames,
            annotator=annotator,
            source_software="CSV",
        )

        annotations.append(ann)

    if len(annotations) == 1:
        return annotations[0]

    return xr.concat(annotations, dim="annotators")


def read_boris_csv(boris_file: Path) -> pd.DataFrame:
    """Read a BORIS CSV export and return the clean event table.

    BORIS CSV exports may contain metadata rows before the actual event table.
    This function detects the event-table header and reads from that point.

    If the file is already a clean BORIS event table, it is read directly.
    """
    boris_file = Path(boris_file)

    with boris_file.open("r", encoding="utf-8-sig") as f:
        text = f.read()

    start = text.find(BORIS_EVENT_HEADER)

    if start == -1:
        # The file may already be a clean BORIS event table.
        df = pd.read_csv(boris_file)
        required_cols = {"Time", "Media file path", "Behavior", "Status"}

        if required_cols.issubset(df.columns):
            return df

        raise ValueError(
            "Could not find the BORIS event-table header and the file does not "
            "appear to be a clean BORIS event table. Required columns are: "
            f"{sorted(required_cols)}"
        )

    event_text = text[start:]
    return pd.read_csv(StringIO(event_text))


def boris_events_to_intervals(boris_events: pd.DataFrame) -> pd.DataFrame:
    """Convert BORIS START/STOP rows to an interval annotation table."""
    required_cols = ["Time", "Media file path", "Behavior", "Status"]
    missing = [col for col in required_cols if col not in boris_events.columns]
    if missing:
        raise ValueError(f"Missing required BORIS columns: {missing}")

    df = boris_events.copy()

    if "Subject" not in df.columns:
        df["Subject"] = ""

    df["Time"] = pd.to_numeric(df["Time"], errors="raise")
    df["Status"] = df["Status"].astype(str).str.strip().str.upper()
    df["Behavior"] = df["Behavior"].astype(str).str.strip()

    df = df[df["Status"].isin(["START", "STOP"])].copy()

    df = df.sort_values(
        ["Media file path", "Subject", "Behavior", "Time"]
    ).reset_index(drop=True)

    intervals = []
    open_events = {}

    for _, row in df.iterrows():
        media_file = str(row["Media file path"])
        subject = "" if pd.isna(row["Subject"]) else str(row["Subject"])
        behavior = str(row["Behavior"])
        status = str(row["Status"])
        time = float(row["Time"])

        key = (media_file, subject, behavior)

        if status == "START":
            if key in open_events:
                raise ValueError(
                    f"Repeated START without STOP for behavior '{behavior}' "
                    f"in file '{media_file}' at time {time}."
                )
            open_events[key] = time

        elif status == "STOP":
            if key not in open_events:
                raise ValueError(
                    f"STOP without START for behavior '{behavior}' "
                    f"in file '{media_file}' at time {time}."
                )

            start_time = open_events.pop(key)
            end_time = time
            duration = end_time - start_time

            if duration <= 0:
                raise ValueError(
                    f"Invalid interval for behavior '{behavior}': "
                    f"start={start_time}, end={end_time}."
                )

            intervals.append(
                {
                    "video_id": os.path.basename(media_file),
                    "media_file": media_file,
                    "subject": subject,
                    "behavior": behavior,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                }
            )

    if open_events:
        raise ValueError(
            "Some START events do not have matching STOP events: "
            f"{open_events}"
        )

    return pd.DataFrame(intervals)


def load_boris_annotations(
    seq_path: Path,
    n_frames: Optional[int] = None,
    fps: Optional[float] = None,
) -> Optional[xr.Dataset]:
    """Load BORIS tabular CSV annotations and convert them to LISBET format."""
    rx = re.compile(r"(?i)(boris|manual_scoring|annotations).*\.csv$")
    boris_files = [pth for pth in seq_path.iterdir() if rx.search(pth.name)]

    if len(boris_files) == 0:
        return None

    annotations = []

    for idx, boris_file in enumerate(boris_files):
        boris_events = read_boris_csv(boris_file)
        intervals = boris_events_to_intervals(boris_events)

        ann_fps = _infer_fps(
            fps=fps,
            boris_events=boris_events,
            default_fps=30,
        )

        ann_n_frames = _infer_n_frames(
            n_frames=n_frames,
            intervals_df=intervals,
            fps=ann_fps,
        )

        ann = intervals_to_lisbet_xarray(
            intervals,
            fps=ann_fps,
            n_frames=ann_n_frames,
            annotator=f"annotator{idx}",
            source_software="BORIS",
        )

        annotations.append(ann)

    if len(annotations) == 1:
        return annotations[0]

    return xr.concat(annotations, dim="annotators")


def load_annotations(
    seq_path: Path,
    annot_format: str = "movement",
    n_frames: Optional[int] = None,
    fps: Optional[float] = None,
) -> Optional[xr.Dataset]:
    """Load annotations from a sequence directory using the requested format.

    Parameters
    ----------
    seq_path
        Sequence directory.
    annot_format
        Annotation format to load. Supported values are:
        "movement", "csv-events", and "boris".
    n_frames
        Number of frames in the corresponding pose-tracking sequence.
    fps
        Frames per second used to convert interval times to frame indices.
        If missing, BORIS FPS metadata or a default value may be used.

    Returns
    -------
    xarray.Dataset or None
        LISBET-compatible annotation dataset, or None if no annotation file is
        found for the requested format.
    """
    annot_format = str(annot_format).lower()

    if annot_format == "movement":
        return load_movement_annotations(seq_path)

    if annot_format == "csv-events":
        return load_csv_event_annotations(seq_path, n_frames=n_frames, fps=fps)

    if annot_format == "boris":
        return load_boris_annotations(seq_path, n_frames=n_frames, fps=fps)

    raise ValueError(
        f"Unsupported annot_format='{annot_format}'. "
        "Expected one of: movement, csv-events, boris."
    )
