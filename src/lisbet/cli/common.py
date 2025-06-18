"""Common argument parsers for CLI tools."""

import argparse
import textwrap
from pathlib import Path


def add_verbosity_args(parser: argparse.ArgumentParser) -> None:
    """Add verbosity arguments."""
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (can be repeated, e.g. -vv)",
    )
    verbosity_group.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set specific log level",
    )


def add_keypoints_args(parser: argparse.ArgumentParser) -> None:
    """Add common keypoints-related arguments."""
    parser.add_argument(
        "--data_format",
        type=str,
        default="DLC",
        choices=["DLC", "SLEAP", "movement"],
        help="Keypoints dataset format",
    )
    parser.add_argument(
        "--data_scale",
        type=str,
        help=textwrap.dedent(
            """\
            Spatial dimensions of the dataset.

            - For 2D data:  WIDTHxHEIGHT         (e.g., 1920x1080)
            - For 3D data:  WIDTHxHEIGHTxDEPTH   (e.g., 1920x1080x480)

            When specified, input coordinates (x, y, z) are interpreted in data units
            and normalized to the [0, 1] range by dividing by the given scale.

            If omitted, the scale is inferred from the dataset.
            """
        ),
    )
    parser.add_argument(
        "--select_coords",
        type=str,
        metavar="INDIVIDUALS;AXES;KEYPOINTS",
        help=textwrap.dedent(
            """\
            Optional subset of coordinates to load (quote or escape the spec).

              Example:
                'individual1,individual2;x,y;nose,neck,tail'

              Format:  INDIVIDUALS;AXES;KEYPOINTS
                INDIVIDUALS = comma‑separated individuals   | *
                AXES        = comma‑separated spatial axes  | *
                KEYPOINTS   = comma‑separated keypoints     | *

              Wildcards:
                *        include all items at that level

              If omitted, the entire dataset is loaded.

              NOTE: ';' and '*' are shell meta-characters, use single quotes
              on Unix‑like shells, double quotes on Windows, or escape them.
            """
        ),
    )
    parser.add_argument(
        "--rename_coords",
        type=str,
        metavar=(
            "OLD_INDIVIDUALS:NEW_INDIVIDUALS;"
            "OLD_AXES:NEW_AXES;"
            "OLD_KEYPOINTS:NEW_KEYPOINTS"
        ),
        help=textwrap.dedent(
            """\
            Optional mapping to rename coordinates (quote or escape the spec).

              Example:
                '*;*;nose:snout,tail:tailbase'

              Format:  INDIVIDUALS;AXES;KEYPOINTS
                INDIVIDUALS = comma‑separated individual mappings   | *
                AXES        = comma‑separated spatial axis mappings | *
                KEYPOINTS   = comma‑separated keypoint mappings     | *

              Each mapping is OLD_NAME:NEW_NAME. Use * to skip renaming at that level.

              If omitted, original dataset names are used.

              NOTE: ';' and '*' are shell meta-characters. Use single quotes
              on Unix‑like shells, double quotes on Windows, or escape them.
            """
        ),
    )
    parser.add_argument(
        "--window_size",
        default=200,
        type=int,
        help="Number of frames to consider at each time",
    )
    parser.add_argument(
        "--window_offset",
        default=0,
        type=int,
        help="Window offset for classification tasks",
    )
    parser.add_argument(
        "--fps_scaling", default=1.0, type=float, help="FPS scaling factor"
    )


def add_data_io_args(parser: argparse.ArgumentParser, data_help: str) -> None:
    """Add common data input/output arguments."""
    parser.add_argument("data_path", type=str, help=data_help)
    parser.add_argument(
        "--data_filter",
        type=str,
        help=textwrap.dedent(
            """\
            Comma-separated list of sub-keys to keep in the dataset
            For example, 'mouse001,mouse002' or 'approach'
            """
        ),
    )
    parser.add_argument(
        "--output_path", type=Path, default=Path("."), help="Output path"
    )
