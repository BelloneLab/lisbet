"""Fetch datasets and models from the lisbet repository."""

import argparse
from pathlib import Path

from lisbet.cli.common import add_verbosity_args


def configure_fetch_dataset_parser(parser: argparse.ArgumentParser) -> None:
    """Configure fetch_dataset command parser."""
    add_verbosity_args(parser)
    parser.add_argument(
        "dataset_id",
        choices=(
            "CalMS21_Task1",
            "CalMS21_Unlabeled",
            "MABe22_MouseTriplets",
            "SampleData",
        ),
        help="Dataset ID",
    )
    parser.add_argument(
        "--download_path",
        default=Path("."),
        type=Path,
        help="Dataset destination path on the local machine",
    )


def configure_fetch_model_parser(parser: argparse.ArgumentParser) -> None:
    """Configure fetch_dataset command parser."""
    add_verbosity_args(parser)
    parser.add_argument(
        "model_id",
        choices=(
            "lisbet32x4-calms21UftT1-classifier",
            "lisbet32x4-calms21U-embedder",
        ),
        help="Model ID",
    )
    parser.add_argument(
        "--download_path",
        default=Path("models"),
        type=Path,
        help="Model destination path on the local machine",
    )
