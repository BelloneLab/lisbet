"""Evaluate a model on keypoint data."""

import argparse
from pathlib import Path

from lisbet.cli.common import add_data_io_args, add_keypoints_args, add_verbosity_args


def configure_evaluate_model_parser(parser: argparse.ArgumentParser) -> None:
    """Configure evaluate_model command parser."""
    add_verbosity_args(parser)
    add_keypoints_args(parser)
    add_data_io_args(parser, "Keypoint data location")
    parser.add_argument("model_path", type=Path, help="Path to model config")
    parser.add_argument("weights_path", type=Path, help="Path to model weights")
    parser.add_argument(
        "--ignore_index",
        type=int,
        help="Ignore this label index in the evaluation (e.g., background label)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="multiclass",
        choices=["multiclass", "multilabel"],
        help="Classification mode",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for multilabel"
    )
