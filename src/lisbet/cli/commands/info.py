"""Module for handling model information retrieval."""

import argparse
from pathlib import Path

from lisbet.cli.common import add_verbosity_args


def configure_model_info_parser(parser: argparse.ArgumentParser) -> None:
    """Configure model_info command parser."""
    add_verbosity_args(parser)
    parser.add_argument(
        "model_path", type=Path, help="Path to model config (YAML file)"
    )
