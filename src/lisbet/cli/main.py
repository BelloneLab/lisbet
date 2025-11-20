"""LISBET Command Line Interface (CLI)"""

import argparse
import importlib
import inspect
import logging
from importlib.metadata import version as get_version
from typing import Any

from lisbet.cli.commands import (
    configure_annotate_behavior_parser,
    configure_compute_embeddings_parser,
    configure_evaluate_model_parser,
    configure_export_embedder_parser,
    configure_fetch_dataset_parser,
    configure_fetch_model_parser,
    configure_model_info_parser,
    configure_reduce_dimensions_parser,
    configure_segment_motifs_parser,
    configure_select_prototypes_parser,
    configure_train_model_parser,
)
import torch


class RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    """
    Show default values **and** keep all newline / indent formatting
    exactly as written in help strings.
    """

    pass


def lazy_load_handler(
    module_path: str, function_name: str, args: dict[str, Any]
) -> None:
    """Dynamically import and call a command handler."""
    # Always use absolute imports from the lisbet root package
    if not module_path.startswith("lisbet"):
        module_path = f"lisbet{module_path}"
    module = importlib.import_module(module_path)
    handler = getattr(module, function_name)

    # Filter arguments to match handler"s signature
    valid_args = [p.name for p in inspect.signature(handler).parameters.values()]
    filtered_args = {k: v for k, v in args.items() if k in valid_args}

    logging.debug("CLI arguments for %s: %s", function_name, filtered_args)

    # Execute handler
    handler(**filtered_args)


def app() -> None:
    """Main entry point for betman CLI."""
    parser = argparse.ArgumentParser(
        description="betman - Command line interface for LISBET",
        add_help=False,
    )

    # Create basic arguments
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {get_version('lisbet')}",
        help="Show LISBET's version number and exit",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit",
    )

    # Create subcommands
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)

    # Configure command parsers
    commands = {
        "train_model": {
            "description": "Train a new classification model",
            "module": ".cli.commands.train",
            "function": "train_model",
            "configure": configure_train_model_parser,
        },
        "annotate_behavior": {
            "description": "Annotate behaviors using trained model",
            "module": ".inference",
            "function": "annotate_behavior",
            "configure": configure_annotate_behavior_parser,
        },
        "compute_embeddings": {
            "description": "Compute behavioral embeddings using trained model",
            "module": ".inference",
            "function": "compute_embeddings",
            "configure": configure_compute_embeddings_parser,
        },
        "export_embedder": {
            "description": "Export embedding model",
            "module": ".io",
            "function": "export_embedder",
            "configure": configure_export_embedder_parser,
        },
        "reduce_dimensions": {
            "description": "Perform dimensionality reduction with UMAP",
            "module": ".unsupervised",
            "function": "reduce_umap",
            "configure": configure_reduce_dimensions_parser,
        },
        "segment_motifs": {
            "description": "Segment behavioral motifs using HMM",
            "module": ".unsupervised",
            "function": "segment_hmm",
            "configure": configure_segment_motifs_parser,
        },
        "select_prototypes": {
            "description": "Select prototype behaviors from multiple annotations",
            "module": ".postprocessing",
            "function": "select_prototypes",
            "configure": configure_select_prototypes_parser,
        },
        "fetch_dataset": {
            "description": "Download public dataset from the internet",
            "module": ".hub",
            "function": "fetch_dataset",
            "configure": configure_fetch_dataset_parser,
        },
        "fetch_model": {
            "description": "Download public model from the internet",
            "module": ".hub",
            "function": "fetch_model",
            "configure": configure_fetch_model_parser,
        },
        "model_info": {
            "description": "Show information about a LISBET model config file",
            "module": ".modeling",
            "function": "model_info",
            "configure": configure_model_info_parser,
        },
        "evaluate_model": {
            "description": "Evaluate a classification model on a labeled dataset",
            "module": ".evaluation",
            "function": "evaluate",
            "configure": configure_evaluate_model_parser,
        },
    }

    # Add all subparsers
    for cmd_name, cmd_config in commands.items():
        cmd_parser = subparsers.add_parser(
            cmd_name,
            description=cmd_config["description"],
            add_help=False,
            formatter_class=RawDefaultsHelpFormatter,
        )
        cmd_parser.add_argument(
            "-h",
            "--help",
            action="help",
            default=argparse.SUPPRESS,
            help="Show this help message and exit",
        )
        cmd_config["configure"](cmd_parser)

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    if args.log_level:
        level = getattr(logging, args.log_level)
    else:
        level_map = {
            0: logging.WARNING,  # default
            1: logging.INFO,  # -v
            2: logging.DEBUG,  # -vv
        }
        level = level_map.get(args.verbose, logging.DEBUG)

    logging.basicConfig(level=level)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("movement.io.load_poses").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)

    # Get command configuration
    cmd_config = commands[args.command]

    # Temporary workaround for new-style commands
    if cmd_config["function"] == "train_model":
        module_path = f"lisbet{cmd_config['module']}"
        function_name = cmd_config["function"]
        module = importlib.import_module(module_path)
        handler = getattr(module, function_name)
        handler(dict(args._get_kwargs()))

    else:
        # Fallback
        lazy_load_handler(
            module_path=f"lisbet{cmd_config['module']}",
            function_name=cmd_config["function"],
            args=vars(args),
        )

torch.set_float32_matmul_precision('medium')