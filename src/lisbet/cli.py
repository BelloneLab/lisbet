"""Command line interface for LISBET."""

import argparse
import importlib
import inspect
import logging
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(verbose: int = 0, log_level: Optional[str] = None) -> None:
    """Setup logging based on verbosity level."""
    if log_level:
        level = getattr(logging, log_level)
    else:
        level_map = {
            0: logging.WARNING,  # default
            1: logging.INFO,  # -v
            2: logging.DEBUG,  # -vv
        }
        level = level_map.get(verbose, logging.DEBUG)

    logging.basicConfig(level=level)


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
        default="maDLC",
        choices=["maDLC", "saDLC", "h5archive"],
        help="Keypoints dataset format",
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
        help="""Comma-separated list of sub-keys to keep in the dataset
        For example, 'mouse001,mouse002' or 'approach'""",
    )
    parser.add_argument(
        "--output_path", type=Path, default=Path("."), help="Output path"
    )


def lazy_load_handler(
    module_path: str, function_name: str, args: Dict[str, Any]
) -> None:
    """Dynamically import and call a command handler."""
    module = importlib.import_module(module_path)
    handler = getattr(module, function_name)

    # Filter arguments to match handler"s signature
    valid_args = [p.name for p in inspect.signature(handler).parameters.values()]
    filtered_args = {k: v for k, v in args.items() if k in valid_args}

    logging.debug("CLI arguments for %s: %s", function_name, filtered_args)

    # Execute handler
    handler(**filtered_args)


def configure_train_model_parser(parser: argparse.ArgumentParser) -> None:
    """Configure train_model command parser."""
    add_verbosity_args(parser)
    add_keypoints_args(parser)
    add_data_io_args(parser, "Keypoint data location")

    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument(
        "--task_ids",
        type=str,
        default="cfc",
        help="Task ID or comma-separated list of task IDS.",
    )
    parser.add_argument(
        "--task_data",
        type=str,
        help="i.e., cfc:[0],nwp:[0,1]",
    )
    parser.add_argument("--seed", default=1991, type=int, help="Base RNG seed")
    parser.add_argument(
        "--seed_test_split", type=int, help="RNG seed for test set split"
    )
    parser.add_argument("--run_id", type=str, help="ID of the run")
    parser.add_argument(
        "--data_augmentation", action="store_true", help="Enable data augmentation"
    )
    parser.add_argument(
        "--train_sample", type=float, help="Fraction of samples from the training set"
    )
    parser.add_argument(
        "--dev_sample", type=float, help="Fraction of samples from the dev set"
    )
    parser.add_argument(
        "--dev_ratio",
        type=float,
        help="Fraction of the training set to held out for hyper-parameters tuning",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        help="""Fraction of the training set to held out for testing,
        if a test set is not available in the dataset""",
    )

    # Model architecture
    parser.add_argument(
        "--num_layers", default=4, type=int, help="Number of transformer layers"
    )
    parser.add_argument(
        "--emb_dim", default=32, type=int, help="Dimension of embedding"
    )
    parser.add_argument(
        "--num_heads", default=4, type=int, help="Number of attention heads"
    )
    parser.add_argument(
        "--hidden_dim", default=128, type=int, help="Units in dense layers"
    )
    parser.add_argument(
        "--learning_rate", default=1e-4, type=float, help="Learning rate"
    )

    # Model weights and saving options
    parser.add_argument(
        "--load_backbone_weights",
        type=Path,
        help="Path to backbone weights from pretrained model",
    )
    parser.add_argument(
        "--freeze_backbone_weights",
        action="store_true",
        help="Freeze the backbone weights",
    )
    parser.add_argument(
        "--save_weights",
        default="last",
        choices=["all", "last"],
        help="Save 'best', 'all' or 'last' model weights",
    )
    parser.add_argument(
        "--save_history", action="store_true", help="Save model's training history"
    )

    # Performance options
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Run training in mixed precision mode",
    )
    parser.add_argument(
        "--compile_model", action="store_true", help="Compile training with XLA"
    )


def configure_annotate_behavior_parser(parser: argparse.ArgumentParser) -> None:
    """Configure annotate_behavior command parser."""
    add_verbosity_args(parser)
    add_keypoints_args(parser)
    add_data_io_args(parser, "Keypoint data location")
    parser.add_argument("model_path", type=Path, help="Path to model config")
    parser.add_argument("weights_path", type=Path, help="Path to model weights")


def configure_compute_embeddings_parser(parser: argparse.ArgumentParser) -> None:
    """Configure compute_embeddings command parser."""
    add_verbosity_args(parser)
    add_keypoints_args(parser)
    add_data_io_args(parser, "Keypoint data location")
    parser.add_argument("model_path", type=Path, help="Path to model config")
    parser.add_argument("weights_path", type=Path, help="Path to model weights")


def configure_export_embedder_parser(parser: argparse.ArgumentParser) -> None:
    """Configure export_embedder command parser."""
    add_verbosity_args(parser)
    parser.add_argument("model_path", type=Path, help="Path to model config")
    parser.add_argument("weights_path", type=Path, help="Path to model weights")
    parser.add_argument(
        "--output_path", type=Path, default=Path("."), help="Output path"
    )


def configure_reduce_dimensions_parser(parser: argparse.ArgumentParser) -> None:
    """Configure reduce_dimensions command parser."""
    add_verbosity_args(parser)
    add_data_io_args(parser, "Embedding data location")
    parser.add_argument(
        "--num_dims", type=int, default=2, help="Embedding space dimensions"
    )
    parser.add_argument(
        "--num_neighbors", type=int, default=60, help="Size of local neighborhood"
    )
    parser.add_argument("--sample_size", type=int, help="Size of random sample")
    parser.add_argument("--sample_seed", type=int, help="RNG seed for random sample")
    parser.add_argument("--umap_seed", type=int, help="RNG seed for UMAP")


def configure_segment_motifs_parser(parser: argparse.ArgumentParser) -> None:
    """Configure segment_motifs command parser."""
    add_verbosity_args(parser)
    add_data_io_args(parser, "Embedding data location")
    parser.add_argument(
        "--num_states", type=int, default=4, help="Number of hidden states"
    )
    parser.add_argument(
        "--num_iter", type=int, default=10, help="Number of iterations of EM"
    )
    parser.add_argument("--hmm_seed", type=int, help="RNG seed for HMM")


def configure_select_prototypes_parser(parser: argparse.ArgumentParser) -> None:
    """Configure select_prototypes command parser."""
    add_verbosity_args(parser)
    add_data_io_args(parser, "Annotation data location")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--hmm_range",
        type=int,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Range of HMM sizes",
    )
    group.add_argument(
        "--hmm_list", type=int, nargs="+", metavar="HMM_SIZE", help="List of HMM sizes"
    )
    parser.add_argument(
        "--method",
        default="min",
        choices=["min", "best"],
        help="Prototype selection algorithm",
    )
    parser.add_argument(
        "--frame_threshold",
        type=float,
        help="Minimum fraction of allocated frames for motifs to be kept",
    )
    parser.add_argument(
        "--bout_threshold",
        type=float,
        help="Minimum mean bout duration in seconds for motifs to be kept",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        help="Maximum Jaccard distance from the closest motif (pairs only)",
    )
    parser.add_argument(
        "--fps", type=float, help="Frames per second, used to compute bout duration"
    )


def configure_fetch_dataset_parser(parser: argparse.ArgumentParser) -> None:
    """Configure fetch_dataset command parser."""
    add_verbosity_args(parser)
    parser.add_argument(
        "dataset_id",
        choices=(
            "CalMS21_Task1",
            "CalMS21_Unlabeled",
            "SampleData",
        ),
        help="Dataset ID",
    )
    parser.add_argument(
        "--download_path",
        default=Path("datasets"),
        type=Path,
        help="Dataset destination path on the local machine",
    )


def configure_fetch_model_parser(parser: argparse.ArgumentParser) -> None:
    """Configure fetch_dataset command parser."""
    add_verbosity_args(parser)
    parser.add_argument(
        "model_id",
        choices=(
            "lisbet64x8-calms21UftT1",
            "lisbet64x8-calms21U-embedder",
        ),
        help="Model ID",
    )
    parser.add_argument(
        "--download_path",
        default=Path("models"),
        type=Path,
        help="Model destination path on the local machine",
    )


def main() -> None:
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
            "module": ".training",
            "function": "train",
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
            "module": ".modeling",
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
            "module": ".datasets",
            "function": "fetch_dataset",
            "configure": configure_fetch_dataset_parser,
        },
        "fetch_model": {
            "description": "Download public model from the internet",
            "module": ".modeling",
            "function": "fetch_model",
            "configure": configure_fetch_model_parser,
        },
    }

    # Add all subparsers
    for cmd_name, cmd_config in commands.items():
        cmd_parser = subparsers.add_parser(
            cmd_name,
            description=cmd_config["description"],
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    setup_logging(args.verbose, args.log_level)

    # Get command configuration
    cmd_config = commands[args.command]

    # Execute command with lazy loading
    lazy_load_handler(
        module_path=f"lisbet{cmd_config['module']}",
        function_name=cmd_config["function"],
        args=vars(args),
    )
