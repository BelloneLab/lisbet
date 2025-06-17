"""Command line interface for LISBET."""

import argparse
import importlib
import inspect
import logging
import textwrap
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Any, Optional


class RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    """
    Show default values **and** keep all newline / indent formatting
    exactly as written in help strings.
    """

    pass


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
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("movement.io.load_poses").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)


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
        default="multiclass",
        help=textwrap.dedent(
            """\
            Task ID or comma-separated list of task IDS.

            Valid (supervised) tasks are:
              - multiclass: Multi-Class Frame Classification
              - multilabel: Multi-Label Frame Classification

            Valid (self-supervised) tasks are:
              - cons: Group Consistency Classification
              - order: Temporal Order Classification
              - shift: Temporal Shift Classification
              - warp: Temporal Warp Classification

            Example:
              order,cons
            """
        ),
    )
    parser.add_argument(
        "--task_data",
        type=str,
        help="i.e., multiclass:[0],order:[0,1]",
    )
    parser.add_argument("--seed", default=1991, type=int, help="Base RNG seed")
    parser.add_argument("--run_id", type=str, help="ID of the run")
    parser.add_argument(
        "--data_augmentation", action="store_true", help="Enable data augmentation"
    )
    parser.add_argument(
        "--train_sample", type=float, help="Fraction of samples from the train set"
    )
    parser.add_argument(
        "--dev_sample", type=float, help="Fraction of samples from the dev set"
    )
    parser.add_argument(
        "--dev_ratio",
        type=float,
        help="Fraction of the train set to be held out as dev set",
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


def configure_annotate_behavior_parser(parser: argparse.ArgumentParser) -> None:
    """Configure annotate_behavior command parser."""
    add_verbosity_args(parser)
    add_keypoints_args(parser)
    add_data_io_args(parser, "Keypoint data location")
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
        "--min_n_components",
        type=int,
        default=2,
        help="Minimum number of hidden states",
    )
    parser.add_argument(
        "--max_n_components",
        type=int,
        default=32,
        help="Maximum number of hidden states",
    )
    parser.add_argument(
        "--num_iter", type=int, default=10, help="Number of iterations of EM"
    )
    parser.add_argument(
        "--fit_frac", type=float, help="Fraction of data to use for model fitting"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs, use -1 (default) for as many jobs as cores",
    )
    parser.add_argument("--hmm_seed", type=int, help="RNG seed for HMM")
    parser.add_argument("--pretrained_path", type=Path, help="Path to saved HMM models")


def configure_select_prototypes_parser(parser: argparse.ArgumentParser) -> None:
    """Configure select_prototypes command parser."""
    add_verbosity_args(parser)
    add_data_io_args(parser, "Annotation data location")
    parser.add_argument(
        "--min_n_components",
        type=int,
        default=6,
        help="Minimum number of hidden states",
    )
    parser.add_argument(
        "--max_n_components",
        type=int,
        default=32,
        help="Maximum number of hidden states",
    )
    parser.add_argument(
        "--method",
        default="best",
        choices=["min", "best"],
        help="Prototype selection algorithm",
    )
    parser.add_argument(
        "--frame_threshold",
        type=float,
        default=0.05,
        help="Minimum fraction of allocated frames for motifs to be kept",
    )
    parser.add_argument(
        "--bout_threshold",
        type=float,
        default=0.5,
        help="Minimum mean bout duration in seconds for motifs to be kept",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=0.6,
        help="Maximum Jaccard distance from the closest neighbor motif",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30,
        help="Frames per second, used to compute bout duration",
    )


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


def configure_model_info_parser(parser: argparse.ArgumentParser) -> None:
    """Configure model_info command parser."""
    add_verbosity_args(parser)
    parser.add_argument(
        "model_path", type=Path, help="Path to model config (YAML file)"
    )


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
    setup_logging(args.verbose, args.log_level)

    # Get command configuration
    cmd_config = commands[args.command]

    # Execute command with lazy loading (always absolute import from lisbet root)
    lazy_load_handler(
        module_path=f"lisbet{cmd_config['module']}",
        function_name=cmd_config["function"],
        args=vars(args),
    )
