"""Train a model for keypoint classification and export the embedder."""

import argparse
import textwrap
from pathlib import Path

from lisbet.cli.common import add_data_io_args, add_keypoints_args, add_verbosity_args


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
        "--embedding_dim", default=32, type=int, help="Dimension of embedding"
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


def configure_export_embedder_parser(parser: argparse.ArgumentParser) -> None:
    """Configure export_embedder command parser."""
    add_verbosity_args(parser)
    parser.add_argument("model_path", type=Path, help="Path to model config")
    parser.add_argument("weights_path", type=Path, help="Path to model weights")
    parser.add_argument(
        "--output_path", type=Path, default=Path("."), help="Output path"
    )
