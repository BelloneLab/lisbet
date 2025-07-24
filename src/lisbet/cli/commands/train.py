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
        "--learning_rate", default=1e-4, type=float, help="Learning rate"
    )
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
        "--backbone_preset", default="transformer-base", type=str, help="Backbone type"
    )
    parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        action="append",
        help="Override config values, e.g. --set backbone.num_layers=4",
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

    # Miscellaneous
    parser.add_argument("--dry_run", action="store_true", help="Print config and exit")


def configure_export_embedder_parser(parser: argparse.ArgumentParser) -> None:
    """Configure export_embedder command parser."""
    add_verbosity_args(parser)
    parser.add_argument("model_path", type=Path, help="Path to model config")
    parser.add_argument("weights_path", type=Path, help="Path to model weights")
    parser.add_argument(
        "--output_path", type=Path, default=Path("."), help="Output path"
    )


def train_model(kwargs):
    """Train a model for keypoint classification."""
    # Lazy imports to avoid unnecessary dependencies when not training
    from lisbet.config.overrides import apply_overrides
    from lisbet.config.presets import BACKBONE_PRESETS
    from lisbet.config.schemas import (
        DataConfig,
        ExperimentConfig,
        ModelConfig,
        TrainingConfig,
    )
    from lisbet.training import train

    # Configure backbone
    preset_name = kwargs.get("backbone_preset", "transformer-base")
    if preset_name not in BACKBONE_PRESETS:
        raise ValueError(f"Unknown backbone preset: {preset_name}")
    backbone_config = BACKBONE_PRESETS[preset_name]

    # Parse overrides from --set backbone.*=...
    overrides = {}
    for override in kwargs.get("set", []) or []:
        if override.startswith("backbone."):
            keyval = override[len("backbone.") :].split("=", 1)
            if len(keyval) == 2:
                key, val = keyval
                overrides[key] = val

    if overrides:
        backbone_config = apply_overrides(backbone_config, overrides)

    # Configure data
    data_config = DataConfig(
        data_path=kwargs["data_path"],
        data_format=kwargs["data_format"],
        data_scale=kwargs.get("data_scale"),
        data_filter=kwargs.get("data_filter"),
        select_coords=kwargs.get("select_coords"),
        rename_coords=kwargs.get("rename_coords"),
        window_size=kwargs["window_size"],
        window_offset=kwargs["window_offset"],
        fps_scaling=kwargs["fps_scaling"],
        dev_ratio=kwargs.get("dev_ratio"),
        train_sample=kwargs.get("train_sample"),
        dev_sample=kwargs.get("dev_sample"),
    )

    # Configure tasks
    task_ids_list = kwargs["task_ids"].split(",")
    # NOTE: For now we keep the task_data as a string, but it could be parsed into a
    #       dict to simplify `split_multi_records`. Or even better, use a TaskConfig
    #       class to handle task-specific configurations.
    task_data = kwargs["task_data"]

    # Configure model
    model_config = ModelConfig(
        model_id=kwargs["run_id"],
        backbone=backbone_config,
        out_heads={task_id: {} for task_id in task_ids_list},
        input_features={},
        window_size=kwargs["window_size"],
        window_offset=kwargs["window_offset"],
    )

    # Configure training
    training_config = TrainingConfig(
        epochs=kwargs["epochs"],
        batch_size=kwargs["batch_size"],
        learning_rate=kwargs["learning_rate"],
        data_augmentation=kwargs["data_augmentation"],
        save_weights=kwargs["save_weights"],
        save_history=kwargs["save_history"],
        mixed_precision=kwargs["mixed_precision"],
        freeze_backbone_weights=kwargs["freeze_backbone_weights"],
        load_backbone_weights=kwargs.get("load_backbone_weights"),
    )

    # Create experiment configuration
    experiment_config = ExperimentConfig(
        run_id=kwargs["run_id"],
        seed=kwargs["seed"],
        model=model_config,
        training=training_config,
        data=data_config,
        task_ids_list=task_ids_list,
        task_data=task_data,
        output_path=kwargs["output_path"],
    )

    if kwargs.get("dry_run"):
        # If dry run, just print the configuration
        print(experiment_config)

    else:
        # Train the model
        train(experiment_config)
