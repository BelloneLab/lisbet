"""Train a model for keypoint classification and export the embedder."""

import argparse
import textwrap
from pathlib import Path

from lisbet.cli.common import add_data_io_args, add_keypoints_args, add_verbosity_args


def parse_data_augmentation(aug_string):
    """Parse data augmentation string into list of DataAugmentationConfig objects.

    Parameters
    ----------
    aug_string : str or None
        Comma-separated augmentation specifications, each with optional parameters.
        Format: name:p=value:frac=value
        Example: "all_perm_id:p=0.5,blk_perm_id:p=0.3:frac=0.2"

    Returns
    -------
    list[dict] or bool
        List of dictionaries with augmentation configs, or False if None/empty.

    Examples
    --------
    >>> parse_data_augmentation("all_perm_id")
    [{'name': 'all_perm_id', 'p': 1.0}]

    >>> parse_data_augmentation("all_perm_id:p=0.5,blk_perm_id:frac=0.3")
    [{'name': 'all_perm_id', 'p': 0.5}, {'name': 'blk_perm_id', 'p': 1.0, 'frac': 0.3}]
    """
    if not aug_string:
        return False

    augmentations = []
    for aug_spec in aug_string.split(","):
        parts = aug_spec.strip().split(":")
        if not parts[0]:
            continue

        aug_config = {"name": parts[0].strip()}

        # Parse parameters
        for param in parts[1:]:
            if "=" in param:
                key, value = param.split("=", 1)
                key = key.strip()
                value = value.strip()
                try:
                    aug_config[key] = float(value)
                except ValueError as exc:
                    raise ValueError from exc(
                        f"Invalid parameter value in '{aug_spec}': {key}={value}"
                    )

        # Set defaults if not specified
        if "p" not in aug_config:
            aug_config["p"] = 1.0

        augmentations.append(aug_config)

    return augmentations if augmentations else False


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
        "--data_augmentation",
        type=str,
        help=textwrap.dedent(
            """\
            Data augmentation techniques to apply, comma-separated.
            Each augmentation can have optional parameters specified with colons.

            Valid options are:
                - all_perm_id: Randomly permute identities of individuals, applied
                               consistently across all frames in a window.
                - all_perm_ax: Randomly permute x, y (and z) axes, applied consistently
                               across all frames in a window.
                - blk_perm_id: Randomly permute identities of individuals, applied
                               to a contiguous block of frames within a window.

            Parameters (optional):
                - p=<float>: Probability of applying the transformation (default: 1.0)
                - frac=<float>: For blk_perm_id only, fraction of frames to permute
                                (default: 0.5)

            Examples:
                --data_augmentation all_perm_id
                --data_augmentation all_perm_id:p=0.5
                --data_augmentation all_perm_id:p=0.5,blk_perm_id:p=0.3:frac=0.2
                --data_augmentation all_perm_ax:p=0.7,blk_perm_id:frac=0.3
            """
        ),
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
        "--head_type",
        type=str,
        choices=["mlp", "linear"],
        default="mlp",
        help="Classification head type",
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
    from pydantic import TypeAdapter

    from lisbet.config.presets import BACKBONE_PRESETS
    from lisbet.config.schemas import (
        BackboneConfig,
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
    backbone_config_dict = BACKBONE_PRESETS[preset_name]

    # Set max_length for transformer backbones to window_size
    if backbone_config_dict.get("type") == "transformer":
        backbone_config_dict["max_length"] = kwargs.get("window_size")

    # Parse overrides from --set backbone.*=...
    overrides = {}
    for override in kwargs.get("set", []) or []:
        if override.startswith("backbone."):
            keyval = override[len("backbone.") :].split("=", 1)
            if len(keyval) == 2:
                key, val = keyval
                overrides[key] = val
    backbone_config_dict.update(overrides)

    # Create backbone config
    adapter = TypeAdapter(BackboneConfig)
    backbone_config = adapter.validate_python(backbone_config_dict)

    # Configure data
    data_config = DataConfig.model_validate(kwargs)

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

    # Parse and configure data augmentation
    aug_string = kwargs.get("data_augmentation")
    parsed_augmentation = parse_data_augmentation(aug_string)

    # Update kwargs with parsed augmentation
    kwargs_for_training = {**kwargs, "data_augmentation": parsed_augmentation}

    # Configure training
    training_config = TrainingConfig.model_validate(kwargs_for_training)

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
