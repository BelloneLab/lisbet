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
    list[dict] or None
        List of dictionaries with augmentation configs, or None if None/empty.

    Examples
    --------
    >>> parse_data_augmentation("all_perm_id")
    [{'name': 'all_perm_id', 'p': 1.0}]

    >>> parse_data_augmentation("all_perm_id:p=0.5,blk_perm_id:frac=0.3")
    [{'name': 'all_perm_id', 'p': 0.5}, {'name': 'blk_perm_id', 'p': 1.0, 'frac': 0.3}]

    >>> parse_data_augmentation("rotation:p=0.5:max_angle=30")
    [{'name': 'rotation', 'p': 0.5, 'max_angle': 30.0}]
    """
    if not aug_string:
        return None

    augmentations = []
    for aug_spec in aug_string.split(","):
        aug_spec = aug_spec.strip()
        if not aug_spec:
            continue

        parts = aug_spec.split(":")
        aug_config = {"name": parts[0].strip(), "p": 1.0}

        # Parse parameters
        for param in parts[1:]:
            key, _, value = param.partition("=")
            key = key.strip()
            value = value.strip()

            if not key or not value:
                raise ValueError(f"Invalid parameter format in '{aug_spec}': {param}")

            try:
                aug_config[key] = float(value)
            except ValueError as e:
                raise ValueError(
                    f"Invalid parameter value in '{aug_spec}': {key}={value}"
                ) from e

        augmentations.append(aug_config)

    return augmentations if augmentations else None


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
              - geom : Geometric Consistency Classification

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
        "--window_sampling",
        choices=["any", "inside"],
        default="any",
        help=textwrap.dedent(
            """\
            Sampling of the windows of the self-supervised tasks (cons, order, shift, warp, geom).

            - any:    original behaviour, a window may extend past the edges of its record and is
                      zero padded there.
            - inside: only draw samples whose windows lie entirely inside their record. Removes the
                      zero padding at record edges, which otherwise leaks the label of the order,
                      shift and cons tasks. The sign of the shift is drawn 50/50 and the geom views
                      follow the same rule. Records shorter than the window are never sampled.
            """
        ),
    )
    parser.add_argument(
        "--data_augmentation",
        type=str,
        help=textwrap.dedent(
            """\
            Data augmentation techniques to apply, comma-separated.
            Each augmentation can have optional parameters specified with colons.

            NOTE: Use the --data_augmentation= format (with equals sign) to clearly
                  separate the argument from its value. This makes the syntax
                  unambiguous and prevents shell interpretation of special characters.

            Valid options are:
                - all_perm_id: Randomly permute identities of individuals, applied
                               consistently across all frames in a window.
                - all_perm_ax: Randomly permute x, y (and z) axes, applied consistently
                               across all frames in a window.
                - blk_perm_id: Randomly permute identities of individuals, applied
                               to a contiguous block of frames within a window.

                - gauss_jitter: Add independent N(0,sigma) noise to every coordinate
                                in a selected window.

                - kp_ablation: Use Bernoulli(pB) to select (keypoint, individual)
                               pairs, then set all of their coordinates to zero across
                               the selected window. Simulates sustained occlusions or
                               tracking failures.

                - all_translate: Randomly translate all individuals together in x,y
                            consistently across all frames in a window. 
                            Translation computed to keep all keypoints in [0,1] bounds.
                            Provides invariance to location within frame.
                - all_mirror_x: Randomly mirror horizontally around x=0.5.
                            Consistently across all frames in a window.
                            Provides invariance to lateral orientation.
                - all_zoom: Randomly zoom/dezoom around center (0.5, 0.5).
                        Consistently across all frames in a window.
                        Scale computed to keep all keypoints in [0,1] bounds.
                        Provides invariance to depth/distance.
                - rotation: Randomly rotate keypoint coordinates around the center
                            of the normalized [0,1] space. Supports 2D and 3D
                            (auto-detected from data).


            Parameters (optional):
                - p=<float>: Probability of applying the entire transformation
                             (default: 1.0).
                - pB=<float>: Per-(keypoint, individual) ablation probability;
                              required for kp_ablation.
                - frac=<float>: Nominal block size fraction for blk_perm_id
                                (default: 0.5).
                - sigma=<float>: Noise standard deviation for gauss_jitter
                                 (default: 0.01).
                - max_angle=<float>: Maximum rotation angle in degrees for rotation
                                     (default 180.0). Angle sampled from
                                     [-max_angle, +max_angle].


            Examples:
                --data_augmentation="all_perm_id"
                --data_augmentation="all_perm_id:p=0.5"
                --data_augmentation="all_perm_id:p=0.5,blk_perm_id:p=0.3:frac=0.2"
                --data_augmentation="all_perm_ax:p=0.7,blk_perm_id:frac=0.3"
                --data_augmentation="gauss_jitter:p=0.02:sigma=0.01"
                --data_augmentation="kp_ablation:p=0.05:pB=0.01"
                --data_augmentation="all_perm_id:p=0.5,kp_ablation:p=0.03:pB=0.01"
                --data_augmentation="rotation:p=0.5:max_angle=30"

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
        help=(
            "Override config values, e.g. --set backbone.num_layers=4 "
            "or --set task.geom.temperature=0.1"
        ),
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
        DataAugmentationPipeline,
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

    # Parse overrides from --set task.<task_id>.<param>=...
    task_configs: dict[str, dict] = {}
    for override in kwargs.get("set", []) or []:
        if override.startswith("task."):
            path, _, val = override[len("task.") :].partition("=")
            task_id, _, param = path.partition(".")
            if task_id and param and val:
                task_configs.setdefault(task_id, {})[param] = val

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

    if parsed_augmentation is not None:
        pipeline = DataAugmentationPipeline(augmentations=parsed_augmentation)
        validated_augmentations = pipeline.augmentations
    else:
        validated_augmentations = parsed_augmentation

    # Update kwargs with parsed augmentation and per-task overrides
    kwargs_for_training = {
        **kwargs,
        "data_augmentation": validated_augmentations,
        "task_configs": task_configs,
    }

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
