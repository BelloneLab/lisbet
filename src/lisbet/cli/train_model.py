"""Train model command for LISBET Enhanced Backbone System.

This module implements the train_model command with support for transformer
presets and custom configuration files.
"""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from lisbet.config import (
    create_experiment_config,
    get_transformer_backbone_presets,
)
from lisbet.config.models import apply_overrides

# Create rich console for output
console = Console()


def _validate_data_directory(data_dir: Path) -> None:
    """Validate that data directory exists and is accessible.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory.

    Raises
    ------
    typer.Exit
        If the data directory is invalid.
    """
    if not data_dir.exists():
        console.print(f"[red]Error: Data directory not found: {data_dir}[/red]")
        raise typer.Exit(1)

    if not data_dir.is_dir():
        console.print(f"[red]Error: Data path is not a directory: {data_dir}[/red]")
        raise typer.Exit(1)


def _show_training_info(config, data_dir: Path) -> None:
    """Display training configuration information.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    data_dir : Path
        Data directory path.
    """
    table = Table(title="Training Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Experiment Name", config.experiment_name)
    table.add_row("Backbone Type", config.backbone.type)
    table.add_row("Model Size", f"{config.backbone.embedding_dim}d")
    table.add_row("Number of Layers", str(config.backbone.num_layers))
    table.add_row("Number of Heads", str(config.backbone.num_heads))
    table.add_row("Data Directory", str(data_dir))
    table.add_row("Epochs", str(config.training.epochs))
    table.add_row("Batch Size", str(config.training.batch_size))
    table.add_row("Learning Rate", str(config.training.learning_rate))
    table.add_row("Output Directory", config.output_dir)

    console.print(table)


def train_model(
    config_or_preset: Annotated[
        str,
        typer.Argument(
            help=(
                "Configuration file path or preset name "
                "(transformer-small, transformer-base, transformer-large)"
            )
        ),
    ],
    data_dir: Annotated[
        Path, typer.Argument(help="Path to the training data directory")
    ],
    experiment_name: Annotated[
        Optional[str],
        typer.Option("--name", "-n", help="Experiment name (overrides config)"),
    ] = None,
    epochs: Annotated[
        Optional[int],
        typer.Option("--epochs", "-e", help="Number of training epochs"),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option("--batch-size", "-b", help="Training batch size"),
    ] = None,
    learning_rate: Annotated[
        Optional[float], typer.Option("--lr", help="Learning rate")
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", "-o", help="Output directory for experiments"),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show configuration without training")
    ] = False,
) -> None:
    """Train a new classification model.

    Train a model using either a built-in transformer preset or a custom
    configuration file. Supports overriding parameters via command line.

    Examples:
        betman train_model transformer-base ./data
        betman train_model my_config.yml ./data --epochs 50
        betman train_model transformer-large ./data --name my_experiment
    """
    # Validate data directory
    _validate_data_directory(data_dir)

    # Determine if input is a preset or config file
    config_path = Path(config_or_preset)
    if config_path.exists() and config_path.is_file():
        # Load from configuration file
        import yaml

        from lisbet.config.models import (
            BackboneConfig,
            DataConfig,
            ExperimentConfig,
            TaskConfig,
            TrainingConfig,
        )

        try:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)

            # Convert nested dictionaries to proper dataclass instances
            if "data" in config_dict and isinstance(config_dict["data"], dict):
                config_dict["data"] = DataConfig(**config_dict["data"])

            if "backbone" in config_dict and isinstance(config_dict["backbone"], dict):
                config_dict["backbone"] = BackboneConfig(**config_dict["backbone"])

            if (
                "tasks" in config_dict
                and hasattr(config_dict["tasks"], "__iter__")
                and not isinstance(config_dict["tasks"], (str, dict))
            ):
                config_dict["tasks"] = [
                    TaskConfig(**task) if isinstance(task, dict) else task
                    for task in config_dict["tasks"]
                ]

            if "training" in config_dict and isinstance(config_dict["training"], dict):
                config_dict["training"] = TrainingConfig(**config_dict["training"])

            config = ExperimentConfig(**config_dict)

        except Exception as e:
            console.print(f"[red]Error loading configuration file: {e}[/red]")
            raise typer.Exit(1) from e

    else:
        # Use preset
        available_presets = get_transformer_backbone_presets()
        if config_or_preset not in available_presets:
            console.print(f"[red]Error: Unknown preset '{config_or_preset}'[/red]")
            console.print(f"Available presets: {', '.join(available_presets.keys())}")
            raise typer.Exit(1)

        # Create experiment configuration from preset
        exp_name = experiment_name or f"{config_or_preset}_experiment"
        config = create_experiment_config(
            experiment_name=exp_name,
            backbone_preset=config_or_preset,
        )

    # Apply command line overrides
    overrides = {}
    if experiment_name is not None:
        overrides["experiment_name"] = experiment_name
    if epochs is not None:
        overrides["training.epochs"] = epochs
    if batch_size is not None:
        overrides["training.batch_size"] = batch_size
    if learning_rate is not None:
        overrides["training.learning_rate"] = learning_rate
    if output_dir is not None:
        overrides["output_dir"] = str(output_dir)

    if overrides:
        config = apply_overrides(config, overrides)

    # Update data paths
    config.data.train_paths = [str(data_dir / "train")]
    config.data.val_paths = [str(data_dir / "val")]
    if (data_dir / "test").exists():
        config.data.test_paths = [str(data_dir / "test")]

    # Show configuration
    _show_training_info(config, data_dir)

    if dry_run:
        console.print("[yellow]Dry run - training not started[/yellow]")
        return

    # Start training
    console.print("[bold green]Starting training...[/bold green]")

    try:
        # Import training module lazily
        from lisbet.training import train

        # Convert config to dict for training function
        config_dict = {
            "experiment_name": config.experiment_name,
            "data_dir": str(data_dir),
            "backbone_config": config.backbone.__dict__,
            "training_config": config.training.__dict__,
            "output_dir": config.output_dir,
        }

        train(config_dict)
        console.print("[bold green]Training completed successfully![/bold green]")

    except ImportError:
        console.print("[red]Error: Training module not available[/red]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        raise typer.Exit(1) from e
