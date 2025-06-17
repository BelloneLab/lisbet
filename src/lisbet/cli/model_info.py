"""Model info command for LISBET Enhanced Backbone System.

This module implements the model_info command for displaying information
about model configurations and architectures.
"""

from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lisbet.config.models import (
    BackboneConfig,
    DataConfig,
    ExperimentConfig,
    TaskConfig,
    TrainingConfig,
)

# Create rich console for output
console = Console()


def _load_config_file(config_path: Path) -> ExperimentConfig:
    """Load configuration from file.

    Parameters
    ----------
    config_path : Path
        Path to the configuration file.

    Returns
    -------
    ExperimentConfig
        Loaded experiment configuration.

    Raises
    ------
    typer.Exit
        If the configuration file is invalid or missing.
    """
    if not config_path.exists():
        console.print(f"[red]Error: Configuration file not found: {config_path}[/red]")
        raise typer.Exit(1)

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

        # Convert dictionary to ExperimentConfig
        return ExperimentConfig(**config_dict)
    except Exception as e:
        console.print(f"[red]Error loading configuration file: {e}[/red]")
        raise typer.Exit(1) from e


def _format_model_size(embedding_dim: int, num_layers: int, num_heads: int) -> str:
    """Format model size information.

    Parameters
    ----------
    embedding_dim : int
        Embedding dimension.
    num_layers : int
        Number of transformer layers.
    num_heads : int
        Number of attention heads.

    Returns
    -------
    str
        Formatted model size string.
    """
    # Rough parameter estimation for transformer
    # This is a simplified calculation
    params_per_layer = (
        4 * embedding_dim * embedding_dim  # Attention weights
        + 4 * embedding_dim * embedding_dim  # Feed-forward weights
        + 6 * embedding_dim  # Biases and layer norms
    )
    total_params = (
        num_layers * params_per_layer + embedding_dim * 1000
    )  # Rough estimate

    if total_params >= 1e9:
        return f"~{total_params / 1e9:.1f}B parameters"
    elif total_params >= 1e6:
        return f"~{total_params / 1e6:.1f}M parameters"
    elif total_params >= 1e3:
        return f"~{total_params / 1e3:.1f}K parameters"
    else:
        return f"{total_params} parameters"


def model_info(
    config_file: Annotated[Path, typer.Argument(help="Configuration file to analyze")],
    detailed: Annotated[
        bool,
        typer.Option(
            "--detailed",
            "-d",
            help="Show detailed model architecture information",
        ),
    ] = False,
) -> None:
    """Show information about a LISBET model config file.

    Display detailed information about a model's architecture, training
    configuration, and expected data requirements.

    Examples:
        betman model_info my_config.yml
        betman model_info transformer_config.json --detailed
    """
    # Load configuration
    config = _load_config_file(config_file)

    # Display basic model information
    console.print(
        Panel(
            f"[bold cyan]{config.experiment_name}[/bold cyan]",
            title="Model Configuration",
            expand=False,
        )
    )

    # Model architecture table
    arch_table = Table(title="Model Architecture")
    arch_table.add_column("Component", style="cyan")
    arch_table.add_column("Configuration", style="green")

    # Backbone info
    backbone = config.backbone
    arch_table.add_row("Backbone Type", backbone.type.upper())
    arch_table.add_row("Feature Dimension", str(backbone.feature_dim))
    arch_table.add_row("Embedding Dimension", str(backbone.embedding_dim))
    arch_table.add_row("Hidden Dimension", str(backbone.hidden_dim))
    arch_table.add_row("Number of Layers", str(backbone.num_layers))
    arch_table.add_row("Number of Heads", str(backbone.num_heads))
    arch_table.add_row("Max Sequence Length", str(backbone.max_length))
    arch_table.add_row("Dropout Rate", str(backbone.dropout))
    arch_table.add_row("Activation Function", backbone.activation.upper())
    arch_table.add_row(
        "Estimated Size",
        _format_model_size(
            backbone.embedding_dim, backbone.num_layers, backbone.num_heads
        ),
    )

    console.print(arch_table)

    # Task configuration
    if config.tasks:
        task_table = Table(title="Task Configuration")
        task_table.add_column("Task ID", style="cyan")
        task_table.add_column("Type", style="white")
        task_table.add_column("Classes", style="green")
        task_table.add_column("Metrics", style="yellow")

        for task in config.tasks:
            classes_str = str(task.num_classes) if task.num_classes else "N/A"
            metrics_str = ", ".join(task.metrics) if task.metrics else "None"
            task_table.add_row(
                task.task_id,
                task.task_type,
                classes_str,
                metrics_str,
            )

        console.print(task_table)

    # Training configuration
    training_table = Table(title="Training Configuration")
    training_table.add_column("Parameter", style="cyan")
    training_table.add_column("Value", style="green")

    training = config.training
    training_table.add_row("Epochs", str(training.epochs))
    training_table.add_row("Learning Rate", f"{training.learning_rate:.2e}")
    training_table.add_row("Batch Size", str(training.batch_size))
    training_table.add_row("Random Seed", str(training.seed))
    training_table.add_row("Mixed Precision", str(training.mixed_precision))

    console.print(training_table)

    # Data configuration (if detailed)
    if detailed:
        data_table = Table(title="Data Configuration")
        data_table.add_column("Parameter", style="cyan")
        data_table.add_column("Value", style="green")

        data = config.data
        data_table.add_row("Window Size", str(data.window_size))
        data_table.add_row("Window Offset", str(data.window_offset))
        data_table.add_row("Batch Size", str(data.batch_size))
        data_table.add_row("Number of Workers", str(data.num_workers))
        data_table.add_row("Pin Memory", str(data.pin_memory))

        # Data paths
        if data.train_paths:
            data_table.add_row("Train Paths", "\n".join(data.train_paths))
        if data.val_paths:
            data_table.add_row("Val Paths", "\n".join(data.val_paths))
        if data.test_paths:
            data_table.add_row("Test Paths", "\n".join(data.test_paths))

        # Input features
        if data.input_features:
            features_str = ", ".join(data.input_features)
            if len(features_str) > 50:
                features_str = features_str[:47] + "..."
            data_table.add_row("Input Features", features_str)

        console.print(data_table)

    # Experiment metadata (if detailed)
    if config.notes or config.tags or detailed:
        meta_table = Table(title="Experiment Metadata")
        meta_table.add_column("Property", style="cyan")
        meta_table.add_column("Value", style="green")

        meta_table.add_row("Output Directory", config.output_dir)
        if config.run_id:
            meta_table.add_row("Run ID", config.run_id)
        meta_table.add_row("Seed", str(config.seed))

        if config.tags:
            meta_table.add_row("Tags", ", ".join(config.tags))
        if config.notes:
            notes_display = (
                config.notes[:100] + "..." if len(config.notes) > 100 else config.notes
            )
            meta_table.add_row("Notes", notes_display)

        console.print(meta_table)
