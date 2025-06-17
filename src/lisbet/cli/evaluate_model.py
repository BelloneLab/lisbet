"""Evaluate model command for LISBET Enhanced Backbone System.

This module implements the evaluate_model command for evaluating
trained models on labeled test data.
"""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

# Create rich console for output
console = Console()


def _validate_model_path(model_path: Path) -> None:
    """Validate that model path exists and is accessible.

    Parameters
    ----------
    model_path : Path
        Path to the model file or directory.

    Raises
    ------
    typer.Exit
        If the model path is invalid.
    """
    if not model_path.exists():
        console.print(f"[red]Error: Model path not found: {model_path}[/red]")
        raise typer.Exit(1)


def _validate_data_path(data_path: Path) -> None:
    """Validate that data path exists and is accessible.

    Parameters
    ----------
    data_path : Path
        Path to the data file or directory.

    Raises
    ------
    typer.Exit
        If the data path is invalid.
    """
    if not data_path.exists():
        console.print(f"[red]Error: Data path not found: {data_path}[/red]")
        raise typer.Exit(1)


def evaluate_model(
    model_path: Annotated[
        Path, typer.Argument(help="Path to trained model file or directory")
    ],
    data_path: Annotated[Path, typer.Argument(help="Path to labeled test data")],
    output_path: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output path for evaluation results",
        ),
    ] = None,
    weights_path: Annotated[
        Optional[Path],
        typer.Option(
            "--weights",
            "-w",
            help="Path to model weights file (if separate from model)",
        ),
    ] = None,
    data_format: Annotated[
        str,
        typer.Option(
            "--data-format",
            "-f",
            help="Input data format (movement, sleap, dlc, etc.)",
        ),
    ] = "movement",
    data_scale: Annotated[
        Optional[str],
        typer.Option(
            "--data-scale",
            help="Data scaling method",
        ),
    ] = None,
    data_filter: Annotated[
        Optional[str],
        typer.Option(
            "--data-filter",
            help="Data filtering method",
        ),
    ] = None,
    window_size: Annotated[
        int,
        typer.Option(
            "--window-size",
            help="Window size for processing",
        ),
    ] = 200,
    window_offset: Annotated[
        int,
        typer.Option(
            "--window-offset",
            help="Window offset",
        ),
    ] = 0,
    fps_scaling: Annotated[
        float,
        typer.Option(
            "--fps-scaling",
            help="FPS scaling factor",
        ),
    ] = 1.0,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Batch size for evaluation",
        ),
    ] = 128,
    device: Annotated[
        Optional[str],
        typer.Option(
            "--device",
            "-d",
            help="Device to use (cpu, cuda, auto)",
        ),
    ] = None,
    save_predictions: Annotated[
        bool,
        typer.Option(
            "--save-predictions",
            help="Save individual predictions",
        ),
    ] = False,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            help="Verbosity level",
        ),
    ] = 1,
) -> None:
    """Evaluate a classification model on a labeled dataset.

    Compute classification metrics and performance statistics for a trained
    model on a labeled dataset.

    Examples:
        betman evaluate_model ./model.ckpt ./test_data.h5
        betman evaluate_model ./model ./test_data --batch-size 64
        betman evaluate_model ./model.ckpt ./test_data --save-predictions
    """
    # Validate inputs
    _validate_model_path(model_path)
    _validate_data_path(data_path)

    # Set default output path
    if output_path is None:
        if data_path.is_file():
            output_path = data_path.parent / f"{data_path.stem}_evaluation.json"
        else:
            output_path = data_path / "evaluation.json"

    # Set default weights path
    if weights_path is None:
        if model_path.is_dir():
            # Look for common weight file names
            for weights_name in ["weights.pt", "model.pt", "checkpoint.ckpt"]:
                potential_weights = model_path / weights_name
                if potential_weights.exists():
                    weights_path = potential_weights
                    break
        else:
            weights_path = model_path

    # Show configuration
    table = Table(title="Evaluation Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model Path", str(model_path))
    table.add_row("Weights Path", str(weights_path))
    table.add_row("Data Path", str(data_path))
    table.add_row("Output Path", str(output_path))
    table.add_row("Data Format", data_format)
    table.add_row("Window Size", str(window_size))
    table.add_row("Batch Size", str(batch_size))
    table.add_row("Save Predictions", str(save_predictions))

    if data_scale:
        table.add_row("Data Scale", data_scale)
    if data_filter:
        table.add_row("Data Filter", data_filter)
    if device:
        table.add_row("Device", device)

    console.print(table)

    # Prepare arguments
    args = {
        "model_path": str(model_path),
        "weights_path": str(weights_path),
        "data_format": data_format,
        "data_path": str(data_path),
        "window_size": window_size,
        "window_offset": window_offset,
        "fps_scaling": fps_scaling,
        "batch_size": batch_size,
        "save_predictions": save_predictions,
        "verbose": verbose,
        "output_path": str(output_path),
    }

    # Add optional parameters
    if data_scale:
        args["data_scale"] = data_scale
    if data_filter:
        args["data_filter"] = data_filter
    if device:
        args["device"] = device

    # Run evaluation
    console.print("[bold blue]Evaluating model...[/bold blue]")

    try:
        # Import evaluation module lazily
        from lisbet.evaluation import evaluate

        results = evaluate(**args)

        # Display key metrics
        if isinstance(results, dict):
            metrics_table = Table(title="Evaluation Results")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")

            for metric, value in results.items():
                if isinstance(value, float):
                    metrics_table.add_row(metric, f"{value:.4f}")
                else:
                    metrics_table.add_row(metric, str(value))

            console.print(metrics_table)

        console.print(
            f"[bold green]Evaluation completed! "
            f"Results saved to: {output_path}[/bold green]"
        )

    except ImportError:
        console.print("[red]Error: Evaluation module not available[/red]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        raise typer.Exit(1) from e
