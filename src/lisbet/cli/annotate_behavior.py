"""Annotate behavior command for LISBET Enhanced Backbone System.

This module implements the annotate_behavior command for applying trained
models to annotate behaviors in new data.
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


def annotate_behavior(
    model_path: Annotated[
        Path, typer.Argument(help="Path to trained model file or directory")
    ],
    data_path: Annotated[
        Path, typer.Argument(help="Path to input data file or directory")
    ],
    output_path: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output path for annotations (defaults to input_path_annotations.h5)",
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
        typer.Option("--data-scale", help="Data scaling method"),
    ] = None,
    data_filter: Annotated[
        Optional[str],
        typer.Option("--data-filter", help="Data filtering method"),
    ] = None,
    mode: Annotated[
        str,
        typer.Option("--mode", "-m", help="Annotation mode (multiclass, multilabel)"),
    ] = "multiclass",
    threshold: Annotated[
        float, typer.Option("--threshold", "-t", help="Classification threshold")
    ] = 0.5,
    window_size: Annotated[
        int, typer.Option("--window-size", help="Window size for processing")
    ] = 200,
    window_offset: Annotated[
        int, typer.Option("--window-offset", help="Window offset")
    ] = 0,
    fps_scaling: Annotated[
        float, typer.Option("--fps-scaling", help="FPS scaling factor")
    ] = 1.0,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Batch size for inference")
    ] = 128,
    device: Annotated[
        Optional[str],
        typer.Option("--device", "-d", help="Device to use (cpu, cuda, auto)"),
    ] = None,
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", help="Verbosity level")
    ] = 1,
) -> None:
    """Annotate behaviors using trained model.

    Apply a trained classification model to new data to predict behaviors
    and generate annotation files.

    Examples:
        betman annotate_behavior ./model.ckpt ./new_data.h5
        betman annotate_behavior ./model ./data --mode multilabel --threshold 0.7
        betman annotate_behavior ./model.ckpt ./data --weights ./weights.pt
    """
    # Validate inputs
    _validate_model_path(model_path)
    _validate_data_path(data_path)

    # Set default output path
    if output_path is None:
        if data_path.is_file():
            output_path = data_path.parent / f"{data_path.stem}_annotations.h5"
        else:
            output_path = data_path / "annotations.h5"

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
    table = Table(title="Behavior Annotation Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model Path", str(model_path))
    table.add_row("Weights Path", str(weights_path))
    table.add_row("Data Path", str(data_path))
    table.add_row("Output Path", str(output_path))
    table.add_row("Data Format", data_format)
    table.add_row("Mode", mode)
    table.add_row("Threshold", str(threshold))
    table.add_row("Window Size", str(window_size))
    table.add_row("Batch Size", str(batch_size))

    if data_scale:
        table.add_row("Data Scale", data_scale)
    if data_filter:
        table.add_row("Data Filter", data_filter)
    if device:
        table.add_row("Device", device)

    console.print(table)

    # Prepare arguments for the annotation function
    args = {
        "model_path": str(model_path),
        "weights_path": str(weights_path),
        "data_format": data_format,
        "data_path": str(data_path),
        "mode": mode,
        "threshold": threshold,
        "window_size": window_size,
        "window_offset": window_offset,
        "fps_scaling": fps_scaling,
        "batch_size": batch_size,
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

    # Run annotation
    console.print("[bold blue]Starting behavior annotation...[/bold blue]")

    try:
        # Import inference module lazily
        from lisbet.inference import annotate_behavior

        annotate_behavior(**args)
        console.print(f"[bold green]Annotations saved to: {output_path}[/bold green]")

    except ImportError:
        console.print("[red]Error: Inference module not available[/red]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Annotation failed: {e}[/red]")
        raise typer.Exit(1) from e
