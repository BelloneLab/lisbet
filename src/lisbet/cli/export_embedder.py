"""Export embedder command for LISBET Enhanced Backbone System.

This module implements the export_embedder command for exporting
embedding models for deployment.
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


def export_embedder(
    model_path: Annotated[
        Path, typer.Argument(help="Path to trained model file or directory")
    ],
    output_path: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output path for exported model",
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
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Export format (pytorch, onnx, torchscript)",
        ),
    ] = "pytorch",
    layer: Annotated[
        Optional[str],
        typer.Option(
            "--layer",
            "-l",
            help="Layer to export (default: backbone)",
        ),
    ] = None,
    optimize: Annotated[
        bool,
        typer.Option(
            "--optimize",
            help="Optimize exported model for inference",
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
    """Export embedding model.

    Export the embedding part of a trained model for efficient deployment
    and inference in production environments.

    Examples:
        betman export_embedder ./model.ckpt
        betman export_embedder ./model ./embedder.pt --format pytorch
        betman export_embedder ./model ./model.onnx --format onnx --optimize
    """
    # Validate inputs
    _validate_model_path(model_path)

    # Validate format
    valid_formats = ["pytorch", "onnx", "torchscript"]
    if format not in valid_formats:
        console.print(f"[red]Error: Invalid format '{format}'[/red]")
        console.print(f"Valid formats: {', '.join(valid_formats)}")
        raise typer.Exit(1)

    # Set default output path
    if output_path is None:
        base_name = model_path.stem if model_path.is_file() else model_path.name
        extension = {"pytorch": ".pt", "onnx": ".onnx", "torchscript": ".pt"}[format]
        output_path = model_path.parent / f"{base_name}_embedder{extension}"

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
    table = Table(title="Export Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model Path", str(model_path))
    table.add_row("Weights Path", str(weights_path))
    table.add_row("Output Path", str(output_path))
    table.add_row("Format", format.upper())
    table.add_row("Optimize", str(optimize))

    if layer:
        table.add_row("Layer", layer)

    console.print(table)

    # Prepare arguments
    args = {
        "model_path": str(model_path),
        "weights_path": str(weights_path),
        "output_path": str(output_path),
        "format": format,
        "optimize": optimize,
        "verbose": verbose,
    }

    if layer:
        args["layer"] = layer

    # Export model
    console.print(
        f"[bold blue]Exporting model to {format.upper()} format...[/bold blue]"
    )

    try:
        # Import io module lazily
        from lisbet.io import export_embedder

        export_embedder(**args)
        console.print(f"[bold green]Model exported to: {output_path}[/bold green]")

    except ImportError:
        console.print("[red]Error: Export module not available[/red]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")
        raise typer.Exit(1) from e
