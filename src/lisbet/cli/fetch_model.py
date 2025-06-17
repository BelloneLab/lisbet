"""Fetch model command for LISBET Enhanced Backbone System.

This module implements the fetch_model command for downloading
public models from the internet.
"""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

# Create rich console for output
console = Console()


def fetch_model(
    model_name: Annotated[str, typer.Argument(help="Name of the model to fetch")],
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output directory (defaults to current directory)",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Force download even if file exists",
        ),
    ] = False,
    model_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Preferred model format (pytorch, onnx, etc.)",
        ),
    ] = "pytorch",
    include_weights: Annotated[
        bool,
        typer.Option(
            "--include-weights",
            help="Include model weights in download",
        ),
    ] = True,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            help="Verbosity level",
        ),
    ] = 1,
) -> None:
    """Download public model from the internet.

    Fetch publicly available pre-trained models from the LISBET hub or other
    repositories for inference or fine-tuning.

    Examples:
        betman fetch_model pretrained_transformer
        betman fetch_model mouse_classifier --output ./models
        betman fetch_model social_detector --force --format onnx
    """
    # Set default output directory
    if output_dir is None:
        output_dir = Path.cwd()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Show configuration
    table = Table(title="Model Download Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model Name", model_name)
    table.add_row("Output Directory", str(output_dir))
    table.add_row("Model Format", model_format)
    table.add_row("Include Weights", str(include_weights))
    table.add_row("Force Download", str(force))

    console.print(table)

    # Prepare arguments
    args = {
        "model_name": model_name,
        "output_dir": str(output_dir),
        "force": force,
        "model_format": model_format,
        "include_weights": include_weights,
        "verbose": verbose,
    }

    # Download model
    console.print(f"[bold blue]Downloading model: {model_name}...[/bold blue]")

    try:
        # Import hub module lazily
        from lisbet.hub import fetch_model

        result_path = fetch_model(**args)
        console.print(f"[bold green]Model downloaded to: {result_path}[/bold green]")

    except ImportError:
        console.print("[red]Error: Hub module not available[/red]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        raise typer.Exit(1) from e
