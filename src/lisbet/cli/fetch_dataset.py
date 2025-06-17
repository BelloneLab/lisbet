"""Fetch dataset command for LISBET Enhanced Backbone System.

This module implements the fetch_dataset command for downloading
public datasets from the internet.
"""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

# Create rich console for output
console = Console()


def fetch_dataset(
    dataset_name: Annotated[str, typer.Argument(help="Name of the dataset to fetch")],
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
    data_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Preferred data format (h5, csv, etc.)",
        ),
    ] = "h5",
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            help="Verbosity level",
        ),
    ] = 1,
) -> None:
    """Download public dataset from the internet.

    Fetch publicly available datasets from the LISBET hub or other
    repositories for training and evaluation.

    Examples:
        betman fetch_dataset example_dataset
        betman fetch_dataset mouse_behavior --output ./data
        betman fetch_dataset primate_social --force
    """
    # Set default output directory
    if output_dir is None:
        output_dir = Path.cwd()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Show configuration
    table = Table(title="Dataset Download Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Dataset Name", dataset_name)
    table.add_row("Output Directory", str(output_dir))
    table.add_row("Data Format", data_format)
    table.add_row("Force Download", str(force))

    console.print(table)

    # Prepare arguments
    args = {
        "dataset_name": dataset_name,
        "output_dir": str(output_dir),
        "force": force,
        "data_format": data_format,
        "verbose": verbose,
    }

    # Download dataset
    console.print(f"[bold blue]Downloading dataset: {dataset_name}...[/bold blue]")

    try:
        # Import hub module lazily
        from lisbet.hub import fetch_dataset

        result_path = fetch_dataset(**args)
        console.print(f"[bold green]Dataset downloaded to: {result_path}[/bold green]")

    except ImportError:
        console.print("[red]Error: Hub module not available[/red]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        raise typer.Exit(1) from e
