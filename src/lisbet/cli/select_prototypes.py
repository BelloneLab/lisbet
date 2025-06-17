"""Select prototypes command for LISBET Enhanced Backbone System.

This module implements the select_prototypes command for selecting
prototype behaviors from multiple annotations.
"""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

# Create rich console for output
console = Console()


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


def select_prototypes(
    annotations_dir: Annotated[
        Path, typer.Argument(help="Directory containing annotation files")
    ],
    output_path: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output path for prototype annotations",
        ),
    ] = None,
    min_agreement: Annotated[
        float,
        typer.Option(
            "--min-agreement",
            "-a",
            help="Minimum agreement threshold",
        ),
    ] = 0.7,
    method: Annotated[
        str,
        typer.Option(
            "--method",
            "-m",
            help="Selection method (majority, consensus, weighted)",
        ),
    ] = "majority",
    data_format: Annotated[
        str,
        typer.Option(
            "--data-format",
            "-f",
            help="Input data format",
        ),
    ] = "h5",
    keypoints: Annotated[
        Optional[str],
        typer.Option(
            "--keypoints",
            help="Keypoint configuration",
        ),
    ] = None,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            help="Verbosity level",
        ),
    ] = 1,
) -> None:
    """Select prototype behaviors from multiple annotations.

    Combine multiple behavior annotations to select prototypical examples
    based on agreement across annotators or models.

    Examples:
        betman select_prototypes ./annotations_dir
        betman select_prototypes ./annotations --min-agreement 0.8
        betman select_prototypes ./annotations --method consensus
    """
    # Validate inputs
    if not annotations_dir.exists() or not annotations_dir.is_dir():
        console.print(
            f"[red]Error: Annotations directory not found: {annotations_dir}[/red]"
        )
        raise typer.Exit(1)

    # Validate method
    valid_methods = ["majority", "consensus", "weighted"]
    if method not in valid_methods:
        console.print(f"[red]Error: Invalid method '{method}'[/red]")
        console.print(f"Valid methods: {', '.join(valid_methods)}")
        raise typer.Exit(1)

    # Set default output path
    if output_path is None:
        output_path = annotations_dir / "prototypes.h5"

    # Show configuration
    table = Table(title="Prototype Selection Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Annotations Dir", str(annotations_dir))
    table.add_row("Output Path", str(output_path))
    table.add_row("Min Agreement", str(min_agreement))
    table.add_row("Method", method)
    table.add_row("Data Format", data_format)

    if keypoints:
        table.add_row("Keypoints", keypoints)

    console.print(table)

    # Prepare arguments
    args = {
        "annotations_dir": str(annotations_dir),
        "output_path": str(output_path),
        "min_agreement": min_agreement,
        "method": method,
        "data_format": data_format,
        "verbose": verbose,
    }

    if keypoints:
        args["keypoints"] = keypoints

    # Run prototype selection
    console.print("[bold blue]Selecting prototype behaviors...[/bold blue]")

    try:
        # Import postprocessing module lazily
        from lisbet.postprocessing import select_prototypes

        select_prototypes(**args)
        console.print(f"[bold green]Prototypes saved to: {output_path}[/bold green]")

    except ImportError:
        console.print("[red]Error: Postprocessing module not available[/red]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Prototype selection failed: {e}[/red]")
        raise typer.Exit(1) from e
