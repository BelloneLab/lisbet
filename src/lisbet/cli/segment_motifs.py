"""Segment motifs command for LISBET Enhanced Backbone System.

This module implements the segment_motifs command for segmenting
behavioral motifs using HMM.
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


def segment_motifs(
    embeddings_path: Annotated[Path, typer.Argument(help="Path to embeddings file")],
    output_path: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output path for segmentation results",
        ),
    ] = None,
    n_states: Annotated[
        int,
        typer.Option(
            "--states",
            "-s",
            help="Number of hidden states for HMM",
        ),
    ] = 10,
    covariance_type: Annotated[
        str,
        typer.Option(
            "--covariance",
            "-c",
            help="Covariance type (full, diag, tied, spherical)",
        ),
    ] = "full",
    n_iter: Annotated[
        int,
        typer.Option(
            "--iterations",
            "-i",
            help="Maximum number of EM iterations",
        ),
    ] = 100,
    random_seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Random seed for reproducibility",
        ),
    ] = 42,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            help="Verbosity level",
        ),
    ] = 1,
) -> None:
    """Segment behavioral motifs using HMM.

    Discover discrete behavioral motifs in continuous embeddings using
    Hidden Markov Models for temporal segmentation.

    Examples:
        betman segment_motifs ./embeddings.h5
        betman segment_motifs ./embeddings.h5 --states 15 --covariance diag
        betman segment_motifs ./embeddings.h5 --states 8 --iterations 200
    """
    # Validate inputs
    _validate_data_path(embeddings_path)

    # Validate covariance type
    valid_covariance = ["full", "diag", "tied", "spherical"]
    if covariance_type not in valid_covariance:
        console.print(f"[red]Error: Invalid covariance type '{covariance_type}'[/red]")
        console.print(f"Valid types: {', '.join(valid_covariance)}")
        raise typer.Exit(1)

    # Set default output path
    if output_path is None:
        output_path = embeddings_path.parent / f"{embeddings_path.stem}_segments.h5"

    # Show configuration
    table = Table(title="Motif Segmentation Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Input Path", str(embeddings_path))
    table.add_row("Output Path", str(output_path))
    table.add_row("Hidden States", str(n_states))
    table.add_row("Covariance Type", covariance_type)
    table.add_row("Max Iterations", str(n_iter))
    table.add_row("Random Seed", str(random_seed))

    console.print(table)

    # Prepare arguments
    args = {
        "embeddings_path": str(embeddings_path),
        "output_path": str(output_path),
        "n_states": n_states,
        "covariance_type": covariance_type,
        "n_iter": n_iter,
        "random_seed": random_seed,
        "verbose": verbose,
    }

    # Run segmentation
    console.print("[bold blue]Segmenting behavioral motifs...[/bold blue]")

    try:
        # Import unsupervised module lazily
        from lisbet.unsupervised import segment_hmm

        segment_hmm(**args)
        console.print(
            f"[bold green]Segmentation results saved to: {output_path}[/bold green]"
        )

    except ImportError:
        console.print("[red]Error: Unsupervised learning module not available[/red]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Motif segmentation failed: {e}[/red]")
        raise typer.Exit(1) from e
