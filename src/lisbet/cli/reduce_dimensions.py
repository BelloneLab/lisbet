"""Reduce dimensions command for LISBET Enhanced Backbone System.

This module implements the reduce_dimensions command for performing
dimensionality reduction with UMAP.
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


def reduce_dimensions(
    embeddings_path: Annotated[Path, typer.Argument(help="Path to embeddings file")],
    output_path: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output path for reduced embeddings",
        ),
    ] = None,
    method: Annotated[
        str,
        typer.Option(
            "--method",
            "-m",
            help="Dimensionality reduction method (umap, pca, tsne)",
        ),
    ] = "umap",
    n_components: Annotated[
        int,
        typer.Option(
            "--components",
            "-c",
            help="Number of components for reduction",
        ),
    ] = 2,
    n_neighbors: Annotated[
        int,
        typer.Option(
            "--neighbors",
            "-n",
            help="Number of neighbors for UMAP",
        ),
    ] = 15,
    min_dist: Annotated[
        float,
        typer.Option(
            "--min-dist",
            help="Minimum distance for UMAP",
        ),
    ] = 0.1,
    random_seed: Annotated[
        int,
        typer.Option(
            "--seed",
            "-s",
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
    """Perform dimensionality reduction with UMAP.

    Reduce high-dimensional behavioral embeddings to lower dimensions
    for visualization and analysis using UMAP, PCA, or t-SNE.

    Examples:
        betman reduce_dimensions ./embeddings.h5
        betman reduce_dimensions ./embeddings.h5 --method pca --components 3
        betman reduce_dimensions ./embeddings.h5 --neighbors 30 --min-dist 0.05
    """
    # Validate inputs
    _validate_data_path(embeddings_path)

    # Validate method
    valid_methods = ["umap", "pca", "tsne"]
    if method not in valid_methods:
        console.print(f"[red]Error: Invalid method '{method}'[/red]")
        console.print(f"Valid methods: {', '.join(valid_methods)}")
        raise typer.Exit(1)

    # Set default output path
    if output_path is None:
        output_path = (
            embeddings_path.parent / f"{embeddings_path.stem}_reduced_{method}.h5"
        )

    # Show configuration
    table = Table(title="Dimensionality Reduction Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Input Path", str(embeddings_path))
    table.add_row("Output Path", str(output_path))
    table.add_row("Method", method.upper())
    table.add_row("Components", str(n_components))
    table.add_row("Random Seed", str(random_seed))

    if method == "umap":
        table.add_row("Neighbors", str(n_neighbors))
        table.add_row("Min Distance", str(min_dist))

    console.print(table)

    # Prepare arguments
    args = {
        "embeddings_path": str(embeddings_path),
        "output_path": str(output_path),
        "method": method,
        "n_components": n_components,
        "random_seed": random_seed,
        "verbose": verbose,
    }

    if method == "umap":
        args["n_neighbors"] = n_neighbors
        args["min_dist"] = min_dist

    # Run dimensionality reduction
    console.print(
        f"[bold blue]Performing {method.upper()} dimensionality reduction..."
        f"[/bold blue]"
    )

    try:
        # Import unsupervised module lazily
        from lisbet.unsupervised import reduce_umap

        reduce_umap(**args)
        console.print(
            f"[bold green]Reduced embeddings saved to: {output_path}[/bold green]"
        )

    except ImportError:
        console.print("[red]Error: Unsupervised learning module not available[/red]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Dimensionality reduction failed: {e}[/red]")
        raise typer.Exit(1) from e
