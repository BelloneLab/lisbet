"""Main Typer application for LISBET Enhanced Backbone System CLI.

This module contains the main Typer app definition and command registration.
The __init__.py file should only handle imports and package initialization.
"""

from typing import Annotated

import typer
from rich.console import Console

# Import individual command functions
from lisbet.cli.annotate_behavior import annotate_behavior
from lisbet.cli.compute_embeddings import compute_embeddings
from lisbet.cli.evaluate_model import evaluate_model
from lisbet.cli.export_embedder import export_embedder
from lisbet.cli.fetch_dataset import fetch_dataset
from lisbet.cli.fetch_model import fetch_model
from lisbet.cli.model_info import model_info
from lisbet.cli.reduce_dimensions import reduce_dimensions
from lisbet.cli.segment_motifs import segment_motifs
from lisbet.cli.select_prototypes import select_prototypes
from lisbet.cli.train_model import train_model

# Create the main Typer application
app = typer.Typer(
    name="betman",
    help="LISBET Enhanced Backbone System CLI",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Create rich console for output
console = Console()

# Register all 11 commands directly with the app
app.command(name="train_model")(train_model)
app.command(name="annotate_behavior")(annotate_behavior)
app.command(name="compute_embeddings")(compute_embeddings)
app.command(name="export_embedder")(export_embedder)
app.command(name="reduce_dimensions")(reduce_dimensions)
app.command(name="segment_motifs")(segment_motifs)
app.command(name="select_prototypes")(select_prototypes)
app.command(name="fetch_dataset")(fetch_dataset)
app.command(name="fetch_model")(fetch_model)
app.command(name="model_info")(model_info)
app.command(name="evaluate_model")(evaluate_model)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit",
        ),
    ] = False,
) -> None:
    """LISBET Enhanced Backbone System CLI.

    A modern command-line interface for training transformer models
    on behavioral data using built-in presets and custom configurations.
    """
    if version:
        try:
            from lisbet._version import __version__

            console.print(f"betman version {__version__}")
        except ImportError:
            console.print("betman version unknown")
        raise typer.Exit()


if __name__ == "__main__":
    app()
