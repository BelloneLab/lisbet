from lisbet.cli.commands.eval import configure_evaluate_model_parser
from lisbet.cli.commands.fetch import (
    configure_fetch_dataset_parser,
    configure_fetch_model_parser,
)
from lisbet.cli.commands.info import configure_model_info_parser
from lisbet.cli.commands.postproc import (
    configure_reduce_dimensions_parser,
    configure_segment_motifs_parser,
    configure_select_prototypes_parser,
)
from lisbet.cli.commands.predict import (
    configure_annotate_behavior_parser,
    configure_compute_embeddings_parser,
)
from lisbet.cli.commands.train import (
    configure_export_embedder_parser,
    configure_train_model_parser,
)

__all__ = [
    "configure_evaluate_model_parser",
    "configure_fetch_dataset_parser",
    "configure_fetch_model_parser",
    "configure_model_info_parser",
    "configure_reduce_dimensions_parser",
    "configure_segment_motifs_parser",
    "configure_select_prototypes_parser",
    "configure_annotate_behavior_parser",
    "configure_compute_embeddings_parser",
    "configure_export_embedder_parser",
    "configure_train_model_parser",
]

__doc__ = """
LISBET CLI Command Configuration
================================
This module provides configuration functions for various LISBET CLI commands.
Each function configures an `argparse.ArgumentParser` for a specific command,
allowing users to specify command-line arguments and options.
The commands include:
- `evaluate_model`: Evaluate a trained model on a dataset.
- `fetch_dataset`: Download public datasets from the LISBET repository.
- `fetch_model`: Download public models from the LISBET repository.
- `model_info`: Display information about a LISBET model configuration.
- `reduce_dimensions`: Perform dimensionality reduction using UMAP.
- `segment_motifs`: Segment behavioral motifs using HMM.
- `select_prototypes`: Select prototype behaviors from multiple annotations.
- `annotate_behavior`: Annotate behaviors using a trained model.
- `compute_embeddings`: Compute behavioral embeddings using a trained model.
- `export_embedder`: Export an embedding model.
- `train_model`: Train a model on a dataset.
Each configuration function adds relevant arguments to the parser,
allowing users to customize the command's behavior.
"""
