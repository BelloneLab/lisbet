"""Post-processing command line argument parsers."""

import argparse
from pathlib import Path

from lisbet.cli.common import add_data_io_args, add_verbosity_args


def configure_reduce_dimensions_parser(parser: argparse.ArgumentParser) -> None:
    """Configure reduce_dimensions command parser."""
    add_verbosity_args(parser)
    add_data_io_args(parser, "Embedding data location")
    parser.add_argument(
        "--num_dims", type=int, default=2, help="Embedding space dimensions"
    )
    parser.add_argument(
        "--num_neighbors", type=int, default=60, help="Size of local neighborhood"
    )
    parser.add_argument("--sample_size", type=int, help="Size of random sample")
    parser.add_argument("--sample_seed", type=int, help="RNG seed for random sample")
    parser.add_argument("--umap_seed", type=int, help="RNG seed for UMAP")


def configure_segment_motifs_parser(parser: argparse.ArgumentParser) -> None:
    """Configure segment_motifs command parser."""
    add_verbosity_args(parser)
    add_data_io_args(parser, "Embedding data location")
    parser.add_argument(
        "--min_n_components",
        type=int,
        default=2,
        help="Minimum number of hidden states",
    )
    parser.add_argument(
        "--max_n_components",
        type=int,
        default=32,
        help="Maximum number of hidden states",
    )
    parser.add_argument(
        "--num_iter", type=int, default=10, help="Number of iterations of EM"
    )
    parser.add_argument(
        "--fit_frac", type=float, help="Fraction of data to use for model fitting"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs, use -1 (default) for as many jobs as cores",
    )
    parser.add_argument("--hmm_seed", type=int, help="RNG seed for HMM")
    parser.add_argument("--pretrained_path", type=Path, help="Path to saved HMM models")


def configure_select_prototypes_parser(parser: argparse.ArgumentParser) -> None:
    """Configure select_prototypes command parser."""
    add_verbosity_args(parser)
    add_data_io_args(parser, "Annotation data location")
    parser.add_argument(
        "--min_n_components",
        type=int,
        default=6,
        help="Minimum number of hidden states",
    )
    parser.add_argument(
        "--max_n_components",
        type=int,
        default=32,
        help="Maximum number of hidden states",
    )
    parser.add_argument(
        "--method",
        default="best",
        choices=["min", "best"],
        help="Prototype selection algorithm",
    )
    parser.add_argument(
        "--frame_threshold",
        type=float,
        default=0.05,
        help="Minimum fraction of allocated frames for motifs to be kept",
    )
    parser.add_argument(
        "--bout_threshold",
        type=float,
        default=0.5,
        help="Minimum mean bout duration in seconds for motifs to be kept",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=0.6,
        help="Maximum Jaccard distance from the closest neighbor motif",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30,
        help="Frames per second, used to compute bout duration",
    )
