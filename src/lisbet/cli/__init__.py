"""CLI package for LISBET Enhanced Backbone System.

This package provides a modern Typer-based command-line interface with
individual commands for training, inference, and data processing.
"""

from lisbet.cli.main import app

__all__ = ["app"]

__doc__ = """
Command Line Interface
======================

The LISBET CLI provides 11 individual commands for working with behavioral data:

Training
--------
- train_model: Train classification models with presets or custom configs

Inference
---------
- annotate_behavior: Annotate behaviors using trained models
- compute_embeddings: Extract behavioral embeddings
- export_embedder: Export models for deployment
- evaluate_model: Evaluate model performance

Analysis
--------
- reduce_dimensions: Dimensionality reduction with UMAP/PCA/t-SNE
- segment_motifs: Behavioral motif segmentation with HMM
- select_prototypes: Select prototype behaviors from annotations

Data Management
---------------
- fetch_dataset: Download public datasets
- fetch_model: Download pre-trained models
- model_info: Display model configuration details

Usage Examples
--------------
Train with preset:
    betman train_model transformer-base ./data

Train with config:
    betman train_model my_config.yml ./data --epochs 50

Annotate behaviors:
    betman annotate_behavior ./model.ckpt ./data.h5

Get model info:
    betman model_info ./config.yml --detailed
"""
