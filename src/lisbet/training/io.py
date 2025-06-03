"""Utility functions for writing model outputs and configurations."""

import logging
from itertools import repeat
from pathlib import Path

import torch
import yaml

from lisbet.datasets import load_records


def dump_weights(model, output_path, run_id, filename):
    """Internal helper. Saves model weights."""
    weights_path = Path(output_path) / "models" / run_id / "weights" / filename
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)


def dump_model_config(
    output_path,
    run_id,
    window_size,
    window_offset,
    output_token_idx,
    bp_dim,
    emb_dim,
    hidden_dim,
    num_heads,
    num_layers,
    max_len,
    tasks,
    input_features,
):
    """Internal helper. Saves model config."""
    model_config = {
        "model_id": run_id,
        "window_size": window_size,
        "window_offset": window_offset,
        "output_token_idx": output_token_idx,
        "bp_dim": bp_dim,
        "emb_dim": emb_dim,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "max_len": max_len,
        "out_dim": {task.task_id: task.out_dim for task in tasks},
        "input_features": input_features,
    }
    model_path = Path(output_path) / "models" / run_id / "model_config.yml"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f_yaml:
        yaml.safe_dump(model_config, f_yaml)


def dump_profiling_results(output_path, run_id, prof):
    """Internal helper. Saves profiling results."""
    # Create profiling directory
    profiling_path = Path(output_path) / "models" / run_id / "profiler"
    profiling_path.mkdir(parents=True, exist_ok=True)

    # Save profiling results
    prof.export_chrome_trace(str(profiling_path / "chrome_trace.json.gz"))
    prof.export_memory_timeline(str(profiling_path / "memory_trace.html"))
    prof.export_stacks(str(profiling_path / "cpu_stacks.txt"), "self_cpu_time_total")
    prof.export_stacks(str(profiling_path / "cuda_stacks.txt"), "self_cuda_time_total")
    with open(profiling_path / "profiling_summary.txt", "w", encoding="utf-8") as f:
        f.write("CPU Profiling Summary:\n")
        f.write(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        f.write("\n\nCUDA Profiling Summary:\n")
        f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))


def load_multi_records(
    data_format,
    data_path,
    data_scale,
    data_filter,
    select_coords,
    rename_coords,
):
    """Internal helper. Loads and splits records for all tasks."""
    datasets = data_format.split(",")
    datapaths = data_path.split(",")
    if len(datasets) == len(datapaths):
        datasources = list(zip(datasets, datapaths))
    elif len(datapaths) == 1:
        datasources = list(zip(datasets, repeat(datapaths[0])))
    else:
        raise ValueError(
            "Input arguments datasets and datapaths must have the same length, or"
            " datapath must be a single element."
        )
    logging.debug(datasources)

    # Load records
    multi_records = [
        load_records(
            dataset,
            datapath,
            data_scale=data_scale,
            data_filter=data_filter,
            select_coords=select_coords,
            rename_coords=rename_coords,
        )
        for dataset, datapath in datasources
    ]

    # Sanity check: All posetracks must have the same 'features' coordinate across
    #               datasets. As consistency within a dataset is already checked, we
    #               only need to check the first record of each dataset against the
    #               others.
    main_features = [
        recs[0].posetracks.coords["features"].values.tolist() for recs in multi_records
    ]
    ref_features = main_features[0]
    for i, features in enumerate(main_features):
        if features != ref_features:
            raise ValueError(
                f"Inconsistent posetracks coordinates in loaded records, dataset {i}:\n"
                f"Reference features:\n{ref_features}\n"
                f"Record features:\n{features}"
            )

    return multi_records
