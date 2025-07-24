"""IO utilities for LISBET."""

import inspect
import logging
import re
from dataclasses import asdict, dataclass
from functools import partial
from itertools import repeat
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import xarray as xr
import yaml
from movement.io import load_poses
from movement.transforms import scale
from torchinfo import summary
from tqdm.auto import tqdm

from lisbet.config.schemas import (
    BACKBONE_CONFIG_REGISTRY,
    ModelConfig,
)
from lisbet.modeling import (
    EmbeddingHead,
    MultiTaskModel,
    TransformerBackbone,
)
from lisbet.modeling.factory import create_model_from_config


@dataclass
class Record:
    """
    Data structure representing a single pose-tracking record.

    Parameters
    ----------
    id : str
        Unique identifier for the record, typically derived from the relative path.
    posetracks : xarray.Dataset
        Pose-tracking data for the record.
    annotations : xarray.Dataset or None, optional
        Annotations associated with the record, if available.
    """

    id: str
    posetracks: xr.Dataset
    annotations: Optional[xr.Dataset] = None


def _filter_kwargs(kwargs, handler):
    """Filter kwargs to match handler's signature."""
    valid_args = [p.name for p in inspect.signature(handler).parameters.values()]
    return {k: v for k, v in kwargs.items() if k in valid_args}


def _load_posetracks(seq_path, data_format, data_scale, select_coords, rename_coords):
    """
    Load and preprocess a pose-tracking dataset from a sequence directory.

    Applies optional coordinate selection and renaming, and rescales coordinates
    to [0, 1] if requested.

    Parameters
    ----------
    seq_path : Path
        Path to the sequence directory.
    data_format : str
        Format of the dataset ('DLC', 'SLEAP', 'movement').
    data_scale : str or None
        Scaling string or None for auto-scaling.
    select_coords : str or None
        Optional subset string in the format 'INDIVIDUALS;AXES;KEYPOINTS', where each
        field is a comma-separated list or '*' for all.
    rename_coords : str or None
        Optional coordinate names remapping in the format 'INDIVIDUALS;AXES;KEYPOINTS',
        where each field is a comma-separated list of maps 'old_id:new_id' or '*' for
        no remapping at that level.

    Returns
    -------
    xarray.Dataset
        The loaded and preprocessed dataset.

    """
    # Valid filenames and their corresponding loading functions
    # TODO: Test re matching for all supported formats.
    data_readers = {
        "DLC": (r"(?i)(DLC.*?shuffle\d+|tracking).*\.csv$", load_poses.from_dlc_file),
        "SLEAP": (r"(?i)SLEAP.*\.h5$", load_poses.from_sleap_file),
        "movement": (r"(?i)tracking.*\.nc$", partial(xr.open_dataset, engine="scipy")),
    }

    if data_format in data_readers:
        # Find all files matching the regex and load them
        pattern, loader = data_readers[data_format]
        rx = re.compile(pattern)
        dss = [loader(pth) for pth in seq_path.iterdir() if rx.search(pth.name)]

    else:
        raise ValueError(f"Unknown data format {data_format}")

    # Check if any datasets were found
    if len(dss) == 0:
        return None

    # Merge all datasets into a single one
    # NOTE: There should be only one dataset per sequence, but we keep this for
    #       compatibility with multiple single-individual datasets
    ds = xr.concat(dss, dim="individuals")
    logging.debug("Individuals: %s", ds["individuals"].values)

    # Replace nan values with 0.0 in the 'position' variable
    # NOTE: This is a workaround for the issue with NaN values in the 'position' during
    #       training, which can cause issues with the model. In the future, we could try
    #       to handle NaN values more gracefully, e.g., by interpolating them or using a
    #       more sophisticated imputation method in movement.
    ds["position"] = ds["position"].fillna(0.0)
    logging.debug("Replaced NaN values in 'position' with 0.0")

    # Drop confidence variable, if present
    # NOTE: This variable is currently not needed in LISBET, but it may become useful in
    #       the future, especially if we decide to provide a measure of tracking
    #       quality to the model.
    if "confidence" in ds:
        ds = ds.drop_vars("confidence")

    # Apply coordinates selection if requested
    # TODO: Move string parsing to CLI during refactoring and make 'sel_dict' an
    #       argument.
    if select_coords is not None:
        # Parse the subset string: 'INDIVIDUALS;AXES;KEYPOINTS'
        fields = select_coords.split(";")
        if len(fields) != 3:
            raise ValueError(
                "select_coords must have format 'INDIVIDUALS;AXES;KEYPOINTS', "
                "e.g. 'ind1,ind2;x,y;nose,neck,tail'"
            )
        # Use a compact dict comprehension for selection
        sel_keys = ["individuals", "space", "keypoints"]
        sel_dict = {
            key: [item.strip() for item in field.split(",") if item.strip()]
            for key, field in zip(sel_keys, fields)
            if field.strip() and field.strip() != "*"
        }
        if sel_dict:
            ds = ds.sel(**sel_dict)

            logging.debug("Subset selection: %s", sel_dict)

    # Apply coordinates renaming if requested
    # TODO: Move string parsing to CLI during refactoring and make 'remap_dict' an
    #       argument.
    if rename_coords is not None:
        # Parse the remapping string: 'INDIVIDUALS;AXES;KEYPOINTS'
        fields = rename_coords.split(";")
        if len(fields) != 3 or any(f.strip() == "" for f in fields):
            raise ValueError(
                "rename_coords must have format 'INDIVIDUALS;AXES;KEYPOINTS', "
                "using '*' for no remapping at a level, e.g. "
                "'mouse1:resident,mouse2:intruder;*;nose:snout,tail:tailbase'"
            )
        rename_keys = ["individuals", "space", "keypoints"]
        remap_dict = {}
        for key, field in zip(rename_keys, fields):
            if field.strip() != "*":
                mapping = {}
                for pair in field.split(","):
                    old, new = pair.split(":")
                    mapping[old.strip()] = new.strip()
                remap_dict[key] = (
                    key,
                    [mapping.get(val, val) for val in ds.coords[key].values],
                )
        if remap_dict:
            ds = ds.assign_coords(**remap_dict)

    # Rescale coordinates in the (0, 1) range
    if data_scale is not None:
        # Explicit scaling
        factor = [1 / float(val) for val in data_scale.split("x")]
        ds = ds.assign(position=scale(ds["position"], factor=factor))

        logging.debug("Rescaled coordinates by factor %s", factor)

    elif "image_size_px" in ds.attrs:
        # Rescale using image size
        factor = [1 / float(val) for val in ds.attrs["image_size_px"]]
        ds = ds.assign(position=scale(ds["position"], factor=factor))

        logging.debug(
            "Rescaled coordinates by image size %s", ds.attrs["image_size_px"]
        )

    else:
        # Auto-scaling
        reduce_dims = ("time", "keypoints", "individuals")

        pos = ds["position"]
        min_val = pos.min(dim=reduce_dims, skipna=True)
        max_val = pos.max(dim=reduce_dims, skipna=True)

        ds = ds.assign(position=(pos - min_val) / (max_val - min_val))

        # Validate scaling
        assert ds["position"].min() >= 0.0, "Coordinates should be in the [0, 1] range"
        assert ds["position"].max() <= 1.0, "Coordinates should be in the [0, 1] range"

        logging.debug(
            "Rescaled coordinates between min values %s and max values %s",
            min_val.values,
            max_val.values,
        )

    # After scaling, enforce [0, 1] range and raise if not satisfied
    min_val = ds["position"].min()
    max_val = ds["position"].max()
    if min_val < 0.0 or max_val > 1.0:
        raise ValueError(
            f"After applying data_scale={data_scale}, coordinates are not in [0, 1] "
            f"(min={min_val.values}, max={max_val.values}). Explicit scaling assumes "
            "that the video has already been cropped to the region of interest during "
            "pose estimation, its origin is at (0,0), and the maximum dimensions match "
            "the scale provided. If this is not the case, use auto mode "
            "(data_scale=None) for normalization."
        )

    # NOTE: We keep the whole Dataset object, rather than selecting the "position"
    #       variable, to allow for future extensions (e.g., adding more variables) and
    #       to keep the FPS information.
    return ds


def _load_annotations(seq_path):
    """
    Load annotations from a sequence directory, if present.

    Returns
    -------
    xarray.Dataset or None
        The loaded annotations, or None if not found.

    """
    # Find all files matching the annotations regex and load them
    rx = re.compile(r"(?i)(manual_scoring|annotations).*\.nc$")
    annotations = [
        xr.open_dataset(pth, engine="scipy")
        for pth in seq_path.iterdir()
        if rx.search(pth.name)
    ]

    # Check if any annotations were found
    if len(annotations) == 0:
        return None

    # Merge all annotations into a single one
    annotations = xr.concat(annotations, dim="annotators")

    logging.debug("Annotations: %s", annotations.coords["behaviors"].values)

    return annotations


def load_records(
    data_format,
    data_path,
    data_scale=None,
    data_filter=None,
    select_coords=None,
    rename_coords=None,
):
    """
    Load pose-tracking records from a directory, with optional filtering, coordinate
    selection and renaming.

    Parameters
    ----------
    data_format : {'movement', 'DLC', 'SLEAP'}
        Dataset format to load.
    data_path : str or Path
        Root directory containing the sequence sub-directories.
    data_scale : str, optional
        If supplied as WIDTHxHEIGHT or WIDTHxHEIGHTxDEPTH, every input coordinate is
        assumed to be in data units and is divided by the given scale to obtain
        normalized coordinates in the range [0, 1]. Otherwise, the algorithm infers the
        active extent directly from the data.
    data_filter : str, optional
        Comma-separated substrings; a record is kept if any substring occurs in its
        relative path. By default, all records are kept.
    select_coords : str or None
        Optional subset string in the format 'INDIVIDUALS;AXES;KEYPOINTS', where each
        field is a comma-separated list or '*' for all. If None, all data is loaded.
        Example: 'mouse1,mouse2;x,y;nose,tail'.
    rename_coords : str or None
        Optional coordinate names remapping in the format 'INDIVIDUALS;AXES;KEYPOINTS',
        where each field is a comma-separated list of maps 'old_id:new_id' or '*' for
        no remapping at that level. If None, original dataset names are used.
        Example: 'mouse1:resident,mouse2:intruder;*;nose:snout,tail:tailbase'.

    Returns
    -------
    list[Record]
        A list of Record objects, each containing id, posetracks, and optionally
        annotations.

    Raises
    ------
    ValueError
        If data_format is unsupported, or if select_coords/rename_coords are invalid.
    NotImplementedError
        For recognized but unimplemented formats.

    Examples
    --------
    >>> records = load_records(
    ...     data_format="movement",
    ...     data_path="~/datasets/mice",
    ...     select_coords="mouse1,mouse2;x,y;nose,tail",
    ...     rename_coords="mouse1:resident,mouse2:intruder;*;nose:snout,tail:tailbase",
    ... )
    >>> print(len(records))
    42
    >>> print(records[0].id)
    'session1/seq001'
    >>> print(records[0].posetracks)
    <xarray.Dataset ...>
    >>> print(records[0].annotations)
    <xarray.Dataset ...> or None

    """
    # Find all potential record paths
    seq_paths = [f for f in Path(data_path).rglob("*") if f.is_dir()]

    # Filter data, if requested
    if data_filter is not None:
        filters = data_filter.split(",")

        seq_paths = [
            seq_path
            for seq_path in seq_paths
            if any(flt in str(seq_path.relative_to(data_path)) for flt in filters)
        ]

        logging.info("%d potential paths after filtering", len(seq_paths))
        logging.debug(seq_paths)

    # Load and preprocess raw data
    records = []
    for seq_path in tqdm(seq_paths, desc="Loading dataset"):
        # Load pose-tracking data
        posetracks = _load_posetracks(
            seq_path, data_format, data_scale, select_coords, rename_coords
        )
        if posetracks is None:
            logging.debug("Skipping %s, no tracking data found", str(seq_path))
            continue

        # Load annotations
        annotations = _load_annotations(seq_path)

        # Create record id
        rec_id = str(seq_path.relative_to(data_path))

        # Add Record object to the list
        records.append(
            Record(id=rec_id, posetracks=posetracks, annotations=annotations)
        )

    # Sanity check: All posetracks must have the same 'features' coordinate (summary of
    #               individuals/keypoints/space)
    if records:
        ref_coords = {
            dim: records[0].posetracks.coords[dim].values.tolist()
            for dim in ("individuals", "keypoints", "space")
        }
        for rec in records:
            for dim in ("individuals", "keypoints", "space"):
                ds_coords = rec.posetracks.coords[dim].values.tolist()
                if ds_coords != ref_coords[dim]:
                    raise ValueError(
                        f"Inconsistent posetracks coordinates in record '{rec.id}':\n"
                        f"Reference {dim}:\n{ref_coords[dim]}\n"
                        f"Record {dim}:\n{ds_coords}"
                    )
    else:
        raise ValueError(
            "No valid records found in the specified directory. Please check the data "
            "path, format and filters to ensure they match the dataset structure.\n"
            f"Current values are: \n data_path = {data_path}\n "
            f"data_format = {data_format}\n data_filter = {data_filter}\n"
        )

    return records


def load_multi_records(data_config):
    """Internal helper. Loads and splits records for all tasks."""
    datasets = data_config.data_format.split(",")
    datapaths = data_config.data_path.split(",")
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
            data_scale=data_config.data_scale,
            data_filter=data_config.data_filter,
            select_coords=data_config.select_coords,
            rename_coords=data_config.rename_coords,
        )
        for dataset, datapath in datasources
    ]

    # Sanity check: All posetracks must have the same 'individuals', 'keypoints', and
    #               'space' coordinates across datasets. As consistency within a dataset
    #               is already checked, we only need to check the first record of each
    #               dataset against the others.
    main_coords = [
        {
            dim: recs[0].posetracks.coords[dim].values.tolist()
            for dim in ("individuals", "keypoints", "space")
        }
        for recs in multi_records
    ]
    ref_coords = main_coords[0]
    for i, coords in enumerate(main_coords):
        for dim in ("individuals", "keypoints", "space"):
            if coords[dim] != ref_coords[dim]:
                raise ValueError(
                    "Inconsistent posetracks coordinates in loaded records, dataset "
                    f"{i}:\n"
                    f"Reference {dim}:\n{ref_coords[dim]}\n"
                    f"Record {dim}:\n{coords[dim]}"
                )

    return multi_records


def load_model(config_path, weights_path):
    """
    Load a pretrained LISBET model from a configuration file.

    This function supports loading models from YAML configuration files (as used in
    LISBET). It uses the model factory to instantiate the model and loads weights from
    the specified file.

    Parameters
    ----------
    config_path : str or Path or dataclass
        Path to the model configuration YAML file.
    weights_path : str or Path
        Path to the model weights file.

    Returns
    -------
    torch.nn.Module
        The loaded LISBET model.
    """
    with open(config_path, encoding="utf-8") as f_yaml:
        model_config_dict = yaml.safe_load(f_yaml)

    # Create backbone configuration to replace the simple dict loaded from YAML
    backbone_type = model_config_dict["backbone"]["backbone_type"]
    backbone_config = BACKBONE_CONFIG_REGISTRY[backbone_type](
        **model_config_dict["backbone"]
    )
    model_config_dict["backbone"] = backbone_config

    # Create model configuration dataclass
    model_config = ModelConfig(**model_config_dict)

    # Load model configuration as a dataclass
    model = create_model_from_config(model_config)

    # Load weights (strict=False allows for partial loading)
    incompatible_layers = model.load_state_dict(
        torch.load(weights_path, weights_only=True, map_location=torch.device("cpu")),
        strict=False,
    )
    logging.info(
        "Loaded weights from file.\nMissing keys: %s\nUnexpected keys: %s",
        incompatible_layers.missing_keys,
        incompatible_layers.unexpected_keys,
    )

    return model


def export_embedder(model_path, weights_path, output_path=Path(".")):
    # Get hyper-parameters
    with open(model_path, encoding="utf-8") as f_yaml:
        model_config = yaml.safe_load(f_yaml)
    model_id = model_config["model_id"] + "-embedder"

    # Update config
    model_config["model_id"] = model_id
    model_config["out_heads"] = {"embedding": {}}

    # Create behavior embedding model
    backbone_kwargs = _filter_kwargs(model_config, TransformerBackbone)
    backbone = TransformerBackbone(**backbone_kwargs)

    head_kwargs = _filter_kwargs(model_config, EmbeddingHead)
    # TODO: Remove this hack when we have a better solution
    head_kwargs["output_token_idx"] = -(model_config["window_offset"] + 1)
    head = {"embedding": EmbeddingHead(**head_kwargs)}

    embedding_model = MultiTaskModel(backbone, head)
    summary(embedding_model)

    # Load weights from pretrained model
    embedding_model.load_state_dict(
        torch.load(weights_path, weights_only=True, map_location=torch.device("cpu")),
        strict=False,
    )

    # Create output directory
    output_path = output_path / "models" / model_id

    # Store configuration
    model_path = output_path / "model_config.yml"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f_yaml:
        yaml.safe_dump(model_config, f_yaml)

    # Store weights
    weights_path = output_path / "weights" / weights_path.name
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embedding_model.state_dict(), weights_path)


def dump_records(data_path, records):
    """
    Dump a list of records to a file.

    Pose tracks and annotations are saved in a NetCDF format.

    Parameters
    ----------
    data_path : str or Path
        Directory where the records will be saved.

    records : list of Record
        List of Record objects to be saved.

    """
    for rec in tqdm(records, desc="Dumping records to disk"):
        rec_path = Path(data_path) / rec.id
        rec_path.mkdir(parents=True, exist_ok=True)

        # Save posetracks
        rec.posetracks.to_netcdf(rec_path / "tracking.nc", engine="scipy")

        # Save annotations
        if rec.annotations is not None:
            rec.annotations.to_netcdf(rec_path / "annotations.nc", engine="scipy")


def dump_annotations(results, output_path):
    """
    Save LISBET behavior predictions to CSV files.

    Parameters
    ----------
    results : list of (record_id, np.ndarray)
        Output from annotate_behavior.
    output_path : str or Path
        Root directory to save CSVs. Each record will be saved under
        output_path/annotations/<record_id>/machineAnnotation_lisbet.csv
    """
    for key, model_output in tqdm(results, desc="Saving LISBET annotations"):
        dst_path = (
            Path(output_path) / "annotations" / key / "machineAnnotation_lisbet.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(model_output).to_csv(dst_path, index=False)


def dump_embeddings(results, output_path):
    """
    Save LISBET embeddings to CSV files.

    Parameters
    ----------
    results : list of (record_id, np.ndarray)
        Output from compute_embeddings.
    output_path : str or Path
        Root directory to save CSVs. Each record will be saved under
        output_path/embeddings/<record_id>/features_lisbet_embedding.csv
    """
    for key, model_output in tqdm(results, desc="Saving LISBET embeddings"):
        dst_path = (
            Path(output_path) / "embeddings" / key / "features_lisbet_embedding.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(model_output).to_csv(dst_path, index=False)


def dump_evaluation_results(report: dict, output_path: str, model_path: str):
    """
    Save evaluation report to a YAML file in a standardized location.

    Parameters
    ----------
    report : dict
        The evaluation report.
    output_path : str or Path
        Directory to save the report.
    model_path : str
        Path to the model config (used to extract model_id).
    """
    with open(model_path, encoding="utf-8") as f_yaml:
        model_config = yaml.safe_load(f_yaml)
    model_id = model_config.get("model_id", "unknown_model")

    report_path = Path(output_path) / "evaluations" / model_id / "evaluation_report.yml"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f_yaml:
        yaml.safe_dump(report, f_yaml)


def dump_weights(model, output_path, run_id, filename):
    """Internal helper. Saves model weights."""
    weights_path = Path(output_path) / "models" / run_id / "weights" / filename
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)


def dump_model_config(output_path, run_id, model_config):
    """Save model configuration to YAML file."""
    model_path = Path(output_path) / "models" / run_id / "model_config.yml"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "w", encoding="utf-8") as f_yaml:
        yaml.safe_dump(asdict(model_config), f_yaml)


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
