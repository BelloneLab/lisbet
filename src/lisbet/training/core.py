"""Training and fitting functions for LISBET.

Notes
-----
[a] The dictionary of RNG seed could be refactored as a dataclass in the future.

[b] The train/dev split is performed here and not in the input_pipeline module to
    emphasize that the test set is frozen and won't be used for hyper-parameters tuning.

[c] When mixing datasets of different lengths, the training and evaluation loops will
    stop after exhausting the shortest one. Please consider using random sampling.

"""

import logging
import os
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm.auto import trange

from lisbet import modeling
from lisbet.config.schemas import ExperimentConfig
from lisbet.io import (
    dump_model_config,
    dump_profiling_results,
    dump_weights,
    load_multi_records,
)
from lisbet.training.preprocessing import split_multi_records
from lisbet.training.tasks import configure_tasks
from lisbet.training.utils import estimate_num_workers, generate_seeds, worker_init_fn


def _configure_profiler(steps_multiplier):
    """Internal helper. Configures the profiler."""
    if os.environ.get("TORCH_PROFILER", "0") == "1":
        logging.info("Profiler is enabled.")

        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                skip_first=4 * steps_multiplier,
                wait=steps_multiplier,
                warmup=steps_multiplier,
                active=8 * steps_multiplier,
                repeat=1,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            # NOTE: ExperimentalConfig needed until bug in torch.profiler is fixed, see
            #       https://github.com/pytorch/pytorch/issues/100253
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        )

    else:
        logging.debug("Profiler is disabled.")
        profiler = nullcontext()

    return profiler


def _build_model(
    feature_dim,
    embedding_dim,
    hidden_dim,
    num_heads,
    num_layers,
    max_length,
    tasks,
    load_backbone_weights,
    freeze_backbone_weights,
):
    """Internal helper. Builds the LISBET model."""
    model = modeling.MultiTaskModel(
        modeling.TransformerBackbone(
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_length=max_length,
        ),
        {task.task_id: task.head for task in tasks},
    )

    if load_backbone_weights:
        incompatible_layers = model.load_state_dict(
            torch.load(load_backbone_weights, weights_only=True),
            strict=False,
        )
        logging.info(
            "Loaded weights from file.\nMissing keys: %s\nUnexpected keys: %s",
            incompatible_layers.missing_keys,
            incompatible_layers.unexpected_keys,
        )

    if freeze_backbone_weights:
        for param in model.backbone.parameters():
            param.requires_grad = False
    return model


def _configure_optimizer_and_scheduler(model, learning_rate):
    """Internal helper. Configures optimizer, scheduler, and scaler."""
    # Configure optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    # Configure LR (warmup scheduler)
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-2,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    # Configure LR (main scheduler)
    T_0 = 10
    T_mult = 2
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult
    )

    # Configure final LR scheduler
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    return optimizer, scheduler


def _configure_dataloaders(tasks, group, batch_size, sample_ratio, pin_memory):
    """Internal helper. Configures dataloaders for a group."""
    # Estimate number of samples
    n_batches = np.ceil(
        min(getattr(task, f"{group}_dataset").n_frames for task in tasks) / batch_size
    ).astype(int)
    if sample_ratio is not None:
        n_batches = int(n_batches * sample_ratio)
    logging.info("Using %d samples from the %s group", n_batches * batch_size, group)

    # Estimate number of workers
    num_workers = estimate_num_workers(len(tasks), batch_size, batch_size_per_worker=4)

    # Create a dataloader for each task
    dataloaders = []
    for task in tasks:
        dataset = getattr(task, f"{group}_dataset")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
        )

        dataloaders.append(dataloader)

    return dataloaders, n_batches


def _train_one_epoch(
    model,
    dataloaders,
    n_batches,
    optimizer,
    scheduler,
    tasks,
    prof,
    fabric,
):
    """Internal helper. Runs one training epoch."""
    model.train()

    dl_iter = [iter(dl) for dl in dataloaders]

    # Iterate over all batches
    for batch_idx in trange(n_batches, desc="Training batches", leave=False):
        optimizer.zero_grad(set_to_none=True)

        # Iterate over all tasks
        for task, dataloader in zip(tasks, dl_iter):
            data, target = next(dataloader)

            # Forward pass
            output = model(data, task.task_id)
            loss = task.loss_function(output, target)

            # Backward pass
            fabric.backward(loss)

            # Store loss value and metrics for stats
            if batch_idx % 10 == 0:
                task.train_loss.update(loss)
                task.train_score.update(output, target)

            # Step profiler
            if prof is not None:
                prof.step()

        # Step optimizer
        optimizer.step()

    # Step scheduler
    scheduler.step()


def _evaluate(model, dataloaders, n_batches, tasks):
    """Internal helper. Evaluates model on a group."""
    model.eval()

    dl_iter = [iter(dl) for dl in dataloaders]

    with torch.no_grad():
        # Iterate over all batches
        for batch_idx in trange(n_batches, desc="Evaluation batches", leave=False):
            # Iterate over all tasks
            for task, dataloader in zip(tasks, dl_iter):
                data, target = next(dataloader)

                # Forward pass
                output = model(data, task.task_id)
                loss = task.loss_function(output, target)

                # Store loss value and metrics for stats
                if batch_idx % 10 == 0:
                    task.dev_loss.update(loss)
                    task.dev_score.update(output, target)


def _compute_epoch_logs(group_id, tasks):
    """Internal helper. Computes metrics and mean losses for an epoch."""
    epoch_log = {}
    for task in tasks:
        # Compute metrics
        metric_name = f"{task.task_id}_{group_id}_score"
        epoch_log[metric_name] = getattr(task, f"{group_id}_score").compute()
        getattr(task, f"{group_id}_score").reset()

        # Compute mean losses
        loss_name = f"{task.task_id}_{group_id}_loss"
        epoch_log[loss_name] = getattr(task, f"{group_id}_loss").compute()
        getattr(task, f"{group_id}_loss").reset()

    return epoch_log


def train(experiment_config: ExperimentConfig) -> torch.nn.Module:
    """
    Train a LISBET model.

    This function orchestrates the full training pipeline for LISBET, including
    data loading, model construction, training, evaluation, and saving artifacts.
    All parameters match the CLI arguments exactly.

    Parameters
    ----------
    experiment_config : ExperimentConfig
        Configuration object containing all parameters for the training run.
        It includes data paths, model architecture, training hyperparameters,
        and task definitions.

    Returns
    -------
    model : torch.nn.Module
        The trained LISBET model instance.

    Notes
    -----
    All arguments are exposed for CLI and documentation. For advanced usage,
    see the LISBET documentation.
    """
    # Create aliases for configuration parameters
    ec = experiment_config

    # Configure base runtime arguments
    run_id = datetime.now().strftime("%Y%m%d%H%M%S") if ec.run_id is None else ec.run_id

    # Create Fabric instance
    precision = "16-mixed" if ec.training.mixed_precision else "32-true"
    history_logger = CSVLogger(
        ec.output_path / "models" / run_id,
        name="training_history",
        flush_logs_every_n_steps=1,
    )
    fabric = Fabric(precision=precision, loggers=history_logger)

    logging.info("Using %s for training model %s.", fabric.device.type, run_id)

    # Configure RNGs
    run_seeds = generate_seeds(ec.seed, ec.task_ids_list)
    torch.manual_seed(run_seeds["torch"])

    # Load records
    # TODO: Switch to the DataConfig object
    multi_records = load_multi_records(
        data_format=ec.data.data_format,
        data_path=ec.data.data_path,
        data_scale=ec.data.data_scale,
        data_filter=ec.data.data_filter,
        select_coords=ec.data.select_coords,
        rename_coords=ec.data.rename_coords,
    )

    # Split records
    train_rec, dev_rec = split_multi_records(
        multi_records=multi_records,
        dev_ratio=ec.data.dev_ratio,
        dev_seed=run_seeds.get("dev_split"),
        task_ids=ec.task_ids_list,
        task_data=ec.task_data,
    )

    # Determine data shape from first record
    cdim = train_rec[ec.task_ids_list[0]][0].posetracks.coords.sizes
    feature_dim = cdim["individuals"] * cdim["keypoints"] * cdim["space"]

    # Determine input_features list for config consistency
    first_record = train_rec[ec.task_ids_list[0]][0]
    input_features = {
        dim: first_record.posetracks.coords[dim].values.tolist()
        for dim in ("individuals", "keypoints", "space")
    }

    if ec.training.load_backbone_weights is not None:
        logging.warning(
            "Loading backbone weights from %s. If you are not experimenting with "
            "transfer learning, please verify that the input features of the "
            "pre-trained model match those of your data. In the future, this warning "
            "could become a requirement to load the backbone weights.",
            ec.training.load_backbone_weights,
        )

    # Determine max sequence length
    # NOTE: We keep the max_length parameter as we may want to use it in the future to
    #       support variable-length sequences.
    max_length = ec.data.window_size

    # Compute backbone output token idx
    output_token_idx = -(ec.data.window_offset + 1)
    if not (ec.data.window_size > ec.data.window_offset >= 0):
        raise RuntimeError(
            "Window offset must be a positive integer smaller than the window size"
            f" or zero, got {ec.data.window_offset}."
        )
    logging.debug("Output token IDX = %d", output_token_idx)

    # Configure tasks
    tasks = configure_tasks(
        train_rec,
        dev_rec,
        ec.task_ids_list,
        ec.data.window_size,
        ec.data.window_offset,
        ec.backbone.embedding_dim,
        ec.backbone.hidden_dim,
        ec.training.data_augmentation,
        run_seeds,
        fabric.device,
    )
    n_tasks = len(tasks)

    # Build model
    model = _build_model(
        feature_dim,
        ec.backbone.embedding_dim,
        ec.backbone.hidden_dim,
        ec.backbone.num_heads,
        ec.backbone.num_layers,
        max_length,
        tasks,
        ec.training.load_backbone_weights,
        ec.training.freeze_backbone_weights,
    )
    model_stats = summary(model, verbose=0)
    logging.info("Model summary\n" + str(model_stats))

    # Optimizer and scheduler
    optimizer, scheduler = _configure_optimizer_and_scheduler(
        model, ec.training.learning_rate
    )

    # Save model config
    dump_model_config(
        ec.output_path,
        run_id,
        ec.data.window_size,
        ec.data.window_offset,
        feature_dim,
        ec.backbone.embedding_dim,
        ec.backbone.hidden_dim,
        ec.backbone.num_heads,
        ec.backbone.num_layers,
        max_length,
        tasks,
        input_features,
    )

    # Configure dataloaders
    train_dataloaders, train_n_batches = _configure_dataloaders(
        tasks,
        "train",
        ec.training.batch_size,
        ec.data.train_sample,
        fabric.device.type == "cuda",
    )
    if ec.data.dev_ratio is not None:
        dev_dataloaders, dev_n_batches = _configure_dataloaders(
            tasks,
            "dev",
            ec.training.batch_size,
            ec.data.dev_sample,
            fabric.device.type == "cuda",
        )

    # Configure Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloaders = [fabric.setup_dataloaders(dl) for dl in train_dataloaders]
    if ec.data.dev_ratio is not None:
        dev_dataloaders = [fabric.setup_dataloaders(dl) for dl in dev_dataloaders]

    # Training loop
    with _configure_profiler(steps_multiplier=n_tasks) as prof:
        for epoch in range(ec.training.epochs):
            history_entry = {"epoch": epoch}
            print(f"Epoch {epoch}")
            logging.info("Current LR = %f", scheduler.get_last_lr()[0])

            # Run training epoch
            _train_one_epoch(
                model,
                train_dataloaders,
                train_n_batches,
                optimizer,
                scheduler,
                tasks,
                prof,
                fabric,
            )

            # Update history entry for current epoch
            history_entry.update(_compute_epoch_logs("train", tasks))

            # Save weights, if requested
            if ec.training.save_weights == "all":
                dump_weights(model, ec.output_path, run_id, f"weights_epoch{epoch}.pt")

            if ec.data.dev_ratio is not None:
                # Run dev epoch
                _evaluate(model, dev_dataloaders, dev_n_batches, tasks)

                # Update history entry for current epoch
                history_entry.update(_compute_epoch_logs("dev", tasks))

            # Update history
            fabric.log_dict(history_entry, step=epoch)
            logging.info(", ".join(f"{k}: {v:.3f}" for k, v in history_entry.items()))

    # Save profiling results, if requested
    if prof is not None:
        dump_profiling_results(ec.output_path, run_id, prof)

    # Save final weights, if requested
    if ec.training.save_weights == "last":
        dump_weights(model, ec.output_path, run_id, "weights_last.pt")

    return model
