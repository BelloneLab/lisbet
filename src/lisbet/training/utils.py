"""Utility functions for model training."""

import hashlib
import logging
import struct

import numpy as np
import torch
import torch.distributed as dist
from lightning.fabric.utilities.data import suggested_max_num_workers
from torch.utils.data import get_worker_info


def generate_seeds(seed, task_ids):
    """Internal helper. Generates multiple seeds from the base one."""
    rng = np.random.default_rng(seed)
    seed_keys = (
        ["torch", "dev_split", "test_split"]
        + [
            f"{group}_shuffle_{task_id}"
            for task_id in task_ids
            for group in ("train", "dev", "test")
        ]
        + [f"transform_{task_id}" for task_id in task_ids]
        + [f"dataset_{task_id}" for task_id in task_ids]
    )
    run_seeds = {sk: rng.integers(low=0, high=2**31 - 1, dtype=int) for sk in seed_keys}

    logging.debug("Generated seeds: %s", run_seeds)

    return run_seeds


def estimate_num_workers(n_tasks, batch_size, batch_size_per_worker=8):
    """
    Estimate the optimal number of DataLoader worker processes to use, based on the
    number of training tasks, the batch size, and the desired batch size per worker.

    Parameters
    ----------
    n_tasks : int
        The number of training tasks (e.g., datasets or splits) being processed.
    batch_size : int
        The total batch size used for loading data.
    batch_size_per_worker : int, optional
        The target batch size to be handled by each worker process (default: 8).

    Returns
    -------
    num_workers : int
        The estimated number of DataLoader worker processes to use.
    """
    # Estimate number of workers
    local_world_size = 1  # torch.distributed.get_world_size() in distributed training

    max_workers = suggested_max_num_workers(local_world_size) + 1
    fair_share = max_workers // max(1, n_tasks)
    batch_cap = max(1, batch_size // batch_size_per_worker)

    num_workers = max(1, min(max_workers, fair_share, batch_cap))

    logging.debug(
        "Estimated num_workers: %d (max_workers: %d, fair_share: %d, batch_cap: %d)",
        num_workers,
        max_workers,
        fair_share,
        batch_cap,
    )

    return num_workers


def worker_init_fn(worker_id: int):
    """
    Worker initialization function for DataLoader.

    This function sets a unique random seed for each DataLoader worker based on the
    worker ID, global rank, and task seed. This ensures that each worker operates with
    a different random seed, which is crucial for data shuffling and augmentation in
    distributed training scenarios.

    Parameters
    ----------
    worker_id : int
        The ID of the worker being initialized. This is typically an integer
        in the range [0, num_workers - 1].
    """
    info = get_worker_info()
    ds = info.dataset
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Generate a unique seed for the DataLoader worker
    payload = struct.pack(">IHH", ds.base_seed, rank, worker_id)
    seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")

    ds.g = torch.Generator().manual_seed(seed)

    if rank == 0:
        logging.debug(
            "Worker %d initialized with seed %d (base seed: %d, rank: %d)",
            worker_id,
            seed,
            ds.base_seed,
            rank,
        )
