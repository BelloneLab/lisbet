"""Utility functions for model training."""

import hashlib
import logging
import struct

import numpy as np
import torch
import torch.distributed as dist
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
