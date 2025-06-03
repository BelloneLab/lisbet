"""Utility functions for model training."""

import logging

import numpy as np


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
        + [f"dataset_{task_id}" for task_id in task_ids if task_id != "cfc"]
    )
    run_seeds = {sk: rng.integers(low=0, high=2**32) for sk in seed_keys}

    logging.debug("Generated seeds: %s", run_seeds)

    return run_seeds
