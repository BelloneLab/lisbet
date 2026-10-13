"""Legacy golden test: the default ``space_layout`` ("legacy") must reproduce lisbet 77dd947 bit for bit.

The reference hashes in ``tests/golden/legacy_samples.json`` were produced by running this file as a script
on commit 77dd947 (before the 3D layouts existed)::

    python tests/test_legacy_golden.py --write

Each case draws samples from the self-supervised tasks (cons, order, shift, warp, geom) built through
``configure_tasks`` with a fixed seed, on synthetic 2D and 3D dyads, and hashes the float64 bytes of every
tensor. Hashes are platform dependent in principle (floating point); they were written on Windows / CPU.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from lisbet.cli.commands.train import parse_data_augmentation
from lisbet.config.schemas import DataAugmentationConfig
from lisbet.training.tasks import configure_tasks
from lisbet.training.utils import generate_seeds

GOLDEN = Path(__file__).parent / "golden" / "legacy_samples.json"

TASKS = ("cons", "order", "shift", "warp", "geom")
N_SAMPLES = 40
W, OFF = 20, 10

AUGMENTATIONS = {
    "none": "",
    # humanlisbet-code wrapper default
    "wrapper_default": "all_translate:p=0.1,all_mirror_x:p=0.1,all_zoom:p=0.1",
    # every geometric / noise augmentation, always applied. Not included because they crash on
    # the numpy engine at 77dd947: blk_perm_id (RandomBlockPermutation has no _apply_numpy) and
    # rotation (RandomRotation reads posetracks["position"] on a NumPy array)
    "all": (
        "all_translate,all_mirror_x,all_zoom,"
        "gauss_jitter:sigma=0.01,all_perm_ax,all_perm_id,kp_ablation:pB=0.05"
    ),
}


class _Rec:
    def __init__(self, idx, n_frames, n_space, seed):
        rng = np.random.default_rng(seed)
        steps = rng.normal(0.0, 0.01, size=(n_frames, 2, 5, n_space))
        start = rng.uniform(0.3, 0.7, size=(1, 2, 5, n_space))
        pos = np.clip(start + np.cumsum(steps, axis=0), 0.05, 0.95)
        self.posetracks = xr.Dataset(
            {"position": (("time", "individuals", "keypoints", "space"), pos)},
            coords={
                "individuals": ["child", "adult"],
                "keypoints": [f"k{i}" for i in range(5)],
                "space": ["x", "y", "z"][:n_space],
            },
        )
        self.id = f"rec{idx}"


def _records(n_space):
    return [
        _Rec(i, n, n_space, seed=100 * n_space + i)
        for i, n in enumerate((60, 120, 200, 300))
    ]


def _hash(x):
    a = np.asarray(x.numpy() if hasattr(x, "numpy") else x, dtype=np.float64)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def compute_case(n_space, aug_name, window_sampling):
    records = _records(n_space)
    run_seeds = generate_seeds(1789, list(TASKS))
    torch.manual_seed(run_seeds["torch"])  # RandomApply draws from the global generator, as in training
    tasks = configure_tasks(
        train_rec={t: records for t in TASKS},
        dev_rec={t: [] for t in TASKS},
        task_ids=list(TASKS),
        window_size=W,
        window_offset=OFF,
        embedding_dim=8,
        hidden_dim=8,
        data_augmentation=[
            DataAugmentationConfig(**d)
            for d in parse_data_augmentation(AUGMENTATIONS[aug_name]) or []
        ],
        run_seeds=run_seeds,
        device="cpu",
        window_sampling=window_sampling,
    )
    out = {}
    for task in tasks:
        it = iter(task.train_dataset)
        hashes = []
        for _ in range(N_SAMPLES):
            a, b = next(it)
            hashes.append(_hash(a) + ":" + _hash(b))
        out[task.task_id] = hashes
    return out


CASES = [
    (n_space, aug, ws)
    for n_space in (2, 3)
    for aug in AUGMENTATIONS
    for ws in ("any", "inside")
]


def _key(n_space, aug, ws):
    return f"{n_space}d|{aug}|{ws}"


@pytest.mark.parametrize("n_space,aug,ws", CASES, ids=[_key(*c) for c in CASES])
def test_legacy_bit_identical(n_space, aug, ws):
    golden = json.loads(GOLDEN.read_text())
    got = compute_case(n_space, aug, ws)
    assert got == golden[_key(n_space, aug, ws)]


if __name__ == "__main__":
    if "--write" not in sys.argv:
        sys.exit("usage: python tests/test_legacy_golden.py --write")
    GOLDEN.parent.mkdir(exist_ok=True)
    data = {_key(*c): compute_case(*c) for c in CASES}
    GOLDEN.write_text(json.dumps(data, indent=1))
    print(f"wrote {len(data)} cases to {GOLDEN}")
