"""Tests for the ``window_sampling`` option of the self-supervised tasks ("any" vs "inside")."""

import itertools

import numpy as np
import pytest
import torch
import xarray as xr
from pydantic import ValidationError

from lisbet.config.schemas import TrainingConfig
from lisbet.datasets import (
    GeometricInvarianceDataset,
    GroupConsistencyDataset,
    TemporalOrderDataset,
    TemporalShiftDataset,
    TemporalWarpDataset,
)
from lisbet.datasets.common import leakfree_config
from lisbet.training.tasks import configure_tasks
from lisbet.training.utils import generate_seeds

SWITCHES = [
    "LISBET_PAD_MODE",
    "LISBET_SHIFT_MODE",
    "LISBET_SHIFT_SIGN",
    "LISBET_MAX_SHIFT",
    "LISBET_MIN_SHIFT",
    "LISBET_GEOM_PAD",
    "LISBET_CONS_NEG",
    "LISBET_CONS_MIN_GAP",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in SWITCHES:
        monkeypatch.delenv(k, raising=False)


class _Rec:
    """Minimal record: constant pose of 1.0 (so any zero is padding) or of a per-record constant."""

    def __init__(self, n_frames, value=1.0):
        pos = np.full((n_frames, 2, 3, 2), value, dtype=np.float32)
        self.posetracks = xr.Dataset(
            {"position": (("time", "individuals", "keypoints", "space"), pos)},
            coords={"individuals": ["a", "b"]},
        )
        self.id = f"rec{n_frames}-{value}"


def _records(lengths=(5, 30, 60, 120, 200, 400)):
    return [_Rec(n) for n in lengths]  # the shortest ones cannot hold a window and are never drawn


W, OFF = 20, 10


def _datasets(records, window_sampling, seed=3):
    kw = dict(
        window_size=W,
        window_offset=OFF,
        engine="numpy",
        base_seed=seed,
        window_sampling=window_sampling,
    )
    return {
        "cons": GroupConsistencyDataset(records, **kw),
        "order": TemporalOrderDataset(records, **kw),
        "shift": TemporalShiftDataset(records, max_shift=8, **kw),
        "warp": TemporalWarpDataset(records, **kw),
        "geom": GeometricInvarianceDataset(records, **kw),
    }


def _min_values(name, ds, n=400):
    if name == "geom":  # content stays at 1.0 (mirror / zoom / translate would create zeros)
        ds._apply_geometric_transform = lambda x: x
    out = []
    for s in itertools.islice(iter(ds), n):
        xs = s if name == "geom" else (s[0],)
        out.append(min(float(np.min(x)) for x in xs))
    return np.array(out)


@pytest.mark.parametrize("name", ["cons", "order", "shift", "warp", "geom"])
def test_inside_never_pads(name):
    ds = _datasets(_records(), "inside")[name]
    assert _min_values(name, ds).min() > 0.999


@pytest.mark.parametrize("name", ["cons", "order", "shift", "warp"])
def test_any_pads_sometimes(name):
    ds = _datasets(_records(), "any")[name]
    assert (_min_values(name, ds, n=600) < 0.999).mean() > 0.05


def test_inside_shift_sign_is_balanced():
    ds = _datasets(_records(), "inside")["shift"]
    y = np.array([float(lab[0]) for _, lab in itertools.islice(iter(ds), 3000)])
    assert abs(y.mean() - 0.5) < 0.04


def test_default_is_any():
    assert leakfree_config()["pad_mode"] == "none"
    assert leakfree_config("any")["shift_sign"] == "original"
    cfg = leakfree_config("inside")
    assert (cfg["pad_mode"], cfg["shift_sign"], cfg["geom_pad"]) == ("reject", "balanced", "reject")


def test_invalid_window_sampling_raises():
    with pytest.raises(ValueError):
        leakfree_config("sometimes")
    with pytest.raises(ValidationError):
        TrainingConfig(epochs=1, batch_size=1, learning_rate=0.1, window_sampling="sometimes")


def test_training_config_option():
    cfg = TrainingConfig(epochs=1, batch_size=1, learning_rate=0.1)
    assert cfg.window_sampling == "any"
    cfg = TrainingConfig(epochs=1, batch_size=1, learning_rate=0.1, window_sampling="inside")
    assert cfg.window_sampling == "inside"


def test_environment_overrides_window_sampling(monkeypatch):
    monkeypatch.setenv("LISBET_PAD_MODE", "none")
    assert leakfree_config("inside")["pad_mode"] == "none"


def test_configure_tasks_passes_window_sampling():
    task_ids = ["cons", "order", "shift", "warp", "geom"]
    recs = _records()
    tasks = configure_tasks(
        train_rec={t: recs for t in task_ids},
        dev_rec={t: recs for t in task_ids},
        task_ids=task_ids,
        window_size=W,
        window_offset=OFF,
        embedding_dim=4,
        hidden_dim=4,
        data_augmentation=None,
        run_seeds=generate_seeds(5, task_ids),
        device=torch.device("cpu"),
        window_sampling="inside",
    )
    for task in tasks:
        assert task.train_dataset.lf["pad_mode"] == "reject"
        assert task.dev_dataset.lf["pad_mode"] == "reject"
        next(iter(task.train_dataset))
        next(iter(task.dev_dataset))
