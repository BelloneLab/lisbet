import numpy as np
import torch
import xarray as xr

from lisbet.transforms_extra import PoseToTensor, RandomXYSwap


def make_posetracks():
    arr = np.zeros((2, 2, 1, 1), dtype=np.float32)
    arr[0, 0, 0, 0] = 1.0  # time 0, x
    arr[0, 1, 0, 0] = 2.0  # time 0, y
    arr[1, 0, 0, 0] = 3.0  # time 1, x
    arr[1, 1, 0, 0] = 4.0  # time 1, y
    ds = xr.Dataset(
        {"position": (("time", "space", "keypoints", "individuals"), arr)},
        coords={
            "time": [0, 1],
            "space": ["x", "y"],
            "keypoints": ["nose"],
            "individuals": ["mouse"],
        },
    )
    return ds


def test_randomxyswap_no_swap(monkeypatch):
    ds = make_posetracks()
    swap = RandomXYSwap(seed=42)
    # Monkeypatch torch.rand to always return >= 0.5 (no swap)
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor([0.7]))
    ds_swapped = swap(ds.copy(deep=True))
    np.testing.assert_array_equal(ds_swapped["position"].values, ds["position"].values)
    assert (ds_swapped.space.values == ["x", "y"]).all()


def test_randomxyswap_swap(monkeypatch):
    ds = make_posetracks()
    swap = RandomXYSwap(seed=42)
    # Monkeypatch torch.rand to always return < 0.5 (swap)
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor([0.2]))
    ds_swapped = swap(ds.copy(deep=True))
    assert (ds_swapped.space.values == ["y", "x"]).all()
    orig = ds["position"].values
    swapped = ds_swapped["position"].values
    np.testing.assert_array_equal(swapped[:, 0], orig[:, 1])
    np.testing.assert_array_equal(swapped[:, 1], orig[:, 0])
    tensor = PoseToTensor()(ds_swapped)
    assert np.allclose(tensor[0, 0], 2.0)
    assert np.allclose(tensor[0, 1], 1.0)
    assert np.allclose(tensor[1, 0], 4.0)
    assert np.allclose(tensor[1, 1], 3.0)


def test_randomxyswap_does_not_modify_original(monkeypatch):
    ds = make_posetracks()
    swap = RandomXYSwap(seed=0)
    ds_copy = ds.copy(deep=True)
    # Monkeypatch to always swap
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor([0.2]))
    _ = swap(ds_copy)
    # Original ds should remain unchanged
    np.testing.assert_array_equal(
        ds["position"].values, make_posetracks()["position"].values
    )
