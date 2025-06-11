import numpy as np
import torch
import xarray as xr

from lisbet.transforms_extra import PoseToTensor, RandomPermutation, RandomXYSwap


def make_posetracks(n_individuals=2):
    arr = np.zeros((2, 2, 1, n_individuals), dtype=np.float32)
    for i in range(n_individuals):
        arr[0, 0, 0, i] = 1.0 + i  # time 0, x
        arr[0, 1, 0, i] = 2.0 + i  # time 0, y
        arr[1, 0, 0, i] = 3.0 + i  # time 1, x
        arr[1, 1, 0, i] = 4.0 + i  # time 1, y
    ds = xr.Dataset(
        {"position": (("time", "space", "keypoints", "individuals"), arr)},
        coords={
            "time": [0, 1],
            "space": ["x", "y"],
            "keypoints": ["nose"],
            "individuals": [f"ind{i}" for i in range(n_individuals)],
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


def test_randompermutation_all_individuals_present(monkeypatch):
    """
    Test that RandomPermutation on 'individuals' keeps all individuals, shuffles order,
    and preserves data.
    """
    ds = make_posetracks(n_individuals=3)
    permute = RandomPermutation(seed=123, coordinate="individuals")
    # Monkeypatch torch.rand to always return < 0.5 (force permutation)
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor([0.2]))
    # Monkeypatch torch.randperm to a known permutation
    monkeypatch.setattr(
        torch, "randperm", lambda n, generator=None: torch.tensor([2, 0, 1])
    )
    ds_permuted = permute(ds.copy(deep=True))
    # All individuals should be present, just reordered
    orig_inds = list(ds.individuals.values)
    permuted_inds = list(ds_permuted.individuals.values)
    assert set(orig_inds) == set(permuted_inds)
    assert orig_inds != permuted_inds  # Should be permuted
    # Data for each individual should match the original (just reordered)
    for i, ind in enumerate(permuted_inds):
        orig_idx = orig_inds.index(ind)
        np.testing.assert_array_equal(
            ds_permuted["position"].values[..., i],
            ds["position"].values[..., orig_idx],
        )


def test_randompermutation_no_permutation(monkeypatch):
    """
    Test that RandomPermutation does nothing when random value >= 0.5.
    """
    ds = make_posetracks(n_individuals=3)
    permute = RandomPermutation(seed=123, coordinate="individuals")
    # Monkeypatch torch.rand to always return >= 0.5 (no permutation)
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor([0.7]))
    ds_permuted = permute(ds.copy(deep=True))
    # Should be identical to original
    np.testing.assert_array_equal(ds_permuted["position"].values, ds["position"].values)
    assert list(ds_permuted.individuals.values) == list(ds.individuals.values)


def test_randompermutation_nothing_new(monkeypatch):
    """
    Test that RandomPermutation does not introduce new coordinate values or NaNs.
    """
    ds = make_posetracks(n_individuals=4)
    permute = RandomPermutation(seed=42, coordinate="individuals")
    # Monkeypatch torch.rand to always return < 0.5 (force permutation)
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor([0.1]))
    # Monkeypatch torch.randperm to a known permutation
    monkeypatch.setattr(
        torch, "randperm", lambda n, generator=None: torch.tensor([3, 2, 1, 0])
    )
    ds_permuted = permute(ds.copy(deep=True))
    # No new individuals introduced
    orig_inds = set(ds.individuals.values)
    permuted_inds = set(ds_permuted.individuals.values)
    assert permuted_inds == orig_inds
    # No NaNs or unexpected values in data
    assert not np.isnan(ds_permuted["position"].values).any()


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
