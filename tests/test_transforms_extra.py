import numpy as np
import pytest
import torch
import xarray as xr

from lisbet.transforms_extra import (
    GaussianJitter,
    KeypointAblation,
    RandomBlockPermutation,
    RandomPermutation,
)


def make_posetracks(n_individuals=2, n_time=10):
    """Helper to create a posetracks dataset for testing."""
    arr = np.zeros((n_time, 2, 1, n_individuals), dtype=np.float32)
    for t in range(n_time):
        for i in range(n_individuals):
            # Unique values for each time, individual, and space
            arr[t, 0, 0, i] = 1.0 + i + t * 10  # x
            arr[t, 1, 0, i] = 2.0 + i + t * 10  # y

    ds = xr.Dataset(
        {"position": (("time", "space", "keypoints", "individuals"), arr)},
        coords={
            "time": np.arange(n_time),
            "space": ["x", "y"],
            "keypoints": ["nose"],
            "individuals": [f"ind{i}" for i in range(n_individuals)],
        },
    )
    return ds


def test_randompermutation_individuals(monkeypatch):
    """Test RandomPermutation on 'individuals' coordinate for the whole clip."""
    ds = make_posetracks(n_individuals=3, n_time=2)
    permute = RandomPermutation(seed=123, coordinate="individuals")

    # Monkeypatch torch.randperm to a known permutation
    monkeypatch.setattr(
        torch, "randperm", lambda n, generator=None: torch.tensor([2, 0, 1])
    )

    ds_permuted = permute(ds.copy(deep=True))

    # Check that individuals are permuted in the new object's coordinates
    orig_inds = list(ds.individuals.values)
    permuted_inds = list(ds_permuted.individuals.values)
    assert permuted_inds == ["ind2", "ind0", "ind1"]
    assert set(orig_inds) == set(permuted_inds)

    # Check that data is permuted accordingly in the underlying array
    for i, ind_name in enumerate(permuted_inds):
        orig_idx = orig_inds.index(ind_name)
        np.testing.assert_array_equal(
            ds_permuted["position"].values[..., i],
            ds["position"].values[..., orig_idx],
        )


def test_randompermutation_space_swap(monkeypatch):
    """Test RandomPermutation on 'space' coordinate to swap x and y."""
    ds = make_posetracks(n_time=2)
    permute = RandomPermutation(seed=42, coordinate="space")

    # Monkeypatch torch.randperm to force a swap
    monkeypatch.setattr(
        torch, "randperm", lambda n, generator=None: torch.tensor([1, 0])
    )

    ds_swapped = permute(ds.copy(deep=True))

    # Check that space coordinates are swapped
    assert list(ds_swapped.space.values) == ["y", "x"]

    # Check that data is swapped
    orig = ds["position"].values
    swapped = ds_swapped["position"].values
    np.testing.assert_array_equal(swapped[:, 0, ...], orig[:, 1, ...])
    np.testing.assert_array_equal(swapped[:, 1, ...], orig[:, 0, ...])


def test_randompermutation_space_no_swap(monkeypatch):
    """Test RandomPermutation on 'space' with no swap (identity permutation)."""
    ds = make_posetracks(n_time=2)
    permute = RandomPermutation(seed=42, coordinate="space")

    # Monkeypatch torch.randperm to return identity
    monkeypatch.setattr(
        torch, "randperm", lambda n, generator=None: torch.tensor([0, 1])
    )

    ds_no_swap = permute(ds.copy(deep=True))

    # Should be identical to original
    xr.testing.assert_equal(ds_no_swap, ds)


def test_randomblockpermutation_basic(monkeypatch):
    """Test RandomBlockPermutation with permute_fraction."""
    n_time = 20
    ds = make_posetracks(n_individuals=2, n_time=n_time)

    # Permute 20% of the frames, so 4 frames.
    permute = RandomBlockPermutation(
        seed=1, coordinate="individuals", permute_fraction=0.2
    )

    # Monkeypatch randperm to a known permutation (swap)
    monkeypatch.setattr(
        torch, "randperm", lambda n, generator=None: torch.tensor([1, 0])
    )
    # Monkeypatch randint to a known start for the block
    monkeypatch.setattr(
        torch, "randint", lambda low, high, size, generator=None: torch.tensor([10])
    )

    ds_permuted = permute(ds.copy(deep=True))

    block_size = int(0.2 * n_time)  # 4
    start_idx = 10
    end_idx = start_idx + block_size  # 14

    # Check data before the block (should be unchanged)
    xr.testing.assert_equal(
        ds_permuted.isel(time=slice(0, start_idx)), ds.isel(time=slice(0, start_idx))
    )

    # Check data after the block (should be unchanged)
    xr.testing.assert_equal(
        ds_permuted.isel(time=slice(end_idx, None)), ds.isel(time=slice(end_idx, None))
    )

    # Check data inside the block (should be permuted)
    block_orig = ds.isel(time=slice(start_idx, end_idx))
    block_permuted = ds_permuted.isel(time=slice(start_idx, end_idx))

    # In the permuted block, data for ind0 should be from original ind1, and vice-versa
    np.testing.assert_array_equal(
        block_permuted.sel(individuals="ind0")["position"],
        block_orig.sel(individuals="ind1")["position"],
    )
    np.testing.assert_array_equal(
        block_permuted.sel(individuals="ind1")["position"],
        block_orig.sel(individuals="ind0")["position"],
    )
    # The coordinates of the final dataset should be the same as original
    assert list(ds_permuted.individuals.values) == list(ds.individuals.values)


def test_randomblockpermutation_zero_block():
    """Test that nothing happens if permute_fraction results in a zero-sized block."""
    ds = make_posetracks(n_time=4)
    # This will result in block_size = int(0.1 * 4) = 0
    permute = RandomBlockPermutation(
        seed=1, coordinate="individuals", permute_fraction=0.1
    )
    ds_permuted = permute(ds.copy(deep=True))
    xr.testing.assert_equal(ds_permuted, ds)


def test_randompermutation_full_track(monkeypatch):
    """Test RandomPermutation permutes the whole track."""
    ds = make_posetracks(n_individuals=2, n_time=4)

    # Use a known permutation
    monkeypatch.setattr(
        torch, "randperm", lambda n, generator=None: torch.tensor([1, 0])
    )

    permute_none = RandomPermutation(seed=123, coordinate="individuals")
    ds_permuted_none = permute_none(ds.copy(deep=True))

    # Check that it's not equal to original
    assert not ds.equals(ds_permuted_none)

    # Check that coordinates are swapped
    assert list(ds_permuted_none.individuals.values) == ["ind1", "ind0"]

    # Check that data follows the swapped coordinates
    # The data at position 0 should be from original position 1 (which was "ind1")
    np.testing.assert_array_equal(
        ds_permuted_none["position"].values[..., 0],
        ds["position"].values[..., 1],
    )
    np.testing.assert_array_equal(
        ds_permuted_none["position"].values[..., 1],
        ds["position"].values[..., 0],
    )


def test_randomblockpermutation_invalid_fraction():
    """Test for ValueError with invalid permute_fraction."""
    with pytest.raises(
        ValueError, match="permute_fraction must be a float between 0 and 1."
    ):
        RandomBlockPermutation(seed=1, permute_fraction=1.5)
    with pytest.raises(
        ValueError, match="permute_fraction must be a float between 0 and 1."
    ):
        RandomBlockPermutation(seed=1, permute_fraction=0.0)
    with pytest.raises(
        ValueError, match="permute_fraction must be a float between 0 and 1."
    ):
        RandomBlockPermutation(seed=1, permute_fraction=1.0)
    with pytest.raises(
        ValueError, match="permute_fraction must be a float between 0 and 1."
    ):
        RandomBlockPermutation(seed=1, permute_fraction=-0.5)


def test_randompermutation_does_not_modify_original():
    """Test that RandomPermutation does not modify the original dataset."""
    ds = make_posetracks()
    ds_original = ds.copy(deep=True)
    permute = RandomPermutation(seed=0, coordinate="individuals")
    _ = permute(ds)
    # Original ds should remain unchanged
    xr.testing.assert_equal(ds, ds_original)


def test_randomblockpermutation_does_not_modify_original():
    """Test that RandomBlockPermutation does not modify the original dataset."""
    ds = make_posetracks()
    ds_original = ds.copy(deep=True)
    permute = RandomBlockPermutation(
        seed=0, coordinate="individuals", permute_fraction=0.5
    )
    _ = permute(ds)
    # Original ds should remain unchanged
    xr.testing.assert_equal(ds, ds_original)


def test_randompermutation_keypoints(monkeypatch):
    """Test RandomPermutation on 'keypoints' coordinate."""
    # Create a dataset with multiple keypoints
    arr = np.zeros((5, 2, 3, 2), dtype=np.float32)
    for t in range(5):
        for k in range(3):
            for i in range(2):
                arr[t, 0, k, i] = 1.0 + k + i * 10 + t * 100  # x
                arr[t, 1, k, i] = 2.0 + k + i * 10 + t * 100  # y

    ds = xr.Dataset(
        {"position": (("time", "space", "keypoints", "individuals"), arr)},
        coords={
            "time": np.arange(5),
            "space": ["x", "y"],
            "keypoints": ["nose", "ear", "tail"],
            "individuals": ["ind0", "ind1"],
        },
    )

    permute = RandomPermutation(seed=42, coordinate="keypoints")

    # Monkeypatch to a known permutation
    monkeypatch.setattr(
        torch, "randperm", lambda n, generator=None: torch.tensor([2, 0, 1])
    )

    ds_permuted = permute(ds.copy(deep=True))

    # Check that keypoints are permuted
    assert list(ds_permuted.keypoints.values) == ["tail", "nose", "ear"]

    # Check that data is permuted accordingly
    np.testing.assert_array_equal(
        ds_permuted["position"].values[:, :, 0, :],  # Position 0 (tail)
        ds["position"].values[:, :, 2, :],  # Original position 2 (tail)
    )


def test_randompermutation_single_element():
    """Test RandomPermutation with only one element in the coordinate."""
    ds = make_posetracks(n_individuals=1, n_time=5)
    permute = RandomPermutation(seed=123, coordinate="individuals")

    ds_permuted = permute(ds.copy(deep=True))

    # Should be identical (permutation of 1 element is identity)
    xr.testing.assert_equal(ds_permuted, ds)


def test_randompermutation_reproducibility():
    """Test that the same seed produces the same permutation."""
    ds = make_posetracks(n_individuals=3, n_time=10)

    permute1 = RandomPermutation(seed=999, coordinate="individuals")
    permute2 = RandomPermutation(seed=999, coordinate="individuals")

    ds_permuted1 = permute1(ds.copy(deep=True))
    ds_permuted2 = permute2(ds.copy(deep=True))

    # Both should produce identical results
    xr.testing.assert_equal(ds_permuted1, ds_permuted2)


def test_randompermutation_different_seeds():
    """Test that different seeds produce different permutations (probabilistically)."""
    ds = make_posetracks(n_individuals=5, n_time=10)

    permute1 = RandomPermutation(seed=111, coordinate="individuals")
    permute2 = RandomPermutation(seed=222, coordinate="individuals")

    ds_permuted1 = permute1(ds.copy(deep=True))
    ds_permuted2 = permute2(ds.copy(deep=True))

    # Very unlikely to be equal with 5 individuals... probability of same permutation is
    # 1/120
    assert not ds_permuted1.equals(ds_permuted2)


def test_randomblockpermutation_at_start(monkeypatch):
    """Test RandomBlockPermutation when block starts at the beginning."""
    n_time = 20
    ds = make_posetracks(n_individuals=2, n_time=n_time)

    permute = RandomBlockPermutation(
        seed=1, coordinate="individuals", permute_fraction=0.3
    )

    # Force the block to start at index 0
    monkeypatch.setattr(
        torch, "randint", lambda low, high, size, generator=None: torch.tensor([0])
    )
    monkeypatch.setattr(
        torch, "randperm", lambda n, generator=None: torch.tensor([1, 0])
    )

    ds_permuted = permute(ds.copy(deep=True))

    block_size = int(0.3 * n_time)  # 6

    # Check that the first block is permuted
    block_orig = ds.isel(time=slice(0, block_size))
    block_permuted = ds_permuted.isel(time=slice(0, block_size))

    np.testing.assert_array_equal(
        block_permuted.sel(individuals="ind0")["position"],
        block_orig.sel(individuals="ind1")["position"],
    )

    # Check that data after the block is unchanged
    xr.testing.assert_equal(
        ds_permuted.isel(time=slice(block_size, None)),
        ds.isel(time=slice(block_size, None)),
    )


def test_randomblockpermutation_at_end(monkeypatch):
    """Test RandomBlockPermutation when block ends at the end of the time window."""
    n_time = 20
    ds = make_posetracks(n_individuals=2, n_time=n_time)

    permute = RandomBlockPermutation(
        seed=1, coordinate="individuals", permute_fraction=0.25
    )

    block_size = int(0.25 * n_time)  # 5
    start_idx = n_time - block_size  # 15

    # Force the block to end at the last frame
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, generator=None: torch.tensor([start_idx]),
    )
    monkeypatch.setattr(
        torch, "randperm", lambda n, generator=None: torch.tensor([1, 0])
    )

    ds_permuted = permute(ds.copy(deep=True))

    # Check that data before the block is unchanged
    xr.testing.assert_equal(
        ds_permuted.isel(time=slice(0, start_idx)), ds.isel(time=slice(0, start_idx))
    )

    # Check that the last block is permuted
    block_orig = ds.isel(time=slice(start_idx, None))
    block_permuted = ds_permuted.isel(time=slice(start_idx, None))

    np.testing.assert_array_equal(
        block_permuted.sel(individuals="ind0")["position"],
        block_orig.sel(individuals="ind1")["position"],
    )


def test_randompermutation_large_permutation(monkeypatch):
    """Test RandomPermutation with a larger number of elements."""
    # Create dataset with 10 individuals
    arr = np.zeros((5, 2, 1, 10), dtype=np.float32)
    for t in range(5):
        for i in range(10):
            arr[t, 0, 0, i] = 1.0 + i + t * 100
            arr[t, 1, 0, i] = 2.0 + i + t * 100

    ds = xr.Dataset(
        {"position": (("time", "space", "keypoints", "individuals"), arr)},
        coords={
            "time": np.arange(5),
            "space": ["x", "y"],
            "keypoints": ["nose"],
            "individuals": [f"ind{i}" for i in range(10)],
        },
    )

    # Use a known permutation (reverse order)
    perm = list(range(9, -1, -1))  # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    monkeypatch.setattr(torch, "randperm", lambda n, generator=None: torch.tensor(perm))

    permute = RandomPermutation(seed=42, coordinate="individuals")
    ds_permuted = permute(ds.copy(deep=True))

    # Check that coordinates are reversed
    expected_inds = [f"ind{i}" for i in range(9, -1, -1)]
    assert list(ds_permuted.individuals.values) == expected_inds

    # Check that first and last data are swapped
    np.testing.assert_array_equal(
        ds_permuted["position"].values[..., 0],
        ds["position"].values[..., 9],
    )
    np.testing.assert_array_equal(
        ds_permuted["position"].values[..., 9],
        ds["position"].values[..., 0],
    )


def test_gaussian_jitter_basic_mask_properties():
    """
    GaussianJitter should approximately modify p proportion of elements (excluding
    space).
    """
    T, S, K, I = 50, 2, 4, 3  # noqa: E741
    rng = np.random.default_rng(1789)
    arr = rng.random((T, S, K, I), dtype=np.float32)
    ds = xr.Dataset(
        {"position": (("time", "space", "keypoints", "individuals"), arr)},
        coords={
            "time": np.arange(T),
            "space": ["x", "y"],
            "keypoints": [f"kp{k}" for k in range(K)],
            "individuals": [f"ind{i}" for i in range(I)],
        },
    )
    sigma = 0.2
    gj = GaussianJitter(seed=123, sigma=sigma)
    ds_j = gj(ds.copy(deep=True))

    diff = ds_j["position"].values - ds["position"].values
    # Collapse space by max abs to detect any change per element
    changed = np.any(np.abs(diff) > 1e-9, axis=1)  # shape (T, K, I)
    proportion_changed = changed.mean()
    print(f"Proportion changed: {proportion_changed:.3f}")
    assert 0.95 < proportion_changed < 1.1  # loose bounds around p=0.1
    # Check noise statistics roughly
    if changed.sum() > 0:
        observed = diff[:, :, :, :][np.abs(diff) > 1e-9]
        # Mean near 0
        assert abs(observed.mean()) < sigma * 0.2


def test_gaussian_jitter_determinism():
    T, S, K, I = 20, 2, 3, 2  # noqa: E741
    rng = np.random.default_rng(1789)
    arr = rng.random((T, S, K, I)).astype(np.float32)
    ds = xr.Dataset(
        {"position": (("time", "space", "keypoints", "individuals"), arr)},
        coords={
            "time": np.arange(T),
            "space": ["x", "y"],
            "keypoints": [f"kp{k}" for k in range(K)],
            "individuals": [f"ind{i}" for i in range(I)],
        },
    )
    gj1 = GaussianJitter(seed=999,sigma=0.01)
    gj2 = GaussianJitter(seed=999, sigma=0.01)
    out1 = gj1(ds.copy(deep=True))
    out2 = gj2(ds.copy(deep=True))
    xr.testing.assert_equal(out1, out2)



def test_keypoint_ablation_basic():
    """Test KeypointAblation sets selected elements to NaN."""
    T, S, K, I = 50, 2, 4, 3  # noqa: E741
    rng = np.random.default_rng(1789)
    arr = rng.random((T, S, K, I), dtype=np.float32)
    ds = xr.Dataset(
        {"position": (("time", "space", "keypoints", "individuals"), arr)},
        coords={
            "time": np.arange(T),
            "space": ["x", "y"],
            "keypoints": [f"kp{k}" for k in range(K)],
            "individuals": [f"ind{i}" for i in range(I)],
        },
    )
    pB = 0.1
    kp_abl = KeypointAblation(seed=123, pB=pB)
    ds_abl = kp_abl(ds.copy(deep=True))

    # Check that some elements are NaN
    pos_orig = ds["position"].values
    pos_abl = ds_abl["position"].values

    # An element is ablated if all its space coordinates are NaN
    # Shape: (T, S, K, I)
    ablated_elements = np.all(np.isnan(pos_abl), axis=1)  # shape (T, K, I)

    # Check that we have some ablation
    assert ablated_elements.sum() > 0, "No keypoints were ablated"

    # Check proportion is roughly around p
    proportion_ablated = ablated_elements.mean()
    print(f"Proportion ablated: {proportion_ablated:.3f}")
    assert 0.05 < proportion_ablated < 0.2  # loose bounds around p=0.1

    # Check that non-ablated elements remain unchanged
    for t in range(T):
        for k in range(K):
            for i in range(I):
                if not ablated_elements[t, k, i]:
                    np.testing.assert_array_equal(
                        pos_abl[t, :, k, i], pos_orig[t, :, k, i]
                    )


def test_keypoint_ablation_determinism():
    """Test KeypointAblation produces consistent results with same seed."""
    T, S, K, I = 20, 2, 3, 2  # noqa: E741
    rng = np.random.default_rng(1789)
    arr = rng.random((T, S, K, I)).astype(np.float32)
    ds = xr.Dataset(
        {"position": (("time", "space", "keypoints", "individuals"), arr)},
        coords={
            "time": np.arange(T),
            "space": ["x", "y"],
            "keypoints": [f"kp{k}" for k in range(K)],
            "individuals": [f"ind{i}" for i in range(I)],
        },
    )
    kp_abl1 = KeypointAblation(seed=999, pB=0.2)
    kp_abl2 = KeypointAblation(seed=999, pB=0.2)
    out1 = kp_abl1(ds.copy(deep=True))
    out2 = kp_abl2(ds.copy(deep=True))

    # Both should have NaN in the same positions
    nan_mask1 = np.isnan(out1["position"].values)
    nan_mask2 = np.isnan(out2["position"].values)
    np.testing.assert_array_equal(nan_mask1, nan_mask2)


def test_keypoint_ablation_no_change_when_p_zero():
    """Test KeypointAblation doesn't ablate when p=0."""
    T, S, K, I = 30, 2, 2, 2  # noqa: E741
    rng = np.random.default_rng(1789)
    arr = rng.random((T, S, K, I)).astype(np.float32)
    ds = xr.Dataset(
        {"position": (("time", "space", "keypoints", "individuals"), arr)},
        coords={
            "time": np.arange(T),
            "space": ["x", "y"],
            "keypoints": [f"kp{k}" for k in range(K)],
            "individuals": [f"ind{i}" for i in range(I)],
        },
    )
    kp_abl = KeypointAblation(seed=1, pB=0.0)
    out = kp_abl(ds.copy(deep=True))
    xr.testing.assert_equal(out, ds)


def test_keypoint_ablation_missing_dimensions():
    """Test KeypointAblation raises error with missing dimensions."""
    # Create dataset without 'individuals' dimension
    arr = np.zeros((10, 2, 3), dtype=np.float32)
    ds = xr.Dataset(
        {"position": (("time", "space", "keypoints"), arr)},
        coords={
            "time": np.arange(10),
            "space": ["x", "y"],
            "keypoints": ["kp0", "kp1", "kp2"],
        },
    )
    kp_abl = KeypointAblation(seed=1, pB=0.1)

    with pytest.raises(ValueError, match="Missing: {'individuals'}"):
        kp_abl(ds)




def test_keypoint_ablation_all_space_dims_ablated():
    """
    Test that KeypointAblation sets all space dimensions to NaN for selected elements.
    """
    T, S, K, I = 20, 3, 2, 2  # 3 spatial dimensions  # noqa: E741
    rng = np.random.default_rng(1789)
    arr = rng.random((T, S, K, I)).astype(np.float32)
    ds = xr.Dataset(
        {"position": (("time", "space", "keypoints", "individuals"), arr)},
        coords={
            "time": np.arange(T),
            "space": ["x", "y", "z"],
            "keypoints": ["kp0", "kp1"],
            "individuals": ["ind0", "ind1"],
        },
    )
    kp_abl = KeypointAblation(seed=42, pB=0.2)
    ds_abl = kp_abl(ds.copy(deep=True))

    pos_abl = ds_abl["position"].values
    # For each (t, k, i), either all space dims are NaN or none are
    for t in range(T):
        for k in range(K):
            for i in range(I):
                space_vals = pos_abl[t, :, k, i]
                # Either all NaN or none NaN
                assert np.all(np.isnan(space_vals)) or np.all(~np.isnan(space_vals))

