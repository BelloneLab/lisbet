"""Augmentation module for transforming samples in a dataset.

This module provides data augmentation and preprocessing transforms for pose tracking
datasets stored as xarray.Dataset objects. The transforms can be used in training
pipelines to improve model robustness and generalization.

Available Transforms
--------------------
RandomPermutation
    Randomly permutes both coordinate labels and their associated data together across
    the entire time window. Useful for making models invariant to coordinate ordering
    (e.g., individual identities, spatial axes).

RandomBlockPermutation
    Randomly permutes data within a contiguous block of frames while keeping coordinate
    labels unchanged. Creates temporal identity confusion within part of the window.
    Useful for more challenging augmentation scenarios.

KeypointAblation
    Randomly sets keypoint coordinates to NaN with independent Bernoulli sampling
    across (time, keypoints, individuals). Simulates missing or occluded keypoints
    for robustness testing.

KeypointBlockAblation
    Randomly sets keypoint coordinates to NaN within element-specific temporal blocks.
    Each selected (time, keypoint, individual) element triggers ablation for a block
    of frames, simulating sustained occlusion or tracking loss.

PoseToTensor
    Converts pose tracking data from xarray.Dataset format to PyTorch tensors by
    stacking spatial dimensions into a single feature dimension.

PoseToVideo
    Renders pose tracking data as video frames (RGB images) using OpenCV, with
    customizable body specifications for visualization.

VideoToTensor
    Converts video frames from NumPy arrays to PyTorch tensors with optional
    normalization for video model inputs.

Usage Examples
--------------
>>> from lisbet.transforms_extra import RandomPermutation, PoseToTensor
>>> from torchvision import transforms
>>>
>>> # Simple augmentation pipeline
>>> transform = transforms.Compose([
...     RandomPermutation(seed=42, coordinate='individuals'),
...     PoseToTensor(),
... ])
>>>
>>> # Apply with probability using torchvision.transforms.RandomApply
>>> transform = transforms.Compose([
...     transforms.RandomApply([
...         RandomPermutation(seed=42, coordinate='individuals')
...     ], p=0.5),
...     PoseToTensor(),
... ])
>>>
>>> # Block permutation for temporal identity confusion
>>> from lisbet.transforms_extra import RandomBlockPermutation
>>> transform = transforms.Compose([
...     RandomBlockPermutation(seed=42, coordinate='individuals', permute_fraction=0.3),
...     PoseToTensor(),
... ])
>>>
>>> # Keypoint ablation for robustness to missing data
>>> from lisbet.transforms_extra import KeypointAblation
>>> transform = transforms.Compose([
...     transforms.RandomApply([
...         KeypointAblation(seed=42, p=0.05)
...     ], p=1.0),
...     PoseToTensor(),
... ])

Notes
-----
- Augmentations should be applied thoughtfully based on dataset characteristics
- Spatial axis permutation (coordinate='space') is only suitable for top-down view
  datasets where axes are symmetric
- Identity permutations work best for datasets where individual labels are
  interchangeable
"""

import logging

import cv2
import numpy as np
import torch
import xarray as xr

from lisbet.drawing import BodySpecs, body_specs_registry, color_to_bgr


class GaussianJitter:
    """Apply Gaussian jitter with per-element Bernoulli sampling.

    Probability ``p`` is applied independently over (time, keypoints, individuals).
    For every selected element, Gaussian noise N(0, sigma^2) is added *broadcast*
    across the space dimension(s). Coordinates are assumed normalized in [0, 1] and
    are clamped to that range post-perturbation.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility.
    p : float
        Bernoulli probability for each (frame,keypoint,individual) element.
    sigma : float
        Standard deviation of the Gaussian noise.
    """

    def __init__(self, seed: int, p: float, sigma: float):
        self.seed = seed
        self.p = float(p)
        self.sigma = float(sigma)
        self.g = torch.Generator().manual_seed(seed)


    def __call__(self, posetracks: xr.Dataset) -> xr.Dataset:
        pos_var = posetracks["position"]

        dims = list(pos_var.dims)

        # Validate dataset dimensions
        required_dims = {"time", "keypoints", "individuals"}
        missing_dims = required_dims - set(dims)
        if missing_dims:
            raise ValueError(
                f"Position variable must contain {required_dims} dimensions. "
                f"Missing: {missing_dims}"
            )

        shape = pos_var.shape
        # Mask shape excludes space dimension(s) for independence semantics.
        mask_shape = [shape[d] for d in range(len(shape))]
        # Replace space dimension size(s) by 1 for broadcasting (space may be before
        # keypoints as per dataset examples)
        for s_name in ["space"]:
            if s_name in dims:
                s_idx = dims.index(s_name)
                mask_shape[s_idx] = 1
        # Ensure independence only over time,keypoints,individuals by collapsing non
        # listed dims to 1
        for d_name in dims:
            if d_name not in ("time", "keypoints", "individuals", "space"):
                mask_shape[dims.index(d_name)] = 1

        bern = torch.rand(mask_shape, generator=self.g) < self.p
        # Broadcast mask to full position shape
        mask = bern
        # Create noise tensor same full shape
        noise = torch.randn(shape, generator=self.g) * self.sigma
        # Apply
        pos = torch.from_numpy(pos_var.values)
        pos = pos + noise * mask
        # Clamp to [0,1]
        pos.clamp_(0.0, 1.0)
        # print('clamped pos:', pos)
        pos_var = pos.numpy()
        posetracks['position'].values[:] = pos_var
        return posetracks


class GaussianBlockJitter:
    """Apply Gaussian jitter within element-specific temporal blocks.

    Bernoulli(p) is sampled independently over (time, keypoints, individuals) to
    select *start* elements. For each positive start at (t0, k, i), a block of length
    ``block_len = int(frac * window)`` frames [t0, t0+block_len) (clipped at sequence end)
    receives Gaussian noise N(0, sigma^2) only for that (keypoint, individual) pair
    across all space dims. Overlapping blocks (either same or different start
    elements covering the same frame and (k,i)) merge naturally; noise is applied
    once per affected element-frame. A debug log reports overlap when merged
    coverage < raw expected coverage.

    Parameters
    ----------
    seed : int
        RNG seed.
    p : float
        Bernoulli probability for each frame to be a block start.
    sigma : float
        Noise standard deviation.
    frac : float
        Fraction of the total temporal window length used for each block
        (0 < frac < 1). Effective block length is ``max(1, int(frac * window))``.
    """

    def __init__(self, seed: int, p: float, sigma: float, frac: float):
        if not 0 < frac < 1:
            raise ValueError("frac must be between 0 and 1 (exclusive)")
        self.seed = seed
        self.p = float(p)
        self.sigma = float(sigma)
        self.frac = float(frac)
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self, posetracks: xr.Dataset) -> xr.Dataset:
        pos_var = posetracks["position"]
        dims = list(pos_var.dims)
        if "time" not in dims:
            raise ValueError("Position variable must have 'time' dimension.")
        t_idx = dims.index("time")
        shape = pos_var.shape
        T = shape[t_idx]
        if T == 0:
            return posetracks
        try:
            k_idx = dims.index("keypoints")
            i_idx = dims.index("individuals")
        except ValueError as e:
            raise ValueError(
                "Position variable must contain 'keypoints' and 'individuals' "
                "dimensions."
            ) from e

        K = shape[k_idx]
        I = shape[i_idx]  # noqa: E741

        block_len = max(1, int(self.frac * T))
        start_mask = torch.rand(T, K, I, generator=self.g) < self.p
        block_mask = torch.zeros(T, K, I, dtype=torch.bool)
        starts = torch.nonzero(start_mask, as_tuple=False)
        for idx_row in starts:
            t0, k, ind = idx_row.tolist()
            end = min(t0 + block_len, T)
            block_mask[t0:end, k, ind] = True

        if starts.numel() > 0:
            expected_cover = starts.size(0) * block_len
            actual_cover = int(block_mask.sum().item())
            if actual_cover < expected_cover:
                logging.debug(
                    "Overlapping element blocks detected in gauss_block_jitter "
                    "(expected raw=%d, merged=%d).",
                    expected_cover,
                    actual_cover,
                )

        if not block_mask.any():
            return posetracks

        broadcast_shape = [1] * len(shape)
        broadcast_shape[t_idx] = T
        broadcast_shape[k_idx] = K
        broadcast_shape[i_idx] = I
        block_mask_b = block_mask.view(broadcast_shape)

        noise = torch.randn(shape, generator=self.g) * self.sigma
        pos = torch.from_numpy(pos_var.values)
        pos = pos + noise * block_mask_b
        pos.clamp_(0.0, 1.0)
        pos_var.values[:] = pos.numpy()
        return posetracks


class KeypointAblation:
    """Apply keypoint ablation with per-element Bernoulli sampling.

    Probability ``p`` is applied independently over (time, keypoints, individuals).
    For every selected element, all spatial coordinates (x, y, z, etc.) are set to NaN,
    simulating missing or occluded keypoints.

    This augmentation helps models become robust to missing data, which commonly occurs
    due to occlusions, tracking failures, or low-confidence detections.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility.
    p : float
        Bernoulli probability for each (frame, keypoint, individual) element.

    Examples
    --------
    >>> from lisbet.transforms_extra import KeypointAblation
    >>> ablation = KeypointAblation(seed=42, p=0.05)
    >>> ablated_ds = ablation(posetracks)
    """

    def __init__(self, seed: int, p: float):
        self.seed = seed
        self.p = float(p)
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self, posetracks: xr.Dataset) -> xr.Dataset:
        pos_var = posetracks["position"]
        dims = list(pos_var.dims)

        # Validate dataset dimensions
        required_dims = {"time", "keypoints", "individuals"}
        missing_dims = required_dims - set(dims)
        if missing_dims:
            raise ValueError(
                f"Position variable must contain {required_dims} dimensions. "
                f"Missing: {missing_dims}"
            )

        shape = pos_var.shape
        # Mask shape excludes space dimension(s) for independence semantics.
        mask_shape = [shape[d] for d in range(len(shape))]
        # Replace space dimension size(s) by 1 for broadcasting
        for s_name in ["space"]:
            if s_name in dims:
                s_idx = dims.index(s_name)
                mask_shape[s_idx] = 1
        # Ensure independence only over time, keypoints, individuals
        for d_name in dims:
            if d_name not in ("time", "keypoints", "individuals", "space"):
                mask_shape[dims.index(d_name)] = 1

        # Generate Bernoulli mask
        bern = torch.rand(mask_shape, generator=self.g) < self.p
        # Broadcast mask to full position shape
        mask = bern

        # Apply ablation by setting selected elements to NaN
        pos = torch.from_numpy(pos_var.values)
        pos = torch.where(mask, torch.tensor(float('nan')), pos)
        pos_var.values[:] = pos.numpy()
        return posetracks


class KeypointBlockAblation:
    """Apply keypoint ablation within element-specific temporal blocks.

    Bernoulli(p) is sampled independently over (time, keypoints, individuals) to
    select *start* elements. For each positive start at (t0, k, i), a block of length
    ``block_len = int(frac * window)`` frames [t0, t0+block_len) (clipped at sequence end)
    has all spatial coordinates set to NaN only for that (keypoint, individual) pair.
    Overlapping blocks (either same or different start elements covering the same frame
    and (k, i)) merge naturally; ablation is applied once per affected element-frame.
    A debug log reports overlap when merged coverage < raw expected coverage.

    This augmentation simulates sustained occlusion or tracking loss, where a specific
    keypoint for a specific individual becomes unavailable for a period of time.

    Parameters
    ----------
    seed : int
        RNG seed.
    p : float
        Bernoulli probability for each frame to be a block start.
    frac : float
        Fraction of the total temporal window length used for each block
        (0 < frac < 1). Effective block length is ``max(1, int(frac * window))``.

    Examples
    --------
    >>> from lisbet.transforms_extra import KeypointBlockAblation
    >>> ablation = KeypointBlockAblation(seed=42, p=0.05, frac=0.1)
    >>> ablated_ds = ablation(posetracks)
    """

    def __init__(self, seed: int, p: float, frac: float):
        if not 0 < frac < 1:
            raise ValueError("frac must be between 0 and 1 (exclusive)")
        self.seed = seed
        self.p = float(p)
        self.frac = float(frac)
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self, posetracks: xr.Dataset) -> xr.Dataset:
        pos_var = posetracks["position"]
        dims = list(pos_var.dims)
        if "time" not in dims:
            raise ValueError("Position variable must have 'time' dimension.")
        t_idx = dims.index("time")
        shape = pos_var.shape
        T = shape[t_idx]
        if T == 0:
            return posetracks
        try:
            k_idx = dims.index("keypoints")
            i_idx = dims.index("individuals")
        except ValueError as e:
            raise ValueError(
                "Position variable must contain 'keypoints' and 'individuals' "
                "dimensions."
            ) from e

        K = shape[k_idx]
        I = shape[i_idx]  # noqa: E741

        block_len = max(1, int(self.frac * T))
        start_mask = torch.rand(T, K, I, generator=self.g) < self.p
        block_mask = torch.zeros(T, K, I, dtype=torch.bool)
        starts = torch.nonzero(start_mask, as_tuple=False)
        for idx_row in starts:
            t0, k, ind = idx_row.tolist()
            end = min(t0 + block_len, T)
            block_mask[t0:end, k, ind] = True

        if starts.numel() > 0:
            expected_cover = starts.size(0) * block_len
            actual_cover = int(block_mask.sum().item())
            if actual_cover < expected_cover:
                logging.debug(
                    "Overlapping element blocks detected in kp_block_ablation "
                    "(expected raw=%d, merged=%d).",
                    expected_cover,
                    actual_cover,
                )

        if not block_mask.any():
            return posetracks

        broadcast_shape = [1] * len(shape)
        broadcast_shape[t_idx] = T
        broadcast_shape[k_idx] = K
        broadcast_shape[i_idx] = I
        block_mask_b = block_mask.view(broadcast_shape)

        # Apply ablation by setting selected elements to NaN
        pos = torch.from_numpy(pos_var.values)
        pos = torch.where(block_mask_b, torch.tensor(float('nan')), pos)
        pos_var.values[:] = pos.numpy()
        return posetracks


class RandomPermutation:
    """
    Randomly permutes the order of a specified coordinate (e.g., 'individuals') in an
    xarray.Dataset, reordering both the coordinate labels and their associated data
    together.

    This augmentation can be used to increase invariance to coordinate order (e.g.,
    fixed identity, axis orientation). The permutation is applied to the entire dataset.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    coordinate : str
        Name of the coordinate to permute (e.g., 'individuals', 'keypoints', 'space').

    Methods
    -------
    __call__(posetracks)
        Applies the random permutation to the specified coordinate of the input
        xarray.Dataset.

    Examples
    --------
    >>> permute = RandomPermutation(seed=42, coordinate='individuals')
    >>> permuted_ds = permute(posetracks)
    """

    def __init__(self, seed, coordinate="individuals"):
        self.seed = seed
        self.coordinate = coordinate
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self, posetracks):
        """
        Apply random permutation to the specified coordinate.

        Parameters
        ----------
        posetracks : xarray.Dataset
            Pose tracks dataset with a 'position' variable.

        Returns
        -------
        xarray.Dataset
            Dataset with permuted coordinate and data.
        """
        # Get current coordinate values
        coord_vals = list(posetracks.coords[self.coordinate].values)

        # Generate a random permutation
        perm = torch.randperm(len(coord_vals), generator=self.g).tolist()

        # Apply permutation to the entire dataset
        # NOTE: This reorders both coordinates and data together
        posetracks = posetracks.isel({self.coordinate: perm})

        return posetracks


class RandomBlockPermutation:
    """
    Randomly permutes the data (but not coordinate labels) of a specified coordinate
    within a random contiguous block of frames in an xarray.Dataset.

    This augmentation is useful to create identity swaps within a portion of the time
    series, mimicking the effects of a tracking error, while maintaining consistent
    coordinate labels throughout.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    coordinate : str
        Name of the coordinate to permute (e.g., 'individuals', 'keypoints').
    permute_fraction : float
        Fraction of the time window to which the permutation is applied.
        Must be between 0 (exclusive) and 1 (exclusive). A continuous block of frames
        of this relative size will be selected at random, and the permutation will be
        applied only to the data within this block, keeping coordinate labels unchanged.

    Methods
    -------
    __call__(posetracks)
        Applies the random block permutation to the specified coordinate of the input
        xarray.Dataset.

    Examples
    --------
    >>> permute = RandomBlockPermutation(seed=42, coordinate='individuals',
    ...                                   permute_fraction=0.3)
    >>> permuted_ds = permute(posetracks)
    """

    def __init__(self, seed, coordinate="individuals", permute_fraction=0.5):
        self.seed = seed
        self.coordinate = coordinate
        if not 0 < permute_fraction < 1:
            raise ValueError("permute_fraction must be a float between 0 and 1.")
        self.permute_fraction = permute_fraction
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self, posetracks):
        """
        Apply random block permutation to the specified coordinate.

        Parameters
        ----------
        posetracks : xarray.Dataset
            Pose tracks dataset with a 'position' variable.

        Returns
        -------
        xarray.Dataset
            Dataset with permuted data in a random block, coordinates unchanged.
        """
        # Get current coordinate values
        coord_vals = list(posetracks.coords[self.coordinate].values)

        # Generate a random permutation
        perm = torch.randperm(len(coord_vals), generator=self.g).tolist()

        window_size = posetracks.sizes["time"]
        block_size = int(self.permute_fraction * window_size)

        if block_size == 0:
            # No permutation needed
            return posetracks

        start_idx = torch.randint(
            0, window_size - block_size + 1, (1,), generator=self.g
        ).item()
        end_idx = start_idx + block_size

        # For block permutation, we permute only the data
        # while keeping coordinates unchanged across the full time series
        block_to_permute = posetracks.isel(time=slice(start_idx, end_idx))

        # Get the dimension index for the coordinate
        coord_dim = list(posetracks["position"].dims).index(self.coordinate)

        # Permute the data along the coordinate dimension
        permuted_data = np.take(
            block_to_permute["position"].values, perm, axis=coord_dim
        )

        # Create a new block with permuted data but original coordinates
        permuted_block = block_to_permute.copy(deep=True)
        permuted_block["position"].values[:] = permuted_data

        # Split and concatenate
        before_block = posetracks.isel(time=slice(None, start_idx))
        after_block = posetracks.isel(time=slice(end_idx, None))

        posetracks = xr.concat(
            [before_block, permuted_block, after_block], dim="time", join="outer"
        )

        return posetracks


class PoseToTensor:
    """
    Convert the 'position' variable from a posetracks xarray.Dataset into a PyTorch
    tensor.

    This transformation stacks the 'individuals', 'keypoints', and 'space' dimensions
    into a single 'features' dimension, resulting in a tensor of shape
    (time, features), where features = individuals * keypoints * space.

    Parameters
    ----------
    None

    Methods
    -------
    __call__(posetracks)
        Stack the 'individuals', 'keypoints', and 'space' dimensions of the 'position'
        variable and return as a PyTorch tensor.

    Examples
    --------
    >>> tensor = PoseToTensor()(posetracks)
    >>> tensor.shape
    torch.Size([time, features])
    """

    def __call__(self, posetracks):
        """
        Stack the 'individuals', 'keypoints', and 'space' dimensions of the 'position'
        variable in the input xarray.Dataset and return as a PyTorch tensor.

        Parameters
        ----------
        posetracks : xarray.Dataset
            Pose tracks dataset with a 'position' variable of shape
            (time, individuals, keypoints, space).

        Returns
        -------
        torch.Tensor
            Tensor of shape (time, features), where features =
            individuals * keypoints * space, containing the stacked position data.
        """
        return torch.from_numpy(
            posetracks.stack(
                features=("individuals", "keypoints", "space")
            ).position.values.astype("float32")
        )


class PoseToVideo:
    """
    Fast OpenCV-based transformation: posetracks (xarray.Dataset) to a sequence of BGR
    images.
    """

    def __init__(
        self,
        body_specs: dict[str, BodySpecs],
        image_size=(256, 256),
        bg_color="black",
    ):
        """
        Fast OpenCV-based transformation using BodySpecs for each individual.

        Parameters
        ----------
        body_specs : dict of str to BodySpecs
            Dictionary mapping individual_name (or species) to BodySpecs.
        image_size : tuple of int, optional
            (width, height) of output frames. Default is (256, 256).
        bg_color : tuple or str, optional
            BGR tuple or color name/hex for background color (default is black).
        """
        self.body_specs = body_specs
        self.width, self.height = image_size
        self.bg_color = color_to_bgr(bg_color)

    def __call__(self, posetracks, show_progress=False):
        frames = [
            self.render_frame(posetracks, t) for t in range(posetracks.sizes["time"])
        ]

        frames = np.stack(frames, axis=0)

        # # Convert to PyTorch tensor
        # frames = torch.Tensor(frames)

        return frames

    def render_frame(self, posetracks, t_idx):
        """
        Render a single frame of pose tracks as a BGR image.

        Parameters
        ----------
        posetracks : xarray.Dataset
            The pose tracks dataset containing keypoints and individuals.
            Must have a "position" variable with dimensions ("time", "individuals",
            "keypoints", "space").
        t_idx : int
            The time index of the frame to render.

        Returns
        -------
        frame : numpy.ndarray
            The rendered frame as a (height, width, 3) uint8 RGB image.
        """
        frame = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        pos = (
            posetracks["position"]
            .isel(time=t_idx)
            .transpose("individuals", "keypoints", "space")
            .values
        )
        keypoints = list(posetracks.keypoints.values)
        individuals = list(posetracks.individuals.values)
        for ind_idx, ind_name in enumerate(individuals):
            spec = self.body_specs.get(ind_name, body_specs_registry.get(ind_name))
            if spec is None:
                continue
            # Draw polygons (with alpha blending)
            for poly in spec.polygons:
                pts = []
                for kp in poly:
                    if kp in keypoints:
                        idx = keypoints.index(kp)
                        x, y = pos[ind_idx, idx, :]
                        # Skip if coordinates are NaN (ablated keypoints)
                        if not (np.isnan(x) or np.isnan(y)):
                            pts.append([int(x * self.width), int(y * self.height)])
                if len(pts) >= 3:
                    pts_np = np.array([pts], dtype=np.int32)
                    overlay = frame.copy()
                    color = color_to_bgr(spec.polygon_color)
                    cv2.fillPoly(overlay, pts_np, color)
                    frame = cv2.addWeighted(
                        overlay, spec.polygon_alpha, frame, 1 - spec.polygon_alpha, 0
                    )
            # Draw skeleton
            for edge in spec.skeleton_edges:
                if edge[0] in keypoints and edge[1] in keypoints:
                    idx1 = keypoints.index(edge[0])
                    idx2 = keypoints.index(edge[1])
                    x1, y1 = pos[ind_idx, idx1, :]
                    x2, y2 = pos[ind_idx, idx2, :]
                    # Skip if any coordinates are NaN (ablated keypoints)
                    if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                        color = color_to_bgr(spec.skeleton_color)
                        cv2.line(
                            frame,
                            (int(x1 * self.width), int(y1 * self.height)),
                            (int(x2 * self.width), int(y2 * self.height)),
                            color=color,
                            thickness=spec.skeleton_thickness,
                            lineType=cv2.LINE_AA,
                        )
            # Draw keypoints
            for k, kp in enumerate(keypoints):
                x, y = pos[ind_idx, k, :]
                # Skip if coordinates are NaN (ablated keypoints)
                if not (np.isnan(x) or np.isnan(y)):
                    color = color_to_bgr(spec.get_keypoint_color(kp))
                    cv2.circle(
                        frame,
                        (int(x * self.width), int(y * self.height)),
                        spec.keypoint_size,
                        color=color,
                        thickness=-1,
                        lineType=cv2.LINE_AA,
                    )

        # Convert a BGR frame (OpenCV) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame


class VideoToTensor:
    """
    Transform a video (NumPy RGB array) into a PyTorch tensor suitable for video models.

    Converts (frames, H, W, 3) RGB uint8/float arrays to (frames, 3, H, W) float
    tensors, with optional normalization and mean/std normalization.

    Parameters
    ----------
    normalize : bool, optional
        If True, scale pixel values to [0, 1] (default: True).
    mean : tuple or list or np.ndarray or torch.Tensor, optional
        Per-channel mean for normalization (applied after scaling to [0, 1]).
        If None, no mean subtraction is performed.
    std : tuple or list or np.ndarray or torch.Tensor, optional
        Per-channel std for normalization (applied after mean subtraction).
        If None, no std division is performed.
    dtype : torch.dtype, optional
        Output tensor dtype (default: torch.float32).
    """

    def __init__(self, normalize=True, mean=None, std=None, dtype=torch.float32):
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.dtype = dtype

    def __call__(self, video):
        """
        Parameters
        ----------
        video : np.ndarray
            Video as (frames, H, W, 3) RGB, dtype uint8 or float.

        Returns
        -------
        torch.Tensor
            Video as (frames, 3, H, W), dtype as specified.
        """
        if not isinstance(video, np.ndarray):
            raise TypeError("Input video must be a numpy ndarray.")
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError("Input video must have shape (frames, H, W, 3) [RGB].")

        # If uint8, convert to float32 for normalization
        if video.dtype == np.uint8:
            video = video.astype(np.float32)
            if self.normalize:
                video = video / 255.0
        elif self.normalize:
            # Assume already float, but ensure in [0, 1]
            video = np.clip(video, 0.0, 1.0)

        # Rearrange to (frames, 3, H, W)
        video = np.transpose(video, (0, 3, 1, 2))
        tensor = torch.from_numpy(video).type(self.dtype)

        # Optional mean/std normalization (per channel)
        if self.mean is not None:
            mean = torch.as_tensor(self.mean, dtype=self.dtype, device=tensor.device)
            if mean.ndim == 1:
                mean = mean.view(1, 3, 1, 1)
            tensor = tensor - mean
        if self.std is not None:
            std = torch.as_tensor(self.std, dtype=self.dtype, device=tensor.device)
            if std.ndim == 1:
                std = std.view(1, 3, 1, 1)
            tensor = tensor / std

        return tensor
