"""Augmentation module for transforming samples in a dataset."""

import cv2
import numpy as np
import torch

from lisbet.drawing import BodySpecs, body_specs_registry, color_to_bgr


class RandomXYSwap:
    """Random transformation swapping x and y coordinates"""

    def __init__(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample):
        transformed_sample = (
            np.stack((sample[:, 1::2], sample[:, ::2]), axis=2).reshape(sample.shape)
            if self.rng.random() < 0.5
            else sample
        )
        return transformed_sample


class PoseToTensor:
    """Extract the position variable from a record"""

    def __call__(self, sample):
        """Extract the position variable from a record."""
        return torch.Tensor(
            sample.stack(features=("individuals", "keypoints", "space")).position.values
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
                color = color_to_bgr(spec.get_keypoint_color(kp))
                x, y = pos[ind_idx, k, :]
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
