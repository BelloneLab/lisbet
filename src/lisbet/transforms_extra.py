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
        background_color=(0, 0, 0),
    ):
        """
        Args:
            body_specs: dict mapping individual_name (or species) to BodySpecs
            image_size: (width, height) of output frames
            background_color: BGR tuple or color name/hex (default black)
        """
        self.body_specs = body_specs
        self.image_size = image_size
        self.background_color = color_to_bgr(background_color)

    def __call__(self, posetracks):
        frames = np.stack(
            [self.render_frame(posetracks, t) for t in range(posetracks.sizes["time"])],
            axis=0,
        )

        # Convert to PyTorch tensor
        frames = torch.Tensor(frames)

        return frames

    def render_frame(self, posetracks, t):
        H, W = self.image_size[1], self.image_size[0]
        frame = np.full((H, W, 3), self.background_color, dtype=np.uint8)
        pos = (
            posetracks["position"].isel(time=t).values
        )  # (space, keypoints, individuals)
        pos = np.transpose(pos, (2, 1, 0))  # (individuals, keypoints, 2)
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
                        pts.append([int(x * W), int(y * H)])
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
                        (int(x1 * W), int(y1 * H)),
                        (int(x2 * W), int(y2 * H)),
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
                    (int(x * W), int(y * H)),
                    spec.keypoint_size,
                    color=color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
        return frame

    @staticmethod
    def cv2_to_rgb(frame_bgr):
        """
        Convert a BGR frame (OpenCV) to RGB for matplotlib plotting.
        """
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
