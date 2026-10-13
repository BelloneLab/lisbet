"""Drawing specifications for LISBET."""

from dataclasses import dataclass, field

import matplotlib.colors as mcolors


@dataclass
class BodySpecs:
    """
    Specification for drawing a body (skeleton, polygons, keypoints) for a given
    species or individual.

    Parameters
    ----------
    skeleton_edges : list of tuple of str
        List of (keypoint1, keypoint2) pairs to draw as skeleton lines.
    polygons : list of list of str, optional
        List of lists of keypoint names to fill as polygons.
    keypoint_colors : dict of str to str, optional
        Mapping from keypoint name to color.
    skeleton_color : str, optional
        Color for skeleton lines.
    polygon_color : str, optional
        Color for filled polygons.
    polygon_alpha : float, optional
        Alpha (opacity) for polygons.
    keypoint_marker : str, optional
        Marker style for keypoints.
    keypoint_size : int, optional
        Size of keypoint markers.
    """

    skeleton_edges: list[tuple[str, str]]
    polygons: list[list[str]] = field(default_factory=list)
    keypoint_colors: dict[str, str] = field(default_factory=dict)
    skeleton_color: str = "lime"
    skeleton_thickness: int = 1
    polygon_color: str = "cyan"
    polygon_alpha: float = 0.3
    keypoint_marker: str = "o"
    keypoint_size: int = 6

    def get_keypoint_color(self, keypoint: str) -> str:
        """Return the color for a given keypoint, or white if not specified."""
        return self.keypoint_colors.get(keypoint, "white")


def color_to_bgr(color):
    """
    Convert a matplotlib color name or hex string to a BGR tuple for OpenCV.
    Accepts color names (e.g. 'red'), hex strings (e.g. '#ff0000'), or BGR tuples.
    """
    if (
        isinstance(color, tuple)
        and len(color) == 3
        and all(isinstance(x, int) for x in color)
    ):
        return color  # Already BGR
    if (
        isinstance(color, tuple)
        and len(color) == 3
        and all(isinstance(x, float) for x in color)
    ):
        # Assume RGB float 0-1
        rgb = tuple(int(255 * x) for x in color)
        return (rgb[2], rgb[1], rgb[0])
    if isinstance(color, str):
        if color.startswith("#"):
            rgb = mcolors.to_rgb(color)
        else:
            rgb = mcolors.to_rgb(mcolors.CSS4_COLORS.get(color, color))
        rgb = tuple(int(255 * x) for x in rgb)
        return (rgb[2], rgb[1], rgb[0])
    # fallback
    return (255, 255, 255)


# Registry of common species/body layouts for LISBET
body_specs_registry: dict[str, BodySpecs] = {
    "mouse": BodySpecs(
        skeleton_edges=[
            ("nose", "neck"),
            ("neck", "tail"),
            ("nose", "left_ear"),
            ("left_ear", "neck"),
            ("nose", "right_ear"),
            ("right_ear", "neck"),
            ("neck", "left_hip"),
            ("left_hip", "tail"),
            ("neck", "right_hip"),
            ("right_hip", "tail"),
        ],
        polygons=[
            ["nose", "left_ear", "neck", "right_ear"],
            ["neck", "left_hip", "tail", "right_hip"],
        ],
        keypoint_colors={
            "nose": "red",
            "left_ear": "orange",
            "right_ear": "orange",
            "neck": "yellow",
            "tail": "blue",
            "left_hip": "green",
            "right_hip": "green",
        },
        skeleton_color="green",
        polygon_color="dodgerblue",
        polygon_alpha=0.7,
        keypoint_marker="o",
        keypoint_size=3,
    ),
    "fly": BodySpecs(
        skeleton_edges=[
            ("head", "thorax"),
            ("thorax", "abdomen"),
        ],
        polygons=[],
        keypoint_colors={
            "head": "orange",
            "thorax": "yellow",
            "abdomen": "blue",
        },
        skeleton_color="purple",
        polygon_color="cyan",
        polygon_alpha=0.3,
        keypoint_marker="^",
        keypoint_size=6,
    ),
    "human": BodySpecs(
        skeleton_edges=[
            # Face
            ("nose", "left_eye"),
            ("left_eye", "left_ear"),
            ("nose", "right_eye"),
            ("right_eye", "right_ear"),
            # Upper body
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            # Torso
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            # Lower body
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
        ],
        polygons=[],
        keypoint_colors={
            "nose": "red",
            "left_eye": "orange",
            "right_eye": "orange",
            "left_ear": "yellow",
            "right_ear": "yellow",
            "left_shoulder": "lime",
            "right_shoulder": "lime",
            "left_elbow": "cyan",
            "right_elbow": "cyan",
            "left_wrist": "blue",
            "right_wrist": "blue",
            "left_hip": "magenta",
            "right_hip": "magenta",
            "left_knee": "purple",
            "right_knee": "purple",
            "left_ankle": "pink",
            "right_ankle": "pink",
        },
        skeleton_color="lime",
        polygon_color="cyan",
        polygon_alpha=0.3,
        keypoint_marker="o",
        keypoint_size=4,
    ),
}
