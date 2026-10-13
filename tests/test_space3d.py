"""3D spatial layouts (``space_layout`` "xyz_up_y" / "xyz_up_z").

Rigid augmentations, geom, configs, training and embedding.
"""

import numpy as np
import pytest
import torch
import xarray as xr
from pydantic import ValidationError

from lisbet.config.schemas import (
    DataAugmentationConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    TransformerBackboneConfig,
)
from lisbet.datasets import GeometricInvarianceDataset
from lisbet.inference.embedding import compute_embeddings
from lisbet.io.core import export_embedder
from lisbet.training import train
from lisbet.training.tasks import _build_augmentation_transforms, configure_tasks
from lisbet.training.utils import generate_seeds
from lisbet.transforms_extra import RandomRotation, RandomTranslate, layout_axes

LAYOUTS = ("xyz_up_y", "xyz_up_z")
TASKS = ("cons", "order", "shift", "warp", "geom")


def _window(n_space=3, T=20, seed=0, extent=0.2):
    """Canonical (time, individuals, keypoints, space) window around the box centre."""
    rng = np.random.default_rng(seed)
    return 0.5 + rng.uniform(-extent, extent, size=(T, 2, 5, n_space)).astype(
        np.float32
    )


def _dist(pos):
    """All pairwise distances between (individual, keypoint) points, per frame."""
    p = pos.reshape(pos.shape[0], -1, pos.shape[-1]).astype(np.float64)
    return np.linalg.norm(p[:, :, None] - p[:, None], axis=-1)


def _as_dataset(pos):
    return xr.Dataset(
        {"position": (("time", "individuals", "keypoints", "space"), pos.copy())},
        coords={"space": ["x", "y", "z"][: pos.shape[-1]]},
    )


# --------------------------------------------------------------------------- transforms


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("container", ["numpy", "xarray"])
def test_yaw_keeps_vertical_and_distances(layout, container):
    up, floor = layout_axes(layout)
    rot = RandomRotation(seed=1, max_angle=180.0, space_layout=layout)
    for i in range(20):
        pos = _window(seed=i)
        x = pos.copy() if container == "numpy" else _as_dataset(pos)
        out = rot(x)
        out = out if container == "numpy" else out["position"].values
        np.testing.assert_allclose(out[..., up], pos[..., up], atol=1e-6)
        np.testing.assert_allclose(_dist(out), _dist(pos), atol=1e-5)
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert out.dtype == pos.dtype


@pytest.mark.parametrize("layout", LAYOUTS)
def test_yaw_actually_rotates(layout):
    _, (a, b) = layout_axes(layout)
    rot = RandomRotation(seed=2, max_angle=180.0, space_layout=layout)
    pos = _window()
    angles = []
    for _ in range(50):
        out = rot(pos.copy())
        # heading of the vector between the two first points in the floor plane
        v0 = pos[0, 0, 1] - pos[0, 0, 0]
        v1 = out[0, 0, 1] - out[0, 0, 0]
        angles.append(np.arctan2(v1[b], v1[a]) - np.arctan2(v0[b], v0[a]))
    angles = np.angle(np.exp(1j * np.array(angles)))
    assert np.ptp(angles) > np.pi  # spread over most of the circle


@pytest.mark.parametrize("layout", LAYOUTS)
def test_yaw_shifts_back_inside_box(layout):
    """A scene near the box edge is shifted back rigidly, not clamped."""
    _, (a, _) = layout_axes(layout)
    pos = _window(extent=0.1)
    pos[..., a] += 0.35  # floor axis a spans about [0.75, 0.95]
    rot = RandomRotation(seed=3, max_angle=180.0, space_layout=layout)
    for _ in range(20):
        out = rot(pos.copy())
        np.testing.assert_allclose(_dist(out), _dist(pos), atol=1e-5)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_translate_floor_only(layout):
    up, floor = layout_axes(layout)
    tr = RandomTranslate(seed=4, space_layout=layout)
    moved = np.zeros(3)
    for i in range(20):
        pos = _window(seed=i)
        out = tr(pos.copy())
        np.testing.assert_array_equal(out[..., up], pos[..., up])
        np.testing.assert_allclose(_dist(out), _dist(pos), atol=1e-5)
        assert out.min() >= 0.0 and out.max() <= 1.0
        moved += np.abs(out - pos).reshape(-1, 3).max(axis=0)
    assert moved[up] == 0.0 and all(moved[f] > 0 for f in floor)


@pytest.mark.parametrize("cls", [RandomTranslate, RandomRotation])
def test_3d_layout_on_2d_data_raises(cls):
    with pytest.raises(ValueError, match="needs 3 spatial axes"):
        cls(seed=0, space_layout="xyz_up_y")(_window(n_space=2))


def test_unknown_layout_raises():
    with pytest.raises(ValueError):
        RandomTranslate(seed=0, space_layout="up_y")
    with pytest.raises(ValueError, match="only supports mode='truncate'"):
        RandomRotation(seed=0, mode="rescale", space_layout="xyz_up_y")


@pytest.mark.parametrize("n_space", [2, 3])
def test_legacy_rotation_numpy_matches_xarray(n_space):
    """Legacy rotation now accepts the canonical NumPy windows used by training."""
    pos = _window(n_space=n_space)
    out_np = RandomRotation(seed=5, max_angle=30.0)(pos.copy())
    out_xr = RandomRotation(seed=5, max_angle=30.0)(_as_dataset(pos))
    np.testing.assert_allclose(out_np, out_xr["position"].values, atol=1e-6)


@pytest.mark.parametrize("name", ["all_zoom", "all_mirror_x", "all_perm_ax"])
def test_build_transforms_refuses_non_rigid(name):
    augs = [DataAugmentationConfig(name=name)]
    _build_augmentation_transforms(augs, seed=0)  # legacy: fine
    with pytest.raises(ValueError, match="not allowed"):
        _build_augmentation_transforms(augs, seed=0, space_layout="xyz_up_y")


# --------------------------------------------------------------------------- configs


def _training(layout, aug):
    return TrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        data_augmentation=[DataAugmentationConfig(name=a) for a in aug],
        space_layout=layout,
    )


@pytest.mark.parametrize("name", ["all_zoom", "all_mirror_x", "all_perm_ax"])
def test_schema_refuses_non_rigid(name):
    _training("legacy", [name])
    with pytest.raises(ValidationError, match="not allowed"):
        _training("xyz_up_y", [name])


def test_schema_accepts_rigid():
    cfg = _training("xyz_up_y", ["all_translate", "rotation", "gauss_jitter"])
    assert cfg.space_layout == "xyz_up_y"
    assert (
        TrainingConfig(epochs=1, batch_size=1, learning_rate=1).space_layout == "legacy"
    )


def _experiment(layout, data_scale, tmp_path="."):
    return ExperimentConfig(
        model=ModelConfig(
            backbone=TransformerBackboneConfig(
                embedding_dim=8, hidden_dim=8, num_heads=1, num_layers=1, max_length=10
            ),
            out_heads={},
            input_features={},
            window_size=10,
            window_offset=0,
        ),
        training=_training(layout, ["all_translate", "rotation"]),
        data=DataConfig(
            data_path=str(tmp_path),
            data_format="movement",
            data_scale=data_scale,
            window_size=10,
        ),
        task_ids_list=list(TASKS),
        output_path=tmp_path,
    )


@pytest.mark.parametrize("scale", [None, "1x2x1", "1x1"])
def test_3d_layout_requires_isotropic_scale(scale):
    _experiment("legacy", scale)  # legacy: anything goes
    with pytest.raises(ValidationError, match="data_scale"):
        _experiment("xyz_up_y", scale)
    _experiment("xyz_up_y", "1x1x1")


# --------------------------------------------------------------------------- tasks


class _Rec:
    def __init__(self, idx, n_frames, static=False):
        rng = np.random.default_rng(idx)
        if static:  # same pose in every frame: every window of a record is identical
            pos = np.repeat(_window(T=1, seed=idx), n_frames, axis=0)
        else:
            steps = rng.normal(0.0, 0.005, size=(n_frames, 2, 5, 3))
            pos = np.clip(_window(T=1, seed=idx) + np.cumsum(steps, axis=0), 0.05, 0.95)
        self.posetracks = xr.Dataset(
            {"position": (("time", "individuals", "keypoints", "space"), pos)},
            coords={"individuals": ["child", "adult"], "space": ["x", "y", "z"]},
        )
        self.id = f"rec{idx}"


@pytest.mark.parametrize("layout", LAYOUTS)
def test_geom_target_is_rigid_motion(layout):
    up, _ = layout_axes(layout)
    records = [_Rec(i, 100, static=True) for i in range(3)]
    ds = GeometricInvarianceDataset(
        records,
        window_size=10,
        engine="numpy",
        base_seed=7,
        window_sampling="inside",
        space_layout=layout,
    )
    it = iter(ds)
    for _ in range(30):
        x_orig, x_tr = next(it)
        np.testing.assert_allclose(_dist(x_tr), _dist(x_orig), atol=1e-5)
        np.testing.assert_allclose(x_tr[..., up], x_orig[..., up], atol=1e-6)
        assert not np.allclose(x_tr, x_orig)


@pytest.mark.parametrize("layout", ["legacy", *LAYOUTS])
def test_all_tasks_yield_3d_features(layout):
    records = [_Rec(i, n) for i, n in enumerate((80, 120, 200))]
    augs = [
        DataAugmentationConfig(name="all_translate", p=0.5),
        DataAugmentationConfig(name="rotation", p=0.5, max_angle=180.0),
    ]
    tasks = configure_tasks(
        train_rec={t: records for t in TASKS},
        dev_rec={t: [] for t in TASKS},
        task_ids=list(TASKS),
        window_size=20,
        window_offset=10,
        embedding_dim=8,
        hidden_dim=8,
        data_augmentation=augs,
        run_seeds=generate_seeds(1, list(TASKS)),
        device="cpu",
        window_sampling="inside",
        space_layout=layout,
    )
    for task in tasks:
        it = iter(task.train_dataset)
        labels = []
        for _ in range(200):
            x, y = next(it)
            assert x.shape == (20, 2 * 5 * 3)
            assert torch.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
            if task.task_id != "geom":
                labels.append(float(np.asarray(y).ravel()[0]))
        if labels:
            assert 0.3 < np.mean(labels) < 0.7, task.task_id


# --------------------------------------------------------------------------- training


def _write_movement(root, n_records=3, n_frames=150):
    for i in range(n_records):
        rec = _Rec(i, n_frames).posetracks.assign_coords(
            time=np.arange(n_frames), keypoints=[f"k{j}" for j in range(5)]
        )
        rec.attrs["fps"] = 30
        d = root / f"session{i}_dyad"
        d.mkdir(parents=True)
        rec.to_netcdf(d / "tracking.nc", engine="scipy")


@pytest.mark.parametrize("layout", ["xyz_up_y"])
def test_train_3d_end_to_end(tmp_path, layout):
    _write_movement(tmp_path / "data")
    cfg = _experiment(layout, "1x1x1", tmp_path)
    cfg.data.data_path = str(tmp_path / "data")
    cfg.training.window_sampling = "inside"
    cfg.training.save_weights = "last"
    cfg.model.model_id = cfg.run_id = "space3d"
    model = train(cfg)
    assert hasattr(model, "state_dict")
    model_dir = tmp_path / "models" / "space3d"
    saved = (model_dir / "model_config.yml").read_text()
    assert "space_layout: xyz_up_y" in saved
    assert "- z" in saved  # input_features.space

    # Embedder export keeps the layout; embeddings are computed on the 3D records
    export_embedder(
        model_dir / "model_config.yml",
        model_dir / "weights" / "weights_last.pt",
        tmp_path,
    )
    emb_dir = tmp_path / "models" / "space3d-embedder"
    assert "space_layout: xyz_up_y" in (emb_dir / "model_config.yml").read_text()
    out = compute_embeddings(
        str(emb_dir / "model_config.yml"),
        str(emb_dir / "weights" / "weights_last.pt"),
        data_format="movement",
        data_path=str(tmp_path / "data"),
        data_scale="1x1x1",
        window_size=10,
    )
    assert len(out) == 3
    assert all(emb.shape == (150, 8) for _, emb in out)


def test_legacy_model_config_has_no_layout_key(tmp_path):
    from lisbet.io.core import dump_model_config

    cfg = _experiment("legacy", None, tmp_path)
    dump_model_config(tmp_path, "m", cfg.model)
    assert (
        "space_layout"
        not in (tmp_path / "models" / "m" / "model_config.yml").read_text()
    )


def test_train_3d_layout_refuses_wrong_axis_names(tmp_path):
    _write_movement(tmp_path / "data")
    for d in (tmp_path / "data").iterdir():
        ds = xr.load_dataset(d / "tracking.nc", engine="scipy")
        ds.assign_coords(space=["a", "b", "c"]).to_netcdf(
            d / "tracking.nc", engine="scipy"
        )
    cfg = _experiment("xyz_up_y", "1x1x1", tmp_path)
    cfg.data.data_path = str(tmp_path / "data")
    with pytest.raises(ValueError, match=r"\['x', 'y', 'z'\]"):
        train(cfg)
