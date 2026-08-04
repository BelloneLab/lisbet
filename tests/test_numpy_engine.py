"""Tests for the xarray/NumPy window-processing engines."""

import inspect

import numpy as np
import pytest
import torch
import xarray as xr

import lisbet.evaluation as evaluation
import lisbet.inference.common as inference_common
from lisbet.config.schemas import DataAugmentationConfig
from lisbet.datasets import (
    AnnotatedWindowDataset,
    AnnotatedWindowSelector,
    GeometricInvarianceDataset,
    GroupConsistencyDataset,
    SocialBehaviorDataset,
    TemporalOrderDataset,
    TemporalShiftDataset,
    TemporalWarpDataset,
    WindowDataset,
    WindowSelector,
)
from lisbet.io import Record
from lisbet.training.tasks import configure_tasks
from lisbet.training.utils import generate_seeds
from lisbet.transforms_extra import (
    GaussianJitter,
    KeypointAblation,
    PoseToTensor,
    PoseToVideo,
    RandomBlockPermutation,
    RandomMirrorX,
    RandomPermutation,
    RandomRotation,
    RandomTranslate,
    RandomZoom,
)


def _canonical_values(dataset):
    return (
        dataset["position"]
        .transpose("time", "individuals", "keypoints", "space")
        .values
    )


def _legacy_xarray_window(record, window_size, window_offset, fps_scaling, frame_idx):
    """Reproduce window selection before engine dispatch was introduced."""
    rel_time_coords = np.linspace(0, window_size - 1, window_size, dtype=int)
    posetracks = record.posetracks

    if fps_scaling == 1.0:
        start_idx = frame_idx - window_size + window_offset + 1
        stop_idx = frame_idx + window_offset
        time_coords = np.linspace(start_idx, stop_idx, window_size, dtype=int)
        return posetracks.reindex(time=time_coords, fill_value=0).assign_coords(
            time=rel_time_coords
        )

    scaled_window_size = int(np.rint(fps_scaling * window_size))
    scaled_window_offset = int(np.rint(fps_scaling * window_offset))
    scaled_start_idx = frame_idx - scaled_window_size + scaled_window_offset + 1
    scaled_stop_idx = frame_idx + scaled_window_offset
    scaled_time_coords = np.linspace(
        scaled_start_idx, scaled_stop_idx, scaled_window_size, dtype=int
    )
    interp_time_coords = np.linspace(scaled_start_idx, scaled_stop_idx, window_size)

    return (
        posetracks.reindex(time=scaled_time_coords, fill_value=0)
        .interp(time=interp_time_coords)
        .assign_coords(time=rel_time_coords)
    )


def _legacy_annotation_value(annotations, frame_idx, annot_format):
    """Reproduce the annotation selection used before the NumPy engine."""
    target = annotations["target_cls"].isel(time=frame_idx)
    if annot_format == "binary":
        return target.values
    if annot_format == "multiclass":
        return target.argmax("behaviors").squeeze().values
    return target.squeeze().values


def _make_records(dtype=np.float32):
    records = []
    for rec_idx, n_frames in enumerate((12, 9)):
        canonical = np.arange(n_frames * 2 * 3 * 2, dtype=dtype).reshape(
            n_frames, 2, 3, 2
        )
        canonical = canonical + rec_idx * 1_000
        stored = canonical.transpose(0, 3, 2, 1)
        posetracks = xr.Dataset(
            {
                "position": (
                    ("time", "space", "keypoints", "individuals"),
                    stored,
                ),
                "quality": (("time",), np.arange(n_frames, dtype=np.int16)),
            },
            coords={
                "time": np.arange(n_frames),
                "space": ["x", "y"],
                "keypoints": ["nose", "ear", "tail"],
                "individuals": ["mouse1", "mouse2"],
            },
            attrs={"record": rec_idx},
        )

        target = np.zeros((n_frames, 3, 1), dtype=np.int16)
        target[np.arange(n_frames), np.arange(n_frames) % 3, 0] = 1
        annotations = xr.Dataset(
            {
                "target_cls": (
                    ("time", "behaviors", "annotators"),
                    target,
                )
            },
            coords={
                "time": np.arange(n_frames),
                "behaviors": ["rest", "follow", "attack"],
                "annotators": ["annotator0"],
            },
        )
        records.append(
            Record(
                id=f"record{rec_idx}",
                posetracks=posetracks,
                annotations=annotations,
            )
        )
    return records


@pytest.fixture
def records():
    return _make_records()


_PUBLIC_DATASET_CLASSES = (
    WindowSelector,
    AnnotatedWindowSelector,
    WindowDataset,
    AnnotatedWindowDataset,
    SocialBehaviorDataset,
    GroupConsistencyDataset,
    TemporalOrderDataset,
    TemporalShiftDataset,
    TemporalWarpDataset,
    GeometricInvarianceDataset,
)


def test_engine_is_final_backward_compatible_constructor_argument():
    for cls in _PUBLIC_DATASET_CLASSES:
        parameters = list(inspect.signature(cls).parameters.values())
        assert parameters[-1].name == "engine"
        assert parameters[-1].default == "xarray"


@pytest.mark.parametrize("cls", _PUBLIC_DATASET_CLASSES)
def test_invalid_engine_is_rejected_by_every_public_dataset(cls, records):
    with pytest.raises(ValueError, match="Invalid engine 'pandas'"):
        cls(records, window_size=5, engine="pandas")


def test_default_engine_keeps_xarray_data_and_metadata(records):
    sample = WindowSelector(records, window_size=5).select(0, 4)

    assert isinstance(sample, xr.Dataset)
    assert "quality" in sample
    assert sample.attrs == records[0].posetracks.attrs
    np.testing.assert_array_equal(sample.time, np.arange(5))


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
@pytest.mark.parametrize(
    ("fps_scaling", "window_offset", "frame_idx"),
    [
        (1.0, 0, 0),
        (1.0, 2, 11),
        (1.0, -20, 5),
        (0.5, 2, 0),
        (1.5, 2, 11),
        (1.5, 20, 5),
    ],
)
def test_default_xarray_selector_matches_legacy_implementation(
    dtype, fps_scaling, window_offset, frame_idx
):
    records = _make_records(dtype)
    selector = WindowSelector(
        records,
        window_size=5,
        window_offset=window_offset,
        fps_scaling=fps_scaling,
    )
    expected = _legacy_xarray_window(
        records[0],
        window_size=5,
        window_offset=window_offset,
        fps_scaling=fps_scaling,
        frame_idx=frame_idx,
    )

    actual = selector.select(0, frame_idx)

    xr.testing.assert_identical(actual, expected)


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_numpy_unscaled_windows_match_xarray_exactly(dtype):
    records = _make_records(dtype)
    xarray_selector = WindowSelector(
        records, window_size=5, window_offset=2, engine="xarray"
    )
    numpy_selector = WindowSelector(
        records, window_size=5, window_offset=2, engine="numpy"
    )

    for frame_idx in (0, 3, 8, 11):
        expected = _canonical_values(xarray_selector.select(0, frame_idx))
        actual = numpy_selector.select(0, frame_idx)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == dtype
        assert actual.shape == (5, 2, 3, 2)
        assert actual.flags.owndata
        assert actual.flags.writeable


def test_numpy_boundary_padding_offsets_and_record_immutability(records):
    source_before = _canonical_values(records[0].posetracks).copy()
    selector = WindowSelector(records, window_size=5, engine="numpy")

    start = selector.select(0, 1)
    np.testing.assert_array_equal(start[:3], 0)
    np.testing.assert_array_equal(start[3:], source_before[:2])

    offset_selector = WindowSelector(
        records, window_size=5, window_offset=2, engine="numpy"
    )
    end = offset_selector.select(0, 11)
    np.testing.assert_array_equal(end[:3], source_before[9:12])
    np.testing.assert_array_equal(end[3:], 0)

    start[...] = -123
    end[...] = -456
    np.testing.assert_array_equal(
        _canonical_values(records[0].posetracks), source_before
    )


@pytest.mark.parametrize("fps_scaling", [1.0, 1.5])
@pytest.mark.parametrize("window_offset", [-20, 20])
def test_numpy_fully_out_of_range_windows_match_xarray(
    records, fps_scaling, window_offset
):
    xarray_selector = WindowSelector(
        records,
        window_size=5,
        window_offset=window_offset,
        fps_scaling=fps_scaling,
        engine="xarray",
    )
    numpy_selector = WindowSelector(
        records,
        window_size=5,
        window_offset=window_offset,
        fps_scaling=fps_scaling,
        engine="numpy",
    )

    expected = _canonical_values(xarray_selector.select(0, frame_idx=5))
    actual = numpy_selector.select(0, frame_idx=5)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(actual, np.zeros_like(actual))


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
@pytest.mark.parametrize("fps_scaling", [0.5, 0.8, 1.5, 2.0])
@pytest.mark.parametrize("frame_idx", [0, 4, 11])
def test_numpy_interpolation_matches_xarray(dtype, fps_scaling, frame_idx):
    records = _make_records(dtype)
    xarray_selector = WindowSelector(
        records,
        window_size=6,
        window_offset=2,
        fps_scaling=fps_scaling,
        engine="xarray",
    )
    numpy_selector = WindowSelector(
        records,
        window_size=6,
        window_offset=2,
        fps_scaling=fps_scaling,
        engine="numpy",
    )

    expected = _canonical_values(xarray_selector.select(0, frame_idx))
    actual = numpy_selector.select(0, frame_idx)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual.dtype == expected.dtype
    assert np.issubdtype(actual.dtype, np.floating)
    assert actual.flags.owndata


@pytest.mark.parametrize("annot_format", ["binary", "multiclass", "multilabel"])
def test_numpy_annotations_match_xarray_shape_and_dtype(records, annot_format):
    xarray_selector = AnnotatedWindowSelector(
        records, window_size=4, annot_format=annot_format, engine="xarray"
    )
    numpy_selector = AnnotatedWindowSelector(
        records, window_size=4, annot_format=annot_format, engine="numpy"
    )

    expected = _legacy_annotation_value(records[0].annotations, 5, annot_format)
    _, xarray_actual = xarray_selector.select(0, 5)
    _, numpy_actual = numpy_selector.select(0, 5)

    for actual in (xarray_actual, numpy_actual):
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype

    numpy_actual[...] = 99
    _, selected_again = numpy_selector.select(0, 5)
    np.testing.assert_array_equal(selected_again, expected)


@pytest.mark.parametrize("annot_format", ["binary", "multiclass", "multilabel"])
@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
@pytest.mark.parametrize(
    ("n_behaviors", "n_annotators"), [(1, 1), (1, 2), (3, 1), (3, 2)]
)
def test_annotation_formatting_matches_legacy_xarray_for_edge_cases(
    annot_format, dtype, n_behaviors, n_annotators
):
    record = _make_records()[0]
    n_frames = record.posetracks.sizes["time"]
    target = np.zeros((n_frames, n_behaviors, n_annotators), dtype=dtype)

    for annotator_idx in range(n_annotators):
        target[:, annotator_idx % n_behaviors, annotator_idx] = 1

    # xarray skips NaNs when taking argmax over floating-point annotations. Keep
    # NaNs away from the winning behavior so the expected class is unambiguous.
    if np.issubdtype(dtype, np.floating) and n_behaviors == 3:
        target[5, :, 0] = [0, np.nan, 1]
        if n_annotators == 2:
            target[5, :, 1] = [1, 0, np.nan]

    annotations = xr.Dataset(
        {
            "target_cls": (
                ("time", "behaviors", "annotators"),
                target,
            )
        },
        coords={
            "time": np.arange(n_frames),
            "behaviors": [f"behavior{i}" for i in range(n_behaviors)],
            "annotators": [f"annotator{i}" for i in range(n_annotators)],
        },
    )
    record.annotations = annotations
    expected = _legacy_annotation_value(annotations, 5, annot_format)

    for engine in ("xarray", "numpy"):
        selector = AnnotatedWindowSelector(
            [record], window_size=4, annot_format=annot_format, engine=engine
        )
        _, actual = selector.select(0, 5)

        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        if engine == "numpy":
            assert actual.flags.owndata


def test_map_style_datasets_and_pose_to_tensor_match(records):
    xarray_dataset = WindowDataset(
        records, window_size=5, transform=PoseToTensor(), engine="xarray"
    )
    numpy_dataset = WindowDataset(
        records, window_size=5, transform=PoseToTensor(), engine="numpy"
    )
    torch.testing.assert_close(numpy_dataset[7], xarray_dataset[7], rtol=0, atol=0)

    xarray_annotated = AnnotatedWindowDataset(
        records,
        window_size=5,
        transform=PoseToTensor(),
        annot_format="multilabel",
        engine="xarray",
    )
    numpy_annotated = AnnotatedWindowDataset(
        records,
        window_size=5,
        transform=PoseToTensor(),
        annot_format="multilabel",
        engine="numpy",
    )
    x_xarray, y_xarray = xarray_annotated[13]
    x_numpy, y_numpy = numpy_annotated[13]
    torch.testing.assert_close(x_numpy, x_xarray, rtol=0, atol=0)
    np.testing.assert_array_equal(y_numpy, y_xarray)


def test_pose_to_tensor_handles_negative_stride_numpy_views():
    canonical = np.arange(3 * 2 * 3 * 2, dtype=np.float64).reshape(3, 2, 3, 2)
    reversed_time = canonical[::-1]
    expected = torch.from_numpy(
        reversed_time.copy().reshape(reversed_time.shape[0], -1).astype(np.float32)
    )

    actual = PoseToTensor()(reversed_time)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.float32
    assert actual.is_contiguous()


@pytest.mark.parametrize(
    ("dataset_cls", "kwargs", "debug_attrs"),
    [
        (SocialBehaviorDataset, {}, ()),
        (
            GroupConsistencyDataset,
            {},
            ("orig_coords", "swap_coords"),
        ),
        (
            TemporalOrderDataset,
            {"method": "simple"},
            ("pre_coords", "post_coords"),
        ),
        (
            TemporalOrderDataset,
            {"method": "strict"},
            ("pre_coords", "post_coords"),
        ),
        (
            TemporalShiftDataset,
            {"max_shift": 3},
            ("orig_coords", "shift_coords"),
        ),
        (
            TemporalWarpDataset,
            {"max_warp": 40},
            ("orig_coords", "warp_coords"),
        ),
    ],
)
def test_iterable_datasets_are_deterministic_and_equivalent_across_engines(
    records, dataset_cls, kwargs, debug_attrs
):
    xarray_dataset = dataset_cls(
        records, window_size=6, base_seed=19, engine="xarray", **kwargs
    )
    numpy_dataset = dataset_cls(
        records, window_size=6, base_seed=19, engine="numpy", **kwargs
    )
    xarray_iterator = iter(xarray_dataset)
    numpy_iterator = iter(numpy_dataset)

    for _ in range(8):
        x_xarray, y_xarray = next(xarray_iterator)
        x_numpy, y_numpy = next(numpy_iterator)
        assert isinstance(x_xarray, xr.Dataset)
        assert isinstance(x_numpy, np.ndarray)
        assert "quality" in x_xarray
        if dataset_cls is TemporalWarpDataset:
            np.testing.assert_allclose(
                x_numpy, _canonical_values(x_xarray), rtol=1e-12, atol=1e-12
            )
        else:
            np.testing.assert_array_equal(x_numpy, _canonical_values(x_xarray))
        np.testing.assert_array_equal(y_numpy, y_xarray)
        for attr in debug_attrs:
            assert attr in x_xarray.attrs
        assert not hasattr(x_numpy, "attrs")


def test_geometric_invariance_dataset_deterministic_and_equivalent_across_engines(
    records,
):
    xarray_dataset = GeometricInvarianceDataset(
        records, window_size=6, base_seed=23, engine="xarray"
    )
    numpy_dataset = GeometricInvarianceDataset(
        records, window_size=6, base_seed=23, engine="numpy"
    )
    xarray_iterator = iter(xarray_dataset)
    numpy_iterator = iter(numpy_dataset)

    for _ in range(8):
        x_orig_xarray, x_transform_xarray = next(xarray_iterator)
        x_orig_numpy, x_transform_numpy = next(numpy_iterator)

        assert isinstance(x_orig_xarray, xr.Dataset)
        assert isinstance(x_transform_xarray, xr.Dataset)
        assert isinstance(x_orig_numpy, np.ndarray)
        assert isinstance(x_transform_numpy, np.ndarray)

        np.testing.assert_allclose(
            x_orig_numpy, _canonical_values(x_orig_xarray), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            x_transform_numpy,
            _canonical_values(x_transform_xarray),
            rtol=1e-12,
            atol=1e-12,
        )

        assert "geometric_transforms_applied" in x_transform_xarray.attrs
        assert not hasattr(x_orig_numpy, "attrs")
        assert not hasattr(x_transform_numpy, "attrs")


def test_gaussian_jitter_matches_legacy_xarray_rng_layout():
    values = np.full((4, 2, 3, 2), 0.5, dtype=np.float32)
    posetracks = xr.Dataset(
        {
            "position": (
                ("time", "space", "keypoints", "individuals"),
                values,
            )
        }
    )
    seed = 123
    sigma = 0.1

    expected = posetracks.copy(deep=True)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(expected["position"].shape, generator=generator) * sigma
    transformed = torch.from_numpy(expected["position"].values) + noise
    transformed.clamp_(0.0, 1.0)
    expected["position"].values[...] = transformed.numpy()

    xarray_actual = GaussianJitter(seed, sigma)(posetracks.copy(deep=True))
    numpy_input = values.transpose(0, 3, 2, 1).copy()
    numpy_actual = GaussianJitter(seed, sigma)(numpy_input)

    xr.testing.assert_identical(xarray_actual, expected)
    np.testing.assert_array_equal(numpy_actual, _canonical_values(expected))


def test_keypoint_ablation_matches_legacy_xarray_rng_layout():
    values = np.ones((4, 2, 3, 2), dtype=np.float32)
    posetracks = xr.Dataset(
        {
            "position": (
                ("time", "space", "keypoints", "individuals"),
                values,
            )
        }
    )
    seed = 123
    probability = 0.5

    expected = posetracks.copy(deep=True)
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand((1, 1, 3, 2), generator=generator) < probability
    transformed = torch.where(
        mask,
        torch.tensor(0.0),
        torch.from_numpy(expected["position"].values),
    )
    expected["position"].values[...] = transformed.numpy()

    xarray_actual = KeypointAblation(seed, probability)(posetracks.copy(deep=True))
    numpy_input = values.transpose(0, 3, 2, 1).copy()
    numpy_actual = KeypointAblation(seed, probability)(numpy_input)

    xr.testing.assert_identical(xarray_actual, expected)
    np.testing.assert_array_equal(numpy_actual, _canonical_values(expected))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_space", [2, 3])
@pytest.mark.parametrize("mode", ["none", "truncate", "rescale"])
def test_random_rotation_matches_legacy_xarray_implementation(dtype, n_space, mode):
    rng = np.random.default_rng(1789)
    values = rng.random((5, n_space, 3, 2)).astype(dtype)
    posetracks = xr.Dataset(
        {
            "position": (
                ("time", "space", "keypoints", "individuals"),
                values,
            ),
            "quality": (("time",), np.arange(5, dtype=np.int16)),
        },
        coords={
            "time": np.arange(5),
            "space": ["x", "y", "z"][:n_space],
            "keypoints": ["nose", "ear", "tail"],
            "individuals": ["mouse1", "mouse2"],
        },
        attrs={"record": "legacy-parity"},
    )
    seed = 123
    max_angle = 135.0

    # Reproduce the implementation used before NumPy transform dispatch was added.
    expected = posetracks.copy(deep=True)
    generator = torch.Generator().manual_seed(seed)
    dims = list(expected["position"].dims)
    space_idx = dims.index("space")
    angle_deg = (torch.rand(1, generator=generator).item() * 2.0 - 1.0) * max_angle
    angle_rad = angle_deg * (np.pi / 180.0)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if n_space == 2:
        rotation = np.array([[c, -s], [s, c]])
    else:
        axis = torch.randn(3, generator=generator).numpy()
        axis = axis / np.linalg.norm(axis)
        kx, ky, kz = axis
        cross_product = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
        rotation = (
            np.eye(3) + s * cross_product + (1.0 - c) * (cross_product @ cross_product)
        )

    transformed = (
        np.moveaxis(expected["position"].values - 0.5, space_idx, -1) @ rotation.T
    )
    transformed = np.moveaxis(transformed, -1, space_idx) + 0.5
    if mode == "truncate":
        np.clip(transformed, 0.0, 1.0, out=transformed)
    elif mode == "rescale" and (np.any(transformed < 0.0) or np.any(transformed > 1.0)):
        for space_index in range(n_space):
            slices = [slice(None)] * transformed.ndim
            slices[space_idx] = space_index
            spatial_slice = transformed[tuple(slices)]
            vmin, vmax = spatial_slice.min(), spatial_slice.max()
            if vmin != vmax:
                transformed[tuple(slices)] = (spatial_slice - vmin) / (vmax - vmin)
    expected["position"].values[...] = transformed

    xarray_actual = RandomRotation(seed, max_angle, mode)(posetracks.copy(deep=True))
    numpy_actual = RandomRotation(seed, max_angle, mode)(
        values.transpose(0, 3, 2, 1).copy()
    )

    xr.testing.assert_identical(xarray_actual, expected)
    np.testing.assert_array_equal(numpy_actual, _canonical_values(expected))


@pytest.mark.parametrize(
    "transform_factory",
    [
        lambda: GaussianJitter(seed=3, sigma=0.05),
        lambda: KeypointAblation(seed=3, pB=0.4),
        lambda: RandomPermutation(
            seed=3, coordinate="individuals", exclude_identity=True
        ),
        lambda: RandomPermutation(
            seed=3, coordinate="keypoints", exclude_identity=True
        ),
        lambda: RandomPermutation(seed=3, coordinate="space", exclude_identity=True),
        lambda: RandomBlockPermutation(
            seed=3,
            coordinate="individuals",
            permute_fraction=0.5,
            exclude_identity=True,
        ),
        lambda: RandomBlockPermutation(
            seed=3,
            coordinate="keypoints",
            permute_fraction=0.5,
            exclude_identity=True,
        ),
        lambda: RandomRotation(seed=3, max_angle=60, mode="none"),
        lambda: RandomTranslate(seed=3),
        lambda: RandomMirrorX(seed=3),
        lambda: RandomZoom(seed=3),
    ],
)
def test_pose_augmentations_dispatch_with_equivalent_results(
    records, transform_factory
):
    xarray_input = records[0].posetracks.copy(deep=True)
    numpy_input = _canonical_values(xarray_input).copy()
    expected_quality = xarray_input["quality"].copy(deep=True)
    expected_attrs = xarray_input.attrs.copy()

    xarray_output = transform_factory()(xarray_input)
    numpy_output = transform_factory()(numpy_input)

    assert isinstance(xarray_output, xr.Dataset)
    assert isinstance(numpy_output, np.ndarray)
    np.testing.assert_array_equal(_canonical_values(xarray_output), numpy_output)
    xr.testing.assert_identical(xarray_output["quality"], expected_quality)
    assert xarray_output.attrs == expected_attrs


def test_composed_augmentations_produce_equivalent_float32_tensors(records):
    xarray_sample = records[0].posetracks.drop_vars("quality").copy(deep=True)
    numpy_sample = _canonical_values(xarray_sample).copy()

    def pipeline():
        return [
            RandomPermutation(1, "individuals", exclude_identity=True),
            RandomBlockPermutation(2, "individuals", 0.5, exclude_identity=True),
            GaussianJitter(3, 0.01),
            KeypointAblation(4, 0.2),
            RandomRotation(5, 30, "truncate"),
            PoseToTensor(),
        ]

    for xarray_transform, numpy_transform in zip(pipeline(), pipeline(), strict=True):
        xarray_sample = xarray_transform(xarray_sample)
        numpy_sample = numpy_transform(numpy_sample)

    assert xarray_sample.dtype == torch.float32
    assert numpy_sample.dtype == torch.float32
    torch.testing.assert_close(numpy_sample, xarray_sample, rtol=0, atol=0)


def test_pose_to_video_is_explicitly_xarray_only():
    with pytest.raises(TypeError, match="requires an xarray Dataset"):
        PoseToVideo({})(np.zeros((3, 2, 1, 2), dtype=np.float32))


@pytest.mark.parametrize(
    "config",
    [
        DataAugmentationConfig(name="all_perm_id"),
        DataAugmentationConfig(name="all_perm_ax"),
        DataAugmentationConfig(name="blk_perm_id", frac=0.5),
        DataAugmentationConfig(name="gauss_jitter", sigma=0.01),
        DataAugmentationConfig(name="kp_ablation", pB=0.2),
        DataAugmentationConfig(name="rotation", max_angle=45, mode="truncate"),
    ],
)
def test_training_augmentation_configs_accept_numpy(records, config):
    task_ids = ["multiclass"]
    task = configure_tasks(
        train_rec={"multiclass": records},
        dev_rec={"multiclass": []},
        task_ids=task_ids,
        window_size=6,
        window_offset=0,
        embedding_dim=4,
        hidden_dim=4,
        data_augmentation=[config],
        run_seeds=generate_seeds(7, task_ids),
        device=torch.device("cpu"),
    )[0]

    output, _ = next(iter(task.train_dataset))

    assert task.train_dataset.engine == "numpy"
    assert output.shape == (6, 12)
    assert output.dtype == torch.float32


def test_all_training_and_development_tasks_use_numpy(records):
    task_ids = ["multiclass", "multilabel", "cons", "order", "shift", "warp", "geom"]
    train_records = {task_id: records for task_id in task_ids}
    dev_records = {task_id: records for task_id in task_ids}
    tasks = configure_tasks(
        train_rec=train_records,
        dev_rec=dev_records,
        task_ids=task_ids,
        window_size=6,
        window_offset=0,
        embedding_dim=4,
        hidden_dim=4,
        data_augmentation=None,
        run_seeds=generate_seeds(11, task_ids),
        device=torch.device("cpu"),
    )

    for task in tasks:
        assert task.train_dataset.engine == "numpy"
        assert task.dev_dataset.engine == "numpy"
        x_train, _ = next(iter(task.train_dataset))
        x_dev, _ = next(iter(task.dev_dataset))
        assert isinstance(x_train, torch.Tensor)
        assert isinstance(x_dev, torch.Tensor)
        assert x_train.shape == (6, 12)
        assert x_dev.shape == (6, 12)


def test_prediction_dataset_uses_numpy(records, monkeypatch):
    captured = {}
    real_dataset = WindowDataset

    class SpyWindowDataset(real_dataset):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(inference_common, "WindowDataset", SpyWindowDataset)

    output = inference_common.predict_record(
        record=records[0],
        model=object(),
        device=torch.device("cpu"),
        window_size=6,
        window_offset=0,
        fps_scaling=1.0,
        batch_size=2,
        forward_fn=lambda _model, data: torch.zeros((data.shape[0], 1)),
    )

    assert captured["engine"] == "numpy"
    assert output.shape == (12, 1)


def test_evaluation_dataset_uses_numpy(records, monkeypatch):
    captured = {}
    real_dataset = AnnotatedWindowDataset

    class SpyAnnotatedWindowDataset(real_dataset):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    class DummyModel(torch.nn.Module):
        def forward(self, data, mode):
            assert mode == "multiclass"
            return torch.zeros((data.shape[0], 3))

    monkeypatch.setattr(
        evaluation, "load_model_and_config", lambda *_: (DummyModel(), {})
    )
    monkeypatch.setattr(evaluation, "load_records", lambda **_: records)
    monkeypatch.setattr(evaluation, "check_feature_compatibility", lambda *_: None)
    monkeypatch.setattr(evaluation, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(evaluation, "AnnotatedWindowDataset", SpyAnnotatedWindowDataset)

    report = evaluation.evaluate(
        model_path="model.yml",
        weights_path="weights.pt",
        data_format="movement",
        data_path="dataset",
        window_size=6,
        batch_size=2,
        mode="multiclass",
    )

    assert captured["engine"] == "numpy"
    assert report["mode"] == "multiclass"
