import pytest
import torch
import yaml

import lisbet.inference as inference


def make_dummy_dataset(root, keypoints=("nose", "tail")):
    from pathlib import Path

    import numpy as np
    import xarray as xr

    exp_dir = Path(root) / "exp0"
    exp_dir.mkdir(parents=True)
    arr = np.arange(10 * 1 * len(keypoints) * 2).reshape((10, 1, len(keypoints), 2))
    data = xr.Dataset(
        {
            "position": (
                ("time", "individuals", "keypoints", "space"),
                arr,
            )
        },
        coords={
            "time": np.arange(10),
            "individuals": ["mouse"],
            "keypoints": list(keypoints),
            "space": ["x", "y"],
        },
    )
    data.to_netcdf(exp_dir / "tracking.nc", engine="scipy")
    return root


def test_process_inference_dataset_input_features_success(tmp_path):
    """Test _process_inference_dataset succeeds when input_features match."""
    root = make_dummy_dataset(tmp_path / "ds", keypoints=("nose", "tail"))
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    config_path = model_dir / "model_config.yml"
    weights_path = model_dir / "weights.pt"
    input_features = [
        ["mouse", "nose", "x"],
        ["mouse", "nose", "y"],
        ["mouse", "tail", "x"],
        ["mouse", "tail", "y"],
    ]
    config = {
        "bp_dim": 4,
        "emb_dim": 2,
        "hidden_dim": 2,
        "num_heads": 1,
        "num_layers": 1,
        "max_len": 10,
        "output_token_idx": -1,
        "out_dim": {"cfc": 2},
        "input_features": input_features,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    model = torch.nn.Linear(4, 2)
    torch.save(model.state_dict(), weights_path)

    class DummyMultiTaskModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.task_heads = {"cfc": torch.nn.Linear(2, 2)}
            self.backbone = torch.nn.Linear(4, 2)

        def forward(self, x, task_id):
            return torch.zeros((x.shape[0], 2))

    inference.modeling.load_model = lambda *a, **k: DummyMultiTaskModel()
    result = inference._process_inference_dataset(
        model_path=str(config_path),
        weights_path=str(weights_path),
        forward_fn=lambda model, data: torch.zeros((data.shape[0], 2)),
        data_format="movement",
        data_path=str(root),
        data_scale=None,
        window_size=10,
        window_offset=0,
        fps_scaling=1.0,
        batch_size=2,
        select_coords=None,
        rename_coords=None,
    )
    assert isinstance(result, list)


def test_process_inference_dataset_input_features_incompatible(tmp_path):
    """
    Test _process_inference_dataset raises ValueError when input_features do not match.
    """
    root = make_dummy_dataset(tmp_path / "ds", keypoints=("nose", "tail"))
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    config_path = model_dir / "model_config.yml"
    weights_path = model_dir / "weights.pt"
    input_features = [["mouse", "nose", "x"], ["mouse", "nose", "y"]]
    config = {
        "bp_dim": 2,
        "emb_dim": 2,
        "hidden_dim": 2,
        "num_heads": 1,
        "num_layers": 1,
        "max_len": 10,
        "output_token_idx": -1,
        "out_dim": {"cfc": 2},
        "input_features": input_features,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    model = torch.nn.Linear(2, 2)
    torch.save(model.state_dict(), weights_path)

    class DummyMultiTaskModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.task_heads = {"cfc": torch.nn.Linear(2, 2)}
            self.backbone = torch.nn.Linear(2, 2)

        def forward(self, x, task_id):
            return torch.zeros((x.shape[0], 2))

    inference.modeling.load_model = lambda *a, **k: DummyMultiTaskModel()
    with pytest.raises(ValueError, match="Incompatible input features"):
        inference._process_inference_dataset(
            model_path=str(config_path),
            weights_path=str(weights_path),
            forward_fn=lambda model, data: torch.zeros((data.shape[0], 2)),
            data_format="movement",
            data_path=str(root),
            data_scale=None,
            window_size=10,
            window_offset=0,
            fps_scaling=1.0,
            batch_size=2,
            select_coords=None,
            rename_coords=None,
        )
