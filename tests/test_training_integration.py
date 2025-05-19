import pytest

import lisbet.datasets.core as datasets_core
import lisbet.training as training


@pytest.mark.integration
def test_train_integration(tmp_path):
    # Download a small sample dataset using the LISBET API
    datasets_core.fetch_dataset("SampleData", download_path=tmp_path)
    data_path = tmp_path / "sample_keypoints"
    # Run a minimal training (1 epoch, small batch, minimal model)
    model = training.train(
        data_format="DLC",
        data_path=str(data_path),
        window_size=4,
        window_offset=0,
        epochs=1,
        batch_size=2,
        task_ids="smp",
        emb_dim=4,
        num_layers=1,
        num_heads=1,
        hidden_dim=8,
        learning_rate=1e-3,
        output_path=tmp_path,
        save_weights="last",
        save_history=True,
        dev_ratio=None,
        test_ratio=None,
        data_augmentation=False,
        train_sample=None,
        dev_sample=None,
        compile_model=False,
        mixed_precision=False,
        run_id="pytest_run",
        select_coords=None,
        rename_coords=None,
    )
    # Check that model is returned and weights/history are saved
    assert hasattr(model, "state_dict")
    weights_path = tmp_path / "models" / "pytest_run" / "weights" / "weights_last.pt"
    assert weights_path.exists()
    hist_path = tmp_path / "models" / "pytest_run" / "training_history.log"
    assert hist_path.exists()
