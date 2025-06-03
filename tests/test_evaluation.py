import numpy as np
import pytest
import yaml
from sklearn.metrics import classification_report

import lisbet.evaluation as evaluation
from lisbet.datasets.core import Record


@pytest.fixture
def dummy_inference(monkeypatch):
    # Patch _process_inference_dataset to return dummy predictions
    monkeypatch.setattr(
        evaluation.inference,
        "_process_inference_dataset",
        lambda **kwargs: [
            ("rec1", np.array([[1, 0], [0, 1], [1, 0]])),
            ("rec2", np.array([[0, 1], [1, 0]])),
        ],
    )


@pytest.fixture
def dummy_load_records(monkeypatch):
    # Patch load_records to return dummy ground-truth labels
    def _dummy_load_records(*args, **kwargs):
        return [
            Record(id="rec1", posetracks=None, annotations=DummyAnnotation([0, 1, 0])),
            Record(id="rec2", posetracks=None, annotations=DummyAnnotation([1, 0])),
        ]

    monkeypatch.setattr(evaluation, "load_records", _dummy_load_records)


class DummyAnnotation:
    def __init__(self, labels):
        self.target_cls = DummyTargetCls(labels)

    def __getitem__(self, item):
        return getattr(self, item)


class DummyTargetCls:
    def __init__(self, labels):
        self.labels = np.array(labels)

    def argmax(self, axis):
        return self

    def squeeze(self):
        return self

    @property
    def values(self):
        return self.labels


def test_evaluate_model_basic(tmp_path, dummy_inference, dummy_load_records):
    # Prepare dummy model config file
    model_path = tmp_path / "model_config.yml"
    with open(model_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"model_id": "dummy_model"}, f)
    weights_path = tmp_path / "weights.pt"
    weights_path.write_bytes(b"dummy")  # Just to have a file

    # Run evaluation
    report = evaluation.evaluate_model(
        model_path=str(model_path),
        weights_path=str(weights_path),
        data_format="movement",
        data_path="dummy",
        output_path=str(tmp_path),
    )
    # Should return a dict with keys like 'accuracy', 'macro avg', etc.
    assert isinstance(report, dict)
    assert "accuracy" in report

    # Check that the YAML file was written
    report_file = tmp_path / "evaluations" / "dummy_model" / "classification_report.yml"
    assert report_file.exists()
    with open(report_file, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded["accuracy"] == report["accuracy"]


def test_evaluate_model_label_filtering(tmp_path, dummy_inference, dummy_load_records):
    # Prepare dummy model config file
    model_path = tmp_path / "model_config.yml"
    with open(model_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"model_id": "dummy_model"}, f)
    weights_path = tmp_path / "weights.pt"
    weights_path.write_bytes(b"dummy")

    # Only include label 0 in the report
    report = evaluation.evaluate_model(
        model_path=str(model_path),
        weights_path=str(weights_path),
        data_format="movement",
        data_path="dummy",
        labels="0",
    )
    assert isinstance(report, dict)
    assert "0" in report


def test_evaluate_model_f1_score_correctness(tmp_path, monkeypatch):
    # Controlled dummy predictions and labels
    dummy_preds = [
        ("rec1", np.array([[1, 0], [0, 1], [1, 0]])),
        ("rec2", np.array([[0, 1], [1, 0]])),
    ]
    dummy_labels = {
        "rec1": [0, 1, 0],
        "rec2": [1, 0],
    }

    # Patch _process_inference_dataset
    monkeypatch.setattr(
        evaluation.inference,
        "_process_inference_dataset",
        lambda **kwargs: dummy_preds,
    )

    # Patch load_records
    def _dummy_load_records(*args, **kwargs):
        return [
            Record(
                id="rec1",
                posetracks=None,
                annotations=DummyAnnotation(dummy_labels["rec1"]),
            ),
            Record(
                id="rec2",
                posetracks=None,
                annotations=DummyAnnotation(dummy_labels["rec2"]),
            ),
        ]

    monkeypatch.setattr(evaluation, "load_records", _dummy_load_records)

    # Prepare dummy model config file
    model_path = tmp_path / "model_config.yml"
    with open(model_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"model_id": "dummy_model"}, f)
    weights_path = tmp_path / "weights.pt"
    weights_path.write_bytes(b"dummy")

    # Run evaluation
    report = evaluation.evaluate_model(
        model_path=str(model_path),
        weights_path=str(weights_path),
        data_format="movement",
        data_path="dummy",
    )

    # Compute expected F1 score using sklearn
    y_true = np.array(dummy_labels["rec1"] + dummy_labels["rec2"])
    y_pred = np.array([0, 1, 0, 1, 0])
    expected = classification_report(y_true, y_pred, digits=3, output_dict=True)

    # Compare all relevant keys
    for label in ["0", "1"]:
        assert report[label]["f1-score"] == pytest.approx(expected[label]["f1-score"])
        assert report[label]["precision"] == pytest.approx(expected[label]["precision"])
        assert report[label]["recall"] == pytest.approx(expected[label]["recall"])
    assert report["accuracy"] == pytest.approx(expected["accuracy"])
