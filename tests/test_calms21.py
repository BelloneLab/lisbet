import argparse
import json
from pathlib import Path

import numpy as np

from lisbet import hub
from lisbet.cli.commands.fetch import configure_fetch_dataset_parser
from lisbet.io import dump_records, load_records
from lisbet.io.ext_sources import calms21

BEHAVIOR_VOCAB = {
    "other": 3,
    "mount": 2,
    "attack": 0,
    "investigation": 1,
}


def _make_raw_record(record_id, annotator_id, offset):
    keypoints = np.arange(2 * 2 * 2 * 7).reshape(2, 2, 2, 7) + offset
    scores = np.linspace(0.1, 0.9, 2 * 2 * 7).reshape(2, 2, 7)
    return record_id, {
        "keypoints": keypoints.tolist(),
        "scores": scores.tolist(),
        "annotations": [3, (annotator_id - 1) % 3],
        "metadata": {
            "annotator-id": annotator_id,
            "vocab": BEHAVIOR_VOCAB,
        },
    }


def _write_task2_json(datapath):
    task_path = datapath / "task2_annotation_styles"
    task_path.mkdir()

    raw_by_split = {}
    for split in ("train", "test"):
        raw_data = {}
        for annotator_id in range(1, 6):
            record_id = (
                f"task2/annotator{annotator_id}/{split}/"
                f"mouse{annotator_id:03d}_task2_annotator{annotator_id}"
            )
            raw_data[f"annotator-id_{annotator_id}"] = dict(
                [_make_raw_record(record_id, annotator_id, annotator_id * 100)]
            )

        json_path = task_path / f"calms21_task2_{split}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f)
        raw_by_split[split] = raw_data

    return raw_by_split


def test_load_task2_preserves_movement_and_annotation_metadata(tmp_path):
    raw_by_split = _write_task2_json(tmp_path)

    train_records, test_records = calms21.load_taskx(tmp_path, taskid=2)

    assert len(train_records) == 5
    assert len(test_records) == 5
    assert [record.id for record in train_records] == [
        next(iter(group)) for group in raw_by_split["train"].values()
    ]
    assert [record.id for record in test_records] == [
        next(iter(group)) for group in raw_by_split["test"].values()
    ]

    record = train_records[0]
    raw_record = next(iter(raw_by_split["train"]["annotator-id_1"].values()))

    assert record.posetracks.coords["individuals"].values.tolist() == [
        "resident",
        "intruder",
    ]
    assert record.posetracks.coords["keypoints"].values.tolist() == [
        "nose",
        "left_ear",
        "right_ear",
        "neck",
        "left_hip",
        "right_hip",
        "tail",
    ]
    assert record.posetracks.coords["space"].values.tolist() == ["x", "y"]
    assert record.posetracks.attrs["fps"] == 30
    assert record.posetracks.attrs["image_size_px"] == [1024, 570]

    np.testing.assert_array_equal(
        record.posetracks["position"]
        .transpose("time", "individuals", "space", "keypoints")
        .values,
        raw_record["keypoints"],
    )
    np.testing.assert_allclose(
        record.posetracks["confidence"]
        .transpose("time", "individuals", "keypoints")
        .values,
        raw_record["scores"],
    )

    assert record.annotations.coords["behaviors"].values.tolist() == [
        "attack",
        "investigation",
        "mount",
        "other",
    ]
    assert record.annotations.coords["annotators"].values.tolist() == ["annotator1"]
    np.testing.assert_array_equal(
        record.annotations["target_cls"].values,
        np.array(
            [
                [[0], [0], [0], [1]],
                [[1], [0], [0], [0]],
            ]
        ),
    )


def test_task2_dump_round_trip_supports_path_filtering(tmp_path):
    _write_task2_json(tmp_path)
    train_records, test_records = calms21.load_taskx(tmp_path, taskid=2)

    output_path = tmp_path / "processed"
    dump_records(output_path, train_records)
    dump_records(output_path, test_records)

    record_path = (
        output_path / "task2" / "annotator1" / "train" / "mouse001_task2_annotator1"
    )
    assert (record_path / "tracking.nc").is_file()
    assert (record_path / "annotations.nc").is_file()

    loaded_records = load_records(
        data_format="movement",
        data_path=output_path,
        data_filter="annotator1/train",
    )
    assert len(loaded_records) == 1
    assert loaded_records[0].id == "task2/annotator1/train/mouse001_task2_annotator1"
    assert loaded_records[0].annotations.coords["annotators"].values.tolist() == [
        "annotator1"
    ]


def test_fetch_dataset_task2_uses_official_archive(monkeypatch, tmp_path):
    retrieve_kwargs = {}
    rawdata_path = tmp_path / "raw"
    extracted_path = rawdata_path / "task2_annotation_styles"
    extracted_files = [
        extracted_path / "calms21_task2_train.json",
        extracted_path / "calms21_task2_test.json",
    ]

    def fake_retrieve(**kwargs):
        retrieve_kwargs.update(kwargs)
        return extracted_files

    train_records = [object()]
    test_records = [object()]
    load_calls = []

    def fake_load_taskx(datapath, taskid):
        load_calls.append((datapath, taskid))
        return train_records, test_records

    dump_calls = []

    def fake_dump_records(datapath, records):
        dump_calls.append((datapath, records))

    monkeypatch.setattr(hub.pooch, "retrieve", fake_retrieve)
    monkeypatch.setattr(hub.calms21, "load_taskx", fake_load_taskx)
    monkeypatch.setattr(hub, "dump_records", fake_dump_records)

    hub.fetch_dataset("CalMS21_Task2", download_path=tmp_path)

    assert retrieve_kwargs["url"] == (
        "https://data.caltech.edu/records/s0vdx-0k302/files/"
        "task2_annotation_styles.zip?download=1"
    )
    assert retrieve_kwargs["known_hash"] == "md5:c97e87e13e77ffb80c073e05c05a4683"
    assert retrieve_kwargs["path"] == tmp_path / "datasets" / ".cache" / "lisbet"
    assert retrieve_kwargs["processor"].members == [
        "task2_annotation_styles/calms21_task2_train.json",
        "task2_annotation_styles/calms21_task2_test.json",
    ]
    assert retrieve_kwargs["progressbar"] is True
    assert load_calls == [(rawdata_path, 2)]

    output_path = tmp_path / "datasets" / "CalMS21" / "task2_annotation_styles"
    assert dump_calls == [
        (output_path, train_records),
        (output_path, test_records),
    ]


def test_fetch_dataset_parser_accepts_task2():
    parser = argparse.ArgumentParser()
    configure_fetch_dataset_parser(parser)

    args = parser.parse_args(["CalMS21_Task2"])

    assert args.dataset_id == "CalMS21_Task2"
    assert args.download_path == Path(".")
