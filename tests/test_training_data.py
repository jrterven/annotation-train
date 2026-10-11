import copy
import hashlib

import pytest
from pydantic import ValidationError
from app.training.data import DatasetInput, fingerprint, make_dataset, review_key
from app.training.store import LocalRepository


def source(count=10):
    categories = [{"id": 7, "name": "Object", "color": "#abcdef"}]
    images = [
        {
            "id": i,
            "file_name": f"{i}.png",
            "width": 20,
            "height": 10,
            "sha256": hashlib.sha256(str(i).encode()).hexdigest(),
            "revision": 3,
            "annotations": [
                {"id": str(i), "kind": "bbox", "category_id": 7, "bbox": [1, 2, 5, 6], "iscrowd": 0}
            ],
        }
        for i in range(1, count + 1)
    ]
    return {"categories": categories, "images": images}


def reviewed(data, task="detection"):
    return {
        review_key(task, im["id"]): {"fingerprint": fingerprint(im, data["categories"], task)}
        for im in data["images"]
    }


def test_optional_test_is_disjoint_deterministic_and_immutable():
    data = source()
    original = copy.deepcopy(data)
    a = make_dataset(data, reviewed(data), DatasetInput(task="detection"))
    assert a["counts"] == {"train": 8, "validation": 2}
    assert "test" not in {im["split"] for im in a["images"]}
    options = DatasetInput(task="detection", test_percent=10)
    b = make_dataset(data, reviewed(data), options)
    assert b["counts"] == {"train": 7, "validation": 2, "test": 1}
    assert b == make_dataset(data, reviewed(data), options)
    assert data == original
    assert len({im["id"] for im in b["images"]}) == 10
    assert a["fingerprint"] != b["fingerprint"]


def test_duplicate_pixels_never_leak_between_partitions():
    data = source()
    data["images"][1]["sha256"] = data["images"][0]["sha256"]
    result = make_dataset(data, reviewed(data), DatasetInput(task="detection", test_percent=10))
    assert result["images"][0]["split"] == result["images"][1]["split"]
    with pytest.raises(ValueError, match="Identical"):
        make_dataset(
            data,
            reviewed(data),
            DatasetInput(task="detection", assignments={1: "train", 2: "validation"}),
        )


def test_review_is_task_specific_content_bound_and_empty_requires_review():
    data = source(3)
    data["images"][0]["annotations"] = []
    marks = reviewed(data)
    data["images"][1]["revision"] += 1
    assert len(make_dataset(data, marks, DatasetInput(task="detection"))["images"]) == 3
    data["images"][1]["annotations"][0]["bbox"][0] += 1
    result = make_dataset(
        data, marks, DatasetInput(task="detection", assignments={1: "validation", 3: "train"})
    )
    assert {im["id"] for im in result["images"]} == {1, 3}
    with pytest.raises(ValueError, match="Mark complete"):
        make_dataset(data, marks, DatasetInput(task="segmentation"))


def test_invalid_split_and_missing_training_class_fail_before_creation():
    data = source(2)
    with pytest.raises(ValueError, match="3 distinct"):
        make_dataset(data, reviewed(data), DatasetInput(task="detection", test_percent=10))
    with pytest.raises(ValidationError):
        DatasetInput(task="detection", validation_percent=90, test_percent=20)
    with pytest.raises(ValidationError):
        DatasetInput(task="detection", assignments={1: "test"})
    data["images"][0]["annotations"] = []
    with pytest.raises(ValueError, match="Every annotated class"):
        make_dataset(
            data,
            reviewed(data),
            DatasetInput(task="detection", assignments={1: "train", 2: "validation"}),
        )


def test_local_records_rollback_and_survive_reopen(tmp_path):
    repo = LocalRepository(tmp_path)
    with repo.transaction() as records:
        records.put("review", "one", {"reviewed": True})
    with pytest.raises(RuntimeError), repo.transaction() as records:
        records.delete("review", "one")
        raise RuntimeError()
    with LocalRepository(tmp_path).transaction() as records:
        assert records.get("review", "one") == {"reviewed": True}
