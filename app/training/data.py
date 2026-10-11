"""Immutable, original-pixel training datasets and explicit review fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.storage import RevisionConflict

Task = Literal["detection", "segmentation"]
Split = Literal["train", "validation", "test"]


def encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ReviewInput(Strict):
    task: Task
    images: dict[int, int] = Field(min_length=1, max_length=10000)
    reviewed: bool = True


class DatasetInput(Strict):
    task: Task
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    validation_percent: int = Field(default=20, ge=1, le=99)
    test_percent: int = Field(default=0, ge=0, le=98)
    assignments: dict[int, Split] = Field(default_factory=dict)

    @model_validator(mode="after")
    def percentages(self):
        if self.validation_percent + self.test_percent >= 100:
            raise ValueError("Leave at least 1% for training.")
        if not self.test_percent and "test" in self.assignments.values():
            raise ValueError("Enable the test partition before assigning test images.")
        return self


def capture(store, ready=None):
    """One SQLite read generation, no previews or mutable ProjectStore handles."""
    with store._connect() as connection:
        connection.execute("BEGIN")
        categories = [
            json.loads(row[0])
            for row in connection.execute("SELECT data FROM categories ORDER BY id")
        ]
        images = []
        for row in connection.execute(
            "SELECT id,file_name,width,height,sha256,revision,state FROM images ORDER BY id"
        ):
            if ready is not None and row["id"] not in ready:
                continue
            image = {
                key: row[key]
                for key in ("id", "file_name", "width", "height", "sha256", "revision")
            }
            image["annotations"] = json.loads(row["state"])["annotations"]
            if ready is not None:
                image["object_key"] = ready[row["id"]][0]
            images.append(image)
    return {"categories": categories, "images": images}


def task_image(image, task):
    annotations = [
        a for a in image["annotations"] if (a.get("kind") == "bbox") == (task == "detection")
    ]
    return {**image, "annotations": annotations}


def fingerprint(image, categories, task):
    # Drafts, proposals, display colors, and the other task do not invalidate review.
    selected = task_image(image, task)
    ids = {a["category_id"] for a in selected["annotations"]}
    return digest(
        {
            "sha256": image["sha256"],
            "task": task,
            "size": [image["width"], image["height"]],
            "annotations": selected["annotations"],
            "classes": [{"id": c["id"], "name": c["name"]} for c in categories if c["id"] in ids],
        }
    )


def review_key(task, image_id):
    return f"{task}:{image_id}"


def review_rows(source, reviews, task):
    return [
        {
            "id": im["id"],
            "file_name": im["file_name"],
            "revision": im["revision"],
            "annotation_count": len(task_image(im, task)["annotations"]),
            "reviewed": reviews.get(review_key(task, im["id"]), {}).get("fingerprint")
            == fingerprint(im, source["categories"], task),
        }
        for im in source["images"]
    ]


def make_dataset(source, reviews, options: DatasetInput):
    categories = source["categories"]
    images = [
        task_image(im, options.task)
        for im in source["images"]
        if reviews.get(review_key(options.task, im["id"]), {}).get("fingerprint")
        == fingerprint(im, categories, options.task)
    ]
    if not images:
        raise ValueError("Mark complete images as reviewed before creating a dataset.")
    image_ids = {im["id"] for im in images}
    if set(options.assignments) - image_ids:
        raise ValueError("Assignments contain images that are not reviewed for this task.")
    groups = {}
    for image in images:
        groups.setdefault(image["sha256"], []).append(image)
    splits = ["train", "validation"] + (["test"] if options.test_percent else [])
    if len(groups) < len(splits):
        raise ValueError(
            f"At least {len(splits)} distinct reviewed images are required for these partitions."
        )
    ordered = sorted(groups, key=lambda sha: digest([options.seed, sha]))
    counts = {
        "validation": max(1, math.floor(len(groups) * options.validation_percent / 100)),
        "test": (
            max(1, math.floor(len(groups) * options.test_percent / 100))
            if options.test_percent
            else 0
        ),
    }
    while sum(counts.values()) >= len(groups):
        largest = max(counts, key=counts.get)
        if counts[largest] <= 1:
            raise ValueError("Not enough distinct images for these partitions.")
        counts[largest] -= 1
    assignment = {}
    for index, sha in enumerate(ordered):
        automatic = (
            "test"
            if index < counts["test"]
            else "validation" if index < sum(counts.values()) else "train"
        )
        overrides = {
            options.assignments[im["id"]] for im in groups[sha] if im["id"] in options.assignments
        }
        if len(overrides) > 1:
            raise ValueError("Identical images must stay in the same partition.")
        assignment[sha] = next(iter(overrides), automatic)
    for image in images:
        image["split"] = assignment[image["sha256"]]
    if any(not any(im["split"] == split for im in images) for split in splits):
        raise ValueError("Every enabled partition must contain an image.")
    present = {
        split: {
            a["category_id"] for im in images if im["split"] == split for a in im["annotations"]
        }
        for split in splits
    }
    used = set().union(*present.values())
    if not used:
        raise ValueError(
            "Add at least one confirmed object. Negative images alone cannot train a model."
        )
    if used - present["train"]:
        raise ValueError(
            "Every annotated class must appear in training. Adjust the image assignments."
        )
    classes = [
        {"index": i, "category_id": c["id"], "name": c["name"]}
        for i, c in enumerate(c for c in categories if c["id"] in used)
    ]
    warnings = [
        f"{split}: no examples of {c['name']}"
        for split in splits
        if split != "train"
        for c in classes
        if c["category_id"] not in present[split]
    ]
    result = {
        "schema_version": 1,
        "task": options.task,
        "options": options.model_dump(mode="json"),
        "classes": classes,
        "images": images,
        "warnings": warnings,
        "counts": {split: sum(im["split"] == split for im in images) for split in splits},
    }
    result["fingerprint"] = digest(result)
    return result
