"""Opt-in real GPU acceptance probe; writes disposable data and evidence outside Git."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw
from app.geometry import mask_payload
from app.training.adapters import train, evaluate, predict
from app.training.catalog import model_profile, Parameters
from app.training.data import DatasetInput, make_dataset, fingerprint, review_key
from app.training.agent import capabilities


def validate(root, weights, families, tasks, size):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This acceptance probe requires a real NVIDIA GPU")
    runtime = capabilities(weights)[2]
    report = {
        "runtime": runtime,
        "architecture": platform.machine(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "started_at": time.time(),
        "checks": [],
    }
    root.mkdir(parents=True, exist_ok=False)
    originals = root / "originals"
    originals.mkdir()
    images = []
    for index in range(10):
        image = Image.new("RGB", (128, 96), (20 + index * 7, 30, 40))
        x, y = 12 + index * 3, 14 + index * 2
        ImageDraw.Draw(image).rectangle([x, y, x + 32, y + 30], fill=(210, 160, 30))
        path = originals / f"{index}.png"
        image.save(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        path.rename(originals / sha)
        mask = np.zeros((96, 128), bool)
        mask[y : y + 31, x : x + 33] = True
        mask[y + 5 : y + 8, x + 5 : x + 8] = False
        images.append(
            {
                "id": index + 1,
                "file_name": f"{index}.png",
                "sha256": sha,
                "width": 128,
                "height": 96,
                "revision": 0,
                "annotations": [
                    {
                        "id": f"b{index}",
                        "kind": "bbox",
                        "category_id": 1,
                        "bbox": [x, y, 33, 31],
                        "iscrowd": 0,
                    },
                    {
                        "id": f"s{index}",
                        "category_id": 1,
                        "iscrowd": 0,
                        **mask_payload(mask, include_preview=False),
                    },
                ],
            }
        )
    source = {"images": images, "categories": [{"id": 1, "name": "Object"}]}
    for family in families:
        for task in tasks:
            for has_test in (False, True):
                profile = model_profile(f"{family}-{size}-{task}")
                marks = {
                    review_key(task, im["id"]): {
                        "fingerprint": fingerprint(im, source["categories"], task)
                    }
                    for im in images
                }
                snapshot = make_dataset(
                    source, marks, DatasetInput(task=task, test_percent=10 if has_test else 0)
                )
                snapshot["id"] = str(uuid4())
                manifest = {
                    **snapshot,
                    "images": [im for im in snapshot["images"] if im["split"] != "test"],
                }
                directory = root / f"{family}-{task}-{'test' if has_test else 'validation'}"
                directory.mkdir()
                shutil.copytree(originals, directory / "originals")
                work = {
                    "id": str(uuid4()),
                    "dataset_id": snapshot["id"],
                    "model": profile,
                    "parameters": Parameters(
                        epochs=3, batch=2, resolution=profile["resolution"]
                    ).model_dump(),
                    "runtime": runtime,
                }
                metrics = []

                def emit(event):
                    if event.get("metric"):
                        metrics.append(event["metric"])

                paused = train(work, manifest, directory, weights, emit, lambda: len(metrics) >= 1)
                assert paused["status"] == "paused" and metrics, paused
                last = metrics[-1]["epoch"]
                result = train(
                    {**work, "resume": True, "metrics": list(metrics)},
                    manifest,
                    directory,
                    weights,
                    emit,
                    lambda: False,
                )
                assert result["status"] == "syncing" and max(m["epoch"] for m in metrics) > last
                model = directory / result["model_file"]
                work["checkpoint_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
                predictions = predict(
                    work, {**snapshot, "images": snapshot["images"][:1]}, directory
                )
                assert len(predictions["images"]) == 1
                evaluation = None
                if has_test:
                    evaluation = evaluate(
                        work,
                        {
                            **snapshot,
                            "images": [im for im in snapshot["images"] if im["split"] == "test"],
                        },
                        directory,
                    )
                    assert (
                        evaluation["snapshot_id"] == snapshot["id"]
                        and evaluation["checkpoint_sha256"] == work["checkpoint_sha256"]
                    )
                report["checks"].append(
                    {
                        "model": profile["id"],
                        "test": has_test,
                        "metrics": metrics,
                        "evaluation": evaluation,
                        "resume": True,
                        "prediction": True,
                    }
                )
                (root / "report.json").write_text(json.dumps(report, indent=2))
                print(f"Passed {profile['id']} test={has_test}", flush=True)
    report["finished_at"] = time.time()
    (root / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--family", choices=["yolo26", "rfdetr"], action="append")
    parser.add_argument("--task", choices=["detection", "segmentation"], action="append")
    parser.add_argument("--size", choices=["nano", "small", "medium"], default="nano")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(repo):
        parser.error("Keep GPU data, weights, and evidence outside this public checkout")
    validate(
        args.output.resolve(),
        args.weights.resolve(),
        args.family or ["yolo26", "rfdetr"],
        args.task or ["detection", "segmentation"],
        args.size,
    )
