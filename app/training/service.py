"""Shared project operations. No training framework or cloud client is imported."""

from __future__ import annotations

import hashlib
import time
import uuid
from fastapi import HTTPException

from .catalog import catalog, model_profile
from .data import encoded, fingerprint, make_dataset, review_key, review_rows
from app.storage import RevisionConflict

ACTIVE = {"queued", "preparing", "running", "pausing", "syncing"}


def identity():
    return str(uuid.uuid4())


def require(records, kind, key):
    value = records.get(kind, key)
    if value is None:
        raise HTTPException(404, f"{kind.title()} not found")
    return value


def public(value):
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items() if k not in {"object_key", "token_hash"}}
    if isinstance(value, list):
        return [public(v) for v in value]
    return value


def expire(records):
    for kind in ("run", "operation"):
        for work in records.list(kind):
            if work["status"] in ACTIVE - {"queued"} and work.get("lease_until", 0) < time.time():
                work.update(status="interrupted", control=None, events=work.get("events", 0) + 1)
                records.put(kind, work["id"], work)
                if kind == "operation":
                    parent = records.get("run", work["run_id"])
                    if parent:
                        key = "evaluation" if work["operation"] == "evaluate" else "prediction"
                        if parent.get(key, {}).get("id") == work["id"]:
                            parent[key]["status"] = "interrupted"
                            parent["events"] += 1
                            records.put("run", parent["id"], parent)


def summary(dataset):
    return {key: value for key, value in dataset.items() if key != "images"}


def reviews_by_key(records):
    return {row["id"]: row for row in records.list("review")}


class TrainingService:
    def __init__(self, backend):
        self.backend = backend
        self.repo = backend.repo

    def status(self):
        with self.repo.transaction() as records:
            expire(records)
            return {"active_runs": sum(run["status"] in ACTIVE for run in records.list("run"))}

    def overview(self, task):
        source = self.backend.capture()
        devices = self.backend.devices()
        with self.repo.transaction() as records:
            expire(records)
            return {
                "active_runs": sum(run["status"] in ACTIVE for run in records.list("run")),
                "catalog": catalog(),
                "images": review_rows(source, reviews_by_key(records), task),
                "datasets": [
                    summary(row) for row in records.list("dataset") if row["task"] == task
                ],
                "runs": [row for row in records.list("run") if row["task"] == task],
                "devices": devices,
                "hosted": self.backend.hosted,
            }

    def review(self, body):
        source = self.backend.capture()
        images = {im["id"]: im for im in source["images"]}
        with self.repo.transaction() as records:
            for image_id, revision in body.images.items():
                image = images.get(image_id)
                if not image:
                    raise HTTPException(404, "Image not found")
                if revision != image["revision"]:
                    raise RevisionConflict("An image changed. Save and review it again.")
                key = review_key(body.task, image_id)
                if body.reviewed:
                    records.put(
                        "review",
                        key,
                        {
                            "id": key,
                            "image_id": image_id,
                            "task": body.task,
                            "fingerprint": fingerprint(image, source["categories"], body.task),
                        },
                    )
                else:
                    records.delete("review", key)
        return {"ok": True}

    def dataset(self, body, *, preview=False):
        source = self.backend.capture()
        with self.repo.transaction() as records:
            result = make_dataset(source, reviews_by_key(records), body)
        if body.expected_fingerprint and body.expected_fingerprint != result["fingerprint"]:
            raise RevisionConflict(
                "Reviewed data changed. Preview the split again before creating a version."
            )
        if preview:
            return result
        # Retain original bytes before committing the immutable manifest. Crash
        # residues are collectable; incomplete manifests are never published.
        self.backend.retain(result)
        with self.repo.transaction() as records:
            datasets = records.list("dataset")
            for previous in datasets:
                if previous["fingerprint"] == result["fingerprint"]:
                    return previous
            # Keep the high-water mark even after the newest (or all) snapshots
            # are deleted. Existing projects initialize it from retained data.
            sequence = records.get("sequence", "dataset") or {"last_version": 0}
            version = max(
                sequence["last_version"],
                max((row["version"] for row in datasets), default=0),
            ) + 1
            result.update(
                id=identity(), created_at=time.time(), version=version
            )
            records.put("sequence", "dataset", {"last_version": version})
            records.put("dataset", result["id"], result)
        return result

    def run(self, body):
        profile = model_profile(body.model_id)
        devices = self.backend.devices()
        with self.repo.transaction() as records:
            for previous in records.list("run"):
                if previous.get("request_id") == body.request_id:
                    if previous["request"] != body.model_dump(mode="json"):
                        raise RevisionConflict(
                            "This request ID was already used for different settings."
                        )
                    return previous
            dataset = require(records, "dataset", body.dataset_id)
            if dataset["task"] != profile["task"]:
                raise ValueError("The model and dataset tasks must match.")
            issues = []
            if profile["family"] == "yolo26":
                for im in dataset["images"]:
                    for ann in im["annotations"]:
                        if ann.get("iscrowd"):
                            raise ValueError(
                                "YOLO cannot represent crowd annotations. Choose RF-DETR or revise the data."
                            )
                        components = ann.get("components", [])
                        if len(components) > 1 or any(c.get("holes") for c in components):
                            issues.append(im["file_name"])
                if issues and not body.allow_lossy:
                    raise ValueError(
                        "YOLO fills holes and connects separate mask parts. Accept these approximations before training."
                    )
            device = next(
                (d for d in devices if d["id"] == body.device_id and body.model_id in d["models"]),
                None,
            )
            if not device or not device["online"] or not device["enabled"]:
                raise ValueError("Connect an enabled, compatible GPU before training.")
            parameters = body.parameters.model_dump()
            parameters["resolution"] = parameters["resolution"] or profile["resolution"]
            if profile["family"] == "yolo26" and parameters["resolution"] % 32:
                raise ValueError("YOLO resolution must be a multiple of 32.")
            if profile["family"] == "rfdetr" and parameters["resolution"] != profile["resolution"]:
                raise ValueError("Use this RF-DETR profile's validated input resolution.")
            run = {
                "id": identity(),
                "dataset_id": dataset["id"],
                "task": dataset["task"],
                "model": profile,
                "parameters": parameters,
                "device_id": body.device_id,
                "device_identity": device.get("identity", ""),
                "created_at": time.time(),
                "status": "queued",
                "epoch": 0,
                "request_id": body.request_id,
                "request": body.model_dump(mode="json"),
                "attempt_id": None,
                "events": 0,
                "metrics": [],
                "artifacts": [],
                "warnings": sorted(set(issues)),
                "control": None,
            }
            records.put("run", run["id"], run)
            # Each epoch is retained both in the run and in its durable event.
            records.put(
                "budget",
                run["id"],
                {
                    "remaining": parameters["epochs"] * (2048 + 256 * len(dataset["classes"]))
                    + 128 * 1024
                },
            )
        return run

    def action(self, run_id, action):
        with self.repo.transaction() as records:
            run = require(records, "run", run_id)
            if action == "pause":
                if run["status"] == "queued":
                    run["status"] = "paused"
                elif run["status"] in {"running", "preparing", "pausing"}:
                    run["status"], run["control"] = "pausing", "pause"
                else:
                    raise ValueError("This execution cannot be paused.")
            elif action == "cancel":
                if run["status"] not in ACTIVE | {"paused", "interrupted"}:
                    raise ValueError("This execution is already finished.")
                run["control"] = "cancel"
                if run["status"] in {"queued", "paused", "interrupted"}:
                    run["status"] = "cancelled"
            elif action == "resume":
                if run["status"] not in {"paused", "interrupted", "failed"}:
                    raise ValueError("Only paused or interrupted executions can resume.")
                run.update(status="queued", control=None, resume=bool(run.get("checkpoint")))
                run.pop("error", None)
                run.pop("finished_at", None)
            else:
                raise HTTPException(404, "Action not found")
            run["events"] += 1
            records.put("run", run_id, run)
            if run["status"] == "cancelled":
                records.delete("budget", run_id)
        return run

    def delete(self, kind, key):
        if kind not in {"run", "dataset"}:
            raise HTTPException(404, "Resource not found")
        artifacts = []
        with self.repo.transaction() as records:
            value = require(records, kind, key)
            if kind == "dataset":
                if any(run["dataset_id"] == key for run in records.list("run")):
                    raise ValueError("Delete executions using this dataset first")
            else:
                operations = [op for op in records.list("operation") if op["run_id"] == key]
                if value["status"] in ACTIVE or any(op["status"] in ACTIVE for op in operations):
                    raise ValueError("Cancel active work before deleting this execution")
                work_ids = {key} | {op["id"] for op in operations}
                for work_id in work_ids:
                    records.delete("budget", work_id)
                for record_kind in ("event", "attempt"):
                    for record in records.list(record_kind):
                        if record["work_id"] in work_ids:
                            record_key = (
                                record["id"]
                                if record_kind == "attempt"
                                else f"{record['work_id']}:{record['cursor']}"
                            )
                            records.delete(record_kind, record_key)
                for op in operations:
                    records.delete("operation", op["id"])
                artifacts = [a for a in records.list("artifact") if a["work_id"] == key]
                for artifact in artifacts:
                    records.delete("artifact", artifact["id"])
            records.delete(kind, key)
        if not self.backend.hosted:
            from .transport import root

            for artifact in artifacts:
                (root(self.backend) / artifact["id"]).unlink(missing_ok=True)
        # Hosted objects are collected by reconciliation, respecting backup references.
        return {"ok": True}
