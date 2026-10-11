"""Durable, fenced GPU assignments shared by local and hosted control planes."""

import copy
import time
from fastapi import HTTPException
from app.storage import RevisionConflict
from .data import encoded
from .service import require, identity, ACTIVE, public, expire
from .protocol import LEASE_SECONDS


def device_rows(nodes):
    return [
        {
            **d,
            "id": f"{n['id']}:{d['slot']}",
            "node_id": n["id"],
            "name": f"{n['name']} · {d['name']}",
            "models": n["models"],
            "online": not n.get("revoked") and time.time() - n.get("seen_at", 0) < 45,
        }
        for n in nodes
        for d in n["devices"]
    ]


def check_attempt(work, node, attempt_id, now=None):
    now = time.time() if now is None else now
    if (
        work.get("node_id") != node["id"]
        or work.get("attempt_id") != attempt_id
        or work.get("lease_until", 0) <= now
        or work["status"] not in ACTIVE
    ):
        raise HTTPException(409, "Attempt expired or superseded; stop this process.")


def claim(records, node, slot, project_id):
    now = time.time()
    device = next((d for d in node["devices"] if d["slot"] == slot and d["enabled"]), None)
    if not device:
        return None
    # Slots are per node, across all projects. Caller persists node and records
    # atomically (PG), or holds both SQLite write locks in a fixed order (local).
    lease = node.setdefault("leases", {}).get(slot)
    if lease and lease["until"] > now:
        return None
    expire(records)
    for kind in ("operation", "run"):
        work_items = sorted(records.list(kind), key=lambda row: row["created_at"])
        for work in work_items:
            if (
                work["status"] != "queued"
                or work["device_id"] != f"{node['id']}:{slot}"
                or work["model"]["id"] not in node["models"]
            ):
                continue
            if work.get("device_identity", "") != device.get("identity", ""):
                work.update(
                    status="interrupted",
                    error="The original GPU identity changed. Restore the original device.",
                )
                records.put(kind, work["id"], work)
                continue
            if work.get("runtime") and work["runtime"] != node["runtime"]:
                work.update(status="interrupted", error="Resume on the original pinned runtime.")
                records.put(kind, work["id"], work)
                continue
            attempt = identity()
            work.update(
                status="preparing",
                node_id=node["id"],
                slot=slot,
                attempt_id=attempt,
                lease_until=now + LEASE_SECONDS,
                started_at=work.get("started_at", now),
                runtime=node["runtime"],
                sequence=-1,
                control=None,
            )
            work["events"] = work.get("events", 0) + 1
            node["leases"][slot] = {
                "until": work["lease_until"],
                "attempt_id": attempt,
                "project_id": project_id,
                "work_id": work["id"],
                "kind": kind,
            }
            records.put(kind, work["id"], work)
            records.put(
                "attempt",
                attempt,
                {
                    "id": attempt,
                    "work_id": work["id"],
                    "kind": kind,
                    "node_id": node["id"],
                    "slot": slot,
                    "started_at": now,
                    "status": "preparing",
                    "runtime": node["runtime"],
                },
            )
            dataset = require(records, "dataset", work["dataset_id"])
            operation = work.get("operation", "train")
            manifest = copy.deepcopy(work.get("input", dataset))
            if operation == "train":
                manifest["images"] = [im for im in manifest["images"] if im["split"] != "test"]
            elif operation == "evaluate":
                manifest["images"] = [im for im in manifest["images"] if im["split"] == "test"]
            # Never disclose server object keys to a worker.
            for image in manifest["images"]:
                image.pop("object_key", None)
            return {
                "project_id": project_id,
                "kind": kind,
                "work": public(work),
                "dataset": manifest,
                "lease_seconds": LEASE_SECONDS,
            }
    return None


def update(records, node, body):
    work = require(records, body.kind, body.work_id)
    if (
        work.get("node_id") == node["id"]
        and work.get("attempt_id") == body.attempt_id
        and work.get("sequence") == body.sequence
        and work["status"] == body.status
        and body.status not in ACTIVE
    ):
        return {
            "control": work.get("control"),
            "lease_seconds": LEASE_SECONDS,
            "status": work["status"],
        }
    check_attempt(work, node, body.attempt_id)
    now = time.time()
    lease = node.get("leases", {}).get(work["slot"])
    if not lease or lease["attempt_id"] != body.attempt_id:
        raise HTTPException(409, "GPU assignment has changed.")
    if body.sequence > work.get("sequence", -1):
        if work.get("control") == "cancel" and body.status == "completed":
            raise HTTPException(409, "Execution was cancelled before completion.")
        work["sequence"] = body.sequence
        if body.metric:
            if body.kind != "run":
                raise ValueError("Only training executions report epoch metrics.")
            metric = body.metric.model_dump(exclude_none=True)
            dataset = require(records, "dataset", work["dataset_id"])
            valid_classes = {entry["category_id"] for entry in dataset["classes"]}
            reported_classes = [entry["category_id"] for entry in metric["per_class"]]
            if (
                len(set(reported_classes)) != len(reported_classes)
                or set(reported_classes) - valid_classes
            ):
                raise ValueError("Class metrics must match the dataset classes.")
            if metric["epoch"] > work["parameters"]["epochs"]:
                raise ValueError("Epoch exceeds the configured training budget.")
            existing = {m["epoch"]: m for m in work.get("metrics", [])}
            existing[metric["epoch"]] = metric
            work["metrics"] = [existing[key] for key in sorted(existing)]
            work["epoch"] = metric["epoch"]
        if body.progress is not None:
            work["progress"] = body.progress
        if body.checkpoint:
            work["checkpoint"] = True
        if body.error:
            work["error"] = body.error
        if body.status != "running" or work["status"] != "pausing":
            work["status"] = body.status
        if body.status == "completed" and body.kind == "run":
            artifacts = [
                a
                for a in records.list("artifact")
                if a.get("work_id") == work["id"]
                and a["status"] == "ready"
                and a["attempt_id"] == body.attempt_id
            ]
            if not any(a["name"] in {"model.pt", "model.pth"} for a in artifacts):
                raise ValueError("Upload the model before completing synchronization.")
            work["artifacts"] = [
                {"id": a["id"], "name": a["name"], "status": "ready"} for a in artifacts
            ]
        if body.status == "completed" and body.kind == "operation":
            if body.result is None:
                raise ValueError("Completed operations require a result")
            if work["operation"] == "evaluate" and any(
                body.result.get(key) != expected
                for key, expected in {
                    "snapshot_id": work["dataset_id"],
                    "checkpoint_sha256": work["checkpoint_sha256"],
                    "partition": "test",
                }.items()
            ):
                raise ValueError(
                    "Evaluation does not match the assigned checkpoint and test snapshot"
                )
        if body.result is not None:
            if len(encoded(body.result)) > 20 * 1024**2:
                raise ValueError("Result exceeds the training output budget.")
            work["result"] = body.result
        if work["status"] not in ACTIVE:
            work["finished_at"] = now
            node["leases"].pop(work["slot"], None)
        work["events"] = work.get("events", 0) + 1
        if (
            body.metric
            or body.status != "running"
            or work.get("last_recorded_status") != work["status"]
        ):
            records.put(
                "event",
                f"{work['id']}:{work['events']}",
                {
                    "work_id": work["id"],
                    "kind": body.kind,
                    "cursor": work["events"],
                    "attempt_id": body.attempt_id,
                    "sequence": body.sequence,
                    "created_at": now,
                    "status": work["status"],
                    "metric": body.metric.model_dump(exclude_none=True) if body.metric else None,
                },
            )
            work["last_recorded_status"] = work["status"]
        attempt = records.get("attempt", body.attempt_id)
        if attempt:
            attempt.update(status=work["status"], heartbeat_at=now)
            records.put("attempt", body.attempt_id, attempt)
        records.put(body.kind, body.work_id, work)
        if work["status"] in {"completed", "cancelled"}:
            records.delete("budget", work["id"])
        if body.kind == "operation":
            parent = require(records, "run", work["run_id"])
            key = "evaluation" if work["operation"] == "evaluate" else "prediction"
            parent[key] = {
                "id": work["id"],
                "status": work["status"],
                "error": work.get("error"),
                "purpose": work.get("purpose"),
            }
            if key == "evaluation" and work["status"] == "completed":
                parent[key].update(work.get("result", {}))
            parent["events"] += 1
            records.put("run", parent["id"], parent)
    if work["status"] in ACTIVE:
        work["lease_until"] = now + LEASE_SECONDS
        node["leases"][work["slot"]]["until"] = work["lease_until"]
        records.put(body.kind, body.work_id, work)
    node["seen_at"] = now
    return {
        "control": work.get("control"),
        "lease_seconds": LEASE_SECONDS,
        "status": work["status"],
    }


def enqueue_operation(service, run_id, operation, prediction=None):
    source = service.backend.capture() if operation == "predict" else None
    if source is not None:
        service.backend.retain(
            {"images": [im for im in source["images"] if im["id"] in prediction.image_ids]}
        )
    with service.repo.transaction() as records:
        parent = require(records, "run", run_id)
        if parent["status"] != "completed":
            raise ValueError("Finish training and synchronize the model first.")
        dataset = require(records, "dataset", parent["dataset_id"])
        key = "evaluation" if operation == "evaluate" else "prediction"
        existing = parent.get(key)
        if existing and (
            existing["status"] in ACTIVE
            or (operation == "evaluate" and existing["status"] == "completed")
        ):
            return existing
        artifact = next(
            (
                require(records, "artifact", a["id"])
                for a in parent["artifacts"]
                if a["name"] in {"model.pt", "model.pth"}
            ),
            None,
        )
        if not artifact:
            raise ValueError("Model is not available.")
        if operation == "evaluate" and not dataset["counts"].get("test"):
            raise ValueError("This dataset has no reserved test partition.")
        work = {
            "id": identity(),
            "run_id": run_id,
            "operation": operation,
            "task": parent["task"],
            "dataset_id": parent["dataset_id"],
            "model": parent["model"],
            "parameters": parent["parameters"],
            "device_id": parent["device_id"],
            "device_identity": parent.get("device_identity", ""),
            "created_at": time.time(),
            "status": "queued",
            "events": 0,
            "artifact_id": artifact["id"],
            "checkpoint_sha256": artifact["sha256"],
            "epoch": 0,
            "runtime": parent["runtime"],
        }
        if source is not None:
            ids = set(prediction.image_ids)
            images = [im for im in source["images"] if im["id"] in ids]
            if len(images) != len(ids):
                raise HTTPException(404, "Image not found")
            current_categories = {c["id"] for c in source["categories"]}
            if any(c["category_id"] not in current_categories for c in dataset["classes"]):
                raise RevisionConflict(
                    "A trained class was removed. Restore it before applying this model."
                )
            work["input"] = {
                "images": images,
                "classes": dataset["classes"],
                "task": parent["task"],
            }
            work["confidence"] = prediction.confidence
            work["purpose"] = prediction.purpose
        records.put("operation", work["id"], work)
        # The final test report is also retained on its parent run.
        records.put(
            "budget", work["id"], {"remaining": (42 if operation == "evaluate" else 21) * 1024**2}
        )
        parent[key] = {"id": work["id"], "status": "queued", "purpose": work.get("purpose")}
        parent["events"] += 1
        records.put("run", run_id, parent)
    return parent[key]
