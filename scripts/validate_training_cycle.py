"""Opt-in real GPU + HTTP + storage cycle; use only disposable private directories."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import threading
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.geometry import mask_payload
from app.storage import ProjectStore
from app.training.agent import Job, capabilities
from app.training.control import LocalControl
from app.training.local import LocalBackend
from app.training.routes import project_router
from app.training.transport import install_agent


def validate(root, weights, family, task):
    root.mkdir(parents=True, exist_ok=False)
    images = root / "originals"
    images.mkdir()
    for index in range(10):
        image = Image.new("RGB", (128, 96), (index * 10, 30, 40))
        ImageDraw.Draw(image).rectangle((20, 20, 60, 55), fill=(220, 140, 30))
        image.save(images / f"{index}.png")
    project = ProjectStore.open(root / "project", images)
    category = project.add_category("Object", "#abcdef")
    mask = np.zeros((96, 128), bool)
    mask[20:56, 20:61] = True
    mask[25:28, 25:28] = False
    for image in project.project()["images"]:
        state = project.get_state(image["id"])
        annotation = (
            {"kind": "bbox", "bbox": [20, 20, 41, 36]}
            if task == "detection"
            else mask_payload(mask, include_preview=False)
        )
        state.update(
            schema_version=2,
            annotations=[
                {"id": str(uuid4()), "category_id": category["id"], "iscrowd": 0, **annotation}
            ],
        )
        project.save_state(image["id"], state)

    control = LocalControl(root / "control")
    backend = LocalBackend(project, lambda: control.devices(project.directory))
    backend.control = control
    application = FastAPI()
    application.include_router(project_router(lambda _: backend), prefix="/training")
    install_agent(application, control, "/agent")
    web = TestClient(application)
    agent = TestClient(application, base_url="http://testserver/agent/")

    def ok(response):
        response.raise_for_status()
        return response.json()

    overview = ok(web.get("/training", params={"task": task}))
    ok(
        web.post(
            "/training/reviews",
            json={
                "task": task,
                "images": {image["id"]: image["revision"] for image in overview["images"]},
            },
        )
    )
    snapshot = ok(web.post("/training/datasets", json={"task": task, "test_percent": 10}))
    devices, models, runtime = capabilities(weights)
    device = devices[0]
    pairing = ok(web.post("/training/compute/pair"))
    node = ok(
        agent.post(
            "/enroll",
            json={
                "code": pairing["code"],
                "name": "Disposable GPU validation",
                "devices": [{k: v for k, v in device.items() if k != "uuid"}],
                "models": models,
                "runtime": runtime,
            },
        )
    )
    agent.headers["Authorization"] = "Bearer " + node["token"]
    run = ok(
        web.post(
            "/training/runs",
            json={
                "dataset_id": snapshot["id"],
                "model_id": f"{family}-nano-{task}",
                "device_id": node["node_id"] + ":" + device["slot"],
                "allow_lossy": True,
                "request_id": str(uuid4()),
                "parameters": {"epochs": 3, "batch": 2},
            },
        )
    )

    def execute(pause=False):
        assignment = ok(agent.post("/claim", json={"slot": device["slot"]}))
        assert assignment is not None
        if assignment["kind"] == "run":
            assert all(image["split"] != "test" for image in assignment["dataset"]["images"])
        if pause:
            ok(web.post(f"/training/runs/{run['id']}/pause"))
        key = hashlib.sha256(
            (assignment["project_id"] + ":" + assignment["work"]["id"]).encode()
        ).hexdigest()
        Job(agent, assignment, root / "jobs" / key, weights, device, threading.Event()).run()
        return ok(web.get(f"/training/runs/{run['id']}"))

    paused = execute(pause=True)
    assert paused["status"] == "paused" and paused["checkpoint"], paused
    ok(web.post(f"/training/runs/{run['id']}/resume"))
    completed = execute()
    assert completed["status"] == "completed" and completed["epoch"] > paused["epoch"], completed
    ok(web.post(f"/training/runs/{run['id']}/evaluate"))
    evaluated = execute()
    report = evaluated["evaluation"]
    assert report["status"] == "completed" and report["snapshot_id"] == snapshot["id"], report
    model = next(
        artifact for artifact in completed["artifacts"] if artifact["name"].startswith("model.")
    )
    downloaded = web.get(f"/training/artifacts/{model['id']}")
    assert (
        downloaded.is_success
        and hashlib.sha256(downloaded.content).hexdigest() == report["checkpoint_sha256"]
    )
    ok(
        web.post(
            f"/training/runs/{run['id']}/predict",
            # Three epochs establish execution, not calibrated confidence.
            json={"image_ids": [1], "confidence": 0.001, "purpose": "preannotate"},
        )
    )
    predicted = execute()
    assert predicted["prediction"]["status"] == "completed", predicted
    batch = predicted["prediction"]["id"]
    before = project.get_state(1)
    ok(web.post(f"/training/predictions/{batch}/apply"))
    after = project.get_state(1)
    assert after["annotations"] == before["annotations"]
    proposals = after["detection"]["proposals"] if task == "detection" else after["proposals"]
    assert proposals and all(not proposal["selected"] for proposal in proposals)
    if task == "segmentation":
        assert all(proposal["mask"]["size"] == [96, 128] for proposal in proposals)
    evidence = {
        "architecture": platform.machine(),
        "runtime": runtime,
        "family": family,
        "task": task,
        "run_id": run["id"],
        "snapshot_id": snapshot["id"],
        "evaluation": report,
        "proposal_count": len(proposals),
        "confirmed_annotations_preserved": True,
        "paused_epoch": paused["epoch"],
        "completed_epoch": completed["epoch"],
    }
    (root / "report.json").write_text(json.dumps(evidence, indent=2))
    print("Passed review → real agent → pause/resume → test → proposals", flush=True)
    agent.close()
    web.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--family", choices=["yolo26", "rfdetr"], default="yolo26")
    parser.add_argument("--task", choices=["detection", "segmentation"], default="detection")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("Keep disposable data and evidence outside the public checkout")
    validate(args.output.resolve(), args.weights.resolve(), args.family, args.task)
