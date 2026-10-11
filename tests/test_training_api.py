"""CPU contract tests: real storage/HTTP, simulated GPU callbacks and opaque weights."""

from dataclasses import replace
import hashlib
import io
import json
import time
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import pytest
from sqlalchemy import select

from app import main
from app.geometry import mask_payload
from app.training.catalog import catalog
from app.training.agent import server_url
from app.training.control import LocalControl
from app.hosted.models import TrainingRecord, User, ResourceCounter
from test_hosted_backend import hosted


def ok(response, status=None):
    assert response.status_code == status if status else response.is_success, response.text
    return response.json()


@pytest.fixture(params=["local", "hosted"])
def environment(request, tmp_path, monkeypatch, hosted):
    mode = request.param
    if mode == "local":
        monkeypatch.setattr(main, "_store", None)
        # Route closures refer to the original control object. Isolate its repository.
        control = LocalControl(tmp_path / "control")
        monkeypatch.setattr(main.training_control, "repo", control.repo)
        directory = tmp_path / "project"
        images = directory / "images"
        images.mkdir(parents=True)
        for index in range(10):
            Image.new("RGB", (16, 12), (index * 20, 0, 0)).save(images / f"{index}.png")
        client = TestClient(main.app, raise_server_exceptions=False)
        ok(
            client.post(
                "/api/projects/open", json={"directory": str(directory), "image_root": str(images)}
            )
        )
        root, agent_root, category_path = "/api", "/api/training-agent", "/api/project/categories"
        other = None
    else:
        settings, db, objects, app, clients = hosted
        app.state.settings = replace(settings, training_enabled=True)
        # Control and enable closure use app.state; control settings need no flag.
        client, other = clients["alice"], clients["bob"]
        project = ok(client.post("/api/v1/projects", json={"name": "Training"}))
        root = f"/api/v1/projects/{project['id']}"
        agent_root, category_path = "/api/v1/training-agent", root + "/categories"
        for index in range(10):
            output = io.BytesIO()
            Image.new("RGB", (16, 12), (index * 20, 0, 0)).save(output, "PNG")
            ok(
                client.post(
                    root + "/images",
                    files={"file": (f"{index}.png", output.getvalue(), "image/png")},
                )
            )
    client.headers["X-Annotation-State-Version"] = "2"
    category = ok(client.post(category_path, json={"name": "Object", "color": "#abcdef"}))
    mask = np.zeros((12, 16), bool)
    mask[1:10, 1:10] = True
    mask[3:5, 3:5] = False
    mask[10, 14] = True
    for image_id in range(1, 11):
        state = ok(client.get(root + f"/images/{image_id}/state"))
        state.update(
            schema_version=2,
            annotations=[
                {
                    "id": f"box-{image_id}",
                    "kind": "bbox",
                    "category_id": category["id"],
                    "bbox": [1, 1, 8, 9],
                    "iscrowd": 0,
                },
                {
                    "id": f"mask-{image_id}",
                    "category_id": category["id"],
                    "iscrowd": 0,
                    **mask_payload(mask),
                },
            ],
        )
        ok(
            client.put(
                root + f"/images/{image_id}/state",
                json=state,
                headers={"X-Annotation-State-Version": "2"},
            )
        )
    yield {
        "client": client,
        "other": other,
        "root": root,
        "agent_root": agent_root,
        "mode": mode,
        "hosted": hosted,
        "mask": mask,
    }
    if mode == "local":
        client.close()


def prepare(env, task="detection", test=True, family="yolo26"):
    client, base = env["client"], env["root"] + "/training"
    overview = ok(client.get(base, params={"task": task}))
    assert not any(image["reviewed"] for image in overview["images"])
    ok(
        client.post(
            base + "/reviews",
            json={"task": task, "images": {im["id"]: im["revision"] for im in overview["images"]}},
        )
    )
    dataset = ok(
        client.post(base + "/datasets", json={"task": task, "test_percent": 10 if test else 0})
    )
    assert "object_key" not in json.dumps(dataset)
    pairing = ok(client.post(base + "/compute/pair"))
    enrollment = {
        "code": pairing["code"],
        "name": "Disposable contract agent",
        "models": [m["id"] for m in catalog()],
        "runtime": {"adapter": "test-only"},
        "devices": [
            {"slot": "0", "name": "Simulated GPU"},
            {"slot": "1", "name": "Second simulated GPU"},
        ],
    }
    agent = TestClient(client.app, base_url=str(client.base_url), raise_server_exceptions=False)
    connected = ok(agent.post(env["agent_root"] + "/enroll", json=enrollment))
    assert agent.post(env["agent_root"] + "/enroll", json=enrollment).status_code == 401
    agent.headers["Authorization"] = "Bearer " + connected["token"]
    request = {
        "dataset_id": dataset["id"],
        "model_id": f"{family}-small-{task}",
        "device_id": connected["node_id"] + ":0",
        "allow_lossy": True,
        "request_id": str(uuid4()),
    }
    run = ok(client.post(base + "/runs", json=request))
    assert ok(client.post(base + "/runs", json=request))["id"] == run["id"]
    assert (
        client.post(base + "/runs", json={**request, "parameters": {"epochs": 2}}).status_code
        == 409
    )
    assigned = ok(agent.post(env["agent_root"] + "/claim", json={"slot": "0"}))
    assert assigned["work"]["id"] == run["id"]
    assert ok(agent.post(env["agent_root"] + "/claim", json={"slot": "0"})) is None
    return agent, dataset, run, assigned


def test_agent_discovers_real_local_and_hosted_loopback_routes(environment):
    env = environment
    assert server_url("http://localhost:8765", client=env["client"]) == (
        "http://localhost:8765" + env["agent_root"]
    )


def test_dataset_versions_survive_deleting_older_latest_and_all_snapshots(environment):
    client, base = environment["client"], environment["root"] + "/training"
    overview = ok(client.get(base, params={"task": "detection"}))
    ok(client.post(base + "/reviews", json={
        "task": "detection", "images": {im["id"]: im["revision"] for im in overview["images"]}
    }))

    def create(seed, version):
        body = {"task": "detection", "seed": seed}
        dataset = ok(client.post(base + "/datasets", json=body))
        assert dataset["version"] == version
        assert ok(client.post(base + "/datasets", json=body)) == dataset
        return dataset["id"]

    first, second = create(1, 1), create(2, 2)
    ok(client.delete(base + "/datasets/" + first))
    third = create(3, 3)
    ok(client.delete(base + "/datasets/" + third))
    fourth = create(4, 4)
    for id in (second, fourth):
        ok(client.delete(base + "/datasets/" + id))
    create(5, 5)


@pytest.mark.parametrize("deleted_versions", [(1,), (2,), (1, 2)])
def test_legacy_dataset_versions_survive_deletion_before_first_new_snapshot(environment, deleted_versions):
    from app.training.store import LocalRepository
    from app.hosted.training import PostgresRepository
    from app.hosted.models import Project

    env = environment
    client, base = env["client"], env["root"] + "/training"
    overview = ok(client.get(base, params={"task": "detection"}))
    ok(client.post(base + "/reviews", json={
        "task": "detection", "images": {im["id"]: im["revision"] for im in overview["images"]}
    }))
    snapshots = {
        seed: ok(client.post(base + "/datasets", json={"task": "detection", "seed": seed}))
        for seed in (1, 2)
    }
    if env["mode"] == "local":
        repo = LocalRepository(main._store.directory)
    else:
        settings, db, objects, _, _ = env["hosted"]
        project_id = env["root"].split("/")[-1]
        with db.session() as session:
            user_id = session.get(Project, project_id).user_id
        repo = PostgresRepository(settings, db, objects, user_id, project_id)
    # Simulate an upgrade from snapshots stored before sequence records existed.
    with repo.transaction() as records:
        records.delete("sequence", "dataset")
    for version in deleted_versions:
        ok(client.delete(base + "/datasets/" + snapshots[version]["id"]))
    created = ok(client.post(base + "/datasets", json={"task": "detection", "seed": 3}))
    assert created["version"] == 3


def update(env, agent, assigned, sequence=0, **event):
    return agent.post(
        env["agent_root"] + "/update",
        json={
            "project_id": assigned["project_id"],
            "kind": assigned["kind"],
            "work_id": assigned["work"]["id"],
            "attempt_id": assigned["work"]["attempt_id"],
            "sequence": sequence,
            **event,
        },
    )


def upload(env, agent, assigned):
    data = b"opaque fake weights: never deserialized by the server"
    model_name = "model.pt" if assigned["work"]["model"]["family"] == "yolo26" else "model.pth"
    artifact = ok(
        agent.post(
            env["agent_root"] + "/artifacts",
            json={
                "project_id": assigned["project_id"],
                "work_id": assigned["work"]["id"],
                "attempt_id": assigned["work"]["attempt_id"],
                "name": model_name,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            },
        )
    )
    path = env["agent_root"] + f"/projects/{assigned['project_id']}/artifacts/{artifact['id']}"
    ok(agent.put(path, params={"offset": 0}, content=data[:10]))
    ok(agent.put(path, params={"offset": 0}, content=data[:10]))
    assert agent.put(path, params={"offset": 0}, content=b"different").status_code == 409
    ok(agent.put(path, params={"offset": 10}, content=data[10:]))
    return ok(agent.post(path + "/complete")), data


@pytest.mark.parametrize("task", ["detection", "segmentation"])
@pytest.mark.parametrize("family", ["yolo26", "rfdetr"])
@pytest.mark.parametrize("test", [False, True])
def test_full_training_test_and_proposal_contract(environment, task, family, test):
    env = environment
    client = env["client"]
    base = env["root"] + "/training"
    agent, dataset, run, assigned = prepare(env, task, test, family)
    assert {im["split"] for im in assigned["dataset"]["images"]} == {"train", "validation"}
    route = (
        env["agent_root"]
        + f"/projects/{assigned['project_id']}/run/{run['id']}/{assigned['work']['attempt_id']}"
    )
    for im in dataset["images"]:
        response = agent.get(route + f"/images/{im['id']}")
        assert response.status_code == (404 if im["split"] == "test" else 200)
    metric = {"epoch": 1, "train_loss": 1.2, "validation_loss": 1.4, "map": 0.2}
    ok(update(env, agent, assigned, metric=metric, checkpoint=True))
    ok(update(env, agent, assigned, metric=metric, checkpoint=True))
    assert len(ok(client.get(base + f"/runs/{run['id']}"))["metrics"]) == 1
    artifact, data = upload(env, agent, assigned)
    done = {"status": "completed", "checkpoint": True}
    ok(update(env, agent, assigned, 1, **done))
    ok(update(env, agent, assigned, 1, **done))
    assert client.get(base + f"/artifacts/{artifact['id']}").content == data
    if env["other"]:
        assert env["other"].get(base + f"/artifacts/{artifact['id']}").status_code == 404
        assert (
            TestClient(client.app, base_url=str(client.base_url))
            .get(base + f"/artifacts/{artifact['id']}")
            .status_code
            == 401
        )
    evaluated = client.post(base + f"/runs/{run['id']}/evaluate")
    if test:
        evaluation = ok(evaluated)
        assignment = ok(agent.post(env["agent_root"] + "/claim", json={"slot": "0"}))
        assert {im["split"] for im in assignment["dataset"]["images"]} == {"test"}
        report = {
            "metrics": {"mAP50-95": 0.3},
            "snapshot_id": dataset["id"],
            "checkpoint_sha256": artifact["sha256"],
            "partition": "test",
        }
        assert (
            update(
                env,
                agent,
                assignment,
                result={**report, "partition": "validation"},
                status="completed",
            ).status_code
            == 422
        )
        ok(update(env, agent, assignment, result=report, status="completed"))
        saved = ok(client.get(base + f"/runs/{run['id']}"))["evaluation"]
        assert saved["metrics"] == report["metrics"]
        assert ok(client.post(base + f"/runs/{run['id']}/evaluate"))["id"] == evaluation["id"]
    else:
        assert evaluated.status_code == 422
    prediction = ok(
        client.post(base + f"/runs/{run['id']}/predict", json={"image_ids": [1], "confidence": 0.5})
    )
    assignment = ok(agent.post(env["agent_root"] + "/claim", json={"slot": "0"}))
    assert (
        agent.get(
            env["agent_root"]
            + f"/projects/{assignment['project_id']}/operation/{assignment['work']['id']}/{assignment['work']['attempt_id']}/model"
        ).content
        == data
    )
    proposal = (
        {"kind": "bbox", "bbox": [2, 2, 3, 4]}
        if task == "detection"
        else mask_payload(env["mask"], include_preview=False)
    )
    proposal.update(category_id=dataset["classes"][0]["category_id"], score=0.9)
    ok(
        update(
            env,
            agent,
            assignment,
            status="completed",
            result={"images": [{"image_id": 1, "proposals": [proposal]}]},
        )
    )
    before = ok(client.get(env["root"] + "/images/1/state"))
    ok(client.post(base + f"/predictions/{prediction['id']}/apply"))
    after = ok(client.get(env["root"] + "/images/1/state"))
    assert after["annotations"] == before["annotations"]
    target = after["detection"] if task == "detection" else after
    assert len(target["proposals"]) == 1
    assert target["proposals"][0]["selected"] is False
    ok(client.post(base + f"/predictions/{prediction['id']}/apply"))
    assert ok(client.get(env["root"] + "/images/1/state")) == after
    if env["mode"] == "hosted":
        _, db, _, _, _ = env["hosted"]
        with db.session() as session:
            assert session.get(ResourceCounter, "global").storage_bytes == sum(
                u.storage_bytes for u in session.scalars(select(User))
            )
            assert all(r.size_bytes > 0 for r in session.scalars(select(TrainingRecord)))
    ok(client.delete(base + f"/runs/{run['id']}"))
    assert client.get(base + f"/artifacts/{artifact['id']}").status_code == 404
    ok(client.delete(base + f"/datasets/{dataset['id']}"))


def test_pause_resume_cancel_fencing_and_revocation(environment):
    env = environment
    client = env["client"]
    base = env["root"] + "/training"
    agent, dataset, run, assigned = prepare(env)
    ok(update(env, agent, assigned, checkpoint=True, metric={"epoch": 1, "map": 0.4}))
    ok(client.post(base + f"/runs/{run['id']}/pause"))
    assert ok(update(env, agent, assigned, 1))["control"] == "pause"
    ok(update(env, agent, assigned, 2, status="paused", checkpoint=True))
    ok(client.post(base + f"/runs/{run['id']}/resume"))
    resumed = ok(agent.post(env["agent_root"] + "/claim", json={"slot": "0"}))
    assert resumed["work"]["resume"] is True
    assert resumed["work"]["attempt_id"] != assigned["work"]["attempt_id"]
    assert update(env, agent, assigned, 3).status_code == 409
    ok(client.post(base + f"/runs/{run['id']}/cancel"))
    assert ok(update(env, agent, resumed))["control"] == "cancel"
    ok(update(env, agent, resumed, 1, status="cancelled"))
    node = resumed["work"]["node_id"]
    ok(client.post(base + f"/compute/{node}/revoke"))
    assert agent.post(env["agent_root"] + "/claim", json={"slot": "0"}).status_code == 401


def test_expired_operation_updates_saved_parent_and_allows_explicit_retry(environment, monkeypatch):
    from app.training import coordinator, service
    from types import SimpleNamespace

    env = environment
    client, base = env["client"], env["root"] + "/training"
    agent, _, run, assignment = prepare(env)
    assert ok(client.get(base + "/status"))["active_runs"] == 1
    upload(env, agent, assignment)
    ok(update(env, agent, assignment, status="completed"))
    pending = ok(client.post(base + f"/runs/{run['id']}/evaluate"))
    operation = ok(agent.post(env["agent_root"] + "/claim", json={"slot": "0"}))
    ok(update(env, agent, operation))
    future = time.time() + 120
    monkeypatch.setattr(coordinator, "time", SimpleNamespace(time=lambda: future))
    monkeypatch.setattr(service, "time", SimpleNamespace(time=lambda: future))
    assert ok(agent.post(env["agent_root"] + "/claim", json={"slot": "0"})) is None
    current = ok(client.get(base + f"/runs/{run['id']}"))
    assert current["evaluation"]["status"] == "interrupted"
    assert update(env, agent, operation, 1, status="completed", result={}).status_code == 409
    retry = ok(client.post(base + f"/runs/{run['id']}/evaluate"))
    assert retry["id"] != pending["id"]


def test_changed_revision_and_removed_class_never_overwrite_annotations(environment):
    env = environment
    client = env["client"]
    base = env["root"] + "/training"
    agent, dataset, run, assigned = prepare(env)
    upload(env, agent, assigned)
    ok(update(env, agent, assigned, status="completed"))
    prediction = ok(client.post(base + f"/runs/{run['id']}/predict", json={"image_ids": [1]}))
    operation = ok(agent.post(env["agent_root"] + "/claim", json={"slot": "0"}))
    proposal = {
        "kind": "bbox",
        "bbox": [2, 2, 3, 4],
        "category_id": dataset["classes"][0]["category_id"],
        "score": 0.9,
    }
    ok(
        update(
            env,
            agent,
            operation,
            status="completed",
            result={"images": [{"image_id": 1, "proposals": [proposal]}]},
        )
    )
    state = ok(client.get(env["root"] + "/images/1/state"))
    state["annotations"][0]["bbox"] = [3, 3, 4, 5]
    current = ok(client.put(env["root"] + "/images/1/state", json=state))
    assert client.post(base + f"/predictions/{prediction['id']}/apply").status_code == 409
    assert ok(client.get(env["root"] + "/images/1/state")) == current
    assert not ok(client.get(base, params={"task": "detection"}))["images"][0]["reviewed"]
    assert (
        ok(client.get(base + f"/datasets/{dataset['id']}"))["images"][0]["annotations"][0]["bbox"]
        != current["annotations"][0]["bbox"]
    )


def test_disabled_training_and_csrf_remain_closed(hosted):
    _, _, _, app, clients = hosted
    client = clients["alice"]
    project = ok(client.post("/api/v1/projects", json={"name": "Private"}))
    base = f"/api/v1/projects/{project['id']}/training"
    assert client.get(base).status_code == 404
    app.state.settings = replace(app.state.settings, training_enabled=True)
    token = client.headers.pop("X-CSRF-Token")
    assert client.post(base + "/compute/pair").status_code == 403
    client.headers["X-CSRF-Token"] = token
    assert clients["bob"].post(base + "/compute/pair").status_code == 404


@pytest.mark.parametrize("task", ["detection", "segmentation"])
def test_compact_split_preview_binds_creation_to_the_approved_data(environment, task):
    env = environment
    _, dataset, *_ = prepare(env, task=task)
    client, root = env["client"], env["root"]
    base = root + "/training"
    options = {"task": task, "test_percent": 10}
    preview = ok(client.post(base + "/datasets/preview?summary_only=true", json=options))
    assert "images" not in preview and "annotations" not in json.dumps(preview)
    assert preview["counts"] == dataset["counts"]
    approved = {**options, "expected_fingerprint": preview["fingerprint"]}
    created = ok(client.post(base + "/datasets?summary_only=true", json=approved), 201)
    assert "images" not in created and created["id"] == dataset["id"]
    assert ok(client.get(base + "/datasets/" + created["id"])) == dataset

    # Same images/counts and re-reviewed content, but a different confirmed generation.
    state = ok(client.get(root + "/images/1/state"))
    annotation = next(
        a for a in state["annotations"] if (a.get("kind") == "bbox") == (task == "detection")
    )
    annotation["iscrowd"] = 1
    saved = ok(client.put(root + "/images/1/state", json=state))
    ok(client.post(base + "/reviews", json={"task": task, "images": {1: saved["revision"]}}))
    rejected = client.post(base + "/datasets?summary_only=true", json=approved)
    assert rejected.status_code == 409 and "Preview" in rejected.json()["detail"]
    assert len(ok(client.get(base, params={"task": task}))["datasets"]) == 1
    refreshed = ok(client.post(base + "/datasets/preview?summary_only=true", json=options))
    assert refreshed["fingerprint"] != preview["fingerprint"]
    assert refreshed["counts"] == preview["counts"]
    updated = ok(
        client.post(base + "/datasets?summary_only=true", json={
            **options, "expected_fingerprint": refreshed["fingerprint"],
        }), 201,
    )
    assert updated["id"] != dataset["id"]
    assert updated["fingerprint"] == refreshed["fingerprint"]


@pytest.mark.parametrize("environment", ["hosted"], indirect=True)
def test_training_artifact_reserves_quota_before_any_bytes_are_uploaded(environment):
    env = environment
    agent, _, _, assigned = prepare(env)
    _, db, _, app, _ = env["hosted"]
    body = {
        "project_id": assigned["project_id"], "work_id": assigned["work"]["id"],
        "attempt_id": assigned["work"]["attempt_id"], "name": "model.pt",
        "size_bytes": 8 * 1024**3, "sha256": hashlib.sha256(b"model").hexdigest(),
    }
    with db.session() as session:
        before = session.get(ResourceCounter, "global").storage_bytes
    path = env["agent_root"] + "/artifacts"
    assert agent.post(path, json=body).status_code == 413
    with db.session() as session:
        assert session.get(ResourceCounter, "global").storage_bytes == before
        assert not list(session.scalars(select(TrainingRecord).where(TrainingRecord.kind == "artifact")))
    assert not (app.state.settings.data_dir / "training-uploads").exists()
    body["size_bytes"] = 4096
    artifact = ok(agent.post(path, json=body))
    with db.session() as session:
        record = session.get(TrainingRecord, (assigned["project_id"], "artifact", artifact["id"]))
        assert record.value["status"] == "uploading" and record.value["offset"] == 0
        assert record.size_bytes > body["size_bytes"]
        charged = session.get(ResourceCounter, "global").storage_bytes
        assert charged == before + record.size_bytes
        assert charged == sum(user.storage_bytes for user in session.scalars(select(User)))
    assert ok(agent.post(path, json=body))["id"] == artifact["id"]
    with db.session() as session:
        assert session.get(ResourceCounter, "global").storage_bytes == charged


def test_production_local_sam_uses_the_training_gpu_reservation(tmp_path, monkeypatch):
    from app.training import gpu
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(gpu.subprocess, "check_output", lambda *_args, **_kwargs: "test-gpu\n")
    # Exercise the callback actually installed on the production entry-point engine.
    with main.engine._cuda_reservation():
        with pytest.raises(ValueError, match="reserved"):
            with gpu.reserve_gpu({"uuid": "test-gpu"}):
                pytest.fail("Training cannot acquire a GPU held by local SAM")
    with gpu.reserve_gpu({"uuid": "test-gpu"}):
        pass
