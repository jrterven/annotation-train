"""Real PostgreSQL GPU reservation races, isolated by disposable test schemas."""

from dataclasses import replace
from types import SimpleNamespace
import time
from uuid import uuid4
from sqlalchemy import select

from test_hosted_concurrency import postgres, race
from app.hosted.models import Project, TrainingNode
from app.hosted.storage import create_project_store, FilesystemStore
from app.hosted.training import HostedBackend, HostedControl
from app.training.coordinator import claim, update
from app.training.protocol import Update
from app.training.catalog import model_profile, Parameters


def test_gpu_reservations_are_atomic_across_projects_and_independent_devices(postgres, tmp_path):
    db, settings = postgres
    settings = replace(settings, storage_limit_bytes=10**8, global_storage_limit_bytes=10**8)
    objects = FilesystemStore(tmp_path / "objects")
    with db.session() as session:
        session.add(Project(id="alice-second", user_id="alice", name="second", quota_version=1))
        session.get(Project, "alice").quota_version = 1
        session.add(TrainingNode(id="agent", user_id="alice", token_hash="unused", value={}))
    control = HostedControl(
        SimpleNamespace(state=SimpleNamespace(db=db, settings=settings, objects=objects))
    )
    node = {
        "id": "agent",
        "user_id": "alice",
        "name": "Test",
        "devices": [{"slot": "0", "enabled": True}, {"slot": "1", "enabled": True}],
        "models": ["yolo26-small-detection"],
        "runtime": {"test": "true"},
        "leases": {},
    }
    backends = []
    for project_id in ("alice", "alice-second"):
        create_project_store(settings, project_id, project_id)
        backend = HostedBackend(settings, db, objects, "alice", project_id)
        backends.append(backend)
        with backend.repo.transaction() as records:
            records.put(
                "dataset",
                "dataset",
                {"id": "dataset", "images": [], "classes": [], "task": "detection"},
            )
            for slot in ("0", "1"):
                id = str(uuid4())
                records.put(
                    "run",
                    id,
                    {
                        "id": id,
                        "dataset_id": "dataset",
                        "created_at": time.time(),
                        "device_id": f"agent:{slot}",
                        "model": model_profile("yolo26-small-detection"),
                        "parameters": Parameters().model_dump(),
                        "status": "queued",
                        "events": 0,
                        "metrics": [],
                    },
                )

    def reserve(backend, slot):
        with backend.repo.transaction() as records:
            row = records.session.get(TrainingNode, "agent")
            import copy

            fresh = copy.deepcopy(row.value or node)
            result = claim(records, fresh, slot, backend.project_id)
            row.value = fresh
            return result

    results = race(
        [lambda b=b, s=s: reserve(b, s) for b in backends for s in ("0", "1") for _ in range(5)]
    )
    assigned = [result for result in results if isinstance(result, dict)]
    assert len(assigned) == 2
    assert {a["work"]["slot"] for a in assigned} == {"0", "1"}
    # Expiration interrupts, never requeues work automatically.
    with db.session() as session:
        row = session.get(TrainingNode, "agent")
        row.value = {**row.value, "leases": {}}
    for assignment in assigned:
        backend = next(b for b in backends if b.project_id == assignment["project_id"])
        with backend.repo.transaction() as records:
            work = records.get("run", assignment["work"]["id"])
            work["lease_until"] = 0
            records.put("run", work["id"], work)
        reserve(backend, assignment["work"]["slot"])
        with backend.repo.transaction() as records:
            assert records.get("run", assignment["work"]["id"])["status"] == "interrupted"
