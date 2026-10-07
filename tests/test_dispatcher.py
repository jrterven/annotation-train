import hashlib
import io
from datetime import timedelta
from uuid import uuid4

from PIL import Image
import pytest
import httpx
from sqlalchemy import create_engine, select

from app.hosted.config import Settings
from app.hosted.database import Database
from app.hosted.dispatcher import Dispatcher, Worker, WorkerTransport
from app.hosted.jobs import cancel, enqueue
from app.hosted.models import Attempt, ImageObject, InferenceUsage, Job, Project, User, utcnow
from app.hosted.storage import FilesystemStore


class Clock:
    def __init__(self):
        self.now = utcnow()

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


class Transport:
    def __init__(self):
        self.requests = []
        self.attempts = {}
        self.down = set()

    def health(self, worker):
        return {"ready": worker.name not in self.down, "busy": False}

    def submit(self, worker, value):
        self.requests.append((worker.name, value))
        response = {"job_id": value["job_id"], "attempt_id": value["attempt_id"], "status": "running"}
        self.attempts[value["attempt_id"]] = response
        if worker.name in self.down:
            raise TimeoutError()
        return response

    def status(self, worker, attempt_id):
        if worker.name in self.down:
            raise TimeoutError()
        return self.attempts[attempt_id]

    def cancel(self, worker, attempt_id):
        value = self.status(worker, attempt_id)
        value["status"] = "cancelled"
        return value


@pytest.fixture
def setup(tmp_path):
    settings = Settings(database_url="unused", public_url="http://localhost:8765", google_client_id="test",
                        google_client_secret="test", r2_endpoint_url="unused", r2_access_key_id="test",
                        r2_secret_access_key="test", primary_worker_url="http://primary", primary_worker_token="a" * 32,
                        fallback_worker_url="http://fallback", fallback_worker_token="b" * 32,
                        data_dir=tmp_path, cache_dir=tmp_path / "cache", min_free_disk_bytes=0)
    db = Database(settings, engine=create_engine("sqlite://"))
    db.create_schema()
    objects = FilesystemStore(tmp_path / "objects")
    buffer = io.BytesIO()
    Image.new("RGB", (30, 20)).save(buffer, format="PNG")
    data = buffer.getvalue()
    users = []
    with db.session() as session:
        for i in range(2):
            uid, pid = str(uuid4()), str(uuid4())
            session.add(User(id=uid, google_sub=str(i), email=f"{i}@example.test", name=f"User {i}"))
            session.add(Project(id=pid, user_id=uid, name="Project"))
            session.add(ImageObject(id=str(uuid4()), project_id=pid, image_id=1, file_name="a.png", object_key=pid,
                                    sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data), width=30, height=20,
                                    content_type="image/png", status="ready"))
            objects.put(pid, data)
            users.append((uid, pid))
    transport, clock = Transport(), Clock()
    dispatcher = Dispatcher(settings, db, objects, transport=transport, clock=clock)
    return settings, db, dispatcher, transport, clock, users


def add(setup, user=0):
    settings, db, _, _, _, users = setup
    uid, pid = users[user]
    return enqueue(db, settings, uid, pid, 1, "points", {"revision": 1, "part": {"box": [0, 0, 10, 10]}})["id"]


def state(db, job_id):
    with db.session() as session:
        return session.get(Job, job_id)


def success(transport, job):
    transport.attempts[job.attempt_id].update(status="succeeded", result={"mask": {"size": [20, 30], "counts": "test"}})


def test_single_global_active_and_fair_user_rotation(setup):
    _, db, dispatcher, transport, _, users = setup
    first, second, other = add(setup), add(setup), add(setup, 1)
    dispatcher.tick()
    assert state(db, first).status == "running"
    dispatcher.tick()
    assert len(transport.requests) == 1
    success(transport, state(db, first))
    dispatcher.tick()
    dispatcher.tick()
    assert state(db, other).status == "running"
    assert state(db, second).status == "queued"
    with db.session() as session:
        usage = session.get(InferenceUsage, (users[0][0], state(db, first).quota_day))
        assert usage.used == 1 and usage.reserved == 1


def test_lost_worker_waits_for_lease_then_retries_once_and_fences_late_result(setup):
    _, db, dispatcher, transport, clock, _ = setup
    job_id = add(setup)
    dispatcher.tick()
    old = state(db, job_id)
    transport.down.add("primary")
    clock.advance(21)
    dispatcher.tick()
    assert len(transport.requests) == 1
    # A separate dispatcher recovers exactly the persisted attempt.
    resumed = Dispatcher(dispatcher.settings, db, dispatcher.objects, transport=transport, clock=clock)
    clock.advance(105)
    resumed.tick()
    current = state(db, job_id)
    assert current.worker_name == "fallback" and current.attempt_count == 2
    assert current.attempt_id != old.attempt_id
    dispatcher._handle(old, {"job_id": old.id, "attempt_id": old.attempt_id, "status": "succeeded", "result": {"stale": True}})
    assert state(db, job_id).status == "running"
    assert state(db, job_id).result is None
    transport.down.add("fallback")
    clock.advance(126)
    dispatcher.tick()
    assert state(db, job_id).status == "failed"
    assert len(transport.requests) == 2


def test_cancel_before_execution_refunds_and_after_execution_consumes_once(setup):
    _, db, dispatcher, transport, _, users = setup
    pending = add(setup)
    cancel(db, users[0][0], pending)
    assert state(db, pending).status == "cancelled"
    active = add(setup)
    dispatcher.tick()
    cancel(db, users[0][0], active)
    dispatcher.tick()
    finished = state(db, active)
    assert finished.status == "cancelled"
    # Duplicate final response cannot settle twice.
    dispatcher._complete(finished, {"status": "cancelled"})
    with db.session() as session:
        usage = session.get(InferenceUsage, (users[0][0], finished.quota_day))
        assert usage.used == 1 and usage.reserved == 0


def test_invalid_input_fails_without_gpu_retry_and_releases_quota(setup):
    _, db, dispatcher, transport, _, users = setup
    job_id = add(setup)
    dispatcher.tick()
    job = state(db, job_id)
    transport.attempts[job.attempt_id].update(status="failed", error="invalid_input")
    dispatcher.tick()
    assert state(db, job_id).status == "failed"
    assert len(transport.requests) == 1
    with db.session() as session:
        usage = session.get(InferenceUsage, (users[0][0], job.quota_day))
        assert usage.used == 0 and usage.reserved == 0


@pytest.mark.parametrize("error", ["result_expired", "result_too_large"])
def test_output_limits_and_expired_payloads_do_not_repeat_inference(setup, error):
    _, db, dispatcher, transport, _, _ = setup
    job_id = add(setup)
    dispatcher.tick()
    job = state(db, job_id)
    transport.attempts[job.attempt_id].update(status="failed", error=error)
    dispatcher.tick()
    assert state(db, job_id).status == "failed"
    assert state(db, job_id).error == error
    assert len(transport.requests) == 1


def test_primary_recovery_requires_ready_and_new_jobs_return_to_primary(setup):
    _, db, dispatcher, transport, _, _ = setup
    transport.down.add("primary")
    first = add(setup)
    dispatcher.tick()
    assert state(db, first).worker_name == "fallback"
    success(transport, state(db, first))
    dispatcher.tick()
    transport.down.remove("primary")
    second = add(setup)
    dispatcher.tick()
    assert state(db, second).worker_name == "primary"


def test_retained_dispatcher_snapshot_cannot_replay_a_purged_job(setup):
    _, db, dispatcher, transport, _, users = setup
    job_id = add(setup)
    dispatcher.tick()
    stale = state(db, job_id)
    success(transport, stale)
    dispatcher.tick()
    # A different dispatcher finished the job and retention removed its rows
    # while this dispatcher retained a snapshot across a pause or external I/O.
    with db.session() as session:
        db.global_lock(session)
        session.delete(session.get(Attempt, stale.attempt_id))
        session.delete(session.get(Job, stale.id))
    response = {"job_id": stale.id, "attempt_id": stale.attempt_id,
                "status": "succeeded", "result": {"stale": True}}
    assert dispatcher._accepted(stale, response) is False
    assert dispatcher._complete(stale, response) is None
    dispatcher._handle(stale, response)
    dispatcher._submit(stale)
    assert state(db, stale.id) is None
    assert len(transport.requests) == 1
    with db.session() as session:
        usage = session.get(InferenceUsage, (users[0][0], stale.quota_day))
        assert usage.used == 1 and usage.reserved == 0


def test_cpu_transport_bounds_even_a_misbehaving_worker_response(monkeypatch):
    monkeypatch.setattr("app.hosted.dispatcher.MAX_WORKER_RESPONSE_BYTES", 1024)
    transport = WorkerTransport()
    transport.client.close()
    transport.client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"x" * 2048)))
    try:
        with pytest.raises(ValueError, match="output limit"):
            transport.health(Worker("primary", "http://worker", "secret"))
    finally:
        transport.close()


def test_repeated_segmentation_reuses_verified_image_without_r2_download(setup, monkeypatch):
    _, db, dispatcher, transport, _, _ = setup
    first = add(setup)
    dispatcher.tick()
    original = transport.requests[0][1]["image_base64"]
    success(transport, state(db, first))
    dispatcher.tick()
    monkeypatch.setattr(dispatcher.objects, "get", lambda *_: (_ for _ in ()).throw(OSError("R2 unavailable")))
    second = add(setup)
    dispatcher.tick()
    assert state(db, second).status == "running"
    assert transport.requests[-1][1]["image_base64"] == original
    assert transport.requests[-1][1]["attempt_id"] != transport.requests[0][1]["attempt_id"]


def test_dispatcher_rechecks_queue_quickly_but_backs_off_unavailable_workers(setup):
    _, _, dispatcher, transport, _, _ = setup
    class Stop:
        def __init__(self): self.delays = []
        def is_set(self): return len(self.delays) == 1
        def wait(self, seconds): self.delays.append(seconds)
    transport.close = lambda: None
    idle = Stop()
    dispatcher.run(idle)
    assert idle.delays == [0.2]
    add(setup)
    transport.down.update({"primary", "fallback"})
    unavailable = Stop()
    dispatcher.run(unavailable)
    assert unavailable.delays == [5]
    transport.down.clear()
    active = Stop()
    dispatcher.run(active)
    assert active.delays == [0.2]


def test_idle_dispatcher_does_not_query_gpu_health(setup, monkeypatch):
    _, _, dispatcher, transport, _, _ = setup
    calls = []
    monkeypatch.setattr(transport, "health", lambda worker: calls.append(worker))
    dispatcher.tick()
    assert calls == []
