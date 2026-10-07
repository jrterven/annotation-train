import base64
import hashlib
import io
import time
from uuid import uuid4

from fastapi import HTTPException
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from app.hosted.job_protocol import AttemptRequest, ResultTooLarge, decode_image, execute_inference
from app.hosted.worker import WorkerSupervisor, create_worker_app


def request(**updates):
    buffer = io.BytesIO()
    Image.new("RGB", (30, 20), "red").save(buffer, format="PNG")
    data = dict(job_id=str(uuid4()), attempt_id=str(uuid4()), project_id=str(uuid4()), image_id=1,
                sha256=hashlib.sha256(buffer.getvalue()).hexdigest(), deadline_at=time.time() + 1,
                kind="points", payload={"revision": 7, "part": {"points": [{"x": 5, "y": 5, "label": 1}]}},
                image_base64=base64.b64encode(buffer.getvalue()).decode("ascii"))
    data.update(updates)
    return AttemptRequest.model_validate(data)


class Engine:
    def predict_points(self, image, key, part):
        self.key = key
        return np.ones((image.height, image.width), dtype=bool)

    def predict_text(self, image, key, text):
        self.text = text
        return [{"mask": self.predict_points(image, key, None), "score": .8}]

    def predict_visual(self, image, key, reference, reference_box=None, text=None):
        self.reference_size, self.box = reference.size, reference_box
        return self.predict_text(image, key, text)


class Translator:
    def translate(self, text, source_language):
        return "object" if source_language == "es" else text


def echo_child(connection):
    connection.send({"type": "ready"})
    while True:
        raw = connection.recv()
        connection.send({"type": "result", "attempt_id": raw["attempt_id"], "status": "succeeded",
                         "result": {"image_id": raw["image_id"]}, "error": None})


def hanging_child(connection):
    connection.send({"type": "ready"})
    connection.recv()
    time.sleep(30)


def not_reading_child(connection):
    connection.send({"type": "ready"})
    time.sleep(30)


def until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    pytest.fail("Condition did not become true")


def test_points_preserve_revision_and_content_addressed_image_key():
    engine, value = Engine(), request()
    result = execute_inference(engine, Translator(), value)
    assert result["image_id"] == 1 and result["revision"] == 7
    assert result["mask"]["size"] == [20, 30]
    assert engine.key == f"{value.project_id}:1:{value.sha256}"


def test_text_and_visual_preserve_local_contract_and_translate():
    engine = Engine()
    value = request(kind="text", payload={"text": "objeto", "category_id": 3, "source_language": "es", "revision": 2})
    result = execute_inference(engine, Translator(), value)
    assert engine.text == "object"
    assert result["prompt"] == {"original": "objeto", "english": "object", "source_language": "es"}
    assert result["proposals"][0]["category_id"] == 3
    assert result["proposals"][0]["id"] == execute_inference(engine, Translator(), value)["proposals"][0]["id"]
    visual = request(kind="visual", payload={"category_id": 3, "reference_image": value.image_base64,
                                             "reference_box": [1, 1, 10, 10]})
    result = execute_inference(engine, Translator(), visual)
    assert result["method"] == "cross_image_exemplar"
    assert result["reference"] == {"width": 30, "height": 20}
    assert engine.box == [1, 1, 10, 10]


def test_untrusted_input_cannot_supply_paths_urls_or_seed_dimensions():
    with pytest.raises(ValueError):
        request(image_url="http://private/secret")
    with pytest.raises(ValueError):
        decode_image("file:///etc/passwd")
    with pytest.raises(ValueError, match="checksum"):
        execute_inference(Engine(), Translator(), request(sha256="0" * 64))
    with pytest.raises(ValueError, match="Seed mask"):
        execute_inference(Engine(), Translator(), request(payload={"part": {"seed_mask": {"size": [999999, 999999]}}}))


def test_worker_caps_serialized_output_and_cumulative_proposal_pixels(monkeypatch):
    engine = Engine()
    value = request()
    monkeypatch.setattr("app.hosted.job_protocol.MAX_RESULT_BYTES", 10)
    with pytest.raises(ResultTooLarge):
        execute_inference(engine, Translator(), value)
    monkeypatch.setattr("app.hosted.job_protocol.MAX_RESULT_BYTES", 20 * 1024 * 1024)
    monkeypatch.setattr("app.hosted.job_protocol.MAX_RESULT_PIXELS", 100)
    with pytest.raises(ResultTooLarge):
        execute_inference(engine, Translator(), request(kind="text", payload={"text": "object", "category_id": 1}))


def test_worker_authenticates_every_endpoint_before_reading_body(tmp_path):
    supervisor = WorkerSupervisor(tmp_path / "attempts.sqlite3", child_target=echo_child)
    app = create_worker_app(token="a" * 32, supervisor=supervisor)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        for method, path in [("GET", "/v1/health"), ("POST", "/v1/attempts"), ("GET", f"/v1/attempts/{uuid4()}")]:
            assert client.request(method, path, content=b"not even JSON").status_code == 401
        assert client.get("/v1/health", headers={"Authorization": "Bearer " + "a" * 32}).status_code == 200


def test_worker_rejects_other_tailnet_peers_even_with_valid_token(tmp_path):
    supervisor = WorkerSupervisor(tmp_path / "attempts.sqlite3", child_target=echo_child)
    app = create_worker_app(token="a" * 32, supervisor=supervisor, allowed_clients=["100.64.0.2"])
    with TestClient(app, client=("100.64.0.3", 50000)) as client:
        response = client.get("/v1/health", headers={"Authorization": "Bearer " + "a" * 32,
                                                    "X-Forwarded-For": "100.64.0.2"})
        assert response.status_code == 403


def test_timeout_kills_subprocess_before_marking_attempt_finished(tmp_path):
    supervisor = WorkerSupervisor(tmp_path / "attempts.sqlite3", max_runtime=.3, child_target=hanging_child)
    supervisor.start()
    try:
        until(lambda: supervisor.health()["ready"])
        process = supervisor.process
        value = request(deadline_at=time.time() + .3)
        supervisor.submit(value)
        until(lambda: supervisor.get(str(value.attempt_id))["status"] == "timed_out")
        assert supervisor.process is None
        assert not supervisor.health()["ready"]
        assert supervisor.active is None
    finally:
        supervisor.close()


def test_monotonic_deadline_survives_wall_clock_moving_backwards(tmp_path, monkeypatch):
    supervisor = WorkerSupervisor(tmp_path / "attempts.sqlite3", max_runtime=.3, child_target=hanging_child)
    supervisor.start()
    try:
        until(lambda: supervisor.health()["ready"])
        now = time.time()
        value = request(deadline_at=now + .3)
        supervisor.submit(value)
        monkeypatch.setattr("app.hosted.worker.time.time", lambda: now - 3600)
        until(lambda: supervisor.get(str(value.attempt_id))["status"] == "timed_out")
        assert supervisor.process is None
    finally:
        supervisor.close()


def test_blocked_large_pipe_write_does_not_block_watchdog(tmp_path):
    supervisor = WorkerSupervisor(tmp_path / "attempts.sqlite3", max_runtime=.3, child_target=not_reading_child)
    supervisor.start()
    try:
        until(lambda: supervisor.health()["ready"])
        value = request(deadline_at=time.time() + .3, image_base64="A" * 2_000_000)
        supervisor.submit(value)
        until(lambda: supervisor.get(str(value.attempt_id))["status"] == "timed_out")
        assert supervisor.process is None
    finally:
        supervisor.close()


def test_linux_container_pid_one_is_a_valid_supervisor(monkeypatch):
    import ctypes
    import sys
    from types import SimpleNamespace
    from app.hosted.worker import _model_process

    sent = []
    class Connection:
        def send(self, value):
            sent.append(value)
        def recv(self):
            raise EOFError()
        def close(self):
            pass

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("app.hosted.worker.os.getppid", lambda: 1)
    monkeypatch.setattr(ctypes, "CDLL", lambda *_: SimpleNamespace(prctl=lambda *_: 0))
    monkeypatch.setattr("app.inference.Sam3Engine", lambda **_: SimpleNamespace(load=lambda: None))
    monkeypatch.setattr("app.hosted.worker.execute_inference", lambda *_: {})
    _model_process(Connection(), expected_parent=1)
    assert sent == [{"type": "ready"}]


def test_launcher_binds_only_loopback_and_tailnet_and_never_trusts_forwarded_headers(monkeypatch):
    import socket
    import uvicorn
    from app.hosted.worker import main
    binds, configs = [], []

    class Socket:
        def setsockopt(self, *_): pass
        def bind(self, value): binds.append(value)
        def listen(self, *_): pass
        def set_inheritable(self, *_): pass
        def close(self): pass

    class Server:
        def __init__(self, config): configs.append(config)
        def run(self, *, sockets): assert len(sockets) == 2

    monkeypatch.setenv("ANNOTATION_WORKER_BIND", "100.64.0.2")
    monkeypatch.setattr(socket, "socket", lambda *_: Socket())
    monkeypatch.setattr("app.hosted.worker.create_worker_app", lambda: "app")
    monkeypatch.setattr(uvicorn, "Config", lambda app, **kwargs: kwargs)
    monkeypatch.setattr(uvicorn, "Server", Server)
    main()
    assert binds == [("127.0.0.1", 8766), ("100.64.0.2", 8766)]
    assert configs == [{"access_log": False, "proxy_headers": False}]
    monkeypatch.setenv("ANNOTATION_WORKER_BIND", "0.0.0.0")
    with pytest.raises(RuntimeError, match="Tailscale"):
        main()


def test_completed_attempt_replays_after_restart_but_changed_identity_fails(tmp_path):
    path = tmp_path / "attempts.sqlite3"
    supervisor = WorkerSupervisor(path, child_target=echo_child)
    supervisor.start()
    value = request(deadline_at=time.time() + 3)
    try:
        until(lambda: supervisor.health()["ready"])
        supervisor.submit(value)
        until(lambda: supervisor.get(str(value.attempt_id))["status"] == "succeeded")
    finally:
        supervisor.close()
    recovered = WorkerSupervisor(path, child_target=echo_child)
    try:
        assert recovered.submit(value)["status"] == "succeeded"
        changed = value.model_copy(update={"sha256": "0" * 64})
        with pytest.raises(HTTPException) as error:
            recovered.submit(changed)
        assert error.value.status_code == 409
    finally:
        recovered.close()


def test_result_payload_expires_but_replay_fence_survives(tmp_path):
    supervisor = WorkerSupervisor(tmp_path / "attempts.sqlite3", child_target=echo_child)
    supervisor.start()
    value = request(deadline_at=time.time() + 3)
    try:
        until(lambda: supervisor.health()["ready"])
        supervisor.submit(value)
        until(lambda: supervisor.get(str(value.attempt_id))["status"] == "succeeded")
        with supervisor.lock:
            supervisor.db.execute("UPDATE attempts SET finished=? WHERE id=?", (time.time() - 86401, str(value.attempt_id)))
            supervisor._prune_results(time.time())
        replay = supervisor.submit(value)
        assert replay["status"] == "failed" and replay["error"] == "result_expired"
        assert replay["result"] is None and supervisor.active is None
        assert supervisor.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1
    finally:
        supervisor.close()


def test_expired_attempt_and_busy_worker_are_rejected(tmp_path):
    supervisor = WorkerSupervisor(tmp_path / "attempts.sqlite3", child_target=hanging_child)
    supervisor.start()
    try:
        until(lambda: supervisor.health()["ready"])
        with pytest.raises(HTTPException) as error:
            supervisor.submit(request(deadline_at=time.time() - 1))
        assert error.value.status_code == 422
        first = request(deadline_at=time.time() + 10)
        supervisor.submit(first)
        with pytest.raises(HTTPException) as error:
            supervisor.submit(request(deadline_at=time.time() + 10))
        assert error.value.status_code == 409
        assert supervisor.cancel(str(first.attempt_id))["status"] == "cancelled"
        assert supervisor.process is None
    finally:
        supervisor.close()


def test_worker_accepts_large_target_original_without_pixel_or_byte_ceiling():
    source = io.BytesIO()
    Image.new('RGB', (4097, 4096), 'red').save(source, format='BMP')
    data = source.getvalue()
    assert len(data) > 20 * 1024 * 1024
    value = request(image_base64=base64.b64encode(data).decode('ascii'), sha256=hashlib.sha256(data).hexdigest())
    image, digest = decode_image(value.image_base64)
    assert image.size == (4097, 4096)
    assert digest == value.sha256
