"""Supervisor failure modes, without importing a model or requiring CUDA."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time

import httpx
import pytest

from app.training.agent import Job, write_private
from app.training.gpu import reserve_gpu


def job(tmp_path, handler):
    assignment = {
        "project_id": "project",
        "kind": "run",
        "dataset": {"images": []},
        "work": {"id": "run", "attempt_id": "attempt"},
    }
    client = httpx.Client(
        base_url="https://agent.example/api/training-agent/", transport=httpx.MockTransport(handler)
    )
    return Job(
        client,
        assignment,
        tmp_path,
        tmp_path / "weights",
        {"uuid": "disposable"},
        threading.Event(),
    )


def test_finished_training_upload_recovery_does_not_launch_another_model(tmp_path, monkeypatch):
    updates, uploads = [], []

    def handler(request):
        assert request.url.path == "/api/training-agent/update"
        updates.append(json.loads(request.content))
        return httpx.Response(200, json={"lease_seconds": 90, "control": None})

    execution = job(tmp_path, handler)
    write_private(tmp_path / "result.json", {"status": "syncing", "model_file": "model.pt"})
    monkeypatch.setattr(execution, "upload", lambda path: uploads.append(path.name))
    monkeypatch.setattr(
        subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("Must not retrain saved outputs")
    )
    execution.run()
    assert uploads == ["model.pt", "manifest.json"]
    assert [value["status"] for value in updates] == ["running", "syncing", "completed"]
    assert execution.done.is_set()


def test_download_hash_failure_preserves_previous_resource(tmp_path):
    execution = job(tmp_path, lambda _: httpx.Response(200, content=b"incorrect bytes"))
    path = tmp_path / "image"
    path.write_bytes(b"previous")
    with pytest.raises(ValueError, match="immutable manifest"):
        execution.download("/image", path, hashlib.sha256(b"expected").hexdigest())
    assert path.read_bytes() == b"previous"
    assert not path.with_suffix(".download").exists()


def test_disconnection_deadline_and_gpu_exclusion(tmp_path, monkeypatch):
    execution = job(tmp_path, lambda _: httpx.Response(200))
    execution.deadline = time.monotonic() - 1
    with pytest.raises(RuntimeError, match="expired"):
        execution.check()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    with reserve_gpu({"uuid": "one"}):
        with pytest.raises(ValueError, match="reserved"):
            with reserve_gpu({"uuid": "one"}):
                pytest.fail("A GPU cannot be reserved twice")
        with reserve_gpu({"uuid": "two"}):
            pass
    with reserve_gpu({"uuid": "one"}):
        pass


@pytest.mark.parametrize("state", ["expired", "missing", "wrong-parent"])
def test_runner_watchdog_terminates_its_process_group(tmp_path, state):
    if state != "missing":
        write_private(
            tmp_path / "supervisor.json",
            {
                "pid": os.getpid() if state == "expired" else -1,
                "deadline": time.monotonic() - 1 if state == "expired" else time.monotonic() + 60,
            },
        )
    code = "from pathlib import Path; import os,sys; from app.training.runner import watch_supervisor; watch_supervisor(Path(sys.argv[1]), os.getppid())"
    process = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], start_new_session=True)
    try:
        assert process.wait(timeout=10) == -signal.SIGTERM
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
