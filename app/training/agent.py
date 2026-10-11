"""User-owned outbound training supervisor: python -m app.training.agent --help."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from importlib.metadata import version, PackageNotFoundError, distributions
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

import httpx
from .adapters import LOCK, verify_runtime
from .catalog import catalog
from .data import encoded
from .protocol import HEARTBEAT_SECONDS, LEASE_SECONDS


def write_private(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".next")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(encoded(value))
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def server_url(value, *, client=None):
    parsed = urlparse(value)
    local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if (
        parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or not parsed.hostname
    ):
        raise ValueError("Provide the application's origin, without credentials or a path")
    if parsed.scheme != "https" and not (local and parsed.scheme == "http"):
        raise ValueError("Agents require HTTPS; HTTP is allowed only on loopback")
    origin = value.rstrip("/")
    if client is None:
        with httpx.Client(timeout=30, follow_redirects=False) as connection:
            return server_url(value, client=connection)
    response = client.get(origin + "/api/config", follow_redirects=False)
    # Local releases have no config route. Never infer local mode from a
    # successful hosted response, nor from a failed remote discovery request.
    if response.status_code == 404 and local:
        return origin + "/api/training-agent"
    response.raise_for_status()
    config = response.json()
    mode = config.get("mode") if isinstance(config, dict) else None
    if mode == "hosted":
        return origin + "/api/v1/training-agent"
    if mode == "local" and local:
        return origin + "/api/training-agent"
    raise ValueError("The application must report hosted mode, or local mode on loopback")


def capabilities(weights):
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "aarch64"}:
        raise ValueError("Use Linux NVIDIA x86-64/ARM64, or Windows through WSL2")
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"], text=True
    )
    devices = []
    for line in output.splitlines():
        index, name, uuid = [part.strip() for part in line.split(",", 2)]
        devices.append(
            {
                "slot": index,
                "name": name,
                "uuid": uuid,
                "identity": hashlib.sha256(uuid.encode()).hexdigest(),
                "enabled": True,
            }
        )
    models = []
    for model in catalog():
        try:
            verify_runtime(model["id"], weights)
            models.append(model["id"])
        except (FileNotFoundError, ValueError, PackageNotFoundError):
            pass
    if not devices or not models:
        raise ValueError(
            "Install the optional training runtime and provision at least one catalog model first"
        )
    runtime = {
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "adapter_sha256": hashlib.sha256(
            Path(__file__).with_name("adapters.py").read_bytes()
        ).hexdigest(),
        "catalog_sha256": hashlib.sha256(
            Path(__file__).with_name("runtime-lock.json").read_bytes()
        ).hexdigest(),
    }
    runtime["engine_sha256"] = hashlib.sha256(
        encoded(
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).parent.glob("*.py"))
            }
        )
    ).hexdigest()
    runtime["environment_sha256"] = hashlib.sha256(
        encoded(sorted((d.metadata["Name"], d.version) for d in distributions()))
    ).hexdigest()
    for package in ("torch", "torchvision", "ultralytics", "rfdetr", "pytorch-lightning", "numpy"):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            pass
    return devices, models, runtime


def provision(weights, model_ids):
    weights.mkdir(parents=True, exist_ok=True, mode=0o700)
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for model_id in model_ids:
            item = LOCK["weights"].get(model_id)
            if not item:
                raise ValueError("Unknown catalog model")
            path = weights / item["file"]
            temporary = path.with_suffix(".download")
            checksum = hashlib.new(item["algorithm"])
            try:
                with client.stream("GET", item["url"]) as response, temporary.open("wb") as target:
                    response.raise_for_status()
                    for chunk in response.iter_bytes(1024**2):
                        checksum.update(chunk)
                        target.write(chunk)
                if checksum.hexdigest() != item["checksum"]:
                    raise ValueError("Pretrained weight checksum differs from the pinned release")
                os.replace(temporary, path)
                print(f"Provisioned {model_id}", flush=True)
            finally:
                temporary.unlink(missing_ok=True)


from .gpu import reserve_gpu


class Job:
    def __init__(self, client, assignment, directory, weights, device, stop):
        self.client, self.assignment = client, assignment
        self.directory, self.weights, self.device, self.stop = directory, weights, device, stop
        self.work = assignment["work"]
        self.base = {
            "project_id": assignment["project_id"],
            "kind": assignment["kind"],
            "work_id": self.work["id"],
            "attempt_id": self.work["attempt_id"],
        }
        self.route = f"/projects/{self.base['project_id']}/{self.base['kind']}/{self.base['work_id']}/{self.base['attempt_id']}"
        self.sequence = 0
        self.status = "running"
        self.control = None
        self.deadline = time.monotonic() + LEASE_SECONDS - 20
        self.lock = threading.RLock()
        self.done = threading.Event()
        self.abandoned = threading.Event()
        self.child = None

    def update(self, **event):
        with self.lock:
            if self.done.is_set():
                return
            status = event.get("status", self.status)
            payload = {**self.base, "sequence": self.sequence, "status": status, **event}
            while True:
                try:
                    response = self.client.post("/update", json=payload)
                    response.raise_for_status()
                    break
                except httpx.TransportError:
                    self.check()
                    self.stop.wait(1)

            result = response.json()
            self.sequence += 1
            self.status = status
            self.control = result.get("control")
            self.deadline = time.monotonic() + result["lease_seconds"] - 20
            write_private(
                self.directory / "supervisor.json", {"pid": os.getpid(), "deadline": self.deadline}
            )
            if self.control == "pause":
                (self.directory / "pause").touch()
            if status in {"completed", "paused", "failed", "cancelled"}:
                self.done.set()

    def terminate(self):
        child = self.child
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()

    def heartbeat(self):
        while not self.done.wait(HEARTBEAT_SECONDS):
            try:
                self.update()
            except httpx.HTTPStatusError as error:
                if error.response.status_code in {401, 403, 404, 409}:
                    self.abandoned.set()
            except (httpx.HTTPError, RuntimeError):
                if time.monotonic() >= self.deadline:
                    self.abandoned.set()
            if self.abandoned.is_set() or time.monotonic() >= self.deadline or self.stop.is_set():
                self.abandoned.set()
                self.terminate()
                return
            if self.control == "cancel":
                self.terminate()

    def check(self):
        if self.abandoned.is_set() or self.stop.is_set() or time.monotonic() >= self.deadline:
            raise RuntimeError(
                "Assignment disconnected or expired; resume explicitly after reconnecting"
            )
        if self.control == "cancel":
            raise InterruptedError("Execution cancelled")

    def download(self, route, path, expected):
        if path.exists():
            with path.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() == expected:
                    return
        temporary = path.with_suffix(".download")
        checksum = hashlib.sha256()
        try:
            with self.client.stream("GET", route) as response, temporary.open("wb") as output:
                response.raise_for_status()
                for chunk in response.iter_bytes(1024**2):
                    self.check()
                    checksum.update(chunk)
                    output.write(chunk)
            if checksum.hexdigest() != expected:
                raise ValueError("Downloaded resource differs from its immutable manifest")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def upload(self, path):
        with path.open("rb") as source:
            sha = hashlib.file_digest(source, "sha256").hexdigest()
        response = self.client.post(
            "/artifacts",
            json={key: value for key, value in self.base.items() if key != "kind"}
            | {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha},
        )
        response.raise_for_status()
        artifact = response.json()
        if artifact["status"] == "ready":
            return
        route = f"/projects/{self.base['project_id']}/artifacts/{artifact['id']}"
        offset = artifact["offset"]
        with path.open("rb") as source:
            source.seek(offset)
            while chunk := source.read(8 * 1024**2):
                self.check()
                for retry in range(4):
                    try:
                        response = self.client.put(
                            route,
                            params={"offset": offset},
                            content=chunk,
                            headers={"Content-Type": "application/octet-stream"},
                        )
                        response.raise_for_status()
                        offset = response.json()["offset"]
                        break
                    except httpx.TransportError:
                        if retry == 3:
                            raise
                        self.stop.wait(2)
                source.seek(offset)
        response = self.client.post(route + "/complete")
        response.raise_for_status()

    def run(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.directory / "pause").unlink(missing_ok=True)
        write_private(self.directory / "assignment.json", self.assignment)
        thread = threading.Thread(target=self.heartbeat, daemon=True)
        thread.start()
        try:
            self.update()
            # A completed fit survives an upload/network failure. Synchronize its
            # saved outputs on explicit resume, without repeating optimization.
            result_path = self.directory / "result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            if not result or result["status"] == "paused":
                originals = self.directory / "originals"
                originals.mkdir(exist_ok=True)
                for image in self.assignment["dataset"]["images"]:
                    self.download(
                        self.route + f"/images/{image['id']}",
                        originals / image["sha256"],
                        image["sha256"],
                    )
                if self.base["kind"] == "operation":
                    name = "model.pt" if self.work["model"]["family"] == "yolo26" else "model.pth"
                    self.download(
                        self.route + "/model", self.directory / name, self.work["checkpoint_sha256"]
                    )
                result_path.unlink(missing_ok=True)
                (self.directory / "events.jsonl").write_text("")
                with reserve_gpu(self.device), (self.directory / "runtime.log").open("ab") as log:
                    self.check()
                    environment = {
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": self.device["uuid"],
                        "PYTHONUNBUFFERED": "1",
                        "YOLO_CONFIG_DIR": str(self.directory / "config"),
                    }
                    self.child = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "app.training.runner",
                            str(self.directory),
                            str(self.weights),
                        ],
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    with (self.directory / "events.jsonl").open() as events:
                        while self.child.poll() is None:
                            self.check()
                            position = events.tell()
                            line = events.readline()
                            if line and line.endswith("\n"):
                                self.update(**json.loads(line))
                            else:
                                events.seek(position)
                                self.stop.wait(0.25)
                        self.check()
                        for line in events:
                            if line.endswith("\n"):
                                self.update(**json.loads(line))
                    if self.child.returncode or not result_path.exists():
                        raise RuntimeError(
                            "Model process failed. Inspect runtime.log on the connected computer"
                        )
                result = json.loads(result_path.read_text())
            self.check()
            if result["status"] == "syncing":
                self.update(status="syncing", checkpoint=True)
                for name in (result["model_file"], "manifest.json"):
                    self.upload(self.directory / name)
                self.update(status="completed", checkpoint=True, progress=1.0)
            else:
                self.update(**result)
        except Exception as error:
            self.terminate()
            if not self.abandoned.is_set():
                try:
                    self.update(
                        status="cancelled" if self.control == "cancel" else "failed",
                        error=str(error)[:500],
                    )
                except httpx.HTTPError:
                    pass
            print(f"Assignment {self.work['id']}: {type(error).__name__}", flush=True)
        finally:
            self.done.set()
            self.terminate()
            thread.join(timeout=12)


def serve(config, state, weights):
    devices, models, runtime = capabilities(weights)
    if runtime != config["runtime"] or models != config["models"]:
        raise ValueError(
            "Runtime changed since enrollment; restore it to resume existing runs, or enroll again"
        )
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())

    def worker(device):
        with httpx.Client(
            base_url=config["url"] + "/",
            headers={"Authorization": "Bearer " + config["token"]},
            timeout=30,
        ) as client:
            while not stop.is_set():
                try:
                    response = client.post("claim", json={"slot": device["slot"]})
                    response.raise_for_status()
                    assignment = response.json()
                    if assignment:
                        key = hashlib.sha256(
                            (assignment["project_id"] + ":" + assignment["work"]["id"]).encode()
                        ).hexdigest()
                        Job(client, assignment, state / "jobs" / key, weights, device, stop).run()
                        continue
                except httpx.HTTPStatusError as error:
                    if error.response.status_code in {401, 403, 404}:
                        stop.set()
                    print(f"Agent request: HTTP {error.response.status_code}", flush=True)
                except httpx.HTTPError:
                    pass
                stop.wait(5)

    threads = [
        threading.Thread(target=worker, args=(device,)) for device in devices if device["enabled"]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state", type=Path, default=Path.home() / ".local/share/annotation-training"
    )
    parser.add_argument("--weights", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser(
        "provision", help="Explicitly download checksum-pinned pretrained weights"
    )
    setup.add_argument("models", nargs="+", choices=list(LOCK["weights"]))
    enroll = commands.add_parser("enroll")
    enroll.add_argument("--server", required=True)
    enroll.add_argument("--name", default=platform.node())
    commands.add_parser("run")
    args = parser.parse_args()
    state = args.state.expanduser().resolve()
    weights = (args.weights or state / "weights").expanduser().resolve()
    if args.command == "provision":
        provision(weights, args.models)
        return
    config_path = state / "agent.json"
    if args.command == "enroll":
        import getpass

        url = server_url(args.server)
        devices, models, runtime = capabilities(weights)
        code = getpass.getpass("One-time pairing code: ")
        previous = json.loads(config_path.read_text()) if config_path.exists() else {}
        previous_token = previous.get("token") if previous.get("url") == url else None
        with httpx.Client(timeout=30) as client:
            response = client.post(
                url + "/enroll",
                json={
                    "code": code,
                    "name": args.name,
                    "protocol": 1,
                    "previous_token": previous_token,
                    "models": models,
                    "runtime": runtime,
                    "devices": [{k: v for k, v in d.items() if k != "uuid"} for d in devices],
                },
            )
            response.raise_for_status()
        write_private(
            config_path, {**response.json(), "url": url, "runtime": runtime, "models": models}
        )
        print("Computer connected. Start the agent with the run command.")
    else:
        with (state / "supervisor.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            serve(json.loads(config_path.read_text()), state, weights)


if __name__ == "__main__":
    main()
