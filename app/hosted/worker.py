"""Tailscale-only SAM worker with a warm, killable model subprocess.

Run one uvicorn process per GPU. Every request (including health) requires the
private service token. A SQLite attempt ledger fences retries across restarts;
it contains results and identifiers, never source images or prompts.
"""
from __future__ import annotations

import base64
from contextlib import asynccontextmanager
import hashlib
import hmac
import io
import ipaddress
import json
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import threading
import time
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .job_protocol import AttemptRequest, ResultTooLarge, execute_inference


def _model_process(connection, expected_parent=None):
    """Only this process imports model dependencies or allocates CUDA memory."""
    try:
        # Linux GPU hosts: a supervisor SIGKILL must not leave orphaned CUDA
        # work alive while a restarted worker/dispatcher schedules a fallback.
        if os.name == "posix" and __import__("sys").platform == "linux":
            import ctypes
            import signal
            parent = os.getppid()
            if ((expected_parent is not None and parent != expected_parent)
                    or ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0 or os.getppid() != parent):
                return
        from PIL import Image
        from app.inference import Sam3Engine
        from app.translation import PromptTranslator

        engine = Sam3Engine(device=os.environ.get("SAM3_DEVICE", "cuda"))
        translator = PromptTranslator(cache_dir=os.environ.get("HF_HUB_CACHE"))
        engine.load()
        # A successful recovery means inference works, not merely CUDA import.
        buffer = io.BytesIO()
        image = Image.new("RGB", (96, 64), "white")
        image.paste("red", (24, 16, 72, 48))
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        common = dict(job_id=uuid4(), attempt_id=uuid4(), project_id=uuid4(), image_id=1,
                      sha256=hashlib.sha256(buffer.getvalue()).hexdigest(), deadline_at=time.time() + 120,
                      image_base64=encoded)
        probes = [
            ("points", {"part": {"points": [{"x": 48, "y": 32, "label": 1}]}}),
            ("text", {"category_id": 1, "text": "objeto rojo", "source_language": "es"}),
            ("visual", {"category_id": 1, "reference_image": encoded, "reference_box": [24, 16, 72, 48]}),
        ]
        for kind, payload in probes:
            execute_inference(engine, translator, AttemptRequest(kind=kind, payload=payload, **common))
        connection.send({"type": "ready"})
        while True:
            raw = connection.recv()
            if raw is None:
                return
            request = AttemptRequest.model_validate(raw)
            try:
                result = execute_inference(engine, translator, request)
                connection.send({"type": "result", "attempt_id": str(request.attempt_id),
                                 "status": "succeeded", "result": result, "error": None})
            except ResultTooLarge:
                connection.send({"type": "result", "attempt_id": str(request.attempt_id),
                                 "status": "failed", "result": None, "error": "result_too_large"})
            except ValueError:
                connection.send({"type": "result", "attempt_id": str(request.attempt_id),
                                 "status": "failed", "result": None, "error": "invalid_input"})
            except Exception:
                # Model exceptions may contain HF credentials, filenames, URLs.
                connection.send({"type": "result", "attempt_id": str(request.attempt_id),
                                 "status": "failed", "result": None, "error": "inference_failed"})
                return  # Prove readiness again before accepting more work.
    except (EOFError, BrokenPipeError):
        return
    except Exception:
        try:
            connection.send({"type": "unavailable"})
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


class WorkerSupervisor:
    def __init__(self, ledger: Path, *, max_runtime: float = 120, load_timeout: float = 900,
                 child_target=_model_process):
        ledger.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(ledger, check_same_thread=False)
        os.chmod(ledger, 0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS attempts (
            id TEXT PRIMARY KEY, job_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
            deadline REAL NOT NULL, status TEXT NOT NULL, result TEXT, error TEXT, finished REAL)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS results_expiry ON attempts(finished) WHERE result IS NOT NULL")
        self.db.execute("UPDATE attempts SET status='failed', error='worker_restarted', finished=? WHERE status='running'",
                        (time.time(),))
        self.db.commit()
        self.max_runtime, self.load_timeout = max_runtime, load_timeout
        self.child_target = child_target
        self.lock = threading.RLock()
        self.ready = False
        self.active: str | None = None
        self.active_expires_monotonic: float | None = None
        self.process = self.connection = None
        self.started = 0.0
        self.retry_at = 0.0
        self.stop_event = threading.Event()
        self.thread = None
        self.sender = None
        self.next_cleanup_at = 0.0

    def _prune_results(self, now):
        # Retain only the small immutable replay fence. Source images/prompts
        # are never stored here; result payloads expire even on an idle worker.
        self.db.execute("UPDATE attempts SET result=NULL,error='result_expired' WHERE finished < ? AND result IS NOT NULL",
                        (now - 86400,))
        self.db.commit()
        self.next_cleanup_at = now + 60

    def start(self):
        self.thread = threading.Thread(target=self._monitor, name="sam-supervisor", daemon=True)
        self.thread.start()

    def _spawn(self):
        if self.sender is not None and self.sender.is_alive():
            return  # Do not reuse pipe descriptors while an old writer exits.
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        args = (child, os.getpid()) if self.child_target is _model_process else (child,)
        self.process = context.Process(target=self.child_target, args=args, daemon=True)
        self.process.start()
        child.close()
        self.connection = parent
        self.started = time.monotonic()
        self.ready = False

    def _kill(self):
        self.ready = False
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            if self.process.is_alive():
                # Keep the GPU unavailable until OS termination is confirmed.
                return False
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        return True

    def _finish(self, attempt_id, status, result=None, error=None):
        self.db.execute("UPDATE attempts SET status=?,result=?,error=?,finished=? WHERE id=? AND status='running'",
                        (status, json.dumps(result) if result is not None else None, error, time.time(), attempt_id))
        self.db.commit()
        if self.active == attempt_id:
            self.active = None
            self.active_expires_monotonic = None

    def _tick(self):
        now = time.time()
        if now >= self.next_cleanup_at:
            self._prune_results(now)
        if self.process is None:
            if now >= self.retry_at:
                self._spawn()
            return
        if self.active:
            row = self.db.execute("SELECT deadline FROM attempts WHERE id=?", (self.active,)).fetchone()
            if now >= row["deadline"] or time.monotonic() >= self.active_expires_monotonic:
                attempt_id = self.active
                if self._kill():
                    self._finish(attempt_id, "timed_out", error="execution_timeout")
                    self.retry_at = now + 5
                return
        if not self.ready and not self.active and time.monotonic() - self.started > self.load_timeout:
            self._kill()
            self.retry_at = now + 30
            return
        if self.connection is not None and self.connection.poll():
            try:
                message = self.connection.recv()
            except (EOFError, OSError):
                message = {"type": "unavailable"}
            if message["type"] == "ready":
                self.ready = True
            elif message["type"] == "result":
                if message.get("attempt_id") == self.active:
                    self._finish(self.active, message["status"], message.get("result"), message.get("error"))
                    if message.get("error") == "inference_failed":
                        self._kill()
                        self.retry_at = now + 5
            else:
                self.ready = False
        if self.process is not None and not self.process.is_alive():
            if self.active:
                self._finish(self.active, "failed", error="worker_process_exited")
            self._kill()
            self.retry_at = now + 5

    def _monitor(self):
        while not self.stop_event.is_set():
            with self.lock:
                try:
                    self._tick()
                except Exception:
                    # Keep fencing in place even after a pipe/process failure.
                    if self._kill() and self.active:
                        self._finish(self.active, "failed", error="worker_supervisor_failed")
                    self.retry_at = time.time() + 5
            self.stop_event.wait(0.05)

    def health(self):
        with self.lock:
            return {"ready": bool(self.ready and self.process and self.process.is_alive()), "busy": self.active is not None}

    def get(self, attempt_id: str):
        with self.lock:
            row = self.db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if row is None:
                raise HTTPException(404, "Attempt not found.")
            return {"attempt_id": row["id"], "job_id": row["job_id"],
                    "status": "failed" if row["error"] == "result_expired" else row["status"],
                    "result": json.loads(row["result"]) if row["result"] else None, "error": row["error"]}

    def submit(self, request: AttemptRequest):
        with self.lock:
            attempt_id = str(request.attempt_id)
            raw = request.model_dump(mode="json")
            fingerprint = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            previous = self.db.execute("SELECT fingerprint FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if previous:
                if not hmac.compare_digest(previous["fingerprint"], fingerprint):
                    raise HTTPException(409, "Attempt identity was already used for another request.")
                return self.get(attempt_id)
            now = time.time()
            if request.deadline_at <= now or request.deadline_at > now + self.max_runtime + 2:
                raise HTTPException(422, "Attempt deadline is expired or exceeds the execution limit.")
            if self.active is not None:
                raise HTTPException(409, "Worker is busy.")
            if not self.health()["ready"]:
                raise HTTPException(503, "Worker is warming up.")
            # Persist before touching the subprocess; restart can never repeat it.
            deadline = min(request.deadline_at, now + self.max_runtime)
            self.db.execute("INSERT INTO attempts(id,job_id,fingerprint,deadline,status) VALUES (?,?,?,?,'running')",
                            (attempt_id, str(request.job_id), fingerprint, deadline))
            self.db.commit()
            self.active = attempt_id
            self.active_expires_monotonic = time.monotonic() + max(0, deadline - now)
            # A pipe write can block if the child stops consuming bytes. The
            # watchdog must remain free to acquire the lock and kill the child.
            self.sender = threading.Thread(target=self._send, args=(self.connection, raw, attempt_id), daemon=True)
            self.sender.start()
            return self.get(attempt_id)

    def _send(self, connection, raw, attempt_id):
        try:
            connection.send(raw)
        except (EOFError, OSError, TypeError):
            with self.lock:
                if not self.stop_event.is_set() and self.active == attempt_id and self._kill():
                    self._finish(attempt_id, "failed", error="worker_process_exited")

    def cancel(self, attempt_id: str):
        with self.lock:
            current = self.get(attempt_id)
            if self.active == attempt_id:
                if not self._kill():
                    raise HTTPException(503, "Waiting for process termination.")
                self._finish(attempt_id, "cancelled", error="cancelled")
                self.retry_at = time.time() + 5
                return self.get(attempt_id)
            return current

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        with self.lock:
            terminated = self._kill()
            if self.active and terminated:
                self._finish(self.active, "failed", error="worker_shutdown")
            self.db.close()


class PrivateWorkerMiddleware:
    def __init__(self, app, token, allowed_clients):
        self.app, self.token = app, token
        self.allowed_clients = {ipaddress.ip_address(value) for value in allowed_clients}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        try:
            client = ipaddress.ip_address(scope.get("client", ("", 0))[0])
        except ValueError:
            client = None
        if client not in self.allowed_clients:
            return await JSONResponse({"detail": "Client not permitted."}, 403)(scope, receive, send)
        headers = dict(scope.get("headers", []))
        expected = f"Bearer {self.token}".encode()
        if not hmac.compare_digest(headers.get(b"authorization", b""), expected):
            return await JSONResponse({"detail": "Unauthorized."}, 401)(scope, receive, send)
        return await self.app(scope, receive, send)


def create_worker_app(*, token=None, supervisor=None, allowed_clients=None):
    token = token or os.environ.get("ANNOTATION_WORKER_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("Set ANNOTATION_WORKER_TOKEN to at least 32 random characters.")
    if allowed_clients is None:
        configured = os.environ.get("ANNOTATION_WORKER_ALLOWED_CLIENTS", "")
        allowed_clients = [value.strip() for value in configured.split(",") if value.strip()]
    allowed_clients = [*allowed_clients, "127.0.0.1", "::1"]
    supervisor = supervisor or WorkerSupervisor(
        Path(os.environ.get("ANNOTATION_WORKER_DATA_DIR", "/var/lib/annotation-worker")) / "attempts.sqlite3")

    @asynccontextmanager
    async def lifespan(app):
        supervisor.start()
        yield
        supervisor.close()

    app = FastAPI(title="Private annotation worker", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(PrivateWorkerMiddleware, token=token, allowed_clients=allowed_clients)

    @app.get("/v1/health")
    def health():
        return supervisor.health()

    @app.post("/v1/attempts", status_code=202)
    def submit(body: AttemptRequest):
        return supervisor.submit(body)

    @app.get("/v1/attempts/{attempt_id}")
    def status(attempt_id: UUID):
        return supervisor.get(str(attempt_id))

    @app.delete("/v1/attempts/{attempt_id}")
    def cancel(attempt_id: UUID):
        return supervisor.cancel(str(attempt_id))

    return app


def main():
    """Host-network launcher: exact Tailscale bind plus loopback, no proxies.

    Docker's userland proxy can hide the real peer behind the bridge gateway.
    Binding these sockets directly preserves the source address for the worker's
    allowlist without exposing a listener on the host's public interfaces.
    """
    import socket
    import uvicorn

    bind = os.environ.get("ANNOTATION_WORKER_BIND", "127.0.0.1")
    address = ipaddress.ip_address(bind)
    tailnet = ipaddress.ip_network("100.64.0.0/10") if address.version == 4 else ipaddress.ip_network("fd7a:115c:a1e0::/48")
    if not address.is_loopback and address not in tailnet:
        raise RuntimeError("Worker bind must be a Tailscale or loopback address.")
    port = int(os.environ.get("ANNOTATION_WORKER_PORT", "8766"))
    sockets = []
    try:
        for host in dict.fromkeys(["127.0.0.1", bind]):
            listener = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
            listener.listen(128)
            listener.set_inheritable(True)
            sockets.append(listener)
        server = uvicorn.Server(uvicorn.Config(create_worker_app(), access_log=False, proxy_headers=False))
        server.run(sockets=sockets)
    finally:
        for listener in sockets:
            listener.close()


if __name__ == "__main__":
    main()
