"""Durable, fair single-inference dispatcher. No GPU dependencies are imported.

Claims, attempt fencing and quota finalization share the database's global row
lock. Network calls run outside transactions; replaying them is safe because a
worker persists attempt identities before execution. Multiple dispatcher
processes can recover the same attempt without starting additional GPU work.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import json
import signal
import threading
from uuid import uuid4

import httpx
from sqlalchemy import func, select

from .job_protocol import AttemptRequest, MAX_RESULT_BYTES
from .models import Attempt, ImageObject, Job, Project, utcnow
from .storage import ObjectCache

LOG = logging.getLogger(__name__)
TERMINAL = {"succeeded", "failed", "cancelled", "timed_out"}
NON_RETRYABLE = {"invalid_input", "result_expired", "result_too_large"}
MAX_WORKER_RESPONSE_BYTES = MAX_RESULT_BYTES + 64 * 1024


@dataclass(frozen=True)
class Worker:
    name: str
    url: str
    token: str


class WorkerTransport:
    def __init__(self):
        self.client = httpx.Client(timeout=httpx.Timeout(10, connect=3, write=15), trust_env=False)

    def call(self, worker, method, path, payload=None):
        with self.client.stream(method, worker.url.rstrip("/") + path,
                                headers={"Authorization": f"Bearer {worker.token}"}, json=payload) as response:
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_bytes():
                if len(body) + len(chunk) > MAX_WORKER_RESPONSE_BYTES:
                    raise ValueError("Worker response exceeds the output limit")
                body.extend(chunk)
            value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("Invalid worker response")
        return value

    def health(self, worker):
        return self.call(worker, "GET", "/v1/health")

    def submit(self, worker, request):
        return self.call(worker, "POST", "/v1/attempts", request)

    def status(self, worker, attempt_id):
        return self.call(worker, "GET", f"/v1/attempts/{attempt_id}")

    def cancel(self, worker, attempt_id):
        return self.call(worker, "DELETE", f"/v1/attempts/{attempt_id}")

    def close(self):
        self.client.close()


class Dispatcher:
    def __init__(self, settings, db, objects, *, transport=None, clock=utcnow):
        self.settings, self.db, self.objects = settings, db, objects
        self.transport = transport or WorkerTransport()
        self.clock = clock
        self.cache = ObjectCache(settings.cache_dir, settings.cache_limit_bytes)
        self.workers = {}
        for name in ("primary", "fallback"):
            url, token = getattr(settings, f"{name}_worker_url"), getattr(settings, f"{name}_worker_token")
            if url and token:
                self.workers[name] = Worker(name, url, token)

    def _ready(self, worker):
        try:
            status = self.transport.health(worker)
            return status.get("ready") is True and status.get("busy") is False
        except Exception:
            return False

    def _preferred(self, *, except_name=None):
        return next((worker for name, worker in self.workers.items()
                     if name != except_name and self._ready(worker)), None)

    def _new_attempt(self, session, job, worker, now):
        job.status = "running"
        job.attempt_id = str(uuid4())
        job.attempt_count += 1
        job.worker_name = worker.name
        job.deadline_at = now + timedelta(seconds=self.settings.max_runtime_seconds)
        job.heartbeat_at = now
        session.add(Attempt(id=job.attempt_id, job_id=job.id, worker_name=worker.name,
                            status="dispatching", started_at=now, deadline_at=job.deadline_at))

    def _claim(self, worker):
        with self.db.session() as session:
            self.db.global_lock(session)
            if session.scalar(select(Job.id).where(Job.status == "running").limit(1)):
                return None
            queued = list(session.scalars(select(Job).where(Job.status == "queued")
                                           .order_by(Job.created_at, Job.id).limit(self.settings.max_pending_global)))
            if not queued:
                return None
            # Least recently served account first; then FIFO within that account.
            last_served = dict(session.execute(select(Job.user_id, func.max(Attempt.started_at))
                               .join(Attempt, Attempt.job_id == Job.id).group_by(Job.user_id)).all())
            job = min(queued, key=lambda row: (last_served.get(row.user_id) or datetime.min, row.created_at, row.id))
            self._new_attempt(session, job, worker, self.clock())
            return job

    def _payload(self, job):
        with self.db.session() as session:
            project = session.get(Project, job.project_id)
            image = session.scalar(select(ImageObject).where(ImageObject.project_id == job.project_id,
                                                            ImageObject.image_id == job.image_id,
                                                            ImageObject.status == "ready"))
            if project is None or project.deleted_at is not None or image is None:
                raise ValueError("Image is no longer available")
            object_key, digest = image.object_key, image.sha256
        data = self.cache.get(self.objects, object_key, digest)
        request = AttemptRequest(job_id=job.id, attempt_id=job.attempt_id, project_id=job.project_id,
                                 image_id=job.image_id, sha256=digest, kind=job.kind, payload=job.payload,
                                 deadline_at=job.deadline_at.replace(tzinfo=timezone.utc).timestamp(),
                                 image_base64=base64.b64encode(data).decode("ascii"))
        return request.model_dump(mode="json")

    def _accepted(self, job, response):
        if response.get("attempt_id") != job.attempt_id or response.get("job_id") != job.id:
            return False
        if response.get("status") not in TERMINAL | {"running"}:
            return False
        with self.db.session() as session:
            self.db.global_lock(session)
            current = session.get(Job, job.id)
            if current is None or current.status != "running" or current.attempt_id != job.attempt_id:
                return False
            current.heartbeat_at = self.clock()
            current.started_at = current.started_at or self.clock()
            attempt = session.get(Attempt, job.attempt_id)
            if attempt.status == "dispatching":
                attempt.status = "running"
        return True

    def _complete(self, job, response, *, safe_to_retry=False):
        """Fence all outcomes by both job and immutable attempt identifier."""
        from .jobs import finalize_quota
        fallback = None
        if safe_to_retry and job.attempt_count < 2 and not job.cancel_requested:
            fallback = self._preferred(except_name=job.worker_name)
        with self.db.session() as session:
            self.db.global_lock(session)
            current = session.get(Job, job.id)
            if current is None or current.status != "running" or current.attempt_id != job.attempt_id:
                return None
            now = self.clock()
            attempt = session.get(Attempt, job.attempt_id)
            attempt.status = response["status"]
            attempt.finished_at = now
            if current.cancel_requested:
                current.status = "cancelled"
                current.error = None
                finalize_quota(session, current, charged=current.started_at is not None)
            elif response["status"] == "succeeded":
                current.status = "succeeded"
                current.result = response.get("result")
                current.error = None
                finalize_quota(session, current, charged=True)
            elif fallback is not None:
                self._new_attempt(session, current, fallback, now)
                return current
            else:
                current.status = "failed"
                # Worker errors are enumerated, never exception strings/URLs.
                current.error = response["error"] if response.get("error") in NON_RETRYABLE else "inference_unavailable"
                finalize_quota(session, current, charged=False)
            current.finished_at = now
            return None

    def _handle(self, job, response):
        if not self._accepted(job, response):
            return
        if response["status"] in TERMINAL:
            retry = self._complete(job, response, safe_to_retry=response["status"] != "succeeded"
                                   and response.get("error") not in NON_RETRYABLE)
            if retry:
                self._submit(retry)

    def _submit(self, job):
        worker = self.workers.get(job.worker_name)
        if worker is None:
            return  # Changed config: lease must expire before another attempt.
        try:
            payload = self._payload(job)
        except ValueError:
            self._complete(job, {"status": "failed", "error": "invalid_input"})
            return
        except Exception:
            # No HTTP request was sent, so this failure is safe to finalize now.
            self._complete(job, {"status": "failed", "error": "storage_unavailable"})
            return
        # Recheck cancellation/fencing after fetching R2 and before sending bytes.
        with self.db.session() as session:
            current = session.get(Job, job.id)
            if current is None or current.status != "running" or current.attempt_id != job.attempt_id or current.cancel_requested:
                return
        try:
            response = self.transport.submit(worker, payload)
            self._handle(job, response)
        except Exception:
            # Timeout is ambiguous. Poll the same identity; never invent a retry
            # until confirmed termination or the hard execution lease expires.
            return

    def _poll(self, job):
        worker = self.workers.get(job.worker_name)
        now = self.clock()
        lease_expired = now >= job.deadline_at + timedelta(seconds=self.settings.worker_grace_seconds)
        try:
            if worker is None:
                raise RuntimeError("Worker no longer configured")
            response = self.transport.cancel(worker, job.attempt_id) if job.cancel_requested else self.transport.status(worker, job.attempt_id)
            if response.get("status") in TERMINAL and self._accepted(job, response):
                retry = self._complete(job, response, safe_to_retry=response["status"] != "succeeded"
                                       and response.get("error") not in NON_RETRYABLE)
                if retry:
                    self._submit(retry)
                return
            if response.get("status") == "running" and self._accepted(job, response):
                if not lease_expired:
                    return
                # A process explicitly still running is stronger evidence than
                # lease expiry. Require actual termination before a fallback.
                ended = self.transport.cancel(worker, job.attempt_id)
                if ended.get("status") not in TERMINAL or not self._accepted(job, ended):
                    return
                retry = self._complete(job, ended, safe_to_retry=True)
                if retry:
                    self._submit(retry)
                return
        except Exception:
            # A short interruption is tolerated; after 20s try a fenced kill.
            if worker and not lease_expired and not job.cancel_requested and now - job.heartbeat_at >= timedelta(seconds=20):
                try:
                    ended = self.transport.cancel(worker, job.attempt_id)
                    if ended.get("status") in TERMINAL and self._accepted(job, ended):
                        retry = self._complete(job, ended, safe_to_retry=True)
                        if retry:
                            self._submit(retry)
                        return
                except Exception:
                    pass
        if lease_expired:
            retry = self._complete(job, {"status": "timed_out", "error": "worker_unreachable"}, safe_to_retry=True)
            if retry:
                self._submit(retry)

    def tick(self):
        with self.db.session() as session:
            running = session.scalar(select(Job).where(Job.status == "running").order_by(Job.created_at).limit(1))
            queued = running is None and session.scalar(select(Job.id).where(Job.status == "queued").limit(1))
        if running:
            self._poll(running)
            return True
        if not queued:
            return True  # Idle queue checks stay fast without pinging GPU hosts.
        worker = self._preferred()
        if worker is None:
            return False  # Availability retries keep their slower cadence.
        job = self._claim(worker)
        if job:
            self._submit(job)
        return True

    def run(self, stop_event=None):
        stop_event = stop_event or threading.Event()
        try:
            while not stop_event.is_set():
                delay = self.settings.worker_poll_seconds
                try:
                    if self.tick():
                        delay = self.settings.dispatcher_poll_milliseconds / 1000
                except Exception:
                    LOG.error("Dispatcher tick failed; durable attempts will be recovered.")
                stop_event.wait(delay)
        finally:
            self.transport.close()


def main():
    from .config import Settings
    from .database import Database
    from .storage import R2Store

    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    settings.validate()
    db = Database(settings)
    db.create_schema()
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    Dispatcher(settings, db, R2Store(settings)).run(stop)


if __name__ == "__main__":
    main()
