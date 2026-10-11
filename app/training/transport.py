"""Authenticated agent transport; every data read is scoped to a live attempt."""

import fcntl
import hashlib
import os
from pathlib import Path
import time

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import FileResponse, StreamingResponse
from . import coordinator
from .protocol import Enrollment, Claim, Update, ArtifactInput
from .service import require, identity, public

CHUNK_BYTES = 8 * 1024**2


def bearer(request):
    value = request.headers.get("authorization", "")
    if not value.startswith("Bearer ") or len(value) > 200:
        raise HTTPException(401, "Agent authentication required")
    return value[7:]


def root(backend):
    directory = (
        backend.settings.data_dir / "training-uploads"
        if backend.hosted
        else backend.repo.root / "artifacts"
    )
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return directory


def stream_file(source, filename=None):
    headers = {"Cache-Control": "private, no-store"}
    if filename:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    if isinstance(source, Path):
        return FileResponse(source, headers=headers, media_type="application/octet-stream")

    def chunks():
        with source:
            while data := source.read(1024 * 1024):
                yield data

    return StreamingResponse(chunks(), headers=headers, media_type="application/octet-stream")


def artifact_source(backend, artifact):
    if artifact["status"] != "ready":
        raise HTTPException(409, "Artifact is still synchronizing")
    return (
        backend.objects.open_stream(artifact["object_key"])
        if backend.hosted
        else root(backend) / artifact["id"]
    )


def install_agent(app, control, prefix, enabled=lambda: True):
    router = APIRouter()

    def authorize(request):
        if not enabled():
            raise HTTPException(404, "Training is not enabled")
        token = bearer(request)
        return token, control.node(token)

    @router.post("/enroll")
    def enroll(body: Enrollment):
        if not enabled():
            raise HTTPException(404, "Training is not enabled")
        return control.enroll(body)

    @router.post("/heartbeat")
    def heartbeat(request: Request):
        token, _ = authorize(request)
        return control.touch(token)

    @router.post("/claim")
    def claim(request: Request, body: Claim):
        token, node = authorize(request)
        control.touch(token)
        for project_id in control.projects(node):
            backend = control.backend(node, project_id)
            with backend.repo.transaction() as records, control.guard(records, token) as fresh:
                result = coordinator.claim(records, fresh, body.slot, project_id)
                if result:
                    return result
        return None

    @router.post("/update")
    def update(request: Request, body: Update):
        token, node = authorize(request)
        backend = control.backend(node, body.project_id)
        with backend.repo.transaction() as records, control.guard(records, token) as fresh:
            return coordinator.update(records, fresh, body)

    def assignment(request, project_id, kind, work_id, attempt_id):
        if kind not in {"run", "operation"}:
            raise HTTPException(404, "Assignment not found")
        token, node = authorize(request)
        backend = control.backend(node, project_id)
        with backend.repo.transaction() as records, control.guard(records, token) as fresh:
            work = require(records, kind, work_id)
            coordinator.check_attempt(work, fresh, attempt_id)
            dataset = work.get("input") or require(records, "dataset", work["dataset_id"])
        return backend, work, dataset

    @router.get("/projects/{project_id}/{kind}/{work_id}/{attempt_id}/images/{image_id}")
    def image(
        request: Request, project_id: str, kind: str, work_id: str, attempt_id: str, image_id: int
    ):
        backend, work, dataset = assignment(request, project_id, kind, work_id, attempt_id)
        im = next((im for im in dataset["images"] if im["id"] == image_id), None)
        op = work.get("operation", "train")
        if (
            not im
            or (op == "train" and im.get("split") == "test")
            or (op == "evaluate" and im.get("split") != "test")
        ):
            raise HTTPException(404, "Image is not assigned to this operation")
        return stream_file(backend.image(im))

    @router.get("/projects/{project_id}/{kind}/{work_id}/{attempt_id}/model")
    def model(request: Request, project_id: str, kind: str, work_id: str, attempt_id: str):
        backend, work, _ = assignment(request, project_id, kind, work_id, attempt_id)
        if kind != "operation":
            raise HTTPException(404, "No model is assigned")
        with backend.repo.transaction() as records:
            artifact = require(records, "artifact", work["artifact_id"])
        return stream_file(artifact_source(backend, artifact))

    @router.post("/artifacts")
    def artifact(request: Request, body: ArtifactInput):
        token, node = authorize(request)
        backend = control.backend(node, body.project_id)
        with backend.repo.transaction() as records, control.guard(records, token) as fresh:
            work = require(records, "run", body.work_id)
            coordinator.check_attempt(work, fresh, body.attempt_id)
            previous = next(
                (
                    a
                    for a in records.list("artifact")
                    if a["work_id"] == body.work_id
                    and a["sha256"] == body.sha256
                    and a["name"] == body.name
                ),
                None,
            )
            if previous:
                previous["attempt_id"] = body.attempt_id
                records.put("artifact", previous["id"], previous)
                return public(previous)
            value = {
                **body.model_dump(),
                "id": identity(),
                "status": "uploading",
                "offset": 0,
                "created_at": time.time(),
            }
            value["object_key"] = f"projects/{body.project_id}/training/{value['id']}"
            records.put("artifact", value["id"], value)
        return public(value)

    @router.put("/projects/{project_id}/artifacts/{artifact_id}")
    async def upload(
        request: Request, project_id: str, artifact_id: str, offset: int = Query(ge=0)
    ):
        token, node = authorize(request)
        backend = control.backend(node, project_id)
        # Reserve/authorize before consuming the chunk.
        with backend.repo.transaction() as records, control.guard(records, token) as fresh:
            value = require(records, "artifact", artifact_id)
            coordinator.check_attempt(
                require(records, "run", value["work_id"]), fresh, value["attempt_id"]
            )
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > CHUNK_BYTES:
                raise HTTPException(413, "Use chunks of at most 8 MiB")
        path = root(backend) / value["id"]
        with path.open("a+b") as output:
            fcntl.flock(output.fileno(), fcntl.LOCK_EX)
            with backend.repo.transaction() as records, control.guard(records, token) as fresh:
                current = require(records, "artifact", artifact_id)
                coordinator.check_attempt(
                    require(records, "run", current["work_id"]), fresh, current["attempt_id"]
                )
                if current["status"] == "ready":
                    return public(current)
                size = output.seek(0, 2)
                if offset < size:
                    output.seek(offset)
                    if output.read(len(data)) != data:
                        raise HTTPException(409, "Replayed chunk differs")
                elif offset == size and size + len(data) <= current["size_bytes"]:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                else:
                    raise HTTPException(409, "Unexpected upload offset")
                current["offset"] = output.seek(0, 2)
                records.put("artifact", artifact_id, current)
        return public(current)

    @router.post("/projects/{project_id}/artifacts/{artifact_id}/complete")
    def complete(request: Request, project_id: str, artifact_id: str):
        token, node = authorize(request)
        backend = control.backend(node, project_id)
        with backend.repo.transaction() as records, control.guard(records, token) as fresh:
            value = require(records, "artifact", artifact_id)
            coordinator.check_attempt(
                require(records, "run", value["work_id"]), fresh, value["attempt_id"]
            )
            if value["status"] == "ready":
                return public(value)
        path = root(backend) / value["id"]
        with path.open("rb") as source:
            fcntl.flock(source.fileno(), fcntl.LOCK_EX)
            if (
                path.stat().st_size != value["size_bytes"]
                or hashlib.file_digest(source, "sha256").hexdigest() != value["sha256"]
            ):
                raise ValueError("Artifact is incomplete or its checksum differs")
            if backend.hosted:
                source.seek(0)
                backend.objects.put(value["object_key"], source)
            with backend.repo.transaction() as records, control.guard(records, token) as fresh:
                coordinator.check_attempt(
                    require(records, "run", value["work_id"]), fresh, value["attempt_id"]
                )
                value["status"] = "ready"
                records.put("artifact", artifact_id, value)
        if backend.hosted:
            path.unlink(missing_ok=True)
        return public(value)

    app.include_router(router, prefix=prefix)
