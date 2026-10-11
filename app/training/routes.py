"""Shared project API mounted with mode-specific authorization and storage."""

import asyncio
import json

from fastapi import APIRouter, Request, Query
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from .data import DatasetInput, ReviewInput, Task
from .catalog import RunInput
from .service import TrainingService, require, public, summary
from .protocol import PredictionInput, DeviceToggle
from .coordinator import enqueue_operation
from .transport import artifact_source, stream_file


def project_router(resolve):
    router = APIRouter()

    @router.get("/status")
    def status(request: Request):
        return TrainingService(resolve(request)).status()

    @router.get("")
    def overview(request: Request, task: Task = "segmentation"):
        return TrainingService(resolve(request)).overview(task)

    @router.post("/reviews")
    def reviews(request: Request, body: ReviewInput):
        return TrainingService(resolve(request)).review(body)

    @router.post("/datasets/preview")
    def preview(request: Request, body: DatasetInput, summary_only: bool = False):
        value = TrainingService(resolve(request)).dataset(body, preview=True)
        return public(summary(value) if summary_only else value)

    @router.post("/datasets", status_code=201)
    def dataset(request: Request, body: DatasetInput, summary_only: bool = False):
        value = TrainingService(resolve(request)).dataset(body)
        return public(summary(value) if summary_only else value)

    @router.get("/datasets/{dataset_id}")
    def get_dataset(request: Request, dataset_id: str):
        with resolve(request).repo.transaction() as records:
            return public(require(records, "dataset", dataset_id))

    @router.delete("/datasets/{dataset_id}")
    def delete_dataset(request: Request, dataset_id: str):
        return TrainingService(resolve(request)).delete("dataset", dataset_id)

    @router.delete("/runs/{run_id}")
    def delete_run(request: Request, run_id: str):
        return TrainingService(resolve(request)).delete("run", run_id)

    @router.get("/predictions/{batch_id}")
    def get_predictions(request: Request, batch_id: str):
        with resolve(request).repo.transaction() as records:
            batch = require(records, "operation", batch_id)
            return public(batch)

    @router.post("/runs", status_code=202)
    def run(request: Request, body: RunInput):
        return TrainingService(resolve(request)).run(body)

    @router.get("/runs/{run_id}")
    def get_run(request: Request, run_id: str):
        with resolve(request).repo.transaction() as records:
            return require(records, "run", run_id)

    @router.get("/runs/{run_id}/log")
    def log(request: Request, run_id: str):
        with resolve(request).repo.transaction() as records:
            require(records, "run", run_id)
            return {
                "events": sorted(
                    [e for e in records.list("event") if e["work_id"] == run_id],
                    key=lambda e: e["cursor"],
                ),
                "attempts": [a for a in records.list("attempt") if a["work_id"] == run_id],
            }

    @router.post("/compute/pair")
    def pair(request: Request):
        backend = resolve(request)
        return backend.control.pair(backend)

    @router.post("/compute/{node_id}/device")
    def device(request: Request, node_id: str, body: DeviceToggle):
        backend = resolve(request)
        return backend.control.toggle(backend, node_id, body)

    @router.post("/compute/{node_id}/revoke")
    def revoke(request: Request, node_id: str):
        backend = resolve(request)
        return backend.control.revoke(backend, node_id)

    @router.get("/artifacts/{artifact_id}")
    def download(request: Request, artifact_id: str):
        backend = resolve(request)
        with backend.repo.transaction() as records:
            artifact = require(records, "artifact", artifact_id)
        return stream_file(artifact_source(backend, artifact), artifact["name"])

    @router.post("/runs/{run_id}/evaluate", status_code=202)
    def evaluate(request: Request, run_id: str):
        return enqueue_operation(TrainingService(resolve(request)), run_id, "evaluate")

    @router.post("/runs/{run_id}/predict", status_code=202)
    def predict(request: Request, run_id: str, body: PredictionInput):
        return enqueue_operation(TrainingService(resolve(request)), run_id, "predict", body)

    @router.post("/predictions/{batch_id}/apply")
    def apply_predictions(request: Request, batch_id: str):
        from .predictions import apply

        return apply(resolve(request), batch_id)

    @router.post("/runs/{run_id}/{action}")
    def action(request: Request, run_id: str, action: str):
        return TrainingService(resolve(request)).action(run_id, action)

    @router.get("/runs/{run_id}/events")
    async def events(request: Request, run_id: str, after: int = Query(default=0, ge=0)):
        def read():
            # Resolve ownership again on every iteration; a deleted project or
            # expired login must never keep a private stream alive.
            backend = resolve(request)
            with backend.repo.transaction() as records:
                return require(records, "run", run_id)

        await run_in_threadpool(read)

        async def stream():
            try:
                cursor = max(after, int(request.headers.get("last-event-id", "0")))
            except ValueError:
                cursor = after
            for _ in range(30):
                if await request.is_disconnected():
                    return
                current = await run_in_threadpool(read)
                sequence = current["events"]
                if sequence > cursor:
                    yield f"id: {sequence}\ndata: {json.dumps(current, allow_nan=False)}\n\n"
                    cursor = sequence
                else:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(2)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "private, no-store", "X-Accel-Buffering": "no"},
        )

    return router
