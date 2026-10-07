"""CPU-only public API; ownership is checked before every project/object operation."""
from contextlib import contextmanager
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from app.images import Image
from PIL import ImageOps
from sqlalchemy import func, select
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
import httpx

from app.geometry import decode_mask, fill_small_holes, mask_payload, rasterize_components, union_masks
from app.storage import ProjectStore, RevisionConflict, _relative_file
from . import auth, jobs, uploads
from .quota import mutate_project, recover_project, recover_project_locked
from .config import Settings
from .database import Database
from .middleware import HostedMiddleware
from .models import ImageObject, Job, Project, UploadReservation, User, identity, utcnow
from .storage import (ObjectCache, R2Store, create_project_store, project_store, register_image,
                      validate_image)


class ProjectInput(BaseModel):
    name: str = Field(min_length=1, max_length=160)


class UploadInput(BaseModel):
    file_name: str = Field(min_length=1, max_length=1024)
    size: int = Field(gt=0, strict=True)


class CategoryInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    color: str = Field(default="#8B8DE3", pattern=r"^#[0-9a-fA-F]{6}$")


class CategoryUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")


class GeometryInput(BaseModel):
    image_id: int = Field(strict=True)
    components: list[dict[str, Any]]


class UnionInput(BaseModel):
    image_id: int = Field(strict=True)
    masks: list[dict[str, Any]]


class FillInput(BaseModel):
    image_id: int = Field(strict=True)
    mask: dict[str, Any]
    max_area: int = Field(default=16, ge=1, le=16_000_000, strict=True)


class PointInput(BaseModel):
    image_id: int = Field(strict=True)
    revision: int = Field(ge=0, strict=True)
    part: dict[str, Any]


class TextInput(BaseModel):
    image_id: int = Field(strict=True)
    revision: int = Field(ge=0, strict=True)
    text: str = Field(min_length=1, max_length=300)
    category_id: int = Field(strict=True)
    source_language: Literal["en", "es"] = "en"


class VisualInput(TextInput):
    text: str = Field(default="", max_length=300)
    reference_image: str = Field(min_length=1, max_length=4 * ((10 * 1024 * 1024 + 2) // 3))
    reference_box: list[float] | None = Field(default=None, min_length=4, max_length=4)


def create_app(settings: Settings | None = None, *, database=None, objects=None):
    settings = settings or Settings.from_env()
    settings.validate()
    db = database or Database(settings)
    objects = objects or R2Store(settings)
    db.create_schema()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # Account omitted legacy metadata/thumbnail bytes before any new admission.
    with db.session() as session:
        existing_projects = list(session.scalars(select(Project.id).where(Project.deleted_at.is_(None))))
    for project_id in existing_projects:
        recover_project(settings, db, project_id, objects)
    cache = ObjectCache(settings.cache_dir, settings.cache_limit_bytes)
    app = FastAPI(title="Annotation", version="1.0.0", docs_url=None, redoc_url=None)
    app.state.settings, app.state.db, app.state.objects = settings, db, objects
    health_lock, health_cache = threading.Lock(), {"at": 0.0, "model": None}
    app.add_middleware(HostedMiddleware, db=db, settings=settings)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[urlsplit(settings.public_url).hostname])
    app.include_router(auth.router)

    @app.exception_handler(RevisionConflict)
    async def revision_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        # Storage's validation errors may mention relative image names, never server paths.
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(FileNotFoundError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Resource not found"}, status_code=404)

    @app.exception_handler(Exception)
    async def unavailable(request, exc):
        return JSONResponse({"detail": "Service temporarily unavailable; try again"}, status_code=503)

    @contextmanager
    def owned(request, project_id, *, write=False):
        user_id = auth.user_for_request(request)
        with db.session() as session:
            counter = db.global_lock(session)
            query = select(Project).where(Project.id == project_id, Project.user_id == user_id, Project.deleted_at.is_(None))
            project = session.execute(query.with_for_update()).scalar_one_or_none()
            if project is None:
                raise HTTPException(404, "Project not found")
            recover_project_locked(settings, session, counter, project, objects)
            yield session, project, project_store(settings, project.id)

    def dto(session, project, store):
        data = store.project()
        ready = set(session.scalars(select(ImageObject.image_id).where(
            ImageObject.project_id == project.id, ImageObject.status == "ready")))
        return {"id": project.id, "name": project.name, "categories": data["categories"],
                "images": [image for image in data["images"] if image["id"] in ready]}

    def image_record(session, project_id, image_id):
        image = session.execute(select(ImageObject).where(ImageObject.project_id == project_id,
            ImageObject.image_id == image_id, ImageObject.status == "ready")).scalar_one_or_none()
        if image is None:
            raise HTTPException(404, "Image not found")
        return image

    def validate_mask_dimensions(value, image, budget=None):
        """Reject forged dimensions before any native RLE decoder can allocate."""
        if budget is None:
            budget = [settings.max_mask_pixels_per_request]
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"mask", "seed_mask"} and item is not None:
                    if not isinstance(item, dict) or item.get("size") != [image.height, image.width]:
                        raise ValueError("The mask does not match the image dimensions")
                    budget[0] -= image.height * image.width
                    if budget[0] < 0:
                        raise HTTPException(413, "This request contains too many full-image masks for hosted processing")
                validate_mask_dimensions(item, image, budget)
        elif isinstance(value, list):
            for item in value:
                validate_mask_dimensions(item, image, budget)

    @app.get("/api/config")
    @app.get("/api/v1/config")
    def config():
        return {"mode": "hosted"}

    @app.get("/api/health")
    @app.get("/api/v1/health")
    def health():
        with health_lock:
            if health_cache["model"] is None or time.monotonic() - health_cache["at"] >= 5:
                ready = False
                for name in ("primary", "fallback"):
                    url, token = getattr(settings, f"{name}_worker_url"), getattr(settings, f"{name}_worker_token")
                    if not url:
                        continue
                    try:
                        with httpx.Client(timeout=2, trust_env=False) as client:
                            response = client.get(url.rstrip("/") + "/v1/health", headers={"Authorization": f"Bearer {token}"})
                            response.raise_for_status()
                            ready = response.json().get("ready") is True
                        if ready:
                            break
                    except Exception:
                        continue
                health_cache.update(at=time.monotonic(), model={"loaded": ready,
                    "state": "ready" if ready else "unavailable", "device": "remote",
                    "message": "Remote inference available" if ready else "Remote inference temporarily unavailable"})
            return {"status": "ok", "mode": "hosted", "model": health_cache["model"]}

    @app.post("/api/model/load")
    @app.post("/api/v1/model/load")
    def model_load():
        return health()["model"]

    @app.get("/api/v1/projects")
    def projects(request: Request):
        user_id = auth.user_for_request(request)
        with db.session() as session:
            counter = db.global_lock(session)
            projects = list(session.scalars(select(Project).where(Project.user_id == user_id, Project.deleted_at.is_(None))
                                          .order_by(Project.created_at.desc()).with_for_update()))
            for project in projects:
                recover_project_locked(settings, session, counter, project, objects)
            return {"projects": [{"id": p.id, "name": p.name, "created_at": p.created_at.isoformat() + "Z"} for p in projects]}

    @app.post("/api/v1/projects", status_code=201)
    def create_project(body: ProjectInput, request: Request):
        user_id = auth.user_for_request(request)
        name = body.name.strip()
        if not name:
            raise ValueError("Enter a project name")
        project_id = identity()
        mutate_project(settings, db, objects, project_id, user_id, lambda *_: None, create_name=name)
        return {"id": project_id, "name": name, "categories": [], "images": []}

    @app.get("/api/v1/projects/{project_id}")
    def get_project(project_id: str, request: Request):
        with owned(request, project_id) as (session, project, store):
            return dto(session, project, store)

    @app.delete("/api/v1/projects/{project_id}")
    def delete_project(project_id: str, request: Request):
        user_id = auth.user_for_request(request)
        with db.session() as session:
            counter = db.global_lock(session)
            project = session.execute(select(Project).where(Project.id == project_id,
                Project.user_id == user_id, Project.deleted_at.is_(None)).with_for_update()).scalar_one_or_none()
            if not project:
                raise HTTPException(404, "Project not found")
            recover_project_locked(settings, session, counter, project, objects)
            now, total = utcnow(), project.metadata_bytes
            project.deleted_at = now
            for image in session.scalars(select(ImageObject).where(ImageObject.project_id == project_id)):
                if image.status == "ready":
                    total += image.size_bytes + image.thumbnail_bytes
                image.status, image.deleted_at = "deleted", now
            session.get(User, user_id).storage_bytes -= total
            counter.storage_bytes -= total
            for job in session.scalars(select(Job).where(Job.project_id == project_id,
                    Job.status.in_(["queued", "running"])).with_for_update()):
                job.cancel_requested = True
                if job.status == "queued":
                    job.status, job.finished_at = "cancelled", now
                    jobs.finalize_quota(session, job, False)
        return {"ok": True}

    def commit_image(project_id, request, source, name, size, reservation_id):
        user_id = auth.user_for_request(request)
        if shutil.disk_usage(settings.data_dir).free < settings.min_free_disk_bytes:
            raise HTTPException(503, "Server storage is temporarily unavailable")
        name = uploads.image_name(name)
        width, height, content_type = validate_image(source)
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        source.seek(0)
        object_id = reservation_id
        key = f"projects/{project_id}/images/{object_id}/original"
        with db.session() as session:
            db.global_lock(session)
            project = session.execute(select(Project).where(Project.id == project_id, Project.user_id == user_id,
                Project.deleted_at.is_(None)).with_for_update()).scalar_one_or_none()
            if not project:
                raise HTTPException(404, "Project not found")
            if session.scalar(select(ImageObject.id).where(ImageObject.project_id == project_id, ImageObject.file_name == name)):
                raise HTTPException(409, "An image with this relative path already exists")
            image_id = (session.scalar(select(func.max(ImageObject.image_id)).where(ImageObject.project_id == project_id)) or 0) + 1
            record = ImageObject(id=object_id, project_id=project_id, image_id=image_id, file_name=name,
                object_key=key, sha256=digest, size_bytes=size, width=width, height=height,
                content_type=content_type, status="pending")
            session.add(record)
        try:
            objects.put(key, source, content_type)
            source.seek(0)
            with Image.open(source) as original:
                # Keep original pixel grid, consistent with local annotations and COCO.
                original.thumbnail((256, 192))
                thumb = original.convert("RGB")
                output = io.BytesIO()
                thumb.save(output, format="PNG")
            objects.put(key.rsplit("/", 1)[0] + "/thumbnail.png", output.getvalue(), "image/png")
            def register_staged(store, session, project):
                record = session.get(ImageObject, object_id)
                record.thumbnail_bytes = len(output.getvalue())
                register_image(store, record)
            mutate_project(settings, db, objects, project_id, user_id, register_staged,
                           image_object_id=object_id, upload_reservation_id=reservation_id)
        except Exception:
            # If the process dies, pending rows remain discoverable by the maintenance reconciler.
            try:
                with db.session() as session:
                    counter = db.global_lock(session)
                    project = session.execute(select(Project).where(Project.id == project_id).with_for_update()).scalar_one()
                    recover_project_locked(settings, session, counter, project, objects)
                    row = session.get(ImageObject, object_id)
                    if row and row.status == "pending":
                        # A failed COMMIT response is ambiguous: never delete an
                        # object whose metadata actually committed as ready.
                        objects.delete(key)
                        objects.delete(key.rsplit("/", 1)[0] + "/thumbnail.png")
                        session.delete(row)
            except Exception:
                pass
            raise
        with owned(request, project_id) as (session, project, store):
            return dto(session, project, store)

    @app.post("/api/v1/projects/{project_id}/images", status_code=201)
    def upload_image(project_id: str, request: Request, file: UploadFile = File(...), relative_path: str = Form("")):
        if not request.state.reservation_id:
            raise HTTPException(400, "Invalid project identifier")
        return commit_image(project_id, request, file.file, relative_path or file.filename,
                            file.size, request.state.reservation_id)

    @app.post("/api/v1/projects/{project_id}/uploads", status_code=201)
    def begin_upload(project_id: str, body: UploadInput, request: Request):
        return uploads.begin(settings, db, auth.user_for_request(request), project_id, body.file_name, body.size)

    @app.put("/api/v1/projects/{project_id}/uploads/{upload_id}")
    async def upload_chunk(project_id: str, upload_id: str, offset: int, request: Request):
        return await run_in_threadpool(uploads.append, settings, db, auth.user_for_request(request),
                                      project_id, upload_id, offset, await request.body())

    @app.delete("/api/v1/projects/{project_id}/uploads/{upload_id}")
    def abort_upload(project_id: str, upload_id: str, request: Request):
        return uploads.abort(settings, db, auth.user_for_request(request), project_id, upload_id)

    @app.post("/api/v1/projects/{project_id}/uploads/{upload_id}/complete", status_code=201)
    def complete_upload(project_id: str, upload_id: str, request: Request):
        user_id = auth.user_for_request(request)
        # Completion is idempotent even if the first response was lost.
        with owned(request, project_id) as (session, project, store):
            record = session.get(ImageObject, upload_id)
            if record and record.project_id == project_id and record.status == "ready":
                return dto(session, project, store)
        uploads.reservation_for(db, user_id, project_id, upload_id)
        with uploads.upload_lock(settings, upload_id) as path:
            row = uploads.reservation_for(db, user_id, project_id, upload_id, refresh=True)
            if not path.exists() or path.stat().st_size != row.bytes:
                raise HTTPException(409, "Upload is incomplete")
            with path.open("rb") as source:
                result = commit_image(project_id, request, source, row.file_name, row.bytes, upload_id)
            path.unlink(missing_ok=True)
            return result

    @app.get("/api/v1/projects/{project_id}/images/{image_id}/file")
    def image_file(project_id: str, image_id: int, request: Request, thumbnail: bool = False):
        with owned(request, project_id) as (session, project, store):
            record = image_record(session, project_id, image_id)
            key, digest = record.object_key, record.sha256
        # Network/disk I/O must not hold the shared quota/project locks.
        if thumbnail:
            return Response(objects.get(key.rsplit("/", 1)[0] + "/thumbnail.png"), media_type="image/png")
        data = cache.get(objects, key, digest)
        with Image.open(io.BytesIO(data)) as original:
            # Native browser formats need no full decode/re-encode. Do not let
            # EXIF auto-rotation move pixels away from their annotation grid.
            if (original.format in {"JPEG", "PNG", "WEBP"}
                    and original.mode in {"RGB", "L", "P"} and "transparency" not in original.info
                    and original.getexif().get(274, 1) == 1):
                return Response(data, media_type=Image.MIME[original.format])
            output = io.BytesIO()
            rgb = original.convert("RGB")
            rgb.info.pop("exif", None)
            rgb.save(output, format="PNG")
        return Response(output.getvalue(), media_type="image/png")

    @app.get("/api/v1/projects/{project_id}/images/{image_id}/state")
    def get_state(project_id: str, image_id: int, request: Request):
        with owned(request, project_id) as (session, project, store):
            image_record(session, project_id, image_id)
            return store.get_state(image_id)

    @app.put("/api/v1/projects/{project_id}/images/{image_id}/state")
    def save_state(project_id: str, image_id: int, body: dict[str, Any], request: Request):
        def save_staged(store, session, project):
            image = image_record(session, project_id, image_id)
            validate_mask_dimensions(body, image)
            return store.save_state(image_id, body)
        return mutate_project(settings, db, objects, project_id, auth.user_for_request(request), save_staged)

    @app.post("/api/v1/projects/{project_id}/categories")
    def category(project_id: str, body: CategoryInput, request: Request):
        return mutate_project(settings, db, objects, project_id, auth.user_for_request(request),
                              lambda store, *_: store.add_category(body.name, body.color))

    @app.patch("/api/v1/projects/{project_id}/categories/{category_id}")
    def edit_category(project_id: str, category_id: int, body: CategoryUpdate, request: Request):
        return mutate_project(settings, db, objects, project_id, auth.user_for_request(request),
                              lambda store, *_: store.update_category(category_id, body.model_dump(exclude_none=True)))

    @app.post("/api/v1/projects/{project_id}/geometry")
    def geometry(project_id: str, body: GeometryInput, request: Request):
        with owned(request, project_id) as (session, project, store):
            record = image_record(session, project_id, body.image_id)
            mask = rasterize_components(body.components, record.width, record.height)
            if not mask.any():
                raise ValueError("Geometry cannot be empty")
            return {**mask_payload(mask), "components": body.components, "controls": body.components}

    @app.post("/api/v1/projects/{project_id}/masks/union")
    def union(project_id: str, body: UnionInput, request: Request):
        with owned(request, project_id) as (session, project, store):
            record = image_record(session, project_id, body.image_id)
            validate_mask_dimensions([{"mask": mask} for mask in body.masks], record)
            mask = union_masks(body.masks, record.width, record.height)
            if not mask.any():
                raise ValueError("Mask cannot be empty")
            return mask_payload(mask)

    @app.post("/api/v1/projects/{project_id}/masks/fill-holes")
    def fill(project_id: str, body: FillInput, request: Request):
        with owned(request, project_id) as (session, project, store):
            record = image_record(session, project_id, body.image_id)
            validate_mask_dimensions({"mask": body.mask}, record)
            mask = decode_mask(body.mask)
            if mask.shape != (record.height, record.width) or not mask.any():
                raise ValueError("Mask is empty or has incorrect dimensions")
            cleaned, holes, pixels = fill_small_holes(mask, body.max_area)
            return {**mask_payload(cleaned), "filled_holes": holes, "filled_pixels": pixels}

    def submit(project_id, kind, body, request):
        user_id = auth.user_for_request(request)
        with owned(request, project_id) as (session, project, store):
            image = image_record(session, project_id, body.image_id)
            validate_mask_dimensions(body.model_dump(), image)
            state = store.get_state(body.image_id)
            if state["revision"] != body.revision:
                raise HTTPException(409, "Image revision changed; reload before inference")
            if kind != "points" and body.category_id not in {c["id"] for c in store.project()["categories"]}:
                raise ValueError("Select a valid class")
        return jobs.enqueue(db, settings, user_id, project_id, body.image_id, kind, body.model_dump())

    @app.post("/api/v1/projects/{project_id}/infer/points", status_code=202)
    def infer_points(project_id: str, body: PointInput, request: Request):
        return submit(project_id, "points", body, request)

    @app.post("/api/v1/projects/{project_id}/infer/text", status_code=202)
    def infer_text(project_id: str, body: TextInput, request: Request):
        return submit(project_id, "text", body, request)

    @app.post("/api/v1/projects/{project_id}/infer/visual", status_code=202)
    def infer_visual(project_id: str, body: VisualInput, request: Request):
        try:
            raw = base64.b64decode(body.reference_image, validate=True)
        except (ValueError, TypeError):
            raise ValueError("Invalid reference image encoding")
        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(413, "Reference image exceeds 10 MiB")
        validate_image(raw, settings)
        with Image.open(io.BytesIO(raw)) as image:
            oriented = ImageOps.exif_transpose(image)
            if body.reference_box:
                x1, y1, x2, y2 = body.reference_box
                if not (0 <= x1 < x2 <= oriented.width and 0 <= y1 < y2 <= oriented.height):
                    raise ValueError("Reference box lies outside the image")
        return submit(project_id, "visual", body, request)

    @app.get("/api/v1/jobs/{job_id}")
    def job_status(job_id: str, request: Request):
        user_id = auth.user_for_request(request)
        with db.session() as session:
            job = session.get(Job, job_id)
            if not job or job.user_id != user_id:
                raise HTTPException(404, "Job not found")
            project = session.get(Project, job.project_id)
            if project is None or project.deleted_at:
                raise HTTPException(404, "Job not found")
            return jobs.serialize(job)

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel(job_id: str, request: Request):
        return jobs.cancel(db, auth.user_for_request(request), job_id)

    @app.post("/api/v1/projects/{project_id}/coco/import")
    def import_coco(project_id: str, request: Request, file: UploadFile = File(...)):
        source = json.loads(file.file.read(settings.max_request_bytes + 1).decode("utf-8-sig"))
        if not isinstance(source, dict) or any(not isinstance(source.get(k), list) for k in ("images", "categories", "annotations")):
            raise ValueError("A COCO instance dataset is required")
        def import_staged(store, session, project):
            current = store.project()
            if current["categories"] or any(store.get_state(i["id"])["annotations"] or store.get_state(i["id"])["draft"]
                    or store.get_state(i["id"])["proposals"] for i in current["images"]):
                raise HTTPException(409, "Import COCO before adding classes or annotations")
            records = {im.file_name: im for im in session.scalars(select(ImageObject).where(
                ImageObject.project_id == project_id, ImageObject.status == "ready"))}
            mapping, seen = {}, set()
            for item in source["images"]:
                if not isinstance(item, dict) or type(item.get("id")) is not int or item["id"] in mapping:
                    raise ValueError("COCO image IDs must be unique integers")
                name = _relative_file(item.get("file_name"))
                if name not in records or name in seen:
                    raise ValueError("COCO images must match uploaded relative paths")
                seen.add(name)
                mapping[item["id"]] = records[name].image_id
                item["id"] = records[name].image_id
            if seen != set(records):
                raise ValueError("COCO must describe every uploaded image")
            mask_budget = [settings.max_mask_pixels_per_request]
            by_image_id = {record.image_id: record for record in records.values()}
            for item in source["annotations"]:
                if not isinstance(item, dict) or type(item.get("image_id")) is not int or item["image_id"] not in mapping:
                    raise ValueError("Unknown COCO image reference")
                item["image_id"] = mapping[item["image_id"]]
                record = by_image_id[item["image_id"]]
                if isinstance(item.get("segmentation"), dict):
                    validate_mask_dimensions({"mask": item["segmentation"]}, record, mask_budget)
                else:
                    mask_budget[0] -= record.height * record.width
                    if mask_budget[0] < 0:
                        raise HTTPException(413, "This COCO import contains too many full-image masks for hosted processing")
            with tempfile.TemporaryDirectory(dir=settings.data_dir, prefix="coco-") as temporary:
                temporary = Path(temporary)
                root = temporary / "images"
                root.mkdir()
                for name, record in records.items():
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(cache.get(objects, record.object_key, record.sha256))
                json_path = temporary / "source.json"
                json_path.write_text(json.dumps(source, allow_nan=False), encoding="utf-8")
                imported = ProjectStore.import_coco(temporary / "import", root, json_path)
                with imported._connect() as connection:
                    connection.execute("UPDATE meta SET value=? WHERE key='name'", (project.name,))
                    connection.execute("UPDATE meta SET value='unavailable-originals' WHERE key='image_root'")
                os.replace(imported.database, store.database)
            return dto(session, project, store)
        return mutate_project(settings, db, objects, project_id, auth.user_for_request(request), import_staged)

    @app.post("/api/v1/projects/{project_id}/coco/export")
    @app.get("/api/v1/projects/{project_id}/coco/export")
    def export_coco(project_id: str, request: Request):
        with owned(request, project_id, write=True) as (session, project, store):
            path = store.export_coco()
            try:
                # A crash after SQLite committed but before object metadata did
                # can leave an orphan image awaiting reconciliation. Export
                # uses the same availability boundary as the project DTO.
                ready = set(session.scalars(select(ImageObject.image_id).where(
                    ImageObject.project_id == project_id, ImageObject.status == "ready")))
                document = json.loads(path.read_text(encoding="utf-8"))
                document["images"] = [image for image in document["images"] if image["id"] in ready]
                document["annotations"] = [annotation for annotation in document["annotations"]
                                           if annotation["image_id"] in ready]
                data = json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
                objects.put(f"exports/{project_id}/anotaciones.coco.json", data, "application/json")
            finally:
                path.unlink(missing_ok=True)
            return {"file_name": "anotaciones.coco.json"}

    @app.get("/api/v1/projects/{project_id}/coco/download")
    def download_coco(project_id: str, request: Request):
        with owned(request, project_id) as (_, project, store):
            data = objects.get(f"exports/{project_id}/anotaciones.coco.json")
        return Response(data, media_type="application/json",
                        headers={"Content-Disposition": 'attachment; filename="anotaciones.coco.json"'})

    frontend = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    # static assets have no access to data_dir/cache_dir; legal pages use the SPA.
    if (frontend / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

    @app.get("/{path:path}")
    def page(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "Not found")
        if (frontend / "index.html").is_file():
            return FileResponse(frontend / "index.html")
        return {"message": "Build the frontend to use Annotation"}

    return app
