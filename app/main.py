from __future__ import annotations

import base64
import binascii
import io
import math
import os
import threading
import uuid
import warnings
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .geometry import decode_mask, fill_small_holes, mask_payload, rasterize_components, union_masks
from .inference import Sam3Engine
from .storage import ProjectStore, RevisionConflict
from . import yolo
from .translation import PromptTranslator, TranslationUnavailable

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_REFERENCE_BYTES = 10 * 1024 * 1024
MAX_REFERENCE_PIXELS = 16_000_000
MAX_REFERENCE_BASE64 = 4 * ((MAX_REFERENCE_BYTES + 2) // 3)
MAX_VISUAL_REQUEST_BYTES = MAX_REFERENCE_BASE64 + 16 * 1024


class VisualUploadLimitMiddleware:
    """Bound the visual prompt JSON before FastAPI buffers or parses it."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or scope.get("path", "").rstrip("/") != "/api/infer/visual"):
            await self.app(scope, receive, send)
            return
        chunks = []
        length = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            length += len(chunk)
            if length > MAX_VISUAL_REQUEST_BYTES:
                response = JSONResponse({"detail": "The example image exceeds the 10 MiB upload limit."}, status_code=413)
                await response(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


app = FastAPI(title="Annotation and Training", version="0.1.0")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
app.add_middleware(VisualUploadLimitMiddleware)
ORIGINS = {"http://localhost:8765", "http://127.0.0.1:8765", "http://localhost:5173", "http://127.0.0.1:5173"}
app.add_middleware(CORSMiddleware, allow_origins=sorted(ORIGINS), allow_methods=["GET", "POST", "PUT", "PATCH"], allow_headers=["Content-Type", "X-Requested-With", "X-Project-Directory", "X-Annotation-State-Version"])
engine = Sam3Engine()
translator = PromptTranslator()
_store: ProjectStore | None = None
_project_lock = threading.RLock()


@app.middleware("http")
async def local_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.url.path.startswith("/api") and origin and origin not in ORIGINS:
        return JSONResponse({"detail": "Origin not allowed."}, status_code=403)
    # Capture once: another request may change the global project while this
    # request awaits a threadpool operation or even before its handler begins.
    project = _store
    request.state.project = project
    header_project = request.headers.get("x-project-directory")
    expected_project = unquote(header_project) if header_project else request.query_params.get("project")
    if expected_project and request.url.path.startswith("/api"):
        # A second tab may open another project on this single-user server.
        # Never let an old tab silently write image ID 1 in that other dataset.
        expected = str(Path(expected_project).expanduser().resolve())
        if project is None or expected != str(project.directory):
            return JSONResponse({"detail": "Another window opened a different project. Reopen your project before continuing."}, status_code=409)
    response = await call_next(request)
    if request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.exception_handler(RevisionConflict)
async def revision_error(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=409)


@app.exception_handler(ValueError)
async def value_error(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.exception_handler(FileNotFoundError)
async def missing_file(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=404)


@app.exception_handler(PermissionError)
async def denied_file(request, exc):
    return JSONResponse({"detail": f"Permission denied for this location: {exc}"}, status_code=403)


def store(request: Request) -> ProjectStore:
    project = request.state.project
    if project is None:
        raise HTTPException(409, "Open a project first.")
    return project


def image_info(project: ProjectStore, image_id: int) -> dict:
    result = next((im for im in project.project()["images"] if im["id"] == image_id), None)
    if result is None:
        raise HTTPException(404, "Image not found.")
    return result


def read_image(project: ProjectStore, image_id: int) -> Image.Image:
    info = image_info(project, image_id)
    with Image.open(project.image_path(image_id)) as image:
        if image.size != (info["width"], info["height"]):
            raise ValueError("The image dimensions have changed. Relink or check the original files.")
        # Keep the file's pixel grid. COCO coordinates refer to the stored raster,
        # not a browser's implicit EXIF rotation.
        return image.convert("RGB")


class OpenProject(BaseModel):
    directory: str = Field(min_length=1)
    image_root: str | None = None
    files: list[str] | None = None
    recursive: bool = True


class CategoryInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    color: str = Field(default="#8B8DE3", pattern=r"^#[0-9a-fA-F]{6}$")


class CategoryUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")


class RelinkInput(BaseModel):
    image_root: str


class GeometryInput(BaseModel):
    image_id: int
    components: list[dict[str, Any]]


class UnionInput(BaseModel):
    image_id: int
    masks: list[dict[str, Any]]


class FillHolesInput(BaseModel):
    image_id: int
    mask: dict[str, Any]
    max_area: int = Field(default=16, ge=1, le=150_000_000, strict=True)


class PointsInput(BaseModel):
    image_id: int
    revision: int
    part: dict[str, Any]


class TextInput(BaseModel):
    image_id: int
    revision: int
    text: str = Field(min_length=1, max_length=300)
    category_id: int
    source_language: Literal["en", "es"] = "en"


class VisualInput(BaseModel):
    image_id: int = Field(strict=True)
    revision: int = Field(ge=0, strict=True)
    category_id: int = Field(strict=True)
    text: str = Field(default="", max_length=300)
    source_language: Literal["en", "es"] = "en"
    reference_image: str = Field(min_length=1, max_length=MAX_REFERENCE_BASE64)
    reference_box: list[Any] | None = Field(default=None, min_length=4, max_length=4)


def decode_reference_image(encoded: str, box: list[Any] | None) -> tuple[Image.Image, list[float] | None]:
    """Decode an uploaded example in memory; crop coordinates use its EXIF-oriented raster."""
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("The example image must contain valid raw base64 data.") from exc
    if len(raw) > MAX_REFERENCE_BYTES:
        raise HTTPException(413, "The example image exceeds the 10 MiB upload limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as candidate:
                if candidate.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("Use a PNG, JPEG or WebP example image.")
                if candidate.width * candidate.height > MAX_REFERENCE_PIXELS:
                    raise HTTPException(413, "The example image exceeds the 16 megapixel limit.")
                if getattr(candidate, "is_animated", False) or getattr(candidate, "n_frames", 1) != 1:
                    raise ValueError("Use a single still image as the example; animated images are not supported.")
                candidate.verify()
            with Image.open(io.BytesIO(raw)) as candidate:
                oriented = ImageOps.exif_transpose(candidate)
                # Match the preview's white background instead of exposing RGB
                # values hidden behind transparent PNG/WebP pixels.
                if "A" in oriented.getbands() or "transparency" in oriented.info:
                    background = Image.new("RGBA", oriented.size, "white")
                    reference = Image.alpha_composite(background, oriented.convert("RGBA")).convert("RGB")
                else:
                    reference = oriented.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(413, "The example image exceeds the 16 megapixel limit.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("The example image is invalid or damaged.") from exc
    if box is not None:
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in box):
            raise ValueError("The example box must contain four finite numeric coordinates.")
        box = [float(value) for value in box]
        if not (0 <= box[0] < box[2] <= reference.width and 0 <= box[1] < box[3] <= reference.height):
            raise ValueError("The example box must have area and lie inside the oriented example image.")
    return reference, box


class ImportInput(BaseModel):
    directory: str
    image_root: str
    json_path: str


@app.get("/api/health")
def health():
    return {"status": "ok", "model": engine.status()}


@app.get("/api/browse")
def browse(path: str | None = None):
    folder = Path(path).expanduser() if path else Path.home()
    folder = folder.resolve()
    if not folder.is_dir():
        raise ValueError("Select an existing folder.")
    directories, files = [], []
    for entry in sorted(folder.iterdir(), key=lambda p: p.name.casefold()):
        if entry.name.startswith("."):
            continue
        try:
            item = {"name": entry.name, "path": str(entry)}
            if entry.is_dir():
                directories.append(item)
            elif entry.suffix.lower() in IMAGE_EXTENSIONS | {".json"}:
                files.append(item)
        except OSError:
            continue
    return {"path": str(folder), "parent": str(folder.parent) if folder.parent != folder else None, "directories": directories, "files": files, "exists": True}


@app.post("/api/projects/open")
def open_project(body: OpenProject):
    global _store
    with _project_lock:
        candidate = ProjectStore.open(**body.model_dump())
        result = candidate.project()
        _store = candidate
        return result


@app.get("/api/project")
def get_project(request: Request):
    project = getattr(request.state, "project", None)
    return project.project() if project is not None else None


@app.post("/api/project/relink")
def relink(body: RelinkInput, request: Request):
    with _project_lock:
        return store(request).relink(body.image_root)


@app.post("/api/project/categories")
def add_category(body: CategoryInput, request: Request):
    return store(request).add_category(body.name.strip(), body.color)


@app.patch("/api/project/categories/{category_id}")
def update_category(category_id: int, body: CategoryUpdate, request: Request):
    return store(request).update_category(category_id, body.model_dump(exclude_none=True))


@app.get("/api/images/{image_id}/file")
def get_image(image_id: int, request: Request, thumbnail: bool = False):
    image = read_image(store(request), image_id)
    if thumbnail:
        image.thumbnail((256, 192))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(buffer.getvalue(), media_type="image/png")


@app.get("/api/images/{image_id}/state")
def get_state(image_id: int, request: Request):
    return store(request).get_state(image_id, client_version=request.headers.get("x-annotation-state-version"))


@app.put("/api/images/{image_id}/state")
def save_state(image_id: int, body: dict[str, Any], request: Request):
    return store(request).save_state(image_id, body, client_version=request.headers.get("x-annotation-state-version"))


@app.post("/api/geometry")
def geometry(body: GeometryInput, request: Request):
    info = image_info(store(request), body.image_id)
    mask = rasterize_components(body.components, info["width"], info["height"])
    if not mask.any():
        raise ValueError("This edit would leave the object with no pixels. Delete the instance if it is no longer needed.")
    payload = mask_payload(mask)
    # Preserve inserted handles and subpixel coordinates. Retracing here would
    # erase a new collinear vertex immediately, even though its mask is valid.
    payload["components"] = body.components
    payload["controls"] = body.components
    return payload


@app.post("/api/masks/union")
def masks_union(body: UnionInput, request: Request):
    info = image_info(store(request), body.image_id)
    mask = union_masks(body.masks, info["width"], info["height"])
    if not mask.any():
        raise ValueError("The object contains no pixels. Adjust its points or boxes before confirming.")
    return mask_payload(mask)


@app.post("/api/masks/fill-holes")
def masks_fill_holes(body: FillHolesInput, request: Request):
    info = image_info(store(request), body.image_id)
    mask = decode_mask(body.mask)
    if mask.shape != (info["height"], info["width"]):
        raise ValueError("The mask does not match the image dimensions.")
    if not mask.any():
        raise ValueError("An empty mask cannot be cleaned.")
    cleaned, filled_holes, filled_pixels = fill_small_holes(mask, body.max_area)
    return {**mask_payload(cleaned), "filled_holes": filled_holes, "filled_pixels": filled_pixels}


@app.post("/api/model/load")
async def load_model():
    try:
        return await run_in_threadpool(engine.load)
    except Exception as exc:
        raise HTTPException(503, f"SAM 3 is not available: {exc}") from exc


@app.post("/api/infer/points")
async def infer_points(body: PointsInput, request: Request):
    project = store(request)
    image = await run_in_threadpool(read_image, project, body.image_id)
    key = f"{project.project()['directory']}:{body.image_id}:{project.image_path(body.image_id).stat().st_mtime_ns}"
    try:
        mask = await run_in_threadpool(engine.predict_points, image, key, body.part)
        payload = await run_in_threadpool(mask_payload, mask)
    except ValueError:
        raise
    except Exception as exc:
        raise HTTPException(503, f"SAM 3 segmentation failed: {exc}") from exc
    return {"image_id": body.image_id, "revision": body.revision, **payload}


@app.post("/api/infer/text")
async def infer_text(body: TextInput, request: Request):
    project = store(request)
    if not any(c["id"] == body.category_id for c in project.project()["categories"]):
        raise ValueError("Select a valid class.")
    image = await run_in_threadpool(read_image, project, body.image_id)
    key = f"{project.project()['directory']}:{body.image_id}:{project.image_path(body.image_id).stat().st_mtime_ns}"
    try:
        english = await run_in_threadpool(translator.translate, body.text, body.source_language)
        results = await run_in_threadpool(engine.predict_text, image, key, english)
        proposals = []
        for prediction in results:
            payload = await run_in_threadpool(mask_payload, prediction["mask"])
            proposals.append({"id": str(uuid.uuid4()), "category_id": body.category_id, "iscrowd": 0, "score": float(prediction["score"]), "selected": True, **payload})
    except ValueError:
        raise
    except TranslationUnavailable as exc:
        raise HTTPException(503, f"Prompt translation failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(503, f"SAM 3 search failed: {exc}") from exc
    return {"image_id": body.image_id, "revision": body.revision, "proposals": proposals,
            "prompt": {"original": body.text, "english": english, "source_language": body.source_language}}


@app.post("/api/infer/visual")
async def infer_visual(body: VisualInput, request: Request):
    project = store(request)
    if not any(c["id"] == body.category_id for c in project.project()["categories"]):
        raise ValueError("Select a valid class.")
    image = await run_in_threadpool(read_image, project, body.image_id)
    reference, box = await run_in_threadpool(decode_reference_image, body.reference_image, body.reference_box)
    key = f"{project.project()['directory']}:{body.image_id}:{project.image_path(body.image_id).stat().st_mtime_ns}"
    prompt = None
    try:
        english = None
        if body.text.strip():
            english = await run_in_threadpool(translator.translate, body.text, body.source_language)
            prompt = {"original": body.text, "english": english, "source_language": body.source_language}
        results = await run_in_threadpool(engine.predict_visual, image, key, reference, reference_box=box, text=english)
        proposals = []
        for prediction in results:
            payload = await run_in_threadpool(mask_payload, prediction["mask"])
            proposals.append({"id": str(uuid.uuid4()), "category_id": body.category_id, "iscrowd": 0,
                              "score": float(prediction["score"]), "selected": True, **payload})
    except ValueError:
        raise
    except TranslationUnavailable as exc:
        raise HTTPException(503, f"Prompt translation failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(503, f"SAM 3 visual search failed: {exc}") from exc
    result = {"image_id": body.image_id, "revision": body.revision, "proposals": proposals,
              "reference": {"width": reference.width, "height": reference.height},
              "method": "cross_image_exemplar"}
    if prompt is not None:
        result["prompt"] = prompt
    return result


@app.post("/api/coco/import")
def import_coco(body: ImportInput):
    global _store
    with _project_lock:
        candidate = ProjectStore.import_coco(body.directory, body.image_root, body.json_path)
        result = candidate.project()
        _store = candidate
        return result


@app.post("/api/yolo/preview")
def preview_yolo(body: yolo.ExportOptions, request: Request):
    return yolo.prepare(store(request), body)


@app.post("/api/yolo/export")
def export_yolo(body: yolo.ExportRequest, request: Request):
    project = store(request)
    export_id, path = yolo.generate(project, body, project.directory / "exports", project.image_path)
    return {"export_id": export_id, "file_name": f"annotations-{body.task}.zip", "path": str(path)}


@app.get("/api/yolo/download/{export_id}")
def download_yolo(export_id: str, request: Request):
    path = store(request).directory / "exports" / f"{yolo.export_identity(export_id)}.zip"
    if not path.is_file():
        raise FileNotFoundError("Export not found")
    return FileResponse(path, media_type="application/zip", filename="annotations-yolo.zip")


@app.post("/api/coco/export")
def export_coco(request: Request, body: yolo.CocoOptions = yolo.CocoOptions()):
    path = store(request).export_coco(body.task, unique=body.unique)
    if body.unique:
        return {"path": str(path), "file_name": "annotations.coco.json", "export_id": path.stem}
    return {"path": str(path), "file_name": path.name}


@app.get("/api/coco/download/{export_id}")
def download_coco_export(export_id: str, request: Request):
    path = store(request).directory / "exports" / f"{yolo.export_identity(export_id)}.json"
    if not path.is_file():
        raise FileNotFoundError("Export not found")
    return FileResponse(path, media_type="application/json", filename="annotations.coco.json")


@app.get("/api/coco/download")
def download_coco(request: Request):
    project = store(request)
    path = Path(project.project()["directory"]) / "anotaciones.coco.json"
    if not path.is_file():
        raise HTTPException(404, "Export the project first.")
    return FileResponse(path, media_type="application/json", filename=path.name)


if (ROOT / "frontend" / "dist").is_dir():
    app.mount("/", StaticFiles(directory=ROOT / "frontend" / "dist", html=True), name="frontend")
else:
    @app.get("/")
    def missing_frontend():
        return {"message": "Build the interface: cd frontend && npm ci && npm run build", "docs": "/docs"}
