"""Private, bounded inference protocol. Importing it never imports PyTorch."""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import math
import warnings
from typing import Any, Literal
from uuid import UUID

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_REFERENCE_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_RESULT_BYTES = 20 * 1024 * 1024
MAX_RESULT_PIXELS = 1_024_000_000


class ResultTooLarge(ValueError):
    """Hosted inference output exceeds its storage/memory budget."""


def bounded_result(value: dict) -> dict:
    size = 0
    for chunk in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(value):
        size += len(chunk.encode("utf-8"))
        if size > MAX_RESULT_BYTES:
            raise ResultTooLarge("Inference output exceeds the 20 MiB limit; use a narrower prompt.")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PromptBase(StrictModel):
    revision: int = Field(default=0, ge=0, strict=True)
    image_id: int | None = Field(default=None, ge=1, strict=True)


class PointsPrompt(PromptBase):
    part: dict[str, Any]


class TextPrompt(PromptBase):
    text: str = Field(min_length=1, max_length=300)
    category_id: int = Field(ge=1, strict=True)
    source_language: Literal["en", "es"] = "en"


class VisualPrompt(TextPrompt):
    text: str = Field(default="", max_length=300)
    reference_image: str = Field(min_length=1, max_length=4 * ((MAX_REFERENCE_BYTES + 2) // 3))
    reference_box: list[float] | None = Field(default=None, min_length=4, max_length=4)


PROMPTS = {"points": PointsPrompt, "text": TextPrompt, "visual": VisualPrompt}


class AttemptRequest(StrictModel):
    job_id: UUID
    attempt_id: UUID
    project_id: UUID
    image_id: int = Field(ge=1, strict=True)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    deadline_at: float = Field(gt=0)
    kind: Literal["points", "text", "visual"]
    payload: dict[str, Any]
    image_base64: str = Field(min_length=1, max_length=4 * ((MAX_IMAGE_BYTES + 2) // 3))

    @model_validator(mode="after")
    def validate_prompt(self):
        parsed = PROMPTS[self.kind].model_validate(self.payload)
        if parsed.image_id is not None and parsed.image_id != self.image_id:
            raise ValueError("Prompt and target image differ.")
        self.payload = parsed.model_dump(exclude_none=True)
        return self

    @property
    def image_key(self) -> str:
        return f"{self.project_id}:{self.image_id}:{self.sha256}"


def decode_image(encoded: str, *, reference: bool = False) -> tuple[Image.Image, str]:
    """No filenames/URLs are accepted. Preserve the original target raster grid."""
    limit = MAX_REFERENCE_BYTES if reference else MAX_IMAGE_BYTES
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Image must contain raw base64 data.") from error
    if not data or len(data) > limit:
        raise ValueError("Image exceeds the byte limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as candidate:
                formats = {"PNG", "JPEG", "WEBP"} if reference else {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}
                if candidate.format not in formats or getattr(candidate, "n_frames", 1) != 1:
                    raise ValueError("Use a supported still image.")
                if candidate.width * candidate.height > MAX_PIXELS:
                    raise ValueError("Image exceeds the 16 megapixel limit.")
                candidate.verify()
            with Image.open(io.BytesIO(data)) as candidate:
                oriented = ImageOps.exif_transpose(candidate) if reference else candidate
                if reference and ("A" in oriented.getbands() or "transparency" in oriented.info):
                    background = Image.new("RGBA", oriented.size, "white")
                    image = Image.alpha_composite(background, oriented.convert("RGBA")).convert("RGB")
                else:
                    image = oriented.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("Invalid or oversized image.") from error
    return image, hashlib.sha256(data).hexdigest()


def reference_image(payload: dict) -> tuple[Image.Image, list[float] | None]:
    image, _ = decode_image(payload["reference_image"], reference=True)
    box = payload.get("reference_box")
    if box is not None:
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            raise ValueError("Reference box coordinates must be finite numbers.")
        if not 0 <= box[0] < box[2] <= image.width or not 0 <= box[1] < box[3] <= image.height:
            raise ValueError("Reference box must be inside the oriented reference image.")
    return image, box


def execute_inference(engine, translator, request: AttemptRequest) -> dict:
    """Same masks, proposals and revisions as the local editor's API."""
    from app.geometry import mask_payload

    image, digest = decode_image(request.image_base64)
    if digest != request.sha256:
        raise ValueError("Image checksum does not match its immutable identity.")
    prompt = request.payload
    result = {"image_id": request.image_id, "revision": prompt.get("revision", 0)}
    if request.kind == "points":
        seed = prompt["part"].get("seed_mask")
        if seed is not None and (not isinstance(seed, dict) or seed.get("size") != [image.height, image.width]):
            raise ValueError("Seed mask must match the target image.")
        return bounded_result({**result, **mask_payload(engine.predict_points(image, request.image_key, prompt["part"]))})
    english = translator.translate(prompt["text"], prompt["source_language"]) if prompt["text"].strip() else None
    if request.kind == "text":
        predictions = engine.predict_text(image, request.image_key, english)
    else:
        reference, box = reference_image(prompt)
        predictions = engine.predict_visual(image, request.image_key, reference, reference_box=box, text=english)
        result.update(reference={"width": reference.width, "height": reference.height}, method="cross_image_exemplar")
    # Deterministic proposal IDs make a replay of a completed attempt idempotent.
    if sum(int(prediction["mask"].size) for prediction in predictions) > MAX_RESULT_PIXELS:
        raise ResultTooLarge("Too many full-resolution proposals; use a narrower prompt.")
    from uuid import NAMESPACE_URL, uuid5
    result["proposals"] = [
        {"id": str(uuid5(NAMESPACE_URL, f"{request.job_id}:{i}")), "category_id": prompt["category_id"],
         "iscrowd": 0, "score": float(prediction["score"]), "selected": True, **mask_payload(prediction["mask"])}
        for i, prediction in enumerate(predictions)
    ]
    if english is not None:
        result["prompt"] = {"original": prompt["text"], "english": english, "source_language": prompt["source_language"]}
    return bounded_result(result)
