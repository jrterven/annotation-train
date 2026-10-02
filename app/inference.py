"""Local SAM 3 inference with separate concept and interactive-instance models.

Only the official facebook/sam3 weights are used. Imports and weight loading are
lazy so project management and polygon editing also work before model setup.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image


# Public Hugging Face repository revision, verified via /api/models/facebook/sam3.
DEFAULT_MODEL_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"


class ModelUnavailable(RuntimeError):
    """A model could not be loaded or used; the message is safe for the UI."""


def _mask_digest(mask: np.ndarray) -> str:
    return hashlib.blake2b(np.packbits(mask).tobytes(), digest_size=16).hexdigest()


def _is_mps_unsupported(error: Exception) -> bool:
    message = str(error).lower()
    return ("mps" in message or "metal" in message) and any(
        marker in message
        for marker in ("not implemented", "not supported", "unsupported", "not currently supported")
    )


def _safe_error(error: Exception) -> str:
    # Never echo an authentication token, including tokens embedded in HTTP errors.
    return re.sub(r"hf_[A-Za-z0-9]+", "[token redacted]", str(error)).split("\n")[0][:300]


class Sam3Engine:
    """Blocking API. A single RLock serializes inference and model transitions."""

    def __init__(
        self,
        device: str | None = None,
        *,
        cache_size: int = 2,
        confidence_threshold: float = 0.5,
        revision: str | None = None,
    ) -> None:
        self.lock = threading.RLock()
        self._status_lock = threading.Lock()
        self.requested_device = device or os.environ.get("SAM3_DEVICE", "auto")
        if self.requested_device not in {"auto", "mps", "cpu", "cuda"}:
            raise ValueError("SAM3_DEVICE must be auto, mps, cuda, or cpu.")
        self.model_id = "facebook/sam3"
        self.revision = revision or os.environ.get("SAM3_REVISION") or DEFAULT_MODEL_REVISION
        self.cache_size = max(1, min(int(cache_size), 8))
        self.confidence_threshold = float(confidence_threshold)
        if not 0 < self.confidence_threshold < 1:
            raise ValueError("The confidence threshold must be between 0 and 1.")
        self._status = {"state": "unloaded", "device": None, "message": "SAM 3 is not loaded."}
        self._torch = None
        self._detector = self._tracker = None
        self._detector_processor = self._tracker_processor = None
        self._detector_cache: OrderedDict = OrderedDict()
        self._tracker_cache: OrderedDict = OrderedDict()
        self._seed_cache: OrderedDict = OrderedDict()
        self._part_cache: OrderedDict = OrderedDict()
        self._reference_cache: OrderedDict = OrderedDict()
        self._device: str | None = None
        self._fallback_reason: str | None = None
        self.last_timings: dict[str, float] = {}

    def status(self) -> dict:
        # Polling /health must not wait for a download or an inference call.
        with self._status_lock:
            return {
                **self._status,
                "model_id": self.model_id,
                "revision": self.revision,
                "dtype": "float32",
            }

    def _set_status(self, state: str, message: str) -> None:
        with self._status_lock:
            self._status = {"state": state, "device": self._device, "message": message}

    def _choose_device(self) -> str:
        torch = self._torch
        if self.requested_device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if self.requested_device == "cuda" and not torch.cuda.is_available():
            raise ModelUnavailable("CUDA is not available on this computer.")
        if self.requested_device == "mps" and not torch.backends.mps.is_available():
            raise ModelUnavailable("Metal/MPS is not available on this computer.")
        return self.requested_device

    def load(self) -> dict:
        with self.lock:
            if self.status()["state"] == "ready":
                return self.status()
            self._set_status("loading", "Loading SAM 3 and its interactive predictor…")
            started = time.perf_counter()
            try:
                import torch
                from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor

                self._torch = torch
                self._device = self._choose_device()
                kwargs = {"revision": self.revision}
                self._detector_processor = Sam3Processor.from_pretrained(self.model_id, **kwargs)
                self._tracker_processor = Sam3TrackerProcessor.from_pretrained(self.model_id, **kwargs)
                model_kwargs = {**kwargs, "dtype": torch.float32, "attn_implementation": "sdpa"}
                self._detector = Sam3Model.from_pretrained(self.model_id, **model_kwargs).eval()
                self._tracker = Sam3TrackerModel.from_pretrained(self.model_id, **model_kwargs).eval()
                try:
                    self._detector.to(self._device)
                    self._tracker.to(self._device)
                except (RuntimeError, NotImplementedError) as error:
                    if self._device != "mps" or not _is_mps_unsupported(error):
                        raise
                    self._fallback_to_cpu(error)
                resolved = getattr(self._detector.config, "_commit_hash", None)
                if resolved:
                    self.revision = resolved
                self.last_timings = {"load_seconds": time.perf_counter() - started}
                self._ready_status()
            except (ImportError, ModuleNotFoundError) as error:
                self._release_models()
                self._set_status(
                    "missing_dependencies",
                    "Install requirements.txt with Python 3.12 to use SAM 3. " + _safe_error(error),
                )
            except Exception as error:
                self._release_models()
                status_code = getattr(getattr(error, "response", None), "status_code", None)
                if status_code in {401, 403} or any(
                    marker in str(error).lower() for marker in ("gated repo", "gated model", "unauthorized", "access to model")
                ):
                    self._set_status(
                        "auth_required",
                        "Request access to facebook/sam3 on Hugging Face and run hf auth login in a terminal.",
                    )
                else:
                    self._set_status("error", "Could not load SAM 3. " + _safe_error(error))
            result = self.status()
            if result["state"] != "ready":
                raise ModelUnavailable(result["message"])
            return result

    def _ready_status(self) -> None:
        message = f"SAM 3 ready · {self._device.upper()}"
        if self._fallback_reason:
            message += ". Using CPU because an operation is not supported by Metal."
        self._set_status("ready", message)

    def _clear_caches(self) -> None:
        self._detector_cache.clear()
        self._tracker_cache.clear()
        self._part_cache.clear()
        self._seed_cache.clear()
        self._reference_cache.clear()

    def _release_models(self) -> None:
        self._clear_caches()
        self._detector = self._tracker = None
        self._detector_processor = self._tracker_processor = None

    def _fallback_to_cpu(self, error: Exception) -> None:
        self._clear_caches()
        self._detector.to("cpu")
        self._tracker.to("cpu")
        self._device = "cpu"
        self._fallback_reason = _safe_error(error)
        if self._torch.backends.mps.is_available():
            self._torch.mps.empty_cache()
        self._ready_status()

    def _execute(self, operation: Callable[[], Any]) -> Any:
        with self.lock:
            if self.status()["state"] != "ready":
                self.load()
            if self.status()["state"] != "ready":
                raise ModelUnavailable(self.status()["message"])
            try:
                with self._torch.inference_mode():
                    return operation()
            except (RuntimeError, NotImplementedError) as error:
                if self._device == "mps" and _is_mps_unsupported(error):
                    self._fallback_to_cpu(error)
                    with self._torch.inference_mode():
                        return operation()
                raise ModelUnavailable("SAM 3 could not process this image. " + _safe_error(error)) from error

    @staticmethod
    def _put(cache: OrderedDict, key: Any, value: Any, limit: int) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    @staticmethod
    def _get(cache: OrderedDict, key: Any) -> Any:
        if key not in cache:
            return None
        cache.move_to_end(key)
        return cache[key]

    @staticmethod
    def _image_key(image: Image.Image, image_key: str) -> tuple:
        digest = hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()
        return image_key, image.size, digest

    def _device_inputs(self, values: dict) -> dict:
        return {key: value.to(self._device) for key, value in values.items() if hasattr(value, "to")}

    def _tracker_embeddings(self, image: Image.Image, key: tuple):
        cached = self._get(self._tracker_cache, key)
        if cached is not None:
            self.last_timings["embedding_seconds"] = 0.0
            return cached
        started = time.perf_counter()
        inputs = self._tracker_processor(images=image, return_tensors="pt")
        embeddings = self._tracker.get_image_embeddings(inputs["pixel_values"].to(self._device))
        self._put(self._tracker_cache, key, embeddings, self.cache_size)
        self.last_timings["embedding_seconds"] = time.perf_counter() - started
        return embeddings

    def _detector_embeddings(self, image: Image.Image, key: tuple, *, role: str = "target"):
        # An uploaded reference and an annotation target never share cache slots,
        # even if a caller happens to give them the same external image ID.
        cache_key = (role, key)
        cached = self._get(self._detector_cache, cache_key)
        if cached is not None:
            self.last_timings["embedding_seconds"] = 0.0
            return cached
        started = time.perf_counter()
        inputs = self._detector_processor(images=image, return_tensors="pt")
        embeddings = self._detector.get_vision_features(inputs["pixel_values"].to(self._device))
        self._put(self._detector_cache, cache_key, embeddings, self.cache_size)
        self.last_timings["embedding_seconds"] = time.perf_counter() - started
        return embeddings

    @staticmethod
    def _validate_part(part: dict, width: int, height: int) -> tuple:
        if not isinstance(part, dict):
            raise ValueError("The part must contain points, a box, or an initial mask.")
        points = []
        raw_points = part.get("points") or []
        if not isinstance(raw_points, list) or len(raw_points) > 512:
            raise ValueError("Each part supports up to 512 points.")
        for point in raw_points:
            if not isinstance(point, dict) or not all(name in point for name in ("x", "y", "label")):
                raise ValueError("Each point must contain x, y, and label.")
            if any(isinstance(point[name], bool) or not isinstance(point[name], (int, float)) for name in ("x", "y")):
                raise ValueError("Point coordinates must be numbers.")
            x, y, label = float(point["x"]), float(point["y"]), point["label"]
            if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x < width and 0 <= y < height):
                raise ValueError("Points must be inside the image.")
            if type(label) is not int or label not in (0, 1):
                raise ValueError("A point label must be 0 or 1.")
            points.append((x, y, int(label)))
        box = None
        if part.get("box") is not None:
            if not isinstance(part["box"], (list, tuple)) or len(part["box"]) != 4:
                raise ValueError("The box must contain x1, y1, x2, y2.")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in part["box"]):
                raise ValueError("Box coordinates must be numbers.")
            box = tuple(float(value) for value in part["box"])
            if not all(math.isfinite(value) for value in box) or not (
                0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
            ):
                raise ValueError("The box must have positive area and be inside the image.")
        if not any(p[2] == 1 for p in points) and box is None and part.get("seed_mask") is None:
            raise ValueError("Add a positive point, a box, or an initial mask.")
        return tuple(points), box

    def _seed_logits(self, key: tuple, seed: np.ndarray):
        """Map an original-raster seed into the tracker's square mask grid.

        Detector logits are reused while cached. A reopened binary RLE has no
        probabilities, so foreground/background become +/-10 logit prompts.
        Both HF image processors resize anisotropically, without letterboxing.
        Coordinates go through Sam3TrackerProcessor; masks use the matching
        bilinear, align_corners=False transform, at the model's actual grid size.
        """
        torch = self._torch
        raw = self._get(self._seed_cache, (key, _mask_digest(seed)))
        if raw is None:
            raw = torch.from_numpy(np.where(seed, 10.0, -10.0).astype(np.float32))[None, None]
        else:
            raw = raw[None, None]
        target_size = self._tracker.prompt_encoder.mask_input_size
        return torch.nn.functional.interpolate(
            raw.float(), size=target_size, mode="bilinear", align_corners=False, antialias=True
        ).to(self._device)

    def predict_points(self, image: Image.Image, image_key: str, part: dict) -> np.ndarray:
        image = image.convert("RGB")
        points, box = self._validate_part(part, image.width, image.height)
        seed = None
        if part.get("seed_mask") is not None:
            from .geometry import decode_mask

            seed = decode_mask(part["seed_mask"])
            if seed.shape != (image.height, image.width) or not seed.any():
                raise ValueError("The initial mask must match the image and contain an object.")
        key = self._image_key(image, image_key)
        return self._execute(lambda: self._predict_points(image, key, part, points, box, seed))

    def _predict_points(self, image, key, part, points, box, seed):
        started = time.perf_counter()
        self.last_timings = {}
        embeddings = self._tracker_embeddings(image, key)
        prompt_args: dict[str, Any] = {"original_sizes": [[image.height, image.width]], "return_tensors": "pt"}
        if points:
            prompt_args["input_points"] = [[[[p[0], p[1]] for p in points]]]
            prompt_args["input_labels"] = [[[p[2] for p in points]]]
        if box:
            prompt_args["input_boxes"] = [[list(box)]]
        inputs = self._tracker_processor(**prompt_args)
        model_inputs = self._device_inputs({key: val for key, val in inputs.items() if key != "original_sizes"})
        seed_hash = _mask_digest(seed) if seed is not None else None
        part_key = (key, str(part.get("id", "")))
        previous = self._get(self._part_cache, part_key)
        append_only = (
            previous is not None
            and previous["box"] == box
            and previous["seed_hash"] == seed_hash
            and len(points) > len(previous["points"])
            and points[:len(previous["points"])] == previous["points"]
        )
        if append_only:
            model_inputs["input_masks"] = previous["logits"].to(self._device)
        elif seed is not None:
            model_inputs["input_masks"] = self._seed_logits(key, seed)
        # A single ambiguous positive click benefits from three candidates.
        multimask = len(points) == 1 and points[0][2] == 1 and box is None and seed is None
        outputs = self._tracker(image_embeddings=embeddings, multimask_output=multimask, **model_inputs)
        scores = outputs.iou_scores[0, 0].float()
        if not self._torch.isfinite(scores).all() or not self._torch.isfinite(outputs.pred_masks).all():
            raise ModelUnavailable("SAM 3 returned invalid numeric values; try using CPU.")
        best = int(scores.argmax().item())
        low_res = outputs.pred_masks[0, 0, best].float().detach().cpu()
        high_res = self._tracker_processor.post_process_masks(
            outputs.pred_masks[:, :, best:best + 1].float().cpu(),
            [[image.height, image.width]],
            binarize=True,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )[0]
        mask = high_res.reshape(image.height, image.width).numpy().astype(bool)
        self._put(self._part_cache, part_key, {
            "points": points, "box": box, "seed_hash": seed_hash, "logits": low_res[None, None]
        }, 64)
        self.last_timings["total_seconds"] = time.perf_counter() - started
        return mask

    def predict_text(self, image: Image.Image, image_key: str, text: str) -> list[dict]:
        if not isinstance(text, str):
            raise ValueError("The concept must be text.")
        text = text.strip()
        if not text or len(text) > 256:
            raise ValueError("Enter a concept containing 1 to 256 characters.")
        image = image.convert("RGB")
        key = self._image_key(image, image_key)
        return self._execute(lambda: self._predict_text(image, key, text))

    def _predict_text(self, image, key, text):
        started = time.perf_counter()
        self.last_timings = {}
        inputs = self._text_inputs(text)
        embeddings = self._detector_embeddings(image, key)
        outputs = self._detector(vision_embeds=embeddings, **self._device_inputs(inputs))
        proposals = self._concept_proposals(outputs, image, key)
        self.last_timings["total_seconds"] = time.perf_counter() - started
        return proposals

    def _text_inputs(self, text: str):
        inputs = self._detector_processor(text=text, return_tensors="pt")
        # Sam3Processor pads to 32 tokens but deliberately does not truncate.
        # Character count alone cannot protect the text encoder's position table.
        max_tokens = self._detector.config.text_config.max_position_embeddings
        if inputs["input_ids"].shape[-1] > max_tokens:
            raise ValueError(
                f"The description is too long for SAM 3 (maximum {max_tokens} tokens). "
                "Use a shorter description."
            )
        return inputs

    @staticmethod
    def _combine_concept_prompts(text_features, text_mask, geometry_features, geometry_mask):
        """Pure tensor assembly matching Sam3Model's native box-prompt branch.

        Transformers 5.18 reads text_embeds.pooler_output directly and uses it as
        the complete downstream prompt sequence. Supplying the already combined
        sequence avoids hooks, model mutation, and any reference/target collage.
        """
        import torch
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        if text_features.ndim != 3 or geometry_features.ndim != 3:
            raise ModelUnavailable("SAM 3 returned an incompatible visual prompt shape.")
        if text_features.shape[0] != geometry_features.shape[0] or text_features.shape[-1] != geometry_features.shape[-1]:
            raise ModelUnavailable("SAM 3 returned incompatible text and visual prompt features.")
        if text_mask is None:
            text_mask = torch.ones(text_features.shape[:2], device=text_features.device, dtype=torch.bool)
        if geometry_mask is None:
            geometry_mask = torch.ones(geometry_features.shape[:2], device=geometry_features.device, dtype=torch.bool)
        if text_mask.shape != text_features.shape[:2] or geometry_mask.shape != geometry_features.shape[:2]:
            raise ModelUnavailable("SAM 3 returned an incompatible visual prompt attention mask.")
        features = torch.cat([text_features, geometry_features], dim=1)
        mask = torch.cat([text_mask.bool(), geometry_mask.bool()], dim=1)
        if not torch.isfinite(features).all():
            raise ModelUnavailable("SAM 3 returned invalid numeric values in the visual reference.")
        return BaseModelOutputWithPooling(pooler_output=features), mask

    def predict_visual(
        self, image: Image.Image, image_key: str, reference: Image.Image,
        reference_box: list[float] | tuple[float, ...] | None = None,
        text: str | None = None,
    ) -> list[dict]:
        """Find target instances using an external reference and optional text.

        Cross-image reuse is an experimental adaptation of SAM 3's intra-image
        exemplar encoder, not a native separate-reference processor argument.
        The reference raster must already have its intended EXIF orientation;
        box coordinates are measured in that raster, before model resizing.
        """
        if not isinstance(reference, Image.Image) or not isinstance(image, Image.Image):
            raise ValueError("The target and visual reference must be images.")
        if text is not None and not isinstance(text, str):
            raise ValueError("The concept must be text.")
        concept = text.strip() if text is not None else ""
        if len(concept) > 256:
            raise ValueError("Enter a concept containing at most 256 characters.")
        concept = concept or "visual"
        reference = reference.convert("RGB")
        image = image.convert("RGB")
        if min(reference.size) <= 0 or min(image.size) <= 0:
            raise ValueError("The target and visual reference must have positive dimensions.")
        raw_box = reference_box if reference_box is not None else [0, 0, reference.width, reference.height]
        _, box = self._validate_part({"box": raw_box}, reference.width, reference.height)
        key = self._image_key(image, image_key)
        reference_key = self._image_key(reference, "visual-reference")
        return self._execute(lambda: self._predict_visual(image, key, reference, reference_key, box, concept))

    def _reference_prompt(self, reference, reference_key, box):
        cache_key = (reference_key, box)
        cached = self._get(self._reference_cache, cache_key)
        if cached is not None:
            self.last_timings["reference_seconds"] = 0.0
            return cached
        started = time.perf_counter()
        inputs = self._detector_processor(
            original_sizes=[[reference.height, reference.width]], input_boxes=[[list(box)]],
            input_boxes_labels=[[1]], return_tensors="pt",
        )
        inputs = self._device_inputs(inputs)
        embeddings = self._detector_embeddings(reference, reference_key, role="reference")
        geometry = self._detector.geometry_encoder(
            box_embeddings=inputs["input_boxes"],
            box_mask=inputs["input_boxes_labels"] != -10,
            box_labels=inputs["input_boxes_labels"],
            img_feats=embeddings.fpn_hidden_states[:-1],
            img_pos_embeds=embeddings.fpn_position_encoding[:-1],
        )
        if not self._torch.isfinite(geometry.last_hidden_state).all():
            raise ModelUnavailable("SAM 3 returned invalid numeric values in the visual reference.")
        result = (geometry.last_hidden_state.detach(), geometry.attention_mask.detach())
        self._put(self._reference_cache, cache_key, result, self.cache_size * 4)
        self.last_timings["reference_seconds"] = time.perf_counter() - started
        return result

    def _visual_outputs(self, image, key, reference, reference_key, box, text):
        inputs = self._device_inputs(self._text_inputs(text))
        geometry_features, geometry_mask = self._reference_prompt(reference, reference_key, box)
        text_features = self._detector.get_text_features(**inputs, return_dict=True).pooler_output
        combined, attention_mask = self._combine_concept_prompts(
            text_features, inputs.get("attention_mask"), geometry_features, geometry_mask,
        )
        target_embeddings = self._detector_embeddings(image, key)
        return self._detector(
            vision_embeds=target_embeddings, text_embeds=combined, attention_mask=attention_mask,
        )

    def _predict_visual(self, image, key, reference, reference_key, box, text):
        started = time.perf_counter()
        self.last_timings = {}
        outputs = self._visual_outputs(image, key, reference, reference_key, box, text)
        proposals = self._concept_proposals(outputs, image, key)
        self.last_timings["total_seconds"] = time.perf_counter() - started
        return proposals

    def _concept_proposals(self, outputs, image: Image.Image, key: tuple) -> list[dict]:
        """Postprocess target masks and retain their logits for click refinement."""
        if (not self._torch.isfinite(outputs.pred_masks).all()
                or not self._torch.isfinite(outputs.pred_logits).all()
                or (outputs.presence_logits is not None and not self._torch.isfinite(outputs.presence_logits).all())):
            raise ModelUnavailable("SAM 3 returned invalid numeric values; try using CPU.")
        results = self._detector_processor.post_process_instance_segmentation(
            outputs, threshold=self.confidence_threshold, mask_threshold=0.5,
            target_sizes=[[image.height, image.width]],
        )[0]
        # The official HF postprocessor keeps queries in model order after this
        # exact confidence rule. Retain logits to seed later instance refinement.
        query_scores = outputs.pred_logits.sigmoid()
        if outputs.presence_logits is not None:
            query_scores = query_scores * outputs.presence_logits.sigmoid()
        keep = (query_scores[0] > self.confidence_threshold).detach().cpu()
        kept_logits = outputs.pred_masks[0].detach().float().cpu()[keep]
        masks = results["masks"].detach().cpu().numpy().astype(bool)
        scores = results["scores"].detach().cpu().numpy()
        if len(kept_logits) != len(masks):
            raise ModelUnavailable("This Transformers version changed the order of SAM 3 results.")
        proposals = []
        for mask, score, logits in zip(masks, scores, kept_logits):
            if mask.any():
                self._put(self._seed_cache, (key, _mask_digest(mask)), logits, 128)
                proposals.append({"mask": mask, "score": float(score)})
        proposals.sort(key=lambda proposal: proposal["score"], reverse=True)
        return proposals
