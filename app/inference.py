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
    return re.sub(r"hf_[A-Za-z0-9]+", "[token oculto]", str(error)).split("\n")[0][:300]


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
            raise ValueError("SAM3_DEVICE debe ser auto, mps, cuda o cpu.")
        self.model_id = "facebook/sam3"
        self.revision = revision or os.environ.get("SAM3_REVISION") or DEFAULT_MODEL_REVISION
        self.cache_size = max(1, min(int(cache_size), 8))
        self.confidence_threshold = float(confidence_threshold)
        if not 0 < self.confidence_threshold < 1:
            raise ValueError("El umbral de confianza debe estar entre 0 y 1.")
        self._status = {"state": "unloaded", "device": None, "message": "SAM 3 aún no está cargado."}
        self._torch = None
        self._detector = self._tracker = None
        self._detector_processor = self._tracker_processor = None
        self._detector_cache: OrderedDict = OrderedDict()
        self._tracker_cache: OrderedDict = OrderedDict()
        self._seed_cache: OrderedDict = OrderedDict()
        self._part_cache: OrderedDict = OrderedDict()
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
            raise ModelUnavailable("CUDA no está disponible en este equipo.")
        if self.requested_device == "mps" and not torch.backends.mps.is_available():
            raise ModelUnavailable("Metal/MPS no está disponible en este equipo.")
        return self.requested_device

    def load(self) -> dict:
        with self.lock:
            if self.status()["state"] == "ready":
                return self.status()
            self._set_status("loading", "Cargando SAM 3 y su predictor interactivo…")
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
                    "Instala requirements.txt en Python 3.12 para usar SAM 3. " + _safe_error(error),
                )
            except Exception as error:
                self._release_models()
                status_code = getattr(getattr(error, "response", None), "status_code", None)
                if status_code in {401, 403} or any(
                    marker in str(error).lower() for marker in ("gated repo", "gated model", "unauthorized", "access to model")
                ):
                    self._set_status(
                        "auth_required",
                        "Autoriza facebook/sam3 en Hugging Face y ejecuta hf auth login en la terminal.",
                    )
                else:
                    self._set_status("error", "No se pudo cargar SAM 3. " + _safe_error(error))
            result = self.status()
            if result["state"] != "ready":
                raise ModelUnavailable(result["message"])
            return result

    def _ready_status(self) -> None:
        message = f"SAM 3 listo · {self._device.upper()}"
        if self._fallback_reason:
            message += ". Se usa CPU porque una operación no es compatible con Metal."
        self._set_status("ready", message)

    def _clear_caches(self) -> None:
        self._detector_cache.clear()
        self._tracker_cache.clear()
        self._part_cache.clear()
        self._seed_cache.clear()

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
                raise ModelUnavailable("SAM 3 no pudo procesar esta imagen. " + _safe_error(error)) from error

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

    def _detector_embeddings(self, image: Image.Image, key: tuple):
        cached = self._get(self._detector_cache, key)
        if cached is not None:
            self.last_timings["embedding_seconds"] = 0.0
            return cached
        started = time.perf_counter()
        inputs = self._detector_processor(images=image, return_tensors="pt")
        embeddings = self._detector.get_vision_features(inputs["pixel_values"].to(self._device))
        self._put(self._detector_cache, key, embeddings, self.cache_size)
        self.last_timings["embedding_seconds"] = time.perf_counter() - started
        return embeddings

    @staticmethod
    def _validate_part(part: dict, width: int, height: int) -> tuple:
        if not isinstance(part, dict):
            raise ValueError("La parte debe contener puntos, caja o máscara inicial.")
        points = []
        raw_points = part.get("points") or []
        if not isinstance(raw_points, list) or len(raw_points) > 512:
            raise ValueError("Se admiten hasta 512 puntos por parte.")
        for point in raw_points:
            if not isinstance(point, dict) or not all(name in point for name in ("x", "y", "label")):
                raise ValueError("Cada punto debe contener x, y y label.")
            if any(isinstance(point[name], bool) or not isinstance(point[name], (int, float)) for name in ("x", "y")):
                raise ValueError("Las coordenadas de los puntos deben ser números.")
            x, y, label = float(point["x"]), float(point["y"]), point["label"]
            if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x < width and 0 <= y < height):
                raise ValueError("Los puntos deben estar dentro de la imagen.")
            if type(label) is not int or label not in (0, 1):
                raise ValueError("La etiqueta de un punto debe ser 0 o 1.")
            points.append((x, y, int(label)))
        box = None
        if part.get("box") is not None:
            if not isinstance(part["box"], (list, tuple)) or len(part["box"]) != 4:
                raise ValueError("La caja debe contener x1, y1, x2, y2.")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in part["box"]):
                raise ValueError("Las coordenadas de la caja deben ser números.")
            box = tuple(float(value) for value in part["box"])
            if not all(math.isfinite(value) for value in box) or not (
                0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
            ):
                raise ValueError("La caja debe tener área y estar dentro de la imagen.")
        if not any(p[2] == 1 for p in points) and box is None and part.get("seed_mask") is None:
            raise ValueError("Añade un punto positivo, una caja o una máscara inicial.")
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
                raise ValueError("La máscara inicial debe coincidir con la imagen y contener un objeto.")
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
            raise ModelUnavailable("SAM 3 devolvió valores numéricos no válidos; prueba con CPU.")
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
            raise ValueError("El concepto debe ser texto.")
        text = text.strip()
        if not text or len(text) > 256:
            raise ValueError("Escribe un concepto de entre 1 y 256 caracteres.")
        image = image.convert("RGB")
        key = self._image_key(image, image_key)
        return self._execute(lambda: self._predict_text(image, key, text))

    def _predict_text(self, image, key, text):
        started = time.perf_counter()
        self.last_timings = {}
        inputs = self._detector_processor(text=text, return_tensors="pt")
        # Sam3Processor pads to 32 tokens but deliberately does not truncate.
        # Character count alone cannot protect the text encoder's position table.
        max_tokens = self._detector.config.text_config.max_position_embeddings
        if inputs["input_ids"].shape[-1] > max_tokens:
            raise ValueError(
                f"La descripción es demasiado larga para SAM 3 (máximo {max_tokens} tokens). "
                "Usa una descripción más corta."
            )
        embeddings = self._detector_embeddings(image, key)
        outputs = self._detector(vision_embeds=embeddings, **self._device_inputs(inputs))
        if (not self._torch.isfinite(outputs.pred_masks).all()
                or not self._torch.isfinite(outputs.pred_logits).all()
                or (outputs.presence_logits is not None and not self._torch.isfinite(outputs.presence_logits).all())):
            raise ModelUnavailable("SAM 3 devolvió valores numéricos no válidos; prueba con CPU.")
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
            raise ModelUnavailable("La versión de Transformers cambió el orden de resultados de SAM 3.")
        proposals = []
        for mask, score, logits in zip(masks, scores, kept_logits):
            if mask.any():
                self._put(self._seed_cache, (key, _mask_digest(mask)), logits, 128)
                proposals.append({"mask": mask, "score": float(score)})
        proposals.sort(key=lambda proposal: proposal["score"], reverse=True)
        self.last_timings["total_seconds"] = time.perf_counter() - started
        return proposals
