"""Optional-on-use, local Spanish-to-English translation for SAM concepts."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import threading


MODEL_ID = "Helsinki-NLP/opus-mt-es-en"
MODEL_REVISION = "c96e2c5399ebfae4fc43d9669556b9afa74bb69d"


class TranslationUnavailable(RuntimeError):
    """Translation failed; the original prompt must remain available to edit."""


class PromptTranslator:
    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = str(cache_dir or Path(__file__).resolve().parent.parent / ".cache" / "huggingface")
        self._lock = threading.RLock()
        self._model = self._tokenizer = self._torch = None
        self._cache: OrderedDict[str, str] = OrderedDict()

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        options = {"revision": MODEL_REVISION, "cache_dir": self.cache_dir, "token": False}
        # Cached models work offline. Only the first Spanish request downloads
        # public weights; no images or prompt text are sent to a translation API.
        try:
            tokenizer = MarianTokenizer.from_pretrained(MODEL_ID, local_files_only=True, **options)
            model = MarianMTModel.from_pretrained(MODEL_ID, local_files_only=True, use_safetensors=False, **options)
        except OSError:
            tokenizer = MarianTokenizer.from_pretrained(MODEL_ID, **options)
            model = MarianMTModel.from_pretrained(MODEL_ID, use_safetensors=False, **options)
        self._model = model.to("cpu").eval()
        self._tokenizer = tokenizer
        self._torch = torch

    def translate(self, text: str, source_language: str = "en") -> str:
        if source_language not in {"en", "es"}:
            raise ValueError("Prompt language must be English or Spanish.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter an object or concept to segment.")
        text = text.strip()
        if len(text) > 300:
            raise ValueError("Keep the prompt under 300 characters.")
        if source_language == "en":
            return text
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text]
            try:
                self._load()
                inputs = self._tokenizer(text, return_tensors="pt", truncation=False)
                if inputs["input_ids"].shape[-1] > 128:
                    raise ValueError("The Spanish prompt is too long. Use a short object description.")
                with self._torch.inference_mode():
                    output = self._model.generate(**inputs, max_length=None, max_new_tokens=128, num_beams=4, do_sample=False)
                translated = self._tokenizer.decode(output[0], skip_special_tokens=True).strip()
                if not translated:
                    raise TranslationUnavailable("Translation returned no text. Try a shorter prompt or use English.")
            except (ValueError, TranslationUnavailable):
                raise
            except Exception as error:
                # HF errors can include URLs/tokens; expose only an actionable
                # message, while retaining the exception for local diagnostics.
                raise TranslationUnavailable(
                    "Local translation is unavailable. Check your connection for the first model download "
                    "and install requirements.txt, or select English and enter the prompt in English."
                ) from error
            self._cache[text] = translated
            if len(self._cache) > 128:
                self._cache.popitem(last=False)
            return translated
