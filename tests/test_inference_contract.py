"""Processor/transport contracts; these tests do not claim model quality."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, Sam3Config, Sam3ImageProcessor, Sam3Processor, Sam3TrackerProcessor

from app.inference import ModelUnavailable, Sam3Engine, _is_mps_unsupported, _mask_digest, _safe_error


@pytest.fixture
def processor():
    return Sam3TrackerProcessor(image_processor=Sam3ImageProcessor(size={"height": 1008, "width": 1008}))


def test_original_non_square_coordinates_are_scaled_once(processor):
    encoded = processor(original_sizes=[[300, 900]], input_points=[[[[450, 150], [0, 299]]]],
                        input_labels=[[[1, 0]]], input_boxes=[[[90, 30, 810, 270]]], return_tensors="pt")
    torch.testing.assert_close(encoded["input_points"][0, 0, 0], torch.tensor([504.0, 504.0]))
    torch.testing.assert_close(encoded["input_boxes"][0, 0], torch.tensor([100.8, 100.8, 907.2, 907.2]))
    assert encoded["input_labels"].tolist() == [[[1, 0]]]
    assert "pixel_values" not in encoded  # embedding reuse does not re-encode pixels


def test_tracker_mask_postprocess_preserves_raster_axes(processor):
    logits = torch.full((1, 1, 1, 288, 288), -10.0)
    logits[..., :, 144:] = 10.0
    result = processor.post_process_masks(logits, [[100, 600]])[0]
    assert result.shape == (1, 1, 100, 600)
    assert not result[..., :290].any()
    assert result[..., 310:].all()


def test_persisted_seed_uses_model_grid_and_signed_logits():
    engine = Sam3Engine(device="cpu")
    engine._torch = torch
    engine._device = "cpu"
    engine._tracker = SimpleNamespace(prompt_encoder=SimpleNamespace(mask_input_size=(288, 288)))
    mask = np.zeros((100, 600), dtype=bool)
    mask[:, 300:] = True
    logits = engine._seed_logits(("non-square",), mask)
    assert logits.shape == (1, 1, 288, 288)
    assert logits.dtype == torch.float32
    assert (logits[..., :140] < 0).all() and (logits[..., 148:] > 0).all()


def test_text_seed_reuses_original_detector_logits():
    engine = Sam3Engine(device="cpu")
    engine._torch = torch
    engine._device = "cpu"
    engine._tracker = SimpleNamespace(prompt_encoder=SimpleNamespace(mask_input_size=(288, 288)))
    mask = np.ones((20, 30), dtype=bool)
    raw_logits = torch.arange(288 * 288, dtype=torch.float32).reshape(288, 288) / 10000
    engine._seed_cache[(("image",), _mask_digest(mask))] = raw_logits
    torch.testing.assert_close(engine._seed_logits(("image",), mask)[0, 0], raw_logits)


def test_caches_are_bounded_and_image_content_is_part_of_identity():
    cache = OrderedDict()
    for value in range(4):
        Sam3Engine._put(cache, value, value, limit=2)
    assert list(cache) == [2, 3]
    Sam3Engine._get(cache, 2)
    Sam3Engine._put(cache, 4, 4, limit=2)
    assert list(cache) == [2, 4]
    assert Sam3Engine._image_key(Image.new("RGB", (10, 10), "red"), "same") != Sam3Engine._image_key(Image.new("RGB", (10, 10), "blue"), "same")


def test_invalid_prompts_do_not_load_weights():
    engine = Sam3Engine()
    with pytest.raises(ValueError, match="positivo"):
        engine.predict_points(Image.new("RGB", (100, 100)), "image", {"points": [{"x": 1, "y": 1, "label": 0}]})
    with pytest.raises(ValueError, match="dentro"):
        engine.predict_points(Image.new("RGB", (100, 100)), "image", {"points": [{"x": 100, "y": 1, "label": 1}]})
    with pytest.raises(ValueError):
        engine.predict_text(Image.new("RGB", (100, 100)), "image", " ")
    assert engine.status()["state"] == "unloaded"


def test_mps_fallback_only_for_unsupported_operations():
    assert _is_mps_unsupported(NotImplementedError("aten::x is not implemented for the MPS device"))
    assert not _is_mps_unsupported(RuntimeError("MPS out of memory"))
    assert not _is_mps_unsupported(RuntimeError("tensor shapes do not match"))
    assert "hf_secret123" not in _safe_error(RuntimeError("403 hf_secret123"))


@pytest.mark.parametrize("seeded", [False, True])
def test_removing_click_discards_previous_logits_and_restores_original_seed(processor, monkeypatch, seeded):
    """A test double observes model inputs; no model-quality claim is made."""
    class RecordingTracker:
        prompt_encoder = SimpleNamespace(mask_input_size=(288, 288))

        def __init__(self):
            self.mask_inputs = []

        def __call__(self, **kwargs):
            self.mask_inputs.append(kwargs.get("input_masks"))
            return SimpleNamespace(
                iou_scores=torch.ones((1, 1, 1)),
                pred_masks=torch.full((1, 1, 1, 288, 288), float(len(self.mask_inputs))),
            )

    engine = Sam3Engine(device="cpu")
    engine._torch = torch
    engine._device = "cpu"
    engine._tracker_processor = processor
    engine._tracker = RecordingTracker()
    monkeypatch.setattr(engine, "_tracker_embeddings", lambda image, key: [])
    image = Image.new("RGB", (60, 20))
    key = ("test-image",)
    seed = np.ones((20, 60), dtype=bool) if seeded else None
    if seeded:
        engine._seed_cache[(key, _mask_digest(seed))] = torch.full((288, 288), 7.0)
    positive = ((10.0, 10.0, 1),)
    combined = positive + ((40.0, 10.0, 0),)
    part = {"id": "one-instance"}
    engine._predict_points(image, key, part, positive, None, seed)
    engine._predict_points(image, key, part, combined, None, seed)
    engine._predict_points(image, key, part, positive, None, seed)  # Undo negative click
    recorded = engine._tracker.mask_inputs
    assert torch.all(recorded[1] == 1.0)  # appended click uses the first model logits
    if seeded:
        assert torch.all(recorded[0] == 7.0)
        assert torch.all(recorded[2] == 7.0)  # undo starts again from the immutable text seed
    else:
        assert recorded[0] is None and recorded[2] is None


@pytest.mark.parametrize("part", [
    {"points": [None]}, {"points": [{"x": 1, "y": 2}]},
    {"points": [{"x": "x", "y": 2, "label": 1}]},
    {"points": [{"x": 1, "y": 2, "label": True}]}, {"box": 4},
])
def test_malformed_part_is_validation_error_not_model_failure(part):
    with pytest.raises(ValueError):
        Sam3Engine._validate_part(part, 60, 20)


@pytest.fixture
def text_contract_engine(monkeypatch):
    """Real SAM processor/config, local tokenizer, observed forward inputs only."""
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "shoe": 3, "[PAD]": 4}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 1), ("[EOS]", 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]",
                                       eos_token="[EOS]", pad_token="[PAD]", model_max_length=1024)

    class ObservedDetector:
        def __init__(self):
            self.config = Sam3Config()
            self.calls = []
            self.presence = torch.tensor([[10.0]])

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(pred_masks=torch.ones((1, 1, 2, 2)), pred_logits=torch.tensor([[10.0]]),
                                   pred_boxes=torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]), presence_logits=self.presence)

    engine = Sam3Engine(device="cpu")
    engine._torch = torch
    engine._device = "cpu"
    engine._detector_processor = Sam3Processor(image_processor=Sam3ImageProcessor(), tokenizer=tokenizer)
    engine._detector = ObservedDetector()
    embedding_calls = []
    monkeypatch.setattr(engine, "_detector_embeddings", lambda image, key: embedding_calls.append(key))
    return engine, embedding_calls


def test_long_text_rejected_before_embeddings_without_silent_truncation(text_contract_engine):
    engine, embedding_calls = text_contract_engine
    text = "shoe " * 31  # 155 characters, but 33 tokens including BOS/EOS.
    assert len(text) < 256
    encoded = engine._detector_processor(text=text, return_tensors="pt")
    assert encoded["input_ids"].shape[-1] == 33
    with pytest.raises(ValueError, match="máximo 32 tokens"):
        engine._predict_text(Image.new("RGB", (60, 20)), ("test",), text)
    assert embedding_calls == []
    assert engine._detector.calls == []
    assert not engine._seed_cache


@pytest.mark.parametrize("max_tokens,word_count", [(32, 30), (48, 46)])
def test_text_at_configured_token_limit_is_preserved(text_contract_engine, max_tokens, word_count):
    engine, embedding_calls = text_contract_engine
    engine._detector.config.text_config.max_position_embeddings = max_tokens
    proposals = engine._predict_text(Image.new("RGB", (60, 20)), ("test",), "shoe " * word_count)
    assert embedding_calls == [("test",)]
    assert engine._detector.calls[0]["input_ids"].shape[-1] == max_tokens
    assert len(proposals) == 1
    assert proposals[0]["mask"].shape == (20, 60)


@pytest.mark.parametrize("invalid_presence", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_presence_is_model_error_not_empty_detection(text_contract_engine, invalid_presence):
    engine, _ = text_contract_engine
    engine._detector.presence = torch.tensor([[invalid_presence]])
    with pytest.raises(ModelUnavailable, match="valores numéricos no válidos"):
        engine._predict_text(Image.new("RGB", (60, 20)), ("test",), "shoe")
    assert not engine._seed_cache


def test_absent_presence_logit_remains_supported(text_contract_engine):
    engine, _ = text_contract_engine
    engine._detector.presence = None
    proposals = engine._predict_text(Image.new("RGB", (60, 20)), ("test",), "shoe")
    assert len(proposals) == 1
