"""Visual-reference transport contracts only; test doubles do not prove quality."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, Sam3Config, Sam3ImageProcessor, Sam3Processor

from app.inference import ModelUnavailable, Sam3Engine, _mask_digest


def test_combined_tokens_equal_native_box_branch_without_mutating_inputs():
    text = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    geometry = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8) + 50
    text_mask = torch.tensor([[1, 1, 0]])
    geometry_mask = torch.tensor([[True, False]])
    combined, mask = Sam3Engine._combine_concept_prompts(text, text_mask, geometry, geometry_mask)
    torch.testing.assert_close(combined.pooler_output, torch.cat([text, geometry], dim=1))
    assert mask.tolist() == [[True, True, False, True, False]]
    assert text.shape == (1, 3, 8) and geometry.shape == (1, 2, 8)
    no_masks, valid = Sam3Engine._combine_concept_prompts(text, None, geometry, None)
    torch.testing.assert_close(no_masks.pooler_output, combined.pooler_output)
    assert valid.all()


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [-1, 0, 3, 3], [0, 0, 21, 5],
                                 [0, 0, float("nan"), 2], [0, 0, float("inf"), 2],
                                 [True, 0, 4, 4], [0, 0, 4], "0,0,4,4"])
def test_bad_reference_box_rejected_before_loading(box):
    engine = Sam3Engine()
    with pytest.raises(ValueError):
        engine.predict_visual(Image.new("RGB", (5, 5)), "target", Image.new("RGB", (20, 10)), box)
    assert engine.status()["state"] == "unloaded"


@pytest.mark.parametrize("text", [12, "x" * 257])
def test_bad_visual_text_rejected_before_loading(text):
    engine = Sam3Engine()
    with pytest.raises(ValueError):
        engine.predict_visual(Image.new("RGB", (5, 5)), "target", Image.new("RGB", (20, 10)), text=text)
    assert engine.status()["state"] == "unloaded"


def test_invalid_prompt_tokens_fail_instead_of_empty_detection():
    with pytest.raises(ModelUnavailable, match="numeric"):
        Sam3Engine._combine_concept_prompts(torch.ones(1, 3, 4), None, torch.full((1, 2, 4), float("nan")), None)
    with pytest.raises(ModelUnavailable, match="attention mask"):
        Sam3Engine._combine_concept_prompts(torch.ones(1, 3, 4), torch.ones(1, 4), torch.ones(1, 2, 4), None)


@pytest.fixture
def visual_engine(monkeypatch):
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "visual": 3, "[PAD]": 4}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 1), ("[EOS]", 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]",
                                       eos_token="[EOS]", pad_token="[PAD]", model_max_length=1024)

    class ObservedDetector:
        config = Sam3Config()

        def __init__(self):
            self.geometry_calls, self.forward_calls, self.vision_calls = [], [], []

        def get_vision_features(self, pixels):
            self.vision_calls.append(pixels)
            feature = pixels[:, 0].mean().reshape(1, 1, 1, 1).expand(1, 256, 2, 2)
            return SimpleNamespace(fpn_hidden_states=(feature, feature, feature, feature),
                                   fpn_position_encoding=(feature, feature, feature, feature))

        def geometry_encoder(self, **kwargs):
            self.geometry_calls.append(kwargs)
            return SimpleNamespace(last_hidden_state=torch.ones(1, 2, 256), attention_mask=torch.ones(1, 2, dtype=torch.bool))

        def get_text_features(self, **kwargs):
            return SimpleNamespace(pooler_output=kwargs["input_ids"].float().unsqueeze(-1).expand(-1, -1, 256))

        def __call__(self, **kwargs):
            self.forward_calls.append(kwargs)
            return SimpleNamespace(pred_masks=torch.ones((1, 1, 2, 2)), pred_logits=torch.tensor([[10.0]]),
                                   pred_boxes=torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]), presence_logits=torch.tensor([[10.0]]))

    engine = Sam3Engine(device="cpu")
    engine._torch, engine._device = torch, "cpu"
    engine._detector_processor = Sam3Processor(image_processor=Sam3ImageProcessor(size={"height": 28, "width": 28}), tokenizer=tokenizer)
    engine._detector = ObservedDetector()
    monkeypatch.setattr(engine, "_execute", lambda operation: operation())
    return engine


def test_external_reference_uses_reference_geometry_and_target_logits(visual_engine):
    engine = visual_engine
    target = Image.new("RGB", (60, 20), "blue")
    reference = Image.new("RGB", (20, 10), "red")
    proposals = engine.predict_visual(target, "target", reference, [2, 1, 18, 9])
    geometry = engine._detector.geometry_calls[0]
    torch.testing.assert_close(geometry["box_embeddings"], torch.tensor([[[0.5, 0.5, 0.8, 0.8]]]))
    assert len(geometry["img_feats"]) == 3
    assert geometry["box_labels"].tolist() == [[1]]
    forward = engine._detector.forward_calls[0]
    assert set(forward) == {"vision_embeds", "text_embeds", "attention_mask"}
    assert forward["text_embeds"].pooler_output.shape == (1, 34, 256)
    assert not torch.equal(geometry["img_feats"][0], forward["vision_embeds"].fpn_hidden_states[0])
    assert proposals[0]["mask"].shape == (20, 60)
    target_key = engine._image_key(target, "target")
    assert (target_key, _mask_digest(proposals[0]["mask"])) in engine._seed_cache
    assert all(key[0] == target_key for key in engine._seed_cache)
    assert {key[0] for key in engine._detector_cache} == {"target", "reference"}


def test_reference_tokens_reused_only_for_identical_pixels_and_box(visual_engine):
    engine = visual_engine
    reference = Image.new("RGB", (20, 10), "red")
    target = Image.new("RGB", (60, 20), "blue")
    engine.predict_visual(target, "session-A", reference)
    engine.predict_visual(target, "session-B", reference, text=" ")
    assert len(engine._detector.geometry_calls) == 1
    assert len(engine._seed_cache) == 2  # Separate target sessions retain their own seeds.
    engine.predict_visual(target, "session-A", reference, [1, 1, 19, 9])
    engine.predict_visual(target, "session-A", Image.new("RGB", (20, 10), "green"))
    assert len(engine._detector.geometry_calls) == 3
    assert len(engine._detector_cache) <= engine.cache_size
    engine._clear_caches()
    assert not engine._reference_cache and not engine._seed_cache and not engine._detector_cache


def test_long_visual_text_rejected_before_reference_encoding(visual_engine):
    engine = visual_engine
    with pytest.raises(ValueError, match="maximum 32 tokens"):
        engine.predict_visual(Image.new("RGB", (5, 5)), "target", Image.new("RGB", (20, 10)), text="visual " * 31)
    assert not engine._detector.vision_calls
    assert not engine._detector.geometry_calls
    assert not engine._detector.forward_calls
