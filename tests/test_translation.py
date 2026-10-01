from unittest.mock import MagicMock

import pytest
import torch

from app.translation import PromptTranslator, TranslationUnavailable


def test_english_never_loads_a_model(monkeypatch):
    translator = PromptTranslator()
    load = MagicMock(side_effect=AssertionError("English must not load translation weights"))
    monkeypatch.setattr(translator, "_load", load)
    assert translator.translate("  carrots  ") == "carrots"
    load.assert_not_called()


@pytest.mark.parametrize("text,language", [("", "es"), ("   ", "en"), ("x" * 301, "es"), ("carrots", "fr")])
def test_invalid_prompts_fail_before_loading(text, language, monkeypatch):
    translator = PromptTranslator()
    load = MagicMock()
    monkeypatch.setattr(translator, "_load", load)
    with pytest.raises(ValueError):
        translator.translate(text, language)
    load.assert_not_called()


def configured_translator(monkeypatch):
    translator = PromptTranslator()
    monkeypatch.setattr(translator, "_load", MagicMock())
    translator._torch = torch
    translator._tokenizer = MagicMock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})
    translator._tokenizer.decode.return_value = "carrots"
    translator._model = MagicMock()
    translator._model.generate.return_value = torch.tensor([[3, 2, 1]])
    return translator


def test_spanish_generation_is_bounded_and_cached_without_changing_english(monkeypatch):
    translator = configured_translator(monkeypatch)
    assert translator.translate("zanahorias", "es") == "carrots"
    assert translator.translate(" zanahorias ", "es") == "carrots"
    translator._model.generate.assert_called_once()
    assert translator._model.generate.call_args.kwargs["max_new_tokens"] == 128
    assert translator._model.generate.call_args.kwargs["do_sample"] is False
    assert translator.translate("zanahorias", "en") == "zanahorias"


def test_token_overflow_is_rejected_without_truncation(monkeypatch):
    translator = configured_translator(monkeypatch)
    translator._tokenizer.return_value = {"input_ids": torch.ones((1, 129), dtype=torch.long)}
    with pytest.raises(ValueError, match="too long"):
        translator.translate("text", "es")
    translator._model.generate.assert_not_called()
    assert translator._tokenizer.call_args.kwargs["truncation"] is False


def test_failed_download_is_actionable_and_does_not_cache_failure(monkeypatch):
    translator = PromptTranslator()
    load = MagicMock(side_effect=RuntimeError("private URL hf_notforoutput"))
    monkeypatch.setattr(translator, "_load", load)
    with pytest.raises(TranslationUnavailable, match="select English") as error:
        translator.translate("zanahorias", "es")
    assert "hf_" not in str(error.value)
    assert not translator._cache
    assert translator.translate("carrots", "en") == "carrots"


def test_empty_translation_does_not_reach_segmentation(monkeypatch):
    translator = configured_translator(monkeypatch)
    translator._tokenizer.decode.return_value = " "
    with pytest.raises(TranslationUnavailable, match="no text"):
        translator.translate("zanahorias", "es")
    assert not translator._cache
