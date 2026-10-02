"""Visual upload contracts, with model calls observed rather than simulated quality."""

import asyncio
import base64
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.geometry import decode_mask


def image_bytes(format="PNG", size=(20, 12), **kwargs):
    buffer = io.BytesIO()
    Image.new("RGB", size, "#448866").save(buffer, format=format, **kwargs)
    return buffer.getvalue()


@pytest.fixture
def visual_client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_store", None)
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (64, 48), "white").save(images / "target.png")
    with TestClient(main.app) as client:
        opened = client.post("/api/projects/open", json={
            "directory": str(tmp_path / "project"), "image_root": str(images),
        })
        assert opened.status_code == 200, opened.text
        category = client.post("/api/project/categories", json={"name": "Leaves"}).json()
        body = {"image_id": opened.json()["images"][0]["id"], "revision": 7,
                "category_id": category["id"], "reference_image": base64.b64encode(image_bytes()).decode("ascii")}
        yield client, body


def unexpected(*args, **kwargs):
    pytest.fail("Invalid input must be rejected before translation or inference")


def reject_model_calls(monkeypatch):
    monkeypatch.setattr(main.translator, "translate", unexpected)
    monkeypatch.setattr(main.engine, "predict_visual", unexpected, raising=False)
    monkeypatch.setattr(main.engine, "predict_text", unexpected)


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP"])
def test_reference_only_returns_target_proposals_without_persisting(visual_client, monkeypatch, format):
    client, body = visual_client
    body["reference_image"] = base64.b64encode(image_bytes(format)).decode("ascii")
    body["text"] = "  "
    before = client.get(f"/api/images/{body['image_id']}/state").json()
    project = client.get("/api/project").json()
    project_files = {p.relative_to(project["directory"]) for p in Path(project["directory"]).rglob("*")}
    calls = []

    def predict(image, key, reference, reference_box=None, text=None):
        calls.append((image.size, reference.size, reference.mode, reference_box, text, key))
        mask = np.zeros((48, 64), dtype=bool)
        mask[5:25, 10:35] = True
        return [{"mask": mask, "score": 0.93}]

    monkeypatch.setattr(main.translator, "translate", unexpected)
    monkeypatch.setattr(main.engine, "predict_visual", predict, raising=False)
    monkeypatch.setattr(main.engine, "predict_text", unexpected)
    response = client.post("/api/infer/visual", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert calls[0][:5] == ((64, 48), (20, 12), "RGB", None, None)
    assert project["directory"] in calls[0][5]
    assert result["image_id"] == body["image_id"] and result["revision"] == 7
    assert result["reference"] == {"width": 20, "height": 12}
    assert result["method"] == "cross_image_exemplar" and "prompt" not in result
    proposal = result["proposals"][0]
    assert proposal["category_id"] == body["category_id"] and proposal["iscrowd"] == 0
    assert proposal["selected"] is True and proposal["score"] == 0.93
    assert decode_mask(proposal["mask"]).shape == (48, 64)
    assert decode_mask(proposal["mask"]).sum() == 500
    assert client.get(f"/api/images/{body['image_id']}/state").json() == before
    assert {p.relative_to(project["directory"]) for p in Path(project["directory"]).rglob("*")} == project_files
    assert [p.name for p in Path(project["image_root"]).iterdir()] == ["target.png"]


def test_reference_with_spanish_text_translates_and_uses_oriented_crop(visual_client, monkeypatch):
    client, body = visual_client
    exif = Image.Exif()
    exif[274] = 6
    body.update(reference_image=base64.b64encode(image_bytes("JPEG", (64, 32), exif=exif)).decode("ascii"),
                reference_box=[1, 40, 20, 60], text="hojas", source_language="es")
    translations, predictions = [], []

    def translate(text, language):
        translations.append((text, language))
        return "leaves"

    def predict(image, key, reference, reference_box=None, text=None):
        predictions.append((reference.size, reference_box, text))
        return []

    monkeypatch.setattr(main.translator, "translate", translate)
    monkeypatch.setattr(main.engine, "predict_visual", predict, raising=False)
    response = client.post("/api/infer/visual", json=body)
    assert response.status_code == 200, response.text
    assert translations == [("hojas", "es")]
    assert predictions == [((32, 64), [1.0, 40.0, 20.0, 60.0], "leaves")]
    assert response.json()["reference"] == {"width": 32, "height": 64}
    assert response.json()["prompt"] == {"original": "hojas", "english": "leaves", "source_language": "es"}


@pytest.mark.parametrize("encoded", ["", "not base64!", "data:image/png;base64,eA==", "ñ", "eA=="])
def test_invalid_base64_or_image_rejected_before_models(visual_client, monkeypatch, encoded):
    client, body = visual_client
    reject_model_calls(monkeypatch)
    body["reference_image"] = encoded
    assert client.post("/api/infer/visual", json=body).status_code == 422


@pytest.mark.parametrize("variant", ["gif", "truncated", "animated"])
def test_unsupported_damaged_or_animated_images_rejected(visual_client, monkeypatch, variant):
    client, body = visual_client
    reject_model_calls(monkeypatch)
    if variant == "gif":
        raw = image_bytes("GIF")
    elif variant == "truncated":
        raw = image_bytes()[:40]
    else:
        buffer = io.BytesIO()
        Image.new("RGB", (20, 12), "red").save(buffer, format="PNG", save_all=True,
                                              append_images=[Image.new("RGB", (20, 12), "blue")], duration=100)
        raw = buffer.getvalue()
    body["reference_image"] = base64.b64encode(raw).decode("ascii")
    assert client.post("/api/infer/visual", json=body).status_code == 422


@pytest.mark.parametrize("box", [[True, 0, 10, 10], [0, "0", 10, 10], [0, 0, 0, 10],
                                 [0, 0, 30, 10], [-1, 0, 10, 10], [0, 0, 10], [0, 0, None, 10]])
def test_invalid_reference_boxes_rejected_before_models(visual_client, monkeypatch, box):
    client, body = visual_client
    reject_model_calls(monkeypatch)
    body["reference_box"] = box
    assert client.post("/api/infer/visual", json=body).status_code == 422


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_reference_box_rejected_by_decoder(number):
    with pytest.raises(ValueError, match="finite"):
        main.decode_reference_image(base64.b64encode(image_bytes()).decode("ascii"), [0, 0, number, 10])


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
def test_transparent_references_are_composited_on_white(mode):
    image = Image.new(mode, (3, 1))
    if mode == "RGBA":
        image.putdata([(255, 0, 0, 255), (0, 0, 255, 0), (0, 0, 255, 128)])
        expected = [(255, 0, 0), (255, 255, 255), (127, 127, 255)]
    elif mode == "LA":
        image.putdata([(100, 255), (0, 0), (0, 128)])
        expected = [(100, 100, 100), (255, 255, 255), (127, 127, 127)]
    else:
        image.putpalette([255, 0, 0, 0, 0, 255, 0, 255, 0] + [0] * (768 - 9))
        image.putdata([0, 1, 2])
        image.info["transparency"] = bytes([255, 0, 128])
        expected = [(255, 0, 0), (255, 255, 255), (127, 255, 127)]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    reference, _ = main.decode_reference_image(base64.b64encode(buffer.getvalue()).decode("ascii"), None)
    assert reference.mode == "RGB"
    assert [reference.getpixel((x, 0)) for x in range(3)] == expected


@pytest.mark.parametrize("limit", ["bytes", "pixels", "body"])
def test_upload_limits_reject_before_models(visual_client, monkeypatch, limit):
    client, body = visual_client
    reject_model_calls(monkeypatch)
    if limit == "bytes":
        monkeypatch.setattr(main, "MAX_REFERENCE_BYTES", 10)
    elif limit == "pixels":
        monkeypatch.setattr(main, "MAX_REFERENCE_PIXELS", 100)
    else:
        monkeypatch.setattr(main, "MAX_VISUAL_REQUEST_BYTES", 20)
    response = client.post("/api/infer/visual", json=body)
    assert response.status_code == 413, response.text


def test_streaming_body_limit_does_not_trust_content_length(monkeypatch):
    monkeypatch.setattr(main, "MAX_VISUAL_REQUEST_BYTES", 8)
    messages = iter([{"type": "http.request", "body": b"12345", "more_body": True},
                     {"type": "http.request", "body": b"67890", "more_body": False}])
    sent = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    middleware = main.VisualUploadLimitMiddleware(unexpected)
    asyncio.run(middleware({"type": "http", "method": "POST", "path": "/api/infer/visual", "headers": []},
                           receive, send))
    assert sent[0]["status"] == 413


@pytest.mark.parametrize("field,value,status", [("category_id", 999, 422), ("image_id", 999, 404),
                                                 ("source_language", "fr", 422), ("revision", -1, 422),
                                                 ("category_id", True, 422)])
def test_project_inputs_checked_before_reference_and_models(visual_client, monkeypatch, field, value, status):
    client, body = visual_client
    reject_model_calls(monkeypatch)
    monkeypatch.setattr(main, "decode_reference_image", unexpected)
    body[field] = value
    assert client.post("/api/infer/visual", json=body).status_code == status


@pytest.mark.parametrize("failure", ["translation", "model"])
def test_failures_do_not_fallback_or_modify_project(visual_client, monkeypatch, failure):
    client, body = visual_client
    state_url = f"/api/images/{body['image_id']}/state"
    before = client.get(state_url).json()
    monkeypatch.setattr(main.engine, "predict_text", unexpected)

    def unavailable(*args, **kwargs):
        if failure == "translation":
            raise main.TranslationUnavailable("Unavailable")
        raise RuntimeError("Unavailable")

    if failure == "translation":
        body.update(text="hojas", source_language="es")
        monkeypatch.setattr(main.translator, "translate", unavailable)
        monkeypatch.setattr(main.engine, "predict_visual", unexpected, raising=False)
    else:
        monkeypatch.setattr(main.translator, "translate", unexpected)
        monkeypatch.setattr(main.engine, "predict_visual", unavailable, raising=False)
    assert client.post("/api/infer/visual", json=body).status_code == 503
    assert client.get(state_url).json() == before


def test_visual_request_keeps_project_guard(visual_client, monkeypatch):
    client, body = visual_client
    reject_model_calls(monkeypatch)
    response = client.post("/api/infer/visual", json=body, headers={"X-Project-Directory": "/different/project"})
    assert response.status_code == 409
