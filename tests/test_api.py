from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.geometry import encode_mask, decode_mask, mask_payload


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_store", None)
    images = tmp_path / "project" / "imágenes"
    images.mkdir(parents=True)
    Image.new("RGB", (64, 48), "white").save(images / "uno.png")
    with TestClient(main.app) as client:
        response = client.post("/api/projects/open", json={"directory": str(images.parent), "image_root": str(images)})
        assert response.status_code == 200, response.text
        yield client


def test_api_save_reopen_export_and_conflict(client):
    project = client.get("/api/project").json()
    image = project["images"][0]
    category = client.post("/api/project/categories", json={"name": "Vegetal", "color": "#668877"}).json()
    mask = np.zeros((48, 64), dtype=bool)
    mask[3:22, 4:31] = True
    mask[7:11, 9:16] = False
    state = client.get(f"/api/images/{image['id']}/state").json()
    state["annotations"] = [{"id": "qa-instance", "category_id": category["id"], "iscrowd": 0, **mask_payload(mask)}]
    saved = client.put(f"/api/images/{image['id']}/state", json=state)
    assert saved.status_code == 200, saved.text
    assert saved.json()["revision"] == state["revision"] + 1
    assert client.put(f"/api/images/{image['id']}/state", json=state).status_code == 409
    output = client.post("/api/coco/export", json={})
    assert output.status_code == 200, output.text
    coco = client.get("/api/coco/download").json()
    assert coco["images"][0]["file_name"] == "uno.png"
    assert np.array_equal(decode_mask(coco["annotations"][0]["segmentation"]), mask)
    assert Path(output.json()["path"]).parent == Path(project["directory"])
    assert not list(Path(project["image_root"]).glob("*.sqlite*"))


def test_missing_image_root_does_not_create_or_replace_project(client, tmp_path):
    original = client.get("/api/project").json()
    destination = tmp_path / "new-project"
    response = client.post("/api/projects/open", json={"directory": str(destination)})
    assert response.status_code == 422
    assert response.json()["detail"] == "Select an image folder to create a project."
    assert not destination.exists()
    assert client.get("/api/project").json() == original
    reopened = client.post("/api/projects/open", json={"directory": original["directory"]})
    assert reopened.status_code == 200, reopened.text
    assert reopened.json()["image_root"] == original["image_root"]


def test_geometry_and_union_keep_holes(client):
    image_id = client.get("/api/project").json()["images"][0]["id"]
    component = {"outer": [[1, 1], [30, 1], [30, 40], [1, 40]], "holes": [[[8, 8], [20, 8], [20, 20], [8, 20]]]}
    response = client.post("/api/geometry", json={"image_id": image_id, "components": [component]})
    assert response.status_code == 200, response.text
    mask = decode_mask(response.json()["mask"])
    assert mask[2, 2] and not mask[10, 10]
    island = np.zeros(mask.shape, dtype=bool)
    island[45, 60] = True
    combined = client.post("/api/masks/union", json={"image_id": image_id, "masks": [response.json()["mask"], encode_mask(island)]})
    assert combined.status_code == 200, combined.text
    result = decode_mask(combined.json()["mask"])
    assert result[45, 60] and not result[10, 10]


def test_geometry_preserves_manual_handles_and_rejects_empty_mask(client):
    image_id = client.get("/api/project").json()["images"][0]["id"]
    components = [{"outer": [[1.2, 1.2], [7, 1.2], [15.3, 1.2], [15.3, 15.3], [1.2, 15.3]], "holes": []}]
    response = client.post("/api/geometry", json={"image_id": image_id, "components": components})
    assert response.status_code == 200, response.text
    assert response.json()["components"] == components
    assert response.json()["controls"] == components
    tiny = [{"outer": [[0.01, 0.01], [0.1, 0.01], [0.1, 0.1], [0.01, 0.1]], "holes": []}]
    response = client.post("/api/geometry", json={"image_id": image_id, "components": tiny})
    assert response.status_code == 422
    response = client.post("/api/masks/union", json={"image_id": image_id, "masks": [encode_mask(np.zeros((48, 64), bool))]})
    assert response.status_code == 422


def test_hole_cleanup_is_explicit_lossless_outside_holes_and_does_not_save(client):
    image_id = client.get("/api/project").json()["images"][0]["id"]
    category = client.post("/api/project/categories", json={"name": "Object", "color": "#668877"}).json()
    mask = np.zeros((48, 64), dtype=bool)
    mask[3:35, 4:50] = True
    mask[8:12, 8:12] = False
    mask[18:23, 18:23] = False
    mask[45, 60] = True
    state_url = f"/api/images/{image_id}/state"
    state = client.get(state_url).json()
    state["annotations"] = [{"id": "with-holes", "category_id": category["id"], "iscrowd": 0, **mask_payload(mask)}]
    saved = client.put(state_url, json=state).json()
    response = client.post("/api/masks/fill-holes", json={"image_id": image_id, "mask": encode_mask(mask)})
    assert response.status_code == 200, response.text
    result = response.json()
    expected = mask.copy()
    expected[8:12, 8:12] = True
    assert (result["filled_holes"], result["filled_pixels"]) == (1, 16)
    assert np.array_equal(decode_mask(result["mask"]), expected)
    assert client.get(state_url).json() == saved

    # Only an explicit normal state save applies the returned mask to the project.
    geometry = {name: result[name] for name in ("mask", "components", "controls")}
    saved["annotations"][0].update(geometry)
    assert client.put(state_url, json=saved).status_code == 200
    assert client.post("/api/coco/export", json={}).status_code == 200
    coco = client.get("/api/coco/download").json()
    assert np.array_equal(decode_mask(coco["annotations"][0]["segmentation"]), expected)
    assert coco["annotations"][0]["area"] == int(expected.sum())
    again = client.post("/api/masks/fill-holes", json={"image_id": image_id, "mask": result["mask"]})
    assert again.status_code == 200
    assert (again.json()["filled_holes"], again.json()["filled_pixels"]) == (0, 0)
    assert np.array_equal(decode_mask(again.json()["mask"]), expected)


@pytest.mark.parametrize("limit", [0, -1, True, 1.0, "16", 150_000_001])
def test_hole_cleanup_api_rejects_nonpositive_or_noninteger_limits(client, limit):
    before = client.get("/api/images/1/state").json()
    response = client.post("/api/masks/fill-holes", json={
        "image_id": 1, "mask": encode_mask(np.ones((48, 64), dtype=bool)), "max_area": limit,
    })
    assert response.status_code == 422
    assert client.get("/api/images/1/state").json() == before


def test_hole_cleanup_api_rejects_empty_or_mismatched_masks(client):
    for mask in (np.ones((64, 48), dtype=bool), np.zeros((48, 64), dtype=bool)):
        response = client.post("/api/masks/fill-holes", json={"image_id": 1, "mask": encode_mask(mask)})
        assert response.status_code == 422


def test_crossed_polygon_can_autosave_but_cannot_refine_until_fixed(client):
    image_id = client.get("/api/project").json()["images"][0]["id"]
    category = client.post("/api/project/categories", json={"name": "Región", "color": "#668877"}).json()
    state_url = f"/api/images/{image_id}/state"
    state = client.get(state_url).json()
    polygon = {"vertices": [[1, 1], [30, 40], [30, 1], [1, 40]], "closed": True}
    state["draft"] = {"id": "polygon-draft", "category_id": category["id"], "active_part_id": "part",
                      "parts": [{"id": "part", "points": [], "polygon": polygon}]}
    saved = client.put(state_url, json=state)
    assert saved.status_code == 200, saved.text
    rejected = client.post("/api/geometry", json={"image_id": image_id,
                           "components": [{"outer": polygon["vertices"], "holes": []}]})
    assert rejected.status_code == 422
    assert client.get(state_url).json() == saved.json()
    assert saved.json()["draft"]["parts"][0]["polygon"] == polygon

    repaired = saved.json()
    part = repaired["draft"]["parts"][0]
    part["polygon"]["vertices"] = [[1, 1], [30, 1], [30, 40], [1, 40]]
    response = client.post("/api/geometry", json={"image_id": image_id,
                           "components": [{"outer": part["polygon"]["vertices"], "holes": []}]})
    assert response.status_code == 200, response.text
    part["seed_mask"] = response.json()["mask"]
    saved = client.put(state_url, json=repaired)
    assert saved.status_code == 200, saved.text
    restored_part = client.get(state_url).json()["draft"]["parts"][0]
    assert restored_part == part
    assert decode_mask(restored_part["seed_mask"]).sum() == 29 * 39
    assert "mask" not in restored_part  # Rasterizing an input is not a SAM result.


@pytest.mark.parametrize("text,language,english", [
    ("hojas de planta", "es", "plant leaves"),
    ("plant leaves", None, "plant leaves"),
])
def test_text_prompt_language_and_translation_metadata(client, monkeypatch, text, language, english):
    image_id = client.get("/api/project").json()["images"][0]["id"]
    category = client.post("/api/project/categories", json={"name": "Hojas", "color": "#668877"}).json()
    translated = []
    model_calls = []

    def translate(value, source_language):
        translated.append((value, source_language))
        return english

    def predict(image, key, prompt):
        model_calls.append((image.size, prompt))
        mask = np.zeros((48, 64), dtype=bool)
        mask[4:20, 5:30] = True
        return [{"mask": mask, "score": 0.95}]

    monkeypatch.setattr(main.translator, "translate", translate)
    monkeypatch.setattr(main.engine, "predict_text", predict)
    body = {"image_id": image_id, "revision": 7, "text": text, "category_id": category["id"]}
    if language is not None:
        body["source_language"] = language
    response = client.post("/api/infer/text", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert translated == [(text, language or "en")]
    assert model_calls == [((64, 48), english)]
    assert result["prompt"] == {"original": text, "english": english, "source_language": language or "en"}
    assert result["image_id"] == image_id and result["revision"] == 7
    assert result["proposals"][0]["category_id"] == category["id"]
    assert client.get("/api/project").json()["categories"][0]["name"] == "Hojas"
    assert client.get(f"/api/images/{image_id}/state").json()["proposals"] == []


def test_unsupported_prompt_language_rejected_before_translation_or_sam(client, monkeypatch):
    def unexpected(*args):
        pytest.fail("Invalid language must be rejected before model calls")

    monkeypatch.setattr(main.translator, "translate", unexpected)
    monkeypatch.setattr(main.engine, "predict_text", unexpected)
    response = client.post("/api/infer/text", json={
        "image_id": 1, "revision": 0, "text": "feuilles", "category_id": 1, "source_language": "fr",
    })
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "source_language"]


def test_translation_failure_does_not_call_sam_or_change_saved_state(client, monkeypatch):
    image_id = client.get("/api/project").json()["images"][0]["id"]
    category = client.post("/api/project/categories", json={"name": "Leaves", "color": "#668877"}).json()
    state_url = f"/api/images/{image_id}/state"
    before = client.get(state_url).json()

    def translate(*args):
        raise main.TranslationUnavailable("Translation model is unavailable.")

    def unexpected(*args):
        pytest.fail("SAM must not run when translation fails")

    monkeypatch.setattr(main.translator, "translate", translate)
    monkeypatch.setattr(main.engine, "predict_text", unexpected)
    response = client.post("/api/infer/text", json={
        "image_id": image_id, "revision": 0, "text": "hojas", "category_id": category["id"],
        "source_language": "es",
    })
    assert response.status_code == 503
    assert response.json()["detail"] == "Prompt translation failed: Translation model is unavailable."
    assert client.get(state_url).json() == before


def test_local_origin_and_host_protection(client):
    assert client.get("/api/health", headers={"Origin": "https://attacker.example"}).status_code == 403
    assert client.get("/api/health", headers={"Host": "attacker.example"}).status_code == 400
    assert client.get("/api/health", headers={"Origin": "http://127.0.0.1:8765"}).status_code == 200


def test_relink_requires_exact_relative_paths(client, tmp_path):
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    Image.new("RGB", (12, 12), "white").save(wrong / "uno.png")
    before = client.get("/api/project").json()["image_root"]
    response = client.post("/api/project/relink", json={"image_root": str(wrong)})
    assert response.status_code == 422
    assert client.get("/api/project").json()["image_root"] == before


def test_pixel_orientation_and_thumbnail(client):
    image_id = client.get("/api/project").json()["images"][0]["id"]
    response = client.get(f"/api/images/{image_id}/file?thumbnail=true")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_old_tab_cannot_write_or_download_other_project(client, tmp_path):
    first = client.get("/api/project").json()
    first_state = client.get(f"/api/images/{first['images'][0]['id']}/state").json()
    second_images = tmp_path / "second" / "images"
    second_images.mkdir(parents=True)
    Image.new("RGB", (64, 48), "white").save(second_images / "uno.png")
    assert client.post("/api/projects/open", json={
        "directory": str(second_images.parent), "image_root": str(second_images)
    }).status_code == 200
    stale_headers = {"X-Project-Directory": first["directory"]}
    assert client.put("/api/images/1/state", json=first_state, headers=stale_headers).status_code == 409
    assert client.get("/api/coco/download", params={"project": first["directory"]}).status_code == 409


def test_request_keeps_validated_project_when_global_changes(client, tmp_path, monkeypatch):
    first = main._store
    second_images = tmp_path / "second" / "images"
    second_images.mkdir(parents=True)
    Image.new("RGB", (64, 48), "white").save(second_images / "uno.png")
    second = main.ProjectStore.open(str(second_images.parent), str(second_images))
    original_store = main.store

    def switch_after_middleware(request):
        # Deterministically reproduce switching between middleware validation and
        # the handler's first access to its store, without timing-based sleeps.
        assert request.state.project is first
        main._store = second
        return original_store(request)

    monkeypatch.setattr(main, "store", switch_after_middleware)
    response = client.post("/api/project/categories", headers={"X-Project-Directory": str(first.directory)},
                           json={"name": "First project only", "color": "#668877"})
    assert response.status_code == 200, response.text
    assert any(c["name"] == "First project only" for c in first.project()["categories"])
    assert not any(c["name"] == "First project only" for c in second.project()["categories"])
