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
