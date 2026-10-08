"""Mixed task persistence and export contracts, using disposable local/hosted projects."""
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
import pytest

from app.geometry import decode_mask, mask_payload
from app.storage import ProjectStore, RevisionConflict
from app import yolo
from app.hosted import quota
from test_api import client  # noqa: F401
from test_hosted_backend import hosted, project_and_image  # noqa: F401
from test_hosted_quota import assert_accounted
from test_storage import project  # noqa: F401


def box(category=1, identity="box", coords=None):
    return {"id": identity, "kind": "bbox", "category_id": category, "iscrowd": 0, "bbox": coords or [2, 1, 8, 6]}


def mixed(state, category, width, height):
    mask = np.zeros((height, width), dtype=bool)
    mask[1:7, 2:10] = True
    mask[3:5, 4:6] = False
    mask[height-1, width-1] = True
    state = {**state, "schema_version": 2}
    state["annotations"] = [box(category), {"id": "seg", "category_id": category, "iscrowd": 0, **mask_payload(mask)}]
    state["draft"] = {"id": "seg-draft", "category_id": category, "parts": [{"id": "part", "points": []}], "active_part_id": "part"}
    state["detection"] = {
        "draft": {"id": "box-draft", "category_id": category, "bbox": [0, 0, 2, 2], "points": []},
        "proposals": [{**box(category, "candidate"), "score": .9, "selected": True}],
        "adjustment": {"target_id": "box", "base_bbox": [2, 1, 8, 6], "bbox": [3, 2, 6, 4]},
    }
    return state, mask


@pytest.mark.parametrize("mode", ["local", "hosted"])
def test_mixed_roundtrip_exports_private_downloads_and_legacy_guard(request, mode, tmp_path):
    if mode == "local":
        api = request.getfixturevalue("client")
        root, categories = "/api", "/api/project/categories"
        document = api.get(root + "/project").json()
    else:
        settings, db, _, _, clients = request.getfixturevalue("hosted")
        api = clients["alice"]
        document = project_and_image(api)
        root = f"/api/v1/projects/{document['id']}"
        categories = root + "/categories"
    category = api.post(categories, json={"name": "Object"}).json()["id"]
    state_url = root + "/images/1/state"
    initial = api.get(state_url).json()
    image = document["images"][0]
    candidate, mask = mixed(initial, category, image["width"], image["height"])
    headers = {"X-Annotation-State-Version": "2"}
    saved = api.put(state_url, json=candidate, headers=headers)
    assert saved.status_code == 200, saved.text
    saved = saved.json()
    assert saved["annotations"][0] == box(category)
    assert saved["detection"]["adjustment"] == candidate["detection"]["adjustment"]
    assert np.array_equal(decode_mask(saved["annotations"][1]["mask"]), mask)
    assert api.get(state_url).status_code == 422  # old client cannot consume mixed state
    assert api.put(state_url, json=initial).status_code == 422
    assert api.put(state_url, json=candidate, headers=headers).status_code == 409
    assert api.get(state_url, headers=headers).json() == saved
    api.headers.update(headers)

    # Legacy endpoint still includes both kinds; task exports filter only confirmed instances.
    assert api.post(root + "/coco/export", json={}).status_code == 200
    coco = api.get(root + "/coco/download").json()
    assert len(coco["annotations"]) == 2
    exported_box = next(a for a in coco["annotations"] if "segmentation" not in a)
    assert exported_box["bbox"] == [2, 1, 8, 6] and exported_box["area"] == 48
    coco_ids = [a["id"] for a in coco["annotations"]]
    unique = api.post(root + "/coco/export", json={"task": "detection", "unique": True}).json()
    unique_url = root + "/coco/download/" + unique["export_id"]
    assert len(api.get(unique_url).json()["annotations"]) == 1

    options = {"task": "detection", "include_images": True}
    preview = api.post(root + "/yolo/preview", json=options).json()
    assert preview["annotation_count"] == 1 and not preview["warnings"]
    exported = api.post(root + "/yolo/export", json={**options, "snapshot": preview["snapshot"]})
    assert exported.status_code == 200, exported.text
    download = root + "/yolo/download/" + exported.json()["export_id"]
    response = api.get(download)
    assert response.status_code == 200 and "no-store" in response.headers["cache-control"]
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        label = next(n for n in archive.namelist() if n.startswith("labels/"))
        numbers = list(map(float, archive.read(label).split()))
        assert numbers == pytest.approx([0, 6/image["width"], 4/image["height"], 8/image["width"], 6/image["height"]])
        png = next(n for n in archive.namelist() if n.startswith("images/"))
        with Image.open(io.BytesIO(archive.read(png))) as output:
            assert output.size == (image["width"], image["height"])
            assert not output.getexif()

    # A mixed COCO import retains stable COCO IDs and both geometry kinds.
    if mode == "local":
        source = tmp_path / "mixed.json"
        source.write_text(json.dumps(coco))
        imported = ProjectStore.import_coco(tmp_path / "roundtrip", document["image_root"], source)
        again = json.loads(imported.export_coco().read_text())
    else:
        target = project_and_image(api)
        target_root = f"/api/v1/projects/{target['id']}"
        imported = api.post(target_root + "/coco/import", files={"file": ("mixed.json", json.dumps(coco))})
        assert imported.status_code == 200, imported.text
        assert api.post(target_root + "/coco/export").status_code == 200
        again = api.get(target_root + "/coco/download").json()
        assert clients["bob"].get(download).status_code == 404
        assert clients["bob"].get(unique_url).status_code == 404
        assert clients["bob"].post(root + "/yolo/preview", json=options).status_code == 404
        assert api.post(root + "/yolo/export", json={**options, "snapshot": preview["snapshot"]}, headers={"X-CSRF-Token": "wrong"}).status_code == 403
        assert_accounted(settings, db)
    assert [a["id"] for a in again["annotations"]] == coco_ids
    assert next(a for a in again["annotations"] if "segmentation" not in a)["bbox"] == exported_box["bbox"]
    assert api.get(state_url).json() == saved  # exporting never applies pending adjustments


@pytest.mark.parametrize("bad", [[0, 0, 0, 1], [-1, 0, 2, 2], [0, 0, 13, 1], [0, 0, True, 1], [0, 0, float('inf'), 1]])
def test_box_validation_is_atomic(project, bad):
    original = project.get_state(1)
    candidate = {**original, "schema_version": 2, "annotations": [box(coords=bad)]}
    with pytest.raises(ValueError):
        project.save_state(1, candidate)
    assert project.get_state(1) == original
    project.require_client(None)  # rejected writes did not migrate the project


def test_yolo_snapshot_approximations_empty_images_and_crowds(project, tmp_path):
    state, mask = mixed(project.get_state(1), 1, 12, 8)
    saved = project.save_state(1, state)
    options = yolo.ExportOptions(task="segmentation", include_empty=True)
    preview = yolo.prepare(project, options)
    assert preview["image_count"] == 2 and preview["annotation_count"] == 1
    assert "holes filled" in preview["warnings"][0]["reason"]
    assert "separate parts connected" in preview["warnings"][0]["reason"]
    request = yolo.ExportRequest(**options.model_dump(), snapshot=preview["snapshot"])
    with pytest.raises(ValueError, match="accept"):
        yolo.generate(project, request, tmp_path / "exports", project.image_path)
    request.allow_lossy = True
    _, output = yolo.generate(project, request, tmp_path / "exports", project.image_path)
    with zipfile.ZipFile(output) as archive:
        assert archive.read("labels/sub/b.txt") == b""
        values = list(map(float, archive.read("labels/a.txt").split()))
        assert values[0] == 0 and all(0 <= n <= 1 for n in values[1:])
        assert len(values) >= 7 and len(values) % 2 == 1
        assert len(archive.read("labels/a.txt").decode().splitlines()) == 1
        assert not any(n.startswith("images/") for n in archive.namelist())
        # Independent COCO polygon rasterizer reads one connected polygon.
        points = np.array(values[1:]).reshape(-1, 2) * [12, 8]
        encoded = coco_mask.frPyObjects([points.flatten().tolist()], 8, 12)
        recovered = coco_mask.decode(coco_mask.merge(encoded))
        assert recovered[3, 4]  # documented hole approximation
        assert recovered[7, 11]  # separate island retained
    assert np.array_equal(decode_mask(project.get_state(1)["annotations"][1]["mask"]), mask)
    project.update_category(1, {"name": "Renamed"})
    with pytest.raises(RevisionConflict):
        yolo.generate(project, request, tmp_path / "exports", project.image_path)
    saved["annotations"][1]["iscrowd"] = 1
    project.save_state(1, saved)
    assert yolo.prepare(project, options)["errors"][0]["annotation_id"] == "seg"


def test_yolo_class_mapping_filename_collisions_and_exif(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    exif = Image.Exif(); exif[274] = 6
    Image.new("RGB", (12, 8), "red").save(images / "a.jpg", exif=exif)
    store = ProjectStore.open(tmp_path / "labels", images)
    store.add_category("First", "#aabbcc")
    with store._connect() as connection:
        connection.execute("INSERT INTO categories VALUES(?,?)", (77, json.dumps({"id": 77, "name": "Last", "color": "#112233"})))
    state = {**store.get_state(1), "schema_version": 2, "annotations": [box(77)]}
    store.save_state(1, state)
    options = yolo.ExportOptions(task="detection", include_images=True)
    preview = yolo.prepare(store, options)
    _, path = yolo.generate(store, yolo.ExportRequest(**options.model_dump(), snapshot=preview["snapshot"]), tmp_path / "exports", store.image_path)
    with zipfile.ZipFile(path) as archive:
        assert archive.read("labels/a.txt").startswith(b"1 ")
        with Image.open(io.BytesIO(archive.read("images/a.png"))) as output:
            assert output.size == (12, 8) and not output.getexif()
    Image.new("RGB", (12, 8)).save(images / "a.png")
    store = ProjectStore.open(store.directory, images, files=["a.png"])
    options.include_empty = True
    assert "filename" in yolo.prepare(store, options)["errors"][0]["reason"]


def test_yolo_rejects_file_directory_collisions_and_changed_options(tmp_path):
    images = tmp_path / "images"
    (images / "a.png").mkdir(parents=True)
    Image.new("RGB", (12, 8)).save(images / "a.jpg")
    Image.new("RGB", (12, 8)).save(images / "a.png" / "nested.jpg")
    store = ProjectStore.open(tmp_path / "project", images)
    options = yolo.ExportOptions(task="detection", include_empty=True)
    preview = yolo.prepare(store, options)
    assert not preview["errors"]
    changed = yolo.ExportRequest(**{**options.model_dump(), "include_images": True}, snapshot=preview["snapshot"])
    with pytest.raises(RevisionConflict):
        yolo.generate(store, changed, tmp_path / "exports", store.image_path)
    assert not list((tmp_path / "exports").glob("*.zip"))
    options.include_images = True
    preview = yolo.prepare(store, options)
    assert preview["errors"] and "filename collides" in preview["errors"][0]["reason"]


@pytest.mark.parametrize("failure", ["after_reservation", "after_replace"])
def test_v2_migration_recovers_once_with_quota(hosted, monkeypatch, failure):
    settings, db, _, _, clients = hosted
    api = clients["alice"]
    project = project_and_image(api)
    root = f"/api/v1/projects/{project['id']}"
    category = api.post(root + "/categories", json={"name": "Object"}).json()["id"]
    state, _ = mixed(api.get(root + "/images/1/state").json(), category, 16, 12)
    if failure == "after_reservation":
        monkeypatch.setattr(quota, "recover_project", lambda *a, **kw: (_ for _ in ()).throw(OSError("simulated crash")))
    else:
        original = quota._durable_file
        def fsync(path):
            if "projects" in Path(path).parts:
                raise OSError("simulated crash after replace")
            original(path)
        monkeypatch.setattr(quota, "_durable_file", fsync)
    headers = {"X-Annotation-State-Version": "2"}
    assert api.put(root + "/images/1/state", json=state, headers=headers).status_code == 503
    monkeypatch.undo()
    recovered = api.get(root + "/images/1/state", headers=headers).json()
    assert recovered["revision"] == 1 and recovered["annotations"][0]["kind"] == "bbox"
    assert_accounted(settings, db)
    assert api.get(root + "/images/1/state", headers=headers).json() == recovered
    assert_accounted(settings, db)
