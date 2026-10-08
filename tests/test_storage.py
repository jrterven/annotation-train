import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sqlite3
import threading
import uuid

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import pytest

from app.geometry import decode_mask, encode_mask, mask_payload
from app.storage import DATABASE_NAME, ProjectStore, RevisionConflict


def make_image(path: Path, size=(12, 8), color="white"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


@pytest.fixture
def project(tmp_path):
    images = tmp_path / "images"
    make_image(images / "a.png")
    make_image(images / "sub" / "b.png")
    store = ProjectStore.open(tmp_path / "labels", images)
    store.add_category("Vegetal", "#33aa99")
    return store


def annotation(category_id=1, identity=None):
    mask = np.zeros((8, 12), dtype=bool)
    mask[1:7, 2:10] = True
    mask[3:5, 4:6] = False
    return {"id": identity or str(uuid.uuid4()), "category_id": category_id, "iscrowd": 0, **mask_payload(mask)}


def test_project_keeps_images_external_and_selected_files(tmp_path):
    root = tmp_path / "originals"
    make_image(root / "a.png")
    make_image(root / "nested" / "a.png")
    store = ProjectStore.open(tmp_path / "labels", root, files=["nested/a.png"])
    data = store.project()
    assert [image["file_name"] for image in data["images"]] == ["nested/a.png"]
    assert data["image_root"] == str(root)
    assert not list(store.directory.rglob("*.png"))
    assert store.get_state(data["images"][0]["id"])["revision"] == 0
    with pytest.raises(ValueError, match="outside"):
        ProjectStore.open(root / "nested" / "labels", root)


def test_state_roundtrip_revision_conflict_and_no_base64_storage(project):
    state = project.get_state(1)
    original = copy.deepcopy(state)
    state["annotations"].append(annotation())
    saved = project.save_state(1, state)
    assert saved["revision"] == 1
    assert saved["annotations"][0]["preview"].startswith("data:image/png")
    with pytest.raises(RevisionConflict):
        project.save_state(1, original)
    reopened = ProjectStore.open(project.directory)
    assert reopened.get_state(1) == saved
    assert reopened.project()["images"][0]["annotation_count"] == 1
    with sqlite3.connect(project.database) as connection:
        assert "base64" not in connection.execute("SELECT state FROM images WHERE id=1").fetchone()[0]


def test_compact_state_preserves_masks_controls_draft_and_proposals(project, monkeypatch):
    state = project.get_state(1)
    item = annotation()
    state["annotations"] = [item]
    state["proposals"] = [{**annotation(), "score": .9, "selected": False}]
    state["draft"] = {"id": "draft", "category_id": 1, "active_part_id": "part",
                      "parts": [{"id": "part", "mask": item["mask"], "components": item["components"]}]}
    saved = project.save_state(1, state)
    expected = copy.deepcopy(saved)
    for record in expected["annotations"] + expected["proposals"] + expected["draft"]["parts"]:
        record.pop("preview", None)

    def unexpected(*args, **kwargs):
        pytest.fail("Compact reads must not decode or render full-image masks")

    monkeypatch.setattr("app.storage.decode_mask", unexpected)
    monkeypatch.setattr("app.storage.mask_preview", unexpected)
    assert project.get_state(1, include_previews=False) == expected


def test_invalid_update_is_atomic(project):
    before = project.get_state(1)
    invalid = copy.deepcopy(before)
    invalid["annotations"] = [annotation(category_id=999)]
    with pytest.raises(ValueError):
        project.save_state(1, invalid)
    assert project.get_state(1) == before
    invalid["annotations"] = [annotation()]
    invalid["annotations"][0]["components"] = []
    with pytest.raises(ValueError, match="do not match"):
        project.save_state(1, invalid)
    assert project.get_state(1) == before


def test_simultaneous_saves_accept_exactly_one_revision(project):
    barrier = threading.Barrier(2)
    def save():
        store = ProjectStore(project.directory)
        state = store.get_state(1)
        state["annotations"] = [annotation()]
        barrier.wait()
        try:
            return store.save_state(1, state)["revision"]
        except RevisionConflict:
            return "conflict"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(save), pool.submit(save)]
        results = [future.result(timeout=10) for future in futures]
    assert sorted(map(str, results)) == ["1", "conflict"]
    assert len(project.get_state(1)["annotations"]) == 1


def test_draft_and_proposals_persist_separately_and_do_not_export(project):
    state = project.get_state(1)
    item = annotation()
    part_id = str(uuid.uuid4())
    state["draft"] = {"id": str(uuid.uuid4()), "category_id": 1, "active_part_id": part_id,
                      "parts": [{"id": part_id, "points": [{"x": 4, "y": 3, "label": 1}], "box": [1, 1, 10, 7],
                                 "mask": item["mask"], "seed_mask": item["mask"], "components": item["components"]}]}
    state["proposals"] = [{**annotation(), "score": .92, "selected": True}]
    saved = project.save_state(1, state)
    assert saved["draft"]["parts"][0]["seed_mask"] == item["mask"]
    assert len(saved["proposals"]) == 1
    assert json.loads(project.export_coco().read_text())["annotations"] == []


@pytest.mark.parametrize("vertices,closed", [
    ([], False), ([[0, 0]], False), ([[0, 0], [12, 8]], False),
    ([[0, 0], [12, 0], [12, 8], [0, 8]], True),
])
def test_polygon_draft_reopens_without_becoming_an_annotation(project, vertices, closed):
    state = project.get_state(1)
    polygon = {"vertices": vertices, "closed": closed}
    state["draft"] = {"id": "polygon-draft", "category_id": 1, "active_part_id": "part",
                      "parts": [{"id": "part", "points": [], "polygon": polygon}]}
    project.save_state(1, state)
    reopened = ProjectStore.open(project.directory)
    restored = reopened.get_state(1)
    assert restored["draft"]["parts"] == state["draft"]["parts"]
    assert restored["annotations"] == []
    assert json.loads(reopened.export_coco().read_text())["annotations"] == []


def test_refined_polygon_preserves_input_separately_from_output_and_clicks(project):
    state = project.get_state(1)
    polygon = {"vertices": [[1.5, 1], [11, 1], [11, 7.5], [1, 7]], "closed": True}
    seed = np.ones((8, 12), dtype=bool)
    prediction = annotation()
    part = {"id": "part", "points": [{"x": 3, "y": 3, "label": 0}], "polygon": polygon,
            "seed_mask": encode_mask(seed), **{k: prediction[k] for k in ("mask", "components")}}
    state["draft"] = {"id": "polygon-draft", "category_id": 1, "active_part_id": "part", "parts": [part]}
    project.save_state(1, state)
    restored = ProjectStore.open(project.directory).get_state(1)["draft"]["parts"][0]
    assert restored["polygon"] == polygon
    assert restored["points"] == part["points"]
    assert np.array_equal(decode_mask(restored["seed_mask"]), seed)
    assert np.array_equal(decode_mask(restored["mask"]), decode_mask(prediction["mask"]))
    assert not np.array_equal(decode_mask(restored["mask"]), seed)


@pytest.mark.parametrize("polygon", [
    None, [], {}, {"vertices": [], "closed": "false"}, {"vertices": None, "closed": False},
    {"vertices": [[1, 1], [2, 2]], "closed": True},
    {"vertices": [[1]], "closed": False}, {"vertices": [[1, 2, 3]], "closed": False},
    {"vertices": [[True, 1]], "closed": False}, {"vertices": [["1", 1]], "closed": False},
    {"vertices": [[float("nan"), 1]], "closed": False},
    {"vertices": [[1, float("inf")]], "closed": False},
    {"vertices": [[-0.1, 1]], "closed": False}, {"vertices": [[12.01, 1]], "closed": False},
    {"vertices": [[1, 8.01]], "closed": False},
])
def test_malformed_polygon_does_not_overwrite_saved_draft(project, polygon):
    state = project.get_state(1)
    state["draft"] = {"id": "polygon-draft", "category_id": 1, "active_part_id": "part",
                      "parts": [{"id": "part", "points": [], "polygon": {"vertices": [[1, 2]], "closed": False}}]}
    saved = project.save_state(1, state)
    invalid = copy.deepcopy(saved)
    invalid["draft"]["parts"][0]["polygon"] = polygon
    with pytest.raises(ValueError):
        project.save_state(1, invalid)
    assert project.get_state(1) == saved


def test_coco_ids_stable_across_delete_undo_and_new_objects(project):
    state = project.get_state(1)
    first = annotation()
    state["annotations"] = [first]
    state = project.save_state(1, state)
    initial_id = json.loads(project.export_coco().read_text())["annotations"][0]["id"]
    state["annotations"] = []
    state = project.save_state(1, state)
    state["annotations"] = [first, annotation()]
    project.save_state(1, state)
    records = json.loads(project.export_coco().read_text())["annotations"]
    assert [record["id"] for record in records] == [initial_id, initial_id + 1]


def test_annotation_id_cannot_move_to_another_image(project):
    state = project.get_state(1)
    item = annotation()
    state["annotations"] = [item]
    project.save_state(1, state)
    other = project.get_state(2)
    other["annotations"] = [item]
    with pytest.raises(ValueError, match="another image"):
        project.save_state(2, other)


def test_project_moves_without_moving_external_originals(project, tmp_path):
    root = Path(project.project()["image_root"])
    moved_directory = tmp_path / "other" / "labels"
    moved_directory.parent.mkdir()
    shutil.move(str(project.directory), moved_directory)
    moved = ProjectStore.open(moved_directory)
    assert moved.image_path(1) == root / "a.png"
    assert moved.project()["image_root"] == str(root)


@pytest.mark.parametrize("image_root", [None, "", "   "])
def test_new_project_requires_explicit_image_root_without_partial_creation(tmp_path, image_root):
    directory = tmp_path / "new-project"
    with pytest.raises(ValueError, match="Select an image folder"):
        ProjectStore.open(directory, image_root=image_root)
    assert not directory.exists()
    # Even a folder matching the old default is never selected implicitly.
    make_image(directory / "imágenes" / "a.png")
    with pytest.raises(ValueError, match="Select an image folder"):
        ProjectStore.open(directory, image_root=image_root)
    assert not (directory / DATABASE_NAME).exists()
    assert not list(directory.glob(".project.*"))


def test_legacy_internal_image_root_reopens_and_moves_without_explicit_root(tmp_path):
    directory = tmp_path / "bundle"
    make_image(directory / "imágenes" / "a.png")
    store = ProjectStore.open(directory, directory / "imágenes")
    with sqlite3.connect(store.database) as connection:
        assert connection.execute("SELECT value FROM meta WHERE key='image_root'").fetchone()[0] == "imágenes"
    reopened = ProjectStore.open(directory)
    assert reopened.image_path(1) == directory / "imágenes" / "a.png"
    moved_directory = tmp_path / "moved"
    shutil.move(str(store.directory), moved_directory)
    moved = ProjectStore.open(moved_directory)
    assert moved.image_path(1) == moved_directory / "imágenes" / "a.png"


def test_reopening_selected_subset_never_adds_other_images(tmp_path):
    root = tmp_path / "originals"
    make_image(root / "a.png")
    make_image(root / "b.png")
    store = ProjectStore.open(tmp_path / "labels", root, files=["b.png"])
    reopened = ProjectStore.open(store.directory)
    assert [image["file_name"] for image in reopened.project()["images"]] == ["b.png"]
    added = ProjectStore.open(store.directory, files=["a.png"])
    assert [image["file_name"] for image in added.project()["images"]] == ["a.png", "b.png"]


def test_relink_requires_exact_originals_and_rolls_back(project, tmp_path):
    replacement = tmp_path / "replacement"
    shutil.copytree(project.project()["image_root"], replacement)
    project.relink(replacement)
    assert project.image_path(1) == replacement / "a.png"
    wrong = tmp_path / "wrong"
    shutil.copytree(replacement, wrong)
    make_image(wrong / "sub" / "b.png", color="red")
    with pytest.raises(ValueError, match="does not match"):
        project.relink(wrong)
    assert project.project()["image_root"] == str(replacement)


def test_modified_original_is_detected(project):
    root = Path(project.project()["image_root"])
    make_image(root / "a.png", color="blue")
    with pytest.raises(ValueError, match="changed"):
        project.image_path(1)
    with pytest.raises(ValueError, match="changed"):
        ProjectStore.open(project.directory)


def make_coco(tmp_path):
    root = tmp_path / "originals"
    make_image(root / "left" / "same.png")
    make_image(root / "right" / "same.png")
    make_image(root / "empty.png")
    mask = decode_mask(annotation()["mask"])
    source = {"info": {"description": "Original"}, "licenses": [{"id": 2, "name": "CC"}],
              "images": [{"id": 9, "file_name": "left/same.png", "width": 12, "height": 8, "license": 2},
                         {"id": 42, "file_name": "right/same.png", "width": 12, "height": 8},
                         {"id": 88, "file_name": "empty.png", "width": 12, "height": 8}],
              "categories": [{"id": 7, "name": "Vegetal", "supercategory": "Alimentos"},
                             {"id": 20, "name": "Sin usar"}],
              "annotations": [{"id": 61, "image_id": 9, "category_id": 7, "iscrowd": 1,
                               "segmentation": encode_mask(mask), "area": 999, "bbox": [0, 0, 1, 1], "custom": "keep"},
                              {"id": 14, "image_id": 42, "category_id": 7, "iscrowd": 0,
                               "segmentation": [[0, 0, 3, 0, 3, 3, 0, 3], [8, 5, 12, 5, 12, 8, 8, 8]]}]}
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    return root, source_path, source


def test_import_export_coco_exact_masks_ids_metadata_and_unused_records(tmp_path):
    root, source_path, source = make_coco(tmp_path)
    store = ProjectStore.import_coco(tmp_path / "labels", root, source_path)
    output_path = store.export_coco()
    original, exported = COCO(str(source_path)), COCO(str(output_path))
    assert set(original.imgs) == set(exported.imgs)
    assert set(original.cats) == set(exported.cats)
    assert set(original.anns) == set(exported.anns)
    for identity in original.anns:
        assert np.array_equal(original.annToMask(original.anns[identity]), exported.annToMask(exported.anns[identity]))
    output = json.loads(output_path.read_text())
    assert output["info"] == source["info"]
    assert output["licenses"] == source["licenses"]
    assert exported.anns[61]["iscrowd"] == 1
    assert exported.anns[61]["custom"] == "keep"
    assert exported.anns[61]["area"] == int(original.annToMask(original.anns[61]).sum())
    assert exported.anns[61]["bbox"] == [2, 1, 8, 6]
    assert store.get_state(9)["revision"] == 0
    assert (store.directory / "source.coco.json").read_text() == source_path.read_text()
    assert {image["file_name"] for image in store.project()["images"]} == {"left/same.png", "right/same.png", "empty.png"}
    # Newly created annotations and classes continue past imported integers.
    category = store.add_category("Nueva", "#AABBCC")
    assert category["id"] == 21
    state = store.get_state(9)
    state["annotations"].append(annotation(category_id=category["id"]))
    store.save_state(9, state)
    assert 62 in COCO(str(store.export_coco())).anns


def test_uncompressed_rle_import(tmp_path):
    root, source_path, source = make_coco(tmp_path)
    source["annotations"] = [{"id": 1, "image_id": 9, "category_id": 7,
                              "segmentation": {"size": [8, 12], "counts": [0, 1, 95]}}]
    source_path.write_text(json.dumps(source))
    store = ProjectStore.import_coco(tmp_path / "labels", root, source_path)
    state = store.get_state(9)
    assert decode_mask(state["annotations"][0]["mask"]).sum() == 1
    assert len(state["annotations"][0]["components"][0]["outer"]) == 4


@pytest.mark.parametrize("corruption", ["duplicate_id", "missing_image", "bad_size", "bad_category", "bbox_only", "bad_rle", "path_escape"])
def test_invalid_import_never_creates_partial_project(tmp_path, corruption):
    root, source_path, source = make_coco(tmp_path)
    if corruption == "duplicate_id":
        source["annotations"].append(copy.deepcopy(source["annotations"][0]))
    elif corruption == "missing_image":
        source["images"][0]["file_name"] = "missing.png"
    elif corruption == "bad_size":
        source["images"][0]["width"] = 11
    elif corruption == "bad_category":
        source["annotations"][0]["category_id"] = 99
    elif corruption == "bbox_only":
        del source["annotations"][0]["segmentation"]
    elif corruption == "bad_rle":
        source["annotations"][0]["segmentation"] = {"size": [8, 12], "counts": [100]}
    elif corruption == "path_escape":
        source["images"][0]["file_name"] = "../outside.png"
    source_path.write_text(json.dumps(source))
    destination = tmp_path / "labels"
    with pytest.raises(ValueError):
        ProjectStore.import_coco(destination, root, source_path)
    assert not (destination / DATABASE_NAME).exists()


def test_coco_project_does_not_merge_over_existing(project, tmp_path):
    root, source, _ = make_coco(tmp_path)
    with pytest.raises(ValueError, match="new project"):
        ProjectStore.import_coco(project.directory, root, source)


def test_exif_orientation_does_not_change_coco_coordinates(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    image = Image.new("RGB", (12, 8), "white")
    exif = Image.Exif()
    exif[274] = 6
    image.save(root / "rotated.jpg", exif=exif)
    store = ProjectStore.open(tmp_path / "labels", root)
    assert (store.project()["images"][0]["width"], store.project()["images"][0]["height"]) == (12, 8)


def test_approximate_controls_persist_without_changing_mask_or_export(tmp_path):
    root = tmp_path / "images"
    make_image(root / "circle.png", size=(200, 160))
    store = ProjectStore.open(tmp_path / "labels", root)
    store.add_category("Círculo", "#ABCDEF")
    y, x = np.ogrid[:160, :200]
    mask = (x - 100) ** 2 + (y - 80) ** 2 <= 60 ** 2
    payload = mask_payload(mask)
    state = store.get_state(1)
    state["annotations"] = [{"id": str(uuid.uuid4()), "category_id": 1, "iscrowd": 0, **payload}]
    saved = store.save_state(1, state)
    assert saved["annotations"][0]["controls"] == payload["controls"]
    assert saved["annotations"][0]["components"] == payload["components"]
    assert np.array_equal(decode_mask(saved["annotations"][0]["mask"]), mask)
    exported = COCO(str(store.export_coco()))
    assert np.array_equal(exported.annToMask(next(iter(exported.anns.values()))), mask)


def test_invalid_controls_rejected_without_touching_saved_state(project):
    state = project.get_state(1)
    item = annotation()
    item["controls"] = [{"outer": [[0, 0], [6, 6], [0, 6], [6, 0]], "holes": []}]
    state["annotations"] = [item]
    with pytest.raises(ValueError, match="Invalid polygon"):
        project.save_state(1, state)
    assert project.get_state(1)["revision"] == 0


def test_older_records_regenerate_controls_without_revision_or_mask_changes(project):
    state = project.get_state(1)
    item = annotation()
    state["annotations"] = [item]
    saved = project.save_state(1, state)
    with sqlite3.connect(project.database) as connection:
        stored = json.loads(connection.execute("SELECT state FROM images WHERE id=1").fetchone()[0])
        del stored["annotations"][0]["controls"]
        connection.execute("UPDATE images SET state=? WHERE id=1", (json.dumps(stored),))
    read = project.get_state(1)
    assert read["revision"] == saved["revision"]
    assert read["annotations"][0]["mask"] == item["mask"]
    assert read["annotations"][0]["components"] == item["components"]
    assert read["annotations"][0]["controls"]
    with sqlite3.connect(project.database) as connection:
        assert "controls" not in json.loads(connection.execute("SELECT state FROM images WHERE id=1").fetchone()[0])["annotations"][0]
