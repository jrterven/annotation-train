"""Exact proposal unions use the existing local/hosted state and COCO APIs."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pycocotools.coco import COCO

from app.geometry import decode_mask, rasterize_components
from test_api import client  # noqa: F401 - shared disposable local fixture
from test_hosted_backend import hosted, project_and_image  # noqa: F401


@pytest.mark.parametrize("mode", ["local", "hosted"])
def test_merge_proposals_exact_union_save_reopen_and_coco(request, tmp_path, mode):
    fixture_path = Path(__file__).parents[1] / "frontend/src/__tests__/fixtures/merge-proposals.json"
    fixture = json.loads(fixture_path.read_text())
    if mode == "local":
        api = request.getfixturevalue("client")
        images = tmp_path / "merge-images"
        images.mkdir()
        Image.new("RGB", (16, 12), "white").save(images / "sample.png")
        project = api.post("/api/projects/open", json={
            "directory": str(tmp_path / "merge-project"), "image_root": str(images),
        }).json()
        root = "/api"
        category_path = root + "/project/categories"
    else:
        *_, clients = request.getfixturevalue("hosted")
        api = clients["alice"]
        project = project_and_image(api)
        root = f"/api/v1/projects/{project['id']}"
        category_path = root + "/categories"
    category = api.post(category_path, json={"name": "Combined"}).json()["id"]
    state_url = root + "/images/1/state"
    state = api.get(state_url).json()
    state["annotations"] = [{"id": "existing", "category_id": category, "iscrowd": 1, **fixture["other"]}]
    state["draft"] = {"id": "draft", "category_id": category, "active_part_id": "part",
                      "parts": [{"id": "part", "points": [], **fixture["other"]}]}
    state["proposals"] = [{"id": name, "category_id": category, "iscrowd": 0,
                           "score": 0.9, "selected": name != "other", **fixture[name]}
                          for name in ("first", "second", "other")]
    saved = api.put(state_url, json=state)
    assert saved.status_code == 200, saved.text
    before = saved.json()
    body = {"image_id": 1, "masks": [p["mask"] for p in before["proposals"] if p["selected"]]}
    if mode == "hosted":
        assert clients["bob"].post(root + "/masks/union", json=body).status_code == 404
        assert api.post(root + "/masks/union", json=body, headers={"X-CSRF-Token": "invalid"}).status_code == 403
    union = api.post(root + "/masks/union", json=body)
    assert union.status_code == 200, union.text
    result = union.json()
    expected = decode_mask(fixture["first"]["mask"]) | decode_mask(fixture["second"]["mask"])
    assert np.array_equal(decode_mask(result["mask"]), expected)
    assert np.array_equal(decode_mask(fixture["merged"]["mask"]), expected)
    assert np.array_equal(rasterize_components(result["components"], 16, 12), expected)
    assert len(result["components"]) == 2  # a separate single-pixel island
    assert not expected[3, 3]  # the uncovered hole remains empty
    assert expected[11, 15]  # original-image border pixel is preserved
    assert api.get(state_url).json() == before  # union itself never mutates state

    merged = {"id": "merged", "category_id": category, "iscrowd": 0, **result}
    update = {**before, "annotations": [*before["annotations"], merged],
              "proposals": [p for p in before["proposals"] if not p["selected"]]}
    saved = api.put(state_url, json=update)
    assert saved.status_code == 200, saved.text
    after = saved.json()
    assert after["revision"] == before["revision"] + 1
    assert after["annotations"][0] == before["annotations"][0]
    assert after["draft"] == before["draft"]
    assert [p["id"] for p in after["proposals"]] == ["other"]
    assert api.put(state_url, json=update).status_code == 409
    if mode == "local":
        assert api.post("/api/projects/open", json={"directory": project["directory"]}).status_code == 200
    assert api.get(state_url).json() == after
    assert api.post(root + "/coco/export", json={}).status_code == 200
    exported = api.get(root + "/coco/download").json()
    coco = COCO()
    coco.dataset = exported
    coco.createIndex()
    assert len(exported["annotations"]) == 2
    annotation = next(a for a in exported["annotations"] if a["iscrowd"] == 0)
    assert annotation["category_id"] == category
    assert annotation["area"] == int(expected.sum())
    assert np.array_equal(coco.annToMask(annotation), expected)
    # Undo saves the original proposals; redo preserves the merged COCO ID.
    assert api.put(state_url, json={**before, "revision": after["revision"]}).status_code == 200
    restored = api.get(state_url).json()
    assert api.put(state_url, json={**after, "revision": restored["revision"]}).status_code == 200
    assert api.post(root + "/coco/export", json={}).status_code == 200
    assert api.get(root + "/coco/download").json()["annotations"] == exported["annotations"]
