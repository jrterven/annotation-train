import hashlib
import json
import httpx
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from app.geometry import mask_payload, decode_mask
from app.training.adapters import prepare_dataset, train, total_loss
from app.training.agent import server_url, write_private


def fixture(tmp_path, task):
    originals = tmp_path / "originals"
    originals.mkdir()
    mask = np.zeros((12, 16), bool)
    mask[1:10, 1:10] = True
    mask[4:7, 4:7] = False
    mask[10, 14] = True
    images = []
    for id, split in enumerate(("train", "validation"), 1):
        path = originals / f"{id}.png"
        Image.new("RGB", (16, 12), (id * 50, 10, 20)).save(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        path.rename(originals / sha)
        ann = (
            {"kind": "bbox", "bbox": [1, 1, 8, 9]}
            if task == "detection"
            else mask_payload(mask, include_preview=False)
        )
        images.append(
            {
                "id": id,
                "split": split,
                "sha256": sha,
                "width": 16,
                "height": 12,
                "annotations": [{"category_id": 7, **ann}],
            }
        )
    return (
        {
            "task": task,
            "classes": [{"index": 0, "category_id": 7, "name": "Object"}],
            "images": images,
        },
        originals,
        mask,
    )


@pytest.mark.parametrize("task", ["detection", "segmentation"])
@pytest.mark.parametrize("family", ["yolo26", "rfdetr"])
def test_dataset_export_preserves_original_grid_and_exact_rle(tmp_path, task, family):
    manifest, originals, mask = fixture(tmp_path, task)
    before = {p.name: p.read_bytes() for p in originals.iterdir()}
    output = prepare_dataset(manifest, originals, tmp_path / "dataset", family)
    assert {p.name: p.read_bytes() for p in originals.iterdir()} == before
    assert not (output / "test").exists()
    assert not (output / "images/test").exists()
    if family == "rfdetr":
        data = json.loads((output / "train/_annotations.coco.json").read_text())
        assert Image.open(output / "train/1.png").size == (16, 12)
        assert data["annotations"][0]["category_id"] == 0
        if task == "segmentation":
            assert np.array_equal(decode_mask(data["annotations"][0]["segmentation"]), mask)
    else:
        import yaml

        assert "test" not in yaml.safe_load((output / "data.yaml").read_text())
        row = (output / "labels/train/1.txt").read_text().split()
        assert row[0] == "0" and all(0 <= float(x) <= 1 for x in row[1:])
        assert len(row) == 5 if task == "detection" else len(row) > 6


def test_adapter_rejects_test_data_before_loading_a_model(tmp_path):
    manifest, _, _ = fixture(tmp_path, "detection")
    manifest["images"][0]["split"] = "test"
    with pytest.raises(ValueError, match="never receive test"):
        train({}, manifest, tmp_path, tmp_path, lambda _: None, lambda: False)


def test_named_losses_in_pinned_yolo_runtime():
    assert total_loss({"box": 1.5, "cls": 0.25, "dfl": 0.5}) == 2.25
    assert total_loss({"box": float("nan")}) is None
    assert total_loss(None) is None


def test_final_coco_report_is_serializable_and_uses_only_test(tmp_path, monkeypatch):
    from app.training import adapters
    from app.training.data import encoded

    manifest, _, _ = fixture(tmp_path, "detection")
    with pytest.raises(ValueError, match="only reserved test"):
        adapters.evaluate({}, manifest, tmp_path)
    for image in manifest["images"]:
        image["split"] = "test"
    monkeypatch.setattr(
        adapters,
        "predict",
        lambda *_: {
            "images": [
                {
                    "image_id": image["id"],
                    "proposals": [{"category_id": 7, "bbox": [1, 1, 8, 9], "score": 0.9}],
                }
                for image in manifest["images"]
            ]
        },
    )
    result = adapters.evaluate(
        {"dataset_id": "snapshot", "checkpoint_sha256": "verified"}, manifest, tmp_path
    )
    assert result["metrics"]["mAP50-95"] == pytest.approx(1)
    assert json.loads(encoded(result))["per_class"][0]["category_id"] == 7


def test_agent_transport_only_https_or_loopback_and_private_credentials(tmp_path):
    for value in (
        "http://gpu.example",
        "https://user:pass@example.com",
        "https://example.com/arbitrary",
    ):
        with pytest.raises(ValueError):
            server_url(value)
    path = tmp_path / "agent.json"
    write_private(path, {"token": "disposable"})
    assert path.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("origin,status,config,suffix", [
    ("http://localhost:8765", 200, {"mode": "hosted"}, "/api/v1/training-agent"),
    ("https://annotation.example", 200, {"mode": "hosted"}, "/api/v1/training-agent"),
    ("http://127.0.0.1:8765", 404, {}, "/api/training-agent"),
    ("http://[::1]:8765", 200, {"mode": "local"}, "/api/training-agent"),
])
def test_agent_discovers_application_mode_before_pairing(origin, status, config, suffix):
    def respond(request):
        assert request.method == "GET" and request.url.path == "/api/config"
        assert "authorization" not in request.headers
        return httpx.Response(status, json=config)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert server_url(origin, client=client) == origin + suffix


@pytest.mark.parametrize("origin,status,config", [
    ("https://annotation.example", 404, {}),
    ("http://localhost:8765", 503, {}),
    ("http://localhost:8765", 302, {}),
    ("http://localhost:8765", 200, {"mode": "unknown"}),
    ("https://annotation.example", 200, {"mode": "local"}),
])
def test_agent_discovery_fails_closed(origin, status, config):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status, json=config, headers={"Location": "https://other.example"})

    with httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
        with pytest.raises((ValueError, httpx.HTTPStatusError)):
            server_url(origin, client=client)
    assert len(requests) == 1
