#!/usr/bin/env python3
"""Real model acceptance probe; private URL/token come only from environment.

Run inside the worker runtime after readiness. Prints aggregate measurements,
never service credentials, worker addresses, source pixels, or model paths.
"""
import base64
import hashlib
import io
import json
import os
import time
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from app.geometry import decode_mask, encode_mask


def main():
    url = os.environ.get("ANNOTATION_WORKER_URL", "http://127.0.0.1:8766").rstrip("/")
    token = os.environ["ANNOTATION_WORKER_TOKEN"]

    def call(method, path, body=None):
        request = Request(url + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                          headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        with urlopen(request, timeout=15) as response:
            return json.load(response)

    if call("GET", "/v1/health").get("ready") is not True:
        raise RuntimeError("Worker has not passed its startup probes")
    image = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 60, 240, 180), fill="red", outline="black", width=2)
    data = io.BytesIO()
    image.save(data, format="PNG")
    raw = data.getvalue()
    encoded = base64.b64encode(raw).decode()
    seed = np.zeros((240, 320), dtype=np.uint8)
    seed[60:181, 80:241] = 1
    prompts = [
        ("points", "points", {"part": {"points": [{"x": 160, "y": 120, "label": 1}]}}),
        ("box", "points", {"part": {"box": [80, 60, 240, 180]}}),
        ("polygon_seed", "points", {"part": {"seed_mask": encode_mask(seed), "points": [{"x": 20, "y": 20, "label": 0}]}}),
        ("text_en", "text", {"category_id": 1, "text": "red rectangle", "source_language": "en"}),
        ("text_es", "text", {"category_id": 1, "text": "rectángulo rojo", "source_language": "es"}),
        ("visual", "visual", {"category_id": 1, "reference_image": encoded, "reference_box": [80, 60, 240, 180]}),
    ]
    project_id = str(uuid4())
    measurements = []
    for name, kind, prompt in prompts:
        started = time.monotonic()
        attempt_id = str(uuid4())
        body = {"job_id": str(uuid4()), "attempt_id": attempt_id, "project_id": project_id,
                "image_id": 1, "sha256": hashlib.sha256(raw).hexdigest(), "image_base64": encoded,
                "deadline_at": time.time() + 120, "kind": kind, "payload": {"revision": 1, **prompt}}
        status = call("POST", "/v1/attempts", body)
        while status["status"] == "running" and time.monotonic() - started < 125:
            time.sleep(.2)
            status = call("GET", f"/v1/attempts/{attempt_id}")
        if status["status"] != "succeeded":
            raise RuntimeError(f"{name} failed: {status.get('error', status['status'])}")
        result = status["result"]
        assert result["image_id"] == 1 and result["revision"] == 1
        masks = [result] if kind == "points" else result["proposals"]
        for candidate in masks:
            assert decode_mask(candidate["mask"]).shape == (240, 320)
        if kind == "points":
            assert decode_mask(result["mask"]).any()
        if name == "text_es":
            assert result["prompt"]["source_language"] == "es" and result["prompt"]["english"]
        measurements.append({"mode": name, "seconds": round(time.monotonic() - started, 3), "masks": len(masks)})
    print(json.dumps({"passed": True, "measurements": measurements}, indent=2))


if __name__ == "__main__":
    main()
