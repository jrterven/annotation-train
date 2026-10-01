#!/usr/bin/env python3
"""Run real SAM 3 inference; never substitute generated or fixture masks.

Example: .venv/bin/python scripts/benchmark_sam3.py --image vegetables.jpg \
    --text carrot --device auto --output benchmark.json --artifacts-dir benchmark-masks
Requires approved facebook/sam3 access and `hf auth login` first.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from PIL import Image

from app.geometry import encode_mask
from app.inference import ModelUnavailable, Sam3Engine


def memory(engine: Sam3Engine) -> dict:
    import psutil

    result = {"rss_bytes": psutil.Process().memory_info().rss}
    torch = engine._torch
    if torch is not None and engine.status()["device"] == "mps":
        result["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
        result["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
    elif torch is not None and engine.status()["device"] == "cuda":
        result["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
        result["cuda_peak_bytes"] = torch.cuda.max_memory_allocated()
    return result


def sync(engine: Sam3Engine) -> None:
    if engine.status()["device"] == "mps":
        engine._torch.mps.synchronize()
    elif engine.status()["device"] == "cuda":
        engine._torch.cuda.synchronize()


def save_masks(directory: Path, stem: str, masks: list[np.ndarray], scores: list[float]) -> dict:
    """Persist actual model outputs for visual review and device comparisons."""
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, mask in enumerate(masks):
        path = directory / f"{stem}-{index:03d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        entry = {"png": str(path.resolve()), "rle": encode_mask(mask)}
        if scores:
            entry["score"] = scores[index]
        entries.append(entry)
    manifest = directory / f"{stem}.json"
    manifest.write_text(json.dumps({"case": stem, "masks": entries}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"manifest": str(manifest.resolve()), "pngs": [item["png"] for item in entries]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--text", required=True, help="A visible concept, for example carrot.")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, help="Save binary PNG masks and COCO RLE for every case.")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 30:
        parser.error("--repeats must be between 1 and 30")
    engine = Sam3Engine(device=args.device)
    report = {
        "status": "running", "platform": platform.platform(), "python": platform.python_version(),
        "image": str(args.image.resolve()), "text": args.text,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "transformers", "huggingface_hub")},
        "cases": [],
        "note": "Real model benchmark. Success validates execution, not annotation accuracy.",
        "memory_note": "RSS and MPS allocations are sampled after each case; only CUDA reports a recorded allocator peak.",
    }

    def checkpoint():
        report["model"] = engine.status()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(args.output)

    def measure(name, operation):
        entry = {"case": name, "status": "running"}
        report["cases"].append(entry)
        checkpoint()
        print(f"[{len(report['cases']):02d}] {name}: running", file=sys.stderr, flush=True)
        start = time.perf_counter()
        try:
            sync(engine)
            start = time.perf_counter()
            result = operation()
            sync(engine)
            entry.update(seconds=time.perf_counter() - start, timings=dict(engine.last_timings), memory=memory(engine), device=engine.status()["device"])
            if isinstance(result, list):
                masks = [item["mask"] for item in result]
                scores = [float(item["score"]) for item in result]
                if not all(np.isfinite(score) for score in scores):
                    raise AssertionError("Non-finite proposal confidence")
                entry.update(proposal_count=len(result), scores=scores)
            else:
                masks, scores = [result], []
            for mask in masks:
                if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != (image.height, image.width):
                    raise AssertionError("Incorrect raster shape or dtype")
            entry["foreground_pixels"] = [int(mask.sum()) for mask in masks] if isinstance(result, list) else int(result.sum())
            if args.artifacts_dir:
                entry["artifacts"] = save_masks(args.artifacts_dir, f"{len(report['cases']):02d}-{name}", masks, scores)
            if not masks or any(not mask.any() for mask in masks):
                raise AssertionError("The model returned no instances or an empty instance mask")
            entry["status"] = "passed"
            return result
        except Exception as error:
            entry.update(status="failed", error=str(error), seconds=entry.get("seconds", time.perf_counter() - start))
            raise
        finally:
            checkpoint()
            print(f"[{len(report['cases']):02d}] {name}: {entry['status']} ({entry.get('seconds', 0):.3f}s)", file=sys.stderr, flush=True)

    def point_evidence(mask, positive, negative):
        return {"positive_included": bool(mask[positive["y"], positive["x"]]), "negative_excluded": not bool(mask[negative["y"], negative["x"]])}

    exit_code = 0
    try:
        checkpoint()
        print("Loading image and SAM 3 weights…", file=sys.stderr, flush=True)
        with Image.open(args.image) as source:
            image = source.convert("RGB")  # same unrotated raster as the application
        report["image_size"] = [image.width, image.height]
        report["image_rgb_sha256"] = hashlib.sha256(image.tobytes()).hexdigest()
        start = time.perf_counter()
        engine.load()
        report["load_seconds"] = time.perf_counter() - start
        report["memory_after_load"] = memory(engine)
        checkpoint()
        print(f"Model ready on {engine.status()['device']} ({report['load_seconds']:.3f}s)", file=sys.stderr, flush=True)
        key = str(args.image.resolve())
        proposals = measure("text_cold", lambda: engine.predict_text(image, key, args.text))
        if not proposals:
            raise ValueError("El texto no produjo propuestas; usa un concepto presente para probar el refinamiento.")
        seed = proposals[0]["mask"]
        ys, xs = np.nonzero(seed)
        # Pick a real foreground pixel nearest the mask centroid, and a real
        # background pixel. These are prompts, never substitute predictions.
        inside = int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))
        positive = {"x": int(xs[inside]), "y": int(ys[inside]), "label": 1}
        background = np.argwhere(~seed)
        if not len(background):
            raise ValueError("La propuesta cubre toda la imagen; no hay fondo para probar un clic negativo.")
        # A background point near the object is more informative than a corner.
        nearest = int(np.argmin((background[:, 1] - positive["x"]) ** 2 + (background[:, 0] - positive["y"]) ** 2))
        negative = {"x": int(background[nearest, 1]), "y": int(background[nearest, 0]), "label": 0}
        box = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
        report["prompts"] = {"positive": positive, "negative": negative, "box": box}
        part = {"id": "benchmark-clicks", "points": [positive]}
        first = measure("point_cold", lambda: engine.predict_points(image, key, part))
        part["points"].append(negative)
        refined = measure("point_negative", lambda: engine.predict_points(image, key, part))
        report["click_refinement_changed_pixels"] = int(np.count_nonzero(first != refined))
        report["click_refinement_prompt_checks"] = point_evidence(refined, positive, negative)
        boxed = measure("box", lambda: engine.predict_points(image, key, {"id": "benchmark-box", "box": box, "points": []}))
        if len(proposals) > 1:
            second_y, second_x = np.nonzero(proposals[1]["mask"])
            second_box = [int(second_x.min()), int(second_y.min()), int(second_x.max() + 1), int(second_y.max() + 1)]
            report["prompts"]["second_box"] = second_box
            second = measure("box_second_part", lambda: engine.predict_points(image, key, {"id": "benchmark-second-box", "box": second_box, "points": []}))
            union = np.logical_or(boxed, second)
            report["two_part_union"] = {"foreground_pixels": int(union.sum()), "contains_each_part": bool(np.all(union[boxed]) and np.all(union[second]))}
            if args.artifacts_dir:
                report["two_part_union"]["artifacts"] = save_masks(args.artifacts_dir, "two-part-union", [union], [])
        seeded = {"id": "benchmark-seed", "seed_mask": encode_mask(seed), "points": [positive, negative]}
        result = measure("text_to_clicks", lambda: engine.predict_points(image, key, seeded))
        report["seed_refinement_changed_pixels"] = int(np.count_nonzero(seed != result))
        report["seed_refinement_prompt_checks"] = point_evidence(result, positive, negative)
        # Also exercise reopening: no retained detector logits, only persisted RLE.
        engine._seed_cache.clear()
        engine._part_cache.clear()
        report["reopened_seed_cache_empty_before_call"] = not engine._seed_cache and not engine._part_cache
        seeded["id"] = "benchmark-reopened-seed"
        reopened = measure("reopened_text_to_clicks", lambda: engine.predict_points(image, key, seeded))
        report["reopened_refinement_prompt_checks"] = point_evidence(reopened, positive, negative)
        report["reopened_vs_cached_changed_pixels"] = int(np.count_nonzero(result != reopened))
        for index in range(args.repeats):
            measure("text_warm", lambda: engine.predict_text(image, key, args.text))
            measure("point_warm", lambda: engine.predict_points(image, key, {
                "id": f"benchmark-warm-{index}", "points": [positive, negative]
            }))
        report["latencies"] = {}
        for name in ("text_warm", "point_warm"):
            values = [case["seconds"] for case in report["cases"] if case["case"] == name]
            report["latencies"][name] = {"p50_seconds": statistics.median(values), "p95_seconds": float(np.percentile(values, 95))}
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "blocked" if engine.status()["state"] == "auth_required" else "failed"
        report["error"] = str(error)
        exit_code = 2
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        exit_code = 130
    finally:
        checkpoint()
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
