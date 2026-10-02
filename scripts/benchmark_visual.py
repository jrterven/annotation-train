#!/usr/bin/env python3
"""Validate real SAM 3 visual-reference inference without a composite image.

Example: HF_HUB_OFFLINE=1 HF_HUB_CACHE=.cache/huggingface .venv/bin/python \
    scripts/benchmark_visual.py --reference sample/images/1047.jpg \
    --reference-box 38 57 170 119 --target sample/images/1001.jpg \
    --text carrot --unrelated-reference sample/images/1190.jpg \
    --unrelated-reference-box 83 25 158 91 --device mps

The same-image check compares raw outputs to the native box-prompt path. External
reference results are diagnostic examples, not a ground-truth accuracy benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from PIL import Image, ImageOps

from app.geometry import encode_mask
from app.inference import Sam3Engine, _mask_digest


def read_image(path: Path, *, reference: bool = False) -> Image.Image:
    with Image.open(path) as source:
        if not reference:
            return source.convert("RGB")  # Annotation targets retain original raster coordinates.
        oriented = ImageOps.exif_transpose(source)
        if "A" in oriented.getbands() or "transparency" in oriented.info:
            background = Image.new("RGBA", oriented.size, "white")
            return Image.alpha_composite(background, oriented.convert("RGBA")).convert("RGB")
        return oriented.convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-box", nargs=4, type=int, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--text", default="")
    parser.add_argument("--unrelated-reference", type=Path)
    parser.add_argument("--unrelated-reference-box", nargs=4, type=int)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("output/qa/visual-reference"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    engine = Sam3Engine(device=args.device)
    report = {"status": "running", "reference": str(args.reference.resolve()), "target": str(args.target.resolve()),
              "reference_box": args.reference_box, "cases": [],
              "note": "Status measures execution only and can pass for background or incorrect predictions. Same-image numeric equivalence is checked separately. No annotated ground truth; no accuracy claim.",
              "quality_status": "not_scored_no_ground_truth",
              "quality_checks": {},
              "native_support": "Meta states that cross-image prompting is not officially supported; this tests an adaptation of its geometry tokens.",
              "maintainer_quote": "This not officially supported at the moment.",
              "source": "https://github.com/facebookresearch/sam3/issues/183"}

    def write_report():
        report["model"] = engine.status()
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def sync():
        if engine.status()["device"] == "mps":
            engine._torch.mps.synchronize()
        elif engine.status()["device"] == "cuda":
            engine._torch.cuda.synchronize()

    def measure(name, operation):
        entry = {"case": name, "status": "running"}
        report["cases"].append(entry)
        write_report()
        print(f"{name}: running", flush=True)
        sync()
        start = time.perf_counter()
        try:
            result = operation()
            sync()
            entry.update(status="execution_passed", seconds=time.perf_counter() - start, device=engine.status()["device"], timings=dict(engine.last_timings))
            return result, entry
        except Exception as error:
            entry.update(status="failed", error=str(error), seconds=time.perf_counter() - start)
            raise
        finally:
            write_report()
            print(f"{name}: {entry['status']}", flush=True)

    def artifacts(name, image, proposals):
        rows = []
        overlay = np.asarray(image).astype(np.float32).copy()
        colors = [[74, 222, 128], [96, 165, 250], [251, 191, 36], [244, 114, 182], [167, 139, 250]]
        for index, proposal in enumerate(proposals):
            mask = proposal["mask"]
            if mask.dtype != np.bool_ or mask.shape != (image.height, image.width) or not mask.any():
                raise AssertionError("Unexpected output mask")
            score = proposal.get("score")
            if score is not None and not np.isfinite(score):
                raise AssertionError("Non-finite confidence")
            path = args.output_dir / f"{name}-{index:03d}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(path)
            ys, xs = np.nonzero(mask)
            rows.append({"score": score, "mask": str(path.resolve()), "segmentation": encode_mask(mask),
                         "area": int(mask.sum()), "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]})
            overlay[mask] = overlay[mask] * 0.55 + np.asarray(colors[index % len(colors)]) * 0.45
        path = args.output_dir / f"{name}-overlay.png"
        Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(path)
        manifest = args.output_dir / f"{name}.json"
        manifest.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        return {"proposal_count": len(rows), "scores": [row["score"] for row in rows],
                "overlay": str(path.resolve()), "manifest": str(manifest.resolve())}

    exit_code = 0
    try:
        reference, target = read_image(args.reference, reference=True), read_image(args.target)
        _, box = engine._validate_part({"box": args.reference_box}, reference.width, reference.height)
        cropped = reference.crop(args.reference_box)
        cropped.save(args.output_dir / "reference-crop.png")
        reference.save(args.output_dir / "reference-source.png")
        target.save(args.output_dir / "target.png")
        report.update(reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                      target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                      different_source_images=reference.size != target.size or reference.tobytes() != target.tobytes())
        write_report()
        print("Loading existing local SAM 3 weights…", flush=True)
        engine.load()
        ref_key = engine._image_key(reference, "same-image-validation")

        def native_forward():
            inputs = engine._detector_processor(original_sizes=[[reference.height, reference.width]],
                                                input_boxes=[[list(box)]], input_boxes_labels=[[1]], return_tensors="pt")
            embeddings = engine._detector_embeddings(reference, ref_key)
            return engine._detector(vision_embeds=embeddings,
                                    **engine._device_inputs({key: value for key, value in inputs.items() if key != "original_sizes"}))

        native, native_case = measure("same_image_native_box", lambda: engine._execute(native_forward))
        transferred, transfer_case = measure("same_image_transferred_tokens", lambda: engine._execute(
            lambda: engine._visual_outputs(reference, ref_key, reference, engine._image_key(reference, "visual-reference"), box, "visual")))
        report["same_image_equivalence"] = {}
        for name in ("pred_masks", "pred_boxes", "pred_logits", "presence_logits"):
            a, b = getattr(native, name).detach().cpu(), getattr(transferred, name).detach().cpu()
            maximum = float((a - b).abs().max())
            report["same_image_equivalence"][name + "_max_abs_difference"] = maximum
            engine._torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        native_proposals = engine._concept_proposals(native, reference, ref_key)
        transferred_proposals = engine._concept_proposals(transferred, reference, ref_key)
        native_case.update(artifacts("same-image-native", reference, native_proposals))
        transfer_case.update(artifacts("same-image-transfer", reference, transferred_proposals))
        report["same_image_equivalence"]["passed"] = True
        del native, transferred
        write_report()

        cases = [
            ("external_crop_visual", lambda: engine.predict_visual(target, "external-target", cropped)),
            ("external_full_reference_box", lambda: engine.predict_visual(target, "external-target", reference, box)),
        ]
        if args.text.strip():
            cases.extend([
                ("external_reference_box_plus_text", lambda: engine.predict_visual(target, "external-target", reference, box, text=args.text)),
                ("target_text_baseline", lambda: engine.predict_text(target, "external-target", args.text)),
            ])
        if args.unrelated_reference:
            unrelated = read_image(args.unrelated_reference, reference=True)
            unrelated_box = args.unrelated_reference_box
            if unrelated_box is None:
                raise ValueError("An unrelated reference requires --unrelated-reference-box around its object.")
            report["unrelated_reference"] = str(args.unrelated_reference.resolve())
            report["unrelated_reference_box"] = unrelated_box
            unrelated.save(args.output_dir / "unrelated-reference.png")
            cases.append(("unrelated_reference_box_control", lambda: engine.predict_visual(target, "external-target", unrelated, unrelated_box)))
        for name, operation in cases:
            proposals, entry = measure(name, operation)
            entry.update(artifacts(name, target, proposals))
            if name == "unrelated_reference_box_control":
                report["quality_checks"]["unrelated_reference_returned_zero_instances"] = not proposals
                entry["expected_proposal_count"] = 0
            write_report()
            if name == "external_full_reference_box":
                if not proposals:
                    raise AssertionError("This positive-control reference produced no proposals for tracker validation.")
                seed = proposals[0]["mask"]
                ys, xs = np.nonzero(seed)
                center = int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))
                positive = {"x": int(xs[center]), "y": int(ys[center]), "label": 1}
                background = np.argwhere(~seed)
                if not len(background):
                    raise AssertionError("No background pixel is available for negative-click validation.")
                nearest = int(np.argmin((background[:, 1] - positive["x"]) ** 2 + (background[:, 0] - positive["y"]) ** 2))
                negative = {"x": int(background[nearest, 1]), "y": int(background[nearest, 0]), "label": 0}
                target_key = engine._image_key(target, "external-target")
                report["tracker_prompts"] = {"positive": positive, "negative": negative,
                                             "detector_logits_cached": (target_key, _mask_digest(seed)) in engine._seed_cache}
                part = {"id": "visual-cached-refinement", "points": [positive, negative], "seed_mask": encode_mask(seed)}
                cached, cached_case = measure("visual_seed_tracker_cached", lambda: engine.predict_points(target, "external-target", part))
                cached_case.update(artifacts("visual_seed_tracker_cached", target, [{"mask": cached}]))
                engine._seed_cache.clear()
                engine._part_cache.clear()
                part["id"] = "visual-persisted-refinement"
                persisted, persisted_case = measure("visual_seed_tracker_persisted", lambda: engine.predict_points(target, "external-target", part))
                persisted_case.update(artifacts("visual_seed_tracker_persisted", target, [{"mask": persisted}]))
                union = int(np.logical_or(cached, persisted).sum())
                report["tracker_comparison"] = {
                    "cached_shape": list(cached.shape), "persisted_shape": list(persisted.shape),
                    "cached_positive_included": bool(cached[positive["y"], positive["x"]]),
                    "cached_negative_excluded": not bool(cached[negative["y"], negative["x"]]),
                    "persisted_positive_included": bool(persisted[positive["y"], positive["x"]]),
                    "persisted_negative_excluded": not bool(persisted[negative["y"], negative["x"]]),
                    "cached_vs_persisted_mask_iou": float(np.logical_and(cached, persisted).sum() / union) if union else 1.0,
                    "changed_pixels": int(np.count_nonzero(cached != persisted)),
                    "note": "Agreement between two seed representations, not IoU against annotation ground truth.",
                }
                for mode in ("cached", "persisted"):
                    report["quality_checks"][mode + "_refinement_respected_clicks"] = (
                        report["tracker_comparison"][mode + "_positive_included"]
                        and report["tracker_comparison"][mode + "_negative_excluded"]
                    )
                write_report()
        if not all(report["quality_checks"].values()):
            report["quality_status"] = "control_failures_observed_not_accuracy_scored"
        report["status"] = "execution_passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        exit_code = 2
    finally:
        write_report()
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
