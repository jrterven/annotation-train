"""Portable YOLO exports. No model dependency and no changes to source annotations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid
import zipfile

import numpy as np
from pydantic import BaseModel, ConfigDict
from typing import Literal
from shapely.geometry import MultiPoint
from shapely.ops import nearest_points

from .annotations import bbox
from .images import Image
from .storage import RevisionConflict, _json, _relative_file


class ExportOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Literal["segmentation", "detection"] = "segmentation"
    include_images: bool = False
    include_empty: bool = False


class ExportRequest(ExportOptions):
    snapshot: str
    allow_lossy: bool = False


class CocoOptions(BaseModel):
    unique: bool = False
    task: Literal["all", "segmentation", "detection"] = "all"


def snapshot(connection, options: ExportOptions, ready_ids=None):
    categories = [json.loads(row[0]) for row in connection.execute("SELECT data FROM categories ORDER BY id")]
    images = [dict(row) for row in connection.execute("SELECT id,file_name,width,height,sha256,revision FROM images ORDER BY id")
              if ready_ids is None or row["id"] in ready_ids]
    token = hashlib.sha256(_json([options.model_dump(), categories, images]).encode()).hexdigest()
    return categories, images, token


def polygon(components):
    """One walk per instance, with retraced bridges between nearest vertices.

    Hole rings are deliberately omitted. Exact components remain unchanged.
    GEOS nearest_points avoids allocating an N*M NumPy distance matrix.
    """
    rings = [np.asarray(c["outer"], dtype=float) for c in components]
    rings.sort(key=lambda r: (float(r[:, 0].min()), float(r[:, 1].min()), len(r)))
    if not rings:
        raise ValueError("No polygon geometry")
    merged = rings[0].tolist()
    for ring in rings[1:]:
        left, right = nearest_points(MultiPoint(merged), MultiPoint(ring))
        i = next(
            j for j, p in enumerate(merged) if tuple(p) == tuple(left.coords[0]))
        j = next(j for j, p in enumerate(ring) if tuple(p) == tuple(right.coords[0]))
        walk = np.concatenate((ring[j:], ring[:j + 1])).tolist()
        merged = merged[:i + 1] + walk + [merged[i]] + merged[i + 1:]
    return merged


def prepare(store, options: ExportOptions, ready_ids=None, *, write_entry=None, expected_snapshot=None):
    # Hold one SQLite read snapshot while processing one image's geometry at a
    # time. Only compact filenames/revisions and the review report accumulate.
    with store._connect() as connection:
        connection.execute("BEGIN")
        categories, images, token = snapshot(connection, options, ready_ids)
        if expected_snapshot is not None and token != expected_snapshot:
            raise RevisionConflict("Annotations, images or classes changed. Review the export again.")
        classes = [{"index": i, "category_id": c["id"], "name": c["name"]} for i, c in enumerate(categories)]
        indices = {c["category_id"]: c["index"] for c in classes}
        warnings, errors, exported = [], [], []
        names, directories = {}, {}
        annotation_count = 0
        for row in images:
            state = json.loads(connection.execute("SELECT state FROM images WHERE id=?", (row["id"],)).fetchone()[0])
            annotations = [a for a in state["annotations"]
                           if (a.get("kind") == "bbox") == (options.task == "detection")]
            if not annotations and not options.include_empty:
                continue
            name = _relative_file(row["file_name"])
            label = str(Path(name).with_suffix(".txt"))
            image_name = str(Path(name).with_suffix(".png"))
            destinations = ["labels/" + label]
            if options.include_images:
                destinations.append("images/" + image_name)
            for destination in destinations:
                folded = destination.casefold()
                parents = [str(parent) for parent in Path(folded).parents if str(parent) != "."]
                collision = names.get(folded) or directories.get(folded) or next(
                    (names[parent] for parent in parents if parent in names), None)
                if collision:
                    errors.append({"image_id": row["id"], "file_name": name,
                                   "reason": f"Export filename collides with {collision}. Rename/reimport one image."})
                names[folded] = name
                for parent in parents:
                    directories[parent] = name
            lines = []
            for ann in annotations:
                reference = {"image_id": row["id"], "file_name": name, "annotation_id": ann["id"]}
                if ann.get("iscrowd", 0):
                    errors.append({**reference, "reason": "YOLO has no iscrowd representation. Use COCO or edit this annotation."})
                    continue
                try:
                    if options.task == "detection":
                        x, y, w, h = bbox(ann["bbox"], row["width"], row["height"])
                        values = [(x + w / 2) / row["width"], (y + h / 2) / row["height"], w / row["width"], h / row["height"]]
                    else:
                        components = ann["components"]
                        reasons = []
                        if any(c.get("holes") for c in components):
                            reasons.append("holes filled")
                        if len(components) > 1:
                            reasons.append("separate parts connected")
                        if reasons:
                            warnings.append({**reference, "reason": "; ".join(reasons)})
                        points = polygon(components)
                        if len(points) < 3:
                            raise ValueError("Polygon needs at least three points")
                        values = [n / dimension for point in points for n, dimension in zip(point, (row["width"], row["height"]))]
                    if not all(np.isfinite(n) and 0 <= n <= 1 for n in values):
                        raise ValueError("Invalid normalized coordinates")
                    lines.append(str(indices[ann["category_id"]]) + " " + " ".join(format(n, ".12g") for n in values))
                except (KeyError, ValueError, TypeError) as exc:
                    errors.append({**reference, "reason": str(exc)})
            annotation_count += len(annotations)
            exported.append({"id": row["id"], "source": name,
                             "image": "images/" + image_name, "label": "labels/" + label})
            if write_entry is not None:
                write_entry({"image": row, "label": label, "image_name": image_name,
                             "text": "\n".join(lines) + ("\n" if lines else "")})
        return {"snapshot": token, "options": options.model_dump(), "classes": classes,
                "image_count": len(exported), "annotation_count": annotation_count,
                "warnings": warnings, "errors": errors, "images": exported}


def generate(store, request: ExportRequest, directory: Path, image_path, ready_ids=None):
    options = ExportOptions(**request.model_dump(include=set(ExportOptions.model_fields)))
    directory.mkdir(parents=True, exist_ok=True)
    export_id = str(uuid.uuid4())
    destination = directory / f"{export_id}.zip"
    try:
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            def write_entry(entry):
                archive.writestr("labels/" + entry["label"], entry["text"])
                if options.include_images:
                    with Image.open(image_path(entry["image"]["id"])) as image:
                        if image.size != (entry["image"]["width"], entry["image"]["height"]):
                            raise ValueError("Source image dimensions changed.")
                        clean = image.convert("RGB")
                        clean.info.clear()
                        with archive.open("images/" + entry["image_name"], "w", force_zip64=True) as output:
                            clean.save(output, format="PNG")
            report = prepare(store, options, ready_ids, write_entry=write_entry,
                             expected_snapshot=request.snapshot)
            if report["errors"]:
                raise ValueError("Resolve the errors in the export preview before downloading.")
            if report["warnings"] and not request.allow_lossy:
                raise ValueError("Review and accept the polygon approximations before exporting.")
            if not report["image_count"]:
                raise ValueError("No images to export for this task.")
            archive.writestr("report.json", json.dumps(report, ensure_ascii=False, indent=2))
            archive.writestr("classes.json", json.dumps({c["index"]: c["name"] for c in report["classes"]}, ensure_ascii=False, indent=2))
            archive.writestr("README.txt", "YOLO annotations\n\nlabels/ contains one TXT per exported image.\n"
                "classes.json maps zero-based class indices to names. report.json maps original paths and records approximations.\n"
                "Images, when included, are lossless PNGs on the original pixel grid, without EXIF rotation.\n"
                "When exporting labels only, place matching images under images/ with the same relative stems. Preserve the stored pixel grid (do not apply EXIF rotation).\n"
                "Create your own disjoint train/validation splits and dataset YAML with the classes in this order.\n"
                "Empty labels are included only when requested; they do not imply the image was reviewed.\n"
                "YOLO polygons can approximate pixel masks. COCO RLE preserves the original masks.\n")
        return export_id, destination
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def export_identity(value):
    if str(uuid.UUID(value)) != value:
        raise ValueError("Invalid export identifier")
    return value
