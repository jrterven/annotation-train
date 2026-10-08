"""CPU-only validation for native boxes and task-local detection work."""
import copy
import math


def bbox(value, width, height):
    if (not isinstance(value, list) or len(value) != 4
            or any(type(n) not in (int, float) or not math.isfinite(n) for n in value)):
        raise ValueError("A bounding box needs four finite numbers (x, y, width, height).")
    x, y, w, h = value
    if not (0 <= x < x + w <= width and 0 <= y < y + h <= height):
        raise ValueError("The bounding box is outside the image or empty.")
    return list(value)


def box_record(record, width, height):
    if any(key in record for key in ("mask", "components", "controls", "preview")):
        raise ValueError("A detection must contain a box, not mask geometry.")
    crowd = record.get("iscrowd", 0)
    if type(crowd) is not int or crowd not in (0, 1):
        raise ValueError("iscrowd must be 0 or 1.")
    return {"id": record["id"], "kind": "bbox", "category_id": record["category_id"],
            "iscrowd": crowd, "bbox": bbox(record.get("bbox"), width, height)}


def detection_workspace(value, width, height, identity, category, annotations):
    if not isinstance(value, dict):
        raise ValueError("Detection workspace must be an object.")
    result = {"draft": None, "proposals": [], "adjustment": None}
    if not isinstance(value.get("proposals", []), list):
        raise ValueError("Detection proposals must be a list.")
    for proposal in value.get("proposals", []):
        identity(proposal)
        category(proposal)
        clean = box_record(proposal, width, height)
        score = proposal.get("score")
        if type(score) not in (float, int) or not math.isfinite(score) or not isinstance(proposal.get("selected"), bool):
            raise ValueError("Invalid detection proposal score or selection.")
        result["proposals"].append({**clean, "score": score, "selected": proposal["selected"]})
    draft = value.get("draft")
    if draft is not None:
        identity(draft)
        category(draft)
        clean = {"id": draft["id"], "category_id": draft["category_id"], "points": []}
        for field in ("bbox", "prompt_bbox"):
            if draft.get(field) is not None:
                clean[field] = bbox(draft[field], width, height)
        points = draft.get("points", [])
        if not isinstance(points, list):
            raise ValueError("Detection points must be a list.")
        for point in points:
            if (not isinstance(point, dict) or type(point.get("label")) is not int or point["label"] not in (0, 1)
                    or any(type(point.get(axis)) not in (float, int) or not math.isfinite(point[axis])
                           or not 0 <= point[axis] < limit for axis, limit in (("x", width), ("y", height)))):
                raise ValueError("Invalid detection point.")
            clean["points"].append({k: point[k] for k in ("x", "y", "label")})
        result["draft"] = clean
    adjustment = value.get("adjustment")
    if adjustment is not None:
        if not isinstance(adjustment, dict):
            raise ValueError("Invalid box adjustment.")
        target = next((a for a in annotations if a.get("kind") == "bbox" and a["id"] == adjustment.get("target_id")), None)
        if draft and draft["id"] == adjustment.get("target_id"):
            target = result["draft"]
        if target is None:
            raise ValueError("The adjusted box no longer exists.")
        base = bbox(adjustment.get("base_bbox"), width, height)
        if base != target.get("bbox"):
            raise ValueError("The adjusted box has changed.")
        result["adjustment"] = {"target_id": target["id"], "base_bbox": base,
                                "bbox": bbox(adjustment.get("bbox"), width, height)}
    return copy.deepcopy(result)
