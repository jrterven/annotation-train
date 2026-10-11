"""Revision-checked proposals through the existing annotation persistence path."""

import copy
from fastapi import HTTPException
from app.storage import RevisionConflict
from .service import require


def apply(backend, batch_id):
    with backend.repo.transaction() as records:
        operation = require(records, "operation", batch_id)
        if operation.get("operation") != "predict" or operation["status"] != "completed":
            raise ValueError("Predictions are not ready")
        if operation.get("purpose") == "preview":
            raise ValueError("Use Preannotate to create reviewable proposals")
        if operation.get("applied"):
            return {"image_ids": operation["applied"]}
    images = {im["id"]: im for im in operation["input"]["images"]}
    classes = {c["category_id"] for c in operation["input"]["classes"]}
    results = operation.get("result", {}).get("images", [])
    if not isinstance(results, list) or len(results) > len(images):
        raise ValueError("Invalid prediction result")
    seen = set()
    pixel_budget = 1_024_000_000
    staged = []
    for result in results:
        image_id = result.get("image_id")
        if image_id not in images or image_id in seen:
            raise ValueError("Predictions refer to an unassigned image")
        seen.add(image_id)
        image = images[image_id]
        proposals = result.get("proposals")
        if not isinstance(proposals, list) or len(proposals) > 10000:
            raise ValueError("Too many proposals")
        clean = []
        for index, proposal in enumerate(proposals):
            if proposal.get("category_id") not in classes:
                raise ValueError("Predicted class does not belong to the trained model")
            if operation["task"] == "segmentation":
                if proposal.get("mask", {}).get("size") != [image["height"], image["width"]]:
                    raise ValueError("Prediction mask does not match original pixels")
                pixel_budget -= image["height"] * image["width"]
                if pixel_budget < 0:
                    raise ValueError("Prediction masks exceed the output budget")
            elif proposal.get("kind") != "bbox":
                raise ValueError("Detection predictions must be boxes")
            clean.append(
                {
                    **proposal,
                    "id": f"training:{batch_id}:{image_id}:{index}",
                    "selected": False,
                    "iscrowd": 0,
                }
            )
        staged.append((image, clean))

    def mutate(store, *_):
        # Validate the entire batch before saving any image. IDs also make a
        # replay after a crash safe without re-appending proposals.
        changes = []
        with store._connect() as connection:
            category_ids = {row[0] for row in connection.execute("SELECT id FROM categories")}
            if classes - category_ids:
                raise RevisionConflict(
                    "A trained class was removed. Restore it before applying predictions."
                )
            for image, proposals in staged:
                state = store.get_state(image["id"], include_previews=False)
                target = (
                    state.setdefault(
                        "detection", {"draft": None, "proposals": [], "adjustment": None}
                    )
                    if operation["task"] == "detection"
                    else state
                )
                existing = {a["id"] for a in target["proposals"]} | {
                    a["id"] for a in state["annotations"]
                }
                additions = [a for a in proposals if a["id"] not in existing]
                if not additions:
                    continue
                if state["revision"] != image["revision"]:
                    raise RevisionConflict(
                        "An image changed after prediction. Generate fresh proposals."
                    )
                state["schema_version"] = 2
                target["proposals"].extend(additions)
                row = store._image_row(connection, image["id"])
                if row["sha256"] != image["sha256"] or (row["width"], row["height"]) != (
                    image["width"],
                    image["height"],
                ):
                    raise RevisionConflict("The original image changed. Generate fresh proposals.")
                store._validate_state(state, row, category_ids)
                changes.append(state)
        for state in changes:
            store.save_state(state["image_id"], state, include_previews=False)
        return [image["id"] for image, _ in staged]

    if backend.hosted:
        from app.hosted.quota import mutate_project

        applied = mutate_project(
            backend.settings,
            backend.db,
            backend.objects,
            backend.project_id,
            backend.user_id,
            mutate,
        )
    else:
        # Local writes retain optimistic revision checks. A retry only appends
        # missing stable proposal IDs if a process stopped between images.
        applied = mutate(backend.project)
    with backend.repo.transaction() as records:
        operation = require(records, "operation", batch_id)
        operation["applied"] = applied
        records.put("operation", batch_id, operation)
    return {"image_ids": applied}
