"""Local projects: transactional state, portable image references, and COCO."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Iterable
import uuid

import numpy as np
from .images import Image
from pycocotools import mask as coco_mask

from .annotations import box_record, bbox, detection_workspace
from .geometry import (controls_for_components, decode_mask, encode_mask, mask_payload,
                       mask_preview, rasterize_components, validate_components)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DATABASE_NAME = "proyecto.sqlite3"
PALETTE = ["#8B8DE3", "#63B8A4", "#D5A565", "#CB819D", "#75A8CC", "#AAA16C"]


class RevisionConflict(ValueError):
    """The client attempted to overwrite a newer saved image revision."""


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError) as error:
        raise ValueError("Data must be valid JSON with finite numbers.") from error


def _root_path(directory: Path, image_root: str | Path) -> Path:
    root = Path(image_root).expanduser().resolve()
    if directory == root or directory.is_relative_to(root):
        raise ValueError("The project must be stored outside the image folder.")
    if not root.is_dir():
        raise ValueError(f"Image folder does not exist: {root}")
    return root


def _relative_file(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError("file_name must be a nonempty relative path.")
    # COCO datasets produced on Windows commonly use backslash separators.
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or re.match(r"^[a-zA-Z]:", normalized) or ".." in path.parts:
        raise ValueError(f"The path must stay within the image root: {value}")
    return path.as_posix()


def _file_path(root: Path, name: str) -> Path:
    path = (root / _relative_file(name)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"The image is outside the selected root: {name}")
    if not path.is_file():
        raise ValueError(f"Image not found: {name}")
    return path


def _image_details(path: Path) -> dict:
    try:
        with Image.open(path) as image:
            width, height = image.size  # Deliberately do not apply EXIF transpose.
            image.verify()
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        stat = path.stat()
        return {"width": width, "height": height, "sha256": digest.hexdigest(),
                "file_size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except (OSError, SyntaxError) as error:
        raise ValueError(f"Could not read image {path.name}: {error}") from error


def _schema(connection: sqlite3.Connection, directory: Path, root: Path) -> None:
    connection.executescript("""
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE categories(id INTEGER PRIMARY KEY, data TEXT NOT NULL);
        CREATE TABLE images(
            id INTEGER PRIMARY KEY, file_name TEXT UNIQUE NOT NULL,
            width INTEGER NOT NULL, height INTEGER NOT NULL,
            sha256 TEXT NOT NULL, file_size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
            data TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL
        );
        CREATE TABLE coco_ids(
            internal_id TEXT PRIMARY KEY, coco_id INTEGER UNIQUE NOT NULL,
            image_id INTEGER NOT NULL REFERENCES images(id)
        );
        CREATE TABLE annotation_sources(
            internal_id TEXT PRIMARY KEY REFERENCES coco_ids(internal_id),
            record TEXT NOT NULL, mask TEXT NOT NULL
        );
    """)
    connection.executemany("INSERT INTO meta VALUES(?,?)", [
        ("schema_version", "1"), ("name", directory.name),
        ("image_root", _root_reference(root, directory)), ("coco_metadata", "{}")])


def _root_reference(root: Path, directory: Path) -> str:
    # Only bundled image folders move with the project. External originals keep
    # their absolute location when the annotations directory alone is moved.
    return root.relative_to(directory).as_posix() if root.is_relative_to(directory) else str(root)


@contextmanager
def _connection(path: str | Path):
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _initial_state(image_id: int) -> dict:
    return {"image_id": image_id, "revision": 0, "annotations": [], "draft": None, "proposals": []}


def _insert_image(connection: sqlite3.Connection, image_id: int, name: str, details: dict,
                  metadata: dict | None = None, state: dict | None = None) -> None:
    connection.execute("INSERT INTO images VALUES(?,?,?,?,?,?,?,?,?,?)", (
        image_id, name, details["width"], details["height"], details["sha256"],
        details["file_size"], details["mtime_ns"], _json(metadata or {}),
        0, _json(state or _initial_state(image_id))))


def _atomic_text(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ProjectStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()
        self.database = self.directory / DATABASE_NAME
        if not self.database.is_file():
            raise ValueError("This folder does not contain a project.")
        with self._connect() as connection:
            version = connection.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if not version or version[0] not in ("1", "2"):
                raise ValueError("This project version is not supported.")

    def require_client(self, version: str | None) -> None:
        with self._connect() as connection:
            current = connection.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        if current == "2" and version != "2":
            raise ValueError("This project uses annotation format v2. Reload/update the editor before continuing.")

    def _connect(self):
        return _connection(self.database)

    def _root(self, connection: sqlite3.Connection) -> Path:
        reference = connection.execute("SELECT value FROM meta WHERE key='image_root'").fetchone()[0]
        return (self.directory / reference).resolve()

    @classmethod
    def open(cls, directory: str | Path, image_root: str | Path | None = None,
             files: list[str] | None = None, recursive: bool = True) -> "ProjectStore":
        destination = Path(directory).expanduser().resolve()
        database = destination / DATABASE_NAME
        if database.exists():
            store = cls(destination)
            if image_root is not None:
                with store._connect() as connection:
                    previous_root = store._root(connection)
                if Path(image_root).expanduser().resolve() != previous_root:
                    store.relink(image_root)
            # Reopening must also work with temporarily missing originals, so
            # the user can explicitly relink them without losing saved state.
            with store._connect() as connection:
                root = store._root(connection)
                registered = [row[0] for row in connection.execute("SELECT file_name FROM images")]
            if root.is_dir():
                store._add_images(root, [name for name in registered if (root / name).exists()], recursive)
                if files is not None:
                    store._add_images(root, files, recursive)
            elif files is not None:
                raise ValueError("Relink the image folder before adding files.")
            return store
        if image_root is None or (isinstance(image_root, str) and not image_root.strip()):
            raise ValueError("Select an image folder to create a project.")
        root = _root_path(destination, image_root)
        records = cls._discover(root, files, recursive)
        destination.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".project.", suffix=".sqlite3", dir=destination)
        os.close(descriptor)
        try:
            with _connection(temporary) as connection:
                _schema(connection, destination, root)
                for image_id, (name, details) in enumerate(records, 1):
                    _insert_image(connection, image_id, name, details)
            os.replace(temporary, database)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return cls(destination)

    @staticmethod
    def _discover(root: Path, files: list[str] | None, recursive: bool) -> list[tuple[str, dict]]:
        if files is not None:
            if not isinstance(files, list):
                raise ValueError("files must be a list of relative paths.")
            names = sorted({_relative_file(name) for name in files}, key=str.casefold)
        else:
            candidates: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
            names = sorted((path.relative_to(root).as_posix() for path in candidates
                            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS), key=str.casefold)
        records = []
        for name in names:
            path = _file_path(root, name)
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image format: {name}")
            records.append((name, _image_details(path)))
        return records

    def _add_images(self, root: Path, files: list[str] | None, recursive: bool) -> None:
        records = self._discover(root, files, recursive)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = {row["file_name"]: row for row in connection.execute("SELECT * FROM images")}
            next_id = connection.execute("SELECT COALESCE(MAX(id),0)+1 FROM images").fetchone()[0]
            for name, details in records:
                if name in existing:
                    if details["sha256"] != existing[name]["sha256"]:
                        raise ValueError(f"The image has changed since import: {name}")
                    continue
                _insert_image(connection, next_id, name, details)
                next_id += 1

    def project(self) -> dict:
        with self._connect() as connection:
            name = connection.execute("SELECT value FROM meta WHERE key='name'").fetchone()[0]
            categories = [json.loads(row[0]) for row in connection.execute("SELECT data FROM categories ORDER BY id")]
            images = [{"id": row["id"], "file_name": row["file_name"], "width": row["width"],
                       "height": row["height"], "annotation_count": len(json.loads(row["state"])["annotations"]),
                       "annotation_counts": {task: sum((a.get("kind") == "bbox") == (task == "detection")
                          for a in json.loads(row["state"])["annotations"]) for task in ("segmentation", "detection")}}
                      for row in connection.execute("SELECT * FROM images ORDER BY file_name COLLATE NOCASE,id")]
            return {"name": name, "directory": str(self.directory), "image_root": str(self._root(connection)),
                    "categories": categories, "images": images}

    @staticmethod
    def _image_row(connection: sqlite3.Connection, image_id: int) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
        if row is None:
            raise ValueError("The image does not belong to this project.")
        return row

    def image_path(self, image_id: int) -> Path:
        with self._connect() as connection:
            row = self._image_row(connection, image_id)
            path = _file_path(self._root(connection), row["file_name"])
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (row["file_size"], row["mtime_ns"]):
            if _image_details(path)["sha256"] != row["sha256"]:
                raise ValueError(f"The image has changed since import: {row['file_name']}")
        return path

    def get_state(self, image_id: int, *, include_previews: bool = True) -> dict:
        with self._connect() as connection:
            state = json.loads(self._image_row(connection, image_id)["state"])
            colors = {row["id"]: json.loads(row["data"])["color"]
                      for row in connection.execute("SELECT * FROM categories")}
        # Previews are derived, never persisted as duplicate base64 data.
        for annotation in state["annotations"] + state["proposals"]:
            if annotation.get("kind") == "bbox":
                continue
            if include_previews:
                mask = decode_mask(annotation["mask"])
                annotation["preview"] = mask_preview(mask, colors[annotation["category_id"]])
            if "controls" not in annotation:
                height, width = annotation["mask"]["size"]
                annotation["controls"] = controls_for_components(annotation["components"], width, height)
        if state["draft"]:
            color = colors[state["draft"]["category_id"]]
            for part in state["draft"]["parts"]:
                if part.get("mask"):
                    if include_previews:
                        mask = decode_mask(part["mask"])
                        part["preview"] = mask_preview(mask, color)
                    if "controls" not in part:
                        height, width = part["mask"]["size"]
                        part["controls"] = controls_for_components(part["components"], width, height)
        return state

    @staticmethod
    def _mask_record(record: dict, width: int, height: int, allow_empty: bool = False) -> dict:
        clean = copy.deepcopy(record)
        clean.pop("preview", None)
        mask = decode_mask(clean.get("mask"))
        if mask.shape != (height, width):
            raise ValueError("The mask does not match the image dimensions.")
        if not allow_empty and not mask.any():
            raise ValueError("An empty segmentation cannot be saved.")
        clean["mask"] = encode_mask(mask)
        components = clean.get("components")
        if components is None:
            components = mask_payload(mask)["components"]
        if not np.array_equal(rasterize_components(components, width, height), mask):
            raise ValueError("The contours and mask do not match; apply the geometry edit first.")
        clean["components"] = components
        controls = clean.get("controls")
        if controls is None:
            controls = controls_for_components(components, width, height)
        validate_components(controls, width, height)
        if mask.any() and not controls:
            raise ValueError("A segmentation needs editable contours.")
        clean["controls"] = controls
        return clean

    def _validate_state(self, state: Any, row: sqlite3.Row, category_ids: set[int]) -> dict:
        if not isinstance(state, dict) or state.get("image_id") != row["id"]:
            raise ValueError("The state belongs to another image.")
        if type(state.get("revision")) is not int or state["revision"] < 0:
            raise ValueError("The state revision is invalid.")
        result = {"image_id": row["id"], "revision": state["revision"], "draft": None}
        if not isinstance(state.get("annotations", []), list):
            raise ValueError("annotations must be a list.")
        version = state.get("schema_version", 1)
        if type(version) is not int or version not in (1, 2):
            raise ValueError("Unsupported annotation state version.")
        if version == 1 and ("detection" in state or any(a.get("kind") == "bbox" for a in (state.get("annotations") or []) if isinstance(a, dict))):
            raise ValueError("Boxes require annotation state version 2.")
        if version == 2:
            result["schema_version"] = 2
        identities = set()
        width, height = row["width"], row["height"]

        def identity(value: dict) -> None:
            if not isinstance(value, dict) or not isinstance(value.get("id"), str) or not value["id"]:
                raise ValueError("Each object needs a text identifier.")
            if value["id"] in identities:
                raise ValueError("Object identifiers must be unique.")
            identities.add(value["id"])

        def category(value: dict) -> None:
            if type(value.get("category_id")) is not int or value["category_id"] not in category_ids:
                raise ValueError("The category does not belong to this project.")

        for collection in ("annotations", "proposals"):
            if not isinstance(state.get(collection, []), list):
                raise ValueError(f"{collection} must be a list.")
            result[collection] = []
            for annotation in state.get(collection, []):
                identity(annotation)
                category(annotation)
                kind = annotation.get("kind", "segmentation")
                if kind not in ("segmentation", "bbox") or (collection == "proposals" and kind != "segmentation"):
                    raise ValueError("Invalid annotation kind.")
                clean = (box_record(annotation, width, height) if kind == "bbox"
                         else self._mask_record(annotation, width, height))
                if type(clean.get("iscrowd", 0)) is not int or clean.get("iscrowd", 0) not in (0, 1):
                    raise ValueError("iscrowd must be 0 or 1.")
                clean["iscrowd"] = clean.get("iscrowd", 0)
                if collection == "proposals":
                    score = clean.get("score")
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                        raise ValueError("The proposal score is invalid.")
                    if not isinstance(clean.get("selected"), bool):
                        raise ValueError("selected must be a boolean.")
                result[collection].append(clean)
        draft = state.get("draft")
        if draft is not None:
            identity(draft)
            category(draft)
            clean = copy.deepcopy(draft)
            if not isinstance(clean.get("parts"), list) or not clean["parts"]:
                raise ValueError("The draft needs at least one part.")
            part_ids = set()
            clean["parts"] = []
            for part in draft["parts"]:
                if not isinstance(part, dict) or not isinstance(part.get("id"), str) or not part["id"] or part["id"] in part_ids:
                    raise ValueError("Part identifiers must be unique.")
                part_ids.add(part["id"])
                item = copy.deepcopy(part)
                item.pop("preview", None)
                points = item.get("points", [])
                if not isinstance(points, list):
                    raise ValueError("points must be a list.")
                for point in points:
                    if not isinstance(point, dict) or type(point.get("label")) is not int or point["label"] not in (0, 1):
                        raise ValueError("A point prompt must be positive or negative.")
                    for axis, limit in (("x", width), ("y", height)):
                        number = point.get(axis)
                        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not 0 <= number < limit:
                            raise ValueError("The point is outside the image.")
                if item.get("box") is not None:
                    box = item["box"]
                    if (not isinstance(box, list) or len(box) != 4 or any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) for n in box)
                            or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
                        raise ValueError("The bounding box is outside the image or empty.")
                if "polygon" in item:
                    polygon = item["polygon"]
                    if (not isinstance(polygon, dict) or not isinstance(polygon.get("closed"), bool)
                            or not isinstance(polygon.get("vertices"), list)):
                        raise ValueError("The polygon needs vertices and a boolean closed value.")
                    vertices = polygon["vertices"]
                    if polygon["closed"] and len(vertices) < 3:
                        raise ValueError("A closed polygon needs at least three vertices.")
                    for vertex in vertices:
                        if (not isinstance(vertex, list) or len(vertex) != 2
                                or any(isinstance(n, bool) or not isinstance(n, (int, float))
                                       or not math.isfinite(n) for n in vertex)):
                            raise ValueError("Each polygon vertex must contain two finite numbers.")
                        if not 0 <= vertex[0] <= width or not 0 <= vertex[1] <= height:
                            raise ValueError("A polygon vertex is outside the image.")
                    # A draft can be incomplete or intersect itself while being
                    # drawn. Keep it editable; /geometry validates topology when
                    # the user explicitly turns it into a SAM mask prompt.
                    item["polygon"] = {"vertices": vertices, "closed": polygon["closed"]}
                if item.get("mask") is not None:
                    item = self._mask_record(item, width, height, allow_empty=True)
                if item.get("seed_mask") is not None:
                    seed = decode_mask(item["seed_mask"])
                    if seed.shape != (height, width):
                        raise ValueError("The initial mask does not match the image.")
                    item["seed_mask"] = encode_mask(seed)
                clean["parts"].append(item)
            if clean.get("active_part_id") not in part_ids:
                raise ValueError("The active part does not exist in the draft.")
            result["draft"] = clean
        if version == 2:
            result["detection"] = detection_workspace(state.get("detection", {}), width, height,
                                                        identity, category, result["annotations"])
        _json(result)
        return result

    def save_state(self, image_id: int, state: dict, *, include_previews: bool = True) -> dict:
        if not isinstance(state, dict):
            raise ValueError("The state must be a JSON object.")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._image_row(connection, image_id)
            if state.get("revision") != row["revision"]:
                raise RevisionConflict("The image has newer changes. Reload its state before saving.")
            schema = connection.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
            if schema == "2" and state.get("schema_version") != 2:
                raise ValueError("This project uses annotation format v2. Reload/update the editor before saving.")
            category_ids = {item[0] for item in connection.execute("SELECT id FROM categories")}
            clean = self._validate_state(state, row, category_ids)
            for annotation in clean["annotations"] + clean["proposals"]:
                found = connection.execute("SELECT * FROM coco_ids WHERE internal_id=?", (annotation["id"],)).fetchone()
                if found and found["image_id"] != image_id:
                    raise ValueError("The object identifier belongs to another image.")
                if not found:
                    coco_id = connection.execute("SELECT COALESCE(MAX(coco_id),0)+1 FROM coco_ids").fetchone()[0]
                    connection.execute("INSERT INTO coco_ids VALUES(?,?,?)", (annotation["id"], coco_id, image_id))
            if clean.get("schema_version") == 2:
                connection.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
            clean["revision"] += 1
            connection.execute("UPDATE images SET revision=?,state=? WHERE id=?", (clean["revision"], _json(clean), image_id))
        return self.get_state(image_id, include_previews=include_previews)

    def relink(self, image_root: str | Path) -> dict:
        root = _root_path(self.directory, image_root)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for row in connection.execute("SELECT * FROM images").fetchall():
                details = _image_details(_file_path(root, row["file_name"]))
                if (details["width"], details["height"], details["sha256"]) != (row["width"], row["height"], row["sha256"]):
                    raise ValueError(f"The image does not match the original: {row['file_name']}")
                connection.execute("UPDATE images SET file_size=?,mtime_ns=? WHERE id=?", (details["file_size"], details["mtime_ns"], row["id"]))
            connection.execute("UPDATE meta SET value=? WHERE key='image_root'", (_root_reference(root, self.directory),))
        return self.project()

    @staticmethod
    def _category_fields(name: Any, color: Any) -> tuple[str, str]:
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 160:
            raise ValueError("The class name must contain 1 to 160 characters.")
        if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError("The color must use #RRGGBB format.")
        return name.strip(), color.upper()

    def add_category(self, name: str, color: str) -> dict:
        name, color = self._category_fields(name, color)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = [json.loads(row[0]) for row in connection.execute("SELECT data FROM categories")]
            if any(category["name"].casefold() == name.casefold() for category in existing):
                raise ValueError("A class with this name already exists.")
            identity = max([category["id"] for category in existing] + [0]) + 1
            category = {"id": identity, "name": name, "color": color, "supercategory": ""}
            connection.execute("INSERT INTO categories VALUES(?,?)", (identity, _json(category)))
        return category

    def update_category(self, id: int, updates: dict) -> dict:
        if not isinstance(updates, dict) or set(updates) - {"name", "color"}:
            raise ValueError("Only a class name or color can be changed.")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT data FROM categories WHERE id=?", (id,)).fetchone()
            if row is None:
                raise ValueError("The class does not exist.")
            category = json.loads(row[0])
            category.update(updates)
            category["name"], category["color"] = self._category_fields(category["name"], category["color"])
            for other in connection.execute("SELECT data FROM categories WHERE id<>?", (id,)):
                if json.loads(other[0])["name"].casefold() == category["name"].casefold():
                    raise ValueError("A class with this name already exists.")
            connection.execute("UPDATE categories SET data=? WHERE id=?", (_json(category), id))
        return category

    @classmethod
    def import_coco(cls, directory: str | Path, image_root: str | Path, json_path: str | Path) -> "ProjectStore":
        destination = Path(directory).expanduser().resolve()
        if (destination / DATABASE_NAME).exists():
            raise ValueError("Import COCO into a new project; existing projects cannot be merged.")
        root = _root_path(destination, image_root)
        try:
            source_text = Path(json_path).expanduser().read_text(encoding="utf-8-sig")
            source = json.loads(source_text)
        except (OSError, ValueError) as error:
            raise ValueError(f"Could not read the COCO JSON: {error}") from error
        if not isinstance(source, dict) or any(not isinstance(source.get(key), list) for key in ("images", "annotations", "categories")):
            raise ValueError("An instance COCO dataset with images, annotations, and categories is required.")
        _json(source)

        def indexed(items: list, label: str) -> dict[int, dict]:
            result = {}
            for item in items:
                if not isinstance(item, dict) or type(item.get("id")) is not int:
                    raise ValueError(f"Each item in {label} must have an integer ID.")
                if item["id"] in result:
                    raise ValueError(f"Duplicate ID in {label}: {item['id']}")
                result[item["id"]] = item
            return result

        images = indexed(source["images"], "images")
        annotations = indexed(source["annotations"], "annotations")
        categories = indexed(source["categories"], "categories")
        category_records = []
        for index, category in enumerate(categories.values()):
            data = copy.deepcopy(category)
            color = data.get("color", PALETTE[index % len(PALETTE)])
            if not isinstance(color, str):
                color = PALETTE[index % len(PALETTE)]
            data["name"], data["color"] = cls._category_fields(data.get("name"), color)
            category_records.append(data)
        image_records = {}
        file_names = set()
        for image_id, image in images.items():
            name = _relative_file(image.get("file_name"))
            if name in file_names:
                raise ValueError(f"Two COCO records reference the same image: {name}")
            file_names.add(name)
            details = _image_details(_file_path(root, name))
            if type(image.get("width")) is not int or type(image.get("height")) is not int or (image["width"], image["height"]) != (details["width"], details["height"]):
                raise ValueError(f"The COCO dimensions do not match the image: {name}")
            image_records[image_id] = (name, details, _initial_state(image_id))
        source_annotations = []
        for coco_id, annotation in annotations.items():
            image_id, category_id = annotation.get("image_id"), annotation.get("category_id")
            if type(image_id) is not int or image_id not in images or type(category_id) is not int or category_id not in categories:
                raise ValueError(f"Annotation {coco_id} references an unknown image or class.")
            crowd = annotation.get("iscrowd", 0)
            if type(crowd) is not int or crowd not in (0, 1):
                raise ValueError(f"Invalid iscrowd value in annotation {coco_id}.")
            _, details, state = image_records[image_id]
            height, width = details["height"], details["width"]
            segmentation = annotation.get("segmentation")
            if segmentation is None or segmentation == []:
                try:
                    box = bbox(annotation.get("bbox"), width, height)
                except ValueError as error:
                    raise ValueError(f"Annotation {coco_id}: {error}") from error
                identity = str(uuid.uuid4())
                state["annotations"].append({"id": identity, "kind": "bbox", "category_id": category_id,
                                             "iscrowd": crowd, "bbox": box})
                state["schema_version"] = 2
                source_annotations.append((identity, coco_id, image_id, annotation, None))
                continue
            try:
                if isinstance(segmentation, dict):
                    mask = decode_mask(segmentation)
                elif isinstance(segmentation, list) and segmentation:
                    for polygon in segmentation:
                        if (not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2
                                or any(isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) for number in polygon)):
                            raise ValueError("Invalid COCO polygon.")
                    encoded = coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))
                    mask = np.asarray(coco_mask.decode(encoded), dtype=bool)
                else:
                    raise ValueError("A supported segmentation is required; boxes are not converted to masks.")
                if mask.shape != (height, width) or not mask.any():
                    raise ValueError("The mask is empty or has incorrect dimensions.")
                payload = mask_payload(mask)
                payload.pop("preview")
            except Exception as error:
                raise ValueError(f"Annotation {coco_id}: {error}") from error
            identity = str(uuid.uuid4())
            state["annotations"].append({"id": identity, "category_id": category_id, "iscrowd": crowd, **payload})
            source_annotations.append((identity, coco_id, image_id, annotation, payload["mask"]))

        destination.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".import.", suffix=".sqlite3", dir=destination)
        os.close(descriptor)
        try:
            with _connection(temporary) as connection:
                _schema(connection, destination, root)
                metadata = {key: value for key, value in source.items() if key not in ("images", "annotations", "categories")}
                connection.execute("UPDATE meta SET value=? WHERE key='coco_metadata'", (_json(metadata),))
                for category in category_records:
                    connection.execute("INSERT INTO categories VALUES(?,?)", (category["id"], _json(category)))
                if any(state.get("schema_version") == 2 for _, _, state in image_records.values()):
                    connection.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
                for image_id, (name, details, state) in image_records.items():
                    _insert_image(connection, image_id, name, details, images[image_id], state)
                for identity, coco_id, image_id, annotation, mask in source_annotations:
                    connection.execute("INSERT INTO coco_ids VALUES(?,?,?)", (identity, coco_id, image_id))
                    connection.execute("INSERT INTO annotation_sources VALUES(?,?,?)", (identity, _json(annotation), _json(mask)))
            _atomic_text(destination / "source.coco.json", source_text)
            os.replace(temporary, destination / DATABASE_NAME)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return cls(destination)

    def export_coco(self, task: str = "all", *, unique: bool = False) -> Path:
        if task not in ("all", "segmentation", "detection"):
            raise ValueError("Invalid export task.")
        with self._connect() as connection:
            connection.execute("BEGIN")
            output = json.loads(connection.execute("SELECT value FROM meta WHERE key='coco_metadata'").fetchone()[0])
            output.setdefault("info", {"description": "Segmentations", "version": "1.0"})
            output.setdefault("licenses", [])
            output["categories"] = [{key: value for key, value in json.loads(row[0]).items() if key != "color"}
                                    for row in connection.execute("SELECT data FROM categories ORDER BY id")]
            output["images"], output["annotations"] = [], []
            for row in connection.execute("SELECT * FROM images ORDER BY id").fetchall():
                image = json.loads(row["data"])
                image.update({"id": row["id"], "file_name": row["file_name"], "width": row["width"], "height": row["height"]})
                output["images"].append(image)
                for annotation in json.loads(row["state"])["annotations"]:
                    is_box = annotation.get("kind") == "bbox"
                    if task != "all" and is_box != (task == "detection"):
                        continue
                    identity = connection.execute("SELECT coco_id FROM coco_ids WHERE internal_id=?", (annotation["id"],)).fetchone()[0]
                    original = connection.execute("SELECT * FROM annotation_sources WHERE internal_id=?", (annotation["id"],)).fetchone()
                    record = json.loads(original["record"]) if original else {}
                    if is_box:
                        record.pop("segmentation", None)
                        record.update({"id": identity, "image_id": row["id"], "category_id": annotation["category_id"],
                                       "iscrowd": annotation.get("iscrowd", 0), "bbox": annotation["bbox"],
                                       "area": annotation["bbox"][2] * annotation["bbox"][3]})
                        output["annotations"].append(record)
                        continue
                    mask = annotation["mask"]
                    record.update({"id": identity, "image_id": row["id"], "category_id": annotation["category_id"],
                                   "iscrowd": annotation.get("iscrowd", 0), "segmentation": mask})
                    rle = {"size": mask["size"], "counts": mask["counts"].encode("ascii")}
                    record["area"] = int(coco_mask.area(rle))
                    record["bbox"] = [float(number) for number in coco_mask.toBbox(rle)]
                    output["annotations"].append(record)
            output["annotations"].sort(key=lambda annotation: annotation["id"])
        destination = self.directory / "exports" / f"{uuid.uuid4()}.json" if unique else self.directory / "anotaciones.coco.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        _atomic_text(destination, json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2))
        return destination
