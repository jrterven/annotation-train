"""Lossless COCO masks and editable polygons in original-image coordinates."""

from __future__ import annotations

import base64
import io
import math
import re
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from shapely import box, intersects_xy, union_all
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import explain_validity


def encode_mask(mask: np.ndarray) -> dict:
    array = np.asarray(mask)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError("The mask must be a nonempty two-dimensional array.")
    rle = coco_mask.encode(np.asfortranarray(array.astype(np.uint8) != 0, dtype=np.uint8))
    return {"size": [int(n) for n in rle["size"]], "counts": rle["counts"].decode("ascii")}


def decode_mask(rle: dict) -> np.ndarray:
    if not isinstance(rle, dict):
        raise ValueError("The mask must use COCO RLE.")
    size, counts = rle.get("size"), rle.get("counts")
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(type(n) is not int or n <= 0 for n in size)):
        raise ValueError("RLE size must contain a positive height and width.")
    height, width = size
    if height * width > 150_000_000:
        raise ValueError("The mask exceeds the 150-megapixel limit.")
    # Validate runs ourselves before entering the C decoder. Invalid compressed
    # counts can otherwise read/write outside the expected allocation.
    if isinstance(counts, str):
        runs, value, shift, previous = [], 0, 0, []
        at = 0
        while at < len(counts):
            value, shift = 0, 0
            while True:
                if at >= len(counts):
                    raise ValueError("Incomplete compressed RLE.")
                code = ord(counts[at]) - 48
                at += 1
                if code < 0 or code > 63 or shift > 60:
                    raise ValueError("Invalid compressed RLE.")
                value |= (code & 0x1F) << shift
                shift += 5
                if not code & 0x20:
                    if code & 0x10:
                        value |= -1 << shift
                    break
            if len(previous) > 2:
                value += previous[-2]
            if value < 0:
                raise ValueError("RLE contains negative run lengths.")
            previous.append(value)
            runs.append(value)
    elif isinstance(counts, list) and all(type(n) is int and n >= 0 for n in counts):
        runs = counts
    else:
        raise ValueError("RLE counts must be text or a list of integers.")
    if not runs or sum(runs) != height * width:
        raise ValueError("RLE run lengths do not match the dimensions.")
    try:
        encoded = {"size": [height, width], "counts": counts}
        if isinstance(counts, list):
            encoded = coco_mask.frPyObjects(encoded, height, width)
        return np.asarray(coco_mask.decode(encoded), dtype=bool)
    except Exception as error:
        raise ValueError("Could not decode the RLE mask.") from error


def _ring(points: Any, width: int, height: int) -> list[tuple[float, float]]:
    if not isinstance(points, (list, tuple)) or len(points) < 3:
        raise ValueError("Each contour needs at least three vertices.")
    output = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each vertex must contain x and y.")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point):
            raise ValueError("Coordinates must be finite numbers.")
        x, y = map(float, point)
        if not 0 <= x <= width or not 0 <= y <= height:
            raise ValueError("A vertex is outside the image.")
        output.append((x, y))
    if len(set(output)) < 3:
        raise ValueError("Each contour needs three distinct vertices.")
    return output


def _validated_polygons(components: list[dict], width: int, height: int) -> list[Polygon]:
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError("Dimensions must be positive integers.")
    if width * height > 150_000_000 or not isinstance(components, list):
        raise ValueError("Invalid geometry or dimensions.")
    polygons = []
    for component in components:
        if not isinstance(component, dict) or not isinstance(component.get("holes", []), list):
            raise ValueError("Invalid geometry component.")
        polygon = Polygon(_ring(component.get("outer"), width, height),
                          [_ring(hole, width, height) for hole in component.get("holes", [])])
        if polygon.is_empty or polygon.area <= 0 or not polygon.is_valid:
            raise ValueError(f"Invalid polygon: {explain_validity(polygon)}")
        polygons.append(polygon)
    return polygons


def validate_components(components: list[dict], width: int, height: int) -> None:
    """Validate editable topology without requiring equality to a pixel mask."""
    _validated_polygons(components, width, height)


def rasterize_components(components: list[dict], width: int, height: int) -> np.ndarray:
    polygons = _validated_polygons(components, width, height)
    mask = np.zeros((height, width), dtype=bool)
    for polygon in polygons:
        x0, y0, x1, y1 = polygon.bounds
        left, top = max(0, math.floor(x0)), max(0, math.floor(y0))
        right, bottom = min(width, math.ceil(x1)), min(height, math.ceil(y1))
        # Chunk rows to avoid allocating two full-image float64 coordinate grids.
        xs = np.arange(left, right, dtype=float) + 0.5
        for start in range(top, bottom, 256):
            stop = min(start + 256, bottom)
            ys = (np.arange(start, stop, dtype=float) + 0.5)[:, None]
            mask[start:stop, left:right] |= intersects_xy(polygon, xs[None, :], ys)
    return mask


def _coordinates(ring: Any) -> list[list[float]]:
    """Remove the closing duplicate; add handles along long straight edges."""
    points = list(ring.coords)[:-1]
    output = []
    for index, (x, y) in enumerate(points):
        nx, ny = points[(index + 1) % len(points)]
        steps = max(1, math.ceil(math.hypot(nx - x, ny - y) / 24))
        output.extend([[float(x + (nx - x) * step / steps), float(y + (ny - y) * step / steps)]
                       for step in range(steps)])
    return output


def _mask_shape(mask: np.ndarray):
    # Union horizontal pixel runs, not centerline contours. Cell boundaries
    # preserve one-pixel objects, holes, disconnected islands, and edge pixels.
    padded = np.pad(mask.astype(np.int8), ((0, 0), (1, 1)))
    transitions = np.diff(padded, axis=1)
    starts_y, starts_x = np.nonzero(transitions == 1)
    ends_y, ends_x = np.nonzero(transitions == -1)
    if not len(starts_x):
        return Polygon()
    rectangles = box(starts_x, starts_y, ends_x, ends_y + 1)
    return union_all(rectangles).simplify(0, preserve_topology=True)


def _shape_polygons(shape) -> list[Polygon]:
    if shape.is_empty:
        return []
    polygons = [shape] if shape.geom_type == "Polygon" else list(shape.geoms)
    polygons.sort(key=lambda item: (item.bounds[1], item.bounds[0], -item.area))
    return polygons


def _shape_components(shape) -> list[dict]:
    return [{"outer": _coordinates(poly.exterior), "holes": [_coordinates(ring) for ring in poly.interiors]}
            for poly in _shape_polygons(shape)]


def _control_components(shape) -> list[dict]:
    """Simplified editing handles, kept separate from the lossless boundary."""
    if shape.is_empty:
        return []
    # Global simplification preserves the relationship between adjacent parts.
    simplified = shape.simplify(1.0, preserve_topology=True)
    originals = [shape] if shape.geom_type == "Polygon" else list(shape.geoms)
    candidates = [simplified] if simplified.geom_type == "Polygon" else list(simplified.geoms)
    if len(originals) != len(candidates):
        return _shape_components(shape)
    safe = []
    for original, candidate in zip(originals, candidates):
        # GEOS can legally reduce a one-pixel square/hole to a triangle. Keep
        # tiny rings exact so editing another part cannot erase their pixel.
        if original.area <= 4 or len(original.interiors) != len(candidate.interiors):
            safe.append(original)
            continue
        holes = [list(source.coords) if Polygon(source).area <= 4 else list(reduced.coords)
                 for source, reduced in zip(original.interiors, candidate.interiors)]
        protected = Polygon(candidate.exterior, holes)
        safe.append(protected if protected.is_valid else original)
    result = safe[0] if len(safe) == 1 else MultiPolygon(safe)
    # Restoring a tiny ring must not introduce intersections with a neighboring
    # component. Falling back retains every component and hole exactly.
    return _shape_components(result if result.is_valid else shape)


def controls_for_components(components: list[dict], width: int, height: int) -> list[dict]:
    """Upgrade an older saved record without decoding/retracing its mask."""
    polygons = _validated_polygons(components, width, height)
    if not polygons:
        return []
    shape = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
    if not shape.is_valid:
        # Manually edited components may overlap; their pixel semantics is union.
        shape = union_all(polygons)
    return _control_components(shape)


def mask_preview(mask: np.ndarray, color: str = "#8B8DE3") -> str:
    """Render only the derived overlay, without retracing stored contours."""
    array = np.asarray(mask, dtype=bool)
    if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise ValueError("The color must use #RRGGBB format.")
    rgba = np.zeros((*array.shape, 4), dtype=np.uint8)
    rgba[array, :3] = [int(color[offset:offset + 2], 16) for offset in (1, 3, 5)]
    rgba[array, 3] = 115
    buffer = io.BytesIO()
    Image.fromarray(rgba).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def mask_payload(mask: np.ndarray, color: str = "#8B8DE3") -> dict:
    array = np.asarray(mask, dtype=bool)
    encoded = encode_mask(array)
    shape = _mask_shape(array)
    return {"mask": encoded, "components": _shape_components(shape),
            "controls": _control_components(shape), "preview": mask_preview(array, color)}


def fill_small_holes(mask: np.ndarray, max_area: int = 16) -> tuple[np.ndarray, int, int]:
    """Fill enclosed background regions up to max_area original-image pixels.

    Pixel cells sharing an edge belong to the same background region; diagonal
    contact alone does not connect them (4-connectivity). Regions touching any
    image edge remain background. Foreground islands inside a hole are excluded
    from its area and are never removed. The input array is not modified.
    """
    if type(max_area) is not int or not 1 <= max_area <= 150_000_000:
        raise ValueError("The hole area limit must be an integer from 1 to 150000000 pixels.")
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2 or min(array.shape) < 1 or array.size > 150_000_000:
        raise ValueError("The mask must be a nonempty 2D array of at most 150 megapixels.")
    height, width = array.shape
    output = array.copy()
    # Subtracting exact pixel-cell geometry measures only background pixels,
    # even when a hole encloses a disconnected foreground island.
    background = box(0, 0, width, height).difference(_mask_shape(array))
    filled_holes = filled_pixels = 0
    for region in _shape_polygons(background):
        left, top, right, bottom = region.bounds
        if left <= 0 or top <= 0 or right >= width or bottom >= height:
            continue
        if region.area > max_area:
            continue
        left, top, right, bottom = map(int, (left, top, right, bottom))
        xs = np.arange(left, right, dtype=float) + 0.5
        added = 0
        for start in range(top, bottom, 256):
            stop = min(start + 256, bottom)
            ys = (np.arange(start, stop, dtype=float) + 0.5)[:, None]
            selected = intersects_xy(region, xs[None, :], ys)
            target = output[start:stop, left:right]
            pixels = selected & ~target
            added += int(pixels.sum())
            target |= pixels
        if added:
            filled_holes += 1
            filled_pixels += added
    return output, filled_holes, filled_pixels


def union_masks(masks: list[dict], width: int, height: int) -> np.ndarray:
    if not isinstance(masks, list) or not masks:
        raise ValueError("At least one mask is required.")
    output = np.zeros((height, width), dtype=bool)
    for rle in masks:
        mask = decode_mask(rle)
        if mask.shape != (height, width):
            raise ValueError("The masks do not match the image dimensions.")
        output |= mask
    return output
