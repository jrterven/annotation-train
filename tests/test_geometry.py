import base64
import io

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
import pytest

from app.geometry import decode_mask, encode_mask, fill_small_holes, mask_payload, rasterize_components, union_masks, validate_components


def test_rle_roundtrip_matches_official_decoder():
    mask = np.zeros((9, 15), dtype=bool)
    mask[0, 14] = True
    mask[2:7, 3:8] = True
    mask[4, 5] = False
    encoded = encode_mask(mask)
    assert encoded["size"] == [9, 15]
    assert isinstance(encoded["counts"], str)
    assert np.array_equal(decode_mask(encoded), mask)
    assert np.array_equal(coco_mask.decode(encoded), mask)


def test_uncompressed_rle_is_column_major():
    mask = decode_mask({"size": [3, 5], "counts": [4, 2, 9]})
    assert set(map(tuple, np.argwhere(mask))) == {(1, 1), (2, 1)}


@pytest.mark.parametrize("rle", [
    {"size": [0, 2], "counts": "2"},
    {"size": [2, 2], "counts": ""},
    {"size": [2, 2], "counts": "P"},
    {"size": [2, 2], "counts": "💀"},
    {"size": [2, 2], "counts": [0, 5]},
    {"size": [2, 2], "counts": [1, -1, 4]},
    {"size": [2, 2], "counts": [True, 3]},
])
def test_invalid_rle_is_rejected_before_c_decoder(rle):
    with pytest.raises(ValueError):
        decode_mask(rle)


@pytest.mark.parametrize("kind", ["empty", "single_pixel", "donut", "diagonal", "nested", "edge", "full"])
def test_cell_contours_preserve_every_pixel(kind):
    mask = np.zeros((13, 19), dtype=bool)
    if kind == "single_pixel":
        mask[4, 7] = True
    elif kind == "donut":
        mask[1:12, 2:17] = True
        mask[3:10, 4:15] = False
    elif kind == "diagonal":
        mask[2, 2] = mask[3, 3] = mask[4, 4] = True
    elif kind == "nested":
        mask[1:12, 1:18] = True
        mask[2:11, 2:17] = False
        mask[4:8, 6:12] = True
        mask[5, 7] = False
    elif kind == "edge":
        mask[0, :] = True
        mask[:, 0] = True
        mask[-1, -1] = True
    elif kind == "full":
        mask[:] = True
    payload = mask_payload(mask)
    assert np.array_equal(rasterize_components(payload["components"], 19, 13), mask)
    for component in payload["components"]:
        assert len(component["outer"]) >= 3
        assert component["outer"][0] != component["outer"][-1]
    if kind == "donut":
        assert len(payload["components"]) == 1
        assert len(payload["components"][0]["holes"]) == 1


def test_fuzz_cell_contours_and_rle():
    random = np.random.default_rng(41)
    for _ in range(25):
        mask = random.random((15, 20)) > 0.62
        payload = mask_payload(mask)
        assert np.array_equal(decode_mask(payload["mask"]), mask)
        assert np.array_equal(rasterize_components(payload["components"], 20, 15), mask)


def test_preview_is_transparent_rgba_with_requested_color():
    mask = np.zeros((4, 5), dtype=bool)
    mask[2, 3] = True
    preview = mask_payload(mask, "#12ABEF")["preview"]
    image = Image.open(io.BytesIO(base64.b64decode(preview.split(",")[1])))
    assert image.mode == "RGBA"
    assert image.getpixel((0, 0)) == (0, 0, 0, 0)
    assert image.getpixel((3, 2)) == (18, 171, 239, 115)


def test_union_keeps_parts_in_one_mask_and_preserves_holes():
    one = np.zeros((8, 12), dtype=bool)
    one[1:7, 1:7] = True
    one[3:5, 3:5] = False
    two = np.zeros_like(one)
    two[2:5, 9:11] = True
    assert np.array_equal(union_masks([encode_mask(one), encode_mask(two)], 12, 8), one | two)
    with pytest.raises(ValueError):
        union_masks([encode_mask(two)], 8, 12)


@pytest.mark.parametrize("component", [
    {"outer": [[0, 0], [5, 5], [0, 5], [5, 0]], "holes": []},
    {"outer": [[0, 0], [5, 0], [5, 5]], "holes": [[[7, 7], [8, 7], [8, 8]]]},
    {"outer": [[0, 0], [5, 0], [float("nan"), 5]], "holes": []},
    {"outer": [[-1, 0], [5, 0], [5, 5]], "holes": []},
    {"outer": [[0, 0], [1, 1], [2, 2]], "holes": []},
])
def test_invalid_manual_geometry_is_rejected(component):
    with pytest.raises(ValueError):
        rasterize_components([component], 10, 10)


def test_fractional_manual_geometry_uses_pixel_centers():
    components = [{"outer": [[1.2, 1.2], [3.2, 1.2], [3.2, 4.2], [1.2, 4.2]], "holes": []}]
    mask = rasterize_components(components, 6, 6)
    expected = np.zeros((6, 6), dtype=bool)
    expected[1:4, 1:3] = True
    assert np.array_equal(mask, expected)


def test_round_masks_have_manageable_controls_without_changing_exact_mask():
    y, x = np.ogrid[:300, :340]
    distance = (x - 170) ** 2 + (y - 150) ** 2
    mask = (distance <= 120 ** 2) & (distance >= 60 ** 2)
    payload = mask_payload(mask)
    exact = payload["components"]
    controls = payload["controls"]
    count = lambda components: sum(len(c["outer"]) + sum(map(len, c["holes"])) for c in components)
    assert count(exact) > 700
    assert count(controls) < 150
    assert count(controls) < count(exact) / 4
    assert len(controls) == 1 and len(controls[0]["holes"]) == 1
    assert np.array_equal(decode_mask(payload["mask"]), mask)
    assert np.array_equal(rasterize_components(exact, 340, 300), mask)
    # These editing handles intentionally approximate the raster boundary.
    assert not np.array_equal(rasterize_components(controls, 340, 300), mask)
    validate_components(controls, 340, 300)


def test_simplified_controls_preserve_tiny_islands_holes_and_density():
    mask = np.zeros((80, 120), dtype=bool)
    mask[5:65, 5:90] = True
    mask[20, 20] = False
    mask[75, 115] = True
    payload = mask_payload(mask)
    controls = payload["controls"]
    assert len(controls) == 2
    assert len(controls[0]["holes"]) == 1
    assert len(controls[0]["holes"][0]) == 4
    assert len(controls[1]["outer"]) == 4
    controls_mask = rasterize_components(controls, 120, 80)
    assert controls_mask[75, 115]
    assert not controls_mask[20, 20]
    for component in controls:
        for ring in [component["outer"], *component["holes"]]:
            assert len(ring) >= 3
            for first, second in zip(ring, ring[1:] + ring[:1]):
                assert np.linalg.norm(np.asarray(first) - np.asarray(second)) <= 24.000001


def test_control_topology_valid_for_disconnected_and_nested_shapes():
    random = np.random.default_rng(19)
    for _ in range(15):
        mask = random.random((12, 16)) > .5
        payload = mask_payload(mask)
        validate_components(payload["controls"], 16, 12)
        assert len(payload["controls"]) == len(payload["components"])
        assert sorted(len(c["holes"]) for c in payload["controls"]) == sorted(len(c["holes"]) for c in payload["components"])
        assert np.array_equal(decode_mask(payload["mask"]), mask)


def test_fill_small_holes_changes_only_qualifying_background_pixels():
    mask = np.zeros((40, 60), dtype=bool)
    mask[2:36, 2:50] = True
    mask[5:9, 5:9] = False  # Exactly 16 pixels.
    mask[15, 10:27] = False  # 17 pixels: retain this hole.
    mask[20, 40] = False
    mask[2:5, 30] = False  # An opening connected to the exterior.
    mask[38, 58] = True  # Keep a one-pixel foreground island.
    original = mask.copy()
    expected = mask.copy()
    expected[5:9, 5:9] = True
    expected[20, 40] = True
    cleaned, holes, pixels = fill_small_holes(mask)
    assert holes == 2 and pixels == 17
    assert np.array_equal(cleaned, expected)
    assert np.array_equal(mask, original)
    payload = mask_payload(cleaned)
    assert np.array_equal(decode_mask(payload["mask"]), expected)
    assert np.array_equal(coco_mask.decode(payload["mask"]), expected)
    assert np.array_equal(rasterize_components(payload["components"], 60, 40), expected)
    validate_components(payload["controls"], 60, 40)


def test_hole_area_excludes_foreground_islands_inside_it():
    mask = np.zeros((17, 17), dtype=bool)
    mask[1:16, 1:16] = True
    mask[5:8, 5:8] = False
    mask[6, 6] = True  # The surrounding background region has 8, not 9 pixels.
    mask[10:14, 10:14] = False
    mask[11, 11] = True  # This surrounding region has 15 pixels, so retain it.
    expected = mask.copy()
    expected[5:8, 5:8] = True
    cleaned, holes, pixels = fill_small_holes(mask, max_area=8)
    assert holes == 1 and pixels == 8
    assert np.array_equal(cleaned, expected)


def test_hole_cleanup_uses_four_connected_background_and_preserves_image_edges():
    mask = np.ones((8, 8), dtype=bool)
    mask[0, 0] = mask[1, 1] = mask[2, 2] = False
    mask[0:2, 6] = False  # A small edge-connected void is never filled.
    expected = mask.copy()
    expected[1, 1] = expected[2, 2] = True
    cleaned, holes, pixels = fill_small_holes(mask, max_area=2)
    assert holes == 2 and pixels == 2
    assert np.array_equal(cleaned, expected)


@pytest.mark.parametrize("max_area", [0, -1, True, 1.0, "16", 150_000_001])
def test_hole_cleanup_rejects_invalid_area_limits(max_area):
    with pytest.raises(ValueError, match="integer"):
        fill_small_holes(np.ones((3, 3), dtype=bool), max_area)


@pytest.mark.parametrize("fill", [False, True])
def test_hole_cleanup_without_holes_is_an_exact_noop(fill):
    mask = np.full((5, 7), fill, dtype=bool)
    cleaned, holes, pixels = fill_small_holes(mask)
    assert np.array_equal(cleaned, mask)
    assert cleaned is not mask
    assert holes == pixels == 0


def test_hole_cleanup_matches_independent_pixel_flood_fill():
    random = np.random.default_rng(23)
    for _ in range(25):
        mask = random.random((9, 12)) > 0.45
        limit = int(random.integers(1, 8))
        expected = mask.copy()
        seen = set()
        expected_holes = expected_pixels = 0
        for y, x in np.argwhere(~mask):
            point = (int(y), int(x))
            if point in seen:
                continue
            pending = [point]
            region = set()
            while pending:
                row, column = pending.pop()
                if (row, column) in seen:
                    continue
                seen.add((row, column))
                region.add((row, column))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = row + dy, column + dx
                    if 0 <= ny < 9 and 0 <= nx < 12 and not mask[ny, nx] and (ny, nx) not in seen:
                        pending.append((ny, nx))
            if len(region) <= limit and all(0 < row < 8 and 0 < column < 11 for row, column in region):
                expected_holes += 1
                expected_pixels += len(region)
                for row, column in region:
                    expected[row, column] = True
        cleaned, holes, pixels = fill_small_holes(mask, limit)
        assert np.array_equal(cleaned, expected)
        assert (holes, pixels) == (expected_holes, expected_pixels)
