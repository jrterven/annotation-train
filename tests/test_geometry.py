import base64
import io

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
import pytest

from app.geometry import decode_mask, encode_mask, mask_payload, rasterize_components, union_masks, validate_components


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
