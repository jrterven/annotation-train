import { describe, expect, it } from "vitest";
import { decodeCounts, maskContains, maskRaster } from "../masks";
import type { Mask } from "../types";
// Encoded by the backend's pycocotools implementation, not this decoder.
const fixtures: { mask: Mask; rows: number[][] }[] = [
  {
    mask: { size: [4, 5], counts: "d0" },
    rows: Array.from({ length: 4 }, () => [0, 0, 0, 0, 0]),
  },
  {
    mask: { size: [4, 5], counts: "0d0" },
    rows: Array.from({ length: 4 }, () => [1, 1, 1, 1, 1]),
  },
  {
    mask: { size: [3, 4], counts: "11123O" },
    rows: [
      [0, 1, 0, 0],
      [1, 1, 0, 1],
      [0, 1, 0, 1],
    ],
  },
  {
    mask: { size: [3, 3], counts: "0410" },
    rows: [
      [1, 1, 1],
      [1, 0, 1],
      [1, 1, 1],
    ],
  },
];
describe("COCO pixel masks", () => {
  for (const { mask, rows } of fixtures)
    it(`decodes ${mask.counts} with exact column-major positions`, () => {
      for (let y = 0; y < rows.length; y++)
        for (let x = 0; x < rows[y].length; x++)
          expect(maskContains(mask, x, y)).toBe(Boolean(rows[y][x]));
      expect(decodeCounts(mask.counts).reduce((a, b) => a + b, 0)).toBe(
        mask.size[0] * mask.size[1],
      );
      expect(maskContains(mask, -1, 0)).toBe(false);
      expect(maskContains(mask, mask.size[1], 0)).toBe(false);
    });
  it("rejects truncated and invalid encodings", () => {
    expect(() => decodeCounts("~~~~")).toThrow();
    expect(() => decodeCounts("!")).toThrow();
  });
  for (const { mask, rows } of fixtures)
    it(`renders cropped ${mask.counts} at exact original coordinates`, () => {
      const raster = maskRaster(mask, "#123456");
      for (let y = 0; y < rows.length; y++)
        for (let x = 0; x < rows[y].length; x++) {
          const cx = x - raster.x,
            cy = y - raster.y;
          const inside =
            cx >= 0 && cy >= 0 && cx < raster.width && cy < raster.height;
          const pixel = inside
            ? Array.from(
                raster.pixels.slice(
                  (cy * raster.width + cx) * 4,
                  (cy * raster.width + cx + 1) * 4,
                ),
              )
            : [0, 0, 0, 0];
          expect(pixel).toEqual(rows[y][x] ? [18, 52, 86, 255] : [0, 0, 0, 0]);
        }
    });
  it("allocates only the occupied rectangle even on a full-resolution orthophoto", () => {
    // Fixtures encoded independently by pycocotools in the backend.
    const small = maskRaster({ size: [10, 12], counts: "f1280^1" }, "#123456");
    expect([small.x, small.y, small.width, small.height]).toEqual([5, 4, 2, 2]);
    expect(small.pixels.length).toBe(16);
    const huge = maskRaster(
      { size: [5671, 5786], counts: "PTVj:1eWU_d0" },
      "#123456",
    );
    expect([huge.x, huge.y, huge.width, huge.height]).toEqual([
      2000, 2000, 1, 1,
    ]);
    expect(Array.from(huge.pixels)).toEqual([18, 52, 86, 255]);
    expect(() =>
      maskRaster({ size: [1, 1], counts: "20" }, "#123456"),
    ).toThrow();
  });
});

it("computes exact bounding boxes from RLE runs without allocating a raster", async () => {
  const { maskBounds } = await import("../masks");
  expect(maskBounds({ size: [5671, 5786], counts: "PTVj:1eWU_d0" })).toEqual([
    2000, 2000, 1, 1,
  ]);
  expect(maskBounds({ size: [3, 4], counts: "11123O" })).toEqual([0, 0, 4, 3]);
  expect(maskBounds({ size: [3, 3], counts: "0410" })).toEqual([0, 0, 3, 3]);
  expect(maskBounds({ size: [4, 5], counts: "d0" })).toBeNull();
  expect(() => maskBounds({ size: [1, 1], counts: "20" })).toThrow();
});
