import { describe, expect, it } from "vitest";
import { decodeCounts, maskContains } from "../masks";
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
});
