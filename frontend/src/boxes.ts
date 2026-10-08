import type { BBox, XY } from "./types";
export const xyxy = ([x, y, w, h]: BBox): BBox => [x, y, x + w, y + h];
export const xywh = ([x, y, right, bottom]: BBox): BBox => [
  x,
  y,
  right - x,
  bottom - y,
];
export const boxContains = ([x, y, w, h]: BBox, [px, py]: XY) =>
  px >= x && py >= y && px <= x + w && py <= y + h;
export function unionBoxes(boxes: BBox[]): BBox | null {
  if (!boxes.length) return null;
  const x = Math.min(...boxes.map((b) => b[0])),
    y = Math.min(...boxes.map((b) => b[1]));
  return [
    x,
    y,
    Math.max(...boxes.map((b) => b[0] + b[2])) - x,
    Math.max(...boxes.map((b) => b[1] + b[3])) - y,
  ];
}
