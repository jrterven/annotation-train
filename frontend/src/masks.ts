import type { Mask } from "./types";
// COCO compressed RLE, column-major. Keep pixel masks exact; contours are editing controls.
export function decodeCounts(encoded: string): number[] {
  const counts: number[] = [];
  let p = 0;
  while (p < encoded.length) {
    let value = 0,
      shift = 0,
      more = true,
      c = 0;
    while (more) {
      if (p >= encoded.length) throw new Error("Incomplete RLE mask");
      c = encoded.charCodeAt(p++) - 48;
      if (c < 0 || c > 63) throw new Error("Invalid RLE mask");
      value += (c & 31) * 2 ** shift;
      shift += 5;
      more = Boolean(c & 32);
      if (shift > 55) throw new Error("RLE mask is too large");
    }
    if (c & 16) value -= 2 ** shift;
    if (counts.length > 2) value += counts[counts.length - 2];
    if (value < 0 || !Number.isSafeInteger(value))
      throw new Error("Invalid RLE mask");
    counts.push(value);
  }
  return counts;
}
type Overlay = { url: string; x: number; y: number };
const cache = new Map<string, Overlay>();
export function maskContains(mask: Mask, x: number, y: number): boolean {
  const [h, w] = mask.size;
  x = Math.floor(x);
  y = Math.floor(y);
  if (x < 0 || y < 0 || x >= w || y >= h) return false;
  const target = x * h + y;
  let offset = 0,
    filled = false;
  for (const count of decodeCounts(mask.counts)) {
    offset += count;
    if (target < offset) return filled;
    filled = !filled;
  }
  return false;
}
// Keep every original pixel, but allocate only the occupied rectangle. A small
// object on an orthophoto must not require a full-image RGBA canvas per mask.
export function maskRaster(mask: Mask, color: string) {
  const [h, w] = mask.size;
  if (![h, w, h * w].every(Number.isSafeInteger) || h <= 0 || w <= 0)
    throw new Error("Invalid RLE dimensions");
  const counts = decodeCounts(mask.counts);
  let offset = 0,
    left = w,
    top = h,
    right = -1,
    bottom = -1;
  for (let run = 0; run < counts.length; run++) {
    const count = counts[run];
    if (offset + count > w * h) throw new Error("Invalid RLE dimensions");
    if (run % 2 && count) {
      const firstX = Math.floor(offset / h),
        lastX = Math.floor((offset + count - 1) / h);
      left = Math.min(left, firstX);
      right = Math.max(right, lastX);
      top = Math.min(top, firstX === lastX ? offset % h : 0);
      bottom = Math.max(
        bottom,
        firstX === lastX ? (offset + count - 1) % h : h - 1,
      );
    }
    offset += count;
  }
  if (offset !== w * h) throw new Error("Incomplete RLE dimensions");
  if (right < left)
    return {
      x: 0,
      y: 0,
      width: 1,
      height: 1,
      pixels: new Uint8ClampedArray(4),
    };
  const width = right - left + 1,
    height = bottom - top + 1;
  const pixels = new Uint8ClampedArray(width * height * 4);
  const hex = color.replace("#", "");
  const rgb = [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
  offset = 0;
  for (let run = 0; run < counts.length; run++) {
    const count = counts[run];
    if (run % 2) {
      for (let i = offset; i < offset + count; i++) {
        const x = Math.floor(i / h) - left,
          y = (i % h) - top;
        const at = (y * width + x) * 4;
        pixels[at] = rgb[0];
        pixels[at + 1] = rgb[1];
        pixels[at + 2] = rgb[2];
        pixels[at + 3] = 255;
      }
    }
    offset += count;
  }
  return { x: left, y: top, width, height, pixels };
}
export function maskOverlay(mask: Mask, color: string): Overlay {
  const key = `${color}:${mask.size.join(",")}:${mask.counts}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const raster = maskRaster(mask, color);
  const canvas = document.createElement("canvas");
  canvas.width = raster.width;
  canvas.height = raster.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return { url: "", x: raster.x, y: raster.y };
  ctx.putImageData(
    new ImageData(raster.pixels, raster.width, raster.height),
    0,
    0,
  );
  const overlay = { url: canvas.toDataURL(), x: raster.x, y: raster.y };
  cache.set(key, overlay);
  if (cache.size > 60) cache.delete(cache.keys().next().value!);
  return overlay;
}
