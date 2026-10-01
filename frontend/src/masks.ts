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
      if (p >= encoded.length) throw new Error("Máscara RLE incompleta");
      c = encoded.charCodeAt(p++) - 48;
      if (c < 0 || c > 63) throw new Error("Máscara RLE inválida");
      value += (c & 31) * 2 ** shift;
      shift += 5;
      more = Boolean(c & 32);
      if (shift > 55) throw new Error("Máscara RLE demasiado grande");
    }
    if (c & 16) value -= 2 ** shift;
    if (counts.length > 2) value += counts[counts.length - 2];
    if (value < 0 || !Number.isSafeInteger(value))
      throw new Error("Máscara RLE inválida");
    counts.push(value);
  }
  return counts;
}
const cache = new Map<string, string>();
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
export function maskURL(mask: Mask, color: string): string {
  const key = `${color}:${mask.size.join(",")}:${mask.counts}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const [h, w] = mask.size;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  const data = ctx.createImageData(w, h);
  const hex = color.replace("#", "");
  const rgb = [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
  let offset = 0;
  let filled = false;
  for (const count of decodeCounts(mask.counts)) {
    if (offset + count > w * h) throw new Error("Dimensiones RLE inválidas");
    if (filled) {
      for (let i = offset; i < offset + count; i++) {
        const x = Math.floor(i / h),
          y = i % h;
        const at = (y * w + x) * 4;
        data.data[at] = rgb[0];
        data.data[at + 1] = rgb[1];
        data.data[at + 2] = rgb[2];
        data.data[at + 3] = 255;
      }
    }
    offset += count;
    filled = !filled;
  }
  if (offset !== w * h) throw new Error("Dimensiones RLE incompletas");
  ctx.putImageData(data, 0, 0);
  const url = canvas.toDataURL();
  cache.set(key, url);
  if (cache.size > 60) cache.delete(cache.keys().next().value!);
  return url;
}
