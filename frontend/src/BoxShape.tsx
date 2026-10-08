import { useEffect, useState } from "react";
import { Group, Rect } from "react-konva";
import type { BBox } from "./types";

type Props = {
  bbox: BBox;
  color: string;
  scale: number;
  selected?: boolean;
  editable?: boolean;
  width: number;
  height: number;
  onSelect?: () => void;
  onChange?: (bbox: BBox) => void;
  dashed?: boolean;
};
export default function BoxShape(p: Props) {
  const [edit, setEdit] = useState<BBox | null>(null);
  useEffect(() => setEdit(null), [JSON.stringify(p.bbox)]);
  const b = edit || p.bbox;
  const interactive = !!p.onSelect;
  const handles = [
    [-1, -1],
    [0, -1],
    [1, -1],
    [-1, 0],
    [1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
  ];
  return (
    <Group name="bounding-box">
      <Rect
        name="box-outline"
        x={b[0]}
        y={b[1]}
        width={b[2]}
        height={b[3]}
        stroke={p.color}
        strokeWidth={(p.selected ? 2.5 : 1.5) / p.scale}
        fill={p.selected ? "rgba(255,255,255,.04)" : undefined}
        hitStrokeWidth={12 / p.scale}
        dash={p.dashed ? [6 / p.scale, 4 / p.scale] : undefined}
        listening={interactive}
        draggable={p.editable && p.selected}
        onMouseDown={(e) => {
          if (!interactive) return;
          e.cancelBubble = true;
          p.onSelect?.();
        }}
        onDragMove={(e) => {
          const next: BBox = [
            Math.max(0, Math.min(p.width - b[2], e.target.x())),
            Math.max(0, Math.min(p.height - b[3], e.target.y())),
            b[2],
            b[3],
          ];
          e.target.position({ x: next[0], y: next[1] });
          setEdit(next);
        }}
        onDragEnd={() => {
          if (edit) p.onChange?.(edit);
          setEdit(null);
        }}
      />
      {p.selected &&
        p.editable &&
        handles.map(([hx, hy]) => (
          <Rect
            key={`${hx}:${hy}`}
            name="box-handle"
            vertex
            x={b[0] + ((hx + 1) * b[2]) / 2}
            y={b[1] + ((hy + 1) * b[3]) / 2}
            width={8 / p.scale}
            height={8 / p.scale}
            offsetX={4 / p.scale}
            offsetY={4 / p.scale}
            fill="#fff"
            stroke={p.color}
            strokeWidth={1 / p.scale}
            hitStrokeWidth={8 / p.scale}
            draggable
            onMouseDown={(e) => {
              e.cancelBubble = true;
            }}
            onDragMove={(e) => {
              const [x, y, w, h] = p.bbox;
              const px = Math.max(0, Math.min(p.width, e.target.x())),
                py = Math.max(0, Math.min(p.height, e.target.y()));
              const left = hx === -1 ? Math.min(px, x + w - 0.01) : x,
                right = hx === 1 ? Math.max(px, x + 0.01) : x + w;
              const top = hy === -1 ? Math.min(py, y + h - 0.01) : y,
                bottom = hy === 1 ? Math.max(py, y + 0.01) : y + h;
              setEdit([left, top, right - left, bottom - top]);
            }}
            onDragEnd={() => {
              if (edit) p.onChange?.(edit);
              setEdit(null);
            }}
          />
        ))}
    </Group>
  );
}
