import { useEffect, useMemo, useRef, useState } from "react";
import {
  Stage,
  Layer,
  Group,
  Image as KonvaImage,
  Line,
  Circle,
  Rect,
} from "react-konva";
import type Konva from "konva";
import { Focus, Minus, Plus, ScanLine } from "lucide-react";
import type {
  Annotation,
  Category,
  Component,
  Draft,
  Mask,
  Proposal,
  SourceImage,
  Tool,
  XY,
} from "./types";
import { maskContains, maskURL } from "./masks";
import { assetURL } from "./api";
export type Vertex = { component: number; ring: number; index: number };
type Props = {
  image: SourceImage;
  annotations: Annotation[];
  proposals: Proposal[];
  draft: Draft | null;
  categories: Category[];
  selected: string | null;
  tool: Tool;
  showMasks: boolean;
  opacity: number;
  vertex: Vertex | null;
  busy: boolean;
  onVertex: (v: Vertex | null) => void;
  onSelect: (id: string | null) => void;
  onProposal: (id: string) => void;
  onPoint: (point: XY, negative: boolean) => void;
  onBox: (box: [number, number, number, number]) => void;
  onGeometry: (id: string, components: Component[]) => void;
  fitRef: React.RefObject<(() => void) | null>;
};
function useImage(url: string) {
  const [image, setImage] = useState<HTMLImageElement>();
  useEffect(() => {
    let valid = true;
    setImage(undefined);
    const img = new window.Image();
    img.onload = () => {
      if (valid) setImage(img);
    };
    img.src = url;
    return () => {
      valid = false;
    };
  }, [url]);
  return image;
}
function MaskImage({
  mask,
  color,
  opacity,
}: {
  mask: Mask;
  color: string;
  opacity: number;
}) {
  const url = useMemo(() => maskURL(mask, color), [mask, color]);
  const image = useImage(url);
  return <KonvaImage image={image} opacity={opacity} listening={false} />;
}
const flat = (points: XY[]) => points.flat();
const rings = (c: Component) => [c.outer, ...c.holes];
export default function CanvasEditor(p: Props) {
  const container = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const [size, setSize] = useState({ width: 800, height: 600 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [space, setSpace] = useState(false);
  const [dragPan, setDragPan] = useState<{
    x: number;
    y: number;
    ox: number;
    oy: number;
  } | null>(null);
  const [box, setBox] = useState<{ start: XY; end: XY } | null>(null);
  const [edit, setEdit] = useState<Component[] | null>(null);
  const source = useImage(assetURL(`/images/${p.image.id}/file`));
  const fit = Math.min(
    (size.width - 100) / p.image.width,
    (size.height - 100) / p.image.height,
    1,
  );
  const scale = Math.max(0.001, fit * zoom);
  const chosen = p.annotations.find((a) => a.id === p.selected);
  const geometryKey = JSON.stringify(chosen?.controls ?? chosen?.components);
  // Draft inference and autosave can replace the parent state while dragging.
  // Equal geometry must keep its reference so those updates don't reset edits.
  const baseGeometry = useMemo(
    () => chosen?.controls ?? chosen?.components,
    [p.selected, geometryKey],
  );
  const geometry = edit || baseGeometry;
  const fitView = () => {
    setZoom(1);
    setPan({
      x: (size.width - p.image.width * fit) / 2,
      y: (size.height - p.image.height * fit) / 2,
    });
  };
  p.fitRef.current = fitView;
  useEffect(() => {
    if (!container.current) return;
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setSize({ width, height });
    });
    observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    fitView();
    setEdit(null);
    setBox(null);
  }, [p.image.id, size.width, size.height]);
  useEffect(() => {
    setEdit(null);
  }, [p.selected, baseGeometry]);
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (
        e.code === "Space" &&
        !/INPUT|TEXTAREA|SELECT/.test((e.target as HTMLElement).tagName)
      ) {
        e.preventDefault();
        setSpace(true);
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space") setSpace(false);
    };
    const blur = () => {
      setSpace(false);
      setDragPan(null);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
    };
  }, []);
  function coord(): XY | null {
    const at = stageRef.current?.getPointerPosition();
    return at ? [(at.x - pan.x) / scale, (at.y - pan.y) / scale] : null;
  }
  const bounded = (at: XY): XY => [
    Math.min(p.image.width, Math.max(0, at[0])),
    Math.min(p.image.height, Math.max(0, at[1])),
  ];
  function zoomTo(
    next: number,
    anchor = { x: size.width / 2, y: size.height / 2 },
  ) {
    next = Math.max(0.1, Math.min(16, next));
    const ratio = next / zoom;
    setPan({
      x: anchor.x - (anchor.x - pan.x) * ratio,
      y: anchor.y - (anchor.y - pan.y) * ratio,
    });
    setZoom(next);
  }
  function down(e: Konva.KonvaEventObject<MouseEvent>) {
    if (e.target.attrs.vertex) return;
    const raw = stageRef.current?.getPointerPosition();
    if ((space || e.evt.button === 1) && raw) {
      e.evt.preventDefault();
      setDragPan({ ...raw, ox: pan.x, oy: pan.y });
      return;
    }
    const at = coord();
    if (
      !at ||
      at[0] < 0 ||
      at[1] < 0 ||
      at[0] >= p.image.width ||
      at[1] >= p.image.height
    )
      return;
    if (p.tool === "box") {
      setBox({ start: at, end: at });
      return;
    }
    if (p.tool === "positive" || p.tool === "negative") {
      p.onPoint(at, p.tool === "negative" || e.evt.button === 2);
      return;
    }
    const proposal = [...p.proposals]
      .reverse()
      .find((a) => maskContains(a.mask, ...at));
    if (proposal) {
      p.onProposal(proposal.id);
      return;
    }
    const ann = [...p.annotations]
      .reverse()
      .find((a) => maskContains(a.mask, ...at));
    p.onSelect(ann?.id || null);
    p.onVertex(null);
  }
  function move() {
    const raw = stageRef.current?.getPointerPosition();
    if (dragPan && raw)
      setPan({
        x: dragPan.ox + raw.x - dragPan.x,
        y: dragPan.oy + raw.y - dragPan.y,
      });
    if (box) {
      const at = coord();
      if (at) setBox({ ...box, end: bounded(at) });
    }
  }
  function up() {
    if (box) {
      const [x1, y1] = box.start,
        [x2, y2] = box.end;
      if (Math.abs(x1 - x2) * scale > 3 && Math.abs(y1 - y2) * scale > 3)
        p.onBox([
          Math.min(x1, x2),
          Math.min(y1, y2),
          Math.max(x1, x2),
          Math.max(y1, y2),
        ]);
      setBox(null);
    }
    setDragPan(null);
  }
  function changeVertex(v: Vertex, point: XY, commit: boolean) {
    if (!chosen) return;
    const next = structuredClone(edit || chosen.controls || chosen.components);
    const ring =
      v.ring === 0
        ? next[v.component].outer
        : next[v.component].holes[v.ring - 1];
    ring[v.index] = bounded(point);
    setEdit(next);
    if (commit) p.onGeometry(chosen.id, next);
  }
  function insert(c: number, r: number, e: Konva.KonvaEventObject<MouseEvent>) {
    e.cancelBubble = true;
    if (!chosen || p.tool !== "select" || p.busy) return;
    const at = coord();
    if (!at) return;
    const next = structuredClone(edit || chosen.controls || chosen.components);
    const ring = r === 0 ? next[c].outer : next[c].holes[r - 1];
    let nearest = 0,
      distance = Infinity;
    for (let i = 0; i < ring.length; i++) {
      const a = ring[i],
        b = ring[(i + 1) % ring.length];
      const dx = b[0] - a[0],
        dy = b[1] - a[1];
      const t = Math.max(
        0,
        Math.min(
          1,
          ((at[0] - a[0]) * dx + (at[1] - a[1]) * dy) /
            (dx * dx + dy * dy || 1),
        ),
      );
      const d = (at[0] - a[0] - t * dx) ** 2 + (at[1] - a[1] - t * dy) ** 2;
      if (d < distance) {
        nearest = i;
        distance = d;
      }
    }
    ring.splice(nearest + 1, 0, bounded(at));
    setEdit(next);
    p.onVertex({ component: c, ring: r, index: nearest + 1 });
    p.onGeometry(chosen.id, next);
  }
  return (
    <div
      className={`canvas-wrap tool-${p.tool} ${space ? "panning" : ""}`}
      ref={container}
    >
      <Stage
        ref={stageRef}
        width={size.width}
        height={size.height}
        onMouseDown={down}
        onMouseMove={move}
        onMouseUp={up}
        onMouseLeave={up}
        onContextMenu={(e) => e.evt.preventDefault()}
        onWheel={(e) => {
          e.evt.preventDefault();
          const pointer = stageRef.current?.getPointerPosition();
          zoomTo(zoom * (e.evt.deltaY > 0 ? 0.9 : 1.1), pointer || undefined);
        }}
      >
        <Layer>
          <Group x={pan.x} y={pan.y} scaleX={scale} scaleY={scale}>
            <Rect
              width={p.image.width}
              height={p.image.height}
              fill="#e8e7e2"
              shadowColor="#161a1c"
              shadowBlur={30 / scale}
              shadowOpacity={0.18}
              listening={false}
            />
            <KonvaImage
              image={source}
              width={p.image.width}
              height={p.image.height}
              listening={false}
            />
            {p.showMasks &&
              p.annotations.map((a) => (
                <MaskImage
                  key={a.id}
                  mask={a.mask}
                  color={
                    p.categories.find((c) => c.id === a.category_id)?.color ||
                    "#8991bd"
                  }
                  opacity={
                    a.id === p.selected
                      ? Math.min(0.8, p.opacity + 0.12)
                      : p.opacity
                  }
                />
              ))}
            {p.showMasks &&
              p.proposals.map((a) => (
                <MaskImage
                  key={a.id}
                  mask={a.mask}
                  color={a.selected ? "#b3d0a3" : "#aab6dc"}
                  opacity={
                    a.selected
                      ? Math.min(0.8, p.opacity + 0.15)
                      : p.opacity * 0.65
                  }
                />
              ))}
            {p.proposals
              .filter((a) => a.selected)
              .flatMap((a) =>
                a.components.flatMap((c, ci) =>
                  rings(c).map((ring, ri) => (
                    <Line
                      key={`${a.id}-${ci}-${ri}`}
                      points={flat(ring)}
                      closed
                      stroke="#d9f0cd"
                      strokeWidth={1.5 / scale}
                      listening={false}
                    />
                  )),
                ),
              )}
            {p.draft?.parts.map((part) => (
              <Group key={part.id}>
                {part.mask && p.showMasks && (
                  <MaskImage
                    mask={part.mask}
                    color={
                      p.categories.find((c) => c.id === p.draft?.category_id)
                        ?.color || "#b49ee2"
                    }
                    opacity={p.opacity + 0.05}
                  />
                )}
                {part.box && (
                  <Rect
                    x={part.box[0]}
                    y={part.box[1]}
                    width={part.box[2] - part.box[0]}
                    height={part.box[3] - part.box[1]}
                    stroke="#faf8ed"
                    strokeWidth={1.4 / scale}
                    dash={[5 / scale, 4 / scale]}
                    listening={false}
                  />
                )}
                {part.points.map((point, i) => (
                  <Group key={i} x={point.x} y={point.y}>
                    <Circle
                      radius={5 / scale}
                      fill={point.label ? "#e8f3df" : "#f4d9d0"}
                      stroke={point.label ? "#416641" : "#9a4936"}
                      strokeWidth={1.5 / scale}
                      listening={false}
                    />
                    <Line
                      points={[-2 / scale, 0, 2 / scale, 0]}
                      stroke={point.label ? "#416641" : "#9a4936"}
                      strokeWidth={1 / scale}
                      listening={false}
                    />
                    {Boolean(point.label) && (
                      <Line
                        points={[0, -2 / scale, 0, 2 / scale]}
                        stroke="#416641"
                        strokeWidth={1 / scale}
                        listening={false}
                      />
                    )}
                  </Group>
                ))}
              </Group>
            ))}
            {chosen &&
              geometry?.flatMap((component, ci) =>
                rings(component).map((ring, ri) => (
                  <Group key={`${ci}-${ri}`}>
                    <Line
                      listening={!space}
                      points={flat(ring)}
                      closed
                      stroke="#faf5da"
                      strokeWidth={1.5 / scale}
                      dash={ri ? [5 / scale, 3 / scale] : undefined}
                      onMouseDown={(e) => {
                        e.cancelBubble = true;
                      }}
                      onDblClick={(e) => insert(ci, ri, e)}
                      hitStrokeWidth={12 / scale}
                    />
                    {p.tool === "select" &&
                      ring.map((point, vi) => {
                        const v = { component: ci, ring: ri, index: vi };
                        const selected =
                          p.vertex?.component === ci &&
                          p.vertex.ring === ri &&
                          p.vertex.index === vi;
                        return (
                          <Circle
                            key={vi}
                            vertex
                            listening={!space}
                            x={point[0]}
                            y={point[1]}
                            radius={(selected ? 5 : 3.5) / scale}
                            fill={selected ? "#eed5a0" : "#fffef4"}
                            stroke="#544d39"
                            strokeWidth={1 / scale}
                            draggable={!p.busy && !space}
                            onMouseDown={(e) => {
                              e.cancelBubble = true;
                              p.onVertex(v);
                            }}
                            onDragStart={(e) => {
                              e.cancelBubble = true;
                              p.onVertex(v);
                            }}
                            onDragMove={(e) =>
                              changeVertex(
                                v,
                                [e.target.x(), e.target.y()],
                                false,
                              )
                            }
                            onDragEnd={(e) =>
                              changeVertex(
                                v,
                                [e.target.x(), e.target.y()],
                                true,
                              )
                            }
                          />
                        );
                      })}
                  </Group>
                )),
              )}
            {box && (
              <Rect
                x={Math.min(box.start[0], box.end[0])}
                y={Math.min(box.start[1], box.end[1])}
                width={Math.abs(box.end[0] - box.start[0])}
                height={Math.abs(box.end[1] - box.start[1])}
                fill="rgba(241,239,219,.1)"
                stroke="#faf8ed"
                strokeWidth={1.5 / scale}
                dash={[5 / scale, 4 / scale]}
                listening={false}
              />
            )}
          </Group>
        </Layer>
      </Stage>
      <div className="canvas-topline">
        <span className="canvas-file">
          <FileName name={p.image.file_name} />
          <span>
            {p.image.width} × {p.image.height}
          </span>
        </span>
        <span className="canvas-tech">
          <ScanLine size={13} /> Píxeles originales
        </span>
      </div>
      <div className="zoom-controls">
        <button
          onClick={() => zoomTo(zoom / 0.8)}
          title="Acercar"
          aria-label="Acercar"
        >
          <Plus size={16} />
        </button>
        <span>{Math.round(scale * 100)}%</span>
        <button
          onClick={() => zoomTo(zoom * 0.8)}
          title="Alejar"
          aria-label="Alejar"
        >
          <Minus size={16} />
        </button>
        <i />
        <button
          onClick={fitView}
          title="Ajustar imagen · F"
          aria-label="Ajustar imagen"
        >
          <Focus size={17} />
        </button>
      </div>
      <div className="canvas-instruction">
        {space
          ? "Arrastra para desplazar"
          : p.tool === "box"
            ? "Arrastra para delimitar el objeto"
            : p.tool === "negative"
              ? "Clic para excluir una región"
              : p.tool === "positive"
                ? "Clic positivo · clic derecho negativo"
                : chosen
                  ? "Arrastra vértices · doble clic en borde para insertar"
                  : "Selecciona una instancia"}
        <span>Espacio + arrastrar para mover</span>
      </div>
    </div>
  );
}
function FileName({ name }: { name: string }) {
  const parts = name.split("/");
  return <strong title={name}>{parts[parts.length - 1]}</strong>;
}
