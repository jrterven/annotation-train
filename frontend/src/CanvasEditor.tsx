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
  SegmentationAnnotation as Annotation,
  Category,
  Component,
  Draft,
  Mask,
  Proposal,
  SourceImage,
  Tool,
  XY,
  BBox,
  BoxAnnotation,
} from "./types";
import { maskContains, maskOverlay, maskBounds } from "./masks";
import { assetURL } from "./api";
import { sourceImages } from "./imageCache";
import BoxShape from "./BoxShape";
import { boxContains, unionBoxes } from "./boxes";
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
  showBoxes?: boolean;
  boxes?: BoxAnnotation[];
  pendingBoxIds?: string[];
  boxPreview?: BBox;
  onBoxGeometry?: (id: string, box: BBox) => void;
  opacity: number;
  vertex: Vertex | null;
  busy: boolean;
  onVertex: (v: Vertex | null) => void;
  onSelect: (id: string | null) => void;
  onProposal: (id: string) => void;
  onPoint: (point: XY, negative: boolean) => void;
  onBox: (box: [number, number, number, number]) => void;
  onPolygon: (vertices: XY[], closed: boolean) => void;
  onGeometry: (id: string, components: Component[]) => void;
  fitRef: React.RefObject<(() => void) | null>;
};
function useImage(url: string, remember = false) {
  const [loaded, setLoaded] = useState<{
    url: string;
    image: HTMLImageElement;
  }>();
  useEffect(() => {
    let valid = true;
    if (remember) {
      void sourceImages
        .load(url)
        .then((image) => {
          if (valid) setLoaded({ url, image });
        })
        .catch(() => {});
      return () => {
        valid = false;
      };
    }
    const image = new window.Image();
    image.onload = () => {
      if (valid) setLoaded({ url, image });
    };
    image.src = url;
    return () => {
      valid = false;
    };
  }, [url, remember]);
  // Never paint the previous image while a different URL is loading.
  return loaded?.url === url
    ? loaded.image
    : remember
      ? sourceImages.peek(url)
      : undefined;
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
  const overlay = useMemo(() => maskOverlay(mask, color), [mask, color]);
  const image = useImage(overlay.url);
  return (
    <KonvaImage
      name="mask-fill"
      x={overlay.x}
      y={overlay.y}
      image={image}
      opacity={Math.min(1, Math.max(0, opacity))}
      listening={false}
    />
  );
}
const flat = (points: XY[]) => points.flat();
const rings = (c: Component) => [c.outer, ...c.holes];
function MaskOutline({
  components,
  color,
  scale,
}: {
  components: Component[];
  color: string;
  scale: number;
}) {
  return (
    <Group name="mask-outline" listening={false}>
      {components.flatMap((component, ci) =>
        rings(component).map((ring, ri) => (
          <Group key={`${ci}-${ri}`}>
            <Line
              name="mask-outline-halo"
              points={flat(ring)}
              closed
              stroke="#111827"
              strokeWidth={4 / scale}
              lineJoin="round"
              listening={false}
            />
            <Line
              name="mask-outline-color"
              points={flat(ring)}
              closed
              stroke={color}
              strokeWidth={2 / scale}
              lineJoin="round"
              listening={false}
            />
          </Group>
        )),
      )}
    </Group>
  );
}
export default function CanvasEditor(p: Props) {
  const container = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const [size, setSize] = useState({
    width: 800,
    height: 600,
    measured: false,
  });
  const [{ zoom, pan }, setView] = useState({ zoom: 1, pan: { x: 0, y: 0 } });
  const setPan = (pan: { x: number; y: number }) =>
    setView((view) => ({ ...view, pan }));
  const previousViewport = useRef<{
    image: number;
    width: number;
    height: number;
    fit: number;
    measured: boolean;
  } | null>(null);
  const [space, setSpace] = useState(false);
  const [dragPan, setDragPan] = useState<{
    x: number;
    y: number;
    ox: number;
    oy: number;
  } | null>(null);
  const panning = space || p.tool === "pan";
  const [box, setBox] = useState<{ start: XY; end: XY } | null>(null);
  const [edit, setEdit] = useState<Component[] | null>(null);
  const [hover, setHover] = useState<XY | null>(null);
  const [polygonEdit, setPolygonEdit] = useState<{
    id: string;
    vertices: XY[];
  } | null>(null);
  const sourceURL = assetURL(`/images/${p.image.id}/file`);
  const source = useImage(sourceURL, sourceURL.startsWith("/api/v1/"));
  const fit = Math.min(
    (size.width - 100) / p.image.width,
    (size.height - 100) / p.image.height,
    1,
  );
  const scale = Math.max(0.001, fit * zoom);
  const chosen = p.annotations.find((a) => a.id === p.selected);
  const activePart = p.draft?.parts.find(
    (part) => part.id === p.draft?.active_part_id,
  );
  const activePolygon = activePart?.polygon;
  const geometryKey = JSON.stringify(chosen?.controls ?? chosen?.components);
  // Draft inference and autosave can replace the parent state while dragging.
  // Equal geometry must keep its reference so those updates don't reset edits.
  const baseGeometry = useMemo(
    () => chosen?.controls ?? chosen?.components,
    [p.selected, geometryKey],
  );
  const geometry = edit || baseGeometry;
  const fitView = () => {
    setView({
      zoom: 1,
      pan: {
        x: (size.width - p.image.width * fit) / 2,
        y: (size.height - p.image.height * fit) / 2,
      },
    });
  };
  p.fitRef.current = fitView;
  useEffect(() => {
    if (!container.current) return;
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) setSize({ width, height, measured: true });
    });
    observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const previous = previousViewport.current;
    if (!previous || previous.image !== p.image.id || !previous.measured) {
      fitView();
      setEdit(null);
      setBox(null);
    } else {
      // Resizing a panel keeps the same original pixel under the canvas center.
      setView((view) => {
        const next = Math.max(
          0.1,
          Math.min(16, (view.zoom * previous.fit) / fit),
        );
        const ratio = (fit * next) / (previous.fit * view.zoom);
        return {
          zoom: next,
          pan: {
            x: size.width / 2 - (previous.width / 2 - view.pan.x) * ratio,
            y: size.height / 2 - (previous.height / 2 - view.pan.y) * ratio,
          },
        };
      });
    }
    previousViewport.current = { image: p.image.id, ...size, fit };
  }, [p.image.id, size.width, size.height, size.measured]);
  useEffect(() => {
    setEdit(null);
  }, [p.selected, baseGeometry]);
  useEffect(() => {
    setBox(null);
  }, [p.tool]);
  useEffect(() => {
    setPolygonEdit(null);
    setHover(null);
  }, [p.image.id, activePart?.id, JSON.stringify(activePolygon), p.tool]);
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
  function zoomBy(
    factor: number,
    anchor = { x: size.width / 2, y: size.height / 2 },
  ) {
    // Functional updates accumulate wheel events delivered in the same frame.
    setView((view) => {
      const next = Math.max(0.1, Math.min(16, view.zoom * factor));
      const ratio = next / view.zoom;
      return {
        zoom: next,
        pan: {
          x: anchor.x - (anchor.x - view.pan.x) * ratio,
          y: anchor.y - (anchor.y - view.pan.y) * ratio,
        },
      };
    });
  }
  function down(e: Konva.KonvaEventObject<MouseEvent>) {
    const raw = stageRef.current?.getPointerPosition();
    if (((panning && e.evt.button === 0) || e.evt.button === 1) && raw) {
      e.evt.preventDefault();
      setDragPan({ ...raw, ox: pan.x, oy: pan.y });
      return;
    }
    if (panning || e.target.attrs.vertex) return;
    const at = coord();
    if (
      !at ||
      at[0] < 0 ||
      at[1] < 0 ||
      at[0] >= p.image.width ||
      at[1] >= p.image.height
    )
      return;
    if (p.busy) return;
    if (p.tool === "polygon") {
      if (e.evt.button !== 0 || activePolygon?.closed) return;
      const vertices = activePolygon?.vertices || [];
      if (
        vertices.length >= 3 &&
        Math.hypot(at[0] - vertices[0][0], at[1] - vertices[0][1]) * scale <= 9
      )
        p.onPolygon(vertices, true);
      else if (
        !vertices.length ||
        Math.hypot(at[0] - vertices.at(-1)![0], at[1] - vertices.at(-1)![1]) *
          scale >
          2
      )
        p.onPolygon([...vertices, bounded(at)], false);
      return;
    }
    if (p.tool === "box") {
      setBox({ start: at, end: at });
      return;
    }
    if (p.tool === "positive" || p.tool === "negative") {
      p.onPoint(at, p.tool === "negative" || e.evt.button === 2);
      return;
    }
    if (p.boxes) {
      const box = [...p.boxes].reverse().find((a) => boxContains(a.bbox, at));
      p.onSelect(box?.id || null);
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
    if (p.tool === "polygon" && !activePolygon?.closed) {
      const at = coord();
      setHover(at ? bounded(at) : null);
    }
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
      className={`canvas-wrap tool-${p.tool} ${panning ? "panning" : ""} ${dragPan ? "dragging" : ""}`}
      ref={container}
    >
      <Stage
        ref={stageRef}
        width={size.width}
        height={size.height}
        onMouseDown={down}
        onMouseMove={move}
        onMouseUp={up}
        onMouseLeave={() => {
          up();
          setHover(null);
        }}
        onContextMenu={(e) => e.evt.preventDefault()}
        onWheel={(e) => {
          e.evt.preventDefault();
          const pointer = stageRef.current?.getPointerPosition();
          const unit =
            e.evt.deltaMode === 1
              ? 16
              : e.evt.deltaMode === 2
                ? size.height
                : 1;
          const delta = Math.max(-100, Math.min(100, e.evt.deltaY * unit));
          // Small trackpad deltas remain small; large wheel notches are capped.
          zoomBy(Math.exp(-delta * 0.0008), pointer || undefined);
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
            {p.boxes?.map((a) => (
              <BoxShape
                key={a.id}
                bbox={a.bbox}
                color={
                  p.categories.find((c) => c.id === a.category_id)?.color ||
                  "#38bdf8"
                }
                scale={scale}
                width={p.image.width}
                height={p.image.height}
                selected={p.selected === a.id}
                editable={!p.busy && !panning && p.tool === "select"}
                dashed={p.pendingBoxIds?.includes(a.id)}
                onSelect={
                  !panning && p.tool === "select"
                    ? () => p.onSelect(a.id)
                    : undefined
                }
                onChange={(box) => p.onBoxGeometry?.(a.id, box)}
              />
            ))}
            {p.boxPreview && (
              <BoxShape
                bbox={p.boxPreview}
                color="#22c55e"
                scale={scale}
                width={p.image.width}
                height={p.image.height}
                dashed
              />
            )}
            {p.showBoxes &&
              [...p.annotations, ...p.proposals].map((a) => {
                const bounds = maskBounds(a.mask);
                return (
                  bounds && (
                    <BoxShape
                      key={`bounds-${a.id}`}
                      bbox={bounds}
                      color={
                        p.categories.find((c) => c.id === a.category_id)
                          ?.color || "#38bdf8"
                      }
                      scale={scale}
                      width={p.image.width}
                      height={p.image.height}
                    />
                  )
                );
              })}
            {p.showBoxes &&
              (() => {
                const bounds = unionBoxes(
                  (p.draft?.parts || []).flatMap((part) => {
                    const box = part.mask && maskBounds(part.mask);
                    return box ? [box] : [];
                  }),
                );
                return (
                  bounds && (
                    <BoxShape
                      bbox={bounds}
                      color="#faf8ed"
                      scale={scale}
                      width={p.image.width}
                      height={p.image.height}
                      dashed
                    />
                  )
                );
              })()}
            {p.showMasks &&
              p.annotations.map((a) => {
                const color =
                  p.categories.find((c) => c.id === a.category_id)?.color ||
                  "#06b6d4";
                return (
                  <Group key={a.id}>
                    <MaskImage
                      mask={a.mask}
                      color={color}
                      opacity={p.opacity * (a.id === p.selected ? 1.15 : 1)}
                    />
                    <MaskOutline
                      components={a.components}
                      color={color}
                      scale={scale}
                    />
                  </Group>
                );
              })}
            {p.showMasks &&
              p.proposals.map((a) => {
                const color = a.selected ? "#22c55e" : "#38bdf8";
                return (
                  <Group key={a.id}>
                    <MaskImage
                      mask={a.mask}
                      color={color}
                      opacity={p.opacity * (a.selected ? 1.12 : 0.85)}
                    />
                    <MaskOutline
                      components={a.components}
                      color={color}
                      scale={scale}
                    />
                  </Group>
                );
              })}
            {p.draft?.parts.map((part) => {
              const active = part.id === p.draft?.active_part_id;
              const vertices =
                polygonEdit?.id === part.id
                  ? polygonEdit.vertices
                  : part.polygon?.vertices || [];
              const polygonEditable =
                active && p.tool === "polygon" && !p.busy && !space;
              return (
                <Group key={part.id}>
                  {part.mask && p.showMasks && (
                    <Group>
                      <MaskImage
                        mask={part.mask}
                        color={
                          p.categories.find(
                            (c) => c.id === p.draft?.category_id,
                          )?.color || "#06b6d4"
                        }
                        opacity={p.opacity * 1.12}
                      />
                      <MaskOutline
                        components={part.components || []}
                        color={
                          p.categories.find(
                            (c) => c.id === p.draft?.category_id,
                          )?.color || "#06b6d4"
                        }
                        scale={scale}
                      />
                    </Group>
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
                  {part.polygon && (!part.mask || polygonEditable) && (
                    <Group>
                      <Line
                        name="polygon-prompt"
                        points={flat(vertices)}
                        closed={part.polygon.closed}
                        stroke={active ? "#faf5da" : "#a9aca4"}
                        strokeWidth={1.5 / scale}
                        fill={
                          part.polygon.closed && !part.mask
                            ? "rgba(250,245,218,.1)"
                            : undefined
                        }
                        dash={[5 / scale, 3 / scale]}
                        listening={false}
                      />
                      {active &&
                        p.tool === "polygon" &&
                        !part.polygon.closed &&
                        vertices.length > 0 &&
                        hover && (
                          <Line
                            points={flat([vertices.at(-1)!, hover])}
                            stroke="#faf5da"
                            strokeWidth={1 / scale}
                            dash={[4 / scale, 4 / scale]}
                            listening={false}
                          />
                        )}
                      {active &&
                        vertices.map((point, index) => (
                          <Circle
                            key={index}
                            name="polygon-vertex"
                            x={point[0]}
                            y={point[1]}
                            radius={
                              (index === 0 && !part.polygon!.closed ? 5 : 3.5) /
                              scale
                            }
                            fill={index === 0 ? "#eed5a0" : "#fffef4"}
                            stroke="#544d39"
                            strokeWidth={1 / scale}
                            hitStrokeWidth={8 / scale}
                            listening={polygonEditable}
                            draggable={
                              polygonEditable &&
                              (index !== 0 || !!part.polygon!.closed)
                            }
                            onMouseDown={(e) => {
                              e.cancelBubble = true;
                              if (
                                index === 0 &&
                                !part.polygon!.closed &&
                                vertices.length >= 3 &&
                                e.evt.button === 0
                              )
                                p.onPolygon(vertices, true);
                            }}
                            onDragMove={(e) => {
                              const next = structuredClone(vertices);
                              const at = bounded([e.target.x(), e.target.y()]);
                              e.target.position({ x: at[0], y: at[1] });
                              next[index] = at;
                              setPolygonEdit({ id: part.id, vertices: next });
                            }}
                            onDragEnd={(e) => {
                              const next = structuredClone(vertices);
                              const at = bounded([e.target.x(), e.target.y()]);
                              e.target.position({ x: at[0], y: at[1] });
                              next[index] = at;
                              setPolygonEdit(null);
                              p.onPolygon(next, part.polygon!.closed);
                            }}
                          />
                        ))}
                    </Group>
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
              );
            })}
            {chosen &&
              geometry?.flatMap((component, ci) =>
                rings(component).map((ring, ri) => (
                  <Group key={`${ci}-${ri}`}>
                    <Line
                      listening={!panning}
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
                            listening={!panning}
                            x={point[0]}
                            y={point[1]}
                            radius={(selected ? 5 : 3.5) / scale}
                            fill={selected ? "#eed5a0" : "#fffef4"}
                            stroke="#544d39"
                            strokeWidth={1 / scale}
                            draggable={!p.busy && !panning}
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
          <ScanLine size={13} /> Original pixels
        </span>
      </div>
      <div className="zoom-controls">
        <button
          onClick={() => zoomBy(1.1)}
          title="Zoom in"
          aria-label="Zoom in"
        >
          <Plus size={16} />
        </button>
        <span>{Math.round(scale * 100)}%</span>
        <button
          onClick={() => zoomBy(1 / 1.1)}
          title="Zoom out"
          aria-label="Zoom out"
        >
          <Minus size={16} />
        </button>
        <i />
        <button onClick={fitView} title="Fit image · F" aria-label="Fit image">
          <Focus size={17} />
        </button>
      </div>
      <div className="canvas-instruction">
        {panning
          ? "Drag to pan"
          : p.tool === "polygon"
            ? activePolygon?.closed
              ? "Refine with SAM · drag vertices to adjust"
              : "Click to add vertices · click the first or press Enter to close"
            : p.tool === "box"
              ? "Drag a box around the object"
              : p.tool === "negative"
                ? "Click to exclude a region"
                : p.tool === "positive"
                  ? "Positive click · right-click to exclude"
                  : chosen
                    ? "Drag vertices · double-click an edge to insert"
                    : "Select an instance"}
        <span>Space + drag to pan</span>
      </div>
    </div>
  );
}
function FileName({ name }: { name: string }) {
  const parts = name.split("/");
  return <strong title={name}>{parts[parts.length - 1]}</strong>;
}
