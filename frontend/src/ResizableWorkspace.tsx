import { createContext, useContext, useEffect, useRef, useState } from "react";
import type { CSSProperties, ReactNode } from "react";

type Side = "left" | "right";
type Widths = Record<Side, number>;
const defaults: Widths = { left: 235, right: 267 };
const minimum: Widths = { left: 176, right: 224 };
const maximum: Widths = { left: 480, right: 520 };
const storageKey = "annotation.panel-widths.v1";
const clamp = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(max, value));
function savedWidths(): Widths {
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
    if (
      saved &&
      [saved.left, saved.right].every(
        (v) => typeof v === "number" && Number.isFinite(v),
      )
    )
      return {
        left: clamp(saved.left, minimum.left, maximum.left),
        right: clamp(saved.right, minimum.right, maximum.right),
      };
  } catch {
    /* Browser storage is optional. */
  }
  return defaults;
}
const Panels = createContext<{
  widths: Widths;
  limits: Widths;
  resize: (side: Side, width: number) => void;
} | null>(null);

export default function ResizableWorkspace({
  children,
  rightVisible,
}: {
  children: ReactNode;
  rightVisible: boolean;
}) {
  const element = useRef<HTMLDivElement>(null);
  const [preferred, setPreferred] = useState(savedWidths);
  const [availableWidth, setAvailableWidth] = useState(window.innerWidth);
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) =>
      setAvailableWidth(entry.contentRect.width),
    );
    if (element.current) observer.observe(element.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    try {
      localStorage.setItem(storageKey, JSON.stringify(preferred));
    } catch {
      /* Optional preference. */
    }
  }, [preferred]);
  // Preserve room for editing even when saved preferences exceed the viewport.
  const budget = Math.max(
    minimum.left + (rightVisible ? minimum.right : 0),
    availableWidth - 320 - (rightVisible ? 12 : 6),
  );
  const left = clamp(
    preferred.left,
    minimum.left,
    Math.min(maximum.left, budget - (rightVisible ? minimum.right : 0)),
  );
  const right = rightVisible
    ? clamp(
        preferred.right,
        minimum.right,
        Math.min(maximum.right, budget - left),
      )
    : preferred.right;
  const limits = {
    left: Math.min(maximum.left, budget - (rightVisible ? right : 0)),
    right: Math.min(maximum.right, budget - left),
  };
  const resize = (side: Side, width: number) =>
    setPreferred((current) => ({
      ...current,
      [side]: clamp(width, minimum[side], limits[side]),
    }));
  return (
    <Panels.Provider value={{ widths: { left, right }, limits, resize }}>
      <div
        className="workspace"
        ref={element}
        style={
          {
            "--sidebar": `${left}px`,
            "--inspector": `${right}px`,
          } as CSSProperties
        }
      >
        {children}
      </div>
    </Panels.Provider>
  );
}

export function PanelResizeHandle({ side }: { side: Side }) {
  const panels = useContext(Panels);
  const drag = useRef<{ id: number; x: number; width: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  useEffect(() => {
    const stop = () => {
      drag.current = null;
      setDragging(false);
    };
    window.addEventListener("blur", stop);
    return () => window.removeEventListener("blur", stop);
  }, []);
  if (!panels) return null;
  const { widths, limits, resize } = panels;
  const name = side === "left" ? "images" : "annotations";
  const direction = side === "left" ? 1 : -1;
  return (
    <div
      className={`panel-resizer${dragging ? " dragging" : ""}`}
      role="separator"
      aria-label={`Resize ${name} panel`}
      aria-orientation="vertical"
      aria-valuemin={minimum[side]}
      aria-valuemax={limits[side]}
      aria-valuenow={Math.round(widths[side])}
      aria-valuetext={`${Math.round(widths[side])} pixels`}
      tabIndex={0}
      title={`Resize ${name} panel · drag or use arrow keys · double-click to reset`}
      onPointerDown={(event) => {
        if (event.button !== 0 || drag.current) return;
        event.preventDefault();
        event.currentTarget.focus();
        event.currentTarget.setPointerCapture(event.pointerId);
        drag.current = {
          id: event.pointerId,
          x: event.clientX,
          width: widths[side],
        };
        setDragging(true);
      }}
      onPointerMove={(event) => {
        const start = drag.current;
        if (start && event.pointerId === start.id)
          resize(side, start.width + direction * (event.clientX - start.x));
      }}
      onPointerUp={(event) => {
        if (event.pointerId !== drag.current?.id) return;
        drag.current = null;
        setDragging(false);
        event.currentTarget.releasePointerCapture(event.pointerId);
      }}
      onPointerCancel={() => {
        if (drag.current) resize(side, drag.current.width);
        drag.current = null;
        setDragging(false);
      }}
      onLostPointerCapture={() => {
        drag.current = null;
        setDragging(false);
      }}
      onDoubleClick={() => resize(side, defaults[side])}
      onKeyDown={(event) => {
        const step = event.shiftKey ? 48 : 16;
        if (event.key === "ArrowLeft")
          resize(side, widths[side] - direction * step);
        else if (event.key === "ArrowRight")
          resize(side, widths[side] + direction * step);
        else if (event.key === "Home") resize(side, minimum[side]);
        else if (event.key === "End") resize(side, limits[side]);
        else if (event.key === "Enter") resize(side, defaults[side]);
        else if (event.key === "Escape" && drag.current) {
          resize(side, drag.current.width);
          event.currentTarget.releasePointerCapture(drag.current.id);
          drag.current = null;
          setDragging(false);
        } else return;
        event.preventDefault();
        event.stopPropagation();
      }}
    />
  );
}
