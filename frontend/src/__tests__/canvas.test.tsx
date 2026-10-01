// @vitest-environment jsdom
import { cleanup, render } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import Konva from "konva";
import CanvasEditor from "../CanvasEditor";
import type { Draft } from "../types";

// jsdom has no graphics engine. Stub only its drawing surface; keep the real
// Konva scene graph and React reconciler so invalid canvas children still fail.
vi.hoisted(() => {
  Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
    configurable: true,
    value: function (this: HTMLCanvasElement) {
      const canvas = this;
      return new Proxy<Record<string, unknown>>(
        {},
        {
          get: (_target, key) => {
            if (key === "canvas") return canvas;
            if (key === "measureText") return () => ({ width: 0 });
            if (key === "getLineDash") return () => [];
            if (key === "createImageData" || key === "getImageData")
              return (width: number, height: number) => ({
                width,
                height,
                data: new Uint8ClampedArray(width * height * 4),
              });
            return () => {};
          },
          set: () => true,
        },
      );
    },
  });
  Object.defineProperty(HTMLCanvasElement.prototype, "toDataURL", {
    configurable: true,
    value: () => "data:image/png;base64,",
  });
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
});

afterEach(cleanup);

it("renders, updates, and removes draft point/box/mask nodes in the real Konva scene", () => {
  const props = {
    image: {
      id: 20,
      file_name: "test.png",
      width: 4,
      height: 3,
      annotation_count: 0,
    },
    annotations: [],
    proposals: [],
    categories: [{ id: 0, name: "Objeto", color: "#75a88d" }],
    selected: null,
    tool: "positive" as const,
    showMasks: true,
    opacity: 0.4,
    vertex: null,
    busy: false,
    onVertex: vi.fn(),
    onSelect: vi.fn(),
    onProposal: vi.fn(),
    onPoint: vi.fn(),
    onBox: vi.fn(),
    onGeometry: vi.fn(),
    fitRef: { current: null },
  };
  const { rerender } = render(<CanvasEditor {...props} draft={null} />);
  const stage = Konva.stages.at(-1)!;
  expect(stage.find("Circle")).toHaveLength(0);
  const draft: Draft = {
    id: "draft",
    category_id: 0,
    active_part_id: "part",
    parts: [
      {
        id: "part",
        points: [
          { x: 1, y: 1, label: 1 },
          { x: 3, y: 2, label: 0 },
        ],
      },
    ],
  };
  rerender(<CanvasEditor {...props} draft={draft} />);
  expect(stage.find("Circle")).toHaveLength(2);
  expect(stage.find("Line")).toHaveLength(3);
  rerender(
    <CanvasEditor
      {...props}
      draft={{
        ...draft,
        parts: [
          {
            ...draft.parts[0],
            box: [0, 0, 3, 2],
            mask: { size: [3, 4], counts: "11123O" },
          },
        ],
      }}
    />,
  );
  expect(stage.find("Rect")).toHaveLength(2);
  expect(stage.find("Image")).toHaveLength(2);
  rerender(<CanvasEditor {...props} draft={null} />);
  expect(stage.find("Circle")).toHaveLength(0);
  expect(stage.find("Rect")).toHaveLength(1);
});
