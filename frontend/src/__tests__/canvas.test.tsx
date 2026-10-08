// @vitest-environment jsdom
import { act, cleanup, fireEvent, render } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import Konva from "konva";
import CanvasEditor from "../CanvasEditor";
import type { Draft } from "../types";

// jsdom has no graphics engine. Stub only its drawing surface; keep the real
// Konva scene graph and React reconciler so invalid canvas children still fail.
vi.hoisted(() => {
  vi.stubGlobal(
    "ImageData",
    class {
      constructor(
        public data: Uint8ClampedArray,
        public width: number,
        public height: number,
      ) {}
    },
  );
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

it.each(["pan", "space"] as const)(
  "pans with %s after zooming without changing annotations, then restores editing coordinates",
  (mode) => {
    const annotation = {
      id: "ann",
      category_id: 1,
      mask: { size: [3, 4] as [number, number], counts: "11123O" },
      components: [
        {
          outer: [
            [0, 0],
            [3, 0],
            [3, 2],
            [0, 2],
          ] as [number, number][],
          holes: [],
        },
      ],
      iscrowd: 0,
    };
    const props = {
      image: {
        id: 1,
        file_name: "test.png",
        width: 4,
        height: 3,
        annotation_count: 1,
      },
      annotations: [annotation],
      proposals: [
        { ...annotation, id: "proposal", selected: false, score: 0.9 },
      ],
      draft: null,
      categories: [{ id: 1, name: "Object", color: "#06b6d4" }],
      selected: "ann",
      showMasks: true,
      opacity: 0.65,
      vertex: null,
      busy: false,
      onVertex: vi.fn(),
      onSelect: vi.fn(),
      onProposal: vi.fn(),
      onPoint: vi.fn(),
      onBox: vi.fn(),
      onPolygon: vi.fn(),
      onGeometry: vi.fn(),
      fitRef: { current: null },
    };
    const original = JSON.stringify(annotation);
    const { container, rerender } = render(
      <CanvasEditor {...props} tool={mode === "pan" ? "pan" : "select"} />,
    );
    if (mode === "space") fireEvent.keyDown(window, { code: "Space" });
    const wrap = container.querySelector(".canvas-wrap")!;
    const stage = Konva.stages.at(-1)!;
    const group = stage.findOne("Group")!;
    expect(wrap.classList.contains("panning")).toBe(true);
    // Edges and handles must not intercept a drag over a selected object.
    expect(stage.find("Line").every((node) => !node.isListening())).toBe(true);
    expect(stage.find("Circle").every((node) => !node.isListening())).toBe(
      true,
    );
    act(() => {
      stage.setPointersPositions({ clientX: 400, clientY: 300 });
      stage.fire("wheel", { evt: { deltaY: -1, preventDefault: vi.fn() } });
    });
    const before = group.position();
    const start = group.getAbsoluteTransform().point({ x: 1, y: 1 });
    act(() => {
      stage.setPointersPositions({ clientX: start.x, clientY: start.y });
      stage.fire("mousedown", { evt: { button: 0, preventDefault: vi.fn() } });
    });
    expect(wrap.classList.contains("dragging")).toBe(true);
    act(() => {
      stage.setPointersPositions({
        clientX: start.x + 80,
        clientY: start.y - 40,
      });
      stage.fire("mousemove", { evt: {} });
    });
    expect(group.position()).toEqual({ x: before.x + 80, y: before.y - 40 });
    act(() => stage.fire("mouseup", { evt: {} }));
    expect(wrap.classList.contains("dragging")).toBe(false);
    for (const callback of [
      props.onSelect,
      props.onVertex,
      props.onProposal,
      props.onPoint,
      props.onBox,
      props.onPolygon,
      props.onGeometry,
    ])
      expect(callback).not.toHaveBeenCalled();
    expect(JSON.stringify(annotation)).toBe(original);

    if (mode === "space") fireEvent.keyUp(window, { code: "Space" });
    rerender(<CanvasEditor {...props} tool="select" />);
    expect(wrap.classList.contains("panning")).toBe(false);
    expect(
      stage.find("Circle").filter((node) => node.isListening()),
    ).toHaveLength(4);
    rerender(<CanvasEditor {...props} tool="positive" selected={null} />);
    const at = group.getAbsoluteTransform().point({ x: 1, y: 1 });
    act(() => {
      stage.setPointersPositions({ clientX: at.x, clientY: at.y });
      stage.fire("mousedown", { evt: { button: 0, preventDefault: vi.fn() } });
    });
    expect(props.onPoint).toHaveBeenCalledOnce();
    const [point, negative] = props.onPoint.mock.calls[0];
    expect(point[0]).toBeCloseTo(1);
    expect(point[1]).toBeCloseTo(1);
    expect(negative).toBe(false);
  },
);

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
    onPolygon: vi.fn(),
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

it("draws and closes a polygon in original coordinates after zooming, without emitting point prompts", () => {
  const onPolygon = vi.fn();
  const onPoint = vi.fn();
  const props = {
    image: {
      id: 1,
      file_name: "large.png",
      width: 1000,
      height: 800,
      annotation_count: 0,
    },
    annotations: [],
    proposals: [],
    categories: [],
    selected: null,
    tool: "polygon" as const,
    showMasks: true,
    opacity: 0.4,
    vertex: null,
    busy: false,
    onVertex: vi.fn(),
    onSelect: vi.fn(),
    onProposal: vi.fn(),
    onPoint,
    onBox: vi.fn(),
    onPolygon,
    onGeometry: vi.fn(),
    fitRef: { current: null },
  };
  const { rerender } = render(<CanvasEditor {...props} draft={null} />);
  const stage = Konva.stages.at(-1)!;
  function clickAt(x: number, y: number) {
    const transform = stage.findOne("Group")!.getAbsoluteTransform();
    const screen = transform.point({ x, y });
    act(() => {
      stage.setPointersPositions({ clientX: screen.x, clientY: screen.y });
      stage.fire("mousedown", { evt: { button: 0, preventDefault: vi.fn() } });
    });
  }
  clickAt(123, 234);
  expect(onPolygon).toHaveBeenLastCalledWith([[123, 234]], false);
  const draft: Draft = {
    id: "draft",
    category_id: 0,
    active_part_id: "part",
    parts: [
      {
        id: "part",
        points: [],
        polygon: {
          vertices: [
            [123, 234],
            [500, 234],
            [500, 600],
          ],
          closed: false,
        },
      },
    ],
  };
  rerender(<CanvasEditor {...props} draft={draft} />);
  expect(stage.find(".polygon-vertex")).toHaveLength(3);
  act(() => {
    stage.setPointersPositions({ clientX: 400, clientY: 300 });
    stage.fire("wheel", { evt: { deltaY: -1, preventDefault: vi.fn() } });
  });
  clickAt(123, 234);
  expect(onPolygon).toHaveBeenLastCalledWith(
    draft.parts[0].polygon!.vertices,
    true,
  );
  expect(onPoint).not.toHaveBeenCalled();
  rerender(
    <CanvasEditor
      {...props}
      draft={{
        ...draft,
        parts: [
          {
            ...draft.parts[0],
            polygon: { ...draft.parts[0].polygon!, closed: true },
          },
        ],
      }}
    />,
  );
  expect(stage.findOne(".polygon-prompt")!.getAttr("closed")).toBe(true);
  const calls = onPolygon.mock.calls.length;
  clickAt(600, 600);
  expect(onPolygon).toHaveBeenCalledTimes(calls);
  const handle = stage.findOne(".polygon-vertex")!;
  for (const outside of [-20, -100]) {
    act(() => {
      handle.position({ x: outside, y: 200 });
      handle.fire("dragmove", { evt: {} });
    });
    expect(handle.x()).toBe(0);
  }
  act(() => handle.fire("dragend", { evt: {} }));
  expect(onPolygon).toHaveBeenLastCalledWith(
    [
      [0, 200],
      [500, 234],
      [500, 600],
    ],
    true,
  );
  rerender(
    <CanvasEditor
      {...props}
      draft={{
        ...draft,
        parts: [
          {
            ...draft.parts[0],
            polygon: {
              vertices: onPolygon.mock.calls.at(-1)![0],
              closed: true,
            },
          },
        ],
      }}
    />,
  );
  expect(handle.x()).toBe(0);
});

it("outlines every mask, including holes and islands, and honors opacity and mask visibility", () => {
  const mask = { size: [5, 6] as [number, number], counts: "032NO012>N" };
  const components = [
    {
      outer: [
        [0, 0],
        [3, 0],
        [3, 3],
        [0, 3],
      ] as [number, number][],
      holes: [
        [
          [1, 1],
          [2, 1],
          [2, 2],
          [1, 2],
        ] as [number, number][],
      ],
    },
    {
      outer: [
        [5, 4],
        [6, 4],
        [6, 5],
        [5, 5],
      ] as [number, number][],
      holes: [],
    },
  ];
  const annotation = {
    id: "ann",
    category_id: 1,
    mask,
    components,
    iscrowd: 0,
  };
  const original = JSON.stringify(annotation);
  const props = {
    image: {
      id: 1,
      file_name: "islands.png",
      width: 6,
      height: 5,
      annotation_count: 1,
    },
    annotations: [annotation],
    proposals: [{ ...annotation, id: "proposal", selected: false, score: 0.9 }],
    draft: {
      id: "draft",
      category_id: 1,
      active_part_id: "part",
      parts: [{ id: "part", points: [], mask, components }],
    },
    categories: [{ id: 1, name: "Object", color: "#06b6d4" }],
    selected: null as string | null,
    tool: "select" as const,
    showMasks: true,
    opacity: 0.65,
    vertex: null,
    busy: false,
    onVertex: vi.fn(),
    onSelect: vi.fn(),
    onProposal: vi.fn(),
    onPoint: vi.fn(),
    onBox: vi.fn(),
    onPolygon: vi.fn(),
    onGeometry: vi.fn(),
    fitRef: { current: null },
  };
  const { rerender } = render(<CanvasEditor {...props} />);
  const stage = Konva.stages.at(-1)!;
  expect(stage.find(".mask-outline")).toHaveLength(3);
  expect(stage.find(".mask-outline-color")).toHaveLength(9);
  expect(stage.find(".mask-outline-halo")).toHaveLength(9);
  expect(
    stage
      .find(".mask-outline-color")
      .slice(0, 3)
      .map((line) => line.getAttr("points")),
  ).toEqual([
    components[0].outer.flat(),
    components[0].holes[0].flat(),
    components[1].outer.flat(),
  ]);
  expect(stage.find(".mask-fill").map((node) => node.opacity())).toEqual([
    0.65,
    0.65 * 0.85,
    0.65 * 1.12,
  ]);
  expect(stage.find("Circle")).toHaveLength(0);
  act(() => {
    stage.setPointersPositions({ clientX: 400, clientY: 300 });
    stage.fire("wheel", { evt: { deltaY: -1, preventDefault: vi.fn() } });
  });
  expect(
    stage.findOne(".mask-outline-color")!.getAttr("strokeWidth") *
      stage.findOne("Group")!.scaleX(),
  ).toBeCloseTo(2);
  expect(
    stage.findOne(".mask-outline-halo")!.getAttr("strokeWidth") *
      stage.findOne("Group")!.scaleX(),
  ).toBeCloseTo(4);
  rerender(<CanvasEditor {...props} selected="ann" opacity={0} />);
  expect(stage.find(".mask-fill").every((node) => node.opacity() === 0)).toBe(
    true,
  );
  expect(stage.find("Circle")).toHaveLength(12);
  rerender(<CanvasEditor {...props} selected="ann" showMasks={false} />);
  expect(stage.find(".mask-fill")).toHaveLength(0);
  expect(stage.find(".mask-outline")).toHaveLength(0);
  expect(stage.find("Circle")).toHaveLength(12);
  expect(JSON.stringify(annotation)).toBe(original);
  rerender(
    <CanvasEditor
      {...props}
      image={{ ...props.image, width: 12, height: 10 }}
      annotations={[
        { ...annotation, mask: { size: [10, 12], counts: "f1280^1" } },
      ]}
      proposals={[]}
      draft={null}
    />,
  );
  expect(stage.findOne(".mask-fill")!.position()).toEqual({ x: 5, y: 4 });
});
