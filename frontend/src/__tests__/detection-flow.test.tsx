// @vitest-environment jsdom
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "../App";
import { api } from "../api";
import type {
  BBox,
  BoxAnnotation,
  Draft,
  GeometryResult,
  Project,
  ProjectImageState,
  XY,
} from "../types";

type Canvas = {
  image: { id: number };
  boxes?: BoxAnnotation[];
  selected: string | null;
  draft: Draft | null;
  boxPreview?: BBox;
  showBoxes?: boolean;
  onBox: (box: BBox) => void;
  onBoxGeometry: (id: string, box: BBox) => void;
  onSelect: (id: string | null) => void;
  onPoint: (at: XY, negative: boolean) => void;
  onPolygon: (vertices: XY[], closed: boolean) => void;
};
const canvas = vi.hoisted(() => ({ current: null as Canvas | null }));
vi.mock("../CanvasEditor", () => ({
  default: (p: Canvas) => {
    canvas.current = p;
    return <div data-testid="canvas" />;
  },
}));
vi.mock("../api", async (original) => ({
  ...(await original<typeof import("../api")>()),
  api: vi.fn(),
}));
const request = vi.mocked(api);
const project: Project = {
  name: "Mixed",
  directory: "/tmp/mixed",
  image_root: "/tmp/images",
  categories: [{ id: 7, name: "Object", color: "#123456" }],
  images: [1, 2].map((id) => ({
    id,
    file_name: `${id}.png`,
    width: 16,
    height: 12,
    annotation_count: 0,
  })),
};
const geometry: GeometryResult = {
  mask: { size: [12, 16], counts: "V1480000000n2" },
  components: [
    {
      outer: [
        [3, 2],
        [3, 6],
        [8, 6],
        [8, 2],
      ],
      holes: [],
    },
  ],
};
let disk: Map<number, ProjectImageState>;
let predict: (body: {
  image_id: number;
}) => Promise<GeometryResult & { image_id: number }>;
let model = "ready";
const canvasState = () => canvas.current!;
const inferenceCalls = () =>
  request.mock.calls.filter(([path]) => path.startsWith("/infer/"));
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (e: Error) => void;
  const promise = new Promise<T>((r, j) => {
    resolve = r;
    reject = j;
  });
  return { resolve, reject, promise };
}
beforeEach(() => {
  model = "ready";
  canvas.current = null;
  disk = new Map(
    [1, 2].map((id) => [
      id,
      {
        image_id: id,
        revision: 0,
        annotations: [],
        draft: null,
        proposals: [],
      },
    ]),
  );
  predict = async (body) => ({ ...geometry, image_id: body.image_id });
  request.mockReset();
  request.mockImplementation(async (path, method = "GET", body) => {
    if (path === "/project") return structuredClone(project);
    if (path === "/health") return { model: { state: model } };
    const match = path.match(/^\/images\/(\d+)\/state$/);
    if (match) {
      const id = Number(match[1]);
      if (method === "PUT") {
        const next = structuredClone(body as ProjectImageState);
        next.revision++;
        disk.set(id, next);
      }
      return structuredClone(disk.get(id));
    }
    if (path === "/infer/points") return predict(body as { image_id: number });
    if (path === "/infer/text")
      return {
        image_id: 1,
        proposals: [
          { ...geometry, id: "sam", category_id: 7, iscrowd: 0, score: 0.9 },
        ],
      };
    throw new Error(`Unexpected ${method} ${path}`);
  });
});
afterEach(cleanup);
async function boot() {
  render(<App />);
  await screen.findByTestId("canvas");
}
async function detect() {
  fireEvent.click(screen.getByRole("button", { name: "Detection" }));
  await waitFor(() => expect(canvasState().boxes).toBeDefined());
}
function draw() {
  act(() => canvasState().onBox([1, 1, 10, 9]));
}
async function save() {
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(disk.get(1)?.schema_version).toBe(2));
}

it("draws and confirms boxes without SAM, including when the model is unavailable", async () => {
  model = "error";
  await boot();
  await detect();
  draw();
  expect(canvasState().boxes?.[0].bbox).toEqual([1, 1, 9, 8]);
  expect(inferenceCalls()).toHaveLength(0);
  expect(
    (
      screen.getByRole("button", {
        name: "Adjust box with SAM · S",
      }) as HTMLButtonElement
    ).disabled,
  ).toBe(true);
  fireEvent.click(screen.getByRole("button", { name: /Confirm box/ }));
  await save();
  expect(disk.get(1)?.annotations).toHaveLength(1);
  expect(disk.get(1)?.annotations[0]).toMatchObject({
    kind: "bbox",
    bbox: [1, 1, 9, 8],
  });
  expect(disk.get(1)?.annotations[0]).not.toHaveProperty("mask");
  expect(inferenceCalls()).toHaveLength(0);
});

it.each(["key", "button"])(
  "previews SAM adjustment via %s, accepts the same ID and undoes in one step",
  async (trigger) => {
    await boot();
    await detect();
    draw();
    fireEvent.click(screen.getByRole("button", { name: /Confirm box/ }));
    const original = structuredClone(canvasState().boxes![0]);
    await act(async () => {
      if (trigger === "key") fireEvent.keyDown(window, { key: "s" });
      else
        fireEvent.click(
          screen.getByRole("button", { name: "Adjust box with SAM · S" }),
        );
    });
    expect(inferenceCalls()).toHaveLength(1);
    expect(inferenceCalls()[0][2]).toMatchObject({
      image_id: 1,
      part: { box: [1, 1, 10, 9], points: [] },
    });
    expect(canvasState().boxPreview).toEqual([3, 2, 5, 4]);
    expect(canvasState().boxes![0]).toEqual(original);
    await save();
    expect(disk.get(1)?.annotations[0]).toEqual(original);
    fireEvent.click(screen.getByRole("button", { name: "Accept adjustment" }));
    expect(canvasState().boxes![0]).toEqual({
      ...original,
      bbox: [3, 2, 5, 4],
    });
    fireEvent.click(screen.getByRole("button", { name: "Undo detection" }));
    expect(canvasState().boxes![0]).toEqual(original);
    expect(canvasState().boxPreview).toBeUndefined();
  },
);

it("can discard an adjustment on a manual draft without changing the box", async () => {
  await boot();
  await detect();
  draw();
  await act(async () => fireEvent.keyDown(window, { key: "s" }));
  expect(canvasState().boxPreview).toEqual([3, 2, 5, 4]);
  fireEvent.click(screen.getByRole("button", { name: "Keep original box" }));
  expect(canvasState().boxes![0].bbox).toEqual([1, 1, 9, 8]);
  expect(canvasState().boxPreview).toBeUndefined();
  fireEvent.click(screen.getByRole("button", { name: /Confirm box/ }));
  expect(inferenceCalls()).toHaveLength(1);
});

it.each(["edit", "undo", "delete", "error", "empty"])(
  "preserves the original and ignores stale/invalid adjustment: %s",
  async (kind) => {
    const pending = deferred<GeometryResult & { image_id: number }>();
    predict = () => pending.promise;
    await boot();
    await detect();
    draw();
    const original = structuredClone(canvasState().boxes![0]);
    fireEvent.keyDown(window, { key: "s" });
    fireEvent.keyDown(window, { key: "s", repeat: true });
    expect(inferenceCalls()).toHaveLength(1);
    if (kind === "edit" || kind === "undo")
      act(() => canvasState().onBoxGeometry(original.id, [0, 0, 2, 2]));
    if (kind === "undo") fireEvent.keyDown(window, { key: "z", ctrlKey: true });
    if (kind === "delete") fireEvent.keyDown(window, { key: "Delete" });
    await act(async () => {
      if (kind === "error") pending.reject(new Error("SAM failed"));
      else
        pending.resolve({
          ...geometry,
          image_id: 1,
          ...(kind === "empty"
            ? { mask: { size: [12, 16] as [number, number], counts: "P6" } }
            : {}),
        });
    });
    expect(canvasState().boxPreview).toBeUndefined();
    expect(canvasState().boxes).toHaveLength(kind === "delete" ? 0 : 1);
    if (kind !== "delete")
      expect(canvasState().boxes![0].bbox).toEqual(
        kind === "edit" ? [0, 0, 2, 2] : original.bbox,
      );
  },
);

it("does not trigger S in fields/dialogs or on Ctrl+S, which still saves", async () => {
  await boot();
  await detect();
  draw();
  fireEvent.keyDown(screen.getByRole("textbox", { name: "Detection prompt" }), {
    key: "s",
  });
  fireEvent.keyDown(window, { key: "s", ctrlKey: true });
  fireEvent.click(
    screen.getByRole("button", { name: "Keyboard shortcuts · ?" }),
  );
  fireEvent.keyDown(window, { key: "s" });
  expect(inferenceCalls()).toHaveLength(0);
  await waitFor(() =>
    expect(disk.get(1)?.detection?.draft?.bbox).toEqual([1, 1, 9, 8]),
  );
});

it("isolates drafts and undo histories while switching tasks and receiving a SAM result", async () => {
  const pending = deferred<GeometryResult & { image_id: number }>();
  predict = () => pending.promise;
  await boot();
  expect(
    (
      screen.getByRole("checkbox", {
        name: "Show bounding boxes",
      }) as HTMLInputElement
    ).checked,
  ).toBe(false);
  fireEvent.click(
    screen.getByRole("checkbox", { name: "Show bounding boxes" }),
  );
  expect(canvasState().showBoxes).toBe(true);
  act(() =>
    canvasState().onPolygon(
      [
        [1, 1],
        [8, 1],
        [8, 8],
      ],
      false,
    ),
  );
  const polygon = structuredClone(canvasState().draft);
  await detect();
  draw();
  fireEvent.keyDown(window, { key: "s" });
  fireEvent.click(screen.getByRole("button", { name: "Segmentation" }));
  await act(async () => pending.resolve({ ...geometry, image_id: 1 }));
  expect(canvasState().draft).toEqual(polygon);
  expect(canvasState().boxes).toBeUndefined();
  await detect();
  expect(canvasState().boxPreview).toEqual([3, 2, 5, 4]);
  fireEvent.click(screen.getByRole("button", { name: "Keep original box" }));
  fireEvent.click(screen.getByRole("button", { name: /Confirm box/ }));
  fireEvent.click(screen.getByRole("button", { name: "Segmentation" }));
  fireEvent.keyDown(window, { key: "z", ctrlKey: true });
  expect(canvasState().draft).toBeNull();
  await detect();
  expect(canvasState().boxes).toHaveLength(1);
  await save();
  expect(disk.get(1)?.annotations).toHaveLength(1);
});

it("converts point and text results into boxes; only accepted proposals become annotations", async () => {
  await boot();
  await detect();
  await act(async () => canvasState().onPoint([5, 4], false));
  expect(canvasState().boxes![0].bbox).toEqual([3, 2, 5, 4]);
  fireEvent.click(screen.getByRole("button", { name: /Confirm box/ }));
  fireEvent.change(screen.getByRole("textbox", { name: "Detection prompt" }), {
    target: { value: "objects" },
  });
  await act(async () =>
    fireEvent.submit(screen.getByRole("form", { name: "Concept detection" })),
  );
  expect(canvasState().boxes).toHaveLength(2);
  await save();
  expect(disk.get(1)?.annotations).toHaveLength(1);
  fireEvent.click(
    screen.getByRole("checkbox", { name: "Accept box proposal 1" }),
  );
  fireEvent.click(
    screen.getByRole("button", { name: "Accept selected boxes" }),
  );
  await act(async () =>
    fireEvent.click(screen.getByRole("button", { name: "Save" })),
  );
  await waitFor(() => expect(disk.get(1)?.annotations).toHaveLength(2));
  expect(
    disk.get(1)?.annotations.every((a) => a.kind === "bbox" && !("mask" in a)),
  ).toBe(true);
});
