// @vitest-environment jsdom
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { api, ApiError } from "../api";
import type {
  SegmentationAnnotation as Annotation,
  Component,
  GeometryResult,
  ImageState,
  Mask,
  Project,
} from "../types";

type CanvasState = {
  image: { id: number };
  annotations: Annotation[];
  selected: string | null;
  busy: boolean;
  onGeometry: (id: string, components: Component[]) => void;
};
const canvas = vi.hoisted(() => ({ current: null as CanvasState | null }));
vi.mock("../CanvasEditor", () => ({
  default: (props: CanvasState) => {
    canvas.current = props;
    return <div data-testid="canvas" data-image-id={props.image.id} />;
  },
}));
vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api")>()),
  api: vi.fn(),
}));
const request = vi.mocked(api);
// Exact RLE/contours for an 8×8 image, with a 6×6 square and a 2×2 hole.
const originalGeometry: GeometryResult = {
  mask: { size: [8, 8], counts: "96200L00000004007" },
  components: [
    {
      outer: [
        [1, 1],
        [1, 7],
        [7, 7],
        [7, 1],
      ],
      holes: [
        [
          [3, 3],
          [5, 3],
          [5, 5],
          [3, 5],
        ],
      ],
    },
  ],
  controls: [
    {
      outer: [
        [1, 1],
        [1, 7],
        [7, 7],
        [7, 1],
      ],
      holes: [
        [
          [3, 3],
          [5, 3],
          [5, 5],
          [3, 5],
        ],
      ],
    },
  ],
  preview: "original-preview",
};
const filledGeometry: GeometryResult = {
  mask: { size: [8, 8], counts: "9620000000007" },
  components: [
    {
      outer: [
        [1, 1],
        [1, 7],
        [7, 7],
        [7, 1],
      ],
      holes: [],
    },
  ],
  controls: [
    {
      outer: [
        [1, 1],
        [1, 7],
        [7, 7],
        [7, 1],
      ],
      holes: [],
    },
  ],
  preview: "filled-preview",
};
const replacementGeometry: GeometryResult = {
  mask: { size: [8, 8], counts: "b04400000>" },
  components: [
    {
      outer: [
        [2, 2],
        [2, 6],
        [6, 6],
        [6, 2],
      ],
      holes: [],
    },
  ],
  controls: [
    {
      outer: [
        [2, 2],
        [2, 6],
        [6, 6],
        [6, 2],
      ],
      holes: [],
    },
  ],
  preview: "replacement-preview",
};
const target: Annotation = {
  id: "target",
  category_id: 2,
  iscrowd: 1,
  ...originalGeometry,
};
const other: Annotation = {
  id: "other",
  category_id: 1,
  iscrowd: 0,
  ...replacementGeometry,
};
const project: Project = {
  name: "Hole cleanup experiment",
  directory: "/tmp/hole-cleanup-project",
  image_root: "/tmp/hole-cleanup-project/images",
  categories: [
    { id: 1, name: "Other", color: "#397be6" },
    { id: 2, name: "Target", color: "#16a577" },
    { id: 3, name: "Reclassified", color: "#7856e8" },
  ],
  images: [1, 2].map((id) => ({
    id,
    file_name: `image-${id}.png`,
    width: 8,
    height: 8,
    annotation_count: id === 1 ? 2 : 0,
  })),
};
type FillResult = GeometryResult & {
  filled_holes: number;
  filled_pixels: number;
};
let disk: Map<number, ImageState>;
let fill: (body: {
  image_id: number;
  mask: Mask;
  max_area: number;
}) => Promise<FillResult>;
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
function calls(path: string) {
  return request.mock.calls.filter(([url]) => url === path);
}
function current() {
  if (!canvas.current) throw new Error("Canvas has not mounted");
  return canvas.current;
}
function selectedAnnotation() {
  return current().annotations.find((a) => a.id === "target")!;
}
function fillButton() {
  return screen.getByRole("button", {
    name: "Fill small holes",
  }) as HTMLButtonElement;
}
function threshold() {
  return screen.getByRole("spinbutton", {
    name: "Maximum hole area",
  }) as HTMLInputElement;
}
async function boot() {
  render(<App />);
  await screen.findByTestId("canvas");
  // Selection follows the actual instance row, not an internal callback.
  fireEvent.click(screen.getByRole("button", { name: /^Target\s*#01/ }));
}
async function save() {
  fireEvent.click(screen.getByRole("button", { name: /^Save$/ }));
  await waitFor(() =>
    expect(disk.get(current().image.id)?.annotations).toEqual(
      current().annotations,
    ),
  );
}
function changeClass(id: number) {
  fireEvent.change(screen.getByRole("combobox", { name: "Instance class" }), {
    target: { value: String(id) },
  });
}
function undo() {
  fireEvent.keyDown(document.body, { key: "z", ctrlKey: true });
}

beforeEach(() => {
  request.mockReset();
  canvas.current = null;
  disk = new Map([
    [
      1,
      {
        image_id: 1,
        revision: 0,
        annotations: structuredClone([target, other]),
        draft: null,
        proposals: [],
      },
    ],
    [
      2,
      { image_id: 2, revision: 0, annotations: [], draft: null, proposals: [] },
    ],
  ]);
  fill = async () => ({
    ...structuredClone(filledGeometry),
    filled_holes: 1,
    filled_pixels: 4,
  });
  request.mockImplementation((async (
    path: string,
    method = "GET",
    body?: unknown,
  ) => {
    if (path === "/project") return structuredClone(project);
    if (path === "/health") return { model: { state: "ready", device: "mps" } };
    const match = path.match(/^\/images\/(\d+)\/state$/);
    if (match) {
      const id = Number(match[1]);
      if (method === "PUT") {
        const saved = structuredClone(body as ImageState);
        saved.revision = (disk.get(id)?.revision ?? 0) + 1;
        disk.set(id, saved);
        return structuredClone(saved);
      }
      return structuredClone(disk.get(id));
    }
    if (path === "/masks/fill-holes")
      return fill(body as Parameters<typeof fill>[0]);
    if (path === "/geometry") return structuredClone(replacementGeometry);
    throw new Error(`Unexpected API call ${method} ${path}`);
  }) as typeof api);
});
afterEach(async () => {
  if (canvas.current && screen.queryByRole("button", { name: /^Save$/ }))
    await save();
  cleanup();
  canvas.current = null;
});

describe("small-hole cleanup", () => {
  it("updates only the selected instance and can save, undo, and redo its exact mask and geometry", async () => {
    await boot();
    expect(threshold().value).toBe("16");
    fireEvent.click(fillButton());
    await waitFor(() =>
      expect(selectedAnnotation().mask).toEqual(filledGeometry.mask),
    );
    expect(calls("/masks/fill-holes")[0][2]).toEqual({
      image_id: 1,
      mask: originalGeometry.mask,
      max_area: 16,
    });
    expect(selectedAnnotation()).toEqual({ ...target, ...filledGeometry });
    expect(current().annotations.find((a) => a.id === "other")).toEqual(other);
    expect(current().selected).toBe("target");
    await save();
    expect(disk.get(1)?.annotations).toEqual([
      { ...target, ...filledGeometry },
      other,
    ]);
    undo();
    expect(current().annotations).toEqual([target, other]);
    fireEvent.keyDown(document.body, {
      key: "z",
      ctrlKey: true,
      shiftKey: true,
    });
    expect(current().annotations).toEqual([
      { ...target, ...filledGeometry },
      other,
    ]);
  });

  it("uses the requested pixel-area threshold and preserves class edits made while cleanup is pending", async () => {
    const pending = deferred<FillResult>();
    fill = () => pending.promise;
    await boot();
    fireEvent.change(threshold(), { target: { value: "4" } });
    fireEvent.click(fillButton());
    expect(calls("/masks/fill-holes")[0][2]).toMatchObject({ max_area: 4 });
    changeClass(3);
    await act(async () => {
      pending.resolve({ ...filledGeometry, filled_holes: 1, filled_pixels: 4 });
      await pending.promise;
    });
    expect(selectedAnnotation()).toEqual({
      ...target,
      ...filledGeometry,
      category_id: 3,
    });
    expect(current().annotations.find((a) => a.id === "other")).toEqual(other);
  });

  it("does not add history or replace editable geometry when no holes were filled", async () => {
    const pending = deferred<FillResult>();
    fill = () => pending.promise;
    await boot();
    changeClass(3);
    fireEvent.click(fillButton());
    await act(async () => {
      pending.resolve({ ...filledGeometry, filled_holes: 0, filled_pixels: 0 });
      await pending.promise;
    });
    expect(selectedAnnotation()).toEqual({ ...target, category_id: 3 });
    undo();
    expect(current().annotations).toEqual([target, other]);
  });

  it("keeps the mask, editable geometry, and undo history intact when cleanup fails", async () => {
    fill = async () => {
      throw new ApiError(422, "Mask cleanup failed.");
    };
    await boot();
    changeClass(3);
    fireEvent.click(fillButton());
    await screen.findByText(/Mask cleanup failed\./);
    expect(selectedAnnotation()).toEqual({ ...target, category_id: 3 });
    expect(current().busy).toBe(false);
    undo();
    expect(current().annotations).toEqual([target, other]);
  });

  it("rejects invalid area limits without requesting cleanup", async () => {
    await boot();
    for (const value of ["", "0", "-1", "1.5", "65"]) {
      fireEvent.change(threshold(), { target: { value } });
      expect(fillButton().disabled).toBe(true);
      fireEvent.click(fillButton());
    }
    expect(calls("/masks/fill-holes")).toHaveLength(0);
    expect(current().annotations).toEqual([target, other]);
    fireEvent.change(threshold(), { target: { value: "1" } });
    expect(fillButton().disabled).toBe(false);
  });

  it("fills an enclosed background pixel even when separate foreground components have no interior rings", async () => {
    const separate: GeometryResult = {
      mask: { size: [8, 8], counts: "1160K050X1" },
      components: [
        {
          outer: [
            [2.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
          ],
          holes: [],
        },
        {
          outer: [
            [0.0, 2.0],
            [1.0, 2.0],
            [1.0, 1.0],
            [0.0, 1.0],
          ],
          holes: [],
        },
        {
          outer: [
            [3.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [3.0, 2.0],
          ],
          holes: [],
        },
        {
          outer: [
            [1.0, 3.0],
            [2.0, 3.0],
            [2.0, 2.0],
            [1.0, 2.0],
          ],
          holes: [],
        },
      ],
      controls: [
        {
          outer: [
            [2.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
          ],
          holes: [],
        },
        {
          outer: [
            [0.0, 2.0],
            [1.0, 2.0],
            [1.0, 1.0],
            [0.0, 1.0],
          ],
          holes: [],
        },
        {
          outer: [
            [3.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [3.0, 2.0],
          ],
          holes: [],
        },
        {
          outer: [
            [1.0, 3.0],
            [2.0, 3.0],
            [2.0, 2.0],
            [1.0, 2.0],
          ],
          holes: [],
        },
      ],
      preview: "four-parts-preview",
    };
    const filled: GeometryResult = {
      mask: { size: [8, 8], counts: "11620NX1" },
      components: [
        {
          outer: [
            [2.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [2.0, 3.0],
            [2.0, 2.0],
            [3.0, 2.0],
            [3.0, 1.0],
            [2.0, 1.0],
          ],
          holes: [],
        },
      ],
      controls: [
        {
          outer: [
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 3.0],
            [3.0, 2.0],
          ],
          holes: [],
        },
      ],
      preview: "filled-center-preview",
    };
    expect(separate.components).toHaveLength(4);
    expect(
      separate.components.every((component) => component.holes.length === 0),
    ).toBe(true);
    disk.get(1)!.annotations[0] = { ...target, ...separate };
    fill = async () => ({ ...filled, filled_holes: 1, filled_pixels: 1 });
    await boot();
    fireEvent.change(threshold(), { target: { value: "1" } });
    expect(fillButton().disabled).toBe(false);
    fireEvent.click(fillButton());
    await waitFor(() => expect(selectedAnnotation().mask).toEqual(filled.mask));
    expect(calls("/masks/fill-holes")[0][2]).toEqual({
      image_id: 1,
      mask: separate.mask,
      max_area: 1,
    });
    expect(selectedAnnotation()).toEqual({ ...target, ...filled });
    expect(
      current().annotations.find((annotation) => annotation.id === "other"),
    ).toEqual(other);
    undo();
    expect(selectedAnnotation()).toEqual({ ...target, ...separate });
  });

  it("blocks image navigation and undo until cleanup finishes", async () => {
    const pending = deferred<FillResult>();
    fill = () => pending.promise;
    await boot();
    changeClass(3);
    fireEvent.click(fillButton());
    expect(current().busy).toBe(true);
    expect(fillButton().disabled).toBe(true);
    fireEvent.click(screen.getByTitle("image-2.png"));
    undo();
    expect(current().image.id).toBe(1);
    expect(selectedAnnotation().category_id).toBe(3);
    await act(async () => {
      pending.resolve({ ...filledGeometry, filled_holes: 1, filled_pixels: 4 });
      await pending.promise;
    });
    expect(current().busy).toBe(false);
    fireEvent.click(screen.getByTitle("image-2.png"));
    await waitFor(() => expect(current().image.id).toBe(2));
    expect(current().annotations).toEqual([]);
    expect(disk.get(1)?.annotations).toEqual([
      { ...target, ...filledGeometry, category_id: 3 },
      other,
    ]);
  });

  it("discards a late cleanup result after newer geometry replaces the original mask", async () => {
    const pending = deferred<FillResult>();
    fill = () => pending.promise;
    await boot();
    fireEvent.click(fillButton());
    // Simulate a final drag callback already queued when the cleanup began.
    act(() => current().onGeometry("target", replacementGeometry.components));
    await waitFor(() =>
      expect(selectedAnnotation().mask).toEqual(replacementGeometry.mask),
    );
    await act(async () => {
      pending.resolve({ ...filledGeometry, filled_holes: 1, filled_pixels: 4 });
      await pending.promise;
    });
    expect(selectedAnnotation()).toEqual({ ...target, ...replacementGeometry });
    expect(current().annotations.find((a) => a.id === "other")).toEqual(other);
  });
});
