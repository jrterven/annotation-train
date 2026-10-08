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
  Annotation,
  Draft,
  GeometryResult,
  ImageState,
  Part,
  Project,
  Proposal,
  Tool,
  XY,
} from "../types";

type CanvasState = {
  image: { id: number };
  annotations: Annotation[];
  draft: Draft | null;
  tool: Tool;
  selected: string | null;
  onPolygon: (vertices: XY[], closed: boolean) => void;
  onPoint: (point: XY, negative: boolean) => void;
};
const canvas = vi.hoisted(() => ({ current: null as CanvasState | null }));
vi.mock("../CanvasEditor", () => ({
  // Test application orchestration with the actual persistence hook. Canvas
  // coordinate/gesture handling has separate tests against the real Konva tree.
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
const vertices: XY[] = [
  [1, 1],
  [6, 1],
  [6, 6],
  [1, 6],
];
const seed: GeometryResult = {
  mask: { size: [8, 8], counts: "9530000000?" },
  components: [{ outer: vertices, holes: [] }],
  preview: "seed-preview",
};
const prediction: GeometryResult = {
  mask: { size: [8, 8], counts: "a062000g0" },
  components: [
    {
      outer: [
        [2, 1],
        [5, 1],
        [5, 7],
        [2, 7],
      ],
      holes: [],
    },
  ],
  controls: [
    {
      outer: [
        [2, 1],
        [5, 1],
        [5, 7],
        [2, 7],
      ],
      holes: [],
    },
  ],
  preview: "prediction-preview",
};
const corrected: GeometryResult = {
  mask: { size: [8, 8], counts: "b044000f0" },
  components: [
    {
      outer: [
        [2, 2],
        [5, 2],
        [5, 6],
        [2, 6],
      ],
      holes: [],
    },
  ],
  controls: [
    {
      outer: [
        [2, 2],
        [5, 2],
        [5, 6],
        [2, 6],
      ],
      holes: [],
    },
  ],
  preview: "corrected-preview",
};
const project: Project = {
  name: "Polygon experiment",
  directory: "/tmp/polygon-project",
  image_root: "/tmp/polygon-project/images",
  categories: [{ id: 1, name: "Región", color: "#75a88d" }],
  images: [1, 2].map((id) => ({
    id,
    file_name: `image-${id}.png`,
    width: 8,
    height: 8,
    annotation_count: 0,
  })),
};
const blank = (id: number): ImageState => ({
  image_id: id,
  revision: 0,
  annotations: [],
  draft: null,
  proposals: [],
});
let disk: Map<number, ImageState>;
let geometry: () => Promise<GeometryResult>;
let inference: (body: {
  image_id: number;
  revision: number;
  part: Part;
}) => Promise<GeometryResult & { image_id: number; revision: number }>;
let lastPrediction: GeometryResult;
let modelState: string;
let textInference: (body: {
  image_id: number;
  revision: number;
  text: string;
  source_language: "en" | "es";
}) => Promise<{
  image_id: number;
  revision: number;
  proposals: Proposal[];
  prompt: { original: string; english: string; source_language: "en" | "es" };
}>;
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
function activePart() {
  const draft = current().draft;
  return draft?.parts.find((part) => part.id === draft.active_part_id);
}
async function boot() {
  const view = render(<App />);
  await screen.findByTestId("canvas");
  return view;
}
async function bootHosted() {
  modelState = "unavailable";
  const view = render(
    <App
      hostedSession={{
        user: { id: "owner", name: "Owner", email: "owner@example.test" },
        csrf_token: "csrf",
        usage: null,
      }}
    />,
  );
  fireEvent.click(
    await screen.findByRole("button", { name: "Open Polygon experiment" }),
  );
  await waitFor(() =>
    expect(screen.queryByRole("dialog", { name: "Your projects" })).toBeNull(),
  );
  await screen.findByTestId("canvas");
  return view;
}
function polygon(points = vertices, closed = false) {
  act(() => current().onPolygon(structuredClone(points), closed));
}
function enter() {
  fireEvent.keyDown(document.body, { key: "Enter" });
}
async function refine() {
  fireEvent.click(
    screen.getAllByRole("button", { name: /^Refine with SAM$/ })[0],
  );
  await waitFor(() => expect(activePart()?.mask).toEqual(prediction.mask));
}
async function save() {
  fireEvent.click(screen.getByRole("button", { name: /^Save$/ }));
  await waitFor(() => {
    const expected = current().draft;
    expect(disk.get(current().image.id)?.draft).toEqual(expected);
    expect(disk.get(current().image.id)?.annotations).toEqual(
      current().annotations,
    );
  });
}

beforeEach(() => {
  request.mockReset();
  canvas.current = null;
  disk = new Map([
    [1, blank(1)],
    [2, blank(2)],
  ]);
  lastPrediction = prediction;
  modelState = "ready";
  textInference = async (body) => ({
    image_id: body.image_id,
    revision: body.revision,
    proposals: [
      {
        ...prediction,
        id: "text-proposal",
        category_id: 1,
        iscrowd: 0,
        selected: false,
        score: 0.9,
      },
    ],
    prompt: {
      original: body.text,
      english: body.source_language === "es" ? "carrots" : body.text,
      source_language: body.source_language,
    },
  });
  geometry = async () => structuredClone(seed);
  inference = async (body) => ({
    ...structuredClone(lastPrediction),
    image_id: body.image_id,
    revision: body.revision,
  });
  request.mockImplementation((async (
    path: string,
    method = "GET",
    body?: unknown,
  ) => {
    if (path === "/project") return structuredClone(project);
    if (path === "/health")
      return { model: { state: modelState, device: "mps" } };
    if (path === "/projects")
      return { projects: [{ id: "hosted-project", name: project.name }] };
    if (path === "/projects/hosted-project")
      return { ...structuredClone(project), id: "hosted-project" };
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
    if (path === "/infer/text")
      return textInference(body as Parameters<typeof textInference>[0]);
    if (path === "/geometry") return geometry();
    if (path === "/infer/points")
      return inference(body as Parameters<typeof inference>[0]);
    if (path === "/masks/union") return structuredClone(lastPrediction);
    throw new Error(`Unexpected API call ${method} ${path}`);
  }) as typeof api);
});
afterEach(async () => {
  // Flush the real hook's debounce before unmount so one test's pending save
  // cannot reach another test's mocked project.
  if (canvas.current && screen.queryByRole("button", { name: /^Save$/ })) {
    await save();
  }
  cleanup();
  canvas.current = null;
});

describe("polygon to SAM application flow", () => {
  it.each(["ready", "running"])(
    "segments a new draft after discarding a %s draft with the trash button",
    async (stage) => {
      const confirmed = {
        ...prediction,
        id: "kept",
        category_id: 1,
        iscrowd: 0,
      };
      disk.get(1)!.annotations = [confirmed];
      const first = deferred<
        GeometryResult & { image_id: number; revision: number }
      >();
      const second = deferred<
        GeometryResult & { image_id: number; revision: number }
      >();
      let count = 0;
      inference = () => (++count === 1 ? first.promise : second.promise);
      await bootHosted();
      act(() => current().onPoint([3, 3], false));
      const oldDraft = current().draft!;
      const oldSignal = calls("/infer/points")[0][3];
      if (stage === "ready") {
        await act(async () =>
          first.resolve({ ...prediction, image_id: 1, revision: 0 }),
        );
        expect(activePart()?.mask).toEqual(prediction.mask);
      }
      fireEvent.click(screen.getByRole("button", { name: "Discard draft" }));
      expect(current().draft).toBeNull();
      if (stage === "running") expect(oldSignal?.aborted).toBe(true);
      // Autosave can advance the annotation revision while a new prompt runs.
      await save();
      fireEvent.click(
        screen.getByRole("button", { name: "Positive point · P" }),
      );
      act(() => current().onPoint([2, 3], false));
      expect(current().draft?.id).not.toBe(oldDraft.id);
      expect(activePart()?.id).not.toBe(oldDraft.active_part_id);
      await save();
      if (stage === "running") {
        await act(async () =>
          first.resolve({ ...prediction, image_id: 1, revision: 0 }),
        );
        expect(activePart()?.mask).toBeUndefined();
      }
      const { preview: _preview, ...withoutPreview } = corrected;
      await act(async () =>
        second.resolve({ ...withoutPreview, image_id: 1, revision: 1 }),
      );
      expect(activePart()?.mask).toEqual(corrected.mask);
      expect(activePart()?.points).toEqual([{ x: 2, y: 3, label: 1 }]);
      expect(
        (screen.getByRole("button", { name: "Confirm ↵" }) as HTMLButtonElement)
          .disabled,
      ).toBe(false);
      expect(current().annotations).toEqual([confirmed]);
    },
  );

  it("keeps a segmentation failure visible and clears it when the part is retried", async () => {
    inference = async () => {
      throw new ApiError(503, "SAM unavailable");
    };
    await bootHosted();
    act(() => current().onPoint([3, 3], false));
    await screen.findByRole("button", { name: /Part 1.*Failed/ });
    expect(screen.getByRole("alert").textContent).toContain("SAM unavailable");
    fireEvent.click(
      screen.getByRole("button", { name: "Dismiss notification" }),
    );
    expect(screen.getByRole("alert").textContent).toContain(
      "Retry segmentation",
    );
    inference = async (body) => ({
      ...prediction,
      image_id: body.image_id,
      revision: body.revision,
    });
    fireEvent.click(screen.getByRole("button", { name: "Retry segmentation" }));
    await waitFor(() => expect(activePart()?.mask).toEqual(prediction.mask));
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByRole("button", { name: /Part 1.*Ready/ })).toBeTruthy();
  });

  it("only infers on request, preserves the polygon seed through corrections, and confirms the final mask", async () => {
    await boot();
    fireEvent.keyDown(document.body, { key: "g" });
    expect(current().tool).toBe("polygon");
    polygon();
    expect(activePart()?.polygon).toEqual({ vertices, closed: false });
    expect(calls("/geometry")).toHaveLength(0);
    expect(calls("/infer/points")).toHaveLength(0);
    enter();
    expect(activePart()?.polygon?.closed).toBe(true);
    expect(calls("/geometry")).toHaveLength(0);
    enter();
    expect(calls("/masks/union")).toHaveLength(0);
    await refine();
    expect(calls("/geometry")[0][2]).toEqual({
      image_id: 1,
      components: [{ outer: vertices, holes: [] }],
    });
    const initialPrompt = calls("/infer/points")[0][2] as { part: Part };
    expect(initialPrompt.part.seed_mask).toEqual(seed.mask);
    expect(initialPrompt.part.points).toEqual([]);
    expect(initialPrompt.part.box).toBeUndefined();
    expect(current().tool).toBe("positive");
    lastPrediction = corrected;
    act(() => current().onPoint([2, 1], true));
    await waitFor(() => expect(activePart()?.mask).toEqual(corrected.mask));
    const correction = calls("/infer/points")[1][2] as { part: Part };
    expect(correction.part.seed_mask).toEqual(seed.mask);
    expect(correction.part.polygon).toEqual({ vertices, closed: true });
    expect(correction.part.points).toEqual([{ x: 2, y: 1, label: 0 }]);
    act(() => current().onPoint([3, 3], false));
    await waitFor(() => expect(activePart()?.mask).toEqual(corrected.mask));
    const positiveCorrection = calls("/infer/points")[2][2] as { part: Part };
    expect(positiveCorrection.part.seed_mask).toEqual(seed.mask);
    expect(positiveCorrection.part.points).toEqual([
      { x: 2, y: 1, label: 0 },
      { x: 3, y: 3, label: 1 },
    ]);
    expect(calls("/geometry")).toHaveLength(1);
    enter();
    await waitFor(() => expect(current().annotations).toHaveLength(1));
    expect(current().draft).toBeNull();
    expect(current().annotations[0].mask).toEqual(corrected.mask);
    expect(current().annotations[0].controls).toEqual(corrected.controls);
    expect(current().selected).toBe(current().annotations[0].id);
    expect(current().tool).toBe("select");
    expect(calls("/masks/union")[0][2]).toEqual({
      image_id: 1,
      masks: [corrected.mask],
    });
  });

  it("saves an unfinished polygon before navigation and restores it after reopening", async () => {
    const view = await boot();
    polygon(vertices.slice(0, 2));
    fireEvent.click(screen.getByTitle("image-2.png"));
    await waitFor(() => expect(current().image.id).toBe(2));
    expect(disk.get(1)?.draft?.parts[0].polygon).toEqual({
      vertices: vertices.slice(0, 2),
      closed: false,
    });
    expect(current().draft).toBeNull();
    fireEvent.click(screen.getByTitle("image-1.png"));
    await waitFor(() => expect(current().image.id).toBe(1));
    expect(activePart()?.polygon?.vertices).toEqual(vertices.slice(0, 2));
    view.unmount();
    await boot();
    expect(activePart()?.polygon).toEqual({
      vertices: vertices.slice(0, 2),
      closed: false,
    });
    fireEvent.click(screen.getByRole("button", { name: "Resume draft" }));
    expect(current().tool).toBe("polygon");
    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(current().tool).toBe("select");
    fireEvent.keyDown(document.body, { key: "n" });
    expect(current().tool).toBe("polygon");
    expect(activePart()?.polygon?.vertices).toEqual(vertices.slice(0, 2));
    expect(calls("/geometry")).toHaveLength(0);
    expect(calls("/infer/points")).toHaveLength(0);
  });

  it("keeps a rejected polygon editable and allows a corrected retry", async () => {
    await boot();
    const crossing: XY[] = [
      [1, 1],
      [6, 6],
      [1, 6],
      [6, 1],
    ];
    polygon(crossing, true);
    geometry = async () => {
      throw new ApiError(422, "The polygon intersects itself.");
    };
    fireEvent.click(
      screen.getAllByRole("button", { name: /^Refine with SAM$/ })[0],
    );
    expect((await screen.findByRole("alert")).textContent).toContain(
      "The polygon intersects itself",
    );
    expect(activePart()?.polygon).toEqual({ vertices: crossing, closed: true });
    expect(activePart()?.seed_mask).toBeUndefined();
    expect(activePart()?.mask).toBeUndefined();
    expect(calls("/infer/points")).toHaveLength(0);
    fireEvent.click(
      screen.getAllByRole("button", { name: /^Edit polygon$/ })[0],
    );
    expect(activePart()?.polygon?.closed).toBe(false);
    polygon(vertices);
    enter();
    geometry = async () => structuredClone(seed);
    await refine();
    expect(activePart()?.polygon).toEqual({ vertices, closed: true });
  });

  it("discards polygon rasterization that finishes after undo", async () => {
    const pending = deferred<GeometryResult>();
    geometry = () => pending.promise;
    await boot();
    polygon();
    enter();
    fireEvent.click(
      screen.getAllByRole("button", { name: /^Refine with SAM$/ })[0],
    );
    expect(calls("/geometry")).toHaveLength(1);
    fireEvent.keyDown(document.body, { key: "z", ctrlKey: true });
    expect(activePart()?.polygon?.closed).toBe(false);
    await act(async () => {
      pending.resolve(seed);
      await pending.promise;
    });
    expect(activePart()?.seed_mask).toBeUndefined();
    expect(activePart()?.mask).toBeUndefined();
    expect(calls("/infer/points")).toHaveLength(0);
  });

  it("discards a SAM result after undo invalidates its polygon", async () => {
    const pending = deferred<
      GeometryResult & { image_id: number; revision: number }
    >();
    inference = () => pending.promise;
    await boot();
    polygon();
    enter();
    fireEvent.click(
      screen.getAllByRole("button", { name: /^Refine with SAM$/ })[0],
    );
    await waitFor(() => expect(calls("/infer/points")).toHaveLength(1));
    const signal = calls("/infer/points")[0][3];
    fireEvent.keyDown(document.body, { key: "z", ctrlKey: true });
    expect(signal?.aborted).toBe(true);
    expect(activePart()?.polygon?.closed).toBe(false);
    await act(async () => {
      pending.resolve({ ...prediction, image_id: 1, revision: 0 });
      await pending.promise;
    });
    expect(activePart()?.mask).toBeUndefined();
    expect(activePart()?.seed_mask).toBeUndefined();
    expect(current().annotations).toHaveLength(0);
  });

  it("routes late inference to its original image while another image is active", async () => {
    const pending = deferred<
      GeometryResult & { image_id: number; revision: number }
    >();
    inference = () => pending.promise;
    await boot();
    polygon(vertices, true);
    fireEvent.click(
      screen.getAllByRole("button", { name: /^Refine with SAM$/ })[0],
    );
    await waitFor(() => expect(calls("/infer/points")).toHaveLength(1));
    fireEvent.click(screen.getByTitle("image-2.png"));
    await waitFor(() => expect(current().image.id).toBe(2));
    await act(async () => {
      pending.resolve({ ...prediction, image_id: 1, revision: 0 });
      await pending.promise;
    });
    expect(current().image.id).toBe(2);
    expect(current().draft).toBeNull();
    expect(current().annotations).toHaveLength(0);
    fireEvent.click(screen.getByTitle("image-1.png"));
    await waitFor(() => expect(current().image.id).toBe(1));
    expect(activePart()?.mask).toEqual(prediction.mask);
    expect(activePart()?.polygon).toEqual({ vertices, closed: true });
    expect(disk.get(2)?.draft).toBeNull();
  });
});

describe("annotation toolbar", () => {
  it("submits the selected prompt language outside the canvas and keeps translations with their image", async () => {
    await boot();
    const form = screen.getByRole("form", { name: "Concept segmentation" });
    expect(form.closest(".editor-toolbar")).not.toBeNull();
    expect(form.closest(".canvas-area")).toBeNull();
    const language = screen.getByRole("combobox", { name: "Prompt language" });
    expect((language as HTMLSelectElement).value).toBe("en");
    expect(
      screen.getByPlaceholderText("Object to segment (in English)"),
    ).not.toBeNull();
    fireEvent.change(language, { target: { value: "es" } });
    fireEvent.change(
      screen.getByPlaceholderText("Object to segment (in Spanish)"),
      { target: { value: "zanahorias" } },
    );
    fireEvent.submit(form);
    await waitFor(() => expect(calls("/infer/text")).toHaveLength(1));
    expect(calls("/infer/text")[0][2]).toMatchObject({
      text: "zanahorias",
      source_language: "es",
      image_id: 1,
    });
    await screen.findByText("carrots");
    expect(screen.getByText("carrots").closest("p")?.title).toBe(
      "Original: zanahorias",
    );
    fireEvent.click(screen.getByTitle("image-2.png"));
    await waitFor(() => expect(current().image.id).toBe(2));
    expect(screen.queryByText("carrots")).toBeNull();
    fireEvent.click(screen.getByTitle("image-1.png"));
    await waitFor(() => expect(current().image.id).toBe(1));
    expect(screen.getByText("carrots")).not.toBeNull();
  });

  it("explains an empty translated search and associates tooltips with annotation controls", async () => {
    textInference = async (body) => ({
      image_id: body.image_id,
      revision: body.revision,
      proposals: [],
      prompt: {
        original: body.text,
        english: "carrots",
        source_language: "es",
      },
    });
    await boot();
    for (const name of [
      "Select and edit · V",
      "Positive point · P",
      "Negative point · E",
      "Bounding box · B",
      "Polygon · G",
    ]) {
      const button = screen.getByRole("button", { name });
      const tooltip = document.getElementById(
        button.getAttribute("aria-describedby")!,
      );
      expect(tooltip?.getAttribute("role")).toBe("tooltip");
      expect(tooltip?.textContent).toBe(name);
    }
    fireEvent.change(
      screen.getByRole("combobox", { name: "Prompt language" }),
      { target: { value: "es" } },
    );
    fireEvent.change(
      screen.getByRole("textbox", { name: "Object to segment" }),
      { target: { value: "zanahorias" } },
    );
    fireEvent.submit(
      screen.getByRole("form", { name: "Concept segmentation" }),
    );
    await screen.findByText(
      "No objects found for “carrots”. Try a different description.",
    );
  });
});

describe("hosted manual polygons without a GPU", () => {
  it("rasterizes, confirms, saves and reopens a polygon without any inference request", async () => {
    const view = await bootHosted();
    polygon(vertices, true);
    expect(
      (
        screen.getByRole("button", {
          name: "Refine with SAM",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Use polygon" }));
    await waitFor(() => expect(activePart()?.mask).toEqual(seed.mask));
    expect(activePart()?.seed_mask).toEqual(seed.mask);
    expect(calls("/geometry")[0][2]).toEqual({
      image_id: 1,
      components: [{ outer: vertices, holes: [] }],
    });
    lastPrediction = seed;
    fireEvent.click(screen.getByRole("button", { name: /Confirm/ }));
    await waitFor(() => expect(current().annotations).toHaveLength(1));
    expect(current().annotations[0].mask).toEqual(seed.mask);
    expect(current().draft).toBeNull();
    expect(calls("/masks/union")[0][2]).toEqual({
      image_id: 1,
      masks: [seed.mask],
    });
    expect(
      request.mock.calls.some(([path]) => path.startsWith("/infer/")),
    ).toBe(false);
    await save();
    view.unmount();
    await bootHosted();
    expect(current().annotations[0].mask).toEqual(seed.mask);
    expect(current().draft).toBeNull();
  });

  it("preserves an invalid polygon for correction and ignores a discarded CPU result", async () => {
    await bootHosted();
    polygon(vertices, true);
    geometry = async () => {
      throw new ApiError(422, "The polygon intersects itself.");
    };
    fireEvent.click(screen.getByRole("button", { name: "Use polygon" }));
    await screen.findByText("The polygon intersects itself.");
    expect(activePart()?.polygon).toEqual({ vertices, closed: true });
    expect(activePart()?.mask).toBeUndefined();
    expect(
      (screen.getByRole("button", { name: /Confirm/ }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    const pending = deferred<GeometryResult>();
    geometry = () => pending.promise;
    fireEvent.click(screen.getByRole("button", { name: "Use polygon" }));
    fireEvent.click(screen.getByRole("button", { name: "Discard draft" }));
    await act(async () => {
      pending.resolve(seed);
    });
    expect(current().draft).toBeNull();
    expect(current().annotations).toHaveLength(0);
    expect(
      request.mock.calls.some(([path]) => path.startsWith("/infer/")),
    ).toBe(false);
  });
});
