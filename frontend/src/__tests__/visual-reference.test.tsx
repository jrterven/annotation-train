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
import VisualReference, {
  loadVisualFile,
  referencePoint,
} from "../VisualReference";
import { api } from "../api";
import type {
  SegmentationAnnotation as Annotation,
  Draft,
  ImageState,
  Project,
  Proposal,
  Tool,
} from "../types";

const canvas = vi.hoisted(() => ({
  current: null as null | {
    image: { id: number };
    proposals: Proposal[];
    annotations: Annotation[];
    draft: Draft | null;
    tool: Tool;
  },
}));
vi.mock("../CanvasEditor", () => ({
  default: (props: NonNullable<typeof canvas.current>) => {
    canvas.current = props;
    return <div data-testid="canvas" />;
  },
}));
vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api")>()),
  api: vi.fn(),
}));
vi.mock("../ProjectDialog", () => ({
  FileBrowser: () => null,
  ProjectDialog: ({ onOpen }: { onOpen: (project: Project) => void }) => (
    <button
      onClick={() => {
        disk = new Map([
          [1, blank(1)],
          [2, blank(2)],
        ]);
        onOpen({
          ...project,
          name: "Other project",
          directory: "/tmp/other-project",
        });
      }}
    >
      Open another project
    </button>
  ),
}));
const request = vi.mocked(api);
const pngBase64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6WEAAAAAASUVORK5CYII=";
const makeFile = (name = "example.png") =>
  new File([Uint8Array.from(atob(pngBase64), (c) => c.charCodeAt(0))], name, {
    type: "image/png",
  });
let dimensions: { width: number; height: number };
let decodeAutomatically: boolean;
const decoders: {
  onload: (() => void) | null;
  onerror: (() => void) | null;
}[] = [];
const project: Project = {
  name: "Visual test",
  directory: "/tmp/visual-project",
  image_root: "/tmp/images",
  categories: [{ id: 1, name: "Object", color: "#119ca8" }],
  images: [1, 2].map((id) => ({
    id,
    file_name: `image-${id}.png`,
    width: 8,
    height: 8,
    annotation_count: 0,
  })),
};
const prediction = {
  mask: { size: [8, 8] as [number, number], counts: "a062000g0" },
  components: [
    {
      outer: [
        [2, 1],
        [5, 1],
        [5, 7],
        [2, 7],
      ] as [number, number][],
      holes: [],
    },
  ],
  preview: "data:image/png;base64,",
};
const proposal = (id = "visual-proposal"): Proposal => ({
  ...prediction,
  id,
  category_id: 1,
  iscrowd: 0,
  score: 0.9,
  selected: false,
});
const blank = (id: number): ImageState => ({
  image_id: id,
  revision: 0,
  annotations: [],
  draft: null,
  proposals: [],
});
type SearchBody = {
  image_id: number;
  revision: number;
  category_id: number;
  text: string;
  source_language: "en" | "es";
  reference_image?: string;
  reference_box?: number[];
};
type SearchResult = {
  image_id: number;
  revision: number;
  proposals: Proposal[];
  prompt?: { original: string; english: string; source_language: "en" | "es" };
};
let disk: Map<number, ImageState>;
let visualSearch: (body: SearchBody) => Promise<SearchResult>;
const calls = (path: string) =>
  request.mock.calls.filter(([url]) => url === path);
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => (resolve = done));
  return { promise, resolve };
}
async function boot() {
  render(<App />);
  await screen.findByTestId("canvas");
}
async function chooseFile(file = makeFile()) {
  fireEvent.change(screen.getByLabelText("Visual example file"), {
    target: { files: [file] },
  });
  // FileReader and image decoding are asynchronous; allow headroom while the
  // full suite shares the machine with local model inference.
  await screen.findByAltText("Visual example preview", {}, { timeout: 3000 });
}
async function useFile(file = makeFile()) {
  await chooseFile(file);
  cropExample();
  fireEvent.click(screen.getByRole("button", { name: "Use example" }));
}
function cropExample() {
  const image = screen.getByAltText("Visual example preview");
  vi.spyOn(image, "getBoundingClientRect").mockReturnValue({
    left: 0,
    top: 0,
    width: 600,
    height: 400,
  } as DOMRect);
  const surface = image.parentElement!;
  fireEvent.pointerDown(surface, {
    button: 0,
    pointerId: 1,
    clientX: 60,
    clientY: 40,
  });
  fireEvent.pointerUp(surface, { pointerId: 1, clientX: 540, clientY: 360 });
}
function search() {
  fireEvent.submit(screen.getByRole("form", { name: "Concept segmentation" }));
}

beforeEach(() => {
  dimensions = { width: 1200, height: 800 };
  decodeAutomatically = true;
  decoders.length = 0;
  canvas.current = null;
  request.mockReset();
  vi.stubGlobal(
    "Image",
    class {
      naturalWidth = dimensions.width;
      naturalHeight = dimensions.height;
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      constructor() {
        decoders.push(this);
      }
      set src(value: string) {
        if (value && decodeAutomatically) queueMicrotask(() => this.onload?.());
      }
    },
  );
  if (!window.PointerEvent) vi.stubGlobal("PointerEvent", MouseEvent);
  HTMLElement.prototype.setPointerCapture = vi.fn();
  HTMLElement.prototype.hasPointerCapture = () => false;
  HTMLElement.prototype.releasePointerCapture = vi.fn();
  disk = new Map([
    [1, blank(1)],
    [2, blank(2)],
  ]);
  visualSearch = async (body) => ({
    image_id: body.image_id,
    revision: body.revision,
    proposals: [proposal()],
    ...(body.text
      ? {
          prompt: {
            original: body.text,
            english: body.source_language === "es" ? "carrots" : body.text,
            source_language: body.source_language,
          },
        }
      : {}),
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
        return saved;
      }
      return structuredClone(disk.get(id));
    }
    if (path === "/infer/visual") return visualSearch(body as SearchBody);
    if (path === "/infer/text") {
      const input = body as SearchBody;
      return {
        image_id: input.image_id,
        revision: input.revision,
        proposals: [proposal("text-proposal")],
      };
    }
    if (path === "/masks/union") return prediction;
    throw new Error(`Unexpected API ${method} ${path}`);
  }) as typeof api);
});
afterEach(async () => {
  if (canvas.current && screen.queryByRole("button", { name: /^Save$/ })) {
    fireEvent.click(screen.getByRole("button", { name: /^Save$/ }));
    await waitFor(() =>
      expect(disk.get(canvas.current!.image.id)?.proposals).toEqual(
        canvas.current!.proposals,
      ),
    );
    await waitFor(() =>
      expect(disk.get(canvas.current!.image.id)?.annotations).toEqual(
        canvas.current!.annotations,
      ),
    );
  }
  cleanup();
  canvas.current = null;
  vi.unstubAllGlobals();
});

describe("visual example upload and crop", () => {
  it("reads original bytes and returns crop coordinates in the oriented image pixel grid", async () => {
    const onChange = vi.fn();
    render(
      <VisualReference
        value={null}
        scope="project:1"
        onChange={onChange}
        onPickStart={vi.fn()}
        onOpenChange={vi.fn()}
      />,
    );
    await chooseFile();
    expect(
      (screen.getByRole("button", { name: "Use example" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    const image = screen.getByAltText("Visual example preview");
    vi.spyOn(image, "getBoundingClientRect").mockReturnValue({
      left: 10,
      top: 20,
      width: 600,
      height: 400,
    } as DOMRect);
    const surface = image.parentElement!;
    fireEvent.pointerDown(surface, {
      button: 0,
      pointerId: 1,
      clientX: 460,
      clientY: 370,
    });
    fireEvent.pointerMove(surface, { pointerId: 1, clientX: 60, clientY: 70 });
    fireEvent.pointerUp(surface, { pointerId: 1, clientX: 60, clientY: 70 });
    expect(screen.getByRole("button", { name: "Reset crop" })).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Use example" }));
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "example.png",
        base64: pngBase64,
        width: 1200,
        height: 800,
        box: [100, 100, 900, 700],
      }),
    );
    expect(
      referencePoint(
        { left: 10, top: 20, width: 600, height: 400 },
        1200,
        800,
        -200,
        1000,
      ),
    ).toEqual([0, 800]);
  });
  it("rejects wrong formats, oversized uploads and decoded images over 16 megapixels", async () => {
    const signal = new AbortController().signal;
    await expect(
      loadVisualFile(
        new File(["gif"], "test.gif", { type: "image/gif" }),
        signal,
      ),
    ).rejects.toThrow("PNG, JPEG, or WebP");
    await expect(
      loadVisualFile(
        new File(["not an image"], "bad.png", { type: "image/png" }),
        signal,
      ),
    ).rejects.toThrow("not a PNG");
    await expect(
      loadVisualFile(
        new File([new Uint8Array(10 * 1024 * 1024 + 1)], "large.png", {
          type: "image/png",
        }),
        signal,
      ),
    ).rejects.toThrow("10 MiB");
    dimensions = { width: 5000, height: 4000 };
    await expect(loadVisualFile(makeFile(), signal)).rejects.toThrow(
      "16 megapixels",
    );
  });
  it("requires a new object box after Reset crop and displays bad-file errors without accepting a reference", async () => {
    const onChange = vi.fn();
    render(
      <VisualReference
        value={null}
        scope="project:1"
        onChange={onChange}
        onPickStart={vi.fn()}
        onOpenChange={vi.fn()}
      />,
    );
    await chooseFile();
    cropExample();
    expect(
      (screen.getByRole("button", { name: "Use example" }) as HTMLButtonElement)
        .disabled,
    ).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Reset crop" }));
    expect(
      (screen.getByRole("button", { name: "Use example" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    fireEvent.change(screen.getByLabelText("Visual example file"), {
      target: {
        files: [new File(["not an image"], "bad.png", { type: "image/png" })],
      },
    });
    expect((await screen.findByRole("alert")).textContent).toContain(
      "not a PNG",
    );
    expect(
      (screen.getByRole("button", { name: "Use example" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(onChange).not.toHaveBeenCalled();
  });
  it("discards an unfinished image decode when the project or image scope changes", async () => {
    decodeAutomatically = false;
    const onChange = vi.fn();
    const props = {
      value: null,
      onChange,
      onPickStart: vi.fn(),
      onOpenChange: vi.fn(),
    };
    const view = render(<VisualReference {...props} scope="project:1" />);
    fireEvent.change(screen.getByLabelText("Visual example file"), {
      target: { files: [makeFile()] },
    });
    await waitFor(() => expect(decoders).toHaveLength(1));
    view.rerender(<VisualReference {...props} scope="project:2" />);
    await act(async () => {
      decoders[0].onload?.();
    });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe("visual proposal orchestration", () => {
  it("runs with an example and blank text, keeps the reference through navigation, and accepts proposals normally", async () => {
    await boot();
    await useFile();
    expect(
      (
        screen.getByRole("button", {
          name: "Generate proposals with SAM 3",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(false);
    search();
    await waitFor(() => expect(canvas.current!.proposals).toHaveLength(1));
    expect(calls("/infer/visual")[0][2]).toMatchObject({
      image_id: 1,
      text: "",
      source_language: "en",
      reference_image: pngBase64,
    });
    expect(calls("/infer/text")).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "Select proposal 1" }));
    fireEvent.click(screen.getByRole("button", { name: "Accept selected" }));
    expect(canvas.current!.annotations).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "Next image · ↓" }));
    await waitFor(() => expect(canvas.current!.image.id).toBe(2));
    expect(
      screen.getByRole("button", { name: "Edit visual example" }),
    ).toBeDefined();
    search();
    await waitFor(() => expect(calls("/infer/visual")).toHaveLength(2));
    expect(calls("/infer/visual")[1][2]).toMatchObject({
      image_id: 2,
      reference_image: pngBase64,
    });
  });
  it("combines a cropped example with Spanish text and falls back to text-only after removal", async () => {
    await boot();
    await chooseFile();
    const image = screen.getByAltText("Visual example preview");
    vi.spyOn(image, "getBoundingClientRect").mockReturnValue({
      left: 0,
      top: 0,
      width: 600,
      height: 400,
    } as DOMRect);
    const surface = image.parentElement!;
    fireEvent.pointerDown(surface, {
      button: 0,
      pointerId: 1,
      clientX: 20,
      clientY: 30,
    });
    fireEvent.pointerUp(surface, { pointerId: 1, clientX: 300, clientY: 300 });
    fireEvent.click(screen.getByRole("button", { name: "Use example" }));
    fireEvent.change(
      screen.getByRole("combobox", { name: "Prompt language" }),
      { target: { value: "es" } },
    );
    fireEvent.change(
      screen.getByRole("textbox", { name: "Object to segment" }),
      { target: { value: "zanahorias" } },
    );
    search();
    await waitFor(() => expect(canvas.current!.proposals).toHaveLength(1));
    expect(calls("/infer/visual")[0][2]).toMatchObject({
      text: "zanahorias",
      source_language: "es",
      reference_box: [40, 60, 600, 600],
    });
    expect(screen.getByText("carrots")).toBeDefined();
    fireEvent.click(
      screen.getByRole("button", { name: "Remove visual example" }),
    );
    search();
    await waitFor(() => expect(calls("/infer/text")).toHaveLength(1));
    expect(calls("/infer/text")[0][2]).not.toHaveProperty("reference_image");
  });
  it("ignores a late visual response after the example is removed and a text search replaces it", async () => {
    const slow = deferred<SearchResult>();
    visualSearch = () => slow.promise;
    await boot();
    await useFile();
    search();
    await waitFor(() => expect(calls("/infer/visual")).toHaveLength(1));
    const signal = calls("/infer/visual")[0][3] as AbortSignal;
    fireEvent.click(
      screen.getByRole("button", { name: "Remove visual example" }),
    );
    expect(signal.aborted).toBe(true);
    fireEvent.change(
      screen.getByRole("textbox", { name: "Object to segment" }),
      { target: { value: "carrots" } },
    );
    search();
    await waitFor(() =>
      expect(canvas.current!.proposals[0]?.id).toBe("text-proposal"),
    );
    await act(async () => {
      slow.resolve({
        image_id: 1,
        revision: 0,
        proposals: [proposal("obsolete")],
      });
      await slow.promise;
    });
    expect(canvas.current!.proposals[0]?.id).toBe("text-proposal");
  });
  it("clears the reference on project switch and rejects a pending result even when image IDs match", async () => {
    const slow = deferred<SearchResult>();
    visualSearch = () => slow.promise;
    await boot();
    await useFile();
    search();
    await waitFor(() => expect(calls("/infer/visual")).toHaveLength(1));
    fireEvent.click(screen.getByRole("button", { name: "Visual test" }));
    fireEvent.click(
      await screen.findByRole("button", { name: "Open another project" }),
    );
    await screen.findByRole("button", { name: "Other project" });
    await waitFor(() => expect(canvas.current!.proposals).toHaveLength(0));
    expect(
      screen.queryByRole("button", { name: "Edit visual example" }),
    ).toBeNull();
    expect(
      (
        screen.getByRole("button", {
          name: "Generate proposals with SAM 3",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    await act(async () => {
      slow.resolve({
        image_id: 1,
        revision: 0,
        proposals: [proposal("wrong-project")],
      });
      await slow.promise;
    });
    expect(canvas.current!.proposals).toHaveLength(0);
  });
  it("isolates modal keyboard events from drafts and navigation and restores focus on close", async () => {
    disk.get(1)!.draft = {
      id: "draft",
      category_id: 1,
      active_part_id: "part",
      parts: [{ id: "part", points: [], ...prediction }],
    };
    await boot();
    await chooseFile();
    const dialog = screen.getByRole("dialog", { name: "Visual example" });
    expect(document.activeElement).toBe(dialog);
    for (const key of ["Enter", "ArrowDown", "g", "n"])
      fireEvent.keyDown(document.body, { key });
    expect(canvas.current!.image.id).toBe(1);
    expect(canvas.current!.tool).toBe("positive");
    expect(calls("/masks/union")).toHaveLength(0);
    screen.getByRole("button", { name: "Choose image" }).focus();
    fireEvent.keyDown(document.activeElement!, { key: "Tab" });
    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: "Close visual example" }),
    );
    fireEvent.keyDown(dialog, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: "Upload visual example" }),
    );
  });
});
