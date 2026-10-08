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
import { configureApi } from "../api";
import { maskContains } from "../masks";
import type { GeometryResult, ImageState, Project, Proposal } from "../types";
import fixtures from "./fixtures/merge-proposals.json";

const geometry = fixtures as unknown as Record<
  "first" | "second" | "other" | "merged",
  GeometryResult
>;
const canvas = vi.hoisted(() => ({
  current: null as
    | null
    | (ImageState & {
        image: { id: number };
        selected: string | null;
        busy: boolean;
      }),
}));
vi.mock("../CanvasEditor", () => ({
  default: (props: NonNullable<typeof canvas.current>) => {
    canvas.current = props;
    return <div data-testid="canvas" />;
  },
}));
vi.mock("../HostedProjects", () => ({
  HostedProjects: ({ onOpen }: { onOpen: (p: Project) => void }) => (
    <button
      onClick={() =>
        onOpen({
          ...project,
          id: "merge-project",
          directory: "",
          image_root: "",
        })
      }
    >
      Open fixture
    </button>
  ),
  HostedUploads: () => null,
}));
vi.mock("../VisualReference", () => ({
  default: ({ onChange }: { onChange: (v: unknown) => void }) => (
    <button
      type="button"
      onClick={() =>
        onChange({ base64: "synthetic-example", box: [0, 0, 16, 12] })
      }
    >
      Use fixture example
    </button>
  ),
}));

const project: Project = {
  name: "Merge test",
  directory: "/tmp/merge-project",
  image_root: "/tmp/merge-images",
  categories: [
    { id: 1, name: "Object", color: "#397be6" },
    { id: 2, name: "Combined", color: "#16a577" },
  ],
  images: [1, 2].map((id) => ({
    id,
    file_name: `image-${id}.png`,
    width: 16,
    height: 12,
    annotation_count: id === 1 ? 1 : 0,
  })),
};
const existing = {
  ...geometry.other,
  id: "existing",
  category_id: 1,
  iscrowd: 1,
};
const draft = {
  id: "draft",
  category_id: 1,
  active_part_id: "part",
  parts: [{ id: "part", points: [], ...geometry.other }],
};
const proposals = (): Proposal[] =>
  ["first", "second", "other"].map((name) => ({
    ...geometry[name as "first" | "second" | "other"],
    id: name,
    category_id: 1,
    iscrowd: 0,
    score: 0.9,
    selected: false,
  }));
const session = {
  user: { id: "owner", email: "owner@example.test", name: "Owner" },
  csrf_token: "test-csrf",
  usage: null,
};
const fetchMock = vi.fn();
let disk: Map<number, ImageState>;
let merge: () => Promise<GeometryResult>;
let failSave: boolean;
const response = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status });
const calls = (suffix: string) =>
  fetchMock.mock.calls.filter(([url]) => (url as string).endsWith(suffix));
const current = () => canvas.current!;
const mergeButton = () =>
  screen.getByRole("button", { name: "Merge selected" }) as HTMLButtonElement;
const selectTwo = () => {
  fireEvent.click(screen.getByRole("button", { name: "Select proposal 1" }));
  fireEvent.click(screen.getByRole("button", { name: "Select proposal 2" }));
};
const undo = () =>
  fireEvent.keyDown(document.body, { key: "z", ctrlKey: true });
async function boot(mode: "local" | "hosted" = "local") {
  configureApi(mode, mode === "hosted" ? session.csrf_token : null);
  const view = render(
    <App hostedSession={mode === "hosted" ? session : undefined} />,
  );
  if (mode === "hosted")
    fireEvent.click(
      await screen.findByRole("button", { name: "Open fixture" }),
    );
  await screen.findByTestId("canvas");
  return view;
}
async function save() {
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => {
    const saved = disk.get(current().image.id)!;
    expect(saved.annotations).toEqual(current().annotations);
    expect(saved.proposals).toEqual(current().proposals);
    expect(saved.draft).toEqual(current().draft);
  });
}

beforeEach(() => {
  canvas.current = null;
  failSave = false;
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
  disk = new Map([
    [
      1,
      {
        image_id: 1,
        revision: 0,
        annotations: [existing],
        draft,
        proposals: proposals(),
      },
    ],
    [
      2,
      { image_id: 2, revision: 0, annotations: [], draft: null, proposals: [] },
    ],
  ]);
  merge = async () => structuredClone(geometry.merged);
  fetchMock.mockImplementation(async (url: string, options: RequestInit) => {
    const path = url.replace(/^\/api(?:\/v1\/projects\/merge-project)?/, "");
    const body = options.body ? JSON.parse(options.body as string) : undefined;
    if (url.endsWith("/health")) return response({ model: { state: "ready" } });
    if (path === "/project") return response(project);
    const state = path.match(/^\/images\/(\d+)\/state$/);
    if (state) {
      const id = Number(state[1]);
      if (options.method === "PUT") {
        if (failSave)
          return response({ detail: "Temporary save failure" }, 503);
        if (disk.get(id)!.revision !== body.revision)
          return response({ detail: "Revision conflict" }, 409);
        disk.set(id, { ...body, revision: body.revision + 1 });
      }
      return response(disk.get(id));
    }
    if (path === "/masks/union") return response(await merge());
    if (path.startsWith("/infer/"))
      return response({
        image_id: body.image_id,
        revision: body.revision,
        proposals: proposals(),
      });
    throw new Error(`Unexpected API ${options.method} ${url}`);
  });
});
afterEach(async () => {
  failSave = false;
  if (
    canvas.current &&
    !canvas.current.busy &&
    screen.queryByRole("button", { name: "Save" })
  )
    await save();
  cleanup();
  configureApi("local");
  vi.unstubAllGlobals();
});

describe.each(["local", "hosted"] as const)("%s proposal merging", (mode) => {
  it.each(["text", "visual"] as const)(
    "merges %s proposals into one editable instance, with undo/redo and reopening",
    async (source) => {
      const view = await boot(mode);
      if (source === "visual")
        fireEvent.click(
          screen.getByRole("button", { name: "Use fixture example" }),
        );
      else
        fireEvent.change(
          screen.getByRole("textbox", { name: "Object to segment" }),
          { target: { value: "object" } },
        );
      fireEvent.submit(
        screen.getByRole("form", { name: "Concept segmentation" }),
      );
      await waitFor(() => expect(calls(`/infer/${source}`)).toHaveLength(1));
      await waitFor(() =>
        expect(
          (
            screen.getByRole("button", {
              name: "Select proposal 1",
            }) as HTMLButtonElement
          ).disabled,
        ).toBe(false),
      );
      selectTwo();
      fireEvent.change(screen.getByRole("combobox", { name: "Active class" }), {
        target: { value: "2" },
      });
      await act(async () => fireEvent.click(mergeButton()));
      await waitFor(() => {
        expect(current().annotations).toHaveLength(2);
        expect(current().busy).toBe(false);
      });
      const merged = current().annotations[1];
      expect(merged).toEqual({
        ...geometry.merged,
        id: expect.any(String),
        category_id: 2,
        iscrowd: 0,
      });
      expect(["first", "second", "other"]).not.toContain(merged.id);
      expect(current().selected).toBe(merged.id);
      expect(current().annotations[0]).toEqual(existing);
      expect(current().draft).toEqual(draft);
      expect(current().proposals).toEqual([proposals()[2]]);
      expect(merged.components).toHaveLength(2);
      expect(merged.components.some((c) => c.holes.length > 0)).toBe(true);
      for (let y = 0; y < 12; y++)
        for (let x = 0; x < 16; x++)
          expect(maskContains(merged.mask, x, y)).toBe(
            maskContains(geometry.first.mask, x, y) ||
              maskContains(geometry.second.mask, x, y),
          );
      const [url, options] = calls("/masks/union")[0];
      expect(url).toBe(
        mode === "hosted"
          ? "/api/v1/projects/merge-project/masks/union"
          : "/api/masks/union",
      );
      expect(JSON.parse(options.body)).toEqual({
        image_id: 1,
        masks: [geometry.first.mask, geometry.second.mask],
      });
      if (mode === "hosted")
        expect(options.headers["X-CSRF-Token"]).toBe("test-csrf");
      expect(
        fetchMock.mock.calls.filter(([path]) => path.includes("/infer/")),
      ).toHaveLength(1);
      await act(async () => undo());
      await waitFor(() => expect(current().annotations).toEqual([existing]));
      expect(
        current()
          .proposals.filter((p) => p.selected)
          .map((p) => p.id),
      ).toEqual(["first", "second"]);
      fireEvent.keyDown(document.body, {
        key: "z",
        ctrlKey: true,
        shiftKey: true,
      });
      expect(current().annotations[1]).toEqual(merged);
      await waitFor(() => expect(disk.get(1)!.annotations[1]).toEqual(merged));
      fireEvent.click(screen.getByTitle("image-2.png"));
      await waitFor(() => expect(current().image.id).toBe(2));
      fireEvent.click(screen.getByTitle("image-1.png"));
      await waitFor(() => expect(current().image.id).toBe(1));
      expect(current().annotations[1]).toEqual(merged);
      await save();
      view.unmount();
      await boot(mode);
      expect(current().annotations[1]).toEqual(merged);
      expect(current().proposals).toEqual([proposals()[2]]);
    },
  );
});

it("requires two selections, while Accept selected still accepts them separately", async () => {
  await boot();
  expect(mergeButton().disabled).toBe(true);
  fireEvent.click(screen.getByRole("button", { name: "Select proposal 1" }));
  expect(mergeButton().disabled).toBe(true);
  fireEvent.click(screen.getByRole("button", { name: "Select proposal 2" }));
  expect(mergeButton().disabled).toBe(false);
  fireEvent.click(screen.getByRole("button", { name: "Accept selected" }));
  expect(
    current()
      .annotations.slice(1)
      .map((a) => a.mask),
  ).toEqual([geometry.first.mask, geometry.second.mask]);
  expect(calls("/masks/union")).toHaveLength(0);
});

it("preserves proposals and history on a failed merge and permits retry", async () => {
  await boot();
  selectTwo();
  const before = structuredClone(current().proposals);
  merge = async () => {
    throw new Error("Merge unavailable");
  };
  fireEvent.click(mergeButton());
  await screen.findByText("Merge unavailable");
  expect(current().annotations).toEqual([existing]);
  expect(current().proposals).toEqual(before);
  expect(current().draft).toEqual(draft);
  expect(
    (screen.getByRole("button", { name: /Undo ·/ }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
  expect(mergeButton().disabled).toBe(false);
  merge = async () => geometry.merged;
  fireEvent.click(mergeButton());
  await waitFor(() => expect(current().annotations).toHaveLength(2));
});

it("locks conflicting interactions during a merge and submits only one union", async () => {
  let finish!: (value: GeometryResult) => void;
  merge = () =>
    new Promise((resolve) => {
      finish = resolve;
    });
  await boot();
  selectTwo();
  fireEvent.change(screen.getByRole("textbox", { name: "Object to segment" }), {
    target: { value: "object" },
  });
  fireEvent.click(mergeButton());
  for (const name of [
    "Merge selected",
    "Accept selected",
    "Select proposal 1",
    "Discard proposals",
    "Select all",
    "Generate proposals with SAM 3",
  ])
    expect(
      (screen.getByRole("button", { name }) as HTMLButtonElement).disabled,
    ).toBe(true);
  fireEvent.click(mergeButton());
  fireEvent.submit(screen.getByRole("form", { name: "Concept segmentation" }));
  fireEvent.click(screen.getByTitle("image-2.png"));
  undo();
  expect(current().image.id).toBe(1);
  expect(current().busy).toBe(true);
  expect(calls("/masks/union")).toHaveLength(1);
  expect(calls("/infer/text")).toHaveLength(0);
  await act(async () => finish(geometry.merged));
  expect(current().annotations).toHaveLength(2);
  expect(current().busy).toBe(false);
});

it("discards a late merge after the editor unmounts instead of saving it in another session", async () => {
  let finish!: (value: GeometryResult) => void;
  merge = () =>
    new Promise((resolve) => {
      finish = resolve;
    });
  const view = await boot();
  selectTwo();
  await save();
  const before = structuredClone(disk.get(1));
  const writes = calls("/images/1/state").filter(
    ([, options]) => options.method === "PUT",
  ).length;
  fireEvent.click(mergeButton());
  view.unmount();
  await act(async () => {
    finish(geometry.merged);
    await new Promise((resolve) => setTimeout(resolve, 500));
  });
  expect(disk.get(1)).toEqual(before);
  expect(
    calls("/images/1/state").filter(([, options]) => options.method === "PUT"),
  ).toHaveLength(writes);
  canvas.current = null;
});

it("keeps a merged object in memory after a save failure and retries without duplicating it", async () => {
  await boot();
  selectTwo();
  failSave = true;
  fireEvent.click(mergeButton());
  await waitFor(() => expect(current().annotations).toHaveLength(2));
  const merged = structuredClone(current().annotations[1]);
  await screen.findByRole("button", { name: "Not saved · retry" });
  expect(disk.get(1)!.annotations).toEqual([existing]);
  expect(current().annotations[1]).toEqual(merged);
  failSave = false;
  fireEvent.click(screen.getByRole("button", { name: "Not saved · retry" }));
  await waitFor(() =>
    expect(disk.get(1)!.annotations).toEqual([existing, merged]),
  );
  expect(calls("/masks/union")).toHaveLength(1);
});
