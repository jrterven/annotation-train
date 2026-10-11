// @vitest-environment jsdom
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import App from "../App";
import { api, selectProject } from "../api";
import type { Project, ProjectImageState } from "../types";
import type { Workspace } from "../persistence";

const observed = vi.hoisted(() => ({
  workspace: null as Workspace | null,
  apply: null as Promise<void> | null,
}));
vi.mock("../persistence", async (original) => {
  const actual = await original<typeof import("../persistence")>();
  return {
    ...actual,
    useWorkspace: (...args: Parameters<typeof actual.useWorkspace>) => {
      const workspace = actual.useWorkspace(...args);
      observed.workspace = workspace;
      return workspace;
    },
  };
});
vi.mock("../CanvasEditor", () => ({
  default: () => <div data-testid="canvas" />,
}));
vi.mock("../TrainingWorkspace", () => ({
  default: ({
    onPredictions,
  }: {
    onPredictions: (id: string) => Promise<void>;
  }) => (
    <button
      onClick={() => {
        observed.apply = onPredictions("batch");
      }}
    >
      Apply test proposals
    </button>
  ),
}));
vi.mock("../ProjectDialog", () => ({
  ProjectDialog: ({ onOpen }: { onOpen: (project: Project) => void }) => (
    <button onClick={() => onOpen(second)}>Open second project</button>
  ),
}));
vi.mock("../api", async (original) => ({
  ...(await original<typeof import("../api")>()),
  api: vi.fn(),
  selectProject: vi.fn(),
  projectClient: () => async () => ({ image_ids: [1, 2] }),
}));
const first: Project = {
  id: "first",
  name: "First",
  directory: "/tmp/first",
  image_root: "/tmp/first/images",
  categories: [{ id: 1, name: "Object", color: "#123456" }],
  images: [1, 2].map((id) => ({
    id,
    file_name: `${id}.png`,
    width: 16,
    height: 12,
    annotation_count: 0,
  })),
};
const second: Project = {
  ...first,
  id: "second",
  name: "Second",
  directory: "/tmp/second",
};
const state = (id: number): ProjectImageState => ({
  image_id: id,
  revision: 0,
  annotations: [],
  draft: null,
  proposals: [],
});
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("stops a proposal reload batch on project switch and preserves edits to overlapping image IDs", async () => {
  let current = "first",
    hold = false;
  let release!: (value: ProjectImageState) => void;
  const pending = new Promise<ProjectImageState>((resolve) => {
    release = resolve;
  });
  const reads: string[] = [];
  vi.mocked(selectProject).mockImplementation((project) => {
    current = project.id!;
  });
  vi.mocked(api).mockImplementation(async (path, method = "GET") => {
    if (path === "/project") return first;
    if (path === "/health") return { model: { state: "unloaded" } };
    if (path === "/training/status") return { active_runs: 0 };
    const match = path.match(/^\/images\/(\d+)\/state$/);
    if (match && method === "GET") {
      const id = Number(match[1]);
      reads.push(`${current}:${id}`);
      if (hold && current === "first" && id === 1) return pending;
      return state(id);
    }
    throw new Error(`Unexpected ${method} ${path}`);
  });
  render(<App />);
  await screen.findByTestId("canvas");
  fireEvent.click(screen.getByRole("button", { name: "Train" }));
  hold = true;
  fireEvent.click(screen.getByRole("button", { name: "Apply test proposals" }));
  await waitFor(() =>
    expect(reads.filter((id) => id === "first:1")).toHaveLength(2),
  );
  fireEvent.click(screen.getByTitle("Switch project"));
  fireEvent.click(
    await screen.findByRole("button", { name: "Open second project" }),
  );
  await waitFor(() => expect(current).toBe("second"));
  await act(async () => {
    await observed.workspace!.load(2);
  });
  act(() =>
    observed.workspace!.updateDetection(2, (s) => ({
      ...s,
      draft: {
        id: "unsaved-second-project",
        category_id: 1,
        bbox: [1, 1, 4, 4],
        points: [],
      },
    })),
  );
  await act(async () => {
    release(state(1));
    await observed.apply;
  });
  expect(reads.filter((id) => id === "second:2")).toHaveLength(1);
  expect(observed.workspace!.getDetection(2)?.draft?.id).toBe(
    "unsaved-second-project",
  );
  expect(observed.workspace!.records.get(2)?.dirty).toBe(true);
});
