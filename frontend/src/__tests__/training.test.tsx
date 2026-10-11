// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import TrainingWorkspace from "../TrainingWorkspace";
import { configureApi, projectClient, selectProject } from "../api";
import type { Project } from "../types";
import type { TrainingOverview } from "../trainingTypes";

const project = {
  id: "p1",
  name: "Examples",
  directory: "/tmp/examples",
  categories: [],
  images: [],
} as unknown as Project;
const base: TrainingOverview = {
  hosted: false,
  images: [
    {
      id: 1,
      file_name: "one.png",
      revision: 2,
      reviewed: true,
      annotation_count: 1,
    },
  ],
  datasets: [],
  runs: [],
  devices: [],
  catalog: [
    {
      id: "yolo26-small-detection",
      name: "YOLO26 Small",
      family: "yolo26",
      task: "detection",
      size: "small",
      resolution: 640,
      license: "AGPL-3.0",
      license_url: "https://www.ultralytics.com/license",
    },
  ],
};
let fetcher: ReturnType<typeof vi.fn>;
beforeEach(() => {
  configureApi("local");
  selectProject(project);
  fetcher = vi.fn(
    async (_url: string, init?: RequestInit) =>
      new Response(
        JSON.stringify(
          init?.method === "POST"
            ? { counts: { train: 7, validation: 2, test: 1 }, warnings: [] }
            : base,
        ),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
  );
  vi.stubGlobal("fetch", fetcher);
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
function props() {
  return {
    project,
    task: "detection" as const,
    visible: true,
    flush: vi.fn(async () => {}),
    onAnnotate: vi.fn(),
    onStatus: vi.fn(),
    onPredictions: vi.fn(async () => {}),
  };
}

describe("training workspace", () => {
  it("defaults to train/validation and adds test only when explicitly enabled", async () => {
    const p = props();
    render(<TrainingWorkspace {...p} />);
    const toggle = await screen.findByLabelText("Reserve test set");
    expect((toggle as HTMLInputElement).checked).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Preview split" }));
    await waitFor(() =>
      expect(
        fetcher.mock.calls.some(([url]) =>
          String(url).endsWith("/datasets/preview"),
        ),
      ).toBe(true),
    );
    let call = fetcher.mock.calls.find(([url]) =>
      String(url).endsWith("/datasets/preview"),
    )!;
    expect(JSON.parse(call[1]!.body as string)).toMatchObject({
      validation_percent: 20,
      test_percent: 0,
      seed: 42,
    });
    await waitFor(() =>
      expect(
        (
          screen.getByRole("button", {
            name: "Preview split",
          }) as HTMLButtonElement
        ).disabled,
      ).toBe(false),
    );
    fireEvent.click(toggle);
    expect(screen.getByText("70% training")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Preview split" }));
    await waitFor(() =>
      expect(
        fetcher.mock.calls.filter(([url]) =>
          String(url).endsWith("/datasets/preview"),
        ),
      ).toHaveLength(2),
    );
    call = fetcher.mock.calls.filter(([url]) =>
      String(url).endsWith("/datasets/preview"),
    )[1];
    expect(JSON.parse(call[1]!.body as string)).toMatchObject({
      validation_percent: 20,
      test_percent: 10,
    });
    expect(p.flush).toHaveBeenCalledTimes(2);
  });

  it("preserves the configuration when switching between annotate and train", async () => {
    const p = props();
    const view = render(<TrainingWorkspace {...p} />);
    const toggle = await screen.findByLabelText("Reserve test set");
    fireEvent.click(toggle);
    fireEvent.change(screen.getByLabelText("Test percent"), {
      target: { value: "15" },
    });
    view.rerender(<TrainingWorkspace {...p} visible={false} />);
    view.rerender(<TrainingWorkspace {...p} visible />);
    expect(
      (screen.getByLabelText("Reserve test set") as HTMLInputElement).checked,
    ).toBe(true);
    expect(
      (screen.getByLabelText("Test percent") as HTMLInputElement).value,
    ).toBe("15");
  });

  it("keeps validation charts separate from the final test report", async () => {
    const data: TrainingOverview = {
      ...base,
      datasets: [
        {
          id: "d",
          version: 1,
          task: "detection",
          counts: { train: 7, validation: 2, test: 1 },
          warnings: [],
        },
      ],
      runs: [
        {
          id: "r",
          dataset_id: "d",
          task: "detection",
          device_id: "gpu",
          model: base.catalog[0],
          parameters: { epochs: 10 },
          status: "completed",
          epoch: 3,
          created_at: 1,
          events: 4,
          metrics: [
            { epoch: 1, train_loss: 2, validation_loss: 3, map: 0.25 },
            { epoch: 2, train_loss: 1, validation_loss: 2, map: 0.4 },
          ],
          artifacts: [],
          evaluation: { status: "completed", metrics: { "mAP50-95": 0.3 } },
        },
      ],
    };
    fetcher.mockImplementation(
      async () => new Response(JSON.stringify(data), { status: 200 }),
    );
    render(<TrainingWorkspace {...props()} />);
    fireEvent.click(
      await screen.findByRole("button", { name: /YOLO26 Small/ }),
    );
    expect(
      screen.getByRole("img", { name: "Validation mAP 50–95 by epoch" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("heading", { name: "Test evaluation · completed" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Evaluate on test" }),
    ).toBeTruthy();
    expect(screen.getByText("mAP50-95: 0.300")).toBeTruthy();
  });

  it("captures project identity before asynchronous saves or project switching", async () => {
    configureApi("hosted", "csrf");
    selectProject({ id: "first" });
    const scoped = projectClient({ id: "first" });
    selectProject({ id: "second" });
    await scoped("/training/runs", "POST", { dataset_id: "immutable" });
    expect(fetcher.mock.calls.at(-1)![0]).toBe(
      "/api/v1/projects/first/training/runs",
    );
  });
});
