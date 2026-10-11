// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  act,
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
            ? {
                counts: { train: 7, validation: 2, test: 1 },
                warnings: [],
                fingerprint: "a".repeat(64),
              }
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
  vi.restoreAllMocks();
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

function capturePolling() {
  let poll = () => {};
  const original = globalThis.setInterval;
  vi.spyOn(globalThis, "setInterval").mockImplementation((callback, delay) => {
    if (delay !== 2000) return original(callback, delay);
    poll = callback as () => void;
    return 0 as unknown as ReturnType<typeof setInterval>;
  });
  return async () => {
    await act(async () => poll());
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
          String(url).endsWith("/datasets/preview?summary_only=true"),
        ),
      ).toBe(true),
    );
    let call = fetcher.mock.calls.find(([url]) =>
      String(url).endsWith("/datasets/preview?summary_only=true"),
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
          String(url).endsWith("/datasets/preview?summary_only=true"),
        ),
      ).toHaveLength(2),
    );
    call = fetcher.mock.calls.filter(([url]) =>
      String(url).endsWith("/datasets/preview?summary_only=true"),
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
          fingerprint: "a".repeat(64),
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

  it("selects the newest dataset by version even when UUID order differs", async () => {
    const poll = capturePolling();
    const versions = [3, 1, 2].map((version) => ({
      id: `snapshot-${version}`,
      version,
      task: "detection",
      counts: { train: 8, validation: 2 },
      warnings: [],
      fingerprint: "a".repeat(64),
    }));
    fetcher.mockImplementation(
      async () => new Response(JSON.stringify({ ...base, datasets: versions })),
    );
    render(<TrainingWorkspace {...props()} />);
    await waitFor(() =>
      expect(
        (screen.getByLabelText("Dataset version") as HTMLSelectElement).value,
      ).toBe("snapshot-3"),
    );
    fireEvent.change(screen.getByLabelText("Dataset version"), {
      target: { value: "snapshot-1" },
    });
    await poll();
    await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(2));
    expect(
      (screen.getByLabelText("Dataset version") as HTMLSelectElement).value,
    ).toBe("snapshot-1");
  });

  it("binds creation to the compact preview fingerprint and invalidates rejected previews", async () => {
    render(<TrainingWorkspace {...props()} />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Preview split" }),
    );
    const create = screen.getByRole("button", {
      name: "Create dataset version",
    }) as HTMLButtonElement;
    await waitFor(() => expect(create.disabled).toBe(false));
    fetcher.mockImplementation(async (url: string) =>
      String(url).endsWith("/datasets?summary_only=true")
        ? new Response(
            JSON.stringify({
              detail: "Reviewed data changed. Preview the split again.",
            }),
            { status: 409 },
          )
        : new Response(JSON.stringify(base)),
    );
    fireEvent.click(create);
    await screen.findByText("Reviewed data changed. Preview the split again.");
    const call = fetcher.mock.calls.find(([url]) =>
      String(url).endsWith("/datasets?summary_only=true"),
    )!;
    expect(JSON.parse(call[1]!.body as string).expected_fingerprint).toBe(
      "a".repeat(64),
    );
    expect(create.disabled).toBe(true);
  });

  it.each(["revision", "reviewed"])(
    "invalidates a preview when polling observes changed %s",
    async (field) => {
      const poll = capturePolling();
      render(<TrainingWorkspace {...props()} />);
      fireEvent.click(
        await screen.findByRole("button", { name: "Preview split" }),
      );
      const create = screen.getByRole("button", {
        name: "Create dataset version",
      }) as HTMLButtonElement;
      await waitFor(() => expect(create.disabled).toBe(false));
      fetcher.mockResolvedValue(
        new Response(
          JSON.stringify({
            ...base,
            images: [
              { ...base.images[0], [field]: field === "revision" ? 3 : false },
            ],
          }),
        ),
      );
      await poll();
      await waitFor(() => expect(create.disabled).toBe(true));
    },
  );

  it("invalidates split approval when returning from the editor but keeps split settings", async () => {
    const p = props();
    const view = render(<TrainingWorkspace {...p} />);
    fireEvent.click(await screen.findByLabelText("Reserve test set"));
    fireEvent.click(screen.getByRole("button", { name: "Preview split" }));
    await waitFor(() =>
      expect(
        (
          screen.getByRole("button", {
            name: "Create dataset version",
          }) as HTMLButtonElement
        ).disabled,
      ).toBe(false),
    );
    view.rerender(<TrainingWorkspace {...p} visible={false} />);
    view.rerender(<TrainingWorkspace {...p} visible />);
    expect(
      (
        screen.getByRole("button", {
          name: "Create dataset version",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    expect(
      (screen.getByLabelText("Reserve test set") as HTMLInputElement).checked,
    ).toBe(true);
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
