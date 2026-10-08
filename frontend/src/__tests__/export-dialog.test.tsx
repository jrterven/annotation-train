// @vitest-environment jsdom
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "../api";
import ExportDialog from "../ExportDialog";
vi.mock("../api", async (original) => ({
  ...(await original<typeof import("../api")>()),
  api: vi.fn(),
}));
afterEach(() => {
  cleanup();
  vi.mocked(api).mockReset();
});
it("defaults to the active task, flushes changes and requests a unique COCO export", async () => {
  const flush = vi.fn().mockResolvedValue(undefined),
    ready = vi.fn();
  vi.mocked(api).mockResolvedValue({
    export_id: "unique",
    file_name: "annotations.coco.json",
  });
  render(
    <ExportDialog
      task="detection"
      flush={flush}
      onReady={ready}
      onClose={vi.fn()}
    />,
  );
  expect(
    (screen.getByLabelText("Export task") as HTMLSelectElement).value,
  ).toBe("detection");
  await act(async () =>
    fireEvent.click(screen.getByRole("button", { name: "Export" })),
  );
  expect(flush).toHaveBeenCalledOnce();
  expect(api).toHaveBeenCalledWith("/coco/export", "POST", {
    task: "detection",
    unique: true,
  });
  expect(ready).toHaveBeenCalledWith(
    "/coco/download/unique",
    "annotations.coco.json",
  );
});
it("requires preview and explicit approximation acceptance and invalidates it when options change", async () => {
  const preview = {
    snapshot: "revision-set",
    image_count: 1,
    annotation_count: 1,
    warnings: [
      { file_name: "one.png", annotation_id: "one", reason: "holes filled" },
    ],
    errors: [],
  };
  vi.mocked(api).mockResolvedValue(preview);
  render(
    <ExportDialog
      task="segmentation"
      flush={async () => {}}
      onReady={vi.fn()}
      onClose={vi.fn()}
    />,
  );
  fireEvent.change(screen.getByLabelText("Export format"), {
    target: { value: "yolo" },
  });
  expect(
    (
      screen.getByRole("checkbox", {
        name: "Include images",
      }) as HTMLInputElement
    ).checked,
  ).toBe(false);
  await act(async () =>
    fireEvent.click(screen.getByRole("button", { name: "Review export" })),
  );
  expect(
    (screen.getByRole("button", { name: "Export" }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
  fireEvent.click(
    screen.getByRole("checkbox", { name: /Accept approximations/ }),
  );
  expect(
    (screen.getByRole("button", { name: "Export" }) as HTMLButtonElement)
      .disabled,
  ).toBe(false);
  vi.mocked(api).mockResolvedValueOnce({
    export_id: "zip",
    file_name: "annotations-segmentation.zip",
  });
  await act(async () =>
    fireEvent.click(screen.getByRole("button", { name: "Export" })),
  );
  expect(api).toHaveBeenLastCalledWith("/yolo/export", "POST", {
    task: "segmentation",
    include_images: false,
    include_empty: false,
    snapshot: "revision-set",
    allow_lossy: true,
  });
  fireEvent.click(screen.getByRole("checkbox", { name: "Include images" }));
  expect(screen.queryByText("holes filled")).toBeNull();
  expect(screen.getByRole("button", { name: "Review export" })).toBeTruthy();
});
