// @vitest-environment jsdom
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "../api";
import { useWorkspace } from "../persistence";
import { useEditorInference } from "../editorController";
import type { Task } from "../types";
vi.mock("../api", async (original) => ({
  ...(await original<typeof import("../api")>()),
  api: vi.fn(),
}));
afterEach(() => {
  cleanup();
  vi.mocked(api).mockReset();
});
const initial = {
  image_id: 1,
  revision: 3,
  annotations: [],
  draft: null,
  proposals: [],
};
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { resolve, promise };
}
function useEditors() {
  const workspace = useWorkspace(vi.fn());
  const segmentation = useEditorInference("segmentation", workspace);
  const detection = useEditorInference("detection", workspace);
  return { workspace, segmentation, detection };
}
it.each<Task>(["segmentation", "detection"])(
  "rejects an old %s response after a forced state reload, even with identical IDs",
  async (task) => {
    vi.mocked(api).mockResolvedValue(initial);
    const { result } = renderHook(useEditors);
    await act(async () => {
      await result.current.workspace.load(1);
    });
    const late = deferred<{ image_id: number; proposals: [] }>();
    vi.mocked(api).mockReturnValueOnce(late.promise);
    let request!: ReturnType<typeof result.current.segmentation.begin>;
    act(() => {
      request = result.current[task].begin("object", 1, "text");
    });
    const response = request.concept("objects", "en", 7, null);
    await act(async () => {
      await result.current.workspace.load(1, true);
    });
    await act(async () => late.resolve({ image_id: 1, proposals: [] }));
    expect(await response).toBeUndefined();
    const errors = vi.fn();
    request.error(new Error("old failure"), errors);
    expect(errors).not.toHaveBeenCalled();
  },
);
it("cancels and supersedes requests within one task while retaining the other task", async () => {
  vi.mocked(api).mockResolvedValue(initial);
  const { result, unmount } = renderHook(useEditors);
  await act(async () => {
    await result.current.workspace.load(1);
  });
  let first!: ReturnType<typeof result.current.segmentation.begin>,
    second!: typeof first,
    box!: typeof first;
  act(() => {
    first = result.current.segmentation.begin("object", 1, "points");
    box = result.current.detection.begin("object", 1, "points");
    second = result.current.segmentation.begin("object", 1, "points");
  });
  expect(first.controller.signal.aborted).toBe(true);
  expect(box.isCurrent()).toBe(true);
  act(() => result.current.segmentation.cancel());
  expect(second.controller.signal.aborted).toBe(true);
  expect(box.isCurrent()).toBe(true);
  unmount();
  expect(box.controller.signal.aborted).toBe(true);
});
it.each<Task>(["segmentation", "detection"])(
  "uses the shared visual payload and rejects wrong-image results in %s",
  async (task) => {
    vi.mocked(api).mockResolvedValue(initial);
    const { result } = renderHook(useEditors);
    await act(async () => {
      await result.current.workspace.load(1);
    });
    let request!: ReturnType<typeof result.current.segmentation.begin>;
    act(() => {
      request = result.current[task].begin("concept", 1, "text");
    });
    vi.mocked(api).mockResolvedValueOnce({ image_id: 2, proposals: [] });
    const visual = {
      base64: "image",
      dataUrl: "data:image/png;base64,image",
      box: [1, 2, 3, 4] as [number, number, number, number],
      name: "reference",
      width: 8,
      height: 6,
    };
    expect(await request.concept(" objetos ", "es", 7, visual)).toBeUndefined();
    expect(api).toHaveBeenLastCalledWith(
      "/infer/visual",
      "POST",
      {
        image_id: 1,
        revision: 3,
        text: "objetos",
        source_language: "es",
        category_id: 7,
        reference_image: "image",
        reference_box: [1, 2, 3, 4],
      },
      request.controller.signal,
    );
  },
);
