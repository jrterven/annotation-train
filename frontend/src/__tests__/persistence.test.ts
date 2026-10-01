// @vitest-environment jsdom
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api";
import { useWorkspace } from "../persistence";
import type { ImageState } from "../types";
vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api")>()),
  api: vi.fn(),
}));
const request = vi.mocked(api);
const initial: ImageState = {
  image_id: 1,
  revision: 0,
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
function edited(state: ImageState, id: string): ImageState {
  return {
    ...state,
    draft: {
      id,
      category_id: 0,
      parts: [{ id: "part", points: [] }],
      active_part_id: "part",
    },
  };
}
beforeEach(() => {
  vi.useFakeTimers();
  request.mockReset();
});
afterEach(() => {
  cleanup();
  vi.clearAllTimers();
  vi.useRealTimers();
});
describe("durable workspace state", () => {
  it("serializes saves, preserves edits made during PUT, and forwards the new revision", async () => {
    const first = deferred<ImageState>(),
      second = deferred<ImageState>();
    let puts = 0;
    request.mockImplementation(async (_path, method) =>
      method === "PUT"
        ? ++puts === 1
          ? first.promise
          : second.promise
        : structuredClone(initial),
    );
    const { result } = renderHook(() => useWorkspace(vi.fn()));
    await act(async () => {
      await result.current.load(1);
    });
    let saving!: Promise<void>;
    act(() => {
      result.current.update(1, (s) => edited(s, "first"));
      saving = result.current.save(1);
    });
    act(() => {
      result.current.update(1, (s) => edited(s, "second"));
    });
    expect(puts).toBe(1);
    await act(async () => {
      first.resolve({ ...initial, revision: 1 });
      await Promise.resolve();
    });
    expect(puts).toBe(2);
    const sent = request.mock.calls
      .filter((c) => c[1] === "PUT")
      .map((c) => c[2] as ImageState);
    expect(sent[0].revision).toBe(0);
    expect(sent[1].revision).toBe(1);
    expect(sent[1].draft?.id).toBe("second");
    await act(async () => {
      second.resolve({ ...initial, revision: 2 });
      await saving;
    });
    expect(result.current.get(1)?.draft?.id).toBe("second");
    expect(result.current.get(1)?.revision).toBe(2);
    expect(result.current.entry(1)?.dirty).toBe(false);
  });
  it("blocks flush on a revision conflict without dropping the local edit", async () => {
    request.mockResolvedValueOnce(structuredClone(initial));
    const onError = vi.fn();
    const { result } = renderHook(() => useWorkspace(onError));
    await act(async () => {
      await result.current.load(1);
    });
    request.mockRejectedValue(new ApiError(409, "revision conflict"));
    act(() => result.current.update(1, (s) => edited(s, "keep-me")));
    await act(async () => {
      await result.current.save(1);
    });
    await expect(result.current.flush()).rejects.toThrow();
    expect(result.current.entry(1)?.conflict).toBe(true);
    expect(result.current.get(1)?.draft?.id).toBe("keep-me");
    expect(onError).toHaveBeenCalled();
  });
  it("ignores an old project read after switching projects even when image IDs match", async () => {
    const oldRead = deferred<ImageState>();
    request.mockReturnValueOnce(oldRead.promise);
    const { result } = renderHook(() => useWorkspace(vi.fn()));
    let loading!: Promise<ImageState>;
    act(() => {
      loading = result.current.load(1);
      result.current.reset();
    });
    await act(async () => {
      oldRead.resolve(initial);
      await loading;
    });
    expect(result.current.get(1)).toBeUndefined();
  });
  it("does not let a duplicate late read overwrite changes", async () => {
    const slow = deferred<ImageState>();
    request.mockResolvedValueOnce(initial).mockReturnValueOnce(slow.promise);
    const { result } = renderHook(() => useWorkspace(vi.fn()));
    let late!: Promise<ImageState>;
    await act(async () => {
      const fast = result.current.load(1);
      late = result.current.load(1);
      await fast;
    });
    act(() => result.current.update(1, (s) => edited(s, "preserved")));
    await act(async () => {
      slow.resolve(initial);
      await late;
    });
    expect(result.current.get(1)?.draft?.id).toBe("preserved");
  });
  it("undo and redo preserve the latest disk revision", async () => {
    request.mockResolvedValue({ ...initial, revision: 7 });
    const { result } = renderHook(() => useWorkspace(vi.fn()));
    await act(async () => {
      await result.current.load(1);
    });
    act(() => {
      result.current.update(1, (s) => edited(s, "object"));
      result.current.history(1);
    });
    expect(result.current.get(1)?.draft).toBeNull();
    expect(result.current.get(1)?.revision).toBe(7);
    act(() => result.current.history(1, true));
    expect(result.current.get(1)?.draft?.id).toBe("object");
    expect(result.current.get(1)?.revision).toBe(7);
  });
});
