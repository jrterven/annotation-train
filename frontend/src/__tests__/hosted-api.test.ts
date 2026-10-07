// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  api,
  ApiError,
  assetURL,
  configureApi,
  jobChangedEvent,
  selectProject,
  sessionExpiredEvent,
} from "../api";

const fetchMock = vi.fn();
const response = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
  configureApi("hosted", "session-csrf");
  selectProject({ id: "private-project" });
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  fetchMock.mockReset();
  configureApi("local");
});

describe("hosted API boundary", () => {
  it("scopes state and assets to the project ID without local paths and sends CSRF on writes", async () => {
    fetchMock.mockImplementation(async () => response({ revision: 2 }));
    await api("/images/7/state", "PUT", { revision: 1 });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/projects/private-project/images/7/state",
      expect.objectContaining({
        credentials: "same-origin",
        cache: "no-store",
        headers: {
          "X-CSRF-Token": "session-csrf",
          "Content-Type": "application/json",
        },
        body: '{"revision":1}',
      }),
    );
    expect(assetURL("/images/7/file?thumbnail=true")).toBe(
      "/api/v1/projects/private-project/images/7/file?thumbnail=true",
    );
    await api("/project/categories", "POST", { name: "Carrot" });
    expect(fetchMock.mock.calls[1][0]).toBe(
      "/api/v1/projects/private-project/categories",
    );
  });
  it("lets the browser set multipart boundaries without dropping CSRF", async () => {
    fetchMock.mockResolvedValue(response({}));
    const form = new FormData();
    form.append("file", new File(["image"], "carrot.png"));
    await api("/projects/private-project/images", "POST", form);
    expect(fetchMock.mock.calls[0][1].body).toBe(form);
    expect(fetchMock.mock.calls[0][1].headers).toEqual({
      "X-CSRF-Token": "session-csrf",
    });
  });
  it("keeps revision conflicts visible and announces expired sessions", async () => {
    const expired = vi.fn();
    window.addEventListener(sessionExpiredEvent, expired);
    fetchMock.mockResolvedValueOnce(
      response({ detail: "Session expired" }, 401),
    );
    await expect(api("/images/7/state")).rejects.toMatchObject({ status: 401 });
    expect(expired).toHaveBeenCalledOnce();
    fetchMock.mockResolvedValueOnce(
      response({ detail: "Revision conflict" }, 409),
    );
    await expect(api("/images/7/state", "PUT", {})).rejects.toBeInstanceOf(
      ApiError,
    );
    window.removeEventListener(sessionExpiredEvent, expired);
  });
  it("preserves the original local routes and encoded project header", async () => {
    configureApi("local");
    selectProject({ directory: "/local/my project" });
    fetchMock.mockResolvedValue(response({}));
    await api("/images/7/state", "PUT", {});
    expect(fetchMock.mock.calls[0][0]).toBe("/api/images/7/state");
    expect(fetchMock.mock.calls[0][1].headers["X-Project-Directory"]).toBe(
      "%2Flocal%2Fmy%20project",
    );
    expect(assetURL("/images/7/file")).toBe(
      "/api/images/7/file?project=%2Flocal%2Fmy%20project",
    );
  });
});

describe("persistent inference jobs", () => {
  it("polls a queued job and returns the existing inference result shape", async () => {
    vi.useFakeTimers();
    const result = { image_id: 7, revision: 4, proposals: [] };
    fetchMock
      .mockResolvedValueOnce(response({ id: "job-a", status: "queued" }, 202))
      .mockResolvedValueOnce(response({ id: "job-a", status: "running" }))
      .mockResolvedValueOnce(
        response({ id: "job-a", status: "succeeded", result }),
      );
    const task = api("/infer/text", "POST", { image_id: 7 });
    await vi.advanceTimersByTimeAsync(1000);
    await expect(task).resolves.toEqual(result);
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/api/v1/projects/private-project/infer/text",
      "/api/v1/jobs/job-a",
      "/api/v1/jobs/job-a",
    ]);
  });
  it("cancels server work even if the caller aborts before enqueue returns", async () => {
    let accept!: (result: Response) => void;
    fetchMock
      .mockImplementationOnce(
        () =>
          new Promise<Response>((resolve) => {
            accept = resolve;
          }),
      )
      .mockResolvedValue(response({ status: "cancelled" }));
    const controller = new AbortController();
    const task = api(
      "/infer/points",
      "POST",
      { image_id: 7 },
      controller.signal,
    );
    const rejection = expect(task).rejects.toMatchObject({
      name: "AbortError",
    });
    controller.abort();
    accept(response({ id: "job-b", status: "queued" }, 202));
    await rejection;
    expect(fetchMock).toHaveBeenLastCalledWith(
      "/api/v1/jobs/job-b/cancel",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ "X-CSRF-Token": "session-csrf" }),
      }),
    );
  });
  it("recovers from a temporary poll failure without resubmitting inference", async () => {
    vi.useFakeTimers();
    const progress: string[] = [];
    const listener = (event: Event) =>
      progress.push((event as CustomEvent).detail.status);
    window.addEventListener(jobChangedEvent, listener);
    fetchMock
      .mockResolvedValueOnce(response({ id: "job-c", status: "queued" }, 202))
      .mockRejectedValueOnce(new TypeError("Network unavailable"))
      .mockResolvedValueOnce(
        response({
          id: "job-c",
          status: "succeeded",
          result: { mask: "result" },
        }),
      );
    const task = api("/infer/points", "POST", { image_id: 7 });
    await vi.advanceTimersByTimeAsync(1000);
    expect(progress.at(-1)).toBe("reconnecting");
    await vi.advanceTimersByTimeAsync(1000);
    await expect(task).resolves.toEqual({ mask: "result" });
    expect(progress).toContain("reconnecting");
    expect(
      fetchMock.mock.calls.filter(([, options]) => options.method === "POST"),
    ).toHaveLength(1);
    window.removeEventListener(jobChangedEvent, listener);
  });
});

it("reports cancellation that could not reach the server", async () => {
  vi.useFakeTimers();
  const events: string[] = [];
  const listener = (event: Event) =>
    events.push((event as CustomEvent).detail.status);
  window.addEventListener(jobChangedEvent, listener);
  fetchMock
    .mockResolvedValueOnce(
      response({ id: "job-unknown", status: "queued" }, 202),
    )
    .mockRejectedValueOnce(new TypeError("Offline"));
  const controller = new AbortController();
  const pending = api(
    "/infer/points",
    "POST",
    { image_id: 7 },
    controller.signal,
  );
  const rejected = expect(pending).rejects.toMatchObject({
    name: "AbortError",
  });
  await vi.advanceTimersByTimeAsync(0);
  controller.abort();
  await rejected;
  await vi.advanceTimersByTimeAsync(0);
  expect(events).toContain("cancel_unconfirmed");
  window.removeEventListener(jobChangedEvent, listener);
});
