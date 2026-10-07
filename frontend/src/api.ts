export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

let projectDirectory = "";
let projectId = "";
let hosted = false;
let csrfToken: string | null = null;
export const sessionExpiredEvent = "annotation:session-expired";
export const usageChangedEvent = "annotation:usage-changed";
export const jobChangedEvent = "annotation:job-changed";
export type JobProgress = { id: string; status: string; imageId?: number };

export function configureApi(
  mode: "local" | "hosted",
  csrf: string | null = null,
) {
  hosted = mode === "hosted";
  csrfToken = csrf;
  projectId = "";
  projectDirectory = "";
}
export function setCsrfToken(value: string | null) {
  csrfToken = value;
}
export function selectProject(project: { id?: string; directory?: string }) {
  projectId = project.id || "";
  projectDirectory = project.directory || "";
}
function hostedURL(path: string) {
  if (/^\/(auth|projects|jobs|health|model|config)(\/|\?|$)/.test(path))
    return `/api/v1${path}`;
  if (!projectId) throw new Error("Select a project first.");
  const suffix = path.replace(/^\/project(?=\/|$)/, "");
  return `/api/v1/projects/${encodeURIComponent(projectId)}${suffix}`;
}
export function assetURL(path: string) {
  return hosted
    ? hostedURL(path)
    : `/api${path}${path.includes("?") ? "&" : "?"}project=${encodeURIComponent(projectDirectory)}`;
}
async function request<T>(
  url: string,
  method: string,
  body?: unknown,
  signal?: AbortSignal,
  headers: Record<string, string> = {},
): Promise<{ data: T; status: number }> {
  const isForm = body instanceof FormData;
  const isBlob = body instanceof Blob;
  if (body !== undefined && !isForm && !isBlob)
    headers["Content-Type"] = "application/json";
  if (hosted && !["GET", "HEAD", "OPTIONS"].includes(method) && csrfToken)
    headers["X-CSRF-Token"] = csrfToken;
  const res = await fetch(url, {
    method,
    headers,
    credentials: "same-origin",
    ...(hosted ? { cache: "no-store" as const } : {}),
    body:
      body === undefined
        ? undefined
        : isForm || isBlob
          ? body
          : JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let message = `Error ${res.status}`;
    try {
      const data = await res.json();
      message =
        typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail) || message;
    } catch {
      /* Non-JSON proxy errors retain the HTTP status. */
    }
    if (hosted && res.status === 401)
      window.dispatchEvent(new Event(sessionExpiredEvent));
    throw new ApiError(res.status, message);
  }
  return {
    data: res.status === 204 ? undefined : await res.json(),
    status: res.status,
  };
}
function delay(ms: number, signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const abort = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
      reject(new DOMException("Request cancelled", "AbortError"));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", abort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
  });
}
type Job<T> = { id: string; status: string; result?: T; error?: string };
const jobErrors: Record<string, string> = {
  invalid_input:
    "The image or prompt could not be processed. Check it and try again.",
  inference_unavailable:
    "SAM 3 is temporarily unavailable. You can retry or annotate with polygons.",
  result_too_large:
    "The result is too large. Try a narrower description or a smaller image.",
  result_expired: "This inference result has expired. Run segmentation again.",
  queue_expired:
    "This request expired after waiting 24 hours. Its reserved quota was released; you can try again.",
};
async function waitForJob<T>(
  initial: Job<T>,
  imageId?: number,
  signal?: AbortSignal,
): Promise<T> {
  const url = `/api/v1/jobs/${encodeURIComponent(initial.id)}`;
  const announce = (status: string) =>
    window.dispatchEvent(
      new CustomEvent<JobProgress>(jobChangedEvent, {
        detail: { id: initial.id, status, imageId },
      }),
    );
  let job = initial;
  let failures = 0;
  try {
    while (true) {
      if (signal?.aborted)
        throw new DOMException("Request cancelled", "AbortError");
      announce(failures ? "reconnecting" : job.status);
      if (job.status === "succeeded") {
        if (job.result === undefined)
          throw new Error("The inference returned no result.");
        return job.result;
      }
      if (job.status === "failed")
        throw new Error(
          (job.error && (jobErrors[job.error] || job.error)) ||
            "Inference failed. You can retry.",
        );
      if (job.status === "cancelled")
        throw new DOMException("Request cancelled", "AbortError");
      await delay(1000, signal);
      try {
        job = (await request<Job<T>>(url, "GET", undefined, signal)).data;
        failures = 0;
      } catch (error) {
        if (
          signal?.aborted ||
          (error instanceof ApiError && error.status < 500)
        )
          throw error;
        if (++failures >= 10)
          throw new Error(
            "Connection lost. The job could still be running; reconnect before retrying.",
          );
        announce("reconnecting");
      }
    }
  } finally {
    if (signal?.aborted) {
      // Keep enqueue alive until the job ID arrives so cancellation during the
      // initial request can also cancel a job the server has already accepted.
      void request(`${url}/cancel`, "POST", {}).catch(() =>
        announce("cancel_unconfirmed"),
      );
    }
    announce("finished");
    window.dispatchEvent(new Event(usageChangedEvent));
  }
}
export async function api<T>(
  path: string,
  method = "GET",
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  if (hosted && path === "/project" && method === "GET" && !projectId)
    return null as T;
  const exempt =
    path.startsWith("/health") ||
    path.startsWith("/browse") ||
    path === "/model/load" ||
    path === "/projects/open" ||
    path === "/coco/import" ||
    (path === "/project" && method === "GET");
  const headers: Record<string, string> = {};
  if (!hosted && !exempt && projectDirectory)
    headers["X-Project-Directory"] = encodeURIComponent(projectDirectory);
  const inference = hosted && method === "POST" && path.startsWith("/infer/");
  if (signal?.aborted)
    throw new DOMException("Request cancelled", "AbortError");
  const { data, status } = await request<T>(
    hosted ? hostedURL(path) : `/api${path}`,
    method,
    body,
    inference ? undefined : signal,
    headers,
  );
  if (inference && status === 202)
    return waitForJob(
      data as Job<T>,
      (body as { image_id?: number })?.image_id,
      signal,
    );
  if (
    hosted &&
    ["POST", "PUT", "DELETE", "PATCH"].includes(method) &&
    (/^\/projects?(\/|$)/.test(path) ||
      /^\/images\/\d+\/state$/.test(path) ||
      path === "/coco/import")
  )
    window.dispatchEvent(new Event(usageChangedEvent));
  if (
    !hosted &&
    ["/project", "/projects/open", "/coco/import", "/project/relink"].includes(
      path,
    )
  ) {
    const project = data as { directory?: string } | null;
    if (project?.directory) projectDirectory = project.directory;
  }
  return data;
}
export const uid = () => crypto.randomUUID();
export const errorText = (err: unknown) =>
  err instanceof Error ? err.message : String(err);
