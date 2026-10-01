export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}
let projectDirectory = "";
export function assetURL(path: string) {
  return `/api${path}${path.includes("?") ? "&" : "?"}project=${encodeURIComponent(projectDirectory)}`;
}
export async function api<T>(
  path: string,
  method = "GET",
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const exempt =
    path.startsWith("/health") ||
    path.startsWith("/browse") ||
    path === "/model/load" ||
    path === "/projects/open" ||
    path === "/coco/import" ||
    (path === "/project" && method === "GET");
  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (!exempt && projectDirectory)
    headers["X-Project-Directory"] = encodeURIComponent(projectDirectory);
  const res = await fetch(`/api${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let message = `Error ${res.status}`;
    try {
      const data = await res.json();
      message =
        typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail);
    } catch {
      /* no JSON */
    }
    throw new ApiError(res.status, message);
  }
  const data = await res.json();
  if (
    ["/project", "/projects/open", "/coco/import", "/project/relink"].includes(
      path,
    ) &&
    data?.directory
  )
    projectDirectory = data.directory;
  return data;
}
export const uid = () => crypto.randomUUID();
export const errorText = (err: unknown) =>
  err instanceof Error ? err.message : String(err);
