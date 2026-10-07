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
import { HostedProjects, HostedUploads } from "../HostedProjects";
import RuntimeApp from "../RuntimeApp";
import { api, configureApi, sessionExpiredEvent } from "../api";
import type { HostedProject } from "../types";

vi.mock("../App", () => ({
  default: ({ onLogout }: { onLogout?: () => Promise<void> }) => (
    <div>
      Shared editor
      {onLogout && (
        <button onClick={() => void onLogout()}>Mock sign out</button>
      )}
    </div>
  ),
}));
vi.mock("../api", async (original) => ({
  ...(await original<typeof import("../api")>()),
  api: vi.fn(),
}));
const request = vi.mocked(api);
const project: HostedProject = {
  id: "private-one",
  name: "Carrots",
  categories: [],
  images: [],
};
const fetchMock = vi.fn();
beforeEach(() => {
  request.mockReset();
  vi.stubGlobal("fetch", fetchMock);
  window.history.replaceState({}, "", "/");
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  fetchMock.mockReset();
  configureApi("local");
});

describe("runtime and Google sign in", () => {
  it("shows Google sign-in when the hosted session is anonymous", async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ mode: "hosted" })),
    );
    request.mockResolvedValue({ user: null, csrf_token: null, usage: null });
    render(<RuntimeApp />);
    const link = await screen.findByRole("link", {
      name: /Continue with Google/,
    });
    expect(link.getAttribute("href")).toBe("/api/v1/auth/google/login");
    expect(screen.queryByText("Shared editor")).toBeNull();
    expect(request).toHaveBeenCalledWith("/auth/session");
  });
  it("falls back to local only for explicit config 404, never a network failure", async () => {
    fetchMock.mockRejectedValueOnce(new TypeError("Offline"));
    render(<RuntimeApp />);
    expect(await screen.findByRole("alert")).toHaveProperty(
      "textContent",
      "Offline",
    );
    expect(screen.queryByText("Shared editor")).toBeNull();
    fetchMock.mockResolvedValueOnce(new Response("", { status: 404 }));
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("Shared editor")).toBeTruthy();
  });
  it("serves the privacy page without requiring sign in or making API requests", () => {
    window.history.replaceState({}, "", "/privacy");
    render(<RuntimeApp />);
    expect(
      screen.getByRole("heading", { name: "Privacy policy" }),
    ).toBeTruthy();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(request).not.toHaveBeenCalled();
  });
});

describe("private projects", () => {
  it("opens a UUID project without exposing filesystem controls", async () => {
    request.mockImplementation(async (path) =>
      path === "/projects" ? { projects: [project] } : project,
    );
    const onOpen = vi.fn();
    render(
      <HostedProjects onClose={() => {}} onDelete={() => {}} onOpen={onOpen} />,
    );
    fireEvent.click(
      await screen.findByRole("button", { name: "Open Carrots" }),
    );
    await waitFor(() =>
      expect(onOpen).toHaveBeenCalledWith({
        ...project,
        directory: "",
        image_root: "",
      }),
    );
    expect(request).toHaveBeenCalledWith("/projects/private-one");
    expect(screen.queryByLabelText("Folder path")).toBeNull();
  });
  it("requires a concrete delete confirmation and removes only the selected project", async () => {
    request.mockImplementation(async (path) =>
      path === "/projects" ? { projects: [project] } : {},
    );
    const onDelete = vi.fn();
    render(
      <HostedProjects
        onClose={() => {}}
        onDelete={onDelete}
        onOpen={() => {}}
      />,
    );
    fireEvent.click(
      await screen.findByRole("button", { name: "Delete Carrots" }),
    );
    expect(request).not.toHaveBeenCalledWith("/projects/private-one", "DELETE");
    fireEvent.click(screen.getByRole("button", { name: "Delete permanently" }));
    await waitFor(() => expect(onDelete).toHaveBeenCalledWith("private-one"));
    expect(request).toHaveBeenCalledWith("/projects/private-one", "DELETE");
  });
  it("uploads a browser file and its relative filename, then refreshes the project", async () => {
    request.mockImplementation(async (path) =>
      path === "/projects/private-one" ? project : {},
    );
    const onUpdate = vi.fn();
    render(
      <HostedUploads
        project={{ ...project, directory: "", image_root: "" }}
        onClose={() => {}}
        onUpdate={onUpdate}
      />,
    );
    const file = new File(["image"], "carrot.png", { type: "image/png" });
    Object.defineProperty(file, "webkitRelativePath", {
      value: "dataset/carrot.png",
    });
    fireEvent.change(screen.getByLabelText("Image files"), {
      target: { files: [file] },
    });
    await waitFor(() => expect(onUpdate).toHaveBeenCalledOnce());
    const upload = request.mock.calls.find(([path]) =>
      path.endsWith("/images"),
    );
    expect(upload?.[0]).toBe("/projects/private-one/images");
    const body = upload?.[2] as FormData;
    expect(body.get("relative_path")).toBe("dataset/carrot.png");
    expect(body.get("file")).toBe(file);
    expect(screen.getByText("1 image uploaded.")).toBeTruthy();
  });
});

it("keeps folder filenames relative to the selected root for COCO matching", async () => {
  request.mockImplementation(async (path) =>
    path === "/projects/private-one" ? project : {},
  );
  render(
    <HostedUploads
      project={{ ...project, directory: "", image_root: "" }}
      onClose={() => {}}
      onUpdate={() => {}}
    />,
  );
  const file = new File(["image"], "carrot.png", { type: "image/png" });
  Object.defineProperty(file, "webkitRelativePath", {
    value: "dataset/validation/carrot.png",
  });
  fireEvent.change(screen.getByLabelText("Image folder"), {
    target: { files: [file] },
  });
  await waitFor(() =>
    expect(screen.getByText("1 image uploaded.")).toBeTruthy(),
  );
  const form = request.mock.calls.find(([path]) =>
    path.endsWith("/images"),
  )?.[2] as FormData;
  expect(form.get("relative_path")).toBe("validation/carrot.png");
});

const ownerSession = {
  user: { id: "owner", email: "owner@example.test", name: "Owner" },
  csrf_token: "csrf",
  usage: null,
};

it("keeps the editor mounted but suspends interaction when identity expires or changes", async () => {
  fetchMock.mockResolvedValue(new Response(JSON.stringify({ mode: "hosted" })));
  request.mockResolvedValue(ownerSession);
  render(<RuntimeApp />);
  const editor = await screen.findByText("Shared editor");
  act(() => window.dispatchEvent(new Event(sessionExpiredEvent)));
  expect(editor.closest(".hosted-editor")?.hasAttribute("inert")).toBe(true);
  request.mockResolvedValue({
    ...ownerSession,
    user: { ...ownerSession.user, id: "other" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Refresh session" }));
  expect(
    await screen.findByText(/Sign in again as owner@example.test/),
  ).toBeTruthy();
  expect(editor.isConnected).toBe(true);
  request.mockResolvedValue(ownerSession);
  fireEvent.click(screen.getByRole("button", { name: "Refresh session" }));
  await waitFor(() =>
    expect(editor.closest(".hosted-editor")?.hasAttribute("inert")).toBe(false),
  );
});

it("leaves manual editing available during a temporary session refresh network failure", async () => {
  fetchMock.mockResolvedValue(new Response(JSON.stringify({ mode: "hosted" })));
  request
    .mockResolvedValueOnce(ownerSession)
    .mockRejectedValue(new TypeError("Offline"));
  render(<RuntimeApp />);
  const editor = await screen.findByText("Shared editor");
  await act(async () => {
    window.dispatchEvent(new Event("focus"));
  });
  expect(editor.closest(".hosted-editor")?.hasAttribute("inert")).toBe(false);
  expect(screen.queryByRole("button", { name: "Refresh session" })).toBeNull();
});

it("ignores an old session refresh that returns after sign-out", async () => {
  fetchMock.mockResolvedValue(new Response(JSON.stringify({ mode: "hosted" })));
  request.mockResolvedValueOnce(ownerSession);
  render(<RuntimeApp />);
  await screen.findByText("Shared editor");
  let finishRefresh!: (value: unknown) => void;
  request.mockImplementation(async (path) =>
    path === "/auth/session"
      ? new Promise((resolve) => {
          finishRefresh = resolve;
        })
      : {},
  );
  act(() => window.dispatchEvent(new Event("focus")));
  fireEvent.click(screen.getByRole("button", { name: "Mock sign out" }));
  await screen.findByRole("link", { name: /Continue with Google/ });
  await act(async () => {
    finishRefresh(ownerSession);
  });
  expect(screen.queryByText("Shared editor")).toBeNull();
});

it("does not invent an image count when the project list provides only summaries", async () => {
  request.mockResolvedValue({
    projects: [
      {
        id: "private-one",
        name: "Carrots",
        created_at: "2026-10-07T00:00:00Z",
      },
    ],
  });
  render(
    <HostedProjects onClose={() => {}} onDelete={() => {}} onOpen={() => {}} />,
  );
  expect(
    await screen.findByRole("button", { name: "Open Carrots" }),
  ).toBeTruthy();
  expect(screen.queryByText("0 images")).toBeNull();
});

it("transfers files over 20 MB in chunks and keeps their original filename", async () => {
  const chunkBytes = 8 * 1024 * 1024;
  const file = new File([new Uint8Array(21 * 1024 * 1024)], "survey.tif", {
    type: "image/tiff",
  });
  request.mockImplementation(async (path, method, body) => {
    if (path.endsWith("/uploads"))
      return { id: "upload-one", chunk_bytes: chunkBytes };
    if (method === "PUT") {
      const offset = Number(path.split("offset=")[1]);
      expect(body).toBeInstanceOf(Blob);
      expect((body as Blob).size).toBeLessThanOrEqual(chunkBytes);
      return { offset: offset + (body as Blob).size };
    }
    return project;
  });
  render(
    <HostedUploads
      project={{ ...project, directory: "", image_root: "" }}
      onClose={() => {}}
      onUpdate={() => {}}
    />,
  );
  fireEvent.change(screen.getByLabelText("Image files"), {
    target: { files: [file] },
  });
  await waitFor(() =>
    expect(screen.getByText("1 image uploaded.")).toBeTruthy(),
  );
  expect(request).toHaveBeenCalledWith(
    "/projects/private-one/uploads",
    "POST",
    { file_name: "survey.tif", size: file.size },
  );
  expect(
    request.mock.calls.filter(([, method]) => method === "PUT"),
  ).toHaveLength(3);
  expect(request).toHaveBeenCalledWith(
    "/projects/private-one/uploads/upload-one/complete",
    "POST",
  );
});
