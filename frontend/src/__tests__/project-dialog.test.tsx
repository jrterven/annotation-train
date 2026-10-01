// @vitest-environment jsdom
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "../api";
import { ProjectDialog } from "../ProjectDialog";
import type { Project } from "../types";

vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api")>()),
  api: vi.fn(),
}));
const request = vi.mocked(api);
const project: Project = {
  name: "research",
  directory: "/work/research",
  image_root: "/work/research/photos",
  categories: [],
  images: [],
};
type Folder = {
  path: string;
  parent: string;
  directories: { name: string; path: string }[];
  files: { name: string; path: string }[];
};
const folders: Record<string, Folder> = {
  "/work/research": {
    path: "/work/research",
    parent: "/work",
    directories: [{ name: "photos", path: "/work/research/photos" }],
    files: [],
  },
  "/work/research/photos": {
    path: "/work/research/photos",
    parent: "/work/research",
    directories: [{ name: "second", path: "/work/research/photos/second" }],
    files: [{ name: "sample.png", path: "/work/research/photos/sample.png" }],
  },
  "/work/research/photos/second": {
    path: "/work/research/photos/second",
    parent: "/work/research/photos",
    directories: [],
    files: [
      { name: "sample.png", path: "/work/research/photos/second/sample.png" },
    ],
  },
};

beforeEach(() => {
  request.mockReset();
  request.mockImplementation(async (path, method) => {
    if (path.startsWith("/browse?")) {
      const requested = new URLSearchParams(path.split("?")[1]).get("path")!;
      const result = folders[requested];
      if (!result) throw new Error(`Unexpected folder: ${requested}`);
      return structuredClone(result);
    }
    if (method === "POST" && ["/projects/open", "/coco/import"].includes(path))
      return structuredClone(project);
    throw new Error(`Unexpected request: ${method} ${path}`);
  });
});
afterEach(cleanup);

function openDialog() {
  const onOpen = vi.fn();
  render(<ProjectDialog onClose={vi.fn()} onOpen={onOpen} />);
  fireEvent.change(screen.getByLabelText(/^Project folder/), {
    target: { value: project.directory },
  });
  return onOpen;
}
function submitted(path = "/projects/open") {
  return request.mock.calls.find((call) => call[0] === path)?.[2];
}

it("starts image browsing at the project folder and submits the explicitly chosen subfolder", async () => {
  const onOpen = openDialog();
  const imageInput = screen.getByLabelText("Image folder") as HTMLInputElement;
  expect(imageInput.value).toBe("");
  expect(imageInput.placeholder).toBe(project.directory);
  expect(
    (
      screen.getByRole("button", {
        name: "Select individual images",
      }) as HTMLButtonElement
    ).disabled,
  ).toBe(true);
  fireEvent.click(screen.getByTitle("Browse image folder"));
  expect(request).toHaveBeenCalledWith(
    `/browse?path=${encodeURIComponent(project.directory)}`,
  );
  const browser = await screen.findByRole("dialog", { name: "File browser" });
  fireEvent.click(
    await within(browser).findByRole("button", { name: "photos" }),
  );
  await waitFor(() =>
    expect(
      (within(browser).getByLabelText("Folder path") as HTMLInputElement).value,
    ).toBe(project.image_root),
  );
  fireEvent.click(within(browser).getByRole("button", { name: "Use folder" }));
  expect(imageInput.value).toBe(project.image_root);
  expect(
    (
      screen.getByRole("button", {
        name: "Select individual images",
      }) as HTMLButtonElement
    ).disabled,
  ).toBe(false);
  fireEvent.click(screen.getByRole("button", { name: "Open project" }));
  await waitFor(() => expect(onOpen).toHaveBeenCalledWith(project));
  expect(submitted()).toEqual({
    directory: project.directory,
    image_root: project.image_root,
    recursive: true,
  });
});

it("reopens an existing project without fabricating an image root or browsing a default child", async () => {
  const onOpen = openDialog();
  fireEvent.click(screen.getByRole("button", { name: "Open project" }));
  await waitFor(() => expect(onOpen).toHaveBeenCalledWith(project));
  expect(submitted()).toEqual({
    directory: project.directory,
    recursive: true,
  });
  expect(request.mock.calls).toHaveLength(1);
  expect(submitted()).not.toHaveProperty("image_root");
});

it("requires an explicit image folder before importing COCO", async () => {
  const onOpen = openDialog();
  fireEvent.click(screen.getByRole("button", { name: "Import COCO" }));
  fireEvent.change(screen.getByLabelText(/^COCO annotations/), {
    target: { value: "/work/annotations.json" },
  });
  const imageInput = screen.getByLabelText("Image folder") as HTMLInputElement;
  expect(imageInput.required).toBe(true);
  await act(async () =>
    fireEvent.click(screen.getByRole("button", { name: "Open project" })),
  );
  expect(imageInput.validity.valueMissing).toBe(true);
  expect(submitted("/coco/import")).toBeUndefined();
  expect(onOpen).not.toHaveBeenCalled();

  fireEvent.change(imageInput, { target: { value: project.image_root } });
  fireEvent.click(screen.getByRole("button", { name: "Open project" }));
  await waitFor(() => expect(onOpen).toHaveBeenCalledWith(project));
  expect(submitted("/coco/import")).toEqual({
    directory: project.directory,
    image_root: project.image_root,
    json_path: "/work/annotations.json",
  });
});

it("keeps selected files relative to the chosen root across subfolders with repeated names", async () => {
  const onOpen = openDialog();
  fireEvent.change(screen.getByLabelText("Image folder"), {
    target: { value: project.image_root },
  });
  fireEvent.click(
    screen.getByRole("button", { name: "Select individual images" }),
  );
  const browser = await screen.findByRole("dialog", { name: "File browser" });
  fireEvent.click(
    await within(browser).findByRole("button", { name: "sample.png" }),
  );
  fireEvent.click(within(browser).getByRole("button", { name: "second" }));
  await waitFor(() =>
    expect(
      (within(browser).getByLabelText("Folder path") as HTMLInputElement).value,
    ).toBe(`${project.image_root}/second`),
  );
  fireEvent.click(
    await within(browser).findByRole("button", { name: "sample.png" }),
  );
  fireEvent.click(
    within(browser).getByRole("button", { name: "Select" }),
  );
  expect(
    (screen.getByLabelText("Image folder") as HTMLInputElement).value,
  ).toBe(project.image_root);
  expect(
    screen.getByRole("button", { name: "2 images selected" }),
  ).toBeDefined();
  fireEvent.click(screen.getByRole("button", { name: "Open project" }));
  await waitFor(() => expect(onOpen).toHaveBeenCalledWith(project));
  expect(submitted()).toEqual({
    directory: project.directory,
    image_root: project.image_root,
    files: ["sample.png", "second/sample.png"],
    recursive: true,
  });
});
