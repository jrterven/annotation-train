// @vitest-environment jsdom
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import ResizableWorkspace, { PanelResizeHandle } from "../ResizableWorkspace";
import AccountAvatar from "../AccountAvatar";

let measure: (entries: { contentRect: { width: number } }[]) => void;
beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal(
    "ResizeObserver",
    class {
      constructor(callback: typeof measure) {
        measure = callback;
      }
      observe() {}
      disconnect() {}
    },
  );
  Object.defineProperty(HTMLElement.prototype, "setPointerCapture", {
    configurable: true,
    value: vi.fn(),
  });
  Object.defineProperty(HTMLElement.prototype, "releasePointerCapture", {
    configurable: true,
    value: vi.fn(),
  });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
function Workspace({ rightVisible = true }: { rightVisible?: boolean }) {
  return (
    <ResizableWorkspace rightVisible={rightVisible}>
      <PanelResizeHandle side="left" />
      <main />
      {rightVisible && <PanelResizeHandle side="right" />}
    </ResizableWorkspace>
  );
}
const width = (element: HTMLElement) =>
  Number(element.getAttribute("aria-valuenow"));
it("resizes both panels by keyboard, persists preferences, and reserves canvas space when the window shrinks", () => {
  const { unmount } = render(<Workspace />);
  act(() => measure([{ contentRect: { width: 1400 } }]));
  const left = screen.getByRole("separator", { name: "Resize images panel" });
  const right = screen.getByRole("separator", {
    name: "Resize annotations panel",
  });
  fireEvent.keyDown(left, { key: "ArrowRight" });
  fireEvent.keyDown(right, { key: "ArrowLeft" });
  expect(width(left)).toBe(251);
  expect(width(right)).toBe(283);
  act(() => measure([{ contentRect: { width: 790 } }]));
  expect(width(left) + width(right)).toBeLessThanOrEqual(790 - 320 - 12);
  act(() => measure([{ contentRect: { width: 1400 } }]));
  expect(width(right)).toBe(283);
  unmount();
  render(<Workspace />);
  act(() => measure([{ contentRect: { width: 1400 } }]));
  expect(
    width(screen.getByRole("separator", { name: "Resize images panel" })),
  ).toBe(251);
  expect(
    width(screen.getByRole("separator", { name: "Resize annotations panel" })),
  ).toBe(283);
});
it("captures drags, bounds extreme moves, cancels safely and resets on double click", () => {
  render(<Workspace />);
  act(() => measure([{ contentRect: { width: 1400 } }]));
  const left = screen.getByRole("separator", { name: "Resize images panel" });
  fireEvent.pointerDown(left, { button: 0, pointerId: 4, clientX: 235 });
  fireEvent.pointerMove(left, { pointerId: 5, clientX: 500 });
  expect(width(left)).toBe(235);
  fireEvent.pointerMove(left, { pointerId: 4, clientX: 335 });
  expect(width(left)).toBe(335);
  fireEvent.pointerMove(left, { pointerId: 4, clientX: 9000 });
  expect(width(left)).toBe(480);
  fireEvent.pointerCancel(left, { pointerId: 4 });
  expect(width(left)).toBe(235);
  fireEvent.keyDown(left, { key: "Home" });
  expect(width(left)).toBe(176);
  fireEvent.doubleClick(left);
  expect(width(left)).toBe(235);
});
it("restores the annotations width after hiding its panel and tolerates disabled browser storage", () => {
  const storage = vi
    .spyOn(Storage.prototype, "setItem")
    .mockImplementation(() => {
      throw new Error("blocked");
    });
  const { rerender } = render(<Workspace />);
  act(() => measure([{ contentRect: { width: 1400 } }]));
  fireEvent.keyDown(
    screen.getByRole("separator", { name: "Resize annotations panel" }),
    { key: "ArrowLeft" },
  );
  rerender(<Workspace rightVisible={false} />);
  expect(
    screen.queryByRole("separator", { name: "Resize annotations panel" }),
  ).toBeNull();
  rerender(<Workspace />);
  expect(
    width(screen.getByRole("separator", { name: "Resize annotations panel" })),
  ).toBe(283);
  storage.mockRestore();
});
it("uses the Google picture without a referrer and falls back when unavailable or broken", () => {
  const user = {
    id: "one",
    email: "person@example.test",
    name: "Person",
    picture: "https://lh3.googleusercontent.com/a/profile",
  };
  const { container, rerender } = render(<AccountAvatar user={user} />);
  const image = container.querySelector("img")!;
  expect(image.getAttribute("referrerpolicy")).toBe("no-referrer");
  fireEvent.error(image);
  expect(container.querySelector("img")).toBeNull();
  expect(container.querySelector("svg")).toBeTruthy();
  rerender(
    <AccountAvatar
      user={{ ...user, picture: "https://lh3.googleusercontent.com/a/new" }}
    />,
  );
  expect(container.querySelector("img")).toBeTruthy();
  rerender(<AccountAvatar user={{ ...user, picture: null }} />);
  expect(container.querySelector("svg")).toBeTruthy();
});
