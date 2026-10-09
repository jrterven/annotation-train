import { beforeEach, vi } from "vitest";
// jsdom has no layout/ResizeObserver. Canvas/layout tests install their own
// measurement callbacks; tests that mock the canvas only need its lifecycle.
beforeEach(() => {
  if (typeof window !== "undefined" && !window.ResizeObserver) {
    vi.stubGlobal(
      "ResizeObserver",
      class {
        observe() {}
        unobserve() {}
        disconnect() {}
      },
    );
  }
});
