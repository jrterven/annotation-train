// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ImageMemoryCache, sourceImages } from "../imageCache";
import { configureApi, selectProject, setCsrfToken } from "../api";

class TestImage {
  static made: TestImage[] = [];
  src = "";
  naturalWidth = 10;
  naturalHeight = 10;
  onload: (() => Promise<void>) | null = null;
  onerror: (() => void) | null = null;
  decode = vi.fn(async () => {});
  removeAttribute() {
    this.src = "";
  }
  constructor() {
    TestImage.made.push(this);
  }
}
beforeEach(() => {
  TestImage.made = [];
  vi.stubGlobal("Image", TestImage);
});
afterEach(() => {
  sourceImages.clear();
  vi.unstubAllGlobals();
});
it("shares in-flight loads and reuses decoded originals on returning to an image", async () => {
  const cache = new ImageMemoryCache();
  const first = cache.load("/a");
  const again = cache.load("/a");
  expect(again).toBe(first);
  await TestImage.made[0].onload!();
  const image = await first;
  expect(await cache.load("/a")).toBe(image);
  expect(TestImage.made).toHaveLength(1);
  expect(TestImage.made[0].decode).toHaveBeenCalledOnce();
});
it("evicts older decoded images by bytes while allowing an oversized active image", async () => {
  const cache = new ImageMemoryCache(600);
  for (const url of ["/a", "/b"]) {
    const task = cache.load(url);
    await TestImage.made.at(-1)!.onload!();
    await task;
  }
  expect(cache.peek("/a")).toBeUndefined();
  expect(cache.peek("/b")).toBeTruthy();
  const large = cache.load("/large");
  TestImage.made.at(-1)!.naturalWidth = 1000;
  await TestImage.made.at(-1)!.onload!();
  expect(await large).toBeTruthy();
  expect(cache.peek("/large")).toBeUndefined();
});
it("clears private pixels on project or session changes and cancels pending loads", async () => {
  configureApi("hosted", "one");
  selectProject({ id: "one" });
  const first = sourceImages.load("/a");
  await TestImage.made[0].onload!();
  await first;
  selectProject({ id: "two" });
  expect(sourceImages.peek("/a")).toBeUndefined();
  const pending = sourceImages.load("/b");
  const cancelled = expect(pending).rejects.toMatchObject({
    name: "AbortError",
  });
  setCsrfToken(null);
  await cancelled;
  expect(TestImage.made[1].src).toBe("");
  expect(sourceImages.peek("/b")).toBeUndefined();
});
