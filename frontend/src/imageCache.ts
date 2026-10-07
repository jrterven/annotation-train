// Decoded originals live only in this tab. Never use browser persistent storage
// for private pixels; API authorization and no-store response headers stay intact.
export const imageCacheBudget = 256 * 1024 * 1024;
type Entry = {
  image: HTMLImageElement;
  promise: Promise<HTMLImageElement>;
  reject: (reason: unknown) => void;
  bytes: number;
  ready: boolean;
};
export class ImageMemoryCache {
  private entries = new Map<string, Entry>();
  constructor(
    private maxBytes = imageCacheBudget,
    private maxEntries = 4,
  ) {}
  private drop(url: string) {
    const entry = this.entries.get(url);
    if (!entry) return;
    this.entries.delete(url);
    if (!entry.ready) {
      entry.image.onload = entry.image.onerror = null;
      entry.image.removeAttribute("src");
      entry.reject(new DOMException("Image load cancelled", "AbortError"));
    }
  }
  clear() {
    for (const url of this.entries.keys()) this.drop(url);
  }
  peek(url: string) {
    const entry = this.entries.get(url);
    return entry?.ready ? entry.image : undefined;
  }
  load(url: string): Promise<HTMLImageElement> {
    const existing = this.entries.get(url);
    if (existing) {
      this.entries.delete(url);
      this.entries.set(url, existing);
      return existing.promise;
    }
    while (this.entries.size >= this.maxEntries)
      this.drop(this.entries.keys().next().value!);
    const image = new window.Image();
    let resolve!: (image: HTMLImageElement) => void;
    let reject!: (reason: unknown) => void;
    const promise = new Promise<HTMLImageElement>((yes, no) => {
      resolve = yes;
      reject = no;
    });
    const entry: Entry = { image, promise, reject, bytes: 0, ready: false };
    this.entries.set(url, entry);
    image.onload = async () => {
      try {
        if (image.decode) await image.decode();
        if (this.entries.get(url) !== entry) return;
        entry.ready = true;
        entry.bytes = image.naturalWidth * image.naturalHeight * 4;
        image.onload = image.onerror = null;
        // The active image can exceed the cache budget and still display.
        // Only retained inactive copies are bounded; no image size is rejected.
        if (entry.bytes > this.maxBytes) this.entries.delete(url);
        let total = [...this.entries.values()].reduce(
          (n, item) => n + item.bytes,
          0,
        );
        for (const [key, item] of this.entries) {
          if (total <= this.maxBytes) break;
          if (key === url || !item.ready) continue;
          total -= item.bytes;
          this.drop(key);
        }
        resolve(image);
      } catch (error) {
        if (this.entries.get(url) === entry) this.entries.delete(url);
        reject(error);
      }
    };
    image.onerror = () => {
      if (this.entries.get(url) === entry) this.entries.delete(url);
      reject(new Error("Could not load image"));
    };
    image.src = url;
    return promise;
  }
}
export const sourceImages = new ImageMemoryCache();
