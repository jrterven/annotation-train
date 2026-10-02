import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  Check,
  ImagePlus,
  LoaderCircle,
  RotateCcw,
  Upload,
  X,
} from "lucide-react";
import { errorText } from "./api";
import type { XY } from "./types";

export type ReferenceBox = [number, number, number, number];
export type VisualExample = {
  name: string;
  dataUrl: string;
  base64: string;
  width: number;
  height: number;
  box?: ReferenceBox;
};
const MAX_BYTES = 10 * 1024 * 1024;
const MAX_PIXELS = 16_000_000;
const abortError = () => new DOMException("Upload canceled", "AbortError");

export async function loadVisualFile(
  file: File,
  signal: AbortSignal,
): Promise<VisualExample> {
  if (signal.aborted) throw abortError();
  if (!file.size || file.size > MAX_BYTES)
    throw new Error("Choose a nonempty image under 10 MiB.");
  if (
    !(file.type
      ? ["image/png", "image/jpeg", "image/webp"].includes(file.type)
      : /\.(png|jpe?g|webp)$/i.test(file.name))
  )
    throw new Error("Choose a PNG, JPEG, or WebP image.");
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    const cancel = () => {
      reader.abort();
      reject(abortError());
    };
    signal.addEventListener("abort", cancel, { once: true });
    reader.onloadend = () => signal.removeEventListener("abort", cancel);
    reader.onerror = () => reject(new Error("Could not read this image."));
    reader.onabort = () => reject(abortError());
    reader.onload = () =>
      typeof reader.result === "string"
        ? resolve(reader.result)
        : reject(new Error("Could not read this image."));
    reader.readAsDataURL(file);
  });
  if (signal.aborted) throw abortError();
  const base64 = dataUrl.slice(dataUrl.indexOf(",") + 1);
  const header = atob(base64.slice(0, 32));
  const png = header.startsWith("\x89PNG\r\n\x1a\n");
  const jpeg = header.startsWith("\xff\xd8\xff");
  const webp = header.startsWith("RIFF") && header.slice(8, 12) === "WEBP";
  if (!png && !jpeg && !webp)
    throw new Error("This file is not a PNG, JPEG, or WebP image.");
  const dimensions = await new Promise<{ width: number; height: number }>(
    (resolve, reject) => {
      const image = new Image();
      const cleanup = () => signal.removeEventListener("abort", cancel);
      const cancel = () => {
        cleanup();
        image.onload = image.onerror = null;
        image.src = "";
        reject(abortError());
      };
      signal.addEventListener("abort", cancel, { once: true });
      image.onerror = () => {
        cleanup();
        reject(new Error("Could not decode this image."));
      };
      image.onload = () => {
        cleanup();
        const width = image.naturalWidth,
          height = image.naturalHeight;
        if (!width || !height || width * height > MAX_PIXELS)
          reject(new Error("Choose an image with at most 16 megapixels."));
        else resolve({ width, height });
      };
      // Browser image dimensions and rendering use EXIF orientation. The backend
      // applies the same orientation before interpreting original-pixel crops.
      image.src = dataUrl;
    },
  );
  if (signal.aborted) throw abortError();
  return { name: file.name, dataUrl, base64, ...dimensions };
}

export function referencePoint(
  rect: Pick<DOMRect, "left" | "top" | "width" | "height">,
  width: number,
  height: number,
  x: number,
  y: number,
): XY {
  return [
    Math.max(0, Math.min(width, ((x - rect.left) * width) / rect.width)),
    Math.max(0, Math.min(height, ((y - rect.top) * height) / rect.height)),
  ];
}
const cropBetween = (a: XY, b: XY): ReferenceBox => [
  Math.min(a[0], b[0]),
  Math.min(a[1], b[1]),
  Math.max(a[0], b[0]),
  Math.max(a[1], b[1]),
];

type Props = {
  value: VisualExample | null;
  scope: string;
  onChange: (value: VisualExample | null) => void;
  onPickStart: () => void;
  onOpenChange: (open: boolean) => void;
};
export default function VisualReference({
  value,
  scope,
  onChange,
  onPickStart,
  onOpenChange,
}: Props) {
  const input = useRef<HTMLInputElement>(null);
  const uploadButton = useRef<HTMLButtonElement>(null);
  const previewImage = useRef<HTMLImageElement>(null);
  const dialog = useRef<HTMLElement>(null);
  const operation = useRef<AbortController | null>(null);
  const currentScope = useRef(scope);
  currentScope.current = scope;
  const openedScope = useRef(scope);
  const drag = useRef<{ start: XY; previous?: ReferenceBox } | null>(null);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState<VisualExample | null>(null);
  const [error, setError] = useState("");
  const validCrop =
    !!preview?.box &&
    preview.box[2] - preview.box[0] >= 1 &&
    preview.box[3] - preview.box[1] >= 1;
  const tooltipId = useId();
  function close() {
    operation.current?.abort();
    operation.current = null;
    drag.current = null;
    setOpen(false);
    setLoading(false);
    if (open) uploadButton.current?.focus();
  }
  useEffect(() => {
    close();
    return () => operation.current?.abort();
  }, [scope]);
  useEffect(() => {
    onOpenChange(open);
    if (open) dialog.current?.focus();
  }, [open, onOpenChange]);
  async function choose(file: File) {
    operation.current?.abort();
    const controller = new AbortController();
    operation.current = controller;
    openedScope.current = scope;
    onPickStart();
    setPreview(null);
    setError("");
    setOpen(true);
    setLoading(true);
    try {
      const loaded = await loadVisualFile(file, controller.signal);
      if (operation.current === controller && currentScope.current === scope)
        setPreview(loaded);
    } catch (e) {
      if (operation.current === controller && !controller.signal.aborted)
        setError(errorText(e));
    } finally {
      if (operation.current === controller) setLoading(false);
    }
  }
  function at(event: React.PointerEvent): XY | null {
    const rect = previewImage.current?.getBoundingClientRect();
    return rect && rect.width && rect.height && preview
      ? referencePoint(
          rect,
          preview.width,
          preview.height,
          event.clientX,
          event.clientY,
        )
      : null;
  }
  return (
    <div className="visual-reference">
      <input
        ref={input}
        type="file"
        hidden
        aria-label="Visual example file"
        accept="image/png,image/jpeg,image/webp"
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (file) void choose(file);
        }}
      />
      <span className="tool-control">
        <button
          ref={uploadButton}
          type="button"
          aria-label="Upload visual example"
          aria-describedby={tooltipId}
          onClick={() => input.current?.click()}
        >
          <ImagePlus size={17} />
        </button>
        <span className="tool-tooltip" role="tooltip" id={tooltipId}>
          Upload visual example
        </span>
      </span>
      {value && (
        <div className="visual-reference-chip">
          <button
            type="button"
            className="visual-reference-edit"
            aria-label="Edit visual example"
            title={`${value.name} · Visual example · Experimental`}
            onClick={() => {
              openedScope.current = scope;
              setPreview({
                ...value,
                box: value.box ? [...value.box] : undefined,
              });
              setError("");
              setLoading(false);
              setOpen(true);
            }}
          >
            <img src={value.dataUrl} alt="" />
            <span>{value.name}</span>
          </button>
          <button
            type="button"
            className="visual-reference-remove"
            aria-label="Remove visual example"
            title="Remove visual example"
            onClick={() => {
              close();
              onChange(null);
            }}
          >
            <X size={13} />
          </button>
        </div>
      )}
      {open &&
        createPortal(
          <div
            className="modal-backdrop"
            onKeyDown={(event) => {
              event.stopPropagation();
              if (event.key === "Escape") {
                event.preventDefault();
                close();
              }
              if (event.key === "Tab") {
                const items =
                  dialog.current?.querySelectorAll<HTMLButtonElement>(
                    "button:not(:disabled)",
                  );
                if (items?.length) {
                  const first = items[0],
                    last = items[items.length - 1];
                  if (
                    event.shiftKey &&
                    (document.activeElement === first ||
                      document.activeElement === dialog.current)
                  ) {
                    event.preventDefault();
                    last.focus();
                  } else if (
                    !event.shiftKey &&
                    document.activeElement === last
                  ) {
                    event.preventDefault();
                    first.focus();
                  }
                }
              }
            }}
          >
            <section
              ref={dialog}
              tabIndex={-1}
              className="modal visual-reference-modal"
              role="dialog"
              aria-modal="true"
              aria-label="Visual example"
            >
              <div className="modal-heading">
                <div>
                  <h2>
                    Visual example{" "}
                    <small title="Cross-image matching is experimental. Results may vary; review every proposal.">
                      Experimental
                    </small>
                  </h2>
                </div>
                <button
                  className="icon-button"
                  type="button"
                  aria-label="Close visual example"
                  onClick={close}
                >
                  <X size={19} />
                </button>
              </div>
              {loading ? (
                <div className="visual-reference-loading">
                  <LoaderCircle className="spin" size={23} />
                  Loading image…
                </div>
              ) : preview ? (
                <>
                  <p className="field-note">
                    Draw a box around one object. Use a photo with some
                    surrounding background.
                  </p>
                  <p className="field-note">
                    May return unrelated objects. Review proposals before
                    accepting.
                  </p>
                  <div className="visual-reference-stage">
                    <div
                      className="visual-crop-surface"
                      onPointerDown={(event) => {
                        if (event.button !== 0) return;
                        const point = at(event);
                        if (!point) return;
                        event.preventDefault();
                        drag.current = { start: point, previous: preview.box };
                        event.currentTarget.setPointerCapture(event.pointerId);
                      }}
                      onPointerMove={(event) => {
                        const point = at(event);
                        if (drag.current && point)
                          setPreview({
                            ...preview,
                            box: cropBetween(drag.current.start, point),
                          });
                      }}
                      onPointerUp={(event) => {
                        const point = at(event);
                        if (drag.current && point) {
                          const box = cropBetween(drag.current.start, point);
                          setPreview({
                            ...preview,
                            box:
                              box[2] - box[0] >= 1 && box[3] - box[1] >= 1
                                ? box
                                : drag.current.previous,
                          });
                        }
                        drag.current = null;
                        if (
                          event.currentTarget.hasPointerCapture(event.pointerId)
                        )
                          event.currentTarget.releasePointerCapture(
                            event.pointerId,
                          );
                      }}
                      onPointerCancel={() => {
                        if (drag.current)
                          setPreview({
                            ...preview,
                            box: drag.current.previous,
                          });
                        drag.current = null;
                      }}
                    >
                      <img
                        ref={previewImage}
                        src={preview.dataUrl}
                        alt="Visual example preview"
                        draggable={false}
                      />
                      {preview.box && (
                        <div
                          className="visual-crop-box"
                          style={{
                            left: `${(preview.box[0] / preview.width) * 100}%`,
                            top: `${(preview.box[1] / preview.height) * 100}%`,
                            width: `${((preview.box[2] - preview.box[0]) / preview.width) * 100}%`,
                            height: `${((preview.box[3] - preview.box[1]) / preview.height) * 100}%`,
                          }}
                        />
                      )}
                    </div>
                  </div>
                  <div className="visual-reference-info">
                    <span title={preview.name}>{preview.name}</span>
                    <span>
                      {preview.width} × {preview.height}
                    </span>
                  </div>
                </>
              ) : null}
              {error && (
                <p className="form-error" role="alert">
                  {error}
                </p>
              )}
              <footer className="modal-footer">
                <div className="visual-reference-actions">
                  <button
                    className="text-button"
                    type="button"
                    onClick={() => input.current?.click()}
                  >
                    <Upload size={14} />
                    Choose image
                  </button>
                  {preview?.box && (
                    <button
                      className="text-button"
                      type="button"
                      onClick={() => setPreview({ ...preview, box: undefined })}
                    >
                      <RotateCcw size={14} />
                      Reset crop
                    </button>
                  )}
                </div>
                <button
                  className="button primary"
                  type="button"
                  disabled={!preview || !validCrop || loading}
                  onClick={() => {
                    if (preview && validCrop && openedScope.current === scope) {
                      onChange(preview);
                      close();
                    }
                  }}
                >
                  <Check size={15} />
                  Use example
                </button>
              </footer>
            </section>
          </div>,
          document.body,
        )}
    </div>
  );
}
