import { useState } from "react";
import { Download, X, LoaderCircle } from "lucide-react";
import { api, assetURL, errorText } from "./api";
import type { Task } from "./types";
import IconButton from "./IconButton";

type Issue = { file_name: string; annotation_id?: string; reason: string };
type Preview = {
  snapshot: string;
  image_count: number;
  annotation_count: number;
  warnings: Issue[];
  errors: Issue[];
};
export default function ExportDialog(p: {
  task: Task;
  flush: () => Promise<void>;
  onClose: () => void;
  onReady: (url: string, name: string) => void;
}) {
  const [format, setFormat] = useState<"coco" | "yolo">("coco");
  const [task, setTask] = useState<Task | "all">(p.task);
  const [images, setImages] = useState(false),
    [empty, setEmpty] = useState(false);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [lossy, setLossy] = useState(false),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  function invalidate() {
    setPreview(null);
    setLossy(false);
    setError("");
  }
  async function run() {
    setBusy(true);
    setError("");
    try {
      await p.flush();
      if (format === "coco") {
        const result = await api<{ file_name?: string; export_id: string }>(
          "/coco/export",
          "POST",
          { task, unique: true },
        );
        p.onReady(
          `/coco/download/${result.export_id}`,
          result.file_name || "annotations.coco.json",
        );
        p.onClose();
      } else {
        const options = { task, include_images: images, include_empty: empty };
        if (!preview) {
          setPreview(await api<Preview>("/yolo/preview", "POST", options));
          return;
        }
        const result = await api<{ export_id: string; file_name: string }>(
          "/yolo/export",
          "POST",
          { ...options, snapshot: preview.snapshot, allow_lossy: lossy },
        );
        p.onReady(`/yolo/download/${result.export_id}`, result.file_name);
        p.onClose();
      }
    } catch (e) {
      setError(errorText(e));
      setPreview(null);
      setLossy(false);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="modal-backdrop">
      <section
        className="modal export-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Export annotations"
      >
        <div className="modal-heading">
          <h2>Export annotations</h2>
          <IconButton
            icon={X}
            title="Close export"
            disabled={busy}
            onClick={p.onClose}
          />
        </div>
        <div className="export-fields">
          <label>
            Format
            <select
              aria-label="Export format"
              value={format}
              disabled={busy}
              onChange={(e) => {
                setFormat(e.target.value as "coco" | "yolo");
                if (task === "all") setTask(p.task);
                invalidate();
              }}
            >
              <option value="coco">COCO</option>
              <option value="yolo">YOLO</option>
            </select>
          </label>
          <label>
            Task
            <select
              aria-label="Export task"
              value={task}
              disabled={busy}
              onChange={(e) => {
                setTask(e.target.value as Task | "all");
                invalidate();
              }}
            >
              <option value="segmentation">Segmentation</option>
              <option value="detection">Detection</option>
              {format === "coco" && (
                <option value="all">Whole project (mixed)</option>
              )}
            </select>
          </label>
          <p className="field-note">
            Only confirmed annotations are exported. Pending proposals and SAM
            adjustments are excluded.
          </p>
          {format === "yolo" && (
            <>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={images}
                  disabled={busy}
                  onChange={(e) => {
                    setImages(e.target.checked);
                    invalidate();
                  }}
                />
                Include images
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={empty}
                  disabled={busy}
                  onChange={(e) => {
                    setEmpty(e.target.checked);
                    invalidate();
                  }}
                />
                Include images without annotations for this task
              </label>
              <p className="field-note">
                ZIP with labels, classes and export report. No automatic
                train/validation split.
              </p>
            </>
          )}
          {preview && (
            <div className="export-preview">
              <p>
                {preview.image_count} images · {preview.annotation_count}{" "}
                confirmed annotations
              </p>
              {!!preview.errors.length && (
                <>
                  <strong>Resolve these errors before exporting</strong>
                  <ul>
                    {preview.errors.map((issue, i) => (
                      <li key={i}>
                        {issue.file_name}
                        {issue.annotation_id ? ` · ${issue.annotation_id}` : ""}
                        : {issue.reason}
                      </li>
                    ))}
                  </ul>
                </>
              )}
              {!!preview.warnings.length && (
                <>
                  <strong>Polygon approximations</strong>
                  <ul>
                    {preview.warnings.map((issue, i) => (
                      <li key={i}>
                        {issue.file_name} · {issue.annotation_id}:{" "}
                        {issue.reason}
                      </li>
                    ))}
                  </ul>
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={lossy}
                      onChange={(e) => setLossy(e.target.checked)}
                      disabled={busy}
                    />
                    Accept approximations in this export. Original masks stay
                    unchanged.
                  </label>
                </>
              )}
            </div>
          )}
          {error && (
            <p role="alert" className="part-error">
              {error}
            </p>
          )}
        </div>
        <footer className="modal-footer">
          <span />
          <button
            className="button primary"
            disabled={
              busy ||
              (!!preview &&
                (!!preview.errors.length ||
                  !preview.image_count ||
                  (!!preview.warnings.length && !lossy)))
            }
            onClick={() => void run()}
          >
            {busy ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <Download size={16} />
            )}
            {format === "yolo" && !preview ? "Review export" : "Export"}
          </button>
        </footer>
      </section>
    </div>
  );
}
export function downloadExport(path: string, name: string) {
  const link = document.createElement("a");
  link.href = assetURL(path);
  link.download = name;
  link.click();
}
