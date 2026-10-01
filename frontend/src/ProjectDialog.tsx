import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowUp,
  Check,
  ChevronRight,
  FileImage,
  FileJson,
  Folder,
  FolderOpen,
  LoaderCircle,
  X,
} from "lucide-react";
import { api, errorText } from "./api";
import type { Project } from "./types";
type BrowseResult = {
  path: string;
  parent: string | null;
  directories: { name: string; path: string }[];
  files: { name: string; path: string }[];
};
export function FileBrowser({
  kind,
  initial,
  onCancel,
  onChoose,
}: {
  kind: "directory" | "json" | "images";
  initial: string;
  onCancel: () => void;
  onChoose: (paths: string[], base: string, initialRoot: string) => void;
}) {
  const [data, setData] = useState<BrowseResult | null>(null);
  const [path, setPath] = useState(initial);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const initialRoot = useRef("");
  async function browse(target: string) {
    setLoading(true);
    setError("");
    try {
      const next = await api<BrowseResult>(
        `/browse?path=${encodeURIComponent(target)}`,
      );
      setData(next);
      setPath(next.path);
      if (!initialRoot.current) initialRoot.current = next.path;
    } catch (e) {
      setError(errorText(e));
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    void browse(initial);
  }, []);
  function toggle(file: string) {
    setSelected((old) =>
      kind === "json"
        ? [file]
        : old.includes(file)
          ? old.filter((p) => p !== file)
          : [...old, file],
    );
  }
  return (
    <div className="modal-backdrop nested">
      <section
        className="modal file-browser"
        role="dialog"
        aria-modal="true"
        aria-label="File browser"
      >
        <div className="modal-heading">
          <div>
            <h2>
              {kind === "directory"
                ? "Select a folder"
                : kind === "json"
                  ? "Select COCO JSON"
                  : "Select images"}
            </h2>
          </div>
          <button
            className="icon-button"
            onClick={onCancel}
            aria-label="Close file browser"
          >
            <X size={19} />
          </button>
        </div>
        <form
          className="path-bar"
          onSubmit={(e) => {
            e.preventDefault();
            void browse(path);
          }}
        >
          <button
            type="button"
            className="icon-button"
            disabled={!data?.parent || loading}
            onClick={() => data?.parent && void browse(data.parent)}
            title="Parent folder"
          >
            <ArrowUp size={17} />
          </button>
          <input
            aria-label="Folder path"
            value={path}
            onChange={(e) => setPath(e.target.value)}
          />
          <button type="submit" className="icon-button" title="Go to path">
            <ChevronRight size={18} />
          </button>
        </form>
        <div className="browser-list">
          {loading && (
            <div className="empty-panel">
              <LoaderCircle className="spin" size={24} />
            </div>
          )}
          {!loading && data && (
            <>
              {data.directories.map((dir) => (
                <button
                  className="browser-row"
                  key={dir.path}
                  onClick={() => void browse(dir.path)}
                >
                  <Folder size={18} />
                  <span>{dir.name}</span>
                  <ChevronRight size={15} />
                </button>
              ))}
              {data.files
                .filter((f) =>
                  kind === "json"
                    ? f.name.toLowerCase().endsWith(".json")
                    : kind === "images"
                      ? !f.name.toLowerCase().endsWith(".json")
                      : false,
                )
                .map((file) => (
                  <button
                    className={`browser-row ${selected.includes(file.path) ? "chosen" : ""}`}
                    key={file.path}
                    onClick={() => toggle(file.path)}
                  >
                    {kind === "json" ? (
                      <FileJson size={18} />
                    ) : (
                      <FileImage size={18} />
                    )}
                    <span>{file.name}</span>
                    {selected.includes(file.path) && <Check size={16} />}
                  </button>
                ))}
              {!data.directories.length && !data.files.length && (
                <div className="empty-panel">This folder is empty.</div>
              )}
            </>
          )}
        </div>
        {error && <p className="form-error">{error}</p>}
        <footer className="modal-footer">
          <span className="muted">
            {kind === "images"
              ? `${selected.length} selected`
              : kind === "directory"
                ? "Select the current folder."
                : "COCO segmentation format."}
          </span>
          <button
            className="button primary"
            disabled={
              loading || !data || (kind !== "directory" && !selected.length)
            }
            onClick={() =>
              data &&
              onChoose(
                kind === "directory" ? [data.path] : selected,
                data.path,
                initialRoot.current,
              )
            }
          >
            {kind === "directory" ? "Use folder" : "Select"}
            <Check size={16} />
          </button>
        </footer>
      </section>
    </div>
  );
}
export function ProjectDialog({
  onClose,
  onOpen,
}: {
  onClose: () => void;
  onOpen: (project: Project) => void;
}) {
  const [mode, setMode] = useState<"open" | "import">("open");
  const [directory, setDirectory] = useState("");
  const [imageRoot, setImageRoot] = useState("");
  const [jsonPath, setJsonPath] = useState("");
  const [files, setFiles] = useState<string[] | undefined>();
  const [recursive, setRecursive] = useState(true);
  const [browser, setBrowser] = useState<
    "project" | "images" | "json" | "selection" | null
  >(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function submit() {
    setBusy(true);
    setError("");
    try {
      const result =
        mode === "import"
          ? await api<Project>("/coco/import", "POST", {
              directory,
              image_root: imageRoot,
              json_path: jsonPath,
            })
          : await api<Project>("/projects/open", "POST", {
              directory,
              ...(imageRoot ? { image_root: imageRoot } : {}),
              ...(files ? { files } : {}),
              recursive,
            });
      onOpen(result);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="modal-backdrop">
      <section
        className="modal project-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Open project"
      >
        <div className="modal-heading">
          <div>
            <h2>Project</h2>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Close">
            <X size={19} />
          </button>
        </div>
        <div className="tabs">
          <button
            className={mode === "open" ? "active" : ""}
            onClick={() => setMode("open")}
          >
            Create / open project
          </button>
          <button
            className={mode === "import" ? "active" : ""}
            onClick={() => setMode("import")}
          >
            Import COCO
          </button>
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
        >
          <label className="field-label" htmlFor="project-directory">
            Project folder <span>Annotations and settings</span>
          </label>
          <div className="input-browse">
            <input
              id="project-directory"
              placeholder="/path/to/project"
              value={directory}
              onChange={(e) => setDirectory(e.target.value)}
              required
            />
            <button
              type="button"
              className="icon-button"
              title="Browse project folder"
              onClick={() => setBrowser("project")}
            >
              <FolderOpen size={18} />
            </button>
          </div>
          <label className="field-label" htmlFor="image-root">
            Image folder
          </label>
          <div className="input-browse">
            <input
              id="image-root"
              placeholder={directory || "Select an image folder"}
              required={mode === "import"}
              value={imageRoot}
              onChange={(e) => {
                setImageRoot(e.target.value);
                setFiles(undefined);
              }}
            />
            <button
              type="button"
              className="icon-button"
              title="Browse image folder"
              onClick={() => setBrowser("images")}
            >
              <FolderOpen size={18} />
            </button>
          </div>
          <p className="field-note">
            Keep the project outside the image folder.
          </p>
          {mode === "open" ? (
            <div className="selection-settings">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={recursive}
                  onChange={(e) => setRecursive(e.target.checked)}
                  disabled={!!files}
                />{" "}
                Include subfolders
              </label>
              <button
                type="button"
                className="text-button"
                disabled={!imageRoot}
                onClick={() => setBrowser("selection")}
              >
                {files
                  ? `${files.length} images selected`
                  : "Select individual images"}
              </button>
              {files && (
                <button
                  className="text-button"
                  type="button"
                  onClick={() => setFiles(undefined)}
                >
                  Use all
                </button>
              )}
            </div>
          ) : (
            <>
              <label className="field-label" htmlFor="json-file">
                COCO annotations <span>Import into a new project</span>
              </label>
              <div className="input-browse">
                <input
                  id="json-file"
                  value={jsonPath}
                  onChange={(e) => setJsonPath(e.target.value)}
                  placeholder="/path/annotations.json"
                  required
                />
                <button
                  type="button"
                  className="icon-button"
                  title="Select COCO file"
                  onClick={() => setBrowser("json")}
                >
                  <FileJson size={18} />
                </button>
              </div>
            </>
          )}
          {error && <p className="form-error">{error}</p>}
          <footer className="modal-footer">
            <button
              type="button"
              className="button secondary"
              onClick={onClose}
            >
              <ArrowLeft size={16} /> Back
            </button>
            <button className="button primary" disabled={busy || !directory}>
              {busy ? (
                <LoaderCircle size={17} className="spin" />
              ) : (
                <FolderOpen size={17} />
              )}{" "}
              {busy ? "Opening…" : "Open project"}
            </button>
          </footer>
        </form>
      </section>
      {browser && (
        <FileBrowser
          initial={browser === "project" ? directory : imageRoot || directory}
          kind={
            browser === "json"
              ? "json"
              : browser === "selection"
                ? "images"
                : "directory"
          }
          onCancel={() => setBrowser(null)}
          onChoose={(paths, _base, initialRoot) => {
            if (browser === "project") setDirectory(paths[0]);
            if (browser === "images") {
              setImageRoot(paths[0]);
              setFiles(undefined);
            }
            if (browser === "json") setJsonPath(paths[0]);
            if (browser === "selection") {
              const root = initialRoot.replace(/\\/g, "/").replace(/\/+$/, "");
              const normalizedPaths = paths.map((p) => p.replace(/\\/g, "/"));
              const windows = /^[A-Za-z]:|^\/\//.test(root);
              const comparable = (p: string) => (windows ? p.toLowerCase() : p);
              const outside = normalizedPaths.some(
                (p) => !comparable(p).startsWith(`${comparable(root)}/`),
              );
              if (outside) {
                setError("Select files within the chosen image folder.");
              } else {
                setImageRoot(initialRoot);
                setFiles(normalizedPaths.map((p) => p.slice(root.length + 1)));
              }
            }
            setBrowser(null);
          }}
        />
      )}
    </div>
  );
}
