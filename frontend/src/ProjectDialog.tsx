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
        aria-label="Explorador de archivos"
      >
        <div className="modal-heading">
          <div>
            <span className="eyebrow">EN ESTE EQUIPO</span>
            <h2>
              {kind === "directory"
                ? "Selecciona una carpeta"
                : kind === "json"
                  ? "Selecciona COCO JSON"
                  : "Selecciona imágenes"}
            </h2>
          </div>
          <button
            className="icon-button"
            onClick={onCancel}
            aria-label="Cerrar explorador"
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
            title="Carpeta superior"
          >
            <ArrowUp size={17} />
          </button>
          <input
            aria-label="Ruta de carpeta"
            value={path}
            onChange={(e) => setPath(e.target.value)}
          />
          <button type="submit" className="icon-button" title="Ir a la ruta">
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
                <div className="empty-panel">Esta carpeta está vacía.</div>
              )}
            </>
          )}
        </div>
        {error && <p className="form-error">{error}</p>}
        <footer className="modal-footer">
          <span className="muted">
            {kind === "images"
              ? `${selected.length} seleccionadas`
              : kind === "directory"
                ? "Elige la carpeta actual."
                : "Formato de segmentación COCO."}
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
            {kind === "directory" ? "Usar carpeta" : "Seleccionar"}
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
              image_root: imageRoot || `${directory}/imágenes`,
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
        aria-label="Abrir proyecto"
      >
        <div className="modal-heading">
          <div>
            <span className="eyebrow">TU ESPACIO DE TRABAJO</span>
            <h2>Todo empieza con una imagen.</h2>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Cerrar">
            <X size={19} />
          </button>
        </div>
        <div className="tabs">
          <button
            className={mode === "open" ? "active" : ""}
            onClick={() => setMode("open")}
          >
            Crear / abrir proyecto
          </button>
          <button
            className={mode === "import" ? "active" : ""}
            onClick={() => setMode("import")}
          >
            Importar COCO
          </button>
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
        >
          <label className="field-label" htmlFor="project-directory">
            Carpeta del proyecto <span>Anotaciones y configuración</span>
          </label>
          <div className="input-browse">
            <input
              id="project-directory"
              placeholder="/ruta/a/mi-proyecto"
              value={directory}
              onChange={(e) => setDirectory(e.target.value)}
              required
            />
            <button
              type="button"
              className="icon-button"
              title="Explorar carpeta del proyecto"
              onClick={() => setBrowser("project")}
            >
              <FolderOpen size={18} />
            </button>
          </div>
          <label className="field-label" htmlFor="image-root">
            Carpeta de imágenes{" "}
            <span>Los originales permanecen en su lugar</span>
          </label>
          <div className="input-browse">
            <input
              id="image-root"
              placeholder={
                directory
                  ? `${directory}/imágenes`
                  : "Por defecto: proyecto/imágenes"
              }
              value={imageRoot}
              onChange={(e) => {
                setImageRoot(e.target.value);
                setFiles(undefined);
              }}
            />
            <button
              type="button"
              className="icon-button"
              title="Explorar carpeta de imágenes"
              onClick={() => setBrowser("images")}
            >
              <FolderOpen size={18} />
            </button>
          </div>
          <p className="field-note">
            Guarda el proyecto fuera de la carpeta de imágenes.
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
                Incluir subcarpetas
              </label>
              <button
                type="button"
                className="text-button"
                disabled={!imageRoot && !directory}
                onClick={() => setBrowser("selection")}
              >
                {files
                  ? `${files.length} imágenes elegidas`
                  : "Elegir imágenes individuales"}
              </button>
              {files && (
                <button
                  className="text-button"
                  type="button"
                  onClick={() => setFiles(undefined)}
                >
                  Usar todas
                </button>
              )}
            </div>
          ) : (
            <>
              <label className="field-label" htmlFor="json-file">
                Anotaciones COCO <span>Se importan a un proyecto nuevo</span>
              </label>
              <div className="input-browse">
                <input
                  id="json-file"
                  value={jsonPath}
                  onChange={(e) => setJsonPath(e.target.value)}
                  placeholder="/ruta/anotaciones.json"
                  required
                />
                <button
                  type="button"
                  className="icon-button"
                  title="Seleccionar archivo COCO"
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
              <ArrowLeft size={16} /> Volver
            </button>
            <button className="button primary" disabled={busy || !directory}>
              {busy ? (
                <LoaderCircle size={17} className="spin" />
              ) : (
                <FolderOpen size={17} />
              )}{" "}
              {busy ? "Abriendo…" : "Abrir proyecto"}
            </button>
          </footer>
        </form>
      </section>
      {browser && (
        <FileBrowser
          initial={
            browser === "project"
              ? directory
              : browser === "json"
                ? imageRoot
                : imageRoot || `${directory}/imágenes`
          }
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
                setError(
                  "Selecciona archivos dentro de la carpeta de imágenes indicada.",
                );
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
