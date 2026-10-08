import { useEffect, useRef, useState } from "react";
import {
  FolderOpen,
  LoaderCircle,
  Plus,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { api, ApiError, errorText, usageChangedEvent } from "./api";
import type { HostedProject, Project } from "./types";

// Empty local-only fields let the shared editor retain its local contract.
// They are never sent to the hosted API or displayed as filesystem locations.
const editorProject = (project: HostedProject): Project => ({
  ...project,
  directory: "",
  image_root: "",
});
type ProjectSummary = Pick<HostedProject, "id" | "name"> & {
  images?: HostedProject["images"];
  image_count?: number;
};

export function HostedProjects({
  onClose,
  onOpen,
  onDelete,
}: {
  onClose: () => void;
  onOpen: (project: Project) => void;
  onDelete: (id: string) => void;
}) {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [deleting, setDeleting] = useState<ProjectSummary | null>(null);
  useEffect(() => {
    let active = true;
    void api<{ projects: ProjectSummary[] }>("/projects")
      .then((result) => {
        if (active) setProjects(result.projects);
      })
      .catch((e) => {
        if (active) setError(errorText(e));
      })
      .finally(() => {
        if (active) setBusy(false);
      });
    return () => {
      active = false;
    };
  }, []);
  async function run(operation: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await operation();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="modal-backdrop">
      <section
        className="modal hosted-projects"
        role="dialog"
        aria-modal="true"
        aria-label="Your projects"
      >
        <div className="modal-heading">
          <div>
            <h2>Your projects</h2>
            <p className="field-note">Private to your Google account.</p>
          </div>
          <button
            className="icon-button"
            onClick={onClose}
            aria-label="Close projects"
            disabled={busy}
          >
            <X size={19} />
          </button>
        </div>
        <form
          className="hosted-new-project"
          onSubmit={(event) => {
            event.preventDefault();
            void run(async () =>
              onOpen(
                editorProject(
                  await api<HostedProject>("/projects", "POST", {
                    name: name.trim(),
                  }),
                ),
              ),
            );
          }}
        >
          <input
            aria-label="New project name"
            placeholder="New project name"
            value={name}
            maxLength={160}
            required
            onChange={(event) => setName(event.target.value)}
          />
          <button className="button primary" disabled={busy || !name.trim()}>
            <Plus size={16} /> Create
          </button>
        </form>
        {busy && (
          <p className="inline-status" role="status">
            <LoaderCircle size={16} className="spin" /> Loading projects…
          </p>
        )}
        <div className="hosted-project-list">
          {projects.map((project) => (
            <div className="hosted-project-row" key={project.id}>
              <button
                className="hosted-project-open"
                aria-label={`Open ${project.name}`}
                disabled={busy}
                onClick={() =>
                  void run(async () =>
                    onOpen(
                      editorProject(
                        await api<HostedProject>(`/projects/${project.id}`),
                      ),
                    ),
                  )
                }
              >
                <FolderOpen size={18} />
                <span>
                  <strong>{project.name}</strong>
                  {(project.image_count !== undefined ||
                    project.images !== undefined) && (
                    <small>
                      {project.image_count ?? project.images?.length} images
                    </small>
                  )}
                </span>
              </button>
              <button
                className="icon-button"
                disabled={busy}
                aria-label={`Delete ${project.name}`}
                onClick={() => setDeleting(project)}
              >
                <Trash2 size={16} />
              </button>
            </div>
          ))}
        </div>
        {!busy && !projects.length && !error && (
          <p className="empty-panel">
            Create a project, then upload images to begin.
          </p>
        )}
        {deleting && (
          <div className="hosted-delete-confirm" role="alert">
            <p>
              Delete <strong>{deleting.name}</strong> and its images and
              annotations? Access ends immediately. Backup copies can remain for
              about seven days, until scheduled cleanup completes.
            </p>
            <div className="hosted-inline-actions">
              <button
                className="button secondary"
                disabled={busy}
                onClick={() => setDeleting(null)}
              >
                Keep project
              </button>
              <button
                className="button danger"
                disabled={busy}
                onClick={() =>
                  void run(async () => {
                    await api(`/projects/${deleting.id}`, "DELETE");
                    setProjects((old) =>
                      old.filter((project) => project.id !== deleting.id),
                    );
                    onDelete(deleting.id);
                    setDeleting(null);
                    window.dispatchEvent(new Event(usageChangedEvent));
                  })
                }
              >
                Delete permanently
              </button>
            </div>
          </div>
        )}
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
      </section>
    </div>
  );
}

export function HostedUploads({
  project,
  onClose,
  onUpdate,
}: {
  project: Project;
  onClose: () => void;
  onUpdate: (project: Project) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState("");
  const [uploaded, setUploaded] = useState(0);
  const folderInput = useRef<HTMLInputElement>(null);
  const filesInput = useRef<HTMLInputElement>(null);
  const importInput = useRef<HTMLInputElement>(null);
  const cancelled = useRef(false);
  const base = `/projects/${project.id}`;
  useEffect(() => {
    folderInput.current?.setAttribute("webkitdirectory", "");
  }, []);
  async function refresh() {
    onUpdate(editorProject(await api<HostedProject>(base)));
  }
  async function upload(files: FileList | null, folder = false) {
    if (!files?.length) return;
    const images = Array.from(files).filter((file) =>
      /\.(jpe?g|png|webp|bmp|tiff?)$/i.test(file.name),
    );
    const skipped = files.length - images.length;
    if (!images.length) {
      setError("Select JPEG, PNG, WebP, BMP, or TIFF images.");
      return;
    }
    setBusy(true);
    setError("");
    cancelled.current = false;
    let completed = 0;
    const failures: string[] = [];
    try {
      for (const file of images) {
        if (cancelled.current) break;
        setProgress(
          `Uploading ${completed + failures.length + 1} of ${images.length}: ${file.name}`,
        );
        const relativePath = file.webkitRelativePath || file.name;
        const name =
          folder && relativePath.includes("/")
            ? relativePath.slice(relativePath.indexOf("/") + 1)
            : relativePath;
        try {
          if (file.size <= 8 * 1024 * 1024) {
            const body = new FormData();
            body.append("file", file);
            body.append("relative_path", name);
            await api(base + "/images", "POST", body);
          } else {
            const upload = await api<{ id: string; chunk_bytes: number }>(
              base + "/uploads",
              "POST",
              {
                file_name: name,
                size: file.size,
              },
            );
            const endpoint = `${base}/uploads/${upload.id}`;
            try {
              let offset = 0;
              while (offset < file.size) {
                if (cancelled.current) throw new Error("Upload cancelled");
                const chunk = file.slice(offset, offset + upload.chunk_bytes);
                let retries = 0;
                for (;;) {
                  try {
                    const result = await api<{ offset: number }>(
                      `${endpoint}?offset=${offset}`,
                      "PUT",
                      chunk,
                    );
                    if (result.offset <= offset || result.offset > file.size)
                      throw new Error("Invalid upload progress");
                    offset = result.offset;
                    break;
                  } catch (error) {
                    // The same offset can be replayed safely after a lost response.
                    if (
                      cancelled.current ||
                      retries++ >= 2 ||
                      (error instanceof ApiError && error.status < 500)
                    )
                      throw error;
                  }
                }
                setProgress(
                  `Uploading ${file.name}: ${Math.round((100 * offset) / file.size)}%`,
                );
              }
              if (cancelled.current) throw new Error("Upload cancelled");
              setProgress(`Validating ${file.name}…`);
              await api(endpoint + "/complete", "POST");
            } catch (error) {
              try {
                await api(endpoint, "DELETE");
              } catch {
                /* Expiry also releases abandoned uploads. */
              }
              throw error;
            }
          }
          completed++;
        } catch (e) {
          failures.push(`${file.name}: ${errorText(e)}`);
          if (
            e instanceof ApiError &&
            [401, 403, 413, 429, 503, 507].includes(e.status)
          ) {
            cancelled.current = true;
            break;
          }
        }
      }
      setUploaded((old) => old + completed);
      await refresh();
      if (failures.length)
        setError(
          failures.slice(0, 5).join("\n") +
            (failures.length > 5
              ? `\n…and ${failures.length - 5} more files.`
              : ""),
        );
      setProgress(
        `${completed} image${completed === 1 ? "" : "s"} uploaded.${skipped ? ` ${skipped} non-image files skipped.` : ""}${cancelled.current ? " Remaining uploads stopped." : ""}`,
      );
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
      window.dispatchEvent(new Event(usageChangedEvent));
    }
  }
  async function importCoco(file?: File) {
    if (!file) return;
    setBusy(true);
    setError("");
    setProgress("Importing COCO annotations…");
    try {
      const body = new FormData();
      body.append("file", file);
      await api(base + "/coco/import", "POST", body);
      await refresh();
      setProgress("COCO annotations imported.");
    } catch (e) {
      setError(errorText(e));
      setProgress("");
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="modal-backdrop">
      <section
        className="modal hosted-upload"
        role="dialog"
        aria-modal="true"
        aria-label="Upload images"
      >
        <div className="modal-heading">
          <div>
            <h2>Add images to {project.name}</h2>
            <p className="field-note">
              Original resolution is preserved. Uploads use your account storage
              quota.
            </p>
          </div>
          <button
            className="icon-button"
            disabled={busy}
            onClick={onClose}
            aria-label="Close uploads"
          >
            <X size={19} />
          </button>
        </div>
        <div className="hosted-inline-actions">
          <button
            className="button primary"
            disabled={busy}
            onClick={() => filesInput.current?.click()}
          >
            <Upload size={16} /> Select images
          </button>
          <button
            className="button secondary"
            disabled={busy}
            onClick={() => folderInput.current?.click()}
          >
            <FolderOpen size={16} /> Select folder
          </button>
        </div>
        <input
          ref={filesInput}
          type="file"
          multiple
          accept="image/jpeg,image/png,image/webp,image/bmp,image/tiff"
          aria-label="Image files"
          hidden
          onChange={(event) => {
            void upload(event.target.files);
            event.target.value = "";
          }}
        />
        <input
          ref={folderInput}
          type="file"
          multiple
          aria-label="Image folder"
          hidden
          onChange={(event) => {
            void upload(event.target.files, true);
            event.target.value = "";
          }}
        />
        <p className="field-note">
          Folder uploads preserve relative filenames for COCO. Images stay
          private.
        </p>
        <div className="hosted-import">
          <h3>Import COCO annotations</h3>
          <p className="field-note">
            Upload matching images first. Import into a project without existing
            classes or annotations.
          </p>
          <button
            className="button secondary"
            disabled={busy || (!project.images.length && !uploaded)}
            onClick={() => importInput.current?.click()}
          >
            Choose COCO JSON
          </button>
          <input
            ref={importInput}
            type="file"
            accept="application/json,.json"
            hidden
            aria-label="COCO JSON file"
            onChange={(event) => {
              void importCoco(event.target.files?.[0]);
              event.target.value = "";
            }}
          />
        </div>
        {progress && (
          <p className="inline-status" role="status">
            {busy && <LoaderCircle size={15} className="spin" />}
            {progress}
          </p>
        )}
        {error && (
          <p className="form-error upload-errors" role="alert">
            {error}
          </p>
        )}
        <footer className="modal-footer">
          <span />
          {busy ? (
            <button
              className="button secondary"
              onClick={() => {
                cancelled.current = true;
              }}
            >
              Stop after current upload
            </button>
          ) : (
            <button className="button primary" onClick={onClose}>
              Done
            </button>
          )}
        </footer>
      </section>
    </div>
  );
}
