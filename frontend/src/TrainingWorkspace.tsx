import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  Download,
  LoaderCircle,
  Plus,
  RefreshCw,
} from "lucide-react";
import { projectClient, assetURL, errorText } from "./api";
import type { Project, Task } from "./types";
import type {
  DatasetSnapshot,
  EpochMetric,
  Split,
  TrainingOverview,
  TrainingRun,
} from "./trainingTypes";

import TrainingPreview from "./TrainingPreview";
import TrainingLog from "./TrainingLog";

const activeStates = new Set([
  "queued",
  "preparing",
  "running",
  "pausing",
  "syncing",
]);
const format = (v?: number) => (v === undefined ? "—" : v.toFixed(3));
function Curve({
  title,
  rows,
  fields,
}: {
  title: string;
  rows: EpochMetric[];
  fields: {
    key: "train_loss" | "validation_loss" | "map" | "box_map";
    name: string;
  }[];
}) {
  const values = rows.flatMap((r) =>
    fields.flatMap((f) => (r[f.key] === undefined ? [] : [r[f.key]!])),
  );
  const max = Math.max(...values, 0.001),
    min = Math.min(0, ...values);
  const last = Math.max(1, ...rows.map((r) => r.epoch));
  return (
    <section className="training-chart" aria-label={title}>
      <h3>{title}</h3>
      {values.length ? (
        <svg viewBox="0 0 500 200" role="img" aria-label={`${title} by epoch`}>
          {[0, 0.5, 1].map((t) => (
            <g key={t}>
              <line
                x1="42"
                x2="490"
                y1={160 - t * 140}
                y2={160 - t * 140}
                stroke="var(--border)"
              />
              <text x="2" y={164 - t * 140} fontSize="11">
                {(min + t * (max - min)).toFixed(2)}
              </text>
            </g>
          ))}
          {fields.map((f, i) => (
            <polyline
              key={f.key}
              fill="none"
              stroke={i ? "#b47737" : "var(--green)"}
              strokeWidth="2.5"
              points={rows
                .filter((r) => r[f.key] !== undefined)
                .map(
                  (r) =>
                    `${42 + (448 * r.epoch) / last},${160 - (140 * (r[f.key]! - min)) / (max - min)}`,
                )
                .join(" ")}
            />
          ))}
          <text x="42" y="190" fontSize="11">
            Epoch 1
          </text>
          <text x="430" y="190" fontSize="11">
            {last}
          </text>
        </svg>
      ) : (
        <p className="training-empty-chart">Available after the first epoch</p>
      )}
      <div className="training-legend">
        {fields.map((f) => (
          <span key={f.key}>{f.name}</span>
        ))}
      </div>
    </section>
  );
}

export default function TrainingWorkspace(p: {
  project: Project;
  task: Task;
  visible: boolean;
  flush: () => Promise<void>;
  onAnnotate: (imageId?: number) => void;
  onStatus: (label: string) => void;
  onPredictions: (batch: string) => Promise<void>;
}) {
  const api = useMemo(
    () => projectClient(p.project),
    [p.project.id, p.project.directory],
  );
  const [showPreview, setShowPreview] = useState(false);
  const [showLog, setShowLog] = useState(false);
  const [data, setData] = useState<TrainingOverview | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<number[]>([]);
  const [dataset, setDataset] = useState("");
  const [model, setModel] = useState(`yolo26-small-${p.task}`);
  const [device, setDevice] = useState("");
  const [test, setTest] = useState(false);
  const [validationPercent, setValidationPercent] = useState(20);
  const [testPercent, setTestPercent] = useState(10);
  const [assignments, setAssignments] = useState<Record<number, Split>>({});
  const [preview, setPreview] = useState<DatasetSnapshot | null>(null);
  const [epochs, setEpochs] = useState(100);
  const [batch, setBatch] = useState(4);
  const [resolution, setResolution] = useState<number | undefined>();
  const [lr, setLr] = useState("");
  const [augment, setAugment] = useState(true);
  const [lossy, setLossy] = useState(false);
  const [runId, setRunId] = useState("");
  const [configure, setConfigure] = useState(true);
  const [pairing, setPairing] = useState("");
  const [confidence, setConfidence] = useState(0.5);
  const generation = useRef(0);
  const latest = useRef(p);
  latest.current = p;
  const run = data?.runs.find((r) => r.id === runId);
  const chosen = data?.datasets.find(
    (d) => d.id === (configure ? dataset : run?.dataset_id),
  );
  const compatible = data?.devices.find((d) => d.id === device);

  async function refresh(signal?: AbortSignal) {
    const g = generation.current;
    const result = await api<TrainingOverview>(
      `/training?task=${p.task}`,
      "GET",
      undefined,
      signal,
    );
    if (g !== generation.current || signal?.aborted) return;
    setData(result);
    latest.current.onStatus(
      (result.active_runs ??
        result.runs.filter((r) => activeStates.has(r.status)).length) > 0
        ? "Training in progress"
        : "",
    );
    const gpu =
      result.devices.find((d) => d.id === device) ||
      result.devices.find((d) => d.online && d.enabled);
    setDevice((old) => old || gpu?.id || "");
    if (gpu)
      setModel((old) =>
        gpu.models.includes(old)
          ? old
          : result.catalog.find(
              (m) => m.task === p.task && gpu.models.includes(m.id),
            )?.id || old,
      );
    setDataset((old) => old || result.datasets.at(-1)?.id || "");
  }
  useEffect(() => {
    generation.current++;
    setSelected([]);
    setDataset("");
    setRunId("");
    setPreview(null);
    setAssignments({});
    setModel(`yolo26-small-${p.task}`);
    setResolution(undefined);
    setData(null);
    setBusy(false);
    setError("");
    setShowPreview(false);
    setDevice("");
    const controller = new AbortController();
    const update = () => {
      void refresh(controller.signal).catch((e) => {
        if (!controller.signal.aborted) setError(errorText(e));
      });
    };
    update();
    const timer = setInterval(() => {
      if (latest.current.visible) update();
    }, 2000);
    return () => {
      generation.current++;
      controller.abort();
      clearInterval(timer);
    };
  }, [p.project.id, p.project.directory, p.task]);
  useEffect(() => {
    if (!runId || !p.visible || typeof EventSource === "undefined") return;
    const stream = new EventSource(
      assetURL(`/training/runs/${runId}/events?after=0`),
      { withCredentials: true },
    );
    stream.onmessage = (event) => {
      const updated = JSON.parse(event.data) as TrainingRun;
      setData((old) =>
        old
          ? {
              ...old,
              runs: old.runs.map((r) => (r.id === updated.id ? updated : r)),
            }
          : old,
      );
    };
    return () => stream.close();
  }, [runId, p.visible, p.project.id, p.project.directory]);
  async function perform(fn: () => Promise<void>) {
    const g = generation.current;
    setBusy(true);
    setError("");
    try {
      await fn();
      if (g === generation.current) await refresh();
    } catch (e) {
      if (g === generation.current) setError(errorText(e));
    } finally {
      if (g === generation.current) setBusy(false);
    }
  }
  const options = {
    task: p.task,
    seed: 42,
    validation_percent: validationPercent,
    test_percent: test ? testPercent : 0,
    assignments,
  };
  async function review(reviewed: boolean) {
    const g = generation.current;
    await p.flush();
    if (g !== generation.current) return;
    const current = await api<TrainingOverview>(`/training?task=${p.task}`);
    await api("/training/reviews", "POST", {
      task: p.task,
      reviewed,
      images: Object.fromEntries(
        current.images
          .filter((im) => selected.includes(im.id))
          .map((im) => [im.id, im.revision]),
      ),
    });
    if (g === generation.current) {
      setSelected([]);
      setPreview(null);
    }
  }
  async function snapshot(previewOnly: boolean) {
    const g = generation.current;
    await p.flush();
    if (g !== generation.current) return;
    const value = await api<DatasetSnapshot>(
      `/training/datasets${previewOnly ? "/preview" : ""}`,
      "POST",
      options,
    );
    if (g !== generation.current) return;
    if (previewOnly) setPreview(value);
    else {
      setDataset(value.id);
      setPreview(null);
    }
  }
  async function predict(purpose: "preview" | "preannotate") {
    const g = generation.current;
    if (!run || !data?.images.length) return;
    await p.flush();
    if (g !== generation.current) return;
    await api(`/training/runs/${run.id}/predict`, "POST", {
      image_ids: selected.length ? selected : [data.images[0].id],
      confidence: purpose === "preview" ? 0.001 : confidence,
      purpose,
    });
    if (g === generation.current) setShowPreview(purpose === "preview");
  }
  async function start() {
    const g = generation.current;
    await p.flush();
    if (g !== generation.current) return;
    const result = await api<TrainingRun>("/training/runs", "POST", {
      dataset_id: dataset,
      model_id: model,
      device_id: device,
      allow_lossy: lossy,
      request_id: crypto.randomUUID(),
      parameters: {
        epochs,
        batch,
        resolution: resolution || null,
        learning_rate: lr ? Number(lr) : null,
        augment,
      },
    });
    if (g === generation.current) {
      setRunId(result.id);
      setConfigure(false);
    }
  }
  return (
    <main className="training-workspace" hidden={!p.visible}>
      <div className="training-heading">
        <div>
          <h1>Training</h1>
          <span>
            {p.project.name} · {p.task}
          </span>
        </div>
        <button
          className="button"
          onClick={() => {
            setConfigure(true);
            setRunId("");
          }}
        >
          <Plus size={16} /> New training
        </button>
      </div>
      {error && (
        <p role="alert" className="training-error">
          {error}{" "}
          <button onClick={() => void perform(async () => {})}>
            <RefreshCw size={14} /> Retry
          </button>
        </p>
      )}
      {!data && !error && (
        <LoaderCircle className="spin" aria-label="Loading training" />
      )}
      {data && (
        <div className="training-layout">
          <aside className="training-history" aria-label="Training history">
            {data.runs.length === 0 ? (
              <p>No training runs yet</p>
            ) : (
              [...data.runs].reverse().map((r) => (
                <button
                  key={r.id}
                  className={runId === r.id ? "selected" : ""}
                  onClick={() => {
                    setRunId(r.id);
                    setConfigure(false);
                    setShowPreview(false);
                  }}
                >
                  <strong>{r.model.name}</strong>
                  <span>
                    {r.status} ·{" "}
                    {new Date(r.created_at * 1000).toLocaleDateString()}
                  </span>
                </button>
              ))
            )}
          </aside>
          <div className="training-content">
            {configure ? (
              <>
                <section className="training-section">
                  <h2>Dataset</h2>
                  <div className="training-row">
                    <select
                      aria-label="Dataset version"
                      value={dataset}
                      onChange={(e) => setDataset(e.target.value)}
                    >
                      <option value="">Create a reviewed dataset</option>
                      {data.datasets.map((d) => (
                        <option key={d.id} value={d.id}>
                          Version {d.version} ·{" "}
                          {Object.values(d.counts).reduce(
                            (a, b) => a + (b || 0),
                            0,
                          )}{" "}
                          images
                        </option>
                      ))}
                    </select>
                    <button className="button" onClick={() => p.onAnnotate()}>
                      Review in editor
                    </button>
                  </div>
                  {dataset &&
                    !data.runs.some((r) => r.dataset_id === dataset) && (
                      <button
                        className="button"
                        disabled={busy}
                        onClick={() =>
                          void perform(async () => {
                            await api(
                              `/training/datasets/${dataset}`,
                              "DELETE",
                            );
                            setDataset("");
                          })
                        }
                      >
                        Delete unused version
                      </button>
                    )}
                  {chosen && (
                    <p className="training-counts">
                      {Object.entries(chosen.counts)
                        .map(([split, n]) => `${n} ${split}`)
                        .join(" · ")}
                    </p>
                  )}
                  <details open={!dataset}>
                    <summary>Review images and create a version</summary>
                    <div className="training-row">
                      <button
                        className="button"
                        onClick={() =>
                          setSelected(data.images.map((im) => im.id))
                        }
                      >
                        Select all
                      </button>
                      <button
                        className="button"
                        disabled={!selected.length || busy}
                        onClick={() => void perform(() => review(true))}
                      >
                        Mark reviewed
                      </button>
                      <button
                        className="button"
                        disabled={!selected.length || busy}
                        onClick={() => void perform(() => review(false))}
                      >
                        Unmark
                      </button>
                    </div>
                    <div className="training-images">
                      <table>
                        <thead>
                          <tr>
                            <th>Select</th>
                            <th>Image</th>
                            <th>Objects</th>
                            <th>Reviewed</th>
                            <th>Partition</th>
                          </tr>
                        </thead>
                        <tbody>
                          {data.images.map((im) => (
                            <tr key={im.id}>
                              <td>
                                <input
                                  type="checkbox"
                                  aria-label={`Select ${im.file_name}`}
                                  checked={selected.includes(im.id)}
                                  onChange={(e) =>
                                    setSelected((old) =>
                                      e.target.checked
                                        ? [...old, im.id]
                                        : old.filter((id) => id !== im.id),
                                    )
                                  }
                                />
                              </td>
                              <td>
                                <button onClick={() => p.onAnnotate(im.id)}>
                                  {im.file_name}
                                </button>
                              </td>
                              <td>{im.annotation_count}</td>
                              <td>{im.reviewed ? "✓" : "—"}</td>
                              <td>
                                <select
                                  aria-label={`Partition ${im.file_name}`}
                                  value={assignments[im.id] || ""}
                                  onChange={(e) => {
                                    const next = { ...assignments };
                                    if (e.target.value)
                                      next[im.id] = e.target.value as Split;
                                    else delete next[im.id];
                                    setAssignments(next);
                                    setPreview(null);
                                  }}
                                >
                                  <option value="">Automatic</option>
                                  <option value="train">Training</option>
                                  <option value="validation">Validation</option>
                                  {test && <option value="test">Test</option>}
                                </select>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="training-row">
                      <label>
                        Validation %
                        <input
                          aria-label="Validation percent"
                          type="number"
                          min="1"
                          max="99"
                          value={validationPercent}
                          onChange={(e) => {
                            setValidationPercent(Number(e.target.value));
                            setPreview(null);
                          }}
                        />
                      </label>
                      <label className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={test}
                          onChange={(e) => {
                            setTest(e.target.checked);
                            setPreview(null);
                            if (!e.target.checked)
                              setAssignments((old) =>
                                Object.fromEntries(
                                  Object.entries(old).filter(
                                    ([, split]) => split !== "test",
                                  ),
                                ),
                              );
                          }}
                        />
                        Reserve test set
                      </label>
                      {test && (
                        <label>
                          Test %
                          <input
                            aria-label="Test percent"
                            type="number"
                            min="1"
                            max="98"
                            value={testPercent}
                            onChange={(e) => {
                              setTestPercent(Number(e.target.value));
                              setPreview(null);
                            }}
                          />
                        </label>
                      )}
                      <span>
                        {100 - validationPercent - (test ? testPercent : 0)}%
                        training
                      </span>
                    </div>
                    <p className="field-note">
                      Only reviewed images are included. Reviewed empty images
                      are negatives.
                      {test &&
                        " Test stays separate until you request final evaluation."}
                    </p>
                    {preview && (
                      <div>
                        <p>
                          {Object.entries(preview.counts)
                            .map(([split, n]) => `${n} ${split}`)
                            .join(" · ")}
                        </p>
                        {preview.warnings.map((w) => (
                          <p key={w} className="field-note">
                            {w}
                          </p>
                        ))}
                      </div>
                    )}
                    <div className="training-row">
                      <button
                        className="button"
                        disabled={busy}
                        onClick={() => void perform(() => snapshot(true))}
                      >
                        Preview split
                      </button>
                      <button
                        className="button primary"
                        disabled={busy || !preview}
                        onClick={() => void perform(() => snapshot(false))}
                      >
                        Create dataset version
                      </button>
                    </div>
                  </details>
                </section>
                <section className="training-section">
                  <h2>Model and compute</h2>
                  <div className="training-row">
                    <label>
                      Model
                      <select
                        aria-label="Training model"
                        value={model}
                        onChange={(e) => {
                          setModel(e.target.value);
                          setResolution(undefined);
                        }}
                      >
                        {data.catalog
                          .filter(
                            (m) =>
                              m.task === p.task &&
                              (!compatible || compatible.models.includes(m.id)),
                          )
                          .map((m) => (
                            <option key={m.id} value={m.id}>
                              {m.name}
                            </option>
                          ))}
                      </select>
                    </label>
                    <label>
                      GPU
                      <select
                        aria-label="Training GPU"
                        value={device}
                        onChange={(e) => {
                          setDevice(e.target.value);
                          const gpu = data.devices.find(
                            (d) => d.id === e.target.value,
                          );
                          if (gpu && !gpu.models.includes(model))
                            setModel(
                              data.catalog.find(
                                (m) =>
                                  m.task === p.task &&
                                  gpu.models.includes(m.id),
                              )?.id || model,
                            );
                        }}
                      >
                        <option value="">Connect a GPU</option>
                        {data.devices.map((d) => (
                          <option
                            key={d.id}
                            value={d.id}
                            disabled={!d.online || !d.enabled}
                          >
                            {d.name}
                            {!d.online ? " · offline" : ""}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  {!data.devices.length && (
                    <p className="field-note">
                      {data.hosted
                        ? "Connect your Linux NVIDIA computer to train."
                        : "Start the optional training agent on this computer. NVIDIA CUDA is required."}
                    </p>
                  )}
                  <button
                    className="button"
                    disabled={busy}
                    onClick={() =>
                      void perform(async () => {
                        const result = await api<{ code: string }>(
                          "/training/compute/pair",
                          "POST",
                          {},
                        );
                        setPairing(result.code);
                      })
                    }
                  >
                    Connect computer
                  </button>
                  {pairing && (
                    <div className="training-pairing">
                      <p>
                        Enter this one-use code in the training agent. Expires
                        in 10 minutes.
                      </p>
                      <code>{pairing}</code>
                    </div>
                  )}
                  {!data.hosted && (
                    <button
                      className="button"
                      disabled={busy}
                      onClick={() =>
                        void perform(async () => {
                          await api("/model/unload", "POST", {});
                        })
                      }
                    >
                      Release SAM GPU
                    </button>
                  )}
                  {!!data.devices.length && (
                    <details>
                      <summary>Connected computers</summary>
                      {data.devices.map((d) => (
                        <div className="training-row" key={d.id}>
                          <label className="checkbox-label">
                            <input
                              type="checkbox"
                              checked={d.enabled}
                              onChange={(e) =>
                                void perform(async () => {
                                  await api(
                                    `/training/compute/${d.node_id}/device`,
                                    "POST",
                                    { slot: d.slot, enabled: e.target.checked },
                                  );
                                })
                              }
                            />
                            {d.name} · {d.online ? "online" : "offline"}
                          </label>
                          <button
                            className="button"
                            onClick={() =>
                              void perform(async () => {
                                await api(
                                  `/training/compute/${d.node_id}/revoke`,
                                  "POST",
                                  {},
                                );
                              })
                            }
                          >
                            Disconnect
                          </button>
                        </div>
                      ))}
                    </details>
                  )}
                  {model.startsWith("yolo26") && p.task === "segmentation" && (
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={lossy}
                        onChange={(e) => setLossy(e.target.checked)}
                      />
                      Allow YOLO polygon approximations: filled holes and
                      connected parts
                    </label>
                  )}
                  <details>
                    <summary>Advanced</summary>
                    <div className="training-row">
                      <label>
                        Epochs
                        <input
                          type="number"
                          min="1"
                          max="10000"
                          value={epochs}
                          onChange={(e) => setEpochs(Number(e.target.value))}
                        />
                      </label>
                      <label>
                        Batch
                        <input
                          type="number"
                          min="1"
                          max="512"
                          value={batch}
                          onChange={(e) => setBatch(Number(e.target.value))}
                        />
                      </label>
                      <label>
                        Resolution
                        <input
                          type="number"
                          min="128"
                          max="4096"
                          value={
                            resolution ||
                            data.catalog.find((m) => m.id === model)
                              ?.resolution ||
                            640
                          }
                          onChange={(e) =>
                            setResolution(Number(e.target.value))
                          }
                        />
                      </label>
                      <label>
                        Learning rate
                        <input
                          type="number"
                          step="0.0001"
                          placeholder="Model default"
                          value={lr}
                          onChange={(e) => setLr(e.target.value)}
                        />
                      </label>
                    </div>
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={augment}
                        onChange={(e) => setAugment(e.target.checked)}
                      />
                      Training augmentation
                    </label>
                    <p className="field-note">
                      Early stopping: 15 epochs without validation improvement.
                      Seed: 42.
                    </p>
                    <a
                      href={
                        data.catalog.find((m) => m.id === model)?.license_url
                      }
                      target="_blank"
                      rel="noreferrer"
                    >
                      {data.catalog.find((m) => m.id === model)?.license}
                    </a>
                  </details>
                </section>
                <button
                  className="button primary"
                  disabled={
                    !dataset ||
                    !compatible?.online ||
                    !compatible.enabled ||
                    !compatible.models.includes(model) ||
                    busy
                  }
                  onClick={() => void perform(start)}
                >
                  {busy && <LoaderCircle size={16} className="spin" />}Start
                  training
                </button>
              </>
            ) : (
              run && (
                <>
                  <div className="training-row">
                    <button
                      className="button"
                      onClick={() => setConfigure(true)}
                    >
                      <ArrowLeft size={16} /> Setup
                    </button>
                    <h2>{run.model.name}</h2>
                    <span className="count-pill">{run.status}</span>
                  </div>
                  <p>
                    Epoch {run.epoch} / {run.parameters.epochs}
                    {run.metrics.at(-1)?.elapsed !== undefined &&
                      ` · ${Math.round(run.metrics.at(-1)!.elapsed! / 60)} min elapsed`}
                    {run.epoch >= 3 &&
                    run.metrics.at(-1)?.elapsed &&
                    activeStates.has(run.status)
                      ? ` · ~${Math.round(((run.metrics.at(-1)!.elapsed! / run.epoch) * (run.parameters.epochs - run.epoch)) / 60)} min remaining`
                      : ""}
                  </p>
                  <progress
                    max={1}
                    value={Math.max(
                      run.progress || 0,
                      run.epoch / run.parameters.epochs,
                    )}
                  />
                  {run.error && (
                    <p role="alert" className="training-error">
                      {run.error}
                    </p>
                  )}
                  <div className="training-row">
                    {["queued", "preparing", "running"].includes(
                      run.status,
                    ) && (
                      <button
                        className="button"
                        onClick={() =>
                          void perform(async () => {
                            await api(
                              `/training/runs/${run.id}/pause`,
                              "POST",
                              {},
                            );
                          })
                        }
                      >
                        Pause
                      </button>
                    )}
                    {["paused", "interrupted", "failed"].includes(
                      run.status,
                    ) && (
                      <button
                        className="button"
                        onClick={() =>
                          void perform(async () => {
                            await api(
                              `/training/runs/${run.id}/resume`,
                              "POST",
                              {},
                            );
                          })
                        }
                      >
                        Resume
                      </button>
                    )}
                    {(activeStates.has(run.status) ||
                      ["paused", "interrupted"].includes(run.status)) && (
                      <button
                        className="button"
                        onClick={() =>
                          void perform(async () => {
                            await api(
                              `/training/runs/${run.id}/cancel`,
                              "POST",
                              {},
                            );
                          })
                        }
                      >
                        Cancel
                      </button>
                    )}
                  </div>
                  <div className="training-curves">
                    <Curve
                      title="Loss"
                      rows={run.metrics}
                      fields={[
                        { key: "train_loss", name: "Training" },
                        { key: "validation_loss", name: "Validation" },
                      ]}
                    />
                    <Curve
                      title={
                        p.task === "segmentation"
                          ? "Validation mask mAP 50–95"
                          : "Validation mAP 50–95"
                      }
                      rows={run.metrics}
                      fields={[{ key: "map", name: "Validation" }]}
                    />
                  </div>
                  {!activeStates.has(run.status) && (
                    <button
                      className="button"
                      disabled={
                        busy ||
                        (!!run.prediction &&
                          activeStates.has(run.prediction.status)) ||
                        (!!run.evaluation &&
                          activeStates.has(run.evaluation.status))
                      }
                      onClick={() =>
                        void perform(async () => {
                          await api(`/training/runs/${run.id}`, "DELETE");
                          setRunId("");
                          setConfigure(true);
                        })
                      }
                    >
                      Delete execution
                    </button>
                  )}
                  <details>
                    <summary>Metrics and details</summary>
                    <table>
                      <thead>
                        <tr>
                          <th>Epoch</th>
                          <th>mAP</th>
                          <th>Precision</th>
                          <th>Recall</th>
                          {p.task === "segmentation" && <th>Box mAP</th>}
                        </tr>
                      </thead>
                      <tbody>
                        {run.metrics.map((m) => (
                          <tr key={m.epoch}>
                            <td>{m.epoch}</td>
                            <td>{format(m.map)}</td>
                            <td>{format(m.precision)}</td>
                            <td>{format(m.recall)}</td>
                            {p.task === "segmentation" && (
                              <td>{format(m.box_map)}</td>
                            )}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="field-note">
                      Dataset{" "}
                      {
                        data.datasets.find((d) => d.id === run.dataset_id)
                          ?.version
                      }{" "}
                      · {run.id}
                    </p>
                  </details>
                  {!!run.metrics.at(-1)?.per_class?.length && (
                    <details>
                      <summary>Validation by class</summary>
                      <table>
                        <thead>
                          <tr>
                            <th>Class</th>
                            <th>mAP</th>
                            <th>Precision</th>
                            <th>Recall</th>
                          </tr>
                        </thead>
                        <tbody>
                          {run.metrics.at(-1)!.per_class!.map((c) => (
                            <tr key={c.category_id}>
                              <td>
                                {p.project.categories.find(
                                  (cat) => cat.id === c.category_id,
                                )?.name || c.category_id}
                              </td>
                              <td>{format(c.map)}</td>
                              <td>{format(c.precision)}</td>
                              <td>{format(c.recall)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </details>
                  )}
                  <details onToggle={(e) => setShowLog(e.currentTarget.open)}>
                    <summary>Log</summary>
                    {showLog && (
                      <TrainingLog
                        project={p.project}
                        runId={run.id}
                        cursor={run.events}
                      />
                    )}
                  </details>
                  {run.status === "completed" && (
                    <section className="training-section">
                      <h2>Results</h2>
                      <div className="training-row">
                        {run.artifacts
                          .filter((a) => a.status === "ready")
                          .map((a) => (
                            <a
                              className="button"
                              key={a.id}
                              href={assetURL(`/training/artifacts/${a.id}`)}
                              download
                            >
                              <Download size={16} />
                              {a.name}
                            </a>
                          ))}
                        {!!chosen?.counts.test && (
                          <button
                            className="button"
                            disabled={
                              busy ||
                              run.evaluation?.status === "queued" ||
                              run.evaluation?.status === "running"
                            }
                            onClick={() =>
                              void perform(async () => {
                                await api(
                                  `/training/runs/${run.id}/evaluate`,
                                  "POST",
                                  {},
                                );
                              })
                            }
                          >
                            Evaluate on test
                          </button>
                        )}
                      </div>
                      {run.evaluation && (
                        <div>
                          <h3>Test evaluation · {run.evaluation.status}</h3>
                          {Object.entries(run.evaluation.metrics || {}).map(
                            ([key, value]) => (
                              <span key={key}>
                                {key}: {format(value)}{" "}
                              </span>
                            ),
                          )}
                          {run.evaluation.per_class?.map((c) => (
                            <p key={c.category_id}>
                              {p.project.categories.find(
                                (cat) => cat.id === c.category_id,
                              )?.name || c.category_id}
                              : {format(c.map)}
                            </p>
                          ))}
                          {run.evaluation.error && (
                            <p role="alert">{run.evaluation.error}</p>
                          )}
                        </div>
                      )}
                      <details>
                        <summary>
                          Images for predictions · {selected.length} selected
                        </summary>
                        <div className="training-images">
                          {data.images.map((im) => (
                            <label className="checkbox-label" key={im.id}>
                              <input
                                type="checkbox"
                                checked={selected.includes(im.id)}
                                onChange={(e) =>
                                  setSelected((old) =>
                                    e.target.checked
                                      ? [...old, im.id]
                                      : old.filter((id) => id !== im.id),
                                  )
                                }
                              />
                              {im.file_name}
                            </label>
                          ))}
                        </div>
                        <button
                          className="button"
                          onClick={() =>
                            setSelected(data.images.map((im) => im.id))
                          }
                        >
                          Select all
                        </button>
                      </details>
                      <div className="training-row">
                        <label>
                          Confidence
                          <input
                            aria-label="Prediction confidence"
                            type="number"
                            min="0.01"
                            max="1"
                            step=".05"
                            value={confidence}
                            onChange={(e) =>
                              setConfidence(Number(e.target.value))
                            }
                          />
                        </label>
                        <button
                          className="button"
                          disabled={
                            busy ||
                            (!!run.prediction &&
                              activeStates.has(run.prediction.status))
                          }
                          onClick={() => void perform(() => predict("preview"))}
                        >
                          Try model
                        </button>
                        <button
                          className="button"
                          disabled={
                            !selected.length ||
                            busy ||
                            (!!run.prediction &&
                              activeStates.has(run.prediction.status))
                          }
                          onClick={() =>
                            void perform(() => predict("preannotate"))
                          }
                        >
                          Preannotate ({selected.length})
                        </button>
                      </div>
                      {run.prediction && (
                        <div>
                          <p>
                            Predictions: {run.prediction.status}
                            {run.prediction.error}
                          </p>
                          {run.prediction.status === "completed" && (
                            <>
                              <button
                                className="button"
                                onClick={() => setShowPreview((old) => !old)}
                              >
                                Preview predictions
                              </button>
                              {run.prediction.purpose !== "preview" && (
                                <button
                                  className="button"
                                  onClick={() =>
                                    void perform(async () => {
                                      await p.onPredictions(
                                        run.prediction!.id!,
                                      );
                                    })
                                  }
                                >
                                  Review proposals in editor
                                </button>
                              )}
                              {showPreview && (
                                <TrainingPreview
                                  project={p.project}
                                  batchId={run.prediction.id!}
                                />
                              )}
                            </>
                          )}
                        </div>
                      )}
                    </section>
                  )}
                </>
              )
            )}
          </div>
        </div>
      )}
    </main>
  );
}
