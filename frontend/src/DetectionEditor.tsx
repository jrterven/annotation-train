import { useEffect, useRef, useState } from "react";
import {
  BoxSelect,
  MousePointer2,
  Hand,
  ScanLine,
  Plus,
  Minus,
  Undo2,
  Redo2,
  Trash2,
  Check,
  X,
  LoaderCircle,
  Sparkles,
} from "lucide-react";
import type {
  BBox,
  BoxAnnotation,
  BoxDraft,
  DetectionState,
  GeometryResult,
  Project,
  Proposal,
  SourceImage,
  Tool,
  XY,
} from "./types";
import type { Workspace } from "./persistence";
import { api, errorText, uid } from "./api";
import { maskBounds } from "./masks";
import { xywh, xyxy } from "./boxes";
import CanvasEditor from "./CanvasEditor";
import IconButton from "./IconButton";
import VisualReference from "./VisualReference";
import type { VisualExample } from "./VisualReference";

type Props = {
  active: boolean;
  image?: SourceImage;
  project: Project;
  workspace: Workspace;
  categoryId: number;
  setCategoryId: (id: number) => void;
  createClass: () => void;
  editClass: (id: number) => void;
  modelReady: boolean;
  blocked: boolean;
  onError: (message: string) => void;
  navigate: (offset: number) => void;
};
const signature = (value: unknown) => JSON.stringify(value);
export default function DetectionEditor(p: Props) {
  const [tool, setTool] = useState<Tool>("box");
  const [selected, setSelected] = useState<string | null>(null);
  const [text, setText] = useState("");
  const [language, setLanguage] = useState<"en" | "es">("en");
  const [visual, setVisual] = useState<VisualExample | null>(null);
  const [visualOpen, setVisualOpen] = useState(false);
  const [, render] = useState(0);
  const pending = useRef(new Map<string, AbortController>());
  const fitRef = useRef<(() => void) | null>(null);
  const id = p.image?.id;
  const state = id === undefined ? undefined : p.workspace.getDetection(id);
  const entry = id === undefined ? undefined : p.workspace.entry(id);
  const chosen = state?.annotations.find((a) => a.id === selected);
  const selectedProposal = state?.proposals.find((a) => a.id === selected);
  const target = chosen || (!selectedProposal ? state?.draft : null);
  const busy =
    id !== undefined &&
    [...pending.current.keys()].some((k) => k.startsWith(`${id}:`));
  const canAdjust =
    !!target?.bbox && p.modelReady && !busy && !state?.adjustment;
  useEffect(() => {
    setSelected(null);
    setTool("box");
  }, [id]);
  useEffect(
    () => () => {
      for (const request of pending.current.values()) request.abort();
      pending.current.clear();
    },
    [],
  );
  function update(fn: (s: DetectionState) => DetectionState, history = true) {
    if (id !== undefined) p.workspace.updateDetection(id, fn, history);
  }
  function categoryRequired() {
    if (p.project.categories.some((c) => c.id === p.categoryId)) return true;
    p.createClass();
    return false;
  }
  function draw(box: BBox) {
    if (!categoryRequired()) return;
    if (state?.draft) {
      p.onError("Confirm or discard the current box before drawing another.");
      return;
    }
    update((s) => ({
      ...s,
      draft: {
        id: uid(),
        category_id: p.categoryId,
        bbox: xywh(box),
        points: [],
      },
      adjustment: null,
    }));
    setSelected(null);
    setTool("select");
  }
  function changeBox(targetId: string, bbox: BBox) {
    update((s) => ({
      ...s,
      annotations: s.annotations.map((a) =>
        a.id === targetId ? { ...a, bbox } : a,
      ),
      proposals: s.proposals.map((a) =>
        a.id === targetId ? { ...a, bbox } : a,
      ),
      draft:
        s.draft?.id === targetId
          ? { ...s.draft, bbox, points: [], prompt_bbox: undefined }
          : s.draft,
      adjustment: s.adjustment?.target_id === targetId ? null : s.adjustment,
    }));
  }
  function confirm() {
    if (!state?.draft?.bbox || busy || state.adjustment) return;
    const draft = state.draft;
    update((s) => ({
      ...s,
      annotations: [
        ...s.annotations,
        {
          id: draft.id,
          kind: "bbox",
          category_id: draft.category_id,
          iscrowd: 0,
          bbox: draft.bbox!,
        },
      ],
      draft: null,
    }));
    setSelected(draft.id);
    setTool("select");
  }
  function remove() {
    if (!state) return;
    const targetId = selected || state.draft?.id;
    if (!targetId) return;
    update((s) => ({
      ...s,
      annotations: s.annotations.filter((a) => a.id !== targetId),
      proposals: s.proposals.filter((a) => a.id !== targetId),
      draft: s.draft?.id === targetId ? null : s.draft,
      adjustment: s.adjustment?.target_id === targetId ? null : s.adjustment,
    }));
    setSelected(null);
  }
  function begin(imageId: number, kind: string) {
    const key = `${imageId}:${kind}`;
    pending.current.get(key)?.abort();
    const controller = new AbortController();
    pending.current.set(key, controller);
    render((v) => v + 1);
    const sourceEntry = p.workspace.entry(imageId);
    const generation = sourceEntry?.detectionGeneration;
    return {
      controller,
      isCurrent: () =>
        pending.current.get(key) === controller &&
        !controller.signal.aborted &&
        p.workspace.entry(imageId) === sourceEntry &&
        sourceEntry?.detectionGeneration === generation,
      finish: () => {
        if (pending.current.get(key) === controller) {
          pending.current.delete(key);
          render((v) => v + 1);
        }
      },
    };
  }
  async function adjust() {
    if (
      !canAdjust ||
      id === undefined ||
      !target?.bbox ||
      pending.current.has(`${id}:adjust`)
    )
      return;
    const imageId = id,
      original = structuredClone(target),
      bbox = target.bbox;
    const request = begin(imageId, "adjust");
    const stillMatches = () => {
      const fresh = p.workspace.getDetection(imageId);
      const object =
        fresh?.annotations.find((a) => a.id === original.id) ||
        (fresh?.draft?.id === original.id ? fresh.draft : null);
      return request.isCurrent() && signature(object) === signature(original);
    };
    try {
      const result = await api<GeometryResult & { image_id: number }>(
        "/infer/points",
        "POST",
        {
          image_id: imageId,
          revision: p.workspace.get(imageId)?.revision || 0,
          part: { id: original.id, points: [], box: xyxy(bbox) },
        },
        request.controller.signal,
      );
      if (!stillMatches() || result.image_id !== imageId) return;
      const adjusted = maskBounds(result.mask);
      if (!adjusted)
        throw new Error(
          "SAM returned an empty mask. Your original box is unchanged.",
        );
      p.workspace.updateDetection(
        imageId,
        (s) => ({
          ...s,
          adjustment: {
            target_id: original.id,
            base_bbox: bbox,
            bbox: adjusted,
          },
        }),
        false,
      );
    } catch (e) {
      if (stillMatches() && !(e instanceof Error && e.name === "AbortError"))
        p.onError(errorText(e));
    } finally {
      request.finish();
    }
  }
  function acceptAdjustment() {
    const adjustment = state?.adjustment;
    if (!adjustment) return;
    const original =
      state.annotations.find((a) => a.id === adjustment.target_id) ||
      (state.draft?.id === adjustment.target_id ? state.draft : null);
    if (signature(original?.bbox) !== signature(adjustment.base_bbox)) {
      update((s) => ({ ...s, adjustment: null }), false);
      return;
    }
    // History contains the original geometry, not a pending preview.
    update((s) => ({ ...s, adjustment: null }), false);
    changeBox(adjustment.target_id, adjustment.bbox);
  }
  async function point(at: XY, negative: boolean) {
    if (id === undefined || !p.modelReady || !categoryRequired()) return;
    const imageId = id;
    const draft: BoxDraft = structuredClone(
      state?.draft || { id: uid(), category_id: p.categoryId, points: [] },
    );
    if (!draft.points.length && draft.bbox) draft.prompt_bbox = draft.bbox;
    draft.points.push({ x: at[0], y: at[1], label: negative ? 0 : 1 });
    update((s) => ({ ...s, draft, adjustment: null }));
    setSelected(null);
    const request = begin(imageId, "points"),
      captured = signature(draft);
    const stillMatches = () =>
      request.isCurrent() &&
      signature(p.workspace.getDetection(imageId)?.draft) === captured;
    try {
      const result = await api<GeometryResult & { image_id: number }>(
        "/infer/points",
        "POST",
        {
          image_id: imageId,
          revision: p.workspace.get(imageId)?.revision || 0,
          part: {
            id: draft.id,
            points: draft.points,
            ...(draft.prompt_bbox ? { box: xyxy(draft.prompt_bbox) } : {}),
          },
        },
        request.controller.signal,
      );
      if (!stillMatches() || result.image_id !== imageId) return;
      const box = maskBounds(result.mask);
      if (!box)
        throw new Error(
          "SAM returned an empty mask. Add another point or draw a box manually.",
        );
      p.workspace.updateDetection(
        imageId,
        (s) => ({ ...s, draft: s.draft ? { ...s.draft, bbox: box } : null }),
        false,
      );
    } catch (e) {
      if (stillMatches() && !(e instanceof Error && e.name === "AbortError"))
        p.onError(errorText(e));
    } finally {
      request.finish();
    }
  }
  function cancelConceptSearches() {
    for (const [key, request] of pending.current)
      if (key.endsWith(":concept")) {
        request.abort();
        pending.current.delete(key);
      }
    render((v) => v + 1);
  }
  async function inferText() {
    if (
      id === undefined ||
      !state ||
      !p.modelReady ||
      !categoryRequired() ||
      (!text.trim() && !visual)
    )
      return;
    const imageId = id,
      category = p.categoryId,
      original = signature(state.proposals),
      request = begin(id, "concept");
    try {
      const result = await api<{ image_id: number; proposals: Proposal[] }>(
        visual ? "/infer/visual" : "/infer/text",
        "POST",
        {
          image_id: imageId,
          revision: p.workspace.get(imageId)?.revision || 0,
          text: text.trim(),
          source_language: language,
          category_id: category,
          ...(visual
            ? {
                reference_image: visual.base64,
                ...(visual.box ? { reference_box: visual.box } : {}),
              }
            : {}),
        },
        request.controller.signal,
      );
      if (
        !request.isCurrent() ||
        result.image_id !== imageId ||
        signature(p.workspace.getDetection(imageId)?.proposals) !== original
      )
        return;
      const proposals = result.proposals.flatMap((a) => {
        const box = maskBounds(a.mask);
        return box
          ? [
              {
                id: uid(),
                kind: "bbox" as const,
                category_id: category,
                bbox: box,
                iscrowd: 0,
                score: a.score,
                selected: false,
              },
            ]
          : [];
      });
      p.workspace.updateDetection(imageId, (s) => ({ ...s, proposals }));
      if (!proposals.length)
        p.onError("No boxes found. Try a different prompt.");
    } catch (e) {
      if (
        request.isCurrent() &&
        !(e instanceof Error && e.name === "AbortError")
      )
        p.onError(errorText(e));
    } finally {
      request.finish();
    }
  }
  function acceptProposals() {
    update((s) => ({
      ...s,
      annotations: [
        ...s.annotations,
        ...s.proposals
          .filter((a) => a.selected)
          .map(({ score: _score, selected: _selected, ...a }) => a),
      ],
      proposals: s.proposals.filter((a) => !a.selected),
    }));
    setSelected(null);
  }
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      const element = e.target as HTMLElement;
      if (
        !p.active ||
        p.blocked ||
        visualOpen ||
        /INPUT|TEXTAREA|SELECT/.test(element.tagName) ||
        element.isContentEditable
      )
        return;
      const mod = e.ctrlKey || e.metaKey;
      if (mod && ["z", "y"].includes(e.key.toLowerCase())) {
        e.preventDefault();
        if (id !== undefined)
          p.workspace.historyDetection(
            id,
            e.key.toLowerCase() === "y" || e.shiftKey,
          );
        return;
      }
      if (mod || e.altKey) return;
      if (e.key.toLowerCase() === "s") {
        e.preventDefault();
        if (!e.repeat) void adjust();
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (!e.repeat) confirm();
      } else if (e.key.toLowerCase() === "b") {
        setSelected(null);
        setTool("box");
      } else if (e.key.toLowerCase() === "v") setTool("select");
      else if (e.key.toLowerCase() === "p") setTool("positive");
      else if (e.key.toLowerCase() === "e" || e.key === "-")
        setTool("negative");
      else if (e.key.toLowerCase() === "f") fitRef.current?.();
      else if (e.key === "Escape") {
        setSelected(null);
        setTool("select");
      } else if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        remove();
      } else if (e.key === "ArrowDown" && !selected) p.navigate(1);
      else if (e.key === "ArrowUp" && !selected) p.navigate(-1);
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  });
  if (!p.active) return null;
  const draftBox: BoxAnnotation[] = state?.draft?.bbox
    ? [
        {
          id: state.draft.id,
          kind: "bbox",
          category_id: state.draft.category_id,
          iscrowd: 0,
          bbox: state.draft.bbox,
        },
      ]
    : [];
  const boxes = [
    ...(state?.annotations || []),
    ...(state?.proposals || []),
    ...draftBox,
  ];
  const object =
    chosen ||
    selectedProposal ||
    (selected === state?.draft?.id ? state?.draft : null);
  return (
    <>
      <main
        className="editor-main"
        style={{ display: p.active ? undefined : "none" }}
        aria-label="Detection editor"
      >
        <div className="editor-toolbar">
          <div className="toolbar-controls">
            <div className="tool-group">
              <IconButton
                icon={MousePointer2}
                title="Select and edit boxes · V"
                active={tool === "select"}
                onClick={() => setTool("select")}
              />
              <IconButton
                icon={Hand}
                title="Pan image"
                active={tool === "pan"}
                onClick={() => setTool("pan")}
              />
              <IconButton
                icon={BoxSelect}
                title="Draw bounding box · B"
                active={tool === "box"}
                onClick={() => {
                  setSelected(null);
                  setTool("box");
                }}
              />
              <IconButton
                icon={busy ? LoaderCircle : ScanLine}
                title="Adjust box with SAM · S"
                disabled={!canAdjust}
                onClick={() => void adjust()}
              />
              <IconButton
                icon={Plus}
                title="Positive point · P"
                active={tool === "positive"}
                disabled={!p.modelReady}
                onClick={() => setTool("positive")}
              />
              <IconButton
                icon={Minus}
                title="Negative point · E"
                active={tool === "negative"}
                disabled={!p.modelReady}
                onClick={() => setTool("negative")}
              />
            </div>
            <select
              className="detection-class"
              aria-label="Detection class"
              value={p.categoryId}
              onChange={(e) => p.setCategoryId(Number(e.target.value))}
            >
              {!p.project.categories.length && (
                <option value={0}>Create a class</option>
              )}
              {p.project.categories.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
            <IconButton
              icon={Plus}
              title="Create detection class"
              onClick={p.createClass}
            />
            <span className="toolbar-spacer" />
            <IconButton
              icon={Undo2}
              title="Undo detection"
              disabled={!entry?.detectionPast.length}
              onClick={() =>
                id !== undefined && p.workspace.historyDetection(id)
              }
            />
            <IconButton
              icon={Redo2}
              title="Redo detection"
              disabled={!entry?.detectionFuture.length}
              onClick={() =>
                id !== undefined && p.workspace.historyDetection(id, true)
              }
            />
          </div>
          <form
            className="text-prompt-bar"
            aria-label="Concept detection"
            onSubmit={(e) => {
              e.preventDefault();
              void inferText();
            }}
          >
            <Sparkles size={16} />
            <input
              aria-label="Detection prompt"
              placeholder="Describe objects to detect…"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            <select
              className="prompt-language"
              aria-label="Detection prompt language"
              value={language}
              onChange={(e) => setLanguage(e.target.value as "en" | "es")}
            >
              <option value="en">EN</option>
              <option value="es">ES</option>
            </select>
            <VisualReference
              value={visual}
              scope={`${p.project.id || p.project.directory}:${id}`}
              onChange={(value) => {
                cancelConceptSearches();
                setVisual(value);
              }}
              onPickStart={cancelConceptSearches}
              onOpenChange={setVisualOpen}
            />
            <button
              className="button primary compact"
              disabled={busy || !p.modelReady || (!text.trim() && !visual)}
              aria-label="Generate box proposals with SAM"
              title="Generate box proposals with SAM"
            >
              <Sparkles size={16} />
            </button>
          </form>
        </div>
        <div className="canvas-area">
          {p.image && state ? (
            <CanvasEditor
              key={p.image.id}
              image={p.image}
              annotations={[]}
              proposals={[]}
              draft={null}
              boxes={boxes}
              pendingBoxIds={[
                ...state.proposals.map((a) => a.id),
                ...draftBox.map((a) => a.id),
              ]}
              boxPreview={state.adjustment?.bbox}
              onBoxGeometry={changeBox}
              categories={p.project.categories}
              selected={selected || state.draft?.id || null}
              tool={tool}
              showMasks={false}
              opacity={0}
              vertex={null}
              busy={false}
              onVertex={() => {}}
              onSelect={setSelected}
              onProposal={() => {}}
              onPoint={(at, negative) => void point(at, negative)}
              onBox={draw}
              onPolygon={() => {}}
              onGeometry={() => {}}
              fitRef={fitRef}
            />
          ) : (
            <div className="canvas-placeholder">
              Select an image to annotate boxes.
            </div>
          )}
          {(state?.draft || state?.adjustment || busy) && (
            <div className="draft-floating">
              {busy && (
                <span>
                  <LoaderCircle size={14} className="spin" /> SAM is processing…
                </span>
              )}
              {state?.adjustment ? (
                <>
                  <span>Suggested adjustment</span>
                  <button
                    className="button primary compact"
                    onClick={acceptAdjustment}
                  >
                    <Check size={14} />
                    Accept adjustment
                  </button>
                  <button
                    className="button secondary compact"
                    onClick={() =>
                      update((s) => ({ ...s, adjustment: null }), false)
                    }
                  >
                    <X size={14} />
                    Keep original box
                  </button>
                </>
              ) : (
                state?.draft && (
                  <>
                    <span>Box draft</span>
                    <button
                      className="button primary compact"
                      disabled={!state.draft.bbox || busy}
                      onClick={confirm}
                    >
                      Confirm box <kbd>↵</kbd>
                    </button>
                    <button
                      className="button secondary compact"
                      onClick={() => {
                        update((s) => ({
                          ...s,
                          draft: null,
                          adjustment: null,
                        }));
                        setSelected(null);
                      }}
                    >
                      Discard draft
                    </button>
                  </>
                )
              )}
            </div>
          )}
        </div>
      </main>
      <aside
        className="inspector detection-inspector"
        style={{ display: p.active ? undefined : "none" }}
      >
        <div className="inspector-heading">
          <span>Boxes</span>
          <span className="count-pill">{state?.annotations.length || 0}</span>
        </div>
        {!!state?.proposals.length && (
          <section className="detection-section">
            <h3>SAM proposals</h3>
            {state.proposals.map((a, i) => (
              <div className="box-proposal-row" key={a.id}>
                <input
                  type="checkbox"
                  aria-label={`Accept box proposal ${i + 1}`}
                  checked={a.selected}
                  onChange={() =>
                    update(
                      (s) => ({
                        ...s,
                        proposals: s.proposals.map((v) =>
                          v.id === a.id ? { ...v, selected: !v.selected } : v,
                        ),
                      }),
                      false,
                    )
                  }
                />
                <button
                  className="text-button"
                  onClick={() => {
                    setSelected(a.id);
                    setTool("select");
                  }}
                >
                  Box {i + 1} · {Math.round(a.score * 100)}%
                </button>
                <IconButton
                  icon={X}
                  title={`Reject box proposal ${i + 1}`}
                  onClick={() =>
                    update((s) => ({
                      ...s,
                      proposals: s.proposals.filter((v) => v.id !== a.id),
                    }))
                  }
                />
              </div>
            ))}
            <button
              className="text-button"
              onClick={() =>
                update(
                  (s) => ({
                    ...s,
                    proposals: s.proposals.map((a) => ({
                      ...a,
                      selected: !s.proposals.every((v) => v.selected),
                    })),
                  }),
                  false,
                )
              }
            >
              Select all / none
            </button>
            <button
              className="button primary full compact"
              disabled={!state.proposals.some((a) => a.selected)}
              onClick={acceptProposals}
            >
              Accept selected boxes
            </button>
            <button
              className="text-button"
              onClick={() => update((s) => ({ ...s, proposals: [] }))}
            >
              Reject remaining boxes
            </button>
          </section>
        )}
        <section className="annotations-panel">
          <div className="section-caption">CONFIRMED BOXES</div>
          {state?.annotations.map((a, i) => (
            <button
              className={`annotation-row ${selected === a.id ? "active" : ""}`}
              key={a.id}
              onClick={() => {
                setSelected(a.id);
                setTool("select");
              }}
            >
              <i
                style={{
                  background: p.project.categories.find(
                    (c) => c.id === a.category_id,
                  )?.color,
                }}
              />
              <span>
                {p.project.categories.find((c) => c.id === a.category_id)?.name}
                <small>#{i + 1}</small>
              </span>
              <BoxSelect size={14} />
            </button>
          ))}
          <button
            className="text-button new-object"
            onClick={() => {
              setSelected(null);
              setTool("box");
            }}
          >
            Draw a new box
          </button>
        </section>
        {object && (
          <section className="selection-panel">
            <div className="panel-label">
              <span>Edit box</span>
              <IconButton icon={Trash2} title="Delete box" onClick={remove} />
            </div>
            <select
              aria-label="Box class"
              value={object.category_id}
              onChange={(e) => {
                const category_id = Number(e.target.value);
                update((s) => ({
                  ...s,
                  annotations: s.annotations.map((a) =>
                    a.id === object.id ? { ...a, category_id } : a,
                  ),
                  proposals: s.proposals.map((a) =>
                    a.id === object.id ? { ...a, category_id } : a,
                  ),
                  draft:
                    s.draft?.id === object.id
                      ? { ...s.draft, category_id }
                      : s.draft,
                }));
              }}
            >
              {p.project.categories.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
            <p className="field-note">
              Drag the box or its handles. Press S to ask SAM to adjust a draft
              or confirmed box.
            </p>
          </section>
        )}
        <div className="inspector-bottom">
          <section className="classes-panel">
            <div className="panel-label">
              <span>Classes</span>
              <IconButton
                icon={Plus}
                title="New class"
                onClick={p.createClass}
              />
            </div>
            <div className="category-chips">
              {p.project.categories.map((c) => (
                <button
                  key={c.id}
                  className={p.categoryId === c.id ? "active" : ""}
                  onClick={() => p.setCategoryId(c.id)}
                  onDoubleClick={() => p.editClass(c.id)}
                >
                  <i style={{ background: c.color }} />
                  {c.name}
                </button>
              ))}
            </div>
          </section>
        </div>
      </aside>
    </>
  );
}
