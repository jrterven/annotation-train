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
  Project,
  SourceImage,
  Tool,
  XY,
} from "./types";
import type { Workspace } from "./persistence";
import { uid } from "./api";
import {
  useEditorInference,
  useEditorShortcuts,
  proposalCandidates,
  acceptCandidates,
} from "./editorController";
import ClassPanel from "./ClassPanel";
import { PanelResizeHandle } from "./ResizableWorkspace";
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
  const inference = useEditorInference("detection", p.workspace);
  const { pending } = inference;
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
    const entry = p.workspace.entry(imageId),
      generation = entry?.detectionGeneration;
    return inference.begin(
      `${imageId}:${kind}`,
      imageId,
      kind,
      () => entry?.detectionGeneration === generation,
    );
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
      const result = await request.points({
        id: original.id,
        points: [],
        box: xyxy(bbox),
      });
      if (!result || !stillMatches()) return;
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
      request.error(e, p.onError);
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
      const result = await request.points({
        id: draft.id,
        points: draft.points,
        ...(draft.prompt_bbox ? { box: xyxy(draft.prompt_bbox) } : {}),
      });
      if (!result || !stillMatches()) return;
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
      request.error(e, p.onError);
    } finally {
      request.finish();
    }
  }
  function cancelConceptSearches() {
    inference.cancel((request) => request.kind === "concept");
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
      const result = await request.concept(text, language, category, visual);
      if (
        !result ||
        signature(p.workspace.getDetection(imageId)?.proposals) !== original
      )
        return;
      const proposals = proposalCandidates(
        "detection",
        result.proposals,
        category,
      );
      p.workspace.updateDetection(imageId, (s) => ({ ...s, proposals }));
      if (!proposals.length)
        p.onError("No boxes found. Try a different prompt.");
    } catch (e) {
      request.error(e, p.onError);
    } finally {
      request.finish();
    }
  }
  function acceptProposals() {
    update((s) => {
      const { accepted, remaining } = acceptCandidates(s.proposals, (p) => p);
      return {
        ...s,
        annotations: [...s.annotations, ...accepted],
        proposals: remaining,
      };
    });
    setSelected(null);
  }
  useEditorShortcuts({
    active: p.active,
    blocked: p.blocked || visualOpen,
    selected,
    onHistory: (redo) => {
      if (id !== undefined) {
        inference.cancel((r) => r.imageId === id);
        p.workspace.historyDetection(id, redo);
      }
    },
    onConfirm: confirm,
    onAdjust: () => void adjust(),
    onDelete: remove,
    onEscape: () => {
      setSelected(null);
      setTool("select");
    },
    onNavigate: p.navigate,
    onFit: () => fitRef.current?.(),
    onTool: (tool) => {
      if (tool === "box") setSelected(null);
      setTool(tool);
    },
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
      {p.active && <PanelResizeHandle side="right" />}
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
          <ClassPanel
            categories={p.project.categories}
            categoryId={p.categoryId}
            onSelect={p.setCategoryId}
            onEdit={p.editClass}
            onCreate={p.createClass}
            createTitle="New class"
          />
        </div>
      </aside>
    </>
  );
}
