import { useEffect, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpRight,
  BoxSelect,
  Check,
  CheckCheck,
  ChevronDown,
  ChevronRight,
  Circle,
  Command,
  Download,
  Eye,
  EyeOff,
  FolderOpen,
  Hand,
  HelpCircle,
  ImagePlus,
  Layers3,
  LoaderCircle,
  Minus,
  Merge,
  MousePointer2,
  PanelRightClose,
  PanelRightOpen,
  Plus,
  Redo2,
  Save,
  Sparkles,
  Trash2,
  Undo2,
  X,
  AlertTriangle,
  RotateCcw,
  Link2,
  SlidersHorizontal,
  ScanLine,
  Pentagon,
} from "lucide-react";
import IconButton from "./IconButton";
import AccountAvatar from "./AccountAvatar";
import ResizableWorkspace, { PanelResizeHandle } from "./ResizableWorkspace";
import {
  api,
  assetURL,
  errorText,
  jobChangedEvent,
  selectProject,
  uid,
} from "./api";
import type { JobProgress } from "./api";
import { sourceImages, imageCacheBudget } from "./imageCache";
import type {
  SegmentationAnnotation as Annotation,
  Category,
  Component,
  Draft,
  GeometryResult,
  HostedSession,
  ImageState,
  ModelStatus,
  Part,
  Project,
  Proposal,
  Tool,
  Task,
  XY,
} from "./types";
import { useWorkspace } from "./persistence";
import CanvasEditor from "./CanvasEditor";
import DetectionEditor from "./DetectionEditor";
import {
  useEditorInference,
  useEditorShortcuts,
  proposalCandidates,
  acceptCandidates,
  type TextPrompt,
} from "./editorController";
import ClassPanel from "./ClassPanel";
import ExportDialog, { downloadExport } from "./ExportDialog";
import type { Vertex } from "./CanvasEditor";
import { FileBrowser, ProjectDialog } from "./ProjectDialog";
import { HostedProjects, HostedUploads } from "./HostedProjects";
import VisualReference from "./VisualReference";
import type { VisualExample } from "./VisualReference";
const PALETTE = [
  "#7856e8",
  "#16a577",
  "#e87c26",
  "#397be6",
  "#df4d6d",
  "#cfaa13",
  "#119ca8",
  "#b84dcc",
];
const signature = (part: Part) =>
  JSON.stringify({
    points: part.points,
    box: part.box,
    seed_mask: part.seed_mask,
    polygon: part.polygon,
  });
export default function App({
  hostedSession,
  onLogout,
}: {
  hostedSession?: HostedSession;
  onLogout?: () => Promise<void>;
} = {}) {
  const hosted = !!hostedSession;
  const [project, setProject] = useState<Project | null>(null);
  const [imageId, setImageId] = useState<number | null>(null);
  const [projectDialog, setProjectDialog] = useState(hosted);
  const [uploadDialog, setUploadDialog] = useState(false);
  const [jobs, setJobs] = useState<Map<string, JobProgress>>(new Map());
  const [loggingOut, setLoggingOut] = useState(false);
  const [relink, setRelink] = useState(false);
  const [booting, setBooting] = useState(true);
  const [loadingImage, setLoadingImage] = useState(false);
  const [model, setModel] = useState<ModelStatus>({
    state: "unloaded",
    message: "SAM 3 is ready to load",
  });
  const [modelLoading, setModelLoading] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [task, setTask] = useState<Task>("segmentation");
  const [showBoxes, setShowBoxes] = useState(false);
  const [exportDialog, setExportDialog] = useState(false);
  const [exportName, setExportName] = useState("");
  const [tool, setTool] = useState<Tool>("positive");
  const [selected, setSelected] = useState<string | null>(null);
  const [vertex, setVertex] = useState<Vertex | null>(null);
  const [categoryId, setCategoryId] = useState<number>(0);
  const [inspector, setInspector] = useState(true);
  const [showMasks, setShowMasks] = useState(true);
  const [opacity, setOpacity] = useState(0.65);
  const [text, setText] = useState("");
  const [promptLanguage, setPromptLanguage] = useState<"en" | "es">("en");
  const [visualExample, setVisualExample] = useState<VisualExample | null>(
    null,
  );
  const [visualDialogOpen, setVisualDialogOpen] = useState(false);
  const visualRevision = useRef(0);
  const [textPrompts, setTextPrompts] = useState<Record<number, TextPrompt>>(
    {},
  );
  const [help, setHelp] = useState(false);
  const [categoryDialog, setCategoryDialog] = useState<Category | "new" | null>(
    null,
  );
  const [categoryName, setCategoryName] = useState("");
  const [categoryColor, setCategoryColor] = useState(PALETTE[0]);
  const [categoryBusy, setCategoryBusy] = useState(false);
  const [query, setQuery] = useState("");
  const [exportPath, setExportPath] = useState("");
  const [conflictDialog, setConflictDialog] = useState(false);
  const [conflictImage, setConflictImage] = useState<number | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [geometryBusy, setGeometryBusy] = useState(false);
  const [mergingProposals, setMergingProposals] = useState(false);
  const [maxHoleArea, setMaxHoleArea] = useState("16");
  const [workVersion, setWorkVersion] = useState(0);
  const partErrors = useRef(new Map<string, string>());
  const fitRef = useRef<(() => void) | null>(null);
  const geometryToken = useRef("");
  const sessionToken = useRef(0);
  const activeImage = useRef(imageId);
  activeImage.current = imageId;
  const workspace = useWorkspace(setToast);
  const inference = useEditorInference("segmentation", workspace);
  const { pending } = inference;
  const currentImage = project?.images.find((i) => i.id === imageId);
  const state = imageId === null ? undefined : workspace.get(imageId);
  const entry = imageId === null ? undefined : workspace.entry(imageId);
  const selectedAnnotation = state?.annotations.find((a) => a.id === selected);
  const selectedCategory = project?.categories.find((c) => c.id === categoryId);
  const currentPending = [...pending.current.values()].filter(
    (p) => p.imageId === imageId,
  );
  const textBusy = currentPending.some((p) => p.kind === "text");
  const pointBusy = currentPending.some((p) => p.kind === "points");
  const anyDirty = [...workspace.records.values()].some((e) => e.dirty);
  const anySaving = [...workspace.records.values()].some((e) => e.saving);
  const failedRecord = [...workspace.records.entries()].find(
    ([, e]) => e.error,
  );
  const imageIndex = project?.images.findIndex((i) => i.id === imageId) ?? -1;
  const visibleImages =
    project?.images.filter((i) =>
      i.file_name.toLowerCase().includes(query.toLowerCase()),
    ) || [];
  const selectedProposals = state?.proposals.filter((p) => p.selected) || [];
  const activePart = state?.draft?.parts.find(
    (part) => part.id === state.draft?.active_part_id,
  );
  const activePolygon = activePart?.polygon;
  const currentTextPrompt = imageId === null ? undefined : textPrompts[imageId];
  const holeArea = Number(maxHoleArea);
  const validHoleArea =
    Number.isSafeInteger(holeArea) &&
    holeArea >= 1 &&
    holeArea <= (currentImage ? currentImage.width * currentImage.height : 0);
  function invalidate() {
    setWorkVersion((v) => v + 1);
  }
  function abortAll() {
    inference.cancel();
    partErrors.current.clear();
    invalidate();
  }
  function openProject(result: Project) {
    abortAll();
    workspace.reset();
    sessionToken.current++;
    selectProject(result);
    setProject(result);
    setImageId(result.images[0]?.id ?? null);
    setCategoryId(result.categories[0]?.id ?? 0);
    setSelected(null);
    setVertex(null);
    setTool("positive");
    setTask("segmentation");
    setShowBoxes(false);
    setExportPath("");
    setQuery("");
    setTextPrompts({});
    setText("");
    setVisualExample(null);
    visualRevision.current++;
    setVisualDialogOpen(false);
    setProjectDialog(false);
  }
  useEffect(() => {
    if (!hosted) return;
    const change = (event: Event) => {
      const job = (event as CustomEvent<JobProgress>).detail;
      if (job.status === "cancel_unconfirmed")
        setToast(
          "Cancellation could not be confirmed. The request may still finish and count toward your daily limit.",
        );
      setJobs((old) => {
        const next = new Map(old);
        if (["finished", "cancel_unconfirmed"].includes(job.status))
          next.delete(job.id);
        else next.set(job.id, job);
        return next;
      });
    };
    window.addEventListener(jobChangedEvent, change);
    return () => window.removeEventListener(jobChangedEvent, change);
  }, [hosted]);
  useEffect(
    () => () => {
      geometryToken.current = "";
    },
    [],
  );
  useEffect(() => {
    let active = true;
    void api<Project | null>("/project")
      .then((p) => {
        if (active && p) {
          setProject(p);
          setImageId(p.images[0]?.id ?? null);
          setCategoryId(p.categories[0]?.id ?? 0);
        }
      })
      .catch(() => {})
      .finally(() => {
        if (active) setBooting(false);
      });
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    let active = true;
    const check = () =>
      void api<{ model: ModelStatus }>("/health")
        .then((h) => {
          if (active) setModel(h.model);
        })
        .catch(() => {
          if (active)
            setModel({
              state: "error",
              message: hosted
                ? "Could not connect to the server. Manual annotation is still available."
                : "Could not connect to the local server.",
            });
        });
    check();
    const interval = setInterval(check, 10000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);
  useEffect(() => {
    if (imageId === null) return;
    let active = true;
    setLoadingImage(true);
    void workspace
      .load(imageId)
      .catch((e) => {
        if (active) setToast(errorText(e));
      })
      .finally(() => {
        if (active) setLoadingImage(false);
      });
    setSelected(null);
    setVertex(null);
    geometryToken.current = "";
    setGeometryBusy(false);
    return () => {
      active = false;
    };
  }, [imageId, project?.id, project?.directory, sessionToken.current]);
  useEffect(() => {
    if (!hosted || !currentImage || !project) return;
    let active = true;
    const next = project.images[imageIndex + 1];
    const currentURL = assetURL(`/images/${currentImage.id}/file`);
    const nextURL = next ? assetURL(`/images/${next.id}/file`) : null;
    // Start the source fetch alongside annotation loading. Preload only one
    // neighbor after the selected image is decoded, within the memory budget.
    void sourceImages
      .load(currentURL)
      .then(() => {
        if (
          active &&
          nextURL &&
          next &&
          4 *
            (currentImage.width * currentImage.height +
              next.width * next.height) <=
            imageCacheBudget
        )
          void sourceImages.load(nextURL).catch(() => {});
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, [hosted, imageId, project?.id, project?.images.length]);
  useEffect(() => {
    if (!toast) return;
    const timeout = setTimeout(() => setToast(null), 6500);
    return () => clearTimeout(timeout);
  }, [toast]);
  function categoryRequired() {
    if (project?.categories.some((c) => c.id === categoryId)) return true;
    setCategoryDialog("new");
    setCategoryName("");
    setCategoryColor(PALETTE[project?.categories.length || 0]);
    setToast("Create a class to start annotating.");
    return false;
  }
  function update(fn: (s: ImageState) => ImageState, history = true) {
    if (imageId !== null) workspace.update(imageId, fn, history);
  }
  async function navigateTo(id: number) {
    if (geometryBusy || confirming) {
      setToast("Wait for the mask update to finish.");
      return;
    }
    try {
      await workspace.flush();
      setImageId(id);
      setTool("select");
    } catch (e) {
      setToast(errorText(e));
    }
  }
  function navigate(target: number) {
    if (!project || target < 0 || target >= project.images.length) return;
    void navigateTo(project.images[target].id);
  }
  async function ensureLoad() {
    setModelLoading(true);
    try {
      const status = await api<ModelStatus>("/model/load", "POST", {});
      setModel(status);
    } catch (e) {
      setToast(errorText(e));
    } finally {
      setModelLoading(false);
    }
  }
  function prepareDraft(s: ImageState): Draft {
    if (s.draft) return structuredClone(s.draft);
    const part = { id: uid(), points: [] };
    return {
      id: uid(),
      category_id: categoryId,
      parts: [part],
      active_part_id: part.id,
    };
  }
  async function inferPart(id: number, draftId: string, part: Part) {
    const key = `part:${id}:${part.id}`;
    partErrors.current.delete(key);
    let captured = signature(part);
    const request = inference.begin(key, id, "points", () => {
      const fresh = workspace.get(id);
      const freshPart = fresh?.draft?.parts.find((p) => p.id === part.id);
      return (
        fresh?.draft?.id === draftId &&
        !!freshPart &&
        signature(freshPart) === captured
      );
    });
    const { controller, isCurrent } = request;
    try {
      if (part.polygon && !part.seed_mask) {
        if (!part.polygon.closed)
          throw new Error("Close the polygon before refining it.");
        const seed = await api<GeometryResult>(
          "/geometry",
          "POST",
          {
            image_id: id,
            components: [{ outer: part.polygon.vertices, holes: [] }],
          },
          controller.signal,
        );
        if (!isCurrent()) return;
        part = { ...part, seed_mask: seed.mask };
        captured = signature(part);
        workspace.update(
          id,
          (s) => ({
            ...s,
            draft: s.draft
              ? {
                  ...s.draft,
                  parts: s.draft.parts.map((p) =>
                    p.id === part.id ? { ...p, seed_mask: seed.mask } : p,
                  ),
                }
              : null,
          }),
          false,
        );
      }
      const result = await request.points(part);
      if (!result) return;
      workspace.update(
        id,
        (s) => ({
          ...s,
          draft: s.draft
            ? {
                ...s.draft,
                parts: s.draft.parts.map((p) =>
                  p.id === part.id
                    ? {
                        ...p,
                        mask: result.mask,
                        components: result.components,
                        controls: result.controls,
                        preview: result.preview,
                      }
                    : p,
                ),
              }
            : null,
        }),
        false,
      );
      if (
        part.polygon &&
        activeImage.current === id &&
        workspace.get(id)?.draft?.active_part_id === part.id
      )
        setTool((current) => (current === "polygon" ? "positive" : current));
    } catch (e) {
      request.error(e, (message) => {
        partErrors.current.set(key, message);
        setToast(`SAM 3: ${message}`);
      });
    } finally {
      request.finish();
      invalidate();
    }
  }
  function changePolygon(vertices: XY[], closed: boolean) {
    if (imageId === null || !state || confirming || !categoryRequired()) return;
    const draft = prepareDraft(state);
    let index = draft.parts.findIndex((p) => p.id === draft.active_part_id);
    let original = draft.parts[index];
    // A polygon starts a separate part if the current part already has prompts.
    if (
      !original.polygon &&
      (original.points.length ||
        original.box ||
        original.seed_mask ||
        original.mask)
    ) {
      original = { id: uid(), points: [] };
      draft.parts.push(original);
      index = draft.parts.length - 1;
      draft.active_part_id = original.id;
    }
    const key = `part:${imageId}:${original.id}`;
    inference.cancel((_request, pendingKey) => pendingKey === key);
    partErrors.current.delete(key);
    draft.parts[index] = {
      id: original.id,
      points: [],
      polygon: { vertices, closed },
    };
    update((s) => ({ ...s, draft }));
    setSelected(null);
    setVertex(null);
    invalidate();
  }
  function closePolygon() {
    if (!activePolygon || activePolygon.closed) return;
    if (activePolygon.vertices.length < 3) {
      setToast("A polygon needs at least three vertices.");
      return;
    }
    changePolygon(activePolygon.vertices, true);
  }
  function refinePolygon() {
    if (
      imageId === null ||
      !state?.draft ||
      !activePart?.polygon?.closed ||
      pointBusy ||
      confirming ||
      geometryBusy ||
      (hosted && !["ready", "loaded"].includes(model.state))
    )
      return;
    void inferPart(imageId, state.draft.id, activePart);
  }
  async function usePolygon() {
    if (
      imageId === null ||
      !state?.draft ||
      !activePart?.polygon?.closed ||
      pointBusy ||
      geometryBusy ||
      confirming
    )
      return;
    const id = imageId,
      draftId = state.draft.id,
      part = activePart;
    const captured = signature(part),
      session = sessionToken.current,
      token = uid();
    geometryToken.current = token;
    setGeometryBusy(true);
    try {
      const result = await api<GeometryResult>("/geometry", "POST", {
        image_id: id,
        components: [{ outer: part.polygon!.vertices, holes: [] }],
      });
      const fresh = workspace.get(id);
      const freshPart = fresh?.draft?.parts.find(
        (candidate) => candidate.id === part.id,
      );
      if (
        geometryToken.current !== token ||
        sessionToken.current !== session ||
        fresh?.draft?.id !== draftId ||
        !freshPart ||
        signature(freshPart) !== captured
      )
        return;
      workspace.update(id, (current) => ({
        ...current,
        draft: current.draft
          ? {
              ...current.draft,
              parts: current.draft.parts.map((candidate) =>
                candidate.id === part.id
                  ? {
                      ...candidate,
                      ...result,
                      seed_mask: result.mask,
                    }
                  : candidate,
              ),
            }
          : null,
      }));
      if (activeImage.current === id && fresh.draft.active_part_id === part.id)
        setTool("select");
    } catch (error) {
      if (geometryToken.current === token) setToast(errorText(error));
    } finally {
      if (geometryToken.current === token) setGeometryBusy(false);
    }
  }
  function resumeDraft() {
    if (!categoryRequired()) return;
    setSelected(null);
    setVertex(null);
    setTool(
      activePolygon && (!activePolygon.closed || !activePart?.seed_mask)
        ? "polygon"
        : "positive",
    );
  }
  function addPrompt(
    point?: XY,
    negative = false,
    box?: [number, number, number, number],
  ) {
    if (imageId === null || !state || confirming || !categoryRequired()) return;
    const draft = prepareDraft(state);
    const index = draft.parts.findIndex((p) => p.id === draft.active_part_id);
    const original = draft.parts[index];
    if (original.polygon && !original.seed_mask) {
      setToast(
        "Close the polygon, then choose Use polygon or Refine with SAM before adding corrections.",
      );
      return;
    }
    const next = {
      ...original,
      ...(point
        ? {
            points: [
              ...original.points,
              { x: point[0], y: point[1], label: (negative ? 0 : 1) as 0 | 1 },
            ],
          }
        : {}),
      ...(box ? { box } : {}),
    };
    draft.parts[index] = {
      ...next,
      mask: undefined,
      components: undefined,
      controls: undefined,
      preview: undefined,
    };
    update((s) => ({ ...s, draft }));
    setSelected(null);
    setVertex(null);
    void inferPart(imageId, draft.id, next);
  }
  function addPart() {
    if (!state || !categoryRequired()) return;
    const draft = prepareDraft(state);
    const part = { id: uid(), points: [] };
    draft.parts.push(part);
    draft.active_part_id = part.id;
    update((s) => ({ ...s, draft }));
    setTool("positive");
    setSelected(null);
  }
  function removePart(id: string) {
    if (!state?.draft) return;
    const key = `part:${imageId}:${id}`;
    inference.cancel((_request, pendingKey) => pendingKey === key);
    partErrors.current.delete(key);
    invalidate();
    update((s) => {
      if (!s.draft) return s;
      const parts = s.draft.parts.filter((p) => p.id !== id);
      return {
        ...s,
        draft: parts.length
          ? {
              ...s.draft,
              parts,
              active_part_id: parts.some(
                (p) => p.id === s.draft!.active_part_id,
              )
                ? s.draft.active_part_id
                : parts[0].id,
            }
          : null,
      };
    });
  }
  function discardDraft() {
    if (!state?.draft) return;
    for (const part of state.draft.parts) {
      const key = `part:${imageId}:${part.id}`;
      inference.cancel((_request, pendingKey) => pendingKey === key);
      partErrors.current.delete(key);
    }
    invalidate();
    update((s) => ({ ...s, draft: null }));
    setTool("select");
  }
  async function confirmDraft() {
    if (imageId === null || !state?.draft || confirming || geometryBusy) return;
    if (
      !pointBusy &&
      state.draft.parts.some((part) => part.polygon && !part.mask)
    ) {
      setToast("Choose Use polygon or refine with SAM before confirming.");
      return;
    }
    if (pointBusy || state.draft.parts.some((p) => !p.mask)) {
      setToast("Wait for the latest mask; complete or remove empty parts.");
      return;
    }
    const id = imageId,
      draft = state.draft,
      signatureBefore = JSON.stringify(draft);
    setConfirming(true);
    try {
      const masks = draft.parts.map((p) => p.mask!);
      const result = await api<GeometryResult>("/masks/union", "POST", {
        image_id: id,
        masks,
      });
      if (JSON.stringify(workspace.get(id)?.draft) !== signatureBefore) return;
      const annotation: Annotation = {
        id: uid(),
        category_id: draft.category_id,
        ...result,
        iscrowd: 0,
      };
      workspace.update(id, (s) => ({
        ...s,
        annotations: [...s.annotations, annotation],
        draft: null,
      }));
      if (imageId === id) {
        setSelected(annotation.id);
        setTool("select");
        setVertex(null);
      }
    } catch (e) {
      setToast(errorText(e));
    } finally {
      setConfirming(false);
    }
  }
  function cancelProposalSearches() {
    visualRevision.current++;
    inference.cancel((request) => request.kind === "text");
    invalidate();
  }
  function changeVisualExample(value: VisualExample | null) {
    cancelProposalSearches();
    setVisualExample(value);
  }
  async function inferText() {
    if (
      imageId === null ||
      !state ||
      mergingProposals ||
      (!text.trim() && !visualExample) ||
      !categoryRequired()
    )
      return;
    const id = imageId,
      key = `text:${id}`;
    const currentSession = sessionToken.current,
      referenceRevision = visualRevision.current;
    const original = JSON.stringify(state.proposals);
    const request = inference.begin(
      key,
      id,
      "text",
      () =>
        currentSession === sessionToken.current &&
        referenceRevision === visualRevision.current &&
        JSON.stringify(workspace.get(id)?.proposals) === original,
    );
    try {
      const result = await request.concept(
        text,
        promptLanguage,
        categoryId,
        visualExample,
      );
      if (!result) return;
      workspace.update(id, (s) => ({
        ...s,
        proposals: proposalCandidates(
          "segmentation",
          result.proposals,
          categoryId,
        ),
      }));
      const usedPrompt = result.prompt;
      setTextPrompts((prompts) => {
        const next = { ...prompts };
        if (usedPrompt) next[id] = usedPrompt;
        else delete next[id];
        return next;
      });
      if (activeImage.current === id) {
        setSelected(null);
        setTool("select");
      }
      if (!result.proposals.length)
        setToast(
          visualExample
            ? "No objects found for this visual example. Try a tighter crop or add a short description."
            : usedPrompt?.source_language === "es"
              ? `No objects found for “${usedPrompt.english}”. Try a different description.`
              : "No objects found. Try a short description in English.",
        );
    } catch (e) {
      request.error(e, (message) => setToast(`SAM 3: ${message}`));
    } finally {
      request.finish();
    }
  }
  function acceptProposals() {
    if (
      geometryBusy ||
      confirming ||
      !categoryRequired() ||
      !selectedProposals.length
    )
      return;
    update((s) => {
      const { accepted, remaining } = acceptCandidates(s.proposals, (p) => ({
        ...p,
        id: uid(),
        category_id: categoryId,
      }));
      return {
        ...s,
        annotations: [...s.annotations, ...accepted],
        proposals: remaining,
      };
    });
    setTool("select");
  }
  async function mergeProposals() {
    if (
      imageId === null ||
      !state ||
      selectedProposals.length < 2 ||
      geometryBusy ||
      confirming ||
      textBusy ||
      !categoryRequired()
    )
      return;
    const image = imageId,
      session = sessionToken.current,
      token = uid(),
      sourceEntry = workspace.entry(image),
      proposalsSnapshot = JSON.stringify(state.proposals),
      selectedIds = new Set(selectedProposals.map((p) => p.id)),
      category = categoryId;
    geometryToken.current = token;
    setGeometryBusy(true);
    setMergingProposals(true);
    try {
      const result = await api<GeometryResult>("/masks/union", "POST", {
        image_id: image,
        masks: selectedProposals.map((p) => p.mask),
      });
      if (
        geometryToken.current !== token ||
        sessionToken.current !== session ||
        workspace.entry(image) !== sourceEntry
      )
        return;
      if (
        JSON.stringify(workspace.get(image)?.proposals) !== proposalsSnapshot
      ) {
        setToast("Proposals changed. Select them again to merge.");
        return;
      }
      const annotation: Annotation = {
        ...result,
        id: uid(),
        category_id: category,
        iscrowd: 0,
      };
      // One history entry consumes only the selected proposals. Keep any draft
      // and unrelated annotations, including inference completed in the meantime.
      workspace.update(image, (s) => ({
        ...s,
        annotations: [...s.annotations, annotation],
        proposals: s.proposals.filter((p) => !selectedIds.has(p.id)),
      }));
      if (activeImage.current === image) {
        setSelected(annotation.id);
        setVertex(null);
        setTool("select");
      }
    } catch (e) {
      if (geometryToken.current === token && sessionToken.current === session)
        setToast(errorText(e));
    } finally {
      if (geometryToken.current === token) setGeometryBusy(false);
      setMergingProposals(false);
    }
  }
  function refineProposal(proposal: Proposal) {
    if (geometryBusy || confirming) return;
    if (state?.draft) {
      setToast(
        "Confirm or discard the current draft before refining another proposal.",
      );
      return;
    }
    const part: Part = {
      id: uid(),
      points: [],
      seed_mask: proposal.mask,
      mask: proposal.mask,
      components: proposal.components,
      controls: proposal.controls,
      preview: proposal.preview,
    };
    update((s) => ({
      ...s,
      proposals: s.proposals.filter((p) => p.id !== proposal.id),
      draft: {
        id: uid(),
        category_id: project?.categories.some((c) => c.id === categoryId)
          ? categoryId
          : proposal.category_id,
        parts: [part],
        active_part_id: part.id,
      },
    }));
    setSelected(null);
    setTool("positive");
  }
  async function editGeometry(id: string, components: Component[]) {
    if (imageId === null) return;
    const image = imageId,
      token = uid();
    geometryToken.current = token;
    setGeometryBusy(true);
    try {
      const result = await api<GeometryResult>("/geometry", "POST", {
        image_id: image,
        components,
      });
      if (geometryToken.current !== token) return;
      workspace.update(image, (s) => ({
        ...s,
        annotations: s.annotations.map((a) =>
          a.id === id ? { ...a, ...result } : a,
        ),
      }));
    } catch (e) {
      if (geometryToken.current === token) {
        setToast(errorText(e));
        setSelected(null);
        setVertex(null);
      }
    } finally {
      if (geometryToken.current === token) setGeometryBusy(false);
    }
  }
  async function fillSmallHoles() {
    if (
      imageId === null ||
      !selectedAnnotation ||
      !validHoleArea ||
      geometryBusy ||
      confirming
    )
      return;
    const image = imageId,
      annotation = selectedAnnotation,
      token = uid(),
      session = sessionToken.current;
    const geometrySnapshot = (a: Annotation) =>
      JSON.stringify([a.mask, a.components, a.controls]);
    const original = geometrySnapshot(annotation);
    geometryToken.current = token;
    setGeometryBusy(true);
    try {
      const result = await api<
        GeometryResult & { filled_holes: number; filled_pixels: number }
      >("/masks/fill-holes", "POST", {
        image_id: image,
        mask: annotation.mask,
        max_area: holeArea,
      });
      if (geometryToken.current !== token || sessionToken.current !== session)
        return;
      const current = workspace
        .get(image)
        ?.annotations.find((a) => a.id === annotation.id);
      if (!current || geometrySnapshot(current) !== original) return;
      if (!result.filled_holes) {
        setToast(`No holes at or below ${holeArea} px².`);
        return;
      }
      const { filled_holes, filled_pixels, ...geometry } = result;
      workspace.update(image, (s) => ({
        ...s,
        annotations: s.annotations.map((a) =>
          a.id === annotation.id ? { ...a, ...geometry } : a,
        ),
      }));
      setVertex(null);
      setToast(
        `Filled ${filled_holes} ${filled_holes === 1 ? "hole" : "holes"} (${filled_pixels} ${filled_pixels === 1 ? "pixel" : "pixels"}).`,
      );
    } catch (e) {
      if (geometryToken.current === token) setToast(errorText(e));
    } finally {
      if (geometryToken.current === token) setGeometryBusy(false);
    }
  }
  function removeSelected() {
    if (geometryBusy) return;
    if (selectedAnnotation && vertex) {
      const components = structuredClone(
        selectedAnnotation.controls ?? selectedAnnotation.components,
      );
      const ring =
        vertex.ring === 0
          ? components[vertex.component].outer
          : components[vertex.component].holes[vertex.ring - 1];
      if (ring.length <= 3) {
        setToast("A contour needs at least three vertices.");
        return;
      }
      ring.splice(vertex.index, 1);
      setVertex(null);
      void editGeometry(selectedAnnotation.id, components);
    } else if (selected) {
      update((s) => ({
        ...s,
        annotations: s.annotations.filter((a) => a.id !== selected),
      }));
      setSelected(null);
      setVertex(null);
    }
  }
  function undo(redo = false) {
    if (imageId === null || geometryBusy || confirming) return;
    inference.cancel((request) => request.imageId === imageId);
    invalidate();
    workspace.history(imageId, redo);
    const restored = workspace.get(imageId)?.draft;
    if (restored) {
      for (const part of restored.parts) {
        if (!part.mask && (part.points.length || part.box || part.seed_mask))
          void inferPart(imageId, restored.id, part);
      }
    }
    setSelected(null);
    setVertex(null);
  }
  async function saveAll() {
    if (geometryBusy || confirming) {
      setToast("Wait for the mask update to finish.");
      return;
    }
    try {
      await workspace.flush();
      setToast("All changes saved.");
    } catch (e) {
      setToast(errorText(e));
    }
  }
  async function switchProject() {
    if (geometryBusy || confirming) {
      setToast("Wait for the mask update to finish.");
      return;
    }
    try {
      await workspace.flush();
      if (hosted) abortAll();
      setProjectDialog(true);
    } catch (e) {
      setToast(errorText(e));
    }
  }
  async function showUploads() {
    if (geometryBusy || confirming) {
      setToast("Wait for the mask update to finish.");
      return;
    }
    try {
      await workspace.flush();
      abortAll();
      setUploadDialog(true);
    } catch (e) {
      setToast(errorText(e));
    }
  }
  async function logout() {
    if (!onLogout || loggingOut) return;
    if (geometryBusy || confirming) {
      setToast("Wait for the mask update to finish.");
      return;
    }
    setLoggingOut(true);
    try {
      await workspace.flush();
      abortAll();
      await onLogout();
    } catch (e) {
      setToast(errorText(e));
    } finally {
      setLoggingOut(false);
    }
  }
  function editCategory(category: Category | "new") {
    setCategoryDialog(category);
    setCategoryName(category === "new" ? "" : category.name);
    setCategoryColor(
      category === "new"
        ? PALETTE[(project?.categories.length || 0) % PALETTE.length]
        : category.color,
    );
  }
  async function saveCategory() {
    if (!categoryName.trim() || !categoryDialog) return;
    setCategoryBusy(true);
    try {
      const cat = await api<Category>(
        categoryDialog === "new"
          ? "/project/categories"
          : `/project/categories/${categoryDialog.id}`,
        categoryDialog === "new" ? "POST" : "PATCH",
        { name: categoryName.trim(), color: categoryColor },
      );
      setProject((p) =>
        p
          ? {
              ...p,
              categories:
                categoryDialog === "new"
                  ? [...p.categories, cat]
                  : p.categories.map((c) => (c.id === cat.id ? cat : c)),
            }
          : p,
      );
      setCategoryId(cat.id);
      setCategoryDialog(null);
    } catch (e) {
      setToast(errorText(e));
    } finally {
      setCategoryBusy(false);
    }
  }
  function downloadLocal() {
    const recoveryId = conflictImage ?? imageId;
    const recovery =
      recoveryId === null ? undefined : workspace.snapshot(recoveryId);
    if (!recovery) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(recovery, null, 2)], {
        type: "application/json",
      }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `image-recovery-${recoveryId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
  useEditorShortcuts({
    active: task === "segmentation",
    selected,
    blocked: !!(
      projectDialog ||
      help ||
      categoryDialog ||
      relink ||
      conflictDialog ||
      uploadDialog ||
      visualDialogOpen ||
      exportDialog
    ),
    onSave: () => void saveAll(),
    onHistory: undo,
    onConfirm: () => {
      if (activePolygon && !activePolygon.closed) closePolygon();
      else void confirmDraft();
    },
    onEscape: () => {
      setSelected(null);
      setVertex(null);
      setTool("select");
    },
    onDelete: removeSelected,
    onNavigate: (offset) => navigate(imageIndex + offset),
    onTool: setTool,
    onFit: () => fitRef.current?.(),
    extra: {
      n: resumeDraft,
      g: () => {
        setTool("polygon");
        setSelected(null);
        setVertex(null);
      },
      h: () => setShowMasks((v) => !v),
      "?": () => setHelp(true),
    },
  });
  const modelReady = ["ready", "loaded"].includes(model.state);
  const readyToConfirm =
    !!state?.draft &&
    state.draft.parts.every((p) => p.mask) &&
    !pointBusy &&
    !geometryBusy &&
    !confirming;
  return (
    <div className="application" data-work-version={workVersion}>
      <header className="app-header">
        <a
          className="brand"
          href="#"
          onClick={(e) => e.preventDefault()}
          aria-label="Annotation and Training"
        >
          <span className="brand-icon">
            <ScanLine size={23} strokeWidth={1.4} />
          </span>
          <span>Annotation and Training</span>
        </a>
        <span className="header-divider" />
        {project ? (
          <button
            className="project-switch"
            title="Switch project"
            onClick={() => void switchProject()}
          >
            <FolderOpen size={15} />
            <span>{project.name}</span>
            <ChevronDown size={14} />
          </button>
        ) : (
          <span className="header-tagline">Image annotation</span>
        )}
        {project && (
          <div
            className="task-selector"
            role="group"
            aria-label="Annotation task"
          >
            <button
              aria-pressed={task === "segmentation"}
              title="Segmentation: masks and polygons"
              onClick={() => setTask("segmentation")}
            >
              <Pentagon size={16} />
              <span>Segmentation</span>
            </button>
            <button
              aria-pressed={task === "detection"}
              title="Detection: bounding boxes"
              onClick={() => setTask("detection")}
            >
              <BoxSelect size={16} />
              <span>Detection</span>
            </button>
          </div>
        )}
        <div className="header-actions">
          {hosted && (
            <details className="hosted-account">
              <summary
                aria-label={`Google account: ${hostedSession.user?.email || "Account"}`}
                title={hostedSession.user?.name || hostedSession.user?.email}
              >
                <AccountAvatar user={hostedSession.user} />
              </summary>
              <div className="hosted-account-menu">
                <strong>{hostedSession.user?.email}</strong>
                {hostedSession.usage && (
                  <>
                    <p>
                      {(hostedSession.usage.storage_bytes / 1000000).toFixed(1)}{" "}
                      /{" "}
                      {(
                        hostedSession.usage.storage_limit_bytes / 1000000
                      ).toFixed(0)}{" "}
                      MB stored
                    </p>
                    <progress
                      value={hostedSession.usage.storage_bytes}
                      max={hostedSession.usage.storage_limit_bytes}
                      aria-label="Storage used"
                    />
                    <small>Images, thumbnails, and annotation data.</small>
                    <p>
                      {hostedSession.usage.inferences_used} /{" "}
                      {hostedSession.usage.inference_limit} SAM 3 requests today
                    </p>
                    <small>Daily limit resets at midnight UTC.</small>
                  </>
                )}
                <button
                  className="button secondary full"
                  disabled={loggingOut}
                  onClick={() => void logout()}
                >
                  Sign out
                </button>
                <nav>
                  <a href="/privacy" target="_blank" rel="noopener noreferrer">
                    Privacy
                  </a>
                  <a href="/terms" target="_blank" rel="noopener noreferrer">
                    Terms
                  </a>
                </nav>
              </div>
            </details>
          )}
          <button
            className={`model-status ${modelReady ? "ready" : ""}`}
            title={model.message || "Load SAM 3"}
            onClick={() => void ensureLoad()}
            disabled={hosted || modelLoading || model.state === "loading"}
          >
            {modelLoading || model.state === "loading" ? (
              <LoaderCircle size={12} className="spin" />
            ) : (
              <i />
            )}
            <span>
              SAM 3
              {modelReady && model.device
                ? ` · ${model.device.toUpperCase()}`
                : model.state === "auth_required"
                  ? " · access required"
                  : model.state === "loading"
                    ? " · loading"
                    : model.state === "error"
                      ? " · error"
                      : ""}
            </span>
          </button>
          <IconButton
            icon={HelpCircle}
            title="Keyboard shortcuts · ?"
            onClick={() => setHelp(true)}
          />
          {project && (
            <>
              <button
                className="button secondary compact"
                title="Save · ⌘ S"
                onClick={() => void saveAll()}
              >
                <Save size={15} />
                <span>Save</span>
              </button>
              <button
                className="button primary compact"
                onClick={() => setExportDialog(true)}
                disabled={geometryBusy || confirming}
              >
                <Download size={15} />
                <span>Export…</span>
              </button>
            </>
          )}
        </div>
      </header>
      {!project ? (
        <main className="welcome">
          <div className="welcome-copy">
            <h1>Image annotation</h1>
            <button
              className="button primary welcome-cta"
              disabled={booting}
              onClick={() => setProjectDialog(true)}
            >
              {booting ? (
                <LoaderCircle className="spin" size={18} />
              ) : (
                <FolderOpen size={18} />
              )}{" "}
              {hosted ? "Your projects" : "Open project"}{" "}
              <ArrowUpRight size={17} />
            </button>
          </div>
        </main>
      ) : (
        <>
          <ResizableWorkspace rightVisible={task === "detection" || inspector}>
            <aside className="image-sidebar">
              <div className="sidebar-heading">
                <span>Images</span>
                <span className="count-pill">{project.images.length}</span>
                <IconButton
                  icon={hosted ? ImagePlus : Link2}
                  title={
                    hosted
                      ? "Upload images or import COCO"
                      : "Relink image directory"
                  }
                  onClick={() =>
                    hosted ? void showUploads() : setRelink(true)
                  }
                />
              </div>
              <div className="image-search">
                <input
                  aria-label="Filter images"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search images…"
                />
              </div>
              <div className="image-list">
                {visibleImages.map((img, index) => {
                  const local = workspace.get(img.id);
                  const count =
                    task === "detection"
                      ? (workspace.getDetection(img.id)?.annotations.length ??
                        img.annotation_counts?.detection ??
                        0)
                      : (local?.annotations.length ??
                        img.annotation_counts?.segmentation ??
                        img.annotation_count);
                  return (
                    <button
                      key={img.id}
                      className={`image-item ${imageId === img.id ? "active" : ""}`}
                      onClick={() => void navigateTo(img.id)}
                      title={img.file_name}
                    >
                      <div className="thumbnail">
                        <img
                          src={assetURL(
                            `/images/${img.id}/file?thumbnail=true`,
                          )}
                          alt=""
                          loading="lazy"
                        />
                        {count > 0 && (
                          <span className="thumb-check">
                            <Check size={10} />
                          </span>
                        )}
                      </div>
                      <div className="image-item-info">
                        <span>{img.file_name.split("/").pop()}</span>
                        <small>
                          {count
                            ? `${count} instance${count === 1 ? "" : "s"}`
                            : "Unannotated"}
                          {(
                            task === "segmentation"
                              ? local?.draft
                              : workspace.getDetection(img.id)?.draft
                          )
                            ? " · draft"
                            : ""}
                        </small>
                      </div>
                      <span className="image-index">
                        {String(index + 1).padStart(2, "0")}
                      </span>
                    </button>
                  );
                })}
                {!visibleImages.length && (
                  <div className="empty-panel">
                    {project.images.length
                      ? "No matches"
                      : hosted
                        ? "Upload images to begin."
                        : "No images in this directory."}
                  </div>
                )}
              </div>
              <div className="image-list-footer">
                <span>
                  {Math.max(0, imageIndex + 1)} of {project.images.length}
                </span>
                <div>
                  <IconButton
                    icon={ArrowUp}
                    title="Previous image · ↑"
                    disabled={imageIndex <= 0}
                    onClick={() => navigate(imageIndex - 1)}
                  />
                  <IconButton
                    icon={ArrowDown}
                    title="Next image · ↓"
                    disabled={imageIndex >= project.images.length - 1}
                    onClick={() => navigate(imageIndex + 1)}
                  />
                </div>
              </div>
            </aside>
            <PanelResizeHandle side="left" />
            {task === "segmentation" && (
              <main className="editor-main">
                <div className="editor-toolbar">
                  <div className="toolbar-controls">
                    <div className="tool-group">
                      <IconButton
                        icon={MousePointer2}
                        title="Select and edit · V"
                        active={tool === "select"}
                        onClick={() => setTool("select")}
                      />
                      <IconButton
                        icon={Hand}
                        title="Pan image · drag to move"
                        active={tool === "pan"}
                        onClick={() => setTool("pan")}
                      />
                      <IconButton
                        icon={Plus}
                        title="Positive point · P"
                        active={tool === "positive"}
                        onClick={() => {
                          setTool("positive");
                          setSelected(null);
                        }}
                      />
                      <IconButton
                        icon={Minus}
                        title="Negative point · E"
                        active={tool === "negative"}
                        onClick={() => {
                          setTool("negative");
                          setSelected(null);
                        }}
                      />
                      <IconButton
                        icon={BoxSelect}
                        title="Bounding box · B"
                        active={tool === "box"}
                        onClick={() => {
                          setTool("box");
                          setSelected(null);
                        }}
                      />
                      <IconButton
                        icon={Pentagon}
                        title="Polygon · G"
                        active={tool === "polygon"}
                        onClick={() => {
                          setTool("polygon");
                          setSelected(null);
                          setVertex(null);
                        }}
                      />
                    </div>
                    <span className="toolbar-divider" />
                    <div className="category-control">
                      <i
                        style={{
                          background: selectedCategory?.color || "#9b9a90",
                        }}
                      />
                      <select
                        aria-label="Active class"
                        value={categoryId}
                        onChange={(e) => setCategoryId(Number(e.target.value))}
                      >
                        {!project.categories.length && (
                          <option value={0}>Create a class</option>
                        )}
                        {project.categories.map((c) => (
                          <option key={c.id} value={c.id}>
                            {c.name}
                          </option>
                        ))}
                      </select>
                      <button
                        className="icon-button small"
                        title="Create class"
                        onClick={() => editCategory("new")}
                      >
                        <Plus size={15} />
                      </button>
                    </div>
                    <div className="toolbar-spacer" />
                    <IconButton
                      icon={Undo2}
                      title="Undo · ⌘ Z"
                      disabled={!entry?.past.length || geometryBusy}
                      onClick={() => undo()}
                    />
                    <IconButton
                      icon={Redo2}
                      title="Redo · ⇧ ⌘ Z"
                      disabled={!entry?.future.length || geometryBusy}
                      onClick={() => undo(true)}
                    />
                    <span className="toolbar-divider" />
                    <IconButton
                      icon={showMasks ? Eye : EyeOff}
                      title="Show/hide masks · H"
                      active={!showMasks}
                      onClick={() => setShowMasks((v) => !v)}
                    />
                    <IconButton
                      icon={inspector ? PanelRightClose : PanelRightOpen}
                      title={inspector ? "Hide panel" : "Show panel"}
                      onClick={() => setInspector((v) => !v)}
                    />
                  </div>
                  {currentImage && state && (
                    <form
                      className="text-prompt-bar"
                      aria-label="Concept segmentation"
                      onSubmit={(e) => {
                        e.preventDefault();
                        void inferText();
                      }}
                    >
                      <Sparkles size={17} />
                      <select
                        className="prompt-language"
                        aria-label="Prompt language"
                        title={
                          hosted
                            ? "Spanish prompts are translated to English on the GPU server"
                            : "Spanish prompts are translated to English locally"
                        }
                        value={promptLanguage}
                        onChange={(e) =>
                          setPromptLanguage(e.target.value as "en" | "es")
                        }
                      >
                        <option value="en">English</option>
                        <option value="es">Spanish</option>
                      </select>
                      <input
                        aria-label="Object to segment"
                        placeholder={
                          visualExample
                            ? "Optional description"
                            : `Object to segment (in ${promptLanguage === "es" ? "Spanish" : "English"})`
                        }
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                      />
                      <VisualReference
                        value={visualExample}
                        scope={`${project.id || project.directory}:${sessionToken.current}:${imageId}`}
                        onChange={changeVisualExample}
                        onPickStart={cancelProposalSearches}
                        onOpenChange={setVisualDialogOpen}
                      />
                      <button
                        type="submit"
                        disabled={
                          (!text.trim() && !visualExample) ||
                          textBusy ||
                          mergingProposals
                        }
                        title="Generate proposals with SAM 3"
                        aria-label="Generate proposals with SAM 3"
                      >
                        {textBusy ? (
                          <LoaderCircle className="spin" size={16} />
                        ) : (
                          <ArrowUpRight size={18} />
                        )}
                      </button>
                    </form>
                  )}
                </div>
                <div className="canvas-area">
                  {currentImage && state && !loadingImage ? (
                    <CanvasEditor
                      key={currentImage.id}
                      image={currentImage}
                      annotations={state.annotations}
                      proposals={state.proposals}
                      draft={state.draft}
                      categories={project.categories}
                      selected={selected}
                      tool={tool}
                      showMasks={showMasks}
                      showBoxes={showBoxes}
                      opacity={opacity}
                      vertex={vertex}
                      busy={geometryBusy || confirming}
                      onVertex={setVertex}
                      onSelect={setSelected}
                      onProposal={(id) =>
                        update(
                          (s) => ({
                            ...s,
                            proposals: s.proposals.map((p) =>
                              p.id === id ? { ...p, selected: !p.selected } : p,
                            ),
                          }),
                          false,
                        )
                      }
                      onPoint={(point, negative) => addPrompt(point, negative)}
                      onBox={(box) => addPrompt(undefined, false, box)}
                      onPolygon={changePolygon}
                      onGeometry={(id, c) => void editGeometry(id, c)}
                      fitRef={fitRef}
                    />
                  ) : (
                    <div className="canvas-placeholder">
                      {loadingImage ? (
                        <>
                          <LoaderCircle size={26} className="spin" />
                          <span>Loading image…</span>
                        </>
                      ) : (
                        <>
                          <ImagePlus size={38} strokeWidth={1} />
                          <span>No images</span>
                          {hosted && (
                            <button
                              className="button primary"
                              onClick={() => void showUploads()}
                            >
                              <ImagePlus size={16} /> Upload images
                            </button>
                          )}
                        </>
                      )}
                    </div>
                  )}

                  {currentImage && state?.draft && (
                    <div className="draft-floating">
                      <span>
                        <i />
                        {pointBusy ? "Segmenting…" : "Draft"}{" "}
                        <small>
                          {state.draft.parts.length} part
                          {state.draft.parts.length === 1 ? "" : "s"}
                        </small>
                      </span>
                      {activePolygon && !activePolygon.closed ? (
                        <button
                          className="button secondary compact"
                          disabled={
                            activePolygon.vertices.length < 3 || confirming
                          }
                          onClick={closePolygon}
                        >
                          Close polygon <kbd>↵</kbd>
                        </button>
                      ) : activePolygon && !activePart?.mask ? (
                        <>
                          {
                            <button
                              className="button primary compact"
                              disabled={pointBusy || geometryBusy || confirming}
                              onClick={() => void usePolygon()}
                              title="Use the drawn polygon without SAM 3"
                            >
                              {geometryBusy ? (
                                <LoaderCircle className="spin" size={14} />
                              ) : (
                                <Pentagon size={14} />
                              )}
                              Use polygon
                            </button>
                          }
                          <button
                            className="button secondary compact"
                            disabled={
                              pointBusy ||
                              confirming ||
                              geometryBusy ||
                              (hosted && !modelReady)
                            }
                            title={
                              hosted && !modelReady
                                ? "SAM 3 is unavailable. Use polygon for manual annotation."
                                : undefined
                            }
                            onClick={refinePolygon}
                          >
                            {pointBusy ? (
                              <LoaderCircle className="spin" size={14} />
                            ) : (
                              <ScanLine size={14} />
                            )}
                            Refine with SAM
                          </button>
                        </>
                      ) : null}
                      <button
                        className="button primary compact"
                        disabled={!readyToConfirm}
                        onClick={() => void confirmDraft()}
                      >
                        {confirming ? (
                          <LoaderCircle className="spin" size={14} />
                        ) : (
                          <Check size={14} />
                        )}{" "}
                        Confirm <kbd>↵</kbd>
                      </button>
                    </div>
                  )}
                </div>
              </main>
            )}
            {inspector && task === "segmentation" && (
              <PanelResizeHandle side="right" />
            )}
            {inspector && task === "segmentation" && (
              <aside className="inspector">
                <div className="inspector-heading">
                  <span>Objects</span>
                  <span className="count-pill">
                    {state?.annotations.length || 0}
                  </span>
                  <SlidersHorizontal size={15} />
                </div>
                {state?.draft && (
                  <section className="draft-panel">
                    <div className="panel-label">
                      <span>
                        <Circle size={11} fill="currentColor" /> Object in
                        progress
                      </span>
                      <IconButton
                        icon={Trash2}
                        title="Discard draft"
                        onClick={discardDraft}
                      />
                    </div>
                    <div className="draft-category">
                      <i
                        style={{
                          background: project.categories.find(
                            (c) => c.id === state.draft?.category_id,
                          )?.color,
                        }}
                      />
                      <select
                        aria-label="Draft class"
                        value={state.draft.category_id}
                        onChange={(e) =>
                          update((s) => ({
                            ...s,
                            draft: s.draft
                              ? {
                                  ...s.draft,
                                  category_id: Number(e.target.value),
                                }
                              : null,
                          }))
                        }
                      >
                        {project.categories.map((c) => (
                          <option key={c.id} value={c.id}>
                            {c.name}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="parts-list">
                      {state.draft.parts.map((part, i) => (
                        <div
                          className={`part-item ${state.draft?.active_part_id === part.id ? "active" : ""}`}
                          key={part.id}
                        >
                          <button
                            onClick={() => {
                              update(
                                (s) => ({
                                  ...s,
                                  draft: s.draft
                                    ? { ...s.draft, active_part_id: part.id }
                                    : null,
                                }),
                                false,
                              );
                              setTool(
                                part.polygon && !part.seed_mask
                                  ? "polygon"
                                  : "positive",
                              );
                              setSelected(null);
                            }}
                          >
                            <span>Part {i + 1}</span>
                            <small>
                              {part.mask
                                ? "Ready"
                                : partErrors.current.has(
                                      `part:${imageId}:${part.id}`,
                                    )
                                  ? "Failed"
                                  : part.polygon && !part.seed_mask
                                    ? part.polygon.closed
                                      ? "Closed polygon"
                                      : `${part.polygon.vertices.length} vertices`
                                    : part.points.length ||
                                        part.box ||
                                        part.seed_mask
                                      ? "Pending"
                                      : "No prompts"}
                            </small>
                          </button>
                          {!part.mask &&
                            (part.points.length > 0 ||
                              part.box ||
                              part.seed_mask) &&
                            !pending.current.has(
                              `part:${imageId}:${part.id}`,
                            ) && (
                              <IconButton
                                icon={RotateCcw}
                                title="Retry segmentation"
                                onClick={() =>
                                  imageId !== null &&
                                  state.draft &&
                                  void inferPart(imageId, state.draft.id, part)
                                }
                              />
                            )}
                          <IconButton
                            icon={X}
                            title="Remove part"
                            onClick={() => removePart(part.id)}
                          />
                          {!part.mask &&
                            partErrors.current.has(
                              `part:${imageId}:${part.id}`,
                            ) && (
                              <p className="part-error" role="alert">
                                {partErrors.current.get(
                                  `part:${imageId}:${part.id}`,
                                )}{" "}
                                {part.polygon && !part.seed_mask
                                  ? "Check the polygon, then use Refine with SAM to try again."
                                  : "Use Retry segmentation to try again."}
                              </p>
                            )}
                        </div>
                      ))}
                    </div>
                    {activePolygon && (
                      <div className="polygon-actions">
                        {activePolygon.closed && (
                          <button
                            className="text-button"
                            disabled={confirming}
                            onClick={() => {
                              changePolygon(activePolygon.vertices, false);
                              setTool("polygon");
                            }}
                          >
                            <Pentagon size={14} /> Edit polygon
                          </button>
                        )}
                        {!activePolygon.closed &&
                          activePolygon.vertices.length > 0 && (
                            <button
                              className="text-button"
                              disabled={confirming}
                              onClick={() =>
                                changePolygon(
                                  activePolygon.vertices.slice(0, -1),
                                  false,
                                )
                              }
                            >
                              <Undo2 size={14} /> Remove last vertex
                            </button>
                          )}
                      </div>
                    )}
                    <button className="text-button add-part" onClick={addPart}>
                      <Plus size={14} /> Add part to object
                    </button>
                  </section>
                )}
                {(state?.proposals.length || 0) > 0 && (
                  <section className="proposals-panel">
                    <div className="panel-label">
                      <span>
                        <Sparkles size={13} /> Proposals{" "}
                        <b>{state?.proposals.length}</b>
                      </span>
                      <IconButton
                        icon={X}
                        title="Discard proposals"
                        disabled={geometryBusy || confirming}
                        onClick={() => update((s) => ({ ...s, proposals: [] }))}
                      />
                    </div>
                    {currentTextPrompt && (
                      <p
                        className="proposal-prompt"
                        title={
                          currentTextPrompt.source_language === "es"
                            ? `Original: ${currentTextPrompt.original}`
                            : currentTextPrompt.english
                        }
                      >
                        Prompt: <strong>{currentTextPrompt.english}</strong>
                      </p>
                    )}
                    <div className="proposal-actions">
                      <button
                        className="text-button"
                        disabled={geometryBusy || confirming}
                        onClick={() =>
                          update(
                            (s) => ({
                              ...s,
                              proposals: s.proposals.map((p) => ({
                                ...p,
                                selected: !s.proposals.every((a) => a.selected),
                              })),
                            }),
                            false,
                          )
                        }
                      >
                        {state?.proposals.every((p) => p.selected)
                          ? "Deselect all"
                          : "Select all"}
                      </button>
                      <span>{selectedProposals.length} selected</span>
                    </div>
                    {state?.proposals.map((proposal, i) => (
                      <div
                        className={`proposal-item ${proposal.selected ? "active" : ""}`}
                        key={proposal.id}
                      >
                        <button
                          className="proposal-checkbox"
                          aria-label={`Select proposal ${i + 1}`}
                          aria-pressed={proposal.selected}
                          disabled={geometryBusy || confirming}
                          onClick={() =>
                            update(
                              (s) => ({
                                ...s,
                                proposals: s.proposals.map((p) =>
                                  p.id === proposal.id
                                    ? { ...p, selected: !p.selected }
                                    : p,
                                ),
                              }),
                              false,
                            )
                          }
                        >
                          {proposal.selected && <Check size={12} />}
                        </button>
                        <span>
                          Proposal {i + 1}
                          <small>{Math.round(proposal.score * 100)}%</small>
                        </span>
                        <button
                          className="text-button"
                          onClick={() => refineProposal(proposal)}
                          title="Refine with clicks"
                          disabled={geometryBusy || confirming}
                        >
                          Refine
                        </button>
                      </div>
                    ))}
                    <button
                      className="button primary full compact"
                      disabled={
                        !selectedProposals.length || geometryBusy || confirming
                      }
                      onClick={acceptProposals}
                    >
                      <CheckCheck size={15} /> Accept selected
                    </button>
                    <button
                      className="button secondary full compact"
                      disabled={
                        selectedProposals.length < 2 ||
                        geometryBusy ||
                        confirming ||
                        textBusy
                      }
                      onClick={() => void mergeProposals()}
                      title="Combine selected proposals into one instance using the active class"
                    >
                      {mergingProposals ? (
                        <LoaderCircle size={15} className="spin" />
                      ) : (
                        <Merge size={15} />
                      )}
                      Merge selected
                    </button>
                  </section>
                )}
                <section className="annotations-panel">
                  {!state?.annotations.length &&
                  !state?.draft &&
                  !state?.proposals.length ? (
                    <div className="objects-empty">
                      <Layers3 size={27} strokeWidth={1} />
                      <h3>No annotations</h3>
                      <button
                        className="button secondary compact"
                        onClick={resumeDraft}
                      >
                        <Plus size={14} /> New object
                      </button>
                    </div>
                  ) : (
                    <>
                      <div className="section-caption">CONFIRMED INSTANCES</div>
                      {state?.annotations.map((a, i) => {
                        const cat = project.categories.find(
                          (c) => c.id === a.category_id,
                        );
                        return (
                          <button
                            className={`annotation-row ${selected === a.id ? "active" : ""}`}
                            key={a.id}
                            onClick={() => {
                              setSelected(a.id);
                              setTool("select");
                              setVertex(null);
                            }}
                          >
                            <i
                              style={{ background: cat?.color || "#a29bcc" }}
                            />
                            <span>
                              {cat?.name || "No class"}
                              <small>
                                #{String(i + 1).padStart(2, "0")}
                                {a.components.length > 1
                                  ? ` · ${a.components.length} parts`
                                  : ""}
                              </small>
                            </span>
                            {selected === a.id ? (
                              <ChevronRight size={14} />
                            ) : (
                              <Check size={13} />
                            )}
                          </button>
                        );
                      })}
                      <button
                        className="text-button new-object"
                        onClick={resumeDraft}
                      >
                        <Plus size={14} />{" "}
                        {state?.draft ? "Resume draft" : "New object"}
                      </button>
                    </>
                  )}
                </section>
                {selectedAnnotation && (
                  <section className="selection-panel">
                    <div className="panel-label">
                      <span>Edit instance</span>
                      <IconButton
                        icon={Trash2}
                        title={
                          vertex ? "Delete selected vertex" : "Delete instance"
                        }
                        onClick={removeSelected}
                      />
                    </div>
                    <select
                      aria-label="Instance class"
                      value={selectedAnnotation.category_id}
                      onChange={(e) =>
                        update((s) => ({
                          ...s,
                          annotations: s.annotations.map((a) =>
                            a.id === selected
                              ? { ...a, category_id: Number(e.target.value) }
                              : a,
                          ),
                        }))
                      }
                    >
                      {project.categories.map((c) => (
                        <option value={c.id} key={c.id}>
                          {c.name}
                        </option>
                      ))}
                    </select>
                    <p className="field-note">
                      Drag vertices to adjust the contour. Double-click an edge
                      to add a vertex.
                    </p>
                    <div className="hole-cleanup">
                      <label>
                        <span>Holes up to</span>
                        <input
                          type="number"
                          aria-label="Maximum hole area"
                          title="Maximum enclosed hole area in original-image pixels"
                          min={1}
                          max={
                            currentImage
                              ? currentImage.width * currentImage.height
                              : undefined
                          }
                          step={1}
                          value={maxHoleArea}
                          disabled={geometryBusy || confirming}
                          onChange={(e) => setMaxHoleArea(e.target.value)}
                        />
                        <span>px²</span>
                      </label>
                      <button
                        className="button secondary"
                        type="button"
                        title="Fill small enclosed gaps without changing the outer mask boundary. Undo restores the original mask."
                        disabled={geometryBusy || confirming || !validHoleArea}
                        onClick={() => void fillSmallHoles()}
                      >
                        Fill small holes
                      </button>
                    </div>
                    {geometryBusy && (
                      <span className="inline-status">
                        <LoaderCircle className="spin" size={13} /> Updating
                        mask…
                      </span>
                    )}
                  </section>
                )}
                <div className="inspector-bottom">
                  <ClassPanel
                    categories={project.categories}
                    categoryId={categoryId}
                    onSelect={setCategoryId}
                    onEdit={(id) =>
                      editCategory(project.categories.find((c) => c.id === id)!)
                    }
                    onCreate={() => editCategory("new")}
                  />

                  <label className="checkbox-label mask-box-toggle">
                    <input
                      type="checkbox"
                      checked={showBoxes}
                      onChange={(e) => setShowBoxes(e.target.checked)}
                    />
                    Show bounding boxes
                  </label>
                  <div className="opacity-control">
                    <Eye size={14} />
                    <span>Opacity</span>
                    <input
                      type="range"
                      min="0.1"
                      max="1"
                      step="0.05"
                      value={opacity}
                      onChange={(e) => setOpacity(Number(e.target.value))}
                      aria-label="Mask opacity"
                    />
                  </div>
                </div>
              </aside>
            )}
            <DetectionEditor
              key={project.id || project.directory}
              active={task === "detection"}
              image={currentImage}
              project={project}
              workspace={workspace}
              categoryId={categoryId}
              setCategoryId={setCategoryId}
              createClass={() => editCategory("new")}
              editClass={(id) => {
                const c = project.categories.find((c) => c.id === id);
                if (c) editCategory(c);
              }}
              modelReady={modelReady}
              blocked={
                !!(
                  projectDialog ||
                  help ||
                  categoryDialog ||
                  relink ||
                  conflictDialog ||
                  uploadDialog ||
                  visualDialogOpen ||
                  exportDialog
                )
              }
              onError={setToast}
              navigate={(offset) => navigate(imageIndex + offset)}
            />
          </ResizableWorkspace>
          <footer className="statusbar">
            <div className={`save-status ${failedRecord ? "error" : ""}`}>
              {failedRecord ? (
                <>
                  <AlertTriangle size={12} />
                  <button
                    onClick={() => {
                      if (failedRecord[1].conflict) {
                        setConflictImage(failedRecord[0]);
                        setConflictDialog(true);
                      } else void workspace.save(failedRecord[0]);
                    }}
                  >
                    {failedRecord[1].conflict
                      ? "Save conflict · review"
                      : "Not saved · retry"}
                  </button>
                </>
              ) : anySaving ? (
                <>
                  <LoaderCircle size={12} className="spin" /> Saving…
                </>
              ) : anyDirty ? (
                <>
                  <Circle size={8} fill="currentColor" /> Unsaved changes
                </>
              ) : (
                <>
                  <Check size={13} /> {hosted ? "Saved" : "Saved locally"}
                </>
              )}
            </div>
            <div className="status-middle">
              {hosted && jobs.size ? (
                <span className="hosted-job-status">
                  <LoaderCircle size={12} className="spin" />
                  {[...jobs.values()].some(
                    (job) => job.status === "reconnecting",
                  )
                    ? "Reconnecting to inference…"
                    : [...jobs.values()].some((job) =>
                          ["running", "dispatched"].includes(job.status),
                        )
                      ? "SAM 3 is processing…"
                      : "Waiting for a GPU…"}
                  <button onClick={abortAll}>Cancel requests</button>
                </span>
              ) : (
                currentImage?.file_name || project.image_root
              )}
            </div>
            <div>
              {exportPath ? (
                <a href={assetURL(exportPath)} download title={exportPath}>
                  <Download size={12} /> {exportName}
                </a>
              ) : (
                <>
                  <Command size={11} />
                  <span>Esc to suspend · ↵ to confirm</span>
                </>
              )}
            </div>
          </footer>
        </>
      )}
      {exportDialog && (
        <ExportDialog
          task={task}
          flush={workspace.flush}
          onClose={() => setExportDialog(false)}
          onReady={(path, name) => {
            setExportPath(path);
            setExportName(name);
            downloadExport(path, name);
            setToast("Export ready to download.");
          }}
        />
      )}
      {toast && (
        <div className="toast" role="status">
          <span>{toast}</span>
          <button
            onClick={() => setToast(null)}
            aria-label="Dismiss notification"
          >
            <X size={15} />
          </button>
        </div>
      )}
      {projectDialog && !hosted && (
        <ProjectDialog
          onClose={() => setProjectDialog(false)}
          onOpen={openProject}
        />
      )}{" "}
      {projectDialog && hosted && (
        <HostedProjects
          onClose={() => setProjectDialog(false)}
          onOpen={openProject}
          onDelete={(id) => {
            if (project?.id === id) {
              abortAll();
              workspace.reset();
              sessionToken.current++;
              selectProject({});
              setProject(null);
              setImageId(null);
            }
          }}
        />
      )}
      {uploadDialog && project && hosted && (
        <HostedUploads
          project={project}
          onClose={() => setUploadDialog(false)}
          onUpdate={(result) => {
            openProject(result);
          }}
        />
      )}
      {relink && project && !hosted && (
        <FileBrowser
          kind="directory"
          initial={project.image_root}
          onCancel={() => setRelink(false)}
          onChoose={async (paths) => {
            try {
              await workspace.flush();
              const result = await api<Project>("/project/relink", "POST", {
                image_root: paths[0],
              });
              setProject(result);
              setRelink(false);
              setToast("Image directory relinked.");
            } catch (e) {
              setToast(errorText(e));
            }
          }}
        />
      )}{" "}
      {categoryDialog && (
        <div className="modal-backdrop">
          <section
            className="modal small-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Object class"
          >
            <div className="modal-heading">
              <h2>{categoryDialog === "new" ? "New class" : "Edit class"}</h2>
              <IconButton
                icon={X}
                title="Close"
                onClick={() => setCategoryDialog(null)}
              />
            </div>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void saveCategory();
              }}
            >
              <label className="field-label" htmlFor="category-name">
                Name
              </label>
              <input
                autoFocus
                id="category-name"
                value={categoryName}
                onChange={(e) => setCategoryName(e.target.value)}
                placeholder="For example, carrot"
                required
              />
              <label className="field-label">Color</label>
              <div className="color-options">
                {PALETTE.map((color) => (
                  <button
                    type="button"
                    key={color}
                    style={{ background: color }}
                    className={categoryColor === color ? "active" : ""}
                    aria-label={`Color ${color}`}
                    onClick={() => setCategoryColor(color)}
                  >
                    {categoryColor === color && <Check size={16} />}
                  </button>
                ))}
                <input
                  aria-label="Custom color"
                  type="color"
                  value={categoryColor}
                  onChange={(e) => setCategoryColor(e.target.value)}
                />
              </div>
              <footer className="modal-footer">
                <span />
                <button
                  className="button primary"
                  disabled={categoryBusy || !categoryName.trim()}
                >
                  {categoryBusy ? (
                    <LoaderCircle className="spin" size={15} />
                  ) : (
                    <Check size={15} />
                  )}{" "}
                  Save class
                </button>
              </footer>
            </form>
          </section>
        </div>
      )}
      {help && (
        <div className="modal-backdrop">
          <section
            className="modal small-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Keyboard shortcuts"
          >
            <div className="modal-heading">
              <h2>Keyboard shortcuts</h2>
              <IconButton
                icon={X}
                title="Close"
                onClick={() => setHelp(false)}
              />
            </div>
            <div className="shortcut-list">
              {[
                ["N", "New object / resume draft"],
                ["P / E", "Positive point / negative"],
                ["B", "Draw bounding box"],
                ["S", "Adjust active detection box with SAM"],
                ["G", "Draw polygon"],
                ["V", "Select and edit"],
                ["↵", "Close polygon / confirm object"],
                ["Esc", "Suspend draft / deselect"],
                ["↑ / ↓", "Change image when nothing is selected"],
                ["⌘ / Ctrl + Z", "Undo"],
                ["⇧ ⌘ / Ctrl + Y", "Redo"],
                ["⌘ / Ctrl + S", "Save"],
                ["⌫", "Delete vertex / instance"],
                ["F / H", "Fit image / show masks"],
                ["Space + drag", "Pan image"],
              ].map(([key, label]) => (
                <div key={key}>
                  <span>{label}</span>
                  <kbd>{key}</kbd>
                </div>
              ))}
            </div>
          </section>
        </div>
      )}
      {conflictDialog && (
        <div className="modal-backdrop">
          <section
            className="modal small-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Save conflict"
          >
            <div className="modal-heading">
              <h2>Another version has been saved.</h2>
              <IconButton
                icon={X}
                title="Close"
                onClick={() => setConflictDialog(false)}
              />
            </div>
            <p className="modal-description">
              Your changes are still in this window. Download a copy before
              reloading the saved version. Reloading replaces local changes to
              this image.
            </p>
            <div className="conflict-actions">
              <button className="button secondary" onClick={downloadLocal}>
                <Download size={15} /> Download local copy
              </button>
              <button
                className="button primary"
                onClick={async () => {
                  if (conflictImage === null) return;
                  try {
                    await workspace.load(conflictImage, true);
                    setConflictDialog(false);
                    setSelected(null);
                  } catch (e) {
                    setToast(errorText(e));
                  }
                }}
              >
                <RotateCcw size={15} />{" "}
                {hosted ? "Reload saved version" : "Reload from disk"}
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
