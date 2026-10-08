import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, errorText } from "./api";
import type {
  ImageState,
  ProjectImageState,
  DetectionState,
  BoxAnnotation,
} from "./types";
type Entry = {
  state: ImageState;
  detection: DetectionState;
  detectionPast: DetectionState[];
  detectionFuture: DetectionState[];
  detectionGeneration: number;
  dirty: boolean;
  saving: boolean;
  error: string | null;
  conflict: boolean;
  past: ImageState[];
  future: ImageState[];
  generation: number;
  timer?: ReturnType<typeof setTimeout>;
  inflight?: Promise<void>;
};
const clone = <T>(x: T): T => structuredClone(x);
function fullState(e: Entry): ProjectImageState {
  const { annotations: boxes, ...detection } = e.detection;
  return clone({
    ...e.state,
    schema_version: 2,
    annotations: [...e.state.annotations, ...boxes],
    detection,
  });
}
export function useWorkspace(onError: (message: string) => void) {
  const records = useRef(new Map<number, Entry>());
  const epoch = useRef(0);
  const [, render] = useState(0);
  const notify = useCallback(() => render((v) => v + 1), []);
  const errorFn = useRef(onError);
  errorFn.current = onError;
  const savingFn = useRef<(id: number) => Promise<void>>(async () => {});
  const save = useCallback(
    async (id: number) => {
      const e = records.current.get(id);
      if (!e || !e.dirty || e.conflict) return;
      if (e.saving) {
        await e.inflight;
        if (e.dirty && !e.error) await savingFn.current(id);
        return;
      }
      clearTimeout(e.timer);
      e.saving = true;
      e.error = null;
      const generation = e.generation;
      const payload = fullState(e);
      notify();
      e.inflight = (async () => {
        try {
          const saved = await api<ImageState>(
            `/images/${id}/state`,
            "PUT",
            payload,
          );
          e.state = { ...e.state, revision: saved.revision };
          e.dirty = e.generation !== generation;
        } catch (err) {
          e.error = errorText(err);
          e.conflict = err instanceof ApiError && err.status === 409;
          errorFn.current(
            e.conflict
              ? "The project changed in another window. Your changes are preserved; resolve the conflict."
              : `Could not save: ${e.error}`,
          );
        } finally {
          e.saving = false;
          notify();
        }
      })();
      await e.inflight;
      if (e.dirty && !e.error) await savingFn.current(id);
    },
    [notify],
  );
  savingFn.current = save;
  const load = useCallback(
    async (id: number, force = false) => {
      if (!force && records.current.has(id))
        return records.current.get(id)!.state;
      const requestEpoch = epoch.current;
      const full = await api<ProjectImageState>(`/images/${id}/state`);
      const state: ImageState = {
        image_id: full.image_id,
        revision: full.revision,
        annotations: full.annotations.filter((a) => a.kind !== "bbox"),
        draft: full.draft,
        proposals: full.proposals,
      };
      const detection: DetectionState = {
        draft: null,
        proposals: [],
        adjustment: null,
        ...full.detection,
        annotations: full.annotations.filter(
          (a): a is BoxAnnotation => a.kind === "bbox",
        ),
      };
      if (requestEpoch !== epoch.current) return state;
      // React Strict Mode and rapid navigation can overlap reads. A late read
      // must never replace a record that already contains user edits.
      if (!force && records.current.has(id))
        return records.current.get(id)!.state;
      const old = records.current.get(id);
      if (old) clearTimeout(old.timer);
      records.current.set(id, {
        state,
        detection,
        detectionPast: [],
        detectionFuture: [],
        detectionGeneration: 0,
        dirty: false,
        saving: false,
        error: null,
        conflict: false,
        past: [],
        future: [],
        generation: 0,
      });
      notify();
      return state;
    },
    [notify],
  );
  const update = useCallback(
    (id: number, fn: (state: ImageState) => ImageState, history = true) => {
      const e = records.current.get(id);
      if (!e) return;
      if (history) {
        e.past.push(clone(e.state));
        if (e.past.length > 60) e.past.shift();
        e.future = [];
      }
      e.state = fn(clone(e.state));
      e.generation++;
      e.dirty = true;
      clearTimeout(e.timer);
      if (!e.conflict)
        e.timer = setTimeout(() => void savingFn.current(id), 400);
      notify();
    },
    [notify],
  );
  const history = useCallback(
    (id: number, redo = false) => {
      const e = records.current.get(id);
      if (!e) return;
      const from = redo ? e.future : e.past;
      const to = redo ? e.past : e.future;
      const next = from.pop();
      if (!next) return;
      to.push(clone(e.state));
      const revision = e.state.revision;
      update(id, () => ({ ...next, revision }), false);
    },
    [update],
  );
  const updateDetection = useCallback(
    (id: number, fn: (s: DetectionState) => DetectionState, history = true) => {
      const e = records.current.get(id);
      if (!e) return;
      if (history) {
        e.detectionPast.push(clone(e.detection));
        if (e.detectionPast.length > 60) e.detectionPast.shift();
        e.detectionFuture = [];
      }
      e.detection = fn(clone(e.detection));
      e.detectionGeneration++;
      update(id, (s) => s, false);
    },
    [update],
  );
  const historyDetection = useCallback(
    (id: number, redo = false) => {
      const e = records.current.get(id);
      if (!e) return;
      const from = redo ? e.detectionFuture : e.detectionPast;
      const to = redo ? e.detectionPast : e.detectionFuture;
      const next = from.pop();
      if (!next) return;
      to.push(clone(e.detection));
      updateDetection(id, () => next, false);
    },
    [updateDetection],
  );
  const flush = useCallback(async () => {
    await Promise.all(
      [...records.current.keys()].map((id) => savingFn.current(id)),
    );
    if ([...records.current.values()].some((e) => e.dirty || e.conflict))
      throw new Error(
        "There are unsaved changes. Resolve the error before continuing.",
      );
  }, []);
  const reset = useCallback(() => {
    epoch.current++;
    for (const e of records.current.values()) clearTimeout(e.timer);
    records.current.clear();
    notify();
  }, [notify]);
  useEffect(() => {
    const handler = (event: BeforeUnloadEvent) => {
      if ([...records.current.values()].some((e) => e.dirty)) {
        event.preventDefault();
        event.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", handler);
    return () => {
      window.removeEventListener("beforeunload", handler);
      epoch.current++;
      for (const e of records.current.values()) clearTimeout(e.timer);
    };
  }, []);
  return {
    records: records.current,
    getDetection: (id: number) => records.current.get(id)?.detection,
    updateDetection,
    historyDetection,
    load,
    update,
    history,
    save,
    flush,
    reset,
    get: (id: number) => records.current.get(id)?.state,
    snapshot: (id: number) => {
      const e = records.current.get(id);
      return e ? fullState(e) : undefined;
    },
    entry: (id: number) => records.current.get(id),
  };
}

export type Workspace = ReturnType<typeof useWorkspace>;
