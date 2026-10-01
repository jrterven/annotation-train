import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, errorText } from "./api";
import type { ImageState } from "./types";
type Entry = {
  state: ImageState;
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
      const payload = clone(e.state);
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
      const state = await api<ImageState>(`/images/${id}/state`);
      if (requestEpoch !== epoch.current) return state;
      // React Strict Mode and rapid navigation can overlap reads. A late read
      // must never replace a record that already contains user edits.
      if (!force && records.current.has(id))
        return records.current.get(id)!.state;
      const old = records.current.get(id);
      if (old) clearTimeout(old.timer);
      records.current.set(id, {
        state,
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
    return () => window.removeEventListener("beforeunload", handler);
  }, []);
  return {
    records: records.current,
    load,
    update,
    history,
    save,
    flush,
    reset,
    get: (id: number) => records.current.get(id)?.state,
    entry: (id: number) => records.current.get(id),
  };
}
