import { useEffect, useRef, useState } from "react";
import { api, uid, errorText } from "./api";
import type { Workspace } from "./persistence";
import type {
  Task,
  Part,
  Proposal,
  BoxProposal,
  GeometryResult,
  Tool,
} from "./types";
import type { VisualExample } from "./VisualReference";
import { maskBounds } from "./masks";

export type TextPrompt = {
  original: string;
  english: string;
  source_language: "en" | "es";
};
type Pending = { controller: AbortController; imageId: number; kind: string };

/** One lifecycle for both tasks; callers supply only geometry-specific freshness. */
export function useEditorInference(task: Task, workspace: Workspace) {
  const pending = useRef(new Map<string, Pending>());
  const [, render] = useState(0);
  const notify = () => render((v) => v + 1);
  function cancel(
    predicate: (request: Pending, key: string) => boolean = () => true,
  ) {
    for (const [key, request] of pending.current)
      if (predicate(request, key)) {
        request.controller.abort();
        pending.current.delete(key);
      }
    notify();
  }
  useEffect(
    () => () => {
      for (const request of pending.current.values())
        request.controller.abort();
      pending.current.clear();
    },
    [],
  );
  function begin(
    key: string,
    imageId: number,
    kind: string,
    fresh: () => boolean = () => true,
  ) {
    pending.current.get(key)?.controller.abort();
    const request = { controller: new AbortController(), imageId, kind };
    const entry = workspace.entry(imageId);
    const revision = workspace.get(imageId)?.revision || 0;
    pending.current.set(key, request);
    notify();
    const isCurrent = () =>
      pending.current.get(key) === request &&
      !request.controller.signal.aborted &&
      workspace.entry(imageId) === entry &&
      fresh();
    async function run<T extends { image_id: number }>(
      path: string,
      payload: unknown,
    ) {
      const result = await api<T>(
        path,
        "POST",
        payload,
        request.controller.signal,
      );
      return isCurrent() && result.image_id === imageId ? result : undefined;
    }
    return {
      controller: request.controller,
      isCurrent,
      points: (part: Part) =>
        run<GeometryResult & { image_id: number }>("/infer/points", {
          image_id: imageId,
          revision,
          part,
        }),
      concept: (
        text: string,
        language: "en" | "es",
        category: number,
        visual: VisualExample | null,
      ) =>
        run<{ image_id: number; proposals: Proposal[]; prompt?: TextPrompt }>(
          visual ? "/infer/visual" : "/infer/text",
          {
            image_id: imageId,
            revision,
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
        ),
      error: (error: unknown, report: (message: string) => void) => {
        if (
          isCurrent() &&
          !(error instanceof Error && error.name === "AbortError")
        )
          report(errorText(error));
      },
      finish: () => {
        if (pending.current.get(key) === request) {
          pending.current.delete(key);
          notify();
        }
      },
    };
  }
  return { task, pending, begin, cancel };
}

export function proposalCandidates(
  task: "segmentation",
  proposals: Proposal[],
  category: number,
): Proposal[];
export function proposalCandidates(
  task: "detection",
  proposals: Proposal[],
  category: number,
): BoxProposal[];
export function proposalCandidates(
  task: Task,
  proposals: Proposal[],
  category: number,
): (Proposal | BoxProposal)[] {
  return proposals.flatMap((p): (Proposal | BoxProposal)[] => {
    if (task === "segmentation") return [{ ...p, selected: false }];
    const bbox = maskBounds(p.mask);
    return bbox
      ? [
          {
            id: uid(),
            kind: "bbox",
            bbox,
            category_id: category,
            iscrowd: 0,
            score: p.score,
            selected: false,
          },
        ]
      : [];
  });
}
export function acceptCandidates<
  T extends { score: number; selected: boolean },
  A,
>(proposals: T[], map: (annotation: Omit<T, "score" | "selected">) => A) {
  return {
    accepted: proposals
      .filter((p) => p.selected)
      .map(({ score: _score, selected: _selected, ...p }) => map(p)),
    remaining: proposals.filter((p) => !p.selected),
  };
}

type Shortcuts = {
  active: boolean;
  blocked: boolean;
  selected: string | null;
  onSave?: () => void;
  onHistory: (redo: boolean) => void;
  onConfirm: () => void;
  onDelete: () => void;
  onEscape: () => void;
  onNavigate: (offset: number) => void;
  onTool: (tool: Tool) => void;
  onFit: () => void;
  onAdjust?: () => void;
  extra?: Record<string, () => void>;
};
export function useEditorShortcuts(p: Shortcuts) {
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      const element = e.target as HTMLElement;
      if (
        p.blocked ||
        /INPUT|TEXTAREA|SELECT/.test(element.tagName) ||
        element.isContentEditable
      )
        return;
      const mod = e.ctrlKey || e.metaKey,
        name = e.key.toLowerCase();
      if (mod && name === "s" && p.onSave) {
        e.preventDefault();
        p.onSave();
        return;
      }
      if (!p.active) return;
      if (mod && ["z", "y"].includes(name)) {
        e.preventDefault();
        p.onHistory(name === "y" || e.shiftKey);
        return;
      }
      if (mod || e.altKey) return;
      const tools: Record<string, Tool> = {
        v: "select",
        b: "box",
        p: "positive",
        e: "negative",
        "-": "negative",
      };
      if (name === "s" && p.onAdjust) {
        e.preventDefault();
        if (!e.repeat) p.onAdjust();
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (!e.repeat) p.onConfirm();
      } else if (e.key === "Escape") p.onEscape();
      else if (["Delete", "Backspace"].includes(e.key)) {
        e.preventDefault();
        p.onDelete();
      } else if (["ArrowUp", "ArrowDown"].includes(e.key) && !p.selected) {
        e.preventDefault();
        p.onNavigate(e.key === "ArrowUp" ? -1 : 1);
      } else if (name === "f") p.onFit();
      else if (tools[name]) p.onTool(tools[name]);
      else p.extra?.[name]?.();
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  });
}
