import { useEffect, useMemo, useState } from "react";
import { errorText, projectClient } from "./api";
import type { Project } from "./types";
type Event = {
  cursor: number;
  created_at: number;
  status: string;
  metric?: { epoch: number; map?: number };
};
export default function TrainingLog({
  project,
  runId,
  cursor,
}: {
  project: Project;
  runId: string;
  cursor: number;
}) {
  const api = useMemo(
    () => projectClient(project),
    [project.id, project.directory],
  );
  const [events, setEvents] = useState<Event[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    void api<{ events: Event[] }>(
      `/training/runs/${runId}/log`,
      "GET",
      undefined,
      controller.signal,
    )
      .then((data) => {
        if (!controller.signal.aborted) setEvents(data.events);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(errorText(e));
      });
    return () => controller.abort();
  }, [api, runId, cursor]);
  return (
    <>
      {error && <p role="alert">{error}</p>}
      <ol className="training-events">
        {events.map((e) => (
          <li key={e.cursor}>
            <time>{new Date(e.created_at * 1000).toLocaleTimeString()}</time> ·{" "}
            {e.status}
            {e.metric &&
              ` · epoch ${e.metric.epoch}${e.metric.map === undefined ? "" : ` · validation mAP ${e.metric.map.toFixed(3)}`}`}
          </li>
        ))}
      </ol>
      <p className="field-note">
        Detailed runtime logs stay on the connected computer.
      </p>
    </>
  );
}
