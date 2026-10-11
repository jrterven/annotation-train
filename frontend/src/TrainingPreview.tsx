import { useEffect, useMemo, useState } from "react";
import { assetURL, errorText, projectClient } from "./api";
import type { Project } from "./types";

type Proposal = {
  category_id: number;
  score: number;
  bbox?: number[];
  components?: { outer: number[][]; holes: number[][][] }[];
};
type Batch = {
  input: {
    images: { id: number; width: number; height: number; file_name: string }[];
  };
  result: { images: { image_id: number; proposals: Proposal[] }[] };
};
function ring(points: number[][]) {
  return points.map(([x, y], i) => `${i ? "L" : "M"}${x},${y}`).join(" ") + "Z";
}
export default function TrainingPreview({
  project,
  batchId,
}: {
  project: Project;
  batchId: string;
}) {
  const api = useMemo(
    () => projectClient(project),
    [project.id, project.directory],
  );
  const [batch, setBatch] = useState<Batch | null>(null);
  const [imageId, setImageId] = useState(0);
  const [confidence, setConfidence] = useState(0.5);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    void api<Batch>(
      `/training/predictions/${batchId}`,
      "GET",
      undefined,
      controller.signal,
    )
      .then((value) => {
        if (controller.signal.aborted) return;
        setBatch(value);
        setImageId(value.input.images[0]?.id || 0);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(errorText(e));
      });
    return () => controller.abort();
  }, [api, batchId]);
  if (error) return <p role="alert">{error}</p>;
  const image = batch?.input.images.find((im) => im.id === imageId);
  const proposals =
    batch?.result.images
      .find((im) => im.image_id === imageId)
      ?.proposals.filter((p) => p.score >= confidence) || [];
  if (!image) return null;
  return (
    <div className="training-preview">
      <div className="training-row">
        <select
          aria-label="Prediction image"
          value={imageId}
          onChange={(e) => setImageId(Number(e.target.value))}
        >
          {batch!.input.images.map((im) => (
            <option key={im.id} value={im.id}>
              {im.file_name}
            </option>
          ))}
        </select>
        <label>
          Confidence {confidence.toFixed(2)}
          <input
            aria-label="Preview confidence"
            type="range"
            min=".01"
            max="1"
            step=".01"
            value={confidence}
            onChange={(e) => setConfidence(Number(e.target.value))}
          />
        </label>
        <span>{proposals.length} predictions</span>
      </div>
      <svg
        viewBox={`0 0 ${image.width} ${image.height}`}
        role="img"
        aria-label="Model predictions on original image"
      >
        <image
          href={assetURL(`/images/${image.id}/file`)}
          width={image.width}
          height={image.height}
        />
        {proposals.map((p, i) => {
          const category = project.categories.find(
            (c) => c.id === p.category_id,
          );
          const color = category?.color || "#16a577";
          return (
            <g key={i}>
              <title>
                {category?.name} · {p.score.toFixed(2)}
              </title>
              {p.components ? (
                <path
                  d={p.components
                    .map((c) => [ring(c.outer), ...c.holes.map(ring)].join(" "))
                    .join(" ")}
                  fill={color}
                  fillOpacity=".3"
                  fillRule="evenodd"
                  stroke={color}
                  strokeWidth="2"
                  vectorEffect="non-scaling-stroke"
                />
              ) : (
                p.bbox && (
                  <rect
                    x={p.bbox[0]}
                    y={p.bbox[1]}
                    width={p.bbox[2]}
                    height={p.bbox[3]}
                    fill="none"
                    stroke={color}
                    strokeWidth="2"
                    vectorEffect="non-scaling-stroke"
                  />
                )
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
