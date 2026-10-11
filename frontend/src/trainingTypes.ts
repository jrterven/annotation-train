import type { Task } from "./types";
export type Split = "train" | "validation" | "test";
export type TrainingModel = {
  id: string;
  name: string;
  family: string;
  task: Task;
  size: string;
  resolution: number;
  license: string;
  license_url: string;
};
export type ComputeDevice = {
  id: string;
  name: string;
  models: string[];
  online: boolean;
  enabled: boolean;
  node_id: string;
  slot: string;
};
export type ReviewImage = {
  id: number;
  file_name: string;
  revision: number;
  annotation_count: number;
  reviewed: boolean;
};
export type DatasetSnapshot = {
  id: string;
  version: number;
  task: Task;
  counts: Partial<Record<Split, number>>;
  warnings: string[];
  fingerprint: string;
};
export type ClassMetric = {
  category_id: number;
  map?: number;
  precision?: number;
  recall?: number;
};
export type EpochMetric = {
  per_class?: ClassMetric[];
  epoch: number;
  train_loss?: number;
  validation_loss?: number;
  map?: number;
  box_map?: number;
  precision?: number;
  recall?: number;
  elapsed?: number;
};
export type TrainingRun = {
  id: string;
  task: Task;
  dataset_id: string;
  model: TrainingModel;
  device_id: string;
  parameters: { epochs: number };
  status: string;
  epoch: number;
  created_at: number;
  events: number;
  metrics: EpochMetric[];
  error?: string;
  artifacts: { id: string; name: string; status: string }[];
  progress?: number;
  evaluation?: {
    per_class?: ClassMetric[];
    status: string;
    metrics?: Record<string, number>;
    error?: string;
  };
  prediction?: {
    purpose?: "preview" | "preannotate";
    status: string;
    id?: string;
    error?: string;
  };
};
export type TrainingOverview = {
  active_runs?: number;
  catalog: TrainingModel[];
  images: ReviewImage[];
  datasets: DatasetSnapshot[];
  runs: TrainingRun[];
  devices: ComputeDevice[];
  hosted: boolean;
};
