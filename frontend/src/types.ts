export type XY = [number, number];
export type Mask = { size: [number, number]; counts: string };
export type Component = { outer: XY[]; holes: XY[][] };
export type Category = {
  id: number;
  name: string;
  color: string;
  supercategory?: string;
};
export type SourceImage = {
  id: number;
  file_name: string;
  width: number;
  height: number;
  annotation_count: number;
  annotation_counts?: Record<Task, number>;
};
export type Project = {
  id?: string;
  name: string;
  directory: string;
  image_root: string;
  categories: Category[];
  images: SourceImage[];
};
export type HostedProject = Omit<Project, "directory" | "image_root"> & {
  id: string;
};
export type HostedSession = {
  user: { id: string; email: string; name: string } | null;
  csrf_token: string | null;
  usage: {
    storage_bytes: number;
    storage_limit_bytes: number;
    inferences_used: number;
    inference_limit: number;
  } | null;
};
export type SegmentationAnnotation = {
  kind?: "segmentation";
  id: string;
  category_id: number;
  mask: Mask;
  components: Component[];
  controls?: Component[];
  preview?: string;
  iscrowd: number;
};
export type Point = { x: number; y: number; label: 0 | 1 };
export type Part = {
  id: string;
  points: Point[];
  box?: [number, number, number, number];
  mask?: Mask;
  seed_mask?: Mask;
  polygon?: { vertices: XY[]; closed: boolean };
  preview?: string;
  components?: Component[];
  controls?: Component[];
};
export type Draft = {
  id: string;
  category_id: number;
  parts: Part[];
  active_part_id: string;
};
export type Proposal = SegmentationAnnotation & {
  score: number;
  selected: boolean;
};
export type ImageState = {
  image_id: number;
  revision: number;
  annotations: SegmentationAnnotation[];
  draft: Draft | null;
  proposals: Proposal[];
};
export type GeometryResult = {
  mask: Mask;
  components: Component[];
  controls?: Component[];
  preview?: string;
};
export type ModelStatus = { state: string; device?: string; message?: string };
export type Tool =
  "select" | "pan" | "positive" | "negative" | "box" | "polygon";

export type Task = "segmentation" | "detection";
export type BBox = [number, number, number, number]; // x, y, width, height, original pixels
export type BoxAnnotation = {
  kind: "bbox";
  id: string;
  category_id: number;
  iscrowd: number;
  bbox: BBox;
};
export type Annotation = SegmentationAnnotation | BoxAnnotation;
export type BoxProposal = BoxAnnotation & { score: number; selected: boolean };
export type BoxDraft = {
  id: string;
  category_id: number;
  bbox?: BBox;
  prompt_bbox?: BBox;
  points: Point[];
};
export type BoxAdjustment = { target_id: string; base_bbox: BBox; bbox: BBox };
export type DetectionWork = {
  draft: BoxDraft | null;
  proposals: BoxProposal[];
  adjustment: BoxAdjustment | null;
};
export type DetectionState = DetectionWork & { annotations: BoxAnnotation[] };
export type ProjectImageState = Omit<ImageState, "annotations"> & {
  schema_version?: 2;
  annotations: Annotation[];
  detection?: DetectionWork;
};
