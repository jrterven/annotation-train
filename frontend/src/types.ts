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
};
export type Project = {
  name: string;
  directory: string;
  image_root: string;
  categories: Category[];
  images: SourceImage[];
};
export type Annotation = {
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
export type Proposal = Annotation & { score: number; selected: boolean };
export type ImageState = {
  image_id: number;
  revision: number;
  annotations: Annotation[];
  draft: Draft | null;
  proposals: Proposal[];
};
export type GeometryResult = {
  mask: Mask;
  components: Component[];
  controls?: Component[];
  preview: string;
};
export type ModelStatus = { state: string; device?: string; message?: string };
export type Tool = "select" | "pan" | "positive" | "negative" | "box" | "polygon";
