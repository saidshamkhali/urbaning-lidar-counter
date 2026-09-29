// Types mirroring docs/DATA_FORMAT.md (the binding data contract).

/** [x, y, z, l, w, h, yaw] — z is the box centre, yaw counter-clockwise from +x. */
export type Box7 = [number, number, number, number, number, number, number];

export type ClassName =
  | 'car' | 'van' | 'truck' | 'bus' | 'trailer' | 'motorcycle'
  | 'cyclist' | 'escooter' | 'pedestrian' | 'other';

export type Group = 'vehicle' | 'vru' | 'other';

export interface IndexSequence {
  id: string;
  crossing: string;
  n_frames: number;
  duration_s: number;
  unique_vehicles?: number;
  unique_vehicles_gt?: number;
  unique_vehicles_gt_visible?: number;
  unique_vehicles_gt_trackable?: number;
}

export interface IndexFile {
  generated_at?: string;
  sequences: IndexSequence[];
}

export interface LidarInfo {
  name: string;
  index: number;
  position: [number, number, number];
}

export interface PointFormat {
  stride_bytes: number;
  scale: number;
  origin: [number, number, number];
  fields: string[];
}

export interface Meta {
  id: string;
  crossing: string;
  map?: string;
  fps: number;
  n_frames: number;
  timestamps: number[];
  bounds: { xmin: number; xmax: number; ymin: number; ymax: number; zmin: number; zmax: number };
  lidars: LidarInfo[];
  point_format: PointFormat;
  frames: { file: string; n_points: number }[];
}

export interface Detection {
  id: number;
  cls: string;
  box: Box7;
  score?: number;
  moving?: boolean;
  speed?: number;
  n_points?: number;
  /** detector: box predicted by the tracker without a supporting detection */
  coasted?: boolean;
  /** GT: not visible to the LiDARs in this frame */
  ignore?: boolean;
  /** GT: visible in enough frames to be countable (in sensor coverage) */
  trackable?: boolean;
  occlusion?: string;
}

export interface TrackInfo {
  cls: string;
  group?: Group;
  first: number;
  last: number;
  moving?: boolean;
  max_speed?: number;
  n_frames?: number;
}

export interface FrameCounts {
  vehicle: number;
  vru: number;
  other?: number;
  moving?: number;
  parked?: number;
  by_class?: Record<string, number>;
}

export interface UniqueCounts {
  vehicle: number;
  vru: number;
  other?: number;
  by_class?: Record<string, number>;
}

export interface DetectionsFile {
  source?: string;
  frames: Detection[][];
  tracks?: Record<string, TrackInfo>;
  counts?: FrameCounts[];
  unique?: UniqueCounts;
  /** GT only: vehicles with enough LiDAR points in that frame */
  counts_visible?: FrameCounts[];
  unique_visible?: UniqueCounts;
  /** GT only: vehicles visible in enough frames ("in sensor coverage"), counted whenever they exist */
  counts_trackable?: FrameCounts[];
  unique_trackable?: UniqueCounts;
}

export interface CountingMetrics {
  mae?: number;
  rmse?: number;
  bias?: number;
  within_1?: number;
  pred?: number[];
  gt?: number[];
  unique_pred?: number;
  unique_gt?: number;
}

export interface MetricsFile {
  detection?: {
    iou_thresholds?: number[];
    vehicle?: { ap?: Record<string, number>; precision?: number; recall?: number; f1?: number };
    per_class?: Record<string, { ap?: Record<string, number>; n_gt?: number; n_pred?: number }>;
    by_range?: { range: string; recall: number }[];
    pr_curve?: { recall: number[]; precision: number[] };
  };
  counting?: CountingMetrics;
  counting_visible?: CountingMetrics;
  counting_trackable?: CountingMetrics;
  tracking?: { mota?: number; motp?: number; id_switches?: number; unique_pred?: number; unique_gt?: number };
}

export interface MapLine {
  kind: string;
  subtype?: string;
  points: number[][];
}

export interface MapArea {
  kind: string;
  polygon: number[][];
}

export interface MapFile {
  crossing?: string;
  lines: MapLine[];
  areas: MapArea[];
}

export const VEHICLE_CLASSES = ['car', 'van', 'truck', 'bus', 'trailer', 'motorcycle'];
export const VRU_CLASSES = ['cyclist', 'escooter', 'pedestrian'];

export function groupOf(cls: string): Group {
  if (VEHICLE_CLASSES.includes(cls)) return 'vehicle';
  if (VRU_CLASSES.includes(cls)) return 'vru';
  return 'other';
}
