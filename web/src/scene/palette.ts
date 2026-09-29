import { Color } from 'three';

/** Class colours from the data contract (sRGB hex). */
export const CLASS_HEX: Record<string, string> = {
  car: '#2ef2c5',
  van: '#2ef2c5',
  motorcycle: '#4fd8ff',
  truck: '#ffb52e',
  bus: '#ffb52e',
  trailer: '#ffb52e',
  cyclist: '#ff4fd0',
  pedestrian: '#ff4fd0',
  escooter: '#ff4fd0',
  other: '#a3adff',
};

export const HEX = {
  tp: '#3dff8b',
  fp: '#ff3d58',
  fn: '#ff9f2e',
  gt: '#dfe6f0',
  accent: '#39e1ff',
};

const cache = new Map<string, Color>();
export function classColor(cls: string): Color {
  const hex = CLASS_HEX[cls] ?? CLASS_HEX.other;
  let c = cache.get(hex);
  if (!c) {
    c = new Color(hex);
    cache.set(hex, c);
  }
  return c;
}

export function classHex(cls: string): string {
  return CLASS_HEX[cls] ?? CLASS_HEX.other;
}

/** Per-LiDAR colours for the "sensor" colour mode. */
export const LIDAR_HEX = ['#35d6ff', '#ff5fc8', '#ffc53d', '#8cff5a'];
