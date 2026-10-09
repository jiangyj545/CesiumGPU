import { Cartesian3 } from '../math/Cartesian3';

/**
 * Billboard / label styling & anchoring value types, mirroring the Cesium
 * enums consumed by hexapod_gcs label-heavy call sites (P2-4).
 */

/** Label fill style (Cesium `LabelStyle`). */
export const LabelStyle = {
  FILL: 0,
  OUTLINE: 1,
  FILL_AND_OUTLINE: 2,
} as const;
export type LabelStyle = (typeof LabelStyle)[keyof typeof LabelStyle];

/** Horizontal anchor of a billboard relative to its position (Cesium `HorizontalOrigin`). */
export const HorizontalOrigin = {
  LEFT: 0,
  CENTER: 1,
  RIGHT: 2,
} as const;
export type HorizontalOrigin = (typeof HorizontalOrigin)[keyof typeof HorizontalOrigin];

/** Vertical anchor of a billboard relative to its position (Cesium `VerticalOrigin`). */
export const VerticalOrigin = {
  BASELINE: 0,
  BOTTOM: 1,
  CENTER: 2,
  TOP: 3,
} as const;
export type VerticalOrigin = (typeof VerticalOrigin)[keyof typeof VerticalOrigin];

/** Entity height semantics against the terrain surface (Cesium `HeightReference`). */
export const HeightReference = {
  NONE: 0,
  CLAMP_TO_GROUND: 1,
  RELATIVE_TO_GROUND: 2,
} as const;
export type HeightReference = (typeof HeightReference)[keyof typeof HeightReference];

/**
 * Distance-based visibility window in metres (Cesium `DistanceDisplayCondition`):
 * an object renders while `near <= cameraDistance <= far` (inclusive on both
 * ends, matching Cesium's comparison).
 */
export class DistanceDisplayCondition {
  constructor(
    public near = 0.0,
    public far = Number.MAX_VALUE
  ) {}

  isVisible(distance: number): boolean {
    return distance >= this.near && distance <= this.far;
  }
}

/**
 * Offset (CSS px, y down) of a billboard's QUAD CENTRE relative to its
 * projected anchor point for the given origins:
 * - horizontal LEFT/CENTER/RIGHT → ∓width/2 / 0 / +width/2
 * - vertical TOP → +height/2 (element hangs below the anchor),
 *   CENTER → 0, BOTTOM/BASELINE → −height/2 (element sits above).
 */
export function billboardAnchorOffset(
  horizontal: HorizontalOrigin,
  vertical: VerticalOrigin,
  width: number,
  height: number
): { dx: number; dy: number } {
  const dx = ((horizontal - 1) * width) / 2;
  const dy = vertical === VerticalOrigin.CENTER ? 0 : vertical === VerticalOrigin.TOP ? height / 2 : -height / 2;
  return { dx, dy };
}

/**
 * Combined screen offset (P2-4 附注 pixelOffset)：九宫锚点偏移 + 逐像素
 * 偏移（Cesium `pixelOffset: Cartesian2`，hexapod 标签 (0,-40) 用法）。
 * `pixelOffset` 为 undefined 时等价 {@link billboardAnchorOffset}。
 */
export function billboardScreenOffset(
  horizontal: HorizontalOrigin,
  vertical: VerticalOrigin,
  width: number,
  height: number,
  pixelOffset?: { x: number; y: number }
): { dx: number; dy: number } {
  const { dx, dy } = billboardAnchorOffset(horizontal, vertical, width, height);
  return {
    dx: dx + (pixelOffset?.x ?? 0),
    dy: dy + (pixelOffset?.y ?? 0),
  };
}

/**
 * Whether depth testing is disabled for a billboard at `distanceMetres`:
 * Cesium `disableDepthTestDistance` — a number in metres (occlusion ignored
 * while the camera is within that distance) or `Infinity` (always visible).
 */
export function depthTestDisabled(
  disableDepthTestDistance: number | undefined,
  distanceMetres: number
): boolean {
  return disableDepthTestDistance !== undefined && distanceMetres <= disableDepthTestDistance;
}

const EARTH_SCALE = 6378137.0;
const WGS84_POLAR_RATIO = 6356752.3142 / 6378137.0;

/** Terrain height sampler shape (TerrainLayer-compatible). */
export type TerrainHeightSampler = { sampleHeight(lon: number, lat: number): number };

/**
 * Apply an entity-level {@link HeightReference} to a normalised-ECEF position
 * (P2-4): the position keeps its radial direction and is re-scaled against
 * the engine's flattened surface (`surfaceRadius`, same convention as
 * Camera.zoom clamping / Globe tiles):
 *
 * - NONE → position unchanged (no sampler call).
 * - CLAMP_TO_GROUND → altitude := sampled terrain height.
 * - RELATIVE_TO_GROUND → altitude := current altitude + sampled terrain height.
 *
 * Missing sampler / 0 sample behave as the ellipsoid surface (height 0).
 */
export function applyHeightReference(
  position: Cartesian3,
  heightReference: HeightReference | undefined,
  sampleHeight?: TerrainHeightSampler | null
): Cartesian3 {
  if (heightReference === undefined || heightReference === HeightReference.NONE || !sampleHeight) {
    return position;
  }

  const r = Cartesian3.magnitude(position);
  if (r < 1e-9) return position;
  const ux = position.x / r;
  const uy = (position.y / r) / WGS84_POLAR_RATIO;
  const uz = (position.z / r) / WGS84_POLAR_RATIO;
  const n = Math.sqrt(ux * ux + uy * uy + uz * uz);
  const cx2 = (ux / n) * (ux / n); // cosφcosλ 分量的平方（Camera.surfaceRadius 同式）
  const surfaceR = Math.sqrt(cx2 + WGS84_POLAR_RATIO * WGS84_POLAR_RATIO * (1.0 - cx2));

  const lonDeg = (Math.atan2(uy, ux) * 180) / Math.PI;
  const latDeg = (Math.atan2(uz, Math.hypot(ux, uy)) * 180) / Math.PI;
  const hTerrain = sampleHeight.sampleHeight(lonDeg, latDeg);

  const hCurrent = r - surfaceR;
  const h =
    heightReference === HeightReference.CLAMP_TO_GROUND
      ? hTerrain
      : hCurrent * EARTH_SCALE + hTerrain;
  const scale = (surfaceR + h / EARTH_SCALE) / r;
  return new Cartesian3(position.x * scale, position.y * scale, position.z * scale);
}
