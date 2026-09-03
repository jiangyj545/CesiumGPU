/**
 * TileKey - Web Mercator tile coordinates (z/x/y).
 *
 * Level 0 is a single tile covering the whole world (longitude [-π, π],
 * latitude [-π/2, π/2] via Web Mercator). Each level quadrants its parent:
 * children() of (z, x, y) are (z+1, 2x..2x+1, 2y..2y+1), with y growing
 * southward (image space, top row = north) as in the standard XYZ scheme.
 */

/** Latitude/longitude bounds of a tile, in radians. */
export interface TileRectangle {
  /** Western edge longitude (radians). */
  west: number;
  /** Southern edge latitude (radians). */
  south: number;
  /** Eastern edge longitude (radians). */
  east: number;
  /** Northern edge latitude (radians). */
  north: number;
}

export class TileKey {
  /** Zoom level (0 = whole world). */
  readonly level: number;
  /** Tile column, in [0, 2^level - 1]. */
  readonly x: number;
  /** Tile row, in [0, 2^level - 1]; y = 0 is the northernmost row. */
  readonly y: number;

  constructor(level: number, x: number, y: number) {
    this.level = level;
    this.x = x;
    this.y = y;
  }

  /** Canonical string form `"{level}/{x}/{y}"` used as cache key. */
  get key(): string {
    return `${this.level}/${this.x}/${this.y}`;
  }

  /** Parse a key produced by {@link TileKey.key}. */
  static fromKey(key: string): TileKey {
    const parts = key.split('/');
    return new TileKey(Number(parts[0]), Number(parts[1]), Number(parts[2]));
  }

  /** Number of tiles per axis at `level`. */
  static tilesAtLevel(level: number): number {
    return 2 ** level;
  }

  /**
   * Longitude (radians) → tile column at `level`.
   */
  static longitudeToTileX(longitude: number, level: number): number {
    const n = TileKey.tilesAtLevel(level);
    return clamp(
      Math.floor(((longitude + Math.PI) / (2.0 * Math.PI)) * n),
      0,
      n - 1
    );
  }

  /**
   * Latitude (radians) → tile row at `level` (Web Mercator, y grows south).
   */
  static latitudeToTileY(latitude: number, level: number): number {
    const n = TileKey.tilesAtLevel(level);
    const lat = clamp(latitude, -WebMercatorMaxLat, WebMercatorMaxLat);
    const sinLat = Math.sin(lat);
    const y = (0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI)) * n;
    return clamp(Math.floor(y), 0, n - 1);
  }

  /**
   * Tile row → northern edge latitude (radians):
   * `lat = atan(sinh(π (1 - 2y / 2^z)))`.
   */
  static tileYToLatitude(y: number, level: number): number {
    const n = TileKey.tilesAtLevel(level);
    return Math.atan(Math.sinh(Math.PI * (1.0 - (2.0 * y) / n)));
  }

  /**
   * Web Mercator tile bounds in radians. Exact in longitude; in latitude the
   * rectangle is bounded by the Mercator edge latitudes of rows y and y+1.
   */
  rectangle(): TileRectangle {
    const n = TileKey.tilesAtLevel(this.level);
    return {
      west:  (this.x / n) * 2.0 * Math.PI - Math.PI,
      east:  ((this.x + 1) / n) * 2.0 * Math.PI - Math.PI,
      north: TileKey.tileYToLatitude(this.y, this.level),
      south: TileKey.tileYToLatitude(this.y + 1, this.level),
    };
  }

  /** Parent tile, or `null` at level 0. */
  parent(): TileKey | null {
    if (this.level === 0) return null;
    return new TileKey(this.level - 1, this.x >> 1, this.y >> 1);
  }

  /** The four child tiles at `level + 1` (northwest, northeast, southwest, southeast). */
  children(): [TileKey, TileKey, TileKey, TileKey] {
    const level = this.level + 1;
    const x = this.x * 2;
    const y = this.y * 2;
    return [
      new TileKey(level, x, y),
      new TileKey(level, x + 1, y),
      new TileKey(level, x, y + 1),
      new TileKey(level, x + 1, y + 1),
    ];
  }

  toString(): string {
    return this.key;
  }
}

/**
 * Beyond ±85.05° latitude the Web Mercator projection degenerates; tiles
 * never extend past these bounds.
 */
const WebMercatorMaxLat = Math.atan(Math.sinh(Math.PI));

/** Local minimal clamp (avoids importing math/CesiumMath into a value object). */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export default TileKey;
