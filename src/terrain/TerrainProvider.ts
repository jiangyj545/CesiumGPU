/**
 * TerrainProvider - Source of terrain (DEM) tiles.
 *
 * Parallel system to the imagery pipeline: terrain tiles decode into
 * triangulated meshes (normalised ECEF positions) instead of textures.
 * Wave2's TerrainLayer / sampleHeight consume this interface.
 *
 * Radius convention (matches EllipsoidGeometry / the base globe):
 * ECEF positions are normalised (equatorial semi-axis = 1); heights in
 * meters are folded in by multiplying the unit direction with
 * `1 + height / 6378137` (WGS84 semi-major axis).
 */

/** Tiling scheme of a terrain source. */
export type TerrainTilingScheme = 'geographic' | 'webmercator';

/**
 * Triangulated terrain mesh for one tile.
 *
 * Positions are normalised ECEF (unit ellipsoid scaled radially by height);
 * `center` is the tile centre in full-scale ECEF meters (error metrics /
 * bounding spheres).
 */
export interface TerrainMeshData {
  /** Normalised ECEF positions (x, y, z per vertex), |pos| ≈ 1 + h/6378137. */
  positions: Float32Array;
  /** Triangle indices. */
  indices: Uint16Array | Uint32Array;
  /** Minimum tile height (meters). */
  minHeight: number;
  /** Maximum tile height (meters). */
  maxHeight: number;
  /** Tile centre, ECEF meters (for error/bounding-sphere computation). */
  center: [number, number, number];
  /**
   * Index of the first skirt (edge-wall) vertex; `undefined` = mesh without
   * skirts. Vertices at or above this index are tile-rim duplicates pushed
   * radially downward — CPU sampling skips triangles touching them (their
   * interpolated heights are skirt artefacts, hundreds of metres below the
   * real terrain).
   */
  skirtVertexStart?: number;
}

/**
 * TerrainProvider - abstract tile-geometry source.
 *
 * `requestTileGeometry` resolving `null` means "no data for this tile";
 * the caller falls back to the parent tile.
 */
export interface TerrainProvider {
  /** Whether the provider is ready to serve tiles (metadata loaded). */
  readonly ready: boolean;
  /** Highest level for which this provider can serve tiles. */
  readonly maximumLevel: number;
  /** Tiling scheme. geographic = EPSG:4326 TMS-style (z0 = 2×1 tiles). */
  readonly tilingScheme: TerrainTilingScheme;

  /** Fetch and decode one tile. Resolves `null` when no data is available. */
  requestTileGeometry(
    z: number,
    x: number,
    y: number
  ): Promise<TerrainMeshData | null>;

  /** Cheap availability probe (no network). */
  getTileDataAvailable(z: number, x: number, y: number): boolean;

  /** Release resources and abort in-flight requests. Idempotent. */
  destroy(): void;
}

/** Options shared by all HTTP-backed terrain providers. */
export interface HttpTerrainProviderOptions {
  /**
   * URL template. Supported placeholders: `{z}`, `{x}`, `{y}` and the
   * optional `{s}` subdomain (rotated from `subdomains`).
   */
  url: string;
  /** Tiling scheme of the service. */
  tilingScheme: TerrainTilingScheme;
  /** Highest available level. */
  maximumLevel: number;
  /**
   * Extra HTTP headers sent with every tile request (P1-F auth parity with
   * Cesium's `new Resource({ url, headers })`, e.g. Authorization Bearer).
   */
  headers?: Record<string, string>;
  /** Subdomains for `{s}` rotation. Default: `[]` (no rotation). */
  subdomains?: readonly string[];
}

/** WGS84 polar / equatorial semi-axis ratio (mirrors EllipsoidGeometry). */
const WGS84_POLAR_RATIO = 6356752.3142 / 6378137.0;
/** WGS84 semi-major axis (meters); meters → normalised scale factor. */
const WGS84_SEMI_MAJOR = 6378137.0;
/** Default skirt depth floor (meters) for near-flat tiles. */
export const MIN_SKIRT_DEPTH = 50.0;

/**
 * Longitude bounds (degrees) of a geographic (EPSG:4326 TMS-style) tile.
 * At level z there are 2^(z+1) columns and 2^z rows (z0 = 2×1).
 */
export function geographicTileBounds(
  z: number,
  x: number,
  y: number
): { west: number; south: number; east: number; north: number } {
  const cols = 2 ** (z + 1);
  const rows = 2 ** z;
  const west  = (x / cols) * 360.0 - 180.0;
  const east  = ((x + 1) / cols) * 360.0 - 180.0;
  // TMS south-up: y is counted from the south pole (see decoder notes).
  const south = (y / rows) * 180.0 - 90.0;
  const north = ((y + 1) / rows) * 180.0 - 90.0;
  return { west, south, east, north };
}

/**
 * Longitude/latitude bounds (degrees) of a Web Mercator (slippy XYZ,
 * north-up) tile, matching the imagery TileKey formulas.
 */
export function webmercatorTileBounds(
  z: number,
  x: number,
  y: number
): { west: number; south: number; east: number; north: number } {
  const cols = 2 ** z;
  const west = (x / cols) * 360.0 - 180.0;
  const east = ((x + 1) / cols) * 360.0 - 180.0;
  // Inverse Web Mercator: lat = atan(sinh(π(1 - 2y/n))), radians → degrees.
  const north = (Math.atan(Math.sinh(Math.PI * (1.0 - (2.0 * y) / cols))) *
    180.0) / Math.PI;
  const south = (Math.atan(Math.sinh(Math.PI * (1.0 - (2.0 * (y + 1)) / cols))) *
    180.0) / Math.PI;
  return { west, south, east, north };
}

/**
 * Unit-ellipsoid ECEF direction for lon/lat (degrees), scaled radially by a
 * height in meters — identical normalisation to
 * `EllipsoidGeometry.createRegion` (y/z squashed by the WGS84 polar ratio).
 */
export function ecefFromLonLatHeight(
  lon: number,
  lat: number,
  heightMeters: number,
  out: [number, number, number]
): void {
  const rad = Math.PI / 180.0;
  const cosLat = Math.cos(lat * rad);
  const nx = cosLat * Math.cos(lon * rad);
  const ny = cosLat * Math.sin(lon * rad);
  const nz = Math.sin(lat * rad);
  const scale = 1.0 + heightMeters / WGS84_SEMI_MAJOR;
  out[0] = nx * scale;
  out[1] = ny * WGS84_POLAR_RATIO * scale;
  out[2] = nz * WGS84_POLAR_RATIO * scale;
}

/** Full-scale ECEF meters of the ellipsoid surface point at lon/lat (degrees). */
export function ecefMetersFromLonLat(
  lon: number,
  lat: number
): [number, number, number] {
  const rad = Math.PI / 180.0;
  const cosLat = Math.cos(lat * rad);
  return [
    cosLat * Math.cos(lon * rad) * WGS84_SEMI_MAJOR,
    cosLat * Math.sin(lon * rad) * WGS84_SEMI_MAJOR,
    Math.sin(lat * rad) * 6356752.3142,
  ];
}

/**
 * HttpTerrainProviderBase - shared plumbing for HTTP terrain services:
 * URL templating, per-session failure cache (a failed tile is never
 * re-requested), bounded concurrency (4) and cooperative destroy
 * (abort in-flight fetches, drain the queue).
 */
export abstract class HttpTerrainProviderBase implements TerrainProvider {
  readonly url: string;
  readonly tilingScheme: TerrainTilingScheme;
  readonly maximumLevel: number;
  readonly subdomains: readonly string[];
  /** Auth headers attached to every tile request (see options.headers). */
  readonly headers: Record<string, string> | undefined;

  private readonly _maxConcurrent = 4;
  private readonly _failedTiles = new Set<string>();
  private readonly _abortController = new AbortController();
  private _activeRequests = 0;
  private _pendingQueue: Array<() => void> = [];
  private _destroyed = false;

  constructor(options: HttpTerrainProviderOptions) {
    this.url          = options.url;
    this.tilingScheme = options.tilingScheme;
    this.maximumLevel = options.maximumLevel;
    this.subdomains   = options.subdomains ?? [];
    this.headers      = options.headers;
  }

  /** HTTP template services need no metadata negotiation — ready at once. */
  get ready(): boolean {
    return true;
  }

  /** True once {@link TerrainProvider.destroy} has been called. */
  get destroyed(): boolean {
    return this._destroyed;
  }

  /** Substitute `{z}/{x}/{y}` (and optional `{s}`) into the URL template. */
  resolveUrl(z: number, x: number, y: number): string {
    let url = this.url
      .replace('{z}', String(z))
      .replace('{x}', String(x))
      .replace('{y}', String(y));
    if (this.subdomains.length > 0 && url.includes('{s}')) {
      url = url.replace(
        '{s}',
        this.subdomains[(x + y + z) % this.subdomains.length]
      );
    }
    return url;
  }

  getTileDataAvailable(z: number, x: number, y: number): boolean {
    return z <= this.maximumLevel && !this._failedTiles.has(`${z}/${x}/${y}`);
  }

  async requestTileGeometry(
    z: number,
    x: number,
    y: number
  ): Promise<TerrainMeshData | null> {
    if (this._destroyed || z > this.maximumLevel) return null;
    const key = `${z}/${x}/${y}`;
    // 会话内失败缓存：同瓦片不重试（含 HTTP 错误 / 解码失败 / null 结果）。
    if (this._failedTiles.has(key)) return null;

    await this._acquireSlot();
    if (this._destroyed) {
      this._releaseSlot();
      return null;
    }

    try {
      const response = await fetch(this.resolveUrl(z, x, y), {
        mode: 'cors',
        signal: this._abortController.signal,
        headers: this.headers,
      });
      if (!response.ok) {
        console.warn(
          `[${this.constructor.name}] tile ${key} HTTP ${response.status}`
        );
        this._failedTiles.add(key);
        return null;
      }
      const mesh = await this.decodeTile(response, z, x, y);
      if (mesh === null) this._failedTiles.add(key);
      return mesh;
    } catch (error) {
      console.warn(`[${this.constructor.name}] tile ${key} failed`, error);
      this._failedTiles.add(key);
      return null;
    } finally {
      this._releaseSlot();
    }
  }

  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._abortController.abort();
    // Drain the queue: resumed callers observe _destroyed and bail out.
    const queue = this._pendingQueue;
    this._pendingQueue = [];
    for (const resume of queue) resume();
  }

  /** Decode a successful HTTP response into tile geometry. */
  protected abstract decodeTile(
    response: Response,
    z: number,
    x: number,
    y: number
  ): Promise<TerrainMeshData | null>;

  /** Semaphore acquire: caps in-flight requests at {@link _maxConcurrent}. */
  private _acquireSlot(): Promise<void> {
    if (this._activeRequests < this._maxConcurrent) {
      this._activeRequests++;
      return Promise.resolve();
    }
    return new Promise<void>((resume) => this._pendingQueue.push(resume));
  }

  /** Semaphore release: hand the slot to the next queued request, if any. */
  private _releaseSlot(): void {
    const resume = this._pendingQueue.shift();
    if (resume) {
      resume();
    } else if (this._activeRequests > 0) {
      this._activeRequests--;
    }
  }
}
