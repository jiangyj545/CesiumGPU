import { Camera } from '../scene/Camera';
import { Cartesian3 } from '../math/Cartesian3';
import {
  type TerrainProvider,
  type TerrainMeshData,
  geographicTileBounds,
  webmercatorTileBounds,
} from './TerrainProvider';

/**
 * TerrainLayer - Streams terrain meshes for one {@link TerrainProvider} and
 * exposes them to the renderer / height sampling (Wave2-A).
 *
 * Parallel to {@link import('../imagery/ImageryLayer').ImageryLayer}:
 * per update (called once per frame by `Globe.update`)
 * 1. Quadtree selection from the scheme's root grid: geographic tiling has
 *    `2^(z+1)` columns / `2^z` rows (z0 = 2×1, TMS south-up — see
 *    `QuantizedMeshDecoder` notes); webmercator is the standard north-up XYZ
 *    grid. Tiles are refined while their screen-space error exceeds
 *    `maximumScreenSpaceError`, view-frustum + horizon culled.
 * 2. Bounded, distance-ordered `requestTileGeometry` fetches (deduped; the
 *    provider's own failure cache prevents re-requesting missing tiles).
 * 3. Ready meshes land in a ~64-entry LRU. Tiles whose own mesh is missing
 *    render with the nearest loaded ancestor mesh (parent fallback).
 * 4. `onTileReady` fires once per arrived mesh so the globe can rebuild the
 *    corresponding primitive.
 *
 * {@link sampleHeight} reads the loaded meshes directly: the tile containing
 * the query point (finest level first) is interpolated with barycentric
 * coordinates over its triangles. Returns `0` when nothing is loaded.
 */
export class TerrainLayer {
  /** The terrain source backing this layer. */
  readonly provider: TerrainProvider;

  private readonly _maximumScreenSpaceError: number;
  private readonly _maximumLevel: number;
  private readonly _rootLevel: number;
  private readonly _cacheCapacity: number;
  private readonly _maxConcurrentRequests: number;
  private readonly _onTileReady: TerrainLayerOptions['onTileReady'];

  /** LRU mesh cache keyed by `"{z}/{x}/{y}"` (insertion order = recency). */
  private readonly _meshes = new Map<string, TerrainCacheEntry>();

  /** Keys with a fetch in progress (dedup guard). */
  private readonly _inflight = new Set<string>();

  /** Keys the provider reported as unavailable; not retried this session. */
  private readonly _failed = new Set<string>();

  /** Selection result of the last update (sorted near → far). */
  private _selected: SelectedTerrainTile[] = [];

  /** Render set derived from the selection: tiles with a usable mesh. */
  private _renderTiles: TerrainRenderTile[] = [];

  /** CPU-side tile bounding volumes, keyed by tile key. */
  private readonly _volumes = new Map<string, TerrainTileVolume>();

  /** Frustum planes (a, b, c, d) × 6, normalised, refreshed per update. */
  private readonly _frustumPlanes = new Float32Array(24);

  private _destroyed = false;

  constructor(options: TerrainLayerOptions) {
    this.provider                 = options.provider;
    this._maximumScreenSpaceError = options.maximumScreenSpaceError ?? 2.0;
    this._maximumLevel            = options.maximumLevel ?? this.provider.maximumLevel;
    this._rootLevel               = options.rootLevel ?? TerrainLayer.ROOT_LEVEL;
    this._cacheCapacity           = options.cacheCapacity ?? DEFAULT_CACHE_CAPACITY;
    this._maxConcurrentRequests   = options.maxConcurrentRequests ?? 10;
    this._onTileReady             = options.onTileReady;
  }

  /** Root level the quadtree selection starts from. */
  static readonly ROOT_LEVEL = 1;

  /** Aggregate counters for HUD/debug. */
  get stats(): TerrainLayerStats {
    return {
      selectedTiles:    this._selected.length,
      renderTiles:      this._renderTiles.length,
      cachedMeshes:     this._meshes.size,
      inFlightRequests: this._inflight.size,
      failedTiles:      this._failed.size,
    };
  }

  /** Tiles selected for rendering by the last {@link update}. */
  get renderTiles(): readonly TerrainRenderTile[] {
    return this._renderTiles;
  }

  /**
   * One frame of work: reselect tiles, launch bounded mesh requests and
   * rebuild the render set (selected tiles with own-or-ancestor meshes).
   *
   * @param viewportHeight drawing-buffer height in pixels (SSE scaling);
   *   defaults to 1080 when the caller cannot supply it.
   */
  update(camera: Camera, viewportHeight: number = 1080): void {
    if (this._destroyed) return;
    this._selectTiles(camera, viewportHeight);
    this._launchRequests();
    this._buildRenderSet();
  }

  /** Cached mesh for a tile, or `null` (LRU recency refreshed on hit). */
  getMesh(z: number, x: number, y: number): TerrainMeshData | null {
    const entry = this._touch(`${z}/${x}/${y}`);
    return entry ? entry.mesh : null;
  }

  /**
   * Terrain elevation (metres) at a geodetic position, interpolated over the
   * finest loaded tile mesh covering the point. Returns `0` when no loaded
   * tile covers it (consumers treat that as "clamp to the base ellipsoid").
   *
   * `level` (optional) restricts the search to tiles at or below that level.
   */
  sampleHeight(lon: number, lat: number, level?: number): number {
    if (this._destroyed) return 0;

    let best: TerrainCacheEntry | null = null;
    for (const entry of this._meshes.values()) {
      if (level !== undefined && entry.z > level) continue;
      if (lon < entry.bounds.west || lon > entry.bounds.east) continue;
      if (lat < entry.bounds.south || lat > entry.bounds.north) continue;
      if (best === null || entry.z > best.z) best = entry;
    }
    if (best === null) return 0;

    this._touch(best.key); // sampling keeps the tile resident
    return this._interpolateHeight(best, lon, lat);
  }

  /** Release everything and abort activity. In-flight fetches drain to no-ops. */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._meshes.clear();
    this._inflight.clear();
    this._failed.clear();
    this._selected = [];
    this._renderTiles = [];
    this._volumes.clear();
  }

  // ── Private ────────────────────────────────────────────────────────────────

  /**
   * Quadtree selection with view-frustum and horizon culling (iterative,
   * explicit stack). Grid geometry depends on the provider's tiling scheme:
   * geographic = `2^(z+1)` × `2^z` (TMS, y south-up), webmercator = standard
   * `2^z` × `2^z` XYZ (y north-up). Children of (z, x, y) are
   * (z+1, 2x..2x+1, 2y..2y+1) in both.
   */
  private _selectTiles(camera: Camera, viewportHeight: number): void {
    const planes  = this._extractFrustumPlanes(camera.viewProjectionMatrix.values);
    const camPos  = camera.position;
    const camDir  = Cartesian3.normalize(camPos, new Cartesian3());
    const camDist = Cartesian3.magnitude(camPos);

    // Horizon cone: surface points P stay visible while
    // dot(normalize(P), camDir) >= cos(θh), with cos(θh) = R / |cam|, R = 1.
    const horizonCos = Math.min(1.0, 1.0 / Math.max(camDist, 1.000001));
    const horizonSin = Math.sqrt(Math.max(0.0, 1.0 - horizonCos * horizonCos));

    const maxLevel   = Math.min(this._maximumLevel, this.provider.maximumLevel);
    const rootLevel  = Math.min(this._rootLevel, maxLevel);
    const tanHalfFov = Math.tan(camera.frustum.fov * 0.5);

    const selected: SelectedTerrainTile[] = [];
    const stack: TerrainTileRef[] = [];

    const rootCols = this._columnsAtLevel(rootLevel);
    const rootRows = 2 ** rootLevel;
    for (let y = 0; y < rootRows; y++) {
      for (let x = 0; x < rootCols; x++) {
        stack.push({ z: rootLevel, x, y, key: `${rootLevel}/${x}/${y}` });
      }
    }

    while (stack.length > 0) {
      const tile   = stack.pop() as TerrainTileRef;
      const volume = this._getVolume(tile);

      // ── View-frustum culling (6 planes) ────────────────────────────────
      let outside = false;
      for (let i = 0; i < 6; i++) {
        const o = i * 4;
        const d = planes[o]     * volume.center.x +
                  planes[o + 1] * volume.center.y +
                  planes[o + 2] * volume.center.z +
                  planes[o + 3];
        if (d < -volume.radius) {
          outside = true;
          break;
        }
      }
      if (outside) continue;

      // ── Horizon culling (simplified cone test with tile angular margin) ─
      const centerLen = Cartesian3.magnitude(volume.center);
      const facingDot = Cartesian3.dot(volume.center, camDir) / Math.max(centerLen, 1e-9);
      const sinAngle  = Math.min(0.99, volume.radius);
      const cosAngle  = Math.sqrt(Math.max(0.0, 1.0 - sinAngle * sinAngle));
      if (facingDot < horizonCos * cosAngle - horizonSin * sinAngle) continue;

      // ── Screen-space error refinement ──────────────────────────────────
      const distance = Math.max(Cartesian3.distance(camPos, volume.center), 1e-3);
      const sse =
        (volume.worldSize / TERRAIN_TILE_SIZE) *
        (viewportHeight / (distance * 2.0 * tanHalfFov));

      if (
        sse > this._maximumScreenSpaceError &&
        tile.z < maxLevel &&
        selected.length < MAX_SELECTED_TILES
      ) {
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            const z = tile.z + 1;
            const x = tile.x * 2 + dx;
            const y = tile.y * 2 + dy;
            stack.push({ z, x, y, key: `${z}/${x}/${y}` });
          }
        }
      } else {
        selected.push({ z: tile.z, x: tile.x, y: tile.y, key: tile.key, distance });
      }
    }

    // Near → far so request ordering and render-set dedup favour close tiles.
    selected.sort((a, b) => a.distance - b.distance);
    this._selected = selected;
  }

  /**
   * Launch mesh fetches for selected tiles that have nothing cached: deduped
   * against in-flight / failed sets, nearest-first, bounded in-flight count.
   * The provider applies its own failure cache and concurrency cap on top.
   */
  private _launchRequests(): void {
    for (const entry of this._selected) {
      if (this._inflight.size >= this._maxConcurrentRequests) break;

      const key = entry.key;
      if (this._meshes.has(key)) continue;
      if (this._inflight.has(key)) continue;
      if (this._failed.has(key)) continue;
      if (!this.provider.getTileDataAvailable(entry.z, entry.x, entry.y)) {
        this._failed.add(key);
        continue;
      }

      this._inflight.add(key);
      void this.provider.requestTileGeometry(entry.z, entry.x, entry.y).then(mesh => {
        this._inflight.delete(key);
        if (this._destroyed) return;
        if (mesh !== null) {
          this._putMesh(entry.z, entry.x, entry.y, mesh);
        } else {
          // Provider has no data here; ancestor fallback renders instead.
          this._failed.add(key);
        }
      }).catch((error: unknown) => {
        this._inflight.delete(key);
        this._failed.add(key);
        console.warn(`TerrainLayer: request for tile ${key} failed`, error);
      });
    }
  }

  /**
   * Render set: each selected tile with its own mesh, or the nearest loaded
   * ancestor mesh (parent fallback). Ancestor tiles shared by several
   * selected children render once (dedup by key).
   */
  private _buildRenderSet(): void {
    const renderTiles: TerrainRenderTile[] = [];
    const seen = new Set<string>();

    for (const entry of this._selected) {
      if (this._touch(entry.key) !== null) {
        if (!seen.has(entry.key)) {
          seen.add(entry.key);
          renderTiles.push({ z: entry.z, x: entry.x, y: entry.y, ancestorLevels: 0 });
        }
        continue;
      }

      let z = entry.z;
      let x = entry.x;
      let y = entry.y;
      for (let levels = 1; levels <= MAX_FALLBACK_LEVELS && z > 0; levels++) {
        z -= 1;
        x >>= 1;
        y >>= 1;
        const key = `${z}/${x}/${y}`;
        if (this._touch(key) === null) continue;
        if (!seen.has(key)) {
          seen.add(key);
          renderTiles.push({ z, x, y, ancestorLevels: levels });
        }
        break;
      }
    }

    this._renderTiles = renderTiles;
  }

  /** LRU get: refreshes recency and evicts the oldest entry past capacity. */
  private _touch(key: string): TerrainCacheEntry | null {
    const entry = this._meshes.get(key);
    if (entry === undefined) return null;
    this._meshes.delete(key);
    this._meshes.set(key, entry); // re-insert = most recently used
    return entry;
  }

  /** Insert a freshly fetched mesh (evicting the least recently used). */
  private _putMesh(z: number, x: number, y: number, mesh: TerrainMeshData): void {
    const key = `${z}/${x}/${y}`;
    while (this._meshes.size >= this._cacheCapacity) {
      const oldest = this._meshes.keys().next();
      if (oldest.done) break;
      this._meshes.delete(oldest.value);
    }
    this._meshes.set(key, {
      key,
      z, x, y,
      mesh,
      bounds: this._boundsOf(z, x, y),
      llh: null, // built lazily by sampleHeight
    });
    this._onTileReady?.(z, x, y, mesh);
  }

  /**
   * Barycentric height interpolation over the tile's triangles.
   *
   * Vertex lon/lat/height are derived from the normalised ECEF positions by
   * undoing the engine's polar squash (`y, z × polar ratio`), then height =
   * `(|pos| / |unit direction| - 1) × 6378137`. Skirt triangles project to
   * (near-)degenerate 2D triangles and are rejected by the epsilon test.
   */
  private _interpolateHeight(entry: TerrainCacheEntry, lonDeg: number, latDeg: number): number {
    if (entry.llh === null) entry.llh = this._buildVertexLonLatHeight(entry.mesh);
    const llh     = entry.llh;
    const indices = entry.mesh.indices;

    // Work in tile-local coordinates: absolute lon/lat near ±180 have huge
    // magnitude where the squared-area denominators lose precision (a
    // z12 tile spans only ~0.04° while the coordinates sit at ~100°).
    const ox = entry.bounds.west;
    const oy = entry.bounds.south;
    const px = lonDeg - ox;
    const py = latDeg - oy;

    for (let i = 0; i < indices.length; i += 3) {
      const a = indices[i]     * 3;
      const b = indices[i + 1] * 3;
      const c = indices[i + 2] * 3;

      const ax = llh[a] - ox, ay = llh[a + 1] - oy;
      const e0x = llh[b] - ox - ax, e0y = llh[b + 1] - oy - ay;
      const e1x = llh[c] - ox - ax, e1y = llh[c + 1] - oy - ay;
      const v0x = px - ax,          v0y = py - ay;

      const d00 = e0x * e0x + e0y * e0y;
      const d01 = e0x * e1x + e0y * e1y;
      const d11 = e1x * e1x + e1y * e1y;
      const d20 = v0x * e0x + v0y * e0y;
      const d21 = v0x * e1x + v0y * e1y;

      const denom = d00 * d11 - d01 * d01;
      const denomAbs = Math.abs(denom);
      if (denomAbs < 1e-18) continue; // degenerate (skirt wall / pole pinch)

      const u = (d11 * d20 - d01 * d21) / denom;
      const v = (d00 * d21 - d01 * d20) / denom;
      const eps = -1e-6;
      if (u < eps || v < eps || u + v > 1 - eps) continue;

      const h = llh[a + 2] + u * (llh[b + 2] - llh[a + 2]) + v * (llh[c + 2] - llh[a + 2]);
      return Number.isFinite(h) ? h : 0;
    }
    return 0; // point fell in a skirt seam — approximate as ellipsoid level
  }

  /** Per-vertex [lonDeg, latDeg, heightM] triplets derived from positions. */
  private _buildVertexLonLatHeight(mesh: TerrainMeshData): Float64Array {
    const positions = mesh.positions;
    const count     = positions.length / 3;
    const llh       = new Float64Array(count * 3);
    const rad       = 180.0 / Math.PI;

    for (let i = 0; i < count; i++) {
      const x = positions[i * 3];
      const y = positions[i * 3 + 1] / WGS84_POLAR_RATIO; // undo squash
      const z = positions[i * 3 + 2] / WGS84_POLAR_RATIO;

      const lon = Math.atan2(y, x);
      const lat = Math.atan2(z, Math.hypot(x, y));
      // Radial scale folded into the position: |p| = scale·|unit dir|;
      // height = (scale - 1) × WGS84 semi-major axis.
      const scaleSq = x * x + y * y + z * z;
      const height  = (Math.sqrt(scaleSq) - 1.0) * WGS84_SEMI_MAJOR;

      llh[i * 3]     = lon * rad;
      llh[i * 3 + 1] = lat * rad;
      llh[i * 3 + 2] = height;
    }
    return llh;
  }

  /** Tile longitude/latitude bounds (degrees) per the provider's scheme. */
  private _boundsOf(z: number, x: number, y: number): TerrainTileBounds {
    return this.provider.tilingScheme === 'geographic'
      ? geographicTileBounds(z, x, y)
      : webmercatorTileBounds(z, x, y);
  }

  /** Tile columns at a level (geographic doubles the longitude axis). */
  private _columnsAtLevel(level: number): number {
    return this.provider.tilingScheme === 'geographic' ? 2 ** (level + 1) : 2 ** level;
  }

  /** Bounding sphere + world size for a tile (computed once, then cached). */
  private _getVolume(tile: TerrainTileRef): TerrainTileVolume {
    const cached = this._volumes.get(tile.key);
    if (cached) return cached;

    const rect      = this._boundsOf(tile.z, tile.x, tile.y);
    const centerLon = (rect.west + rect.east) / 2.0;
    const centerLat = (rect.south + rect.north) / 2.0;
    const center    = new Cartesian3();
    ecefSurfacePoint(centerLon, centerLat, center);

    let radius = 0.0;
    for (const [lat, lon] of [
      [rect.north, rect.west],
      [rect.north, rect.east],
      [rect.south, rect.west],
      [rect.south, rect.east],
    ] as const) {
      const d = Cartesian3.distance(center, ecefSurfacePoint(lon, lat, new Cartesian3()));
      if (d > radius) radius = d;
    }
    radius *= 1.15; // margin for the plane / horizon tests + terrain relief

    const rad       = Math.PI / 180.0;
    const widthDeg  = (rect.east - rect.west) * rad;
    const heightDeg = (rect.north - rect.south) * rad;
    const worldSize = Math.max(
      widthDeg * Math.cos(centerLat * rad),
      heightDeg
    ) * WGS84_POLAR_RATIO;

    const volume: TerrainTileVolume = { center, radius, worldSize };
    if (this._volumes.size > MAX_VOLUMES) this._volumes.clear();
    this._volumes.set(tile.key, volume);
    return volume;
  }

  /**
   * Extract the 6 view-frustum planes from the viewProj matrix (WebGPU
   * clip space: x,y ∈ [-w, w], z ∈ [0, w]). `values` is column-major, so
   * matrix row `r` is `[v[r], v[4+r], v[8+r], v[12+r]]`.
   */
  private _extractFrustumPlanes(values: Float32Array): Float32Array {
    const p  = this._frustumPlanes;
    const a0 = values[0], a1 = values[4], a2 = values[8],  a3 = values[12]; // row 0
    const b0 = values[1], b1 = values[5], b2 = values[9],  b3 = values[13]; // row 1
    const c0 = values[2], c1 = values[6], c2 = values[10], c3 = values[14]; // row 2
    const d0 = values[3], d1 = values[7], d2 = values[11], d3 = values[15]; // row 3

    const set = (i: number, x: number, y: number, z: number, w: number): void => {
      const len = Math.hypot(x, y, z) || 1.0;
      p[i * 4]     = x / len;
      p[i * 4 + 1] = y / len;
      p[i * 4 + 2] = z / len;
      p[i * 4 + 3] = w / len;
    };

    set(0, d0 + a0, d1 + a1, d2 + a2, d3 + a3); // left   ( x >= -w)
    set(1, d0 - a0, d1 - a1, d2 - a2, d3 - a3); // right  ( x <=  w)
    set(2, d0 + b0, d1 + b1, d2 + b2, d3 + b3); // bottom ( y >= -w)
    set(3, d0 - b0, d1 - b1, d2 - b2, d3 - b3); // top    ( y <=  w)
    set(4, c0,      c1,      c2,      c3     ); // near   ( z >=  0)
    set(5, d0 - c0, d1 - c1, d2 - c2, d3 - c3); // far    ( z <=  w)

    return p;
  }
}

/** @internal Internal quadtree cursor (grid differs from imagery TileKey). */
interface TerrainTileRef {
  z: number;
  x: number;
  y: number;
  key: string;
}

/** @internal Selected tile plus the camera distance used for ordering. */
interface SelectedTerrainTile {
  z: number;
  x: number;
  y: number;
  key: string;
  distance: number;
}

/** @internal Cached mesh plus the metadata height sampling needs. */
interface TerrainCacheEntry {
  key: string;
  z: number;
  x: number;
  y: number;
  mesh: TerrainMeshData;
  /** Tile bounds in degrees (fast containment test for sampleHeight). */
  bounds: TerrainTileBounds;
  /** Lazily built per-vertex [lonDeg, latDeg, heightM] triplets. */
  llh: Float64Array | null;
}

/** @internal CPU-side tile bounds used by the culling / LOD tests. */
interface TerrainTileVolume {
  /** Surface point at the tile centre (normalised ECEF, radius ≈ 1). */
  center: Cartesian3;
  /** Bounding-sphere radius around `center` (normalised units). */
  radius: number;
  /** Largest tile edge length (normalised units) for the SSE test. */
  worldSize: number;
}

/** Tile bounds in degrees. */
export interface TerrainTileBounds {
  west: number;
  south: number;
  east: number;
  north: number;
}

/** Tile selected for rendering (own mesh, or an ancestor's — see `ancestorLevels`). */
export interface TerrainRenderTile {
  z: number;
  x: number;
  y: number;
  /** Levels between this tile and the mesh actually rendered (0 = own mesh). */
  ancestorLevels: number;
}

export interface TerrainLayerOptions {
  /** The terrain source for this layer. */
  provider: TerrainProvider;
  /** Refine tiles while their SSE exceeds this. Default: 2 */
  maximumScreenSpaceError?: number;
  /** Highest level to refine to. Default: provider.maximumLevel */
  maximumLevel?: number;
  /** Quadtree root level. Default: 1 */
  rootLevel?: number;
  /** LRU mesh pool capacity. Default: 64 */
  cacheCapacity?: number;
  /** Concurrent fetch cap for this layer. Default: 10 */
  maxConcurrentRequests?: number;
  /** Invoked once when a tile mesh arrives (globe rebuilds its primitive). */
  onTileReady?: (z: number, x: number, y: number, mesh: TerrainMeshData) => void;
}

export interface TerrainLayerStats {
  /** Tiles selected by the last update. */
  selectedTiles: number;
  /** Tiles with a usable (own or ancestor) mesh. */
  renderTiles: number;
  /** Meshes resident in the LRU pool. */
  cachedMeshes: number;
  /** Fetches currently in flight. */
  inFlightRequests: number;
  /** Tiles the provider reported as unavailable (not retried). */
  failedTiles: number;
}

/**
 * Reference vertex count of a quantized-mesh grid (65 × 65); the "texel
 * size" used by the SSE test. A coarser constant just over/under-refines.
 */
const TERRAIN_TILE_SIZE = 65;

/** Default LRU mesh pool capacity (~64 grids). */
const DEFAULT_CACHE_CAPACITY = 64;

/** Ancestor levels walked for the parent-mesh fallback. */
const MAX_FALLBACK_LEVELS = 6;

/** Safety cap on the selected set (guards against SSE blowups). */
const MAX_SELECTED_TILES = 512;

/** Cap on the CPU bounding-volume cache. */
const MAX_VOLUMES = 8192;

/** WGS84 polar / equatorial ratio (matches EllipsoidGeometry normalisation). */
const WGS84_POLAR_RATIO = 6356752.3142 / 6378137.0;

/** WGS84 semi-major axis (metres). */
const WGS84_SEMI_MAJOR = 6378137.0;

/**
 * Surface point (normalised ECEF, EllipsoidGeometry convention:
 * x = cos(lat)·cos(lon), y and z scaled by the polar ratio).
 */
function ecefSurfacePoint(lonDeg: number, latDeg: number, result: Cartesian3): Cartesian3 {
  const rad    = Math.PI / 180.0;
  const cosLat = Math.cos(latDeg * rad);
  result.x = cosLat * Math.cos(lonDeg * rad);
  result.y = cosLat * Math.sin(lonDeg * rad) * WGS84_POLAR_RATIO;
  result.z = Math.sin(latDeg * rad) * WGS84_POLAR_RATIO;
  return result;
}

export default TerrainLayer;
