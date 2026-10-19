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

  /**
   * Stitch results keyed by tile key, reused while the neighbour-set
   * signature is unchanged (see {@link _stitchRenderSet}).
   */
  private readonly _stitchCache = new Map<string, StitchCacheEntry>();

  /** Keys with a fetch in progress (dedup guard). */
  private readonly _inflight = new Set<string>();

  /** Keys whose fetch failed (transient or data-less) — session-scoped skip. */
  private readonly _failed = new Set<string>();

  /**
   * 分析用途的瓦片 pin 集合（{@link pinRegion} 登记）。LRU 驱逐会跳过
   * 这些键，保证淹没/水深分析期间采样网格不被视锥调度挤出缓存。
   */
  private readonly _pinnedTiles = new Set<string>();

  /** 分析预载（{@link ensureRegion}）在途的请求，按键去重并可等待。 */
  private readonly _analysisInflight = new Map<string, Promise<void>>();

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
    this._maximumScreenSpaceError = options.maximumScreenSpaceError ?? DEFAULT_MAXIMUM_SSE;
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

  /** P2-5：已缓存网格数（按需渲染指纹——地形瓦片到达即变化）。 */
  get meshCount(): number {
    return this._meshes.size;
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
    this._stitchCache.clear();
    this._inflight.clear();
    this._failed.clear();
    this._pinnedTiles.clear();
    this._analysisInflight.clear();
    this._selected = [];
    this._renderTiles = [];
    this._volumes.clear();
  }

  // ── 区域预载与 pin（淹没 / 水深等离线分析用途）─────────────────────────────

  /**
   * 确保 `boundsDeg`（度）覆盖的全部 `level` 级瓦片网格都已进入 LRU 缓存。
   *
   * 绕过视锥选择（{@link _selectTiles}）直接按 bounds 反算瓦片范围，缺的
   * 瓦片走 {@link _maxConcurrentRequests} 并发上限逐个
   * `provider.requestTileGeometry` → `_putMesh` 入缓存。全部 settle 后返回：
   * - 请求 reject（网络/解码失败）→ 收集后抛带完整清单的错（不静默）；
   * - provider 返回 `null`（该瓦片无数据，如边界外）→ 记入失败缓存并
   *   `console.warn`，不算失败（与渲染调度同语义）。
   *
   * 幂等：已在缓存/在途的瓦片不会重复请求。完成后由调用方决定是否
   * {@link pinRegion} 钉住这些瓦片（防 LRU 挤出）。
   */
  async ensureRegion(
    boundsDeg: { west: number; south: number; east: number; north: number },
    level: number
  ): Promise<void> {
    if (this._destroyed) throw new Error('TerrainLayer.ensureRegion: 层已销毁');
    const z = Math.min(level, this._maximumLevel, this.provider.maximumLevel);
    const tiles = this._tilesInBounds(boundsDeg, z);
    if (tiles.length === 0) return;

    const failures: { key: string; reason: string }[] = [];
    const missing: string[] = [];
    let cursor = 0;
    const limit = Math.max(1, Math.min(this._maxConcurrentRequests, tiles.length));

    const fetchOne = async (tile: TerrainTileRef): Promise<void> => {
      const key = tile.key;
      // 与渲染调度（_launchRequests）在途的同一瓦片合并等待，避免双份请求。
      const pending = this._analysisInflight.get(key);
      if (pending) { await pending; return; }

      const task = (async (): Promise<void> => {
        if (this._meshes.has(key)) return;
        if (this._failed.has(key)) { missing.push(key); return; }
        if (!this.provider.getTileDataAvailable(tile.z, tile.x, tile.y)) {
          this._failed.add(key);
          missing.push(key);
          return;
        }
        try {
          const mesh = await this.provider.requestTileGeometry(tile.z, tile.x, tile.y);
          if (this._destroyed) return;
          if (mesh !== null) {
            this._putMesh(tile.z, tile.x, tile.y, mesh);
          } else {
            this._failed.add(key); // provider 声明无数据，与渲染调度同语义
            missing.push(key);
          }
        } catch (error) {
          failures.push({ key, reason: String((error as Error)?.message ?? error) });
        }
      })();

      this._analysisInflight.set(key, task);
      try {
        await task;
      } finally {
        this._analysisInflight.delete(key);
      }
    };

    const workers = Array.from({ length: limit }, async () => {
      while (cursor < tiles.length) {
        const tile = tiles[cursor++];
        await fetchOne(tile);
      }
    });
    await Promise.all(workers);

    if (missing.length > 0) {
      console.warn(
        `TerrainLayer.ensureRegion: ${missing.length}/${tiles.length} 瓦片无数据 ` +
        `（z=${z}）→ ${missing.join(', ')}`
      );
    }
    if (failures.length > 0) {
      throw new Error(
        `TerrainLayer.ensureRegion: ${failures.length}/${tiles.length} 瓦片请求失败 ` +
        `→ ${failures.map((f) => `${f.key}: ${f.reason}`).join('; ')}`
      );
    }
  }

  /**
   * 把 `boundsDeg` 覆盖的 `level` 级瓦片登记进 pin 集合（LRU 驱逐跳过）。
   * 返回当前 pin 总数。幂等（重复登记同一瓦片不重复计数）。
   *
   * 前置校验（防分析数据挤占渲染瓦片池）：`level <= 10` 且 pin 总数
   * `<= 256`，超限抛错。
   */
  pinRegion(
    boundsDeg: { west: number; south: number; east: number; north: number },
    level: number
  ): number {
    if (level > MAX_PIN_LEVEL) {
      throw new Error(`TerrainLayer.pinRegion: level ${level} 超过上限 ${MAX_PIN_LEVEL}（防挤占渲染瓦片）`);
    }
    const merged = new Set(this._pinnedTiles);
    for (const tile of this._tilesInBounds(boundsDeg, level)) merged.add(tile.key);
    if (merged.size > MAX_PINNED_TILES) {
      throw new Error(
        `TerrainLayer.pinRegion: pin 总数 ${merged.size} 超过上限 ${MAX_PINNED_TILES}（防挤占渲染瓦片）`
      );
    }
    for (const key of merged) this._pinnedTiles.add(key);
    return this._pinnedTiles.size;
  }

  /** 解除全部分析 pin（LRU 恢复正常驱逐）。 */
  unpinAll(): void {
    this._pinnedTiles.clear();
  }

  /**
   * 解除 `boundsDeg` 覆盖的 `level` 级瓦片 pin（只删这些键，不动其它
   * 分析 pin；{@link sampleTerrain} 采样周期专用，避免 unpinAll 误伤
   * 淹没/水深等长周期 pin）。
   */
  unpinRegion(
    boundsDeg: { west: number; south: number; east: number; north: number },
    level: number
  ): void {
    for (const tile of this._tilesInBounds(boundsDeg, level)) {
      this._pinnedTiles.delete(tile.key);
    }
  }

  // ── Private ────────────────────────────────────────────────────────────────

  /**
   * `ensureRegion` 实际加载的层级（`level` 钳制到 layer / provider 最大值）。
   * 采样 pin 需与预载层级对齐（见 sampleTerrain）。
   */
  effectiveLevel(level: number): number {
    return Math.min(level, this._maximumLevel, this.provider.maximumLevel);
  }

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
          renderTiles.push(emptyRenderTile(entry.z, entry.x, entry.y, 0));
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
          renderTiles.push(emptyRenderTile(z, x, y, levels));
        }
        break;
      }
    }

    this._stitchRenderSet(renderTiles);
    this._renderTiles = renderTiles;
  }

  // ── Crack stitching (T-junction removal across LOD borders) ───────────────

  /**
   * Snap the rendered positions of every tile in the render set onto the
   * actually-rendered neighbour surfaces.
   *
   * Adjacent tiles at different LODs do not share edge vertices: the fine
   * tile's edge carries more samples than the coarse neighbour's edge grid,
   * and the mismatching heights open hairline (T-junction) cracks between
   * the triangle rows. For every rendered tile each of the four edges is
   * matched against the *rendered* neighbour (same level first, then the
   * nearest rendered ancestor — render-set fallbacks are indexed under the
   * ancestor's own coordinates). A coarser neighbour means this tile is the
   * fine side: its edge vertices are re-sampled from the neighbour's mesh
   * surface, so the fine edge polyline lies exactly on the coarse surface
   * and the two grids meet watertight.
   *
   * Same-level neighbours are skipped: their boundary vertices come from the
   * same quantised grid edge and coincide by construction. Results are kept
   * in {@link _stitchCache} keyed by a signature of the resolved
   * coarse-neighbour set, so stitching recomputes only when the rendered
   * neighbourhood actually changes — never per frame at a stable LOD.
   */
  private _stitchRenderSet(renderTiles: TerrainRenderTile[]): void {
    const rendered = new Map<string, TerrainRenderTile>();
    for (const tile of renderTiles) {
      rendered.set(renderTileKey(tile.z, tile.x, tile.y), tile);
    }

    for (const tile of renderTiles) {
      // Render-set entries are guaranteed resident (see _buildRenderSet);
      // plain `get` avoids churning the LRU order from the stitch pass.
      const self = this._meshes.get(renderTileKey(tile.z, tile.x, tile.y));
      if (self === undefined) continue;

      const west  = this._resolveStitchNeighbour(rendered, self, 'west');
      const east  = this._resolveStitchNeighbour(rendered, self, 'east');
      const south = this._resolveStitchNeighbour(rendered, self, 'south');
      const north = this._resolveStitchNeighbour(rendered, self, 'north');
      const part  = (n: StitchNeighbour): string => (n.entry === null ? '-' : n.entry.key);
      const signature = `${part(west)}|${part(east)}|${part(south)}|${part(north)}`;

      const cached = this._stitchCache.get(self.key);
      if (cached !== undefined && cached.signature === signature) {
        tile.stitchedPositions = cached.positions;
        tile.stitchSignature = signature;
        continue;
      }

      const positions = this._stitchTile(self, west, east, south, north);
      tile.stitchedPositions = positions;
      tile.stitchSignature = signature;

      if (this._stitchCache.size >= STITCH_CACHE_CAPACITY) this._stitchCache.clear();
      this._stitchCache.set(self.key, { signature, positions });
    }
  }

  /**
   * Walk up from the adjacent tile coordinates until a *rendered* tile is
   * found (the render set indexes parent-fallback tiles under the ancestor's
   * own coordinates, so the first hit is what is actually drawn across the
   * edge). Returns `null` when nothing renders there.
   */
  private _findRenderedNeighbour(
    rendered: Map<string, TerrainRenderTile>,
    z: number,
    x: number,
    y: number
  ): TerrainRenderTile | null {
    if (y < 0 || y >= 2 ** z) return null; // no neighbour across the poles
    const cols = this._columnsAtLevel(z);
    x = ((x % cols) + cols) % cols;        // antimeridian wrap
    for (; z >= 0; z--, x >>= 1, y >>= 1) {
      const hit = rendered.get(renderTileKey(z, x, y));
      if (hit !== undefined) return hit;
    }
    return null;
  }

  /**
   * Resolve what to stitch against across one edge of `self`.
   *
   * Same-level neighbours (and missing ones) yield `entry: null` — their
   * boundary grid coincides with this tile's by construction. A coarser
   * neighbour yields its cache entry plus the neighbour boundary facing the
   * edge, or `side: null` when the coarse mesh spans beyond the shared line
   * (ancestor-fallback overlap draws the coarse mesh under this tile) and
   * the full neighbour mesh must be scanned instead of just its edge strip.
   */
  private _resolveStitchNeighbour(
    rendered: Map<string, TerrainRenderTile>,
    self: TerrainCacheEntry,
    dir: EdgeDirection
  ): StitchNeighbour {
    const neighbour = this._findRenderedNeighbour(
      rendered,
      self.z,
      self.x + (dir === 'west' ? -1 : dir === 'east' ? 1 : 0),
      self.y + (dir === 'south' ? -1 : dir === 'north' ? 1 : 0)
    );
    if (neighbour === null || neighbour.z >= self.z) return NO_STITCH_NEIGHBOUR;

    const entry = this._meshes.get(renderTileKey(neighbour.z, neighbour.x, neighbour.y));
    if (entry === undefined) return NO_STITCH_NEIGHBOUR; // evicted; nothing to sample

    const b = self.bounds;
    const n = entry.bounds;
    const eps = 1e-6; // degrees; tile bounds are exact powers of two
    let side: EdgeDirection | null = null;
    if (dir === 'west'  && Math.abs(b.west  - n.east)  < eps) side = 'east';
    else if (dir === 'east'  && Math.abs(b.east  - n.west)  < eps) side = 'west';
    else if (dir === 'south' && Math.abs(b.south - n.north) < eps) side = 'north';
    else if (dir === 'north' && Math.abs(b.north - n.south) < eps) side = 'south';
    return { entry, side };
  }

  /**
   * Snap `self`'s edge vertices onto the resolved coarse neighbour surfaces.
   *
   * Returns a render-only copy of the mesh positions with the snapped
   * heights folded in as a radial rescale (the engine folds height into the
   * position magnitude: new|p| = |p| × (1 + h_new / 6378137)), or `null`
   * when nothing was snapped. The cached mesh is never touched —
   * {@link sampleHeight} and picking keep reading the raw geometry.
   *
   * Edge vertices are grouped by quantised position: a grid rim vertex and
   * its skirt copies share lon/lat but not height, and the whole group moves
   * with one radial factor so the skirt keeps its depth below the snapped
   * rim (residual-seam coverage is preserved).
   */
  private _stitchTile(
    self: TerrainCacheEntry,
    west: StitchNeighbour,
    east: StitchNeighbour,
    south: StitchNeighbour,
    north: StitchNeighbour
  ): Float32Array | null {
    if (self.llh === null) self.llh = this._buildVertexLonLatHeight(self.mesh);
    const llh     = self.llh;
    const b       = self.bounds;
    const lonEps  = (b.east - b.west) * EDGE_EPSILON;
    const latEps  = (b.north - b.south) * EDGE_EPSILON;
    const snapped = new Uint8Array(llh.length / 3); // a corner abuts two edges but snaps once
    const out     = new Float32Array(self.mesh.positions); // render-only copy
    let changed   = false;

    const edges = [
      ['west', west], ['east', east], ['south', south], ['north', north],
    ] as const;

    for (const [dir, neighbour] of edges) {
      if (neighbour.entry === null) continue;

      // Group this edge's vertices by representative position (linear scan —
      // an edge holds ~grid-size groups; the tolerance sits far below the
      // grid spacing and far above float32 lon/lat roundoff).
      const groups: { lon: number; lat: number; rim: number; members: number[] }[] = [];
      for (let i = 0; i < llh.length; i += 3) {
        const vi = i / 3;
        if (snapped[vi]) continue;
        const lon = llh[i];
        const lat = llh[i + 1];
        const onEdge =
          (dir === 'west'  && Math.abs(lon - b.west)  <= lonEps) ||
          (dir === 'east'  && Math.abs(lon - b.east)  <= lonEps) ||
          (dir === 'south' && Math.abs(lat - b.south) <= latEps) ||
          (dir === 'north' && Math.abs(lat - b.north) <= latEps);
        if (!onEdge) continue;

        let gi = -1;
        for (let g = 0; g < groups.length; g++) {
          if (Math.abs(lon - groups[g].lon) <= lonEps && Math.abs(lat - groups[g].lat) <= latEps) {
            gi = g;
            break;
          }
        }
        if (gi < 0) {
          groups.push({ lon, lat, rim: vi, members: [] });
          gi = groups.length - 1;
        }
        const group = groups[gi];
        group.members.push(vi);
        if (llh[i + 2] > llh[group.rim * 3 + 2]) group.rim = vi; // rim = surface copy
      }
      if (groups.length === 0) continue;

      // A boundary-abutting neighbour only needs its edge strip scanned, and
      // the sample sits exactly on the neighbour boundary (projection kills
      // the float32 lon/lat roundoff of the duplicated vertices); a coarse
      // mesh spanning the edge line is scanned in full.
      const triangles = neighbour.side === null
        ? null
        : this._edgeTriangles(neighbour.entry)[neighbour.side];

      for (const group of groups) {
        let sampleLon = group.lon;
        let sampleLat = group.lat;
        if (neighbour.side === 'east')       sampleLon = neighbour.entry.bounds.east;
        else if (neighbour.side === 'west')  sampleLon = neighbour.entry.bounds.west;
        else if (neighbour.side === 'north') sampleLat = neighbour.entry.bounds.north;
        else if (neighbour.side === 'south') sampleLat = neighbour.entry.bounds.south;

        const newH = this._sampleMeshHeight(
          neighbour.entry, sampleLon, sampleLat, triangles, STITCH_BARYCENTRIC_EPSILON
        );
        if (newH === null) continue; // no coarse coverage → keep heights (skirt still hides the seam)

        // Radial rescale folding the sampled height in; applied to the whole
        // group so rim + skirt copies translate together.
        const factor = (1 + newH / WGS84_SEMI_MAJOR) / (1 + llh[group.rim * 3 + 2] / WGS84_SEMI_MAJOR);
        if (!Number.isFinite(factor) || factor <= 0) continue;
        for (const vi of group.members) {
          snapped[vi] = 1;
          out[vi * 3]     *= factor;
          out[vi * 3 + 1] *= factor;
          out[vi * 3 + 2] *= factor;
        }
        changed = true;
      }
    }

    return changed ? out : null;
  }

  /**
   * Per-side lists of triangle start indices touching each tile boundary
   * (built lazily per mesh). Boundary stitching only needs these: a sample
   * on the boundary falls into a triangle that has a vertex on that same
   * boundary, which shrinks one edge's scan from the full grid (~8k
   * triangles) to the edge strip (~2·grid-size).
   */
  private _edgeTriangles(entry: TerrainCacheEntry): EdgeTriangleLists {
    if (entry.edgeTris !== null) return entry.edgeTris;
    if (entry.llh === null) entry.llh = this._buildVertexLonLatHeight(entry.mesh);

    const llh    = entry.llh;
    const b      = entry.bounds;
    const lonEps = (b.east - b.west) * EDGE_EPSILON;
    const latEps = (b.north - b.south) * EDGE_EPSILON;
    const lists: EdgeTriangleLists = { west: [], east: [], south: [], north: [] };
    const indices = entry.mesh.indices;

    for (let i = 0; i < indices.length; i += 3) {
      let west = false, east = false, south = false, north = false;
      for (let k = 0; k < 3; k++) {
        const v = indices[i + k] * 3;
        const lon = llh[v];
        const lat = llh[v + 1];
        if (Math.abs(lon - b.west)  <= lonEps) west = true;
        if (Math.abs(lon - b.east)  <= lonEps) east = true;
        if (Math.abs(lat - b.south) <= latEps) south = true;
        if (Math.abs(lat - b.north) <= latEps) north = true;
      }
      if (west)  lists.west.push(i);
      if (east)  lists.east.push(i);
      if (south) lists.south.push(i);
      if (north) lists.north.push(i);
    }
    entry.edgeTris = lists;
    return lists;
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
      // 分析 pin 的瓦片不参与驱逐（见 pinRegion）；全部驻留被 pin 时允许
      // 缓存临时超限——分析数据的完整性优先于缓存上限。
      let evicted = false;
      for (const candidate of this._meshes.keys()) {
        if (this._pinnedTiles.has(candidate)) continue;
        this._meshes.delete(candidate);
        evicted = true;
        break;
      }
      if (!evicted) break;
    }
    this._meshes.set(key, {
      key,
      z, x, y,
      mesh,
      bounds: this._boundsOf(z, x, y),
      llh: null,       // built lazily by sampleHeight / stitching
      edgeTris: null,  // built lazily by the stitch pass
    });
    this._onTileReady?.(z, x, y, mesh);
  }

  /** Barycentric height over the tile's triangles (0 when nothing covers the point). */
  private _interpolateHeight(entry: TerrainCacheEntry, lonDeg: number, latDeg: number): number {
    // SAMPLE_BARYCENTRIC_EPSILON (not the strict -1e-6 default): vertex
    // lon/lat are back-solved from float32 positions, jittering boundary
    // vertices ~4e-6 of a grid cell outside their own edge triangles — the
    // strict tolerance rejects every edge triangle and sampleHeight returns 0
    // on tile-border nodes (consumers then clamp to the base ellipsoid,
    // hundreds of metres off). Same remedy as the stitch path.
    return this._sampleMeshHeight(entry, lonDeg, latDeg, null, SAMPLE_BARYCENTRIC_EPSILON) ?? 0;
  }

  /**
   * Barycentric height interpolation over one tile's triangles; `null` when
   * the point misses every non-degenerate triangle (outside the mesh, or
   * inside a skirt seam). `triangles` optionally restricts the scan to
   * pre-selected triangle start indices (edge lists, see
   * {@link _edgeTriangles}); `epsilon` widens the barycentric tolerance for
   * stitch samples sitting on (or a float32 hair outside) a boundary.
   *
   * Vertex lon/lat/height are derived from the normalised ECEF positions by
   * undoing the engine's polar squash (`y, z × polar ratio`), then height =
   * `(|pos| / |unit direction| - 1) × 6378137`. Skirt triangles project to
   * (near-)degenerate 2D triangles and are rejected by the denominator test.
   */
  private _sampleMeshHeight(
    entry: TerrainCacheEntry,
    lonDeg: number,
    latDeg: number,
    triangles: ArrayLike<number> | null,
    epsilon: number = -1e-6
  ): number | null {
    if (entry.llh === null) entry.llh = this._buildVertexLonLatHeight(entry.mesh);
    const llh     = entry.llh;
    const indices = entry.mesh.indices;
    // 裙边排除（几何级）：含裙边顶点（index ≥ skirtVertexStart）的三角形
    // 是贴边墙，插值出的高度是裙深伪高程（可比真实地形低数百米）。
    // 宽松的重心容差会让贴边查询点落进这些 2D 退化窄条，须直接跳过。
    const skirtStart = entry.mesh.skirtVertexStart ?? indices.length;

    // Work in tile-local coordinates: absolute lon/lat near ±180 have huge
    // magnitude where the squared-area denominators lose precision (a
    // z12 tile spans only ~0.04° while the coordinates sit at ~100°).
    const ox = entry.bounds.west;
    const oy = entry.bounds.south;
    const px = lonDeg - ox;
    const py = latDeg - oy;

    const triCount = triangles === null ? indices.length / 3 : triangles.length;
    for (let t = 0; t < triCount; t++) {
      const i = triangles === null ? t * 3 : triangles[t];
      const ia = indices[i], ib = indices[i + 1], ic = indices[i + 2];
      if (ia >= skirtStart || ib >= skirtStart || ic >= skirtStart) continue; // skirt wall
      const a = ia * 3;
      const b = ib * 3;
      const c = ic * 3;

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
      if (u < epsilon || v < epsilon || u + v > 1 - epsilon) continue;

      const h = llh[a + 2] + u * (llh[b + 2] - llh[a + 2]) + v * (llh[c + 2] - llh[a + 2]);
      return Number.isFinite(h) ? h : null;
    }
    return null; // point fell in a skirt seam / outside the mesh
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

  /**
   * Bounds（度）覆盖的 `level` 级瓦片列表（供 {@link ensureRegion} /
   * {@link pinRegion} 反算范围）。坐标换算与各 tiling scheme 的 bounds
   * 公式互逆：geographic 为 TMS 南起（`geographicTileBounds` 反解），
   * webmercator 为标准 XYZ 北起（`webmercatorTileBounds` 反解）。
   * 越界经纬度 clamp 到该层有效瓦片范围。
   */
  private _tilesInBounds(
    boundsDeg: { west: number; south: number; east: number; north: number },
    level: number
  ): TerrainTileRef[] {
    const cols = this._columnsAtLevel(level);
    const rows = 2 ** level;
    const clampI = (v: number, max: number): number => Math.min(max - 1, Math.max(0, v));

    const x0 = clampI(Math.floor(((boundsDeg.west + 180.0) / 360.0) * cols), cols);
    const x1 = clampI(Math.floor(((boundsDeg.east + 180.0) / 360.0) * cols), cols);

    let y0: number;
    let y1: number;
    if (this.provider.tilingScheme === 'geographic') {
      y0 = clampI(Math.floor(((boundsDeg.south + 90.0) / 180.0) * rows), rows);
      y1 = clampI(Math.floor(((boundsDeg.north + 90.0) / 180.0) * rows), rows);
    } else {
      // Web Mercator 反解：lat → y（北起），越极纬度 clamp 到投影边界。
      const MAX_LAT = 85.05112878;
      const latToY = (lat: number): number => {
        const clamped = Math.min(MAX_LAT, Math.max(-MAX_LAT, lat));
        const rad = (clamped * Math.PI) / 180.0;
        return ((1.0 - Math.asinh(Math.tan(rad)) / Math.PI) / 2.0) * rows;
      };
      y0 = clampI(Math.floor(latToY(boundsDeg.north)), rows); // 北 → 小 y
      y1 = clampI(Math.floor(latToY(boundsDeg.south)), rows);
    }

    const tiles: TerrainTileRef[] = [];
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        tiles.push({ z: level, x, y, key: `${level}/${x}/${y}` });
      }
    }
    return tiles;
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
  /** Lazily built per-boundary triangle lists (crack stitching). */
  edgeTris: EdgeTriangleLists | null;
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

/** @internal Tile edge names (both schemes count y northward). */
type EdgeDirection = 'west' | 'east' | 'south' | 'north';

/** @internal Resolved stitch source for one tile edge. */
interface StitchNeighbour {
  /** Neighbour mesh to sample heights from; `null` = nothing to stitch against. */
  entry: TerrainCacheEntry | null;
  /**
   * Neighbour boundary facing the edge (edge-strip scan), or `null` when the
   * coarse mesh spans past the shared line (full-mesh scan).
   */
  side: EdgeDirection | null;
}

/** @internal Per-tile stitch result, reused until the neighbour signature changes. */
interface StitchCacheEntry {
  signature: string;
  positions: Float32Array | null;
}

/** @internal Per-boundary triangle start indices along a tile's four edges. */
interface EdgeTriangleLists {
  west: number[];
  east: number[];
  south: number[];
  north: number[];
}

/** Tile bounds in degrees. */
export interface TerrainTileBounds {
  west: number;
  south: number;
  east: number;
  north: number;
}

/**
 * Tile selected for rendering (own mesh, or an ancestor's — see
 * `ancestorLevels`), plus the crack-stitching result for this frame's
 * rendered neighbourhood.
 */
export interface TerrainRenderTile {
  z: number;
  x: number;
  y: number;
  /** Levels between this tile and the mesh actually rendered (0 = own mesh). */
  ancestorLevels: number;
  /**
   * Render positions with edge vertices snapped onto the actually-rendered
   * coarser neighbours (crack stitching). `null` = nothing was snapped; use
   * the mesh positions verbatim. Always a copy — the cached mesh in the LRU
   * is never modified (sampleHeight / picking keep the raw geometry).
   */
  stitchedPositions: Float32Array | null;
  /**
   * Signature of the resolved coarse-neighbour set the stitched positions
   * were computed from (`"westKey|eastKey|southKey|northKey"`, `-` = no
   * coarse neighbour on that edge). Consumers must fold this into geometry
   * cache keys so a changed neighbourhood rebuilds the primitive.
   */
  stitchSignature: string;
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

/**
 * Default SSE refine threshold — same "screen pixels per texel" semantics as
 * ImageryLayer (1.0 = texel/pixel parity). Keeps imagery and terrain LOD
 * depth consistent; 2.0 froze selection at the coarse mosaic levels.
 */
const DEFAULT_MAXIMUM_SSE = 1.0;

/** Ancestor levels walked for the parent-mesh fallback. */
const MAX_FALLBACK_LEVELS = 6;

/** Safety cap on the selected set (guards against SSE blowups). */
const MAX_SELECTED_TILES = 512;

/** Cap on the CPU bounding-volume cache. */
const MAX_VOLUMES = 8192;

/**
 * Edge / vertex-group tolerance as a fraction of the tile span. Float32
 * position roundoff puts a vertex ~1e-5 of a span off its exact boundary
 * while the grid spacing is ~1/64 of the span, so anything in between
 * classifies cleanly.
 */
const EDGE_EPSILON = 2e-3;

/**
 * Barycentric tolerance for stitch samples: boundary vertices may sit a
 * float32 hair outside the neighbour's edge triangles, so the stitch scan
 * tolerates points ~0.2 % beyond a triangle (heights extrapolate linearly;
 * the error is negligible). Plain {@link sampleHeight} keeps the strict
 * default of `-1e-6`.
 */
const STITCH_BARYCENTRIC_EPSILON = -2e-3;

/**
 * Barycentric tolerance for plain {@link sampleHeight} queries. Identical
 * rationale as {@link STITCH_BARYCENTRIC_EPSILON}: tile-border vertices
 * back-solved from float32 positions jitter just outside their own edge
 * triangles, so the strict `-1e-6` default made sampleHeight return 0 on
 * border grid nodes (measured on a 65×65 tile: up to 24 nodes per edge).
 * Consumers treat 0 as "no terrain" and clamp to the base ellipsoid —
 * hundreds of metres of error. Heights extrapolate linearly over the tiny
 * tolerance, so interior queries are unaffected.
 */
const SAMPLE_BARYCENTRIC_EPSILON = -2e-3;

/** Cap on the stitch-result cache. */
const STITCH_CACHE_CAPACITY = 512;

/**
 * 分析 pin 允许的最高瓦片层级（pinRegion 前置校验）。粗层 pin 才不会把
 * 渲染视锥需要的精细瓦片挤出缓存池。
 */
export const MAX_PIN_LEVEL = 10;

/** 分析 pin 的瓦片总数上限（pinRegion 前置校验）。 */
const MAX_PINNED_TILES = 256;

/** Shared "nothing to stitch against" resolution. */
const NO_STITCH_NEIGHBOUR: StitchNeighbour = { entry: null, side: null };

/** `"{z}/{x}/{y}"` render-tile key (same format as the mesh cache keys). */
function renderTileKey(z: number, x: number, y: number): string {
  return `${z}/${x}/${y}`;
}

/** Render-set entry before its stitch result is computed for this frame. */
function emptyRenderTile(z: number, x: number, y: number, ancestorLevels: number): TerrainRenderTile {
  return { z, x, y, ancestorLevels, stitchedPositions: null, stitchSignature: '' };
}

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
