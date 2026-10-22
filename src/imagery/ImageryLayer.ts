import { TileKey } from './TileKey';
import { ImageryProvider } from './ImageryProvider';
import { TileTextureCache, TileTextureCacheStats } from './TileTextureCache';
import { analyseTopRows, readBitmapRowMeans, cropTopRows, MIN_CROP_LEAD_ROWS, CanvasSurface } from './TopRowBlackFill';
import { MipmapGenerator } from '../renderer/MipmapGenerator';
import { Camera } from '../scene/Camera';
import { Cartesian3 } from '../math/Cartesian3';
import { Cartographic } from '../math/Cartographic';

/**
 * ImageryLayer - One imagery source composited onto the globe, owning the
 * quadtree tile selection for T1.1.
 *
 * Per update (called once per frame by {@link Globe.update}):
 * 1. Quadtree selection from z = {@link ImageryLayer.ROOT_LEVEL} roots:
 *    `sse = (tileWorldSize / tileSize) * viewportHeight / (distance * 2 tan(fov/2))`;
 *    subdivide while `sse > maximumScreenSpaceError` and below max level.
 * 2. View-frustum culling (6 planes extracted from the viewProj rows) and
 *    simplified horizon culling (tiles beyond the horizon cone are dropped).
 * 3. Request queue for selected-but-unloaded tiles: deduped, sorted by
 *    distance, bounded in-flight fetches; at most `uploadsPerFrame` texture
 *    uploads per frame (see docs/decisions/0001-tile-texture-pool.md).
 * 4. Ready textures land in the {@link TileTextureCache}; while a tile's own
 *    texture is missing the caller renders with an ancestor tile's texture
 *    (upsampled via the UV sub-rect).
 */
export class ImageryLayer {
  /** Show/hide this layer. */
  show: boolean = true;

  /**
   * Layer opacity in [0, 1]. Reserved for future group-uniform compositing;
   * the current opaque G-Buffer path always renders at full opacity.
   */
  alpha: number = 1.0;

  /** The imagery source backing this layer. */
  readonly provider: ImageryProvider;

  /** Root level the quadtree selection starts from. */
  static readonly ROOT_LEVEL = 2;

  /** Aggregate counters for HUD/debug (selected tiles, cache, queue). */
  get stats(): ImageryLayerStats {
    return {
      selectedTiles:    this._selected.length,
      inFlightRequests: this._inflight.size,
      pendingUploads:   this._uploadQueue.length,
      failedTiles:      this._failed.size,
      cache:            this._cache.stats,
    };
  }

  /** The LRU texture pool backing this layer (exposed for debugging). */
  get textureCache(): TileTextureCache {
    return this._cache;
  }

  private readonly _cache: TileTextureCache;
  private readonly _mipmapGenerator = new MipmapGenerator();
  private readonly _maximumScreenSpaceError: number;
  private readonly _maximumLevel: number;
  private readonly _uploadsPerFrame: number;
  private readonly _maxConcurrentRequests: number;

  /** Last selection result with camera distance (for request ordering). */
  private _selected: SelectedTile[] = [];

  /** Keys with a fetch in progress (dedup guard). */
  private _inflight = new Set<string>();

  /** Bitmaps fetched but not yet uploaded to the GPU, upload-ordered. */
  private _uploadQueue: UploadItem[] = [];

  /** Keys whose fetch failed, → failure timestamp (ms); retried after FAILED_TILE_RETRY_MS. */
  private _failed = new Map<string, number>();

  /** CPU-side tile bounding volumes, keyed by tile key. */
  private _volumes = new Map<string, TileVolume>();

  /** Frustum planes (a, b, c, d) × 6, normalised, refreshed per update. */
  private readonly _frustumPlanes = new Float32Array(24);

  private _sampler: GPUSampler | null = null;
  private _destroyed = false;

  constructor(options: ImageryLayerOptions) {
    this.provider                 = options.provider;
    this.show                     = options.show ?? true;
    this.alpha                    = options.alpha ?? 1.0;
    this._maximumScreenSpaceError = options.maximumScreenSpaceError ?? DEFAULT_MAXIMUM_SSE;
    this._maximumLevel            = options.maximumLevel ?? options.provider.maximumLevel;
    this._uploadsPerFrame         = options.uploadsPerFrame ?? UPLOADS_PER_FRAME;
    this._maxConcurrentRequests   = options.maxConcurrentRequests ?? 10;
    this._cache                   = new TileTextureCache({ capacity: options.textureCacheCapacity ?? 200 });
  }

  /**
   * One frame of work: reselect tiles, launch bounded requests, upload at
   * most `uploadsPerFrame` textures.
   */
  update(device: GPUDevice, camera: Camera, viewportHeight: number): void {
    // Drain the deferred texture-destroy queue first: entries have now been
    // untouched for 2 full frames (their referencing command buffers have all
    // been submitted), so destroying them here — before this frame's encode —
    // can no longer invalidate a submit.
    this._cache.onFrameEnd();

    if (this._destroyed) return;

    if (!this.show) {
      // Hidden layers stay warm at the root level: without this a layer that
      // has been hidden since page load owns zero textures, so the moment it
      // becomes visible the entire globe is a hole until its first tiles
      // arrive (a slow provider shows a full-screen hole for seconds).
      this._warmRootTiles();
      this._flushUploads(device);
      return;
    }

    this._selectTiles(camera, viewportHeight);
    this._launchRequests();
    this._flushUploads(device);
  }

  /** The tiles selected by the last {@link update} (unsorted, culled set). */
  get selectedTiles(): readonly TileKey[] {
    return this._selected.map(entry => entry.tile);
  }

  /**
   * Best available texture for a tile: its own, or the nearest loaded
   * ancestor's (upsampling fallback). The returned `ancestorLevels` tells the
   * caller which UV sub-rect of the texture covers this tile
   * (`u = x / 2^ancestorLevels … (x+1) / 2^ancestorLevels`).
   */
  resolveTexture(tile: TileKey): ResolvedTileTexture | null {
    if (this._destroyed) return null;

    // `has()` first so the per-frame polling of not-yet-loaded tiles does
    // not pollute the cache hit/miss statistics; `get()` on a hit refreshes
    // LRU recency (keeping selected tiles resident).
    if (this._cache.has(tile.key)) {
      const own = this._cache.get(tile.key);
      if (own) return { texture: own, ancestorLevels: 0 };
    }

    let ancestor = tile.parent();
    let levels   = 1;
    // Fallback may walk to the quadtree root: prefetching keeps the chain
    // resident, and a far ancestor (however blurry) beats the base globe.
    while (ancestor !== null && levels <= Math.max(MAX_FALLBACK_LEVELS, tile.level - ImageryLayer.ROOT_LEVEL)) {
      if (this._cache.has(ancestor.key)) {
        const texture = this._cache.get(ancestor.key);
        if (texture) return { texture, ancestorLevels: levels };
      }
      ancestor = ancestor.parent();
      levels++;
    }
    return null;
  }

  /** Shared clamp-to-edge trilinear sampler for this layer's tile textures. */
  getSampler(device: GPUDevice): GPUSampler {
    if (this._sampler === null) {
      this._sampler = device.createSampler({
        minFilter:     'linear',
        magFilter:     'linear',
        mipmapFilter:  'linear',
        addressModeU:  'clamp-to-edge',
        addressModeV:  'clamp-to-edge',
        maxAnisotropy: 4,
      });
    }
    return this._sampler;
  }

  /** Release GPU textures and stop all activity. In-flight fetches drain to no-ops. */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._cache.clear();
    for (const item of this._uploadQueue) {
      if ('close' in item.bitmap) item.bitmap.close();
    }
    this._uploadQueue.length = 0;
    this._selected = [];
    this._inflight.clear();
    this._failed.clear();
    this._volumes.clear();
  }

  // ── Private ────────────────────────────────────────────────────────────────

  /**
   * Quadtree selection with view-frustum and horizon culling, iterative
   * (explicit stack) so depth never blows the call stack at high levels.
   *
   * SSE semantics here are "screen pixels covered by one texel":
   * `(worldSize / tileSize) · viewportHeight / (2·distance·tan(fov/2))`.
   * A value of 1.0 = texel-per-pixel parity, so the refine threshold of
   * DEFAULT_MAXIMUM_SSE refines until tiles are sampled at 1:1 density.
   */
  private _selectTiles(camera: Camera, viewportHeight: number): void {
    const planes  = this._extractFrustumPlanes(camera.viewProjectionMatrix.values);
    const camPos  = camera.position;
    const camDir  = Cartesian3.normalize(camPos, new Cartesian3());
    const camDist = Cartesian3.magnitude(camPos);

    // Horizon cone: surface points P stay visible while
    // dot(normalize(P), camDir) >= cos(θh), with cos(θh) = R / |cam|.
    // R is the LOCAL ellipsoid-surface radius at the camera's latitude, not
    // the equatorial 1.0: at mid-latitude the WGS84 surface norm is < 1
    // (0.99914 at lat 30.5°), so with R = 1 a low-altitude camera (|cam|
    // below 1.0 + ε) collapsed the cone to a razor sliver and culled every
    // tile beyond ~one tile width of the camera's nadir — the entire ground
    // vanished from low-altitude oblique views (W4 SensorCam depth defect:
    // ground pixels read the depth clear value / skirt walls).
    Cartesian3.multiplyByScalar(camPos, Cartographic.EARTH_SCALE, _scratchEcefM);
    const camHeight = Cartographic.fromCartesian(_scratchEcefM, _scratchCarto).height;
    const heightEff = Math.max(camHeight, 1e-7);
    const surfaceR = Math.max(camDist - heightEff, 0.9);
    const horizonCos = Math.min(1.0, surfaceR / Math.max(camDist, surfaceR));
    const horizonSin = Math.sqrt(Math.max(0.0, 1.0 - horizonCos * horizonCos));

    const maxLevel   = Math.min(this._maximumLevel, this.provider.maximumLevel);
    const rootLevel  = Math.min(ImageryLayer.ROOT_LEVEL, maxLevel);
    const tanHalfFov = Math.tan(camera.frustum.fov * 0.5);

    const selected: SelectedTile[] = [];
    const stack: TileKey[] = [];

    const rootCount = 2 ** rootLevel;
    for (let y = 0; y < rootCount; y++) {
      for (let x = 0; x < rootCount; x++) {
        stack.push(new TileKey(rootLevel, x, y));
      }
    }

    while (stack.length > 0) {
      const tile   = stack.pop() as TileKey;
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
        (volume.worldSize / this.provider.tileSize) *
        (viewportHeight / (distance * 2.0 * tanHalfFov));

      if (
        sse > this._maximumScreenSpaceError &&
        tile.level < maxLevel &&
        selected.length < MAX_SELECTED_TILES
      ) {
        for (const child of tile.children()) stack.push(child);
      } else {
        selected.push({ tile, distance });
      }
    }

    this._selected = selected;

    // Render-set eviction guard (2026-10-22 flicker fix): a wide view can
    // select more tiles than the texture-cache capacity — without protection
    // every incoming texture evicts a SELECTED tile, whose primitive is then
    // dropped (or terrain draping falls back to the 1×1 placeholder) until
    // refetch lands: an endless evict→refetch→rebuild flicker loop. Selected
    // tiles are budget-exempt; capacity still bounds the prefetch cache.
    this._cache.protectRenderSet(selected.map(entry => entry.tile.key));
  }

  /**
   * Enqueue fetches for selected tiles missing a texture: deduped against
   * in-flight/failed sets, nearest-first, bounded in-flight count.
   *
   * Ancestor tiles of each selection are enqueued too (lowest priority):
   * the upsampling fallback in {@link resolveTexture} walks up the ancestor
   * chain, and those levels are rarely selected on their own once the camera
   * has zoomed past them — without prefetching them the chain breaks and the
   * base globe bleeds through while a leaf loads.
   */
  private _launchRequests(): void {
    const queue: SelectedTile[] = [];
    const seen = new Set<string>();
    for (const entry of this._selected) {
      const key = entry.tile.key;
      if (this._cache.has(key) || this._inflight.has(key)) continue;
      if (this._isFailed(key)) continue;
      if (!seen.has(key)) { seen.add(key); queue.push(entry); }

      // Prefetch the fallback chain. Ancestors request BEFORE leaves (negative
      // penalty): the chain is the display guarantee — once loaded, coarse
      // levels are pinned in the cache and every descendant can fall back to
      // them, so a missing ancestor is a hole while a missing leaf is merely
      // blur. Few and small, one-time cost (parent-first, as CesiumJS does).
      let ancestor = entry.tile.parent();
      while (ancestor !== null && ancestor.level >= ImageryLayer.ROOT_LEVEL) {
        const aKey = ancestor.key;
        if (!this._cache.has(aKey) && !this._inflight.has(aKey)
            && !this._isFailed(aKey) && !seen.has(aKey)) {
          seen.add(aKey);
          queue.push({ tile: ancestor, distance: entry.distance - ANCESTOR_REQUEST_PENALTY });
        }
        ancestor = ancestor.parent();
      }
    }
    queue.sort((a, b) => a.distance - b.distance);

    for (const item of queue) {
      if (this._inflight.size >= this._maxConcurrentRequests) break;

      const key  = item.tile.key;
      const tile = item.tile;
      this._inflight.add(key);

      void this.provider.requestTile(tile).then(bitmap => {
        this._inflight.delete(key);
        if (this._destroyed) {
          if (bitmap && 'close' in bitmap) bitmap.close();
          return;
        }
        if (bitmap !== null) {
          // Bound the backlog: drop (and free) the oldest entries when the
          // camera has outrun the uploader.
          while (this._uploadQueue.length >= MAX_UPLOAD_QUEUE) {
            {
        const dropped = this._uploadQueue.shift();
        if (dropped && 'close' in dropped.bitmap) dropped.bitmap.close();
      }
          }
          this._uploadQueue.push({ key, bitmap, distance: item.distance });
        } else {
          // Record the failure with a timestamp — rate-limited / transient
          // errors must become retryable (a 429 is not "tile does not exist").
          this._failed.set(key, performance.now());
        }
      }).catch((error: unknown) => {
        this._inflight.delete(key);
        this._failed.set(key, performance.now());
        console.warn(`ImageryLayer: request for tile ${key} failed`, error);
      });
    }
  }

  /** A key counts as failed only inside the retry window. */
  private _isFailed(key: string): boolean {
    const failedAt = this._failed.get(key);
    if (failedAt === undefined) return false;
    if (performance.now() - failedAt > FAILED_TILE_RETRY_MS) {
      this._failed.delete(key);
      return false;
    }
    return true;
  }

  /**
   * Hidden-layer warm-up: request the ROOT_LEVEL grid (16 tiles at L2) when
   * missing, reusing the same fetch/upload machinery as the visible path.
   * Bounded to the quadtree roots so the per-frame cost of a hidden layer is
   * negligible; the pinned coarse level (see TileTextureCache) keeps them
   * resident, so any layer flipped visible renders a blurry-but-complete
   * globe instantly instead of a full-screen hole.
   */
  private _warmRootTiles(): void {
    const rootLevel = Math.min(ImageryLayer.ROOT_LEVEL, this._maximumLevel, this.provider.maximumLevel);
    const n = 2 ** rootLevel;
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const key = new TileKey(rootLevel, x, y).key;
        if (this._cache.has(key) || this._inflight.has(key) || this._isFailed(key)) continue;
        if (this._inflight.size >= this._maxConcurrentRequests) return;
        this._inflight.add(key);
        void this.provider.requestTile(new TileKey(rootLevel, x, y)).then(bitmap => {
          this._inflight.delete(key);
          if (this._destroyed) {
            if (bitmap && 'close' in bitmap) bitmap.close();
            return;
          }
          if (bitmap !== null) {
            this._uploadQueue.push({ key, bitmap, distance: 0 });
          } else {
            this._failed.set(key, performance.now());
          }
        }).catch((error: unknown) => {
          // Same bookkeeping as the visible-path fetch failure (no silent
          // swallow): record for TTL retry and log once.
          this._inflight.delete(key);
          this._failed.set(key, performance.now());
          console.warn(`ImageryLayer: warm-up request for tile ${key} failed`, error);
        });
      }
    }
  }

  /** Upload at most `uploadsPerFrame` fetched bitmaps to the GPU per frame. */
  private _flushUploads(device: GPUDevice): void {
    if (this._uploadQueue.length === 0) return;
    this._uploadQueue.sort((a, b) => a.distance - b.distance);

    let uploaded = 0;
    while (uploaded < this._uploadsPerFrame && this._uploadQueue.length > 0) {
      const item   = this._uploadQueue.shift() as UploadItem;
      // 顶行"深蓝黑填充数据"治理：整张黑 → null（不建纹理、不进缓存，
      // resolveTexture 的祖先回退链自动生效）；部分黑 → 裁剪重绘画布。
      const source = this._prepareTopRowTile(item.key, item.bitmap);
      if (source !== null) {
        this._cache.put(item.key, this._createTileTexture(device, item.key, source));
      }
      // 位图统一在此释放（create/canvas 绘制均为同步操作，关闭安全；
      // 整张黑路径也已跳过上传，同样需要释放）。canvas 源无 close，
      // 交给 GC（无 GPU 资源持有）。
      if ('close' in item.bitmap) item.bitmap.close();
      uploaded++;
    }
  }

  /**
   * 顶行瓦片（y === 0）"深蓝黑填充数据"治理（数据事实见
   * TopRowBlackFill.ts 文件头：天地图/高德卫星 85.05°N 以上无影像，
   * 顶行顶部带 10~15 亮度的深蓝黑行；z4 顶行 11/16、z5 顶行 32/32 整张黑）。
   * 返回值：
   * - `null`      整张黑 → 调用方跳过上传，缓存无此 key 时祖先回退链生效；
   * - 裁剪画布    顶部连续黑行 ≥ MIN_CROP_LEAD_ROWS → 有效区域已拉伸到整幅，
   *               语义为 83°~85.05°N 显示 83° 处影像的拉伸（与天地图官网
   *               "极区不可见"观感一致）；
   * - 原位图      正常影像（含检测环境不可用、黑行不足 8 的情况）→ 现状路径。
   * 南极底行（y === 2^z - 1）数据完全干净，不做任何处理。
   * 开销：仅 y === 0 瓦片上传时各执行一次（每级 2^z 张），逐行隔列采样。
   */
  private _prepareTopRowTile(key: string, bitmap: ImageBitmap | CanvasSurface): ImageBitmap | CanvasSurface | null {
    if (TileKey.fromKey(key).y !== 0) return bitmap;

    const rowMeans = readBitmapRowMeans(bitmap);
    if (rowMeans === null) return bitmap; // node / 无 canvas 环境：按现状路径放行

    const { leadBlackRows, allBlack } = analyseTopRows(rowMeans);
    if (allBlack) {
      console.debug(`ImageryLayer: tile ${key} no-data (all black)`);
      return null; // 不建纹理、不进缓存 → 祖先回退链生效（位图由调用方释放）
    }
    if (leadBlackRows >= MIN_CROP_LEAD_ROWS) {
      const cropped = cropTopRows(bitmap, leadBlackRows);
      if (cropped !== null) return cropped; // 画布创建失败（无 canvas）则回退原位图
    }
    return bitmap;
  }

  /**
   * Create a mipmapped rgba8unorm texture from a decoded tile bitmap
   * （或顶行裁剪重绘产生的画布源——canvas 是合法的 CanvasImageSource，
   * copyExternalImageToTexture 与 mip 生成路径完全一致）。
   * 位图的释放由调用方（_flushUploads）统一负责。
   */
  private _createTileTexture(device: GPUDevice, key: string, source: ImageBitmap | CanvasSurface): GPUTexture {
    const width  = source.width;
    const height = source.height;
    const texture = device.createTexture({
      label:  `Imagery.Tile.${key.replace(/\//g, '_')}`,
      size:   { width, height },
      format: 'rgba8unorm',
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      mipLevelCount: MipmapGenerator.requiredMipLevelCount(width, height),
    });
    device.queue.copyExternalImageToTexture(
      { source },
      { texture },
      { width, height }
    );
    this._mipmapGenerator.generateMips(device, texture, 'rgba8unorm');
    return texture;
  }

  /** Bounding sphere + world size for a tile (computed once, then cached). */
  private _getVolume(tile: TileKey): TileVolume {
    const cached = this._volumes.get(tile.key);
    if (cached) return cached;

    const rect      = tile.rectangle();
    const centerLat = (rect.north + rect.south) / 2.0;
    const centerLon = (rect.west + rect.east) / 2.0;
    const center    = surfacePosition(centerLat, centerLon, new Cartesian3());

    let radius = 0.0;
    const corners: Array<[number, number]> = [
      [rect.north, rect.west],
      [rect.north, rect.east],
      [rect.south, rect.west],
      [rect.south, rect.east],
    ];
    const scratch = new Cartesian3();
    for (const [lat, lon] of corners) {
      const d = Cartesian3.distance(center, surfacePosition(lat, lon, scratch));
      if (d > radius) radius = d;
    }
    radius *= 1.05; // margin for the plane / horizon tests

    const width  = (rect.east - rect.west) * Math.cos(centerLat) * WGS84_POLAR_RATIO;
    const height = (rect.north - rect.south) * WGS84_POLAR_RATIO;

    const volume: TileVolume = {
      center,
      radius,
      worldSize: Math.max(width, height),
    };

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

/** @internal Selected tile plus the camera distance used for ordering. */
interface SelectedTile {
  tile: TileKey;
  distance: number;
}

/** @internal Fetched bitmap/canvas awaiting its frame-budgeted GPU upload. */
interface UploadItem {
  key: string;
  /** Canvas 源来自合成层（见 CompositeImageryProvider.requestTile 注）。 */
  bitmap: ImageBitmap | CanvasSurface;
  distance: number;
}

/** @internal CPU-side tile bounds used by the culling / LOD tests. */
interface TileVolume {
  /** Surface point at the tile centre (normalised ECEF, radius ≈ 1). */
  center: Cartesian3;
  /** Bounding-sphere radius around `center` (normalised units). */
  radius: number;
  /** Largest tile edge length (normalised units) for the SSE test. */
  worldSize: number;
}

/** Texture resolved for rendering: own tile texture or an ancestor's. */
export interface ResolvedTileTexture {
  texture: GPUTexture;
  /** Levels between the tile and the texture's level (0 = own texture). */
  ancestorLevels: number;
}

export interface ImageryLayerStats {
  /** Tiles selected by the last update. */
  selectedTiles: number;
  /** Fetches currently in flight. */
  inFlightRequests: number;
  /** Fetched bitmaps waiting for their upload frame budget. */
  pendingUploads: number;
  /** Tiles whose fetch failed (not retried). */
  failedTiles: number;
  /** LRU texture pool counters. */
  cache: TileTextureCacheStats;
}

export interface ImageryLayerOptions {
  /** The imagery source for this layer. */
  provider: ImageryProvider;
  /** Initial visibility. Default: true */
  show?: boolean;
  /** Initial opacity in [0, 1] (reserved; see {@link ImageryLayer.alpha}). Default: 1 */
  alpha?: number;
  /** Refine tiles while their SSE exceeds this. Default: 2 */
  maximumScreenSpaceError?: number;
  /** Highest level to refine to. Default: provider.maximumLevel */
  maximumLevel?: number;
  /** Texture uploads per frame. Default: 10 (see decision 0001) */
  uploadsPerFrame?: number;
  /** Concurrent fetch cap. Default: 10 */
  maxConcurrentRequests?: number;
  /** LRU texture pool capacity. Default: 200 */
  textureCacheCapacity?: number;
}

/** Upload budget per frame (docs/decisions/0001-tile-texture-pool.md). */
const UPLOADS_PER_FRAME = 10;

/** Fetch backlog cap; older bitmaps are closed when the camera outruns upload. */
const MAX_UPLOAD_QUEUE = 256;

/** Ancestor levels walked for the upsampling fallback. */
const MAX_FALLBACK_LEVELS = 6;

/**
 * Sort distance subtracted for ancestor-prefetch requests: fallback-chain
 * ancestors issue BEFORE the leaf tiles they back (a missing ancestor means
 * a hole; a missing leaf only means blur — see _launchRequests).
 */
const ANCESTOR_REQUEST_PENALTY = 1.0;

/** Failed fetches become retryable after this window (rate limits are transient). */
const FAILED_TILE_RETRY_MS = 30_000;

/**
 * Default SSE refine threshold. The SSE formula yields "screen pixels per
 * texel" (see {@link ImageryLayer._selectTiles}), where 1.0 = texel/pixel
 * parity — the pixel-density match point. The old default of 2.0 let every
 * tile refine at well under 2 px/texel even at L2 from 9000 km, freezing
 * the whole globe at L2/L3 mosaic ("blurry broken imagery").
 */
const DEFAULT_MAXIMUM_SSE = 1.0;

/** Safety cap on the selected set (guards against SSE blowups). */
const MAX_SELECTED_TILES = 512;

/** Scratch ECEF (metres) for the geodetic-height read in `_selectTiles`. */
const _scratchEcefM = new Cartesian3();
/** Scratch cartographic result for `_selectTiles` (height normalised). */
const _scratchCarto = new Cartographic();

/** Cap on the CPU bounding-volume cache. */
const MAX_VOLUMES = 8192;

/** WGS84 polar / equatorial ratio (matches EllipsoidGeometry normalisation). */
const WGS84_POLAR_RATIO = 6356752.3142 / 6378137.0;

/**
 * Surface point (normalised ECEF, EllipsoidGeometry convention:
 * x = cos(lat)·cos(lon), y and z scaled by the polar ratio).
 */
function surfacePosition(latitude: number, longitude: number, result: Cartesian3): Cartesian3 {
  // 真 WGS84 归一化映射（与 EllipsoidGeometry.createRegion、fromDegrees ÷a
  // 同式）——瓦片选择中心必须与渲染几何/采集相机同系：旧压扁系中心在 30.5°N
  // 与真椭球差 22.7km，斜视采集视锥下近场瓦片整体被视锥裁掉（W-A 深度残差）。
  const ratio = WGS84_POLAR_RATIO;
  const e2 = 1 - ratio * ratio;
  const cosLat = Math.cos(latitude);
  const sinLat = Math.sin(latitude);
  const n = 1 / Math.sqrt(1 - e2 * sinLat * sinLat);
  result.x = n * cosLat * Math.cos(longitude);
  result.y = n * cosLat * Math.sin(longitude);
  result.z = n * (1 - e2) * sinLat;
  return result;
}

export default ImageryLayer;
