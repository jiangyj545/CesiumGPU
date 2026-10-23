import { Context } from '../renderer/Context';
import { Primitive } from './Primitive';
import { EllipsoidGeometry } from './EllipsoidGeometry';
import { Camera } from './Camera';
import { ImageryLayerCollection } from '../imagery/ImageryLayerCollection';
import { ImageryLayer } from '../imagery/ImageryLayer';
import type { TileTextureCache } from '../imagery/TileTextureCache';
import { TileKey } from '../imagery/TileKey';
import { Color } from '../math/Color';
import { Matrix4 } from '../math/Matrix4';
import { Cartesian3 } from '../math/Cartesian3';
import { TerrainLayer } from '../terrain/TerrainLayer';
import type { TerrainRenderTile } from '../terrain/TerrainLayer';
import type { TerrainProvider, TerrainMeshData } from '../terrain/TerrainProvider';
import { geographicTileBounds, webmercatorTileBounds } from '../terrain/TerrainProvider';

export interface GlobeOptions {
  /** Number of longitude partitions. Default: 128 */
  stackPartitions?: number;
  /** Number of latitude partitions. Default: 64 */
  slicePartitions?: number;
  /** Base ocean/land color (used when no imagery is loaded). */
  baseColor?: Color;
}

/**
 * Globe - Renders the Earth as a WGS84 ellipsoid.
 * Matches Cesium's Globe API.
 */
export class Globe {
  /** Show/hide the globe. */
  show: boolean = true;

  /**
   * Sun-light shading on the globe surface (CesiumJS `globe.enableLighting`
   * semantics). Default `false`: imagery renders at uniform brightness —
   * with the default sun light the back-lit hemisphere falls to
   * `ambient × albedo` (≈5% for ocean), visually indistinguishable from the
   * space background, which reads as holes. When false the base sphere,
   * imagery tiles and terrain tiles all draw through the unlit
   * G-Buffer pipeline (raw albedo, no shading / haze / tone map).
   */
  enableLighting = false;

  /** Base colour (ocean/land) shown when no imagery layer is loaded. */
  baseColor: Color;

  /**
   * Imagery layers composited onto the globe surface. When the collection is
   * empty the globe renders the legacy single-texture path (procedural /
   * world.jpg); otherwise per-tile primitives surface the topmost visible
   * layer's tiles (the legacy texture stays visible underneath as fallback).
   */
  readonly layers = new ImageryLayerCollection();

  private _context: Context;
  private _primitive: Primitive | null = null;
  private _options: Required<GlobeOptions>;
  private _imageryTexture: GPUTexture | null = null;

  // ── Tile rendering state (active when layers exist) ────────────────────────
  private _opaquePipeline: GPURenderPipeline | null = null;
  private _tileSampler: GPUSampler | null = null;
  /** Live tile primitives, keyed by tile key. */
  private _tilePrimitives = new Map<string, TilePrimitiveEntry>();
  /**
   * Polar fill patches (north + south): Web Mercator imagery has no coverage
   * poleward of ±85.0511°, and pole-row tiles from common providers carry
   * near-black filler pixels in their top rows — smearing them poleward
   * paints the caps black with a hard dark circle at 85.05°. CesiumJS fills
   * imagery-less regions with a flat colour; each pole gets one flat
   * ice-white segment per ROOT column, edge-shared with the pole row tiles
   * (same radius) plus a short skirt tucked underneath (see
   * {@link Globe._syncPolarFills}).
   */
  private _polarFills: { north: TilePrimitiveEntry[] | null; south: TilePrimitiveEntry[] | null } = {
    north: null,
    south: null,
  };
  /** Flat 1×1 ice-white texture shared by all polar fill segments. */
  private _iceTexture: GPUTexture | null = null;
  /** ROOT column count the current fill segments were built for. */
  private _polarFillColumns = -1;
  /** CPU geometry cache keyed by `"<tileKey>#<ancestorLevels>"`. */
  private _geometryCache = new Map<string, CachedTileGeometry>();
  /**
   * Disposable 1×1 texture swapped onto a tile primitive right before its
   * destruction: `Primitive.destroy()` destroys the texture the primitive
   * currently holds, which for tile primitives is a texture shared via the
   * layer's LRU cache (ancestor-fallback textures are referenced by several
   * primitives). Without the swap, destroying one tile's primitive would
   * destroy a texture other primitives still bind → "Destroyed texture used
   * in a submit" → the whole command buffer is dropped (black frame).
   */
  private _detachTexture: GPUTexture | null = null;

  // ── Terrain rendering state (active when a provider is attached) ──────────
  /** Terrain streaming layer (null until setTerrainProvider is called). */
  private _terrainLayer: TerrainLayer | null = null;
  /** Live terrain tile primitives, keyed by rendered tile key. */
  private _terrainPrimitives = new Map<string, TilePrimitiveEntry>();
  /** CPU geometry cache for terrain meshes, keyed `terrain:<key>#<levels>`. */
  private _terrainGeometryCache = new Map<string, CachedTileGeometry>();
  /** 1×1 earthy texture bound to terrain tiles with no resolvable imagery. */
  private _terrainFallbackTexture: GPUTexture | null = null;
  /**
   * Imagery flat tiles to SUPPRESS in {@link drawTiles}: keys whose mercator
   * rectangle is FULLY covered by the terrain tile union (tile-exact — a
   * mesh-free gap keeps its imagery, no black holes). Rebuilt by
   * {@link _syncTerrainPrimitives} only when the terrain/imagery key
   * signature or draping readiness changes (see postmortem in drawTiles).
   */
  private _imagerySkipKeys = new Set<string>();
  /** Last signature the skip-set was computed for ('' = never). */
  private _imagerySkipSig = '';
  /**
   * Imagery cache that holds the live terrain pins (released on clear —
   * pins outliving the terrain set would leak capacity exemption).
   */
  private _terrainPinCache: TileTextureCache | null = null;
  /**
   * Retired tile primitives awaiting their in-flight grace window before
   * GPU destruction (see `_destroyTilePrimitive`: `Primitive.destroy()`
   * frees vertex/index buffers IMMEDIATELY, which invalidates still-
   * executing submits → per-tile black frames during upgrade streams).
   */
  private _retiredPrimitives: Array<{ primitive: Primitive; frame: number }> = [];
  private _frameIndex = 0;

  /** Scale factor: globe radius in normalised units = 1.0 */
  static readonly RADIUS = 1.0;

  constructor(context: Context, options: GlobeOptions = {}) {
    this._context = context;
    this._options = {
      stackPartitions: options.stackPartitions ?? 128,
      slicePartitions: options.slicePartitions ?? 64,
      baseColor:       options.baseColor ?? new Color(0.1, 0.35, 0.6, 1.0),
    };
    this.baseColor = this._options.baseColor;
  }

  /**
   * Build the globe geometry and upload it to the GPU.
   * Call once during scene initialization.
   */
  initialize(): Primitive {
    const geom = new EllipsoidGeometry({
      radii:           new Cartesian3(6378137.0, 6378137.0, 6356752.3142),
      stackPartitions: this._options.stackPartitions,
      slicePartitions: this._options.slicePartitions,
    });

    const { vertices, indices } = geom.createInterleavedBuffer();

    this._primitive = new Primitive(this._context, {
      vertices,
      indices,
      color:       this.baseColor,
      translucent: false,
    });

    // Scale model matrix: the geometry is built on a unit sphere, scale by EARTH_SCALE
    // so the globe fills the normalised space (radius = 1).
    // Since EllipsoidGeometry normalises radii to 1.0, the model matrix is identity.
    Matrix4.identity(this._primitive.modelMatrix);

    return this._primitive;
  }

  get primitive(): Primitive | null { return this._primitive; }

  /**
   * Terrain streaming layer, or `null` while no provider is attached (the
   * flat-ellipsoid path is used). Consumed by terrain-aware picking via
   * `Camera.pickGlobe` / `Measure.clampToTerrain` / draw handlers.
   */
  get terrainLayer(): TerrainLayer | null { return this._terrainLayer; }

  /**
   * True while the imagery quadtree owns the globe surface (any layer in the
   * collection, visible or hidden — hidden layers warm their roots too, so
   * the pinned L2 roots always cover the planet). In that state GBufferPass
   * skips the base sphere entirely (CesiumJS architecture: the tile quadtree
   * IS the globe surface; a second sphere underneath would only compete in
   * the depth buffer and force lift offsets at every seam). The base sphere
   * still draws for the legacy single-texture / procedural path (no layers).
   */
  get surfaceIsQuadtree(): boolean {
    return this.layers.length > 0;
  }

  /**
   * Attach (or detach with `null`) a terrain provider. While attached the
   * globe streams terrain meshes each frame and renders displaced tile
   * geometry instead of the flat per-tile regions.
   */
  setTerrainProvider(provider: TerrainProvider | null): void {
    this._clearTerrainPrimitives();
    this._terrainGeometryCache.clear();
    if (this._terrainLayer !== null) {
      this._terrainLayer.destroy();
      this._terrainLayer = null;
    }
    if (!provider) return;

    this._terrainLayer = new TerrainLayer({
      provider,
      // A tile mesh arrived: drop its cached geometry (it may have been built
      // from a parent-mesh fallback) so the primitive rebuilds from real
      // terrain on the next sync.
      onTileReady: (z, x, y) => {
        const prefix = `terrain:${z}/${x}/${y}`;
        for (const cacheKey of this._terrainGeometryCache.keys()) {
          if (cacheKey.startsWith(prefix)) this._terrainGeometryCache.delete(cacheKey);
        }
        const stale = this._terrainPrimitives.get(`${z}/${x}/${y}`);
        if (stale !== undefined) {
          this._destroyTilePrimitive(stale);
          this._terrainPrimitives.delete(`${z}/${x}/${y}`);
        }
      },
    });
  }

  /**
   * Load a texture from an HTMLImageElement and apply it to the globe.
   */
  loadImageryTexture(image: HTMLImageElement): void {
    if (!this._primitive) return;
    const device = this._context.device;

    createImageBitmap(image).then(bitmap => {
      const texture = device.createTexture({
        label:  'Globe.Imagery',
        size:   { width: bitmap.width, height: bitmap.height },
        format: 'rgba8unorm',
        usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        mipLevelCount: 1,
      });

      device.queue.copyExternalImageToTexture(
        { source: bitmap },
        { texture },
        { width: bitmap.width, height: bitmap.height }
      );

      // TODO: generate mipmaps if needed

      // WebGPU requires mipmapFilter:'linear' whenever maxAnisotropy > 1.
      // mipLevelCount is 1 so mipmapFilter has no visual effect, but the
      // validator enforces the constraint and rejects 'nearest' (the default).
      const sampler = device.createSampler({
        minFilter:    'linear',
        magFilter:    'linear',
        mipmapFilter: 'linear',
        addressModeU: 'repeat',
        addressModeV: 'clamp-to-edge',
        maxAnisotropy: 4,
      });

      this._imageryTexture?.destroy();
      this._imageryTexture = texture;
      this._primitive?.setTexture(texture, sampler);
      bitmap.close();
    }).catch(err => console.error('Globe: failed to create texture bitmap:', err));
  }

  /**
   * Generate and apply a procedural Earth-like texture entirely in the browser.
   * Uses a 2D canvas with trig-based continent approximation, ocean gradients
   * and polar ice caps. No network request needed.
   */
  generateProceduralTexture(): void {
    if (!this._primitive) return;

    const W = 1024, H = 512;
    const canvas = document.createElement('canvas');
    canvas.width  = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d')!;
    const imgData = ctx.createImageData(W, H);
    const data = imgData.data;

    for (let py = 0; py < H; py++) {
      // lat in [-π/2, π/2], lon in [-π, π]
      const lat = (0.5 - (py + 0.5) / H) * Math.PI; // py=0 → +π/2 (north pole), py=H-1 → -π/2 (south pole)
      const absLat = Math.abs(lat);

      for (let px = 0; px < W; px++) {
        const lon = ((px + 0.5) / W - 0.5) * 2 * Math.PI;
        const i = (py * W + px) * 4;

        // ── Polar ice caps ────────────────────────────────────────────────
        if (absLat > 1.22) { // > ~70°
          const blend = Math.min(1, (absLat - 1.22) / 0.35);
          data[i]   = Math.round(210 + 45 * blend);
          data[i+1] = Math.round(225 + 30 * blend);
          data[i+2] = Math.round(235 + 20 * blend);
          data[i+3] = 255;
          continue;
        }

        // ── Continent mask (trig-based approximation) ─────────────────────
        // Weighted sum of harmonics: positive → land, negative → ocean
        const n =
          0.4 * Math.sin(lon * 1.5 - 0.4) * Math.cos(lat * 2.2) +
          0.3 * Math.sin(lon * 3.0 + 1.0) * Math.cos(lat * 1.8 + 0.3) +
          0.2 * Math.sin(lon * 2.5 + 2.5) * Math.cos(lat * 3.5 - 0.5) +
          0.15 * Math.cos(lon * 4.0 - 1.0) * Math.sin(lat * 2.0 + 1.0);
        const land = n > 0.08;

        // ── Ocean: dark blue → lighter blue by latitude ───────────────────
        if (!land) {
          const depthShade = 0.6 + 0.4 * Math.cos(lat);
          data[i]   = Math.round(15  + 25  * depthShade);
          data[i+1] = Math.round(60  + 55  * depthShade);
          data[i+2] = Math.round(130 + 70  * depthShade);
          data[i+3] = 255;
        } else {
          // ── Land: green lowlands → brownish highlands ─────────────────
          const elevation = Math.max(0, n - 0.08);
          const highFactor = Math.min(1, elevation * 5);
          const r = Math.round(40  + 100 * highFactor + 20 * (1 - highFactor));
          const g = Math.round(110 - 30  * highFactor + 30 * (1 - highFactor));
          const b = Math.round(30  + 20  * highFactor);
          // Snow line on high terrain
          if (elevation > 0.18) {
            const snowBlend = Math.min(1, (elevation - 0.18) / 0.10);
            data[i]   = Math.round(r + (220 - r) * snowBlend);
            data[i+1] = Math.round(g + (225 - g) * snowBlend);
            data[i+2] = Math.round(b + (230 - b) * snowBlend);
          } else {
            data[i]   = r;
            data[i+1] = g;
            data[i+2] = b;
          }
          data[i+3] = 255;
        }
      }
    }

    ctx.putImageData(imgData, 0, 0);

    createImageBitmap(canvas).then(bitmap => {
      const device = this._context.device;
      const texture = device.createTexture({
        label:         'Globe.ProceduralEarth',
        size:          { width: bitmap.width, height: bitmap.height },
        format:        'rgba8unorm',
        usage:         GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        mipLevelCount: 1,
      });
      device.queue.copyExternalImageToTexture(
        { source: bitmap },
        { texture },
        { width: bitmap.width, height: bitmap.height }
      );
      bitmap.close();

      const sampler = device.createSampler({
        minFilter:    'linear',
        magFilter:    'linear',
        addressModeU: 'repeat',
        addressModeV: 'clamp-to-edge',
      });

      this._imageryTexture?.destroy();
      this._imageryTexture = texture;
      this._primitive?.setTexture(texture, sampler);
    }).catch(console.error);
  }

  /**
   * Load imagery from a URL.
   */
  async loadImageryFromUrl(url: string): Promise<void> {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    await new Promise<void>((resolve, reject) => {
      img.onload  = () => resolve();
      img.onerror = () => reject(new Error(`Failed to load imagery: ${url}`));
      img.src     = url;
    });
    this.loadImageryTexture(img);
  }

  // ── Imagery tile rendering (T1.1) ───────────────────────────────────────────

  /**
   * Give the globe the shared opaque pipeline so tile primitives can build
   * their bind groups. Called once by {@link Scene.initialize}.
   */
  setOpaquePipeline(pipeline: GPURenderPipeline): void {
    this._opaquePipeline = pipeline;
  }

  /**
   * Per-frame tile update: run the active layer's quadtree selection, then
   * synchronise this globe's per-tile primitives with the selected set.
   * No-op while `layers` is empty (legacy single-texture path keeps drawing).
   */
  update(camera: Camera): void {
    if (!this._opaquePipeline) {
      this._clearTilePrimitives();
      this._clearTerrainPrimitives();
      // Keep advancing the grace window even without a pipeline — otherwise
      // resources retired here would never be destroyed.
      this._frameIndex++;
      this._flushRetiredPrimitives();
      return;
    }

    // Retired-primitive grace window: free GPU resources only once the
    // submits referencing them have completed (2 frames — same policy as
    // TileTextureCache's deferred texture destruction).
    this._frameIndex++;
    this._flushRetiredPrimitives();

    // P3 多图层合成：单可见层 = 原 activeLayer 原样（零改动路径）；≥2 可见
    // 层 = 集合的合成层（CompositeImageryProvider 逐瓦片叠加）。
    const layer = this.layers.renderLayer;
    if (layer === null) {
      this._clearTilePrimitives();
    } else {
      layer.update(this._context.device, camera, this._context.drawingBufferHeight);
      this._syncTilePrimitives(layer);
      this._syncPolarFills(layer);
    }

    // Hidden layers still tick `update` — ImageryLayer uses the hidden branch
    // only to warm its root tiles (16 fetches, one-time), so flipping a layer
    // visible never starts from an empty cache (no full-screen hole).
    // Composite members are skipped: the composite layer drives their
    // providers directly（合成层已取瓦片，避免成员层再跑一套四叉树双份拉取）.
    const members = new Set(this.layers.compositedMembers());
    for (const hidden of this.layers) {
      if (hidden !== layer && !members.has(hidden)) {
        hidden.update(this._context.device, camera, this._context.drawingBufferHeight);
      }
    }

    // Terrain streaming runs alongside imagery: meshes arrive asynchronously
    // and displace the per-tile geometry once resident (see drawTiles).
    if (this._terrainLayer !== null) {
      this._terrainLayer.update(camera, this._context.drawingBufferHeight);
      this._syncTerrainPrimitives(layer);
    } else {
      this._clearTerrainPrimitives();
    }
  }

  /**
   * Draw all live tile primitives in the G-Buffer pass. Terrain meshes and
   * flat imagery tiles are drawn TOGETHER and the depth test arbitrates:
   * elevated terrain sticks out above the (slightly lifted, see
   * TILE_RADIUS_SCALE) flat imagery, while regions the terrain service does
   * not cover (geographic terrain only spans its dataset bounds — e.g. the
   * 7777 Hubei set) keep their flat imagery instead of a black hole.
   *
   * 2026-10-23 postmortem: a "single-layer" variant that skipped flat tiles
   * inside the terrain coverage union-rect was reverted the same day — the
   * union-rect of the live meshes contains mesh-free gaps (L-shaped tile
   * distributions, failed/unloaded tiles), and skipping imagery there opened
   * BLACK HOLES, strictly worse than the flicker it aimed to fix. The flicker
   * root cause (depth alternation between the two shells at far distances)
   * needs a tile-exact coverage test + cached skip-set, not a bounding rect.
   * That v2 now lives here: {@link _imagerySkipKeys} (tile-exact, gap-safe,
   * signature-cached) suppresses the flat shell only where terrain FULLY
   * covers it.
   */
  drawTiles(pass: GPURenderPassEncoder, cameraBindGroup: GPUBindGroup): void {
    for (const entry of this._terrainPrimitives.values()) {
      entry.primitive.drawOpaque(pass, cameraBindGroup);
    }
    const skip = this._imagerySkipKeys;
    for (const [imageryKey, entry] of this._tilePrimitives) {
      if (skip.has(imageryKey)) continue;
      entry.primitive.drawOpaque(pass, cameraBindGroup);
    }
    // Polar fills ride along with the flat imagery path.
    for (const which of ['north', 'south'] as const) {
      const entries = this._polarFills[which];
      if (entries === null) continue;
      for (const entry of entries) entry.primitive.drawOpaque(pass, cameraBindGroup);
    }
  }

  /** Number of live tile primitives (for spike HUDs). */
  get tilePrimitiveCount(): number {
    return this._tilePrimitives.size;
  }

  /**
   * Snapshot of the live tile primitives (for GPU picking: the pick pass
   * re-draws the same set the G-Buffer pass draws). Allocated per call —
   * only invoked when a pick is pending, never per rendered frame.
   *
   * MUST stay same-source with {@link Globe.drawTiles}: both sets (terrain
   * meshes + flat imagery + polar fills) are drawn together, depth-arbitrated.
   */
  get tilePrimitives(): Primitive[] {
    const out: Primitive[] = [];
    for (const entry of this._terrainPrimitives.values()) out.push(entry.primitive);
    for (const entry of this._tilePrimitives.values()) out.push(entry.primitive);
    for (const which of ['north', 'south'] as const) {
      const entries = this._polarFills[which];
      if (entries === null) continue;
      for (const entry of entries) out.push(entry.primitive);
    }
    return out;
  }

  /** Reconcile tile primitives with the layer's selected set. */
  private _syncTilePrimitives(layer: ImageryLayer): void {
    const selected = layer.selectedTiles;
    const keep     = new Set<string>();
    for (const tile of selected) keep.add(tile.key);

    // Remove primitives for tiles no longer selected (frees GPU buffers).
    for (const [key, entry] of this._tilePrimitives) {
      if (!keep.has(key)) {
        this._destroyTilePrimitive(entry);
        this._tilePrimitives.delete(key);
      }
    }

    // Create / retexture primitives for selected tiles.
    for (const tile of selected) {
      const resolved = layer.resolveTexture(tile);
      if (resolved === null) {
        // Nothing loaded (e.g. the tile's texture was just evicted and no
        // ancestor is resident): drop any stale primitive so it cannot keep
        // binding the evicted texture past its deferred-destroy deadline.
        const stale = this._tilePrimitives.get(tile.key);
        if (stale !== undefined) {
          this._destroyTilePrimitive(stale);
          this._tilePrimitives.delete(tile.key);
        }
        continue; // base globe shows until something loads
      }

      let entry = this._tilePrimitives.get(tile.key);
      if (entry !== undefined && entry.texture !== resolved.texture) {
        // Bound texture changed (e.g. own texture replaced the parent's) →
        // rebuild with geometry whose UVs match the new texture's sub-rect.
        this._destroyTilePrimitive(entry);
        entry = undefined;
        this._tilePrimitives.delete(tile.key);
      }
      if (entry === undefined) {
        entry = this._createTilePrimitive(tile, resolved.texture, resolved.ancestorLevels);
        this._tilePrimitives.set(tile.key, entry);
      }
    }
  }

  /**
   * Keep the two polar fill patches in step with the active layer (CesiumJS
   * imagery-gap semantics — regions without imagery coverage render a flat
   * fill colour instead of smeared edge pixels).
   *
   * Pole-row tiles from common providers (Tianditu, Gaode) carry near-black
   * filler rows at their top edge; the historical pole-stretch approach
   * smeared those rows over the caps, painting them black with a hard dark
   * circle at 85.05°. Each pole is instead covered by `n` flat ice-white
   * segments — one per ROOT column — whose edge row sits exactly ON the
   * 85.0511° latitude circle at the same radius as the bordering pole-row
   * tiles (edge-shared, no lift → no depth conflict), plus one skirt row
   * slightly below the surface that hides any T-junction slivers where a
   * refined tile's denser boundary vertices meet the segment edge.
   *
   * The fills are texture-independent (flat colour), so unlike the retired
   * cap patches they exist from the first frame — no load-order seams.
   */
  private _syncPolarFills(layer: ImageryLayer): void {
    const rootLevel = Math.min(ImageryLayer.ROOT_LEVEL, layer.provider.maximumLevel);
    const n = 2 ** rootLevel;
    if (n === this._polarFillColumns && this._polarFills.north !== null) return;
    this._clearPolarFills();
    this._polarFillColumns = n;

    const ice = this._getIceTexture();
    for (const which of ['north', 'south'] as const) {
      const isNorth = which === 'north';
      const entries: TilePrimitiveEntry[] = [];
      for (let x = 0; x < n; x++) {
        const west = -Math.PI + (x / n) * 2 * Math.PI;
        const east = -Math.PI + ((x + 1) / n) * 2 * Math.PI;
        const geometry = this._buildPolarFillGeometry(west, east, isNorth);
        const primitive = new Primitive(this._context, {
          vertices:    geometry.vertices,
          indices:     geometry.indices,
          translucent: false,
        });
        Matrix4.identity(primitive.modelMatrix);
        primitive.setTexture(ice, this._getTileSampler());
        primitive.buildOpaqueBindGroup(this._opaquePipeline as GPURenderPipeline);
        entries.push({ primitive, texture: ice });
      }
      this._polarFills[which] = entries;
    }
  }

  /**
   * Flat grid over one ROOT-column slice of a polar cap (85.0511° → pole),
   * same vertex layout as {@link EllipsoidGeometry.createRegion} ([pos(3),
   * normal(3), uv(2)], x on the unit sphere with y/z squashed by the WGS84
   * polar ratio). Rows run pole → 85.0511° at r = 1 (edge-shared with the
   * pole-row tiles), then one skirt row past the boundary at r slightly
   * below 1 (hidden under the neighbouring tiles, closes T-junction gaps).
   */
  private _buildPolarFillGeometry(
    west: number,
    east: number,
    isNorth: boolean
  ): { vertices: Float32Array; indices: Uint32Array } {
    const stacks = POLAR_FILL_STACKS;
    const slices = POLAR_FILL_SLICES;
    const pole   = isNorth ? Math.PI / 2 : -Math.PI / 2;
    const edge   = isNorth ? MERCATOR_EDGE : -MERCATOR_EDGE;
    // Skirt latitude: one step further past the boundary (toward the equator).
    const skirtLat = edge + (edge - pole) / stacks * 0.5;

    const rows = stacks + 2; // main surface rows (stacks+1) + skirt row
    const vertexCount = rows * (slices + 1);
    const vertices = new Float32Array(vertexCount * 8);
    const ratio = WGS84_POLAR_RATIO;
    const ratioSq = ratio * ratio;

    let vi = 0;
    for (let stack = 0; stack <= stacks + 1; stack++) {
      const isSkirt = stack === stacks + 1;
      const lat = isSkirt
        ? skirtLat
        : pole + (edge - pole) * (stack / stacks);
      const r = isSkirt ? POLAR_FILL_SKIRT_RADIUS : 1.0;
      const cosLat = Math.cos(lat);
      const sinLat = Math.sin(lat);
      // 真椭球映射（与 EllipsoidGeometry.createRegion 同式同源，见彼处注释）——
      // 极区填充与瓦片在 ±85.0511° 共边，两映射必须一致才无缝。
      const nLat = 1 / Math.sqrt(1 - (1 - ratio * ratio) * sinLat * sinLat);
      const zn = nLat * ratio * ratio;
      for (let slice = 0; slice <= slices; slice++) {
        const lon = west + (east - west) * (slice / slices);
        const nx = cosLat * Math.cos(lon);
        const ny = cosLat * Math.sin(lon);
        const nz = sinLat;

        vertices[vi++] = r * nLat * nx;
        vertices[vi++] = r * nLat * ny;
        vertices[vi++] = r * zn * nz;

        const ny2 = ny / ratioSq;
        const nz2 = nz / ratioSq;
        const nLen = Math.sqrt(nx * nx + ny2 * ny2 + nz2 * nz2);
        vertices[vi++] = nx / nLen;
        vertices[vi++] = ny2 / nLen;
        vertices[vi++] = nz2 / nLen;

        vertices[vi++] = 0; // uv unused (flat 1×1 texture)
        vertices[vi++] = 0;
      }
    }

    const indices = new Uint32Array((stacks + 1) * slices * 6);
    let idx = 0;
    const stride = slices + 1;
    for (let stack = 0; stack <= stacks; stack++) {
      for (let slice = 0; slice < slices; slice++) {
        const a = stack * stride + slice;
        const b = a + stride;
        const c = b + 1;
        const d = a + 1;
        // Rows run pole → 85.05°: in the north that matches createRegion's
        // north → south row order (front faces outward), but in the south it
        // is reversed — the same index pattern would face inward there and be
        // backface-culled entirely (south cap rendered as a hole). Flip the
        // winding for the south cap so both caps face outward.
        if (isNorth) {
          indices[idx++] = a;
          indices[idx++] = b;
          indices[idx++] = d;
          indices[idx++] = b;
          indices[idx++] = c;
          indices[idx++] = d;
        } else {
          indices[idx++] = a;
          indices[idx++] = d;
          indices[idx++] = b;
          indices[idx++] = b;
          indices[idx++] = d;
          indices[idx++] = c;
        }
      }
    }
    return { vertices, indices };
  }

  /** Lazily created shared 1×1 ice-white texture (polar fill colour). */
  private _getIceTexture(): GPUTexture {
    if (this._iceTexture !== null) return this._iceTexture;
    const texture = this._context.device.createTexture({
      label: 'Globe.PolarIceFill',
      size:  { width: 1, height: 1 },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    // Ice white (matches the retired procedural base-globe polar ice tone).
    this._context.device.queue.writeTexture(
      { texture },
      new Uint8Array([238, 243, 248, 255]),
      {},
      { width: 1, height: 1 },
    );
    this._iceTexture = texture;
    return texture;
  }

  /** Destroy both polar fill patches (the shared ice texture stays). */
  private _clearPolarFills(): void {
    for (const which of ['north', 'south'] as const) {
      const entries = this._polarFills[which];
      if (entries === null) continue;
      for (const entry of entries) this._destroyTilePrimitive(entry);
      this._polarFills[which] = null;
    }
    this._polarFillColumns = -1;
  }

  /** Build one per-tile primitive (region geometry + tile texture). */
  private _createTilePrimitive(
    tile: TileKey,
    texture: GPUTexture,
    ancestorLevels: number
  ): TilePrimitiveEntry {
    const geometry = this._getTileGeometry(tile, ancestorLevels, this._edgeLiftFor(tile));
    const primitive = new Primitive(this._context, {
      vertices:    geometry.vertices,
      indices:     geometry.indices,
      translucent: false,
      origin:      tileRTCEnabled() ? tileRTCOrigin(tile) : undefined,
    });
    // Geometry is generated directly in normalised ECEF (EllipsoidGeometry
    // convention, Earth radius = 1) — identity model matrix, like the base globe.
    Matrix4.identity(primitive.modelMatrix);
    primitive.setTexture(texture, this._getTileSampler());
    primitive.buildOpaqueBindGroup(this._opaquePipeline as GPURenderPipeline);
    return { primitive, texture };
  }

  /**
   * Edge lifts pinning this tile's north/south boundary rows to a live
   * neighbour tile's lift when that neighbour is coarser (its larger lift
   * would otherwise show as a radial step on the shared latitude circle —
   * the seam continuity rule).
   *
   * Neighbour rows only: with the pole-stretch architecture the top/bottom
   * rows are ordinary tiles (no separate cap mesh at the poles), so no
   * special case is needed there.
   */
  private _edgeLiftFor(tile: TileKey): { north?: number; south?: number } | undefined {
    const own = tileRadiusScale(tile.level);
    const rows = 2 ** tile.level;
    const edge: { north?: number; south?: number } = {};
    // Neighbour rows only (see doc comment).
    for (const key of this._tilePrimitives.keys()) {
      const parts = key.split('/');
      const lvl = parseInt(parts[0], 10);
      const y = parseInt(parts[2], 10);
      const lift = tileRadiusScale(lvl);
      if (lift <= own) continue;
      const vStart = y * rows / 2 ** lvl;
      const vEnd = (y + 1) * rows / 2 ** lvl;
      if (tile.y > 0 && vEnd >= tile.y && vStart < tile.y) {
        edge.north = Math.max(edge.north ?? 0, lift);
      }
      if (tile.y < rows - 1 && vEnd >= tile.y + 1 && vStart < tile.y + 1) {
        edge.south = Math.max(edge.south ?? 0, lift);
      }
    }
    return edge.north !== undefined || edge.south !== undefined ? edge : undefined;
  }

  /** Cached (or newly generated) interleaved geometry for a tile. */
  private _getTileGeometry(
    tile: TileKey,
    ancestorLevels: number,
    edgeLift?: { north?: number; south?: number }
  ): CachedTileGeometry {
    // skirtDepth 编入 key：裙边改变几何内容，旧缓存条目（无裙边）不可复用。
    const cacheKey = `${tile.key}#${ancestorLevels}#${edgeLift?.north ?? '-'}#${edgeLift?.south ?? '-'}#${TILE_SKIRT_DEPTH}`;
    const cached = this._geometryCache.get(cacheKey);
    if (cached) return cached;

    const rect = tile.rectangle();
    // When rendering from an ancestor texture, remap the region's UVs into
    // that texture's sub-rect (upsampling fallback, no shader change needed).
    let uvRect: { u0: number; v0: number; u1: number; v1: number } | undefined;
    if (ancestorLevels > 0) {
      const n = 2 ** ancestorLevels;
      const inv = 1.0 / n;
      const u0 = (tile.x % n) * inv;
      const v0 = (tile.y % n) * inv;
      uvRect = { u0, v0, u1: u0 + inv, v1: v0 + inv };
    }

    const geometry = EllipsoidGeometry.createRegion(
      rect.west,
      rect.south,
      rect.east,
      rect.north,
      TILE_PARTITIONS,
      // Mercator tiles distribute pixel rows in Mercator-y; a linear v
      // offsets every row's content northward (≈7.7° mid-latitude) —
      // the "map squeezed into the Arctic" defect.
      {
        uvRect,
        radiusScale: tileRadiusScale(tile.level),
        edgeLift,
        mercatorV: true,
        skirtDepth: TILE_SKIRT_DEPTH,
        origin: tileRTCEnabled() ? tileRTCOrigin(tile) : undefined,
      }
    );

    if (this._geometryCache.size >= TILE_GEOMETRY_CACHE_CAPACITY) {
      this._geometryCache.clear();
    }
    this._geometryCache.set(cacheKey, geometry);
    return geometry;
  }

  // ── Terrain tile rendering (Wave2-A) ────────────────────────────────────────

  /**
   * Synchronise terrain primitives with the layer's render set (selected
   * tiles backed by an own or ancestor mesh). Mirrors the imagery sync.
   */
  private _syncTerrainPrimitives(layer: ImageryLayer | null): void {
    const terrainLayer = this._terrainLayer;
    if (terrainLayer === null) return;

    const keep = new Set<string>();
    for (const tile of terrainLayer.renderTiles) keep.add(`${tile.z}/${tile.x}/${tile.y}`);

    for (const [key, entry] of this._terrainPrimitives) {
      if (!keep.has(key)) {
        this._unpinTerrainTexture(entry);
        this._destroyTilePrimitive(entry);
        this._terrainPrimitives.delete(key);
      }
    }

    for (const tile of terrainLayer.renderTiles) {
      const key = `${tile.z}/${tile.x}/${tile.y}`;
      const mesh = terrainLayer.getMesh(tile.z, tile.x, tile.y);
      if (mesh === null) continue; // evicted between selection and sync

      // Textures: the webmercator terrain grid aligns 1:1 with imagery
      // TileKeys, so tile textures are reused there. Geographic terrain
      // (EPSG:4326 TMS) resolves the covering WEB-MERCATOR imagery tile via
      // its centre and maps vertex UVs through the mercator projection —
      // a geographic tile is half a mercator tile wide at the same z, so it
      // straddles at most two columns; clamp-to-edge sampling keeps the
      // straddle error to the shared edge row (CesiumJS draping semantics,
      // simplified to a single reference tile).
      let texture: GPUTexture;
      let textureLevels = -1; // -1 → fallback texture, full [0,1] uv
      let refMerc: { z: number; x: number; y: number } | undefined;
      let pinKey: string | undefined;
      if (layer !== null && terrainLayer.provider.tilingScheme === 'webmercator') {
        const resolved = layer.resolveTexture(new TileKey(tile.z, tile.x, tile.y));
        if (resolved !== null) {
          texture = resolved.texture;
          textureLevels = resolved.ancestorLevels;
          pinKey = `${tile.z - resolved.ancestorLevels}/${tile.x >> resolved.ancestorLevels}/${tile.y >> resolved.ancestorLevels}`;
        } else {
          texture = this._getTerrainFallbackTexture();
        }
      } else if (layer !== null) {
        const rect = geographicTileBounds(tile.z, tile.x, tile.y);
        const lonC = (rect.west + rect.east) / 2;
        const latC = (rect.north + rect.south) / 2;
        const ref = mercatorDrapingKey(tile.z, lonC, latC);
        const resolved = layer.resolveTexture(ref);
        if (resolved !== null) {
          texture = resolved.texture;
          textureLevels = resolved.ancestorLevels;
          refMerc = {
            z: ref.level - resolved.ancestorLevels,
            x: ref.x >> resolved.ancestorLevels,
            y: ref.y >> resolved.ancestorLevels,
          };
          pinKey = `${refMerc.z}/${refMerc.x}/${refMerc.y}`;
        } else {
          texture = this._getTerrainFallbackTexture();
        }
      } else {
        texture = this._getTerrainFallbackTexture();
      }

      // The geometry cache key folds in the stitch signature: when the
      // rendered neighbourhood changes (e.g. a coarse fallback neighbour is
      // replaced by finer tiles) the stitched edge positions change and the
      // primitive must rebuild even though its texture did not.
      const geometryKey = this._terrainGeometryKey(tile, textureLevels, refMerc);
      const entry = this._terrainPrimitives.get(key);
      if (entry !== undefined && (entry.texture !== texture || entry.geometryKey !== geometryKey)) {
        // Bound texture changed → rebuild with matching uv sub-rect geometry;
        // stitch signature changed → rebuild with the re-stitched edges.
        if (entry.pinKey !== undefined && entry.pinKey !== pinKey) this._unpinTerrainTexture(entry);
        this._destroyTilePrimitive(entry);
        this._terrainPrimitives.delete(key);
      }
      if (!this._terrainPrimitives.has(key)) {
        if (pinKey !== undefined && layer !== null) {
          layer.textureCache.pin(pinKey); // bind-time pin (flicker fix)
          this._terrainPinCache = layer.textureCache;
        }
        this._terrainPrimitives.set(
          key,
          this._createTerrainPrimitive(tile, mesh, texture, textureLevels, geometryKey, refMerc, pinKey)
        );
      }
    }

    // Tile-exact imagery suppression (flicker fix v2, 2026-10-23): inside
    // regions FULLY covered by terrain tiles the flat imagery shell is
    // skipped — at far view distances the two shells (tens of metres apart)
    // degenerate the depth test into frame-by-frame alternation. Tile-exact
    // means a mesh-free gap KEEPS its imagery (the union-rect variant was
    // reverted for opening black holes); draping must also be fully resident
    // or nothing is skipped. Signature-cached: rebuilt only when the terrain
    // or imagery key set / readiness changes, never per stable frame.
    const fallbackTex = this._terrainFallbackTexture;
    let drapingReady = this._terrainPrimitives.size > 0;
    if (drapingReady) {
      for (const entry of this._terrainPrimitives.values()) {
        if (entry.texture === fallbackTex || entry.texture.width <= 1) { drapingReady = false; break; }
      }
    }
    const sig = (drapingReady
      ? [...this._terrainPrimitives.keys()].sort().join('|')
      : '-') + '#' + [...this._tilePrimitives.keys()].sort().join('|');
    if (sig !== this._imagerySkipSig) {
      this._imagerySkipSig = sig;
      this._imagerySkipKeys = drapingReady
        ? computeImagerySkipKeys(
            new Set(this._terrainPrimitives.keys()),
            this._tilePrimitives.keys())
        : new Set();
    }
  }

  /**
   * Release a terrain entry's eviction pin (see TileTextureCache._pins).
   * Pins are released on primitive destroy AND on rebuild-to-a-different-key;
   * keeping a pin whose texture changed would leak capacity exemption.
   */
  private _unpinTerrainTexture(entry: TilePrimitiveEntry): void {
    if (entry.pinKey === undefined) return;
    if (this._terrainPinCache !== null) this._terrainPinCache.unpin(entry.pinKey);
    entry.pinKey = undefined;
  }

  /** Geometry cache key for a terrain render tile (includes the stitch signature). */
  private _terrainGeometryKey(
    tile: TerrainRenderTile,
    textureLevels: number,
    refMerc?: { z: number; x: number; y: number }
  ): string {
    return `terrain:${tile.z}/${tile.x}/${tile.y}#${tile.ancestorLevels}#${textureLevels}` +
      `#${refMerc ? `m${refMerc.z}/${refMerc.x}/${refMerc.y}` : 'l'}#${tile.stitchSignature}`;
  }

  /** Build one terrain primitive (displaced mesh geometry + tile texture). */
  private _createTerrainPrimitive(
    tile: TerrainRenderTile,
    mesh: TerrainMeshData,
    texture: GPUTexture,
    textureLevels: number,
    geometryKey: string,
    refMerc?: { z: number; x: number; y: number },
    pinKey?: string
  ): TilePrimitiveEntry {
    // uvRect（线性子矩形）仅用于 webmercator 同网格路径；geographic 地形走
    // refMerc 的逐顶点 mercator 投影 UV（见 _getTerrainGeometry）。
    const uvRect = refMerc === undefined && textureLevels > 0
      ? this._tileUvRect(tile.x, tile.y, textureLevels)
      : undefined;
    const geometry = this._getTerrainGeometry(tile, mesh, uvRect, textureLevels, refMerc);
    const primitive = new Primitive(this._context, {
      vertices:    geometry.vertices,
      indices:     geometry.indices,
      translucent: false,
    });
    // Mesh positions are already normalised ECEF at true height — identity
    // model matrix and NO TILE_RADIUS_SCALE lift (that is imagery-only).
    Matrix4.identity(primitive.modelMatrix);
    primitive.setTexture(texture, this._getTileSampler());
    primitive.buildOpaqueBindGroup(this._opaquePipeline as GPURenderPipeline);
    return { primitive, texture, geometryKey, pinKey };
  }

  /**
   * Cached (or newly generated) interleaved geometry for a terrain mesh.
   *
   * Positions come from the layer's crack-stitched snapshot when present
   * (edge vertices snapped onto the actually-rendered coarse neighbours —
   * see `TerrainLayer._stitchRenderSet`), otherwise verbatim from the mesh
   * (normalised ECEF, height folded in). Normals are accumulated from face
   * crosses (robust across skirts); zero-length sums fall back to the radial
   * direction. UVs map each vertex's lon/lat linearly across the tile bounds
   * — the same mapping `EllipsoidGeometry.createRegion` uses, so tile
   * textures line up. The cache key includes the stitch signature, so a
   * changed rendered neighbourhood rebuilds instead of reusing stale edges.
   */
  private _getTerrainGeometry(
    tile: TerrainRenderTile,
    mesh: TerrainMeshData,
    uvRect: { u0: number; v0: number; u1: number; v1: number } | undefined,
    textureLevels: number,
    refMerc?: { z: number; x: number; y: number }
  ): CachedTileGeometry {
    const cacheKey = this._terrainGeometryKey(tile, textureLevels, refMerc);
    const cached = this._terrainGeometryCache.get(cacheKey);
    if (cached) return cached;

    const terrainLayer = this._terrainLayer as TerrainLayer;
    const rect = terrainLayer.provider.tilingScheme === 'geographic'
      ? geographicTileBounds(tile.z, tile.x, tile.y)
      : webmercatorTileBounds(tile.z, tile.x, tile.y);

    const positions = tile.stitchedPositions ?? mesh.positions;
    const indices   = Uint32Array.from(mesh.indices);
    const vertexCount = positions.length / 3;
    const vertices    = new Float32Array(vertexCount * 8); // [pos(3), normal(3), uv(2)]
    const normals     = new Float32Array(vertexCount * 3);
    const rad         = 180.0 / Math.PI;

    // Pass 1: positions + uvs; face-normal accumulation.
    for (let i = 0; i < indices.length; i += 3) {
      const a = indices[i] * 3;
      const b = indices[i + 1] * 3;
      const c = indices[i + 2] * 3;
      const e0x = positions[b]     - positions[a];
      const e0y = positions[b + 1] - positions[a + 1];
      const e0z = positions[b + 2] - positions[a + 2];
      const e1x = positions[c]     - positions[a];
      const e1y = positions[c + 1] - positions[a + 1];
      const e1z = positions[c + 2] - positions[a + 2];
      const nx = e0y * e1z - e0z * e1y;
      const ny = e0z * e1x - e0x * e1z;
      const nz = e0x * e1y - e0y * e1x;
      normals[a] += nx; normals[a + 1] += ny; normals[a + 2] += nz;
      normals[b] += nx; normals[b + 1] += ny; normals[b + 2] += nz;
      normals[c] += nx; normals[c + 1] += ny; normals[c + 2] += nz;
    }

    const u0 = uvRect ? uvRect.u0 : 0;
    const v0 = uvRect ? uvRect.v0 : 0;
    const du = uvRect ? uvRect.u1 - uvRect.u0 : 1;
    const dv = uvRect ? uvRect.v1 - uvRect.v0 : 1;
    const lonSpan = rect.east - rect.west;
    const latSpan = rect.north - rect.south;

    for (let i = 0; i < vertexCount; i++) {
      // True-ellipsoid inverse (matches ecefFromLonLatHeight, 2026-10-22):
      // x = n·nx·s, y = n·ny·s, z = n(1−e²)·nz·s → n cancels, closed form
      // tanφ = z / ((1−e²)·hypot(x, y)). Legacy `y/ρ, z/ρ` undo belonged to
      // the retired squash convention and mis-projected draping UVs.
      const x = positions[i * 3];
      const y = positions[i * 3 + 1];
      const z = positions[i * 3 + 2];
      const e2 = 1.0 - WGS84_POLAR_RATIO * WGS84_POLAR_RATIO;

      const lon = Math.atan2(y, x) * rad;
      const lat = Math.atan2(z / (1.0 - e2), Math.hypot(x, y)) * rad;
      // Geographic terrain draping: vertex lon/lat → web-mercator imagery tile
      // UV. v is NON-linear in latitude (mercator y); u is linear in lon.
      let u: number;
      let v: number;
      if (refMerc !== undefined) {
        const mn = 2 ** refMerc.z;
        u = ((lon + 180) / 360) * mn - refMerc.x;
        const m = (1 - Math.log(Math.tan(Math.PI / 4 + lat / rad / 2)) / Math.PI) / 2;
        v = m * mn - refMerc.y;
        // A geographic tile straddles at most two mercator columns/rows —
        // clamp straddling edge vertices to the reference tile's borders
        // (sampler is clamp-to-edge; error confined to the shared edge row).
        if (u < 0) u = 0; else if (u > 1) u = 1;
        if (v < 0) v = 0; else if (v > 1) v = 1;
      } else {
        u = u0 + du * (lon - rect.west) / lonSpan;
        v = v0 + dv * (rect.north - lat) / latSpan; // row 0 = north edge
      }

      let nx = normals[i * 3];
      let ny = normals[i * 3 + 1];
      let nz = normals[i * 3 + 2];
      const nLen = Math.hypot(nx, ny, nz);
      if (nLen < 1e-12) { nx = x; ny = y; nz = z; } // degenerate → radial
      const invLen = 1.0 / (nLen < 1e-12 ? Math.hypot(x, y, z) : nLen);

      const vi = i * 8;
      vertices[vi]     = x;
      vertices[vi + 1] = positions[i * 3 + 1];
      vertices[vi + 2] = positions[i * 3 + 2];
      vertices[vi + 3] = nx * invLen;
      vertices[vi + 4] = ny * invLen;
      vertices[vi + 5] = nz * invLen;
      vertices[vi + 6] = u;
      vertices[vi + 7] = v;
    }

    const geometry: CachedTileGeometry = { vertices, indices };
    if (this._terrainGeometryCache.size >= TILE_GEOMETRY_CACHE_CAPACITY) {
      this._terrainGeometryCache.clear();
    }
    this._terrainGeometryCache.set(cacheKey, geometry);
    return geometry;
  }

  /** UV sub-rect of an ancestor tile texture covering this tile (imagery path convention). */
  private _tileUvRect(x: number, y: number, ancestorLevels: number) {
    const n   = 2 ** ancestorLevels;
    const inv = 1.0 / n;
    const u0  = (x % n) * inv;
    const v0  = (y % n) * inv;
    return { u0, v0, u1: u0 + inv, v1: v0 + inv };
  }

  /** Lazily created 1×1 earthy texture for terrain without resolvable imagery. */
  private _getTerrainFallbackTexture(): GPUTexture {
    if (this._terrainFallbackTexture === null) {
      const device = this._context.device;
      const texture = device.createTexture({
        label: 'Globe.TerrainFallback',
        size:  { width: 1, height: 1 },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture(
        { texture },
        new Uint8Array([137, 124, 104, 255]),
        { bytesPerRow: 4, rowsPerImage: 1 },
        { width: 1, height: 1 }
      );
      this._terrainFallbackTexture = texture;
    }
    return this._terrainFallbackTexture;
  }

  /** Destroy all terrain primitives (used when no provider is attached). */
  private _clearTerrainPrimitives(): void {
    if (this._terrainPrimitives.size === 0) return;
    for (const entry of this._terrainPrimitives.values()) {
      this._unpinTerrainTexture(entry);
      this._destroyTilePrimitive(entry);
    }
    this._terrainPrimitives.clear();
  }

  /** Lazily created shared tile sampler (driven by the active layer). */
  private _getTileSampler(): GPUSampler {
    if (this._tileSampler === null) {
      this._tileSampler = this._context.device.createSampler({
        minFilter:     'linear',
        magFilter:     'linear',
        mipmapFilter:  'linear',
        addressModeU:  'clamp-to-edge',
        addressModeV:  'clamp-to-edge',
        maxAnisotropy: 4,
      });
    }
    return this._tileSampler;
  }

  /**
   * Destroy a tile primitive without touching the shared LRU-cached texture
   * it renders from: the primitive is first re-bound to a disposable
   * sentinel texture, so `Primitive.destroy()` destroys the sentinel instead
   * (see {@link Globe._detachTexture}).
   */
  private _destroyTilePrimitive(entry: TilePrimitiveEntry): void {
    if (this._detachTexture === null) {
      this._detachTexture = this._context.device.createTexture({
        label: 'Globe.TilePrimitiveDetach',
        size:  { width: 1, height: 1 },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING,
      });
    }
    // Detach the shared tile texture first: the eventual destroy() destroys
    // whatever texture is bound — the sentinel, never the shared imagery
    // texture still referenced by the cache / other primitives.
    entry.primitive.setTexture(this._detachTexture);
    // DEFERRED destroy (flicker fix): Primitive.destroy() frees vertex/index
    // buffers immediately, but the previous frame's render pass may still be
    // executing on the GPU and referencing them — destroying now invalidates
    // that submit and the tile's screen region turns black for a frame (the
    // clear↔black flicker during tile upgrade streams). Retire instead;
    // _flushRetiredPrimitives destroys after the 2-frame grace window.
    this._retiredPrimitives.push({ primitive: entry.primitive, frame: this._frameIndex });
  }

  /** Destroy retired primitives whose in-flight grace window has elapsed. */
  private _flushRetiredPrimitives(): void {
    if (this._retiredPrimitives.length === 0) return;
    const deadline = this._frameIndex - RETIRE_GRACE_FRAMES;
    let write = 0;
    for (let read = 0; read < this._retiredPrimitives.length; read++) {
      const item = this._retiredPrimitives[read];
      if (item.frame <= deadline) {
        item.primitive.destroy();          // consumes the detach sentinel
        this._detachTexture = null;        // sentinel destroyed with it — recreated on next retire
      } else {
        this._retiredPrimitives[write++] = item;
      }
    }
    this._retiredPrimitives.length = write;
  }

  /** Destroy all tile primitives including the polar fills (no layer active). */
  private _clearTilePrimitives(): void {
    if (this._tilePrimitives.size === 0) {
      this._clearPolarFills();
      return;
    }
    for (const entry of this._tilePrimitives.values()) {
      this._destroyTilePrimitive(entry);
    }
    this._tilePrimitives.clear();
    this._clearPolarFills();
  }

  destroy(): void {
    this._clearTilePrimitives();
    this._clearTerrainPrimitives();
    // Teardown: no further frames will reference the retired resources —
    // destroy them all immediately (bypasses the grace window).
    for (const item of this._retiredPrimitives) item.primitive.destroy();
    this._retiredPrimitives.length = 0;
    this._terrainGeometryCache.clear();
    if (this._terrainLayer !== null) {
      this._terrainLayer.destroy();
      this._terrainLayer = null;
    }
    this._terrainFallbackTexture?.destroy();
    this._terrainFallbackTexture = null;
    this._geometryCache.clear();
    this._detachTexture?.destroy();
    this._detachTexture = null;
    this._iceTexture?.destroy();
    this._iceTexture = null;
    this.layers.destroy();
    this._primitive?.destroy();
    this._imageryTexture?.destroy();
  }
}

/** @internal Live tile primitive plus the texture currently bound to it. */
interface TilePrimitiveEntry {
  primitive: Primitive;
  /** Texture reference used to detect bind-group-invalidating changes. */
  texture: GPUTexture;
  /**
   * Terrain only: geometry cache key the primitive was built from (detects
   * stitch-signature changes that require a rebuild).
   */
  geometryKey?: string;
  /**
   * Terrain only: imagery cache key pinned against LRU eviction for the
   * lifetime of this entry (see TileTextureCache.pin — evicting a bound
   * texture triggers the clear↔blur rebuild flicker).
   */
  pinKey?: string;
}

/** @internal Cached interleaved geometry for a tile (+ UV variant). */
interface CachedTileGeometry {
  vertices: Float32Array;
  indices: Uint32Array;
}

/** Subdivisions per axis for tile geometry (a tile is small on screen). */
const TILE_PARTITIONS = 8;

/**
 * 影像瓦片裙边深度（归一化半径比例）。跨 level 相邻瓦片共边处，粗层边缘是
 * 弦线、细层中点顶点浮在弧上，最大楔形缝深 ≈0.0048（TILE_PARTITIONS=8、
 * L2 边段 11.25°）——0.005 留出裕量，裙边面从外侧遮住缝隙
 * （CesiumJS skirt 同款语义，见 EllipsoidGeometry.createRegion）。
 */
const TILE_SKIRT_DEPTH = 0.005;

/**
 * Tile geometry sits ON the true ellipsoid surface (CesiumJS architecture:
 * the quadtree IS the globe surface — there is no separate base sphere
 * competing in the depth buffer beneath the tiles). While imagery layers
 * are active the base globe is not drawn at all (see GBufferPass), so no
 * lift is needed anywhere — a uniform 1.0 keeps every seam pair at the same
 * radius by construction. The historical 1.008/1.02 tiered lifts (and the
 * edgeLift pinning that level mixing required) exist only for the retired
 * "tiles above a drawn base globe" architecture; `edgeLift` stays a no-op.
 */
function tileRadiusScale(_level: number): number {
  // 影像瓦片径向下沉（2026-10-23 拉远闪烁修复）：地形 mesh 与平面影像瓦片
  // 同帧绘制、深度裁决——影像壳沉到地表下 300m（见 IMAGERY_LIFT 注释），
  // 对最低地形块仍有 ≥317m 稳定隔离，任何视距的深度测试都可分辨；
  // h<-300m 的水面/河床由影像覆盖（语义正确），其余区域地形 draping
  // 自带纹理。旧 +50m 抬升对低地只有 ~33m 隔离——拉远视距深度混叠=闪。
  return 1.0 + IMAGERY_LIFT;
}

/**
 * Imagery tile radial offset from the ellipsoid — NEGATIVE (300 m BELOW the
 * surface, 2026-10-23 far-view flicker fix). History: +50 m kept the flat
 * shell just ABOVE zero-height terrain so imagery covered water/plains, but
 * that left only ~33 m of separation against low terrain — at far view
 * distances the depth test cannot resolve tens of metres and the two shells
 * alternate frame by frame ("one rippled layer, one flat layer" flicker).
 * Sinking the shell −300 m instead gives ≥317 m of separation against the
 * lowest Hubei tiles (h≈17 m), stable at every view distance; regions with
 * terrain below −300 m (river gorges / lake beds) read as water and are
 * legitimately covered by imagery; terrain draping carries the texture
 * everywhere else. Polar fills share this offset (same function).
 */
const IMAGERY_LIFT = -300 / 6378137;

/**
 * tile 的 RTC 几何中心：tile 矩形中心点的真 WGS84 归一化 ECEF（与
 * ImageryLayer.surfacePosition / Cartesian3.fromDegrees÷a 同式，f64）。
 * 顶点相对它存储（f32 量化亚毫米），深度走 encoded-camera 精确路径
 * （掠射瓦片量化残差根除，2026-10-18）。
 */
/**
 * 瓦片 RTC 开关：**缺省关**（2026-10-21 回归修复）。b40bea4 起缺省开的主渲
 * 染路径在真机 GPU 上把全球瓦片 mesh 变换错位——视锥/深度只放过 L2 顶行
 * （85°N~66.5°N ≈ 北极圈），画面即"只渲染北极圈"（?waRtcTiles=0 对照 A/B
 * 实证：关=完整地球，开=北极圈；纹理/上传/合成管线全部无辜）。深度收益主
 * 体本是城市墙掠射（瓦片为陡角入射几乎不放大），关闭瓦片 RTC 不损
 * verify-depth 结论；矩阵 bug 修复后可恢复缺省开。?waRtcTiles=1 显式开。
 */
function tileRTCEnabled(): boolean {
  if (typeof location === 'undefined') return false;
  return new URLSearchParams(location.search).get('waRtcTiles') === '1';
}

/**
 * Tile-exact imagery suppression set (flicker fix v2, 2026-10-23): imagery
 * mercator keys whose rectangle is FULLY covered by the union of terrain
 * geographic tiles. The depth test between the terrain shell and the flat
 * imagery shell degenerates into frame-by-frame alternation at far view
 * distances ("one rippled layer, one flat layer" flicker); suppressing the
 * flat shell where terrain FULLY covers it removes the competitor while any
 * mesh-free gap keeps its imagery — tile-exactness is what the reverted
 * union-rect variant lacked (it opened black holes).
 *
 * Coverage test: an 8×8 sample grid over the imagery rectangle; a sample is
 * covered when the geographic tile containing it (at ANY terrain zoom level
 * present in the set — meshes mix levels during refinement) is in
 * `terrainKeys`. Miss-direction-safe: an undetected sub-1/8-tile gap merely
 * keeps the flat shell drawn (status-quo flicker), never a hole.
 *
 * Grids: terrain = geographicTileBounds (cols 2^(z+1), rows 2^z, south-up
 * linear latitudes); imagery = webmercatorTileBounds (cols 2^z, mercator
 * latitudes) — both axis-aligned rectangles in lon/lat space.
 */
/**
 * Mercator imagery TileKey covering a geographic terrain tile's centre —
 * the draping texture reference. The query level is CLAMPED to the imagery
 * quadtree root ({@link ImageryLayer.ROOT_LEVEL}): terrain tiles coarser
 * than the root (z0/z1 far-view fallbacks) have no imagery counterpart at
 * their own level, so an unclamped lookup is structurally null and the
 * draping primitive sticks on the 1×1 fallback forever (permanent
 * flesh-coloured block, fourth flicker case 2026-10-23). A z0/z1 terrain
 * tile resolves to the z2 tile containing its centre; the straddle across
 * two z2 tiles degrades to clamp-to-edge sampling like any other
 * cross-column tile.
 */
export function mercatorDrapingKey(z: number, lonDeg: number, latDeg: number): TileKey {
  const rz = Math.max(z, ImageryLayer.ROOT_LEVEL);
  const n = 2 ** rz;
  const x = Math.min(n - 1, Math.max(0, Math.floor(((lonDeg + 180) / 360) * n)));
  const mercN = (1 - Math.log(Math.tan(Math.PI / 4 + (latDeg * Math.PI / 180) / 2)) / Math.PI) / 2;
  const y = Math.min(n - 1, Math.max(0, Math.floor(mercN * n)));
  return new TileKey(rz, x, y);
}

export function computeImagerySkipKeys(
  terrainKeys: ReadonlySet<string>,
  imageryKeys: Iterable<string>
): Set<string> {
  const skip = new Set<string>();
  if (terrainKeys.size === 0) return skip;
  const levels: number[] = [];
  for (const key of terrainKeys) {
    const slash = key.indexOf('/');
    if (slash <= 0) continue;
    const z = Number(key.slice(0, slash));
    if (Number.isFinite(z) && !levels.includes(z)) levels.push(z);
  }
  if (levels.length === 0) return skip;
  const SAMPLES = 8;
  for (const ikey of imageryKeys) {
    const m = ikey.match(/^(\d+)\/(\d+)\/(\d+)$/);
    if (m === null) continue;
    const iz = +m[1];
    const ix = +m[2];
    const iy = +m[3];
    const n = 2 ** iz;
    const west  = (ix / n) * 360 - 180;
    const east  = ((ix + 1) / n) * 360 - 180;
    const north = (Math.atan(Math.sinh(Math.PI * (1 - (2 * iy) / n))) * 180) / Math.PI;
    const south = (Math.atan(Math.sinh(Math.PI * (1 - (2 * (iy + 1)) / n))) * 180) / Math.PI;
    if (!Number.isFinite(north) || !Number.isFinite(south)) continue;
    let covered = true;
    scan:
    for (let sy = 0; sy < SAMPLES && covered; sy++) {
      const lat = south + ((sy + 0.5) / SAMPLES) * (north - south);
      for (let sx = 0; sx < SAMPLES; sx++) {
        const lon = west + ((sx + 0.5) / SAMPLES) * (east - west);
        let hit = false;
        for (const zl of levels) {
          const cols = 2 ** (zl + 1);
          const rows = 2 ** zl;
          const tx = Math.min(cols - 1, Math.max(0, Math.floor(((lon + 180) / 360) * cols)));
          const ty = Math.min(rows - 1, Math.max(0, Math.floor(((lat + 90) / 180) * rows)));
          if (terrainKeys.has(zl + '/' + tx + '/' + ty)) { hit = true; break; }
        }
        if (!hit) { covered = false; break scan; }
      }
    }
    if (covered) skip.add(ikey);
  }
  return skip;
}

function tileRTCOrigin(tile: TileKey): [number, number, number] {
  const rect = tile.rectangle();
  const lat = (rect.north + rect.south) / 2;
  const lon = (rect.west + rect.east) / 2;
  const ratio = WGS84_POLAR_RATIO;
  const e2 = 1 - ratio * ratio;
  const sinLat = Math.sin(lat);
  const n = 1 / Math.sqrt(1 - e2 * sinLat * sinLat);
  const cosLat = Math.cos(lat);
  return [
    n * cosLat * Math.cos(lon),
    n * cosLat * Math.sin(lon),
    n * (1 - e2) * sinLat,
  ];
}

/** WGS84 polar / equatorial ratio (matches EllipsoidGeometry normalisation). */
const WGS84_POLAR_RATIO = 6356752.3142 / 6378137.0;

/** Web Mercator domain edge: ±atan(sinh π) ≈ ±85.0511° (CesiumJS semantics). */
const MERCATOR_EDGE = Math.atan(Math.sinh(Math.PI));

/** Latitude × longitude subdivisions of one polar fill segment. */
const POLAR_FILL_STACKS = 8;
const POLAR_FILL_SLICES = 8;

/**
 * Skirt radius for the polar fill boundary row — a hair below the surface
 * (≈0.05% ≈ 3.2 km) so the row is depth-occluded by the bordering pole-row
 * tiles and only becomes visible through T-junction slivers it must fill.
 */
const POLAR_FILL_SKIRT_RADIUS = 0.9995;

/** Max cached tile geometries (CPU arrays only; ~4 KB each at 8×8). */
const TILE_GEOMETRY_CACHE_CAPACITY = 1024;

/**
 * Frames a retired tile primitive waits before its GPU resources are
 * destroyed (same policy as TileTextureCache.DEFERRED_DESTROY_FRAMES):
 * submits referencing its buffers must have completed — destroying earlier
 * invalidates the submit and the tile's region flashes black for a frame.
 */
const RETIRE_GRACE_FRAMES = 2;

export default Globe;
