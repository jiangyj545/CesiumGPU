import { Context } from '../renderer/Context';
import { Primitive } from './Primitive';
import { EllipsoidGeometry } from './EllipsoidGeometry';
import { Camera } from './Camera';
import { ImageryLayerCollection } from '../imagery/ImageryLayerCollection';
import { ImageryLayer } from '../imagery/ImageryLayer';
import { TileKey } from '../imagery/TileKey';
import { Color } from '../math/Color';
import { CesiumMath } from '../math/CesiumMath';
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
   * imagery tiles, polar caps and terrain tiles all draw through the unlit
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
   * Polar cap patches (north + south): Web Mercator imagery stops at
   * ±85.05°, so without these the caps show the base colour — reads as a
   * "hole" whenever the poles rotate into view. Each patch re-samples the
   * top/bottom row of the ROOT level grid stretched over the cap.
   */
  private _polarCaps: { north: TilePrimitiveEntry | null; south: TilePrimitiveEntry | null } = {
    north: null,
    south: null,
  };
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
      return;
    }

    const layer = this.layers.activeLayer;
    if (layer === null) {
      this._clearTilePrimitives();
    } else {
      layer.update(this._context.device, camera, this._context.drawingBufferHeight);
      this._syncTilePrimitives(layer);
      this._syncPolarCaps(layer);
    }

    // Hidden layers still tick `update` — ImageryLayer uses the hidden branch
    // only to warm its root tiles (16 fetches, one-time), so flipping a layer
    // visible never starts from an empty cache (no full-screen hole).
    for (const hidden of this.layers) {
      if (hidden !== layer) hidden.update(this._context.device, camera, this._context.drawingBufferHeight);
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
   * Draw all live tile primitives in the G-Buffer pass. While terrain
   * primitives exist they replace the flat imagery tiles (they carry the same
   * tile textures where the grids align); otherwise imagery tiles draw.
   */
  drawTiles(pass: GPURenderPassEncoder, cameraBindGroup: GPUBindGroup): void {
    if (this._terrainPrimitives.size > 0) {
      for (const entry of this._terrainPrimitives.values()) {
        entry.primitive.drawOpaque(pass, cameraBindGroup);
      }
      return;
    }
    for (const entry of this._tilePrimitives.values()) {
      entry.primitive.drawOpaque(pass, cameraBindGroup);
    }
    // Polar caps ride along with the flat imagery path (terrain replaces the
    // whole flat tile set including caps — see the early return above).
    for (const which of ['north', 'south'] as const) {
      this._polarCaps[which]?.primitive.drawOpaque(pass, cameraBindGroup);
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
   * MUST stay same-source with {@link Globe.drawTiles}: while terrain
   * primitives exist, drawTiles draws ONLY the terrain set (imagery tiles and
   * polar caps are skipped by its early return), so the pick pass must paint
   * the terrain geometry too — otherwise picking rasterises the flat imagery
   * tiles lifted at 1.008× radius while the G-Buffer holds displaced terrain
   * meshes, and every pick lands systematically offset.
   */
  get tilePrimitives(): Primitive[] {
    if (this._terrainPrimitives.size > 0) {
      const terrain: Primitive[] = [];
      for (const entry of this._terrainPrimitives.values()) terrain.push(entry.primitive);
      return terrain;
    }
    const out: Primitive[] = [];
    for (const entry of this._tilePrimitives.values()) out.push(entry.primitive);
    for (const which of ['north', 'south'] as const) {
      const cap = this._polarCaps[which];
      if (cap !== null) out.push(cap.primitive);
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
   * Keep the two polar cap patches in step with the active layer: Web
   * Mercator tiles stop at ±85.05°, so the caps (±85.05°…90°) are covered by
   * dedicated geometry that re-samples the ROOT grid's top / bottom row
   * stretched poleward. Without them the base colour bleeds through the
   * caps — visually a "hole" at high-latitude views.
   */
  private _syncPolarCaps(layer: ImageryLayer): void {
    const rootLevel = Math.min(ImageryLayer.ROOT_LEVEL, layer.provider.maximumLevel);
    const n = 2 ** rootLevel;
    const mercatorEdge = Math.atan(Math.sinh(Math.PI)); // ±85.0511°

    // The cap texture is any resident tile covering the top/bottom row at
    // the coarsest level with an own or fallback texture; row 0 = north.
    const caps: Array<'north' | 'south'> = ['north', 'south'];
    for (const which of caps) {
      // Find a resident texture for the cap's reference row (the row's
      // first tile via the fallback chain is enough — every tile in a row
      // resolves up to the same coarse ancestors).
      const y = which === 'north' ? 0 : n - 1;
      const refTile = new TileKey(rootLevel, 0, y);
      const resolved = layer.resolveTexture(refTile);
      const current = this._polarCaps[which];

      if (resolved === null) {
        // Nothing loaded yet for this row — drop any stale cap primitive.
        if (current !== null) {
          this._destroyTilePrimitive(current);
          this._polarCaps[which] = null;
        }
        continue;
      }
      if (current !== null && current.texture === resolved.texture) continue; // up to date

      if (current !== null) this._destroyTilePrimitive(current);

      // Cap geometry: full longitude sweep, Mercator edge → pole. UVs map
      // onto the reference row's sub-rect inside the resolved texture
      // (row stretched over the 4.95° cap — standard global-viewer behaviour).
      const south = which === 'north' ? mercatorEdge : -CesiumMath.PI_OVER_TWO;
      const north = which === 'north' ? CesiumMath.PI_OVER_TWO : -mercatorEdge;
      const rowsTotal = n;
      const vRange = 1.0 / rowsTotal;
      // TileKey y grows southward; texture v likewise (row 0 = north edge).
      const v0 = which === 'north' ? 0.0 : 1.0 - vRange;
      const uvRect = { u0: 0.0, v0, u1: 1.0, v1: v0 + vRange };
      // The cap always sits at the coarse lift: bordering row tiles pin
      // their edge rows to the same value via `_edgeLiftFor`, so the 85.05°
      // seam is continuous no matter which levels render in the row.
      const geometry = EllipsoidGeometry.createRegion(
        -CesiumMath.PI,
        south,
        CesiumMath.PI,
        north,
        POLAR_CAP_PARTITIONS,
        { uvRect, radiusScale: POLAR_CAP_RADIUS_SCALE },
      );
      const primitive = new Primitive(this._context, {
        vertices:    geometry.vertices,
        indices:     geometry.indices,
        translucent: false,
      });
      Matrix4.identity(primitive.modelMatrix);
      primitive.setTexture(resolved.texture, this._getTileSampler());
      primitive.buildOpaqueBindGroup(this._opaquePipeline as GPURenderPipeline);
      this._polarCaps[which] = { primitive, texture: resolved.texture };
    }
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
    });
    // Geometry is generated directly in normalised ECEF (EllipsoidGeometry
    // convention, Earth radius = 1) — identity model matrix, like the base globe.
    Matrix4.identity(primitive.modelMatrix);
    primitive.setTexture(texture, this._getTileSampler());
    primitive.buildOpaqueBindGroup(this._opaquePipeline as GPURenderPipeline);
    return { primitive, texture };
  }

  /**
   * Edge lifts pinning this tile's north/south boundary rows to the lift of
   * whatever renders across that boundary (seam continuity rule: both sides
   * of a shared latitude circle must sit at the same radius, or the step
   * reads as a seam ring at grazing angles). Sources:
   *  - a live neighbour tile in the adjacent quadtree row (may be a coarser
   *    level with a larger lift), or
   *  - the polar cap (constant 1.02) when the row borders ±85.05°.
   */
  private _edgeLiftFor(tile: TileKey): { north?: number; south?: number } | undefined {
    const own = tileRadiusScale(tile.level);
    if (own >= POLAR_CAP_RADIUS_SCALE) return undefined; // already coarse — nothing to pin
    const rows = 2 ** tile.level;
    const edge: { north?: number; south?: number } = {};
    // The cap renders at POLAR_CAP_RADIUS_SCALE across ±85.05° regardless of
    // the row's level, so bordering tiles always ramp to it.
    if (tile.y === 0) edge.north = POLAR_CAP_RADIUS_SCALE;
    if (tile.y === rows - 1) edge.south = POLAR_CAP_RADIUS_SCALE;
    // Neighbour rows: a live coarser tile whose row band covers the latitude
    // circle just north/south of us forces its (larger) lift onto the shared
    // edge. Bands are compared in this tile's row units: a neighbour at level
    // L spans rows [y·rows/2^L, (y+1)·rows/2^L) in our units.
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
    const cacheKey = `${tile.key}#${ancestorLevels}#${edgeLift?.north ?? '-'}#${edgeLift?.south ?? '-'}`;
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
      { uvRect, radiusScale: tileRadiusScale(tile.level), edgeLift, mercatorV: true }
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
        this._destroyTilePrimitive(entry);
        this._terrainPrimitives.delete(key);
      }
    }

    for (const tile of terrainLayer.renderTiles) {
      const key = `${tile.z}/${tile.x}/${tile.y}`;
      const mesh = terrainLayer.getMesh(tile.z, tile.x, tile.y);
      if (mesh === null) continue; // evicted between selection and sync

      // Textures: only the webmercator terrain grid aligns 1:1 with imagery
      // TileKeys, so tile textures are reused there; geographic terrain (and
      // anything unresolvable) renders with the flat fallback texture.
      let texture: GPUTexture;
      let textureLevels = -1; // -1 → fallback texture, full [0,1] uv
      if (layer !== null && terrainLayer.provider.tilingScheme === 'webmercator') {
        const resolved = layer.resolveTexture(new TileKey(tile.z, tile.x, tile.y));
        if (resolved !== null) {
          texture = resolved.texture;
          textureLevels = resolved.ancestorLevels;
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
      const geometryKey = this._terrainGeometryKey(tile, textureLevels);
      const entry = this._terrainPrimitives.get(key);
      if (entry !== undefined && (entry.texture !== texture || entry.geometryKey !== geometryKey)) {
        // Bound texture changed → rebuild with matching uv sub-rect geometry;
        // stitch signature changed → rebuild with the re-stitched edges.
        this._destroyTilePrimitive(entry);
        this._terrainPrimitives.delete(key);
      }
      if (!this._terrainPrimitives.has(key)) {
        this._terrainPrimitives.set(
          key,
          this._createTerrainPrimitive(tile, mesh, texture, textureLevels, geometryKey)
        );
      }
    }
  }

  /** Geometry cache key for a terrain render tile (includes the stitch signature). */
  private _terrainGeometryKey(tile: TerrainRenderTile, textureLevels: number): string {
    return `terrain:${tile.z}/${tile.x}/${tile.y}#${tile.ancestorLevels}#${textureLevels}#${tile.stitchSignature}`;
  }

  /** Build one terrain primitive (displaced mesh geometry + tile texture). */
  private _createTerrainPrimitive(
    tile: TerrainRenderTile,
    mesh: TerrainMeshData,
    texture: GPUTexture,
    textureLevels: number,
    geometryKey: string
  ): TilePrimitiveEntry {
    const uvRect = textureLevels > 0 ? this._tileUvRect(tile.x, tile.y, textureLevels) : undefined;
    const geometry = this._getTerrainGeometry(tile, mesh, uvRect, textureLevels);
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
    return { primitive, texture, geometryKey };
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
    textureLevels: number
  ): CachedTileGeometry {
    const cacheKey = this._terrainGeometryKey(tile, textureLevels);
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
      const x = positions[i * 3];
      const y = positions[i * 3 + 1] / WGS84_POLAR_RATIO; // undo squash
      const z = positions[i * 3 + 2] / WGS84_POLAR_RATIO;

      const lon = Math.atan2(y, x) * rad;
      const lat = Math.atan2(z, Math.hypot(x, y)) * rad;
      const u   = u0 + du * (lon - rect.west) / lonSpan;
      const v   = v0 + dv * (rect.north - lat) / latSpan; // row 0 = north edge

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
    entry.primitive.setTexture(this._detachTexture);
    entry.primitive.destroy(); // consumes the sentinel (destroy is idempotent)
    this._detachTexture = null;
  }

  /** Destroy all tile primitives including the polar caps (no layer active). */
  private _clearTilePrimitives(): void {
    if (this._tilePrimitives.size === 0) return;
    for (const entry of this._tilePrimitives.values()) {
      this._destroyTilePrimitive(entry);
    }
    this._tilePrimitives.clear();
    this._clearPolarCaps();
  }

  /** Destroy both polar cap patches (textures stay owned by the layer cache). */
  private _clearPolarCaps(): void {
    for (const which of ['north', 'south'] as const) {
      const cap = this._polarCaps[which];
      if (cap !== null) {
        this._destroyTilePrimitive(cap);
        this._polarCaps[which] = null;
      }
    }
  }

  destroy(): void {
    this._clearTilePrimitives();
    this._clearTerrainPrimitives();
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
}

/** @internal Cached interleaved geometry for a tile (+ UV variant). */
interface CachedTileGeometry {
  vertices: Float32Array;
  indices: Uint32Array;
}

/** Subdivisions per axis for tile geometry (a tile is small on screen). */
const TILE_PARTITIONS = 8;

/**
 * Tiles are drawn above the base globe so the legacy fallback texture
 * underneath never z-fights with them. The lift must exceed the worst-case
 * chord-vs-arc sagitta of a tile triangle sinking below the (much finer)
 * base-globe mesh: with TILE_PARTITIONS = 8 the largest tiles (L2 roots,
 * ≈11.25° per cell) sag ≈ 1 − cos(5.6°) ≈ 0.0048 below the surface, while
 * the base globe (128 stacks, ≈2.8° cells) sags only ≈ 0.0003. A 0.1% lift
 * (1.001) is swallowed by that gap — mid-tile fragments lose the depth test
 * against the base globe and vanish, leaving the base colour showing through
 * (the "uncovered regions" defect).
 *
 * Level-dependent lift (2026-09-08 eagle-eye fix): small overview canvases
 * (e.g. the 62_eagleEyeMap mini viewer) keep the camera at ~1.94 R nadir and
 * never refine past the L2 roots — and at the disc edge those big tiles are
 * viewed at a ~30° grazing angle, where depth-buffer perspective compression
 * shrinks the effective lift below the 0.0048 worst-case sagitta. Mid-tile
 * fragments then lose the depth test against the base globe and pop out as
 * diamond-shaped holes (20-30% of the disc). Coarse levels (≤ 3) get 2%,
 * which clears the grazing-angle case with margin; finer levels keep 0.8%
 * (≈50 km at true scale, imperceptible at their viewing distances).
 */
function tileRadiusScale(level: number): number {
  return level <= 3 ? 1.02 : 1.008;
}

/**
 * Polar cap patches render at the coarse lift unconditionally; row tiles
 * bordering ±85.05° pin their edge rows here (`Globe._edgeLiftFor`), so the
 * cap/tile seam stays continuous for every mix of levels in the row.
 */
const POLAR_CAP_RADIUS_SCALE = 1.02;

/** Longitude × latitude partitions of a polar cap patch (small — it only
 *  stretches one ROOT-row of imagery over the 4.95° cap). */
const POLAR_CAP_PARTITIONS = 64;

/** WGS84 polar / equatorial ratio (matches EllipsoidGeometry normalisation). */
const WGS84_POLAR_RATIO = 6356752.3142 / 6378137.0;

/** Max cached tile geometries (CPU arrays only; ~4 KB each at 8×8). */
const TILE_GEOMETRY_CACHE_CAPACITY = 1024;

export default Globe;
