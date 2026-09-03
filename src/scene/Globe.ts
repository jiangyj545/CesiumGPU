import { Context } from '../renderer/Context';
import { Primitive } from './Primitive';
import { EllipsoidGeometry } from './EllipsoidGeometry';
import { Camera } from './Camera';
import { ImageryLayerCollection } from '../imagery/ImageryLayerCollection';
import { ImageryLayer } from '../imagery/ImageryLayer';
import { TileKey } from '../imagery/TileKey';
import { Color } from '../math/Color';
import { Matrix4 } from '../math/Matrix4';
import { Cartesian3 } from '../math/Cartesian3';

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
   * Load a texture from an HTMLImageElement and apply it to the globe.
   */
  loadImageryTexture(image: HTMLImageElement): void {
    if (!this._primitive) return;
    const device = this._context.device;

    const bitmap = createImageBitmap(image).then(bitmap => {
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
    });

    void bitmap;
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
      return;
    }

    const layer = this.layers.activeLayer;
    if (layer === null) {
      this._clearTilePrimitives();
      return;
    }

    layer.update(this._context.device, camera, this._context.drawingBufferHeight);
    this._syncTilePrimitives(layer);
  }

  /** Draw all live tile primitives in the G-Buffer pass. */
  drawTiles(pass: GPURenderPassEncoder, cameraBindGroup: GPUBindGroup): void {
    for (const entry of this._tilePrimitives.values()) {
      entry.primitive.drawOpaque(pass, cameraBindGroup);
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
   */
  get tilePrimitives(): Primitive[] {
    const out: Primitive[] = [];
    for (const entry of this._tilePrimitives.values()) out.push(entry.primitive);
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

  /** Build one per-tile primitive (region geometry + tile texture). */
  private _createTilePrimitive(
    tile: TileKey,
    texture: GPUTexture,
    ancestorLevels: number
  ): TilePrimitiveEntry {
    const geometry = this._getTileGeometry(tile, ancestorLevels);
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

  /** Cached (or newly generated) interleaved geometry for a tile. */
  private _getTileGeometry(tile: TileKey, ancestorLevels: number): CachedTileGeometry {
    const cacheKey = `${tile.key}#${ancestorLevels}`;
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
      { uvRect, radiusScale: TILE_RADIUS_SCALE }
    );

    if (this._geometryCache.size >= TILE_GEOMETRY_CACHE_CAPACITY) {
      this._geometryCache.clear();
    }
    this._geometryCache.set(cacheKey, geometry);
    return geometry;
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

  /** Destroy all tile primitives (used when no layer is active). */
  private _clearTilePrimitives(): void {
    if (this._tilePrimitives.size === 0) return;
    for (const entry of this._tilePrimitives.values()) {
      this._destroyTilePrimitive(entry);
    }
    this._tilePrimitives.clear();
  }

  destroy(): void {
    this._clearTilePrimitives();
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
}

/** @internal Cached interleaved geometry for a tile (+ UV variant). */
interface CachedTileGeometry {
  vertices: Float32Array;
  indices: Uint32Array;
}

/** Subdivisions per axis for tile geometry (a tile is small on screen). */
const TILE_PARTITIONS = 8;

/**
 * Tiles are drawn a hair above the base globe so the legacy fallback texture
 * underneath never z-fights with them.
 */
const TILE_RADIUS_SCALE = 1.001;

/** Max cached tile geometries (CPU arrays only; ~4 KB each at 8×8). */
const TILE_GEOMETRY_CACHE_CAPACITY = 1024;

export default Globe;
