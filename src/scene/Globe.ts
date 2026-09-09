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
      return;
    }

    const layer = this.layers.activeLayer;
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
    // Polar fills ride along with the flat imagery path (terrain replaces the
    // whole flat tile set including fills — see the early return above).
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
   * MUST stay same-source with {@link Globe.drawTiles}: while terrain
   * primitives exist, drawTiles draws ONLY the terrain set (flat imagery
   * tiles and polar fills are skipped by its early return), so the pick pass
   * must paint the terrain geometry too — otherwise picking rasterises the
   * flat imagery tiles while the G-Buffer holds displaced terrain meshes,
   * and every pick lands systematically offset.
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
      for (let slice = 0; slice <= slices; slice++) {
        const lon = west + (east - west) * (slice / slices);
        const nx = cosLat * Math.cos(lon);
        const ny = cosLat * Math.sin(lon);
        const nz = sinLat;

        vertices[vi++] = r * nx;
        vertices[vi++] = r * ny * ratio;
        vertices[vi++] = r * nz * ratio;

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
}

/** @internal Cached interleaved geometry for a tile (+ UV variant). */
interface CachedTileGeometry {
  vertices: Float32Array;
  indices: Uint32Array;
}

/** Subdivisions per axis for tile geometry (a tile is small on screen). */
const TILE_PARTITIONS = 8;

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
  return 1.0;
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

export default Globe;
