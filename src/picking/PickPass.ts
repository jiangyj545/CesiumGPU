import { Context } from '../renderer/Context';
import { Frustum } from '../renderer/Frustum';
import type { Camera } from '../scene/Camera';
import type { Primitive, PrimitiveCollection } from '../scene/Primitive';
import { Cartesian3 } from '../math/Cartesian3';
import { PICK_SHADER, PICK_ID_TEST_SHADER } from '../shaders/PickShader';
import {
  decodePickInstanceId,
  tilesetPickRegistry,
  type Cesium3DTileFeature,
} from '../tileset/Cesium3DTileFeature';
import { CAMERA_UBO_SIZE, writeCameraUniforms } from '../renderer/CameraUniforms';

/** Signature of the per-frame encode hook (matches the Scene.frameHooks shape). */
export type PickFrameHook = (
  encoder: GPUCommandEncoder,
  swapChainView: GPUTextureView
) => void;

/**
 * Narrow, structural view of the globe's pickable primitives. Satisfied by
 * the Globe class — PickPass depends on this slice, not on Globe or Scene.
 */
export interface PickPassGlobe {
  readonly show: boolean;
  /** Base sphere primitive, or null before Globe.initialize(). */
  readonly primitive: Primitive | null;
  /** Live per-tile imagery primitives. */
  readonly tilePrimitives: Primitive[];
  readonly tilePrimitiveCount: number;
  /**
   * Quadtree surface mode: tiles ARE the globe (CesiumJS architecture) —
   * the base sphere must be skipped in the pick view just like the G-Buffer
   * pass does, otherwise both sit at r=1.0 and the finer-tessellated sphere
   * z-fight-wins every tile pixel, collapsing all globe picks onto the
   * sphere's id.
   */
  readonly surfaceIsQuadtree: boolean;
}

/**
 * Explicit dependencies injected by the Scene — PickPass has no Scene import.
 * The getters are polled lazily because GPU resources (pipelines, G-Buffer
 * textures) come and go with scene initialize/resize, while a PickPass
 * instance can be created once and reused.
 */
export interface PickPassDeps {
  /** WebGPU context: device, canvas sizing, queue writes, submit. */
  context: Context;
  /** Camera for the pick-view uniform upload + cache fingerprint. */
  camera: Camera;
  /** Scene-graph primitives (the opaque set is re-drawn into the pick view). */
  primitives: PrimitiveCollection;
  /** Globe-provided pickables (base mesh + live tiles). */
  globe: PickPassGlobe;
  /** Shared opaque pipeline — null before the scene is initialized. */
  getOpaquePipeline(): GPURenderPipeline | null;
  /** G-Buffer worldPos texture for the 1×1 position readback (null-safe). */
  getWorldPosTexture(): GPUTexture | null;
  /** Whether the pick set should be frustum-culled (Scene.debugFrustumCulling). */
  frustumCullingEnabled(): boolean;
  /** Register / unregister the per-frame encode hook on the frame encoder. */
  addFrameHook(hook: PickFrameHook): void;
  removeFrameHook(hook: PickFrameHook): void;
}

/**
 * Result of a single GPU pick.
 *
 * `entity` integration (dataSource) is planned for a later task; the picked
 * `primitive` is already resolved so callers can identify the hit object.
 */
export interface PickResult {
  /** 24-bit object id (0 = background / no hit). */
  id: number;
  /** The picked primitive, or null for a miss (id 0). */
  primitive: Primitive | null;
  /**
   * 3D Tiles feature for this pick (P1-E): batch-level when the b3dm carries
   * `_BATCHID` (`getProperty` reads the batch table), tile-level (batchId
   * undefined, properties undefined) otherwise. null for non-tileset hits.
   */
  feature: Cesium3DTileFeature | null;
  /** Normalised world-space position (Earth radius = 1), null on miss. */
  worldPosition: Cartesian3 | null;
  /** Timing breakdown (ms). */
  timing: {
    /** Pick-view encode time inside the frame hook. */
    encodeMs: number;
    /** submit → mapAsync resolve (the actual GPU readback latency). */
    readbackMs: number;
    /** encode start → mapAsync resolve. */
    totalMs: number;
  };
}

interface PendingPick {
  x: number;
  y: number;
  camFp: number;
  sceneFp: number;
  encoded: boolean;
  tEncodeStart: number;
  tEncodeEnd: number;
  promise: Promise<PickResult>;
  resolve: (result: PickResult) => void;
  reject: (error: unknown) => void;
}

interface CacheEntry {
  x: number;
  y: number;
  camFp: number;
  sceneFp: number;
  result: PickResult;
}

/** WebGPU requires COPY_TEXTURE_TO_BUFFER bytesPerRow to be 256-aligned. */
const READBACK_BYTES_PER_ROW = 256;

/**
 * PickPass - Production GPU picking via id-encoded render-to-texture + readback.
 *
 * Every opaque primitive drawn into the G-Buffer (globe base mesh, imagery
 * tiles and the scene's primitive collection) is automatically pickable: each
 * `Primitive` owns a 24-bit `pickId` assigned at construction. When a pick is
 * pending, the pass re-draws that same set into a canvas-sized rgba8unorm
 * "pick view" using the PickShader, appends 1×1 copies of the pick texture and
 * the G-Buffer worldPos texture to the frame encoder (via the injected frame
 * hooks), and decodes them after submit with `mapAsync`.
 *
 * Dependencies are injected via the {@link PickPassDeps} parameter object —
 * this module has no import of the Scene class (the Scene wires its narrow
 * accessors in `Scene._buildPickDeps`).
 *
 * Guarantees:
 *  - at most one readback per rendered frame (concurrent `pick()` calls
 *    share the in-flight promise);
 *  - identical set/order/state as the G-Buffer pass, so pick results match
 *    the visible image;
 *  - the last result is cached and served from cache until the camera or
 *    the scene content changes.
 *
 * Coordinates: `pick(x, y)` takes canvas CSS-pixel coordinates with a
 * top-left origin — exactly the space reported by
 * `ScreenSpaceEventHandler` (`clientX/clientY` minus the canvas rect).
 * The CSS→texel conversion (`pickTextureWidth / canvas.clientWidth`) is
 * applied once here, so callers never scale by devicePixelRatio themselves.
 */
export class PickPass {
  private readonly _deps: PickPassDeps;

  /** Primitives manually registered in addition to the scene graph. */
  private readonly _registered = new Set<Primitive>();

  /** Lazily created id UBO / bind group per pick id. */
  private readonly _idUbos = new Map<number, GPUBuffer>();
  private readonly _idBindGroups = new Map<number, GPUBindGroup>();

  private _pipeline: GPURenderPipeline | null = null;
  /**
   * F12 instance-id variant: second vertex buffer (stride 4, 'uint32' @
   * location 3) carries the per-vertex instance id; fragment emits
   * `0x800000 | instanceId`. Created lazily alongside `_pipeline`.
   */
  private _instancePipeline: GPURenderPipeline | null = null;
  private _pickIdBGL: GPUBindGroupLayout | null = null;
  private _cameraBindGroup: GPUBindGroup | null = null;

  private _cameraUbo: GPUBuffer;
  private _pickTexture: GPUTexture | null = null;
  private _pickDepth: GPUTexture | null = null;
  private _targetWidth  = 0;
  private _targetHeight = 0;

  // Readback buffers (reused; 256 B each for the 256-aligned bytesPerRow)
  private _idReadBuffer: GPUBuffer;
  private _posReadBuffer: GPUBuffer;

  private _pending: PendingPick | null = null;
  private _cache: CacheEntry | null = null;

  /** id → primitive map, rebuilt on every pick-view encode. */
  private _idToPrimitive = new Map<number, Primitive>();

  /** Diagnostics: how often each pipeline stage ran. */
  readonly debug = { hookCalls: 0, encodes: 0, flushes: 0, gpuErrors: 0 };

  /**
   * Debug: render the pick view without the 1×1 scissor so the whole id
   * plane can be dumped via `debugDumpPickTexture`.
   */
  debugFullViewport = false;

  /** Registered once via `deps.addFrameHook`; encodes the pick view. */
  private readonly _frameHook: (encoder: GPUCommandEncoder) => void;

  /** Diagnostic listener for uncaptured GPU errors; removed in destroy(). */
  private readonly _errorListener: (event: Event) => void;

  constructor(deps: PickPassDeps) {
    this._deps = deps;

    const device = deps.context.device;

    this._cameraUbo = device.createBuffer({
      label: 'PickPass.CameraUBO',
      size:  CAMERA_UBO_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bufferUsage = GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ;
    this._idReadBuffer = device.createBuffer({
      label: 'PickPass.IdReadback',       size: READBACK_BYTES_PER_ROW, usage: bufferUsage,
    });
    this._posReadBuffer = device.createBuffer({
      label: 'PickPass.WorldPosReadback', size: READBACK_BYTES_PER_ROW, usage: bufferUsage,
    });

    this._frameHook = (_encoder: GPUCommandEncoder) => {
      this.debug.hookCalls += 1;
      // W-A 修复（实验①命中）：pick 编制走**独立 encoder + 独立 submit**，与主帧
      // 编码完全隔离——同 encoder 混编时 pick 的校验错误会使整个 submission
      // 作废（主帧全空帧，零 GPU 错误表面现象，2026-10-18）。pick 用自持
      // 纹理/缓冲，与主帧无资源依赖，独立提交语义等价且互不污染。
      const enc = device.createCommandEncoder({ label: 'PickPass.Enc' });
      this._encodePickFrame(enc);
      device.queue.submit([enc.finish()]);
    };
    deps.addFrameHook(this._frameHook);

    this._errorListener = (event: Event) => {
      this.debug.gpuErrors += 1;
      console.error('[PickPass] GPU uncaptured error:', (event as GPUUncapturedErrorEvent).error.message);
    };
    device.addEventListener('uncapturederror', this._errorListener as EventListener);
  }

  // ── Registration (primitives outside the scene graph) ────────────────────────

  /**
   * Manually register a primitive that is not reachable through the scene
   * graph. Scene-graph primitives (globe base, imagery tiles,
   * `scene.primitives`) are picked automatically and never need this.
   * @returns the primitive's 24-bit pick id.
   */
  register(primitive: Primitive): number {
    this._registered.add(primitive);
    this._invalidateCache();
    return primitive.pickId;
  }

  /** Remove a manual registration. */
  unregister(primitive: Primitive): void {
    this._registered.delete(primitive);
    this._invalidateCache();
  }

  // ── Picking ──────────────────────────────────────────────────────────────────

  /**
   * Request a pick at canvas CSS-pixel coordinates (top-left origin — the
   * same space as `ScreenSpaceEventHandler` positions). Converted to
   * pick-texture texels internally via the texture/CSS-size ratio.
   *
   * - While a pick is in flight, further calls return the SAME pending
   *   promise (max one readback per frame).
   * - Repeated picks of the same pixel are served from cache until the
   *   camera or the scene content changes.
   *
   * @returns null when the scene is not initialised/destroyed, otherwise a
   * `PickResult` whose `id === 0` means "background / no hit".
   */
  pick(x: number, y: number): Promise<PickResult | null> {
    if (!this._deps.getOpaquePipeline()) return Promise.resolve(null);

    // Convert CSS pixels → pick-texture texels with the ratio between the
    // pick target size and the canvas CSS size (the effective DPR, correct
    // even when the drawing buffer was clamped). Ratio is 1 at DPR 1, so
    // the legacy path is unchanged.
    const canvas = this._deps.context.canvas;
    const w = this._targetWidth  || Math.max(1, this._deps.context.drawingBufferWidth);
    const h = this._targetHeight || Math.max(1, this._deps.context.drawingBufferHeight);
    const sx = w / Math.max(1, canvas.clientWidth);
    const sy = h / Math.max(1, canvas.clientHeight);
    const cx = Math.min(Math.max(0, Math.round(x * sx)), w - 1);
    const cy = Math.min(Math.max(0, Math.round(y * sy)), h - 1);

    const camFp   = this._cameraFingerprint();
    const sceneFp = this._sceneFingerprint();

    if (
      this._cache &&
      this._cache.x === cx && this._cache.y === cy &&
      this._cache.camFp === camFp && this._cache.sceneFp === sceneFp
    ) {
      // Timing of a cache hit is not a fresh measurement — zero it out.
      const cached = this._cache.result;
      return Promise.resolve({ ...cached, timing: { encodeMs: 0, readbackMs: 0, totalMs: 0 } });
    }

    if (this._pending) return this._pending.promise;

    let resolve!: (r: PickResult) => void;
    let reject!: (e: unknown) => void;
    const promise = new Promise<PickResult>((res, rej) => {
      resolve = res;
      reject  = rej;
    });

    this._pending = {
      x: cx,
      y: cy,
      camFp,
      sceneFp,
      encoded: false,
      tEncodeStart: 0,
      tEncodeEnd: 0,
      promise,
      resolve,
      reject,
    };
    return promise;
  }

  /** Drop the cached result (e.g. after manual register/unregister). */
  invalidate(): void {
    this._invalidateCache();
  }

  /**
   * Verify rgba8unorm id precision: render `testId` through the encode path
   * into a 1×1 rgba8unorm target, read it back and decode.
   * Returns the decoded id (=== testId when lossless).
   */
  async verifyIdPrecision(testId: number): Promise<number> {
    const device = this._deps.context.device;

    const target = device.createTexture({
      label: 'PickPass.IdPrecisionTarget',
      size:   { width: 1, height: 1 },
      format: 'rgba8unorm',
      usage:  GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const shader = device.createShaderModule({ code: PICK_ID_TEST_SHADER, label: 'PickIdTestShader' });
    const bgl    = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });
    const pipeline = device.createRenderPipeline({
      label:  'PickPass.IdPrecisionPipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex:   { module: shader, entryPoint: 'vs_main' },
      fragment: { module: shader, entryPoint: 'fs_main', targets: [{ format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list' },
    });
    const ubo = device.createBuffer({
      size:  256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._deps.context.writeBuffer(ubo, 0, new Uint32Array([testId >>> 0, 0, 0, 0]));
    const bg = device.createBindGroup({
      layout: bgl,
      entries: [{ binding: 0, resource: { buffer: ubo } }],
    });

    const encoder = device.createCommandEncoder({ label: 'PickPass.IdPrecision' });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view:       target.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp:     'clear',
        storeOp:    'store',
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
    encoder.copyTextureToBuffer(
      { texture: target },
      { buffer: this._idReadBuffer, bytesPerRow: READBACK_BYTES_PER_ROW },
      { width: 1, height: 1 },
    );
    this._deps.context.submit([encoder.finish()]);

    await this._idReadBuffer.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(this._idReadBuffer.getMappedRange());
    const decoded = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16);
    const alpha   = bytes[3];
    this._idReadBuffer.unmap();

    target.destroy();
    ubo.destroy();

    if (alpha !== 255) {
      throw new Error(`PickPass: precision test alpha=${alpha}, expected 255`);
    }
    return decoded;
  }

  /**
   * Debug: dump the whole pick-texture id plane as tightly packed RGBA bytes.
   * Must be called after a pick resolved (the texture holds that pick's
   * render). Returns null when the pick target does not exist yet.
   */
  async debugDumpPickTexture(): Promise<{ w: number; h: number; data: Uint8Array } | null> {
    const tex = this._pickTexture;
    if (!tex) return null;
    const device = this._deps.context.device;
    const w = this._targetWidth;
    const h = this._targetHeight;
    const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
    const buf = device.createBuffer({
      label:  'PickPass.DebugDump',
      size:   bytesPerRow * h,
      usage:  GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder({ label: 'PickPass.DebugDump' });
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow }, { width: w, height: h });
    this._deps.context.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) {
      out.set(src.subarray(y * bytesPerRow, y * bytesPerRow + w * 4), y * w * 4);
    }
    buf.unmap();
    buf.destroy();
    return { w, h, data: out };
  }

  destroy(): void {
    this._deps.removeFrameHook(this._frameHook);

    this._deps.context.device.removeEventListener('uncapturederror', this._errorListener as EventListener);

    // Reject any in-flight pick so awaiting callers do not hang forever
    // (the frame hook is gone, so the job could never be flushed).
    if (this._pending) {
      this._pending.reject(new Error('PickPass destroyed'));
      this._pending = null;
    }

    for (const ubo of this._idUbos.values()) ubo.destroy();
    this._idUbos.clear();
    this._idBindGroups.clear();
    this._registered.clear();
    this._cache = null;
    this._pending = null;
    this._idToPrimitive.clear();
    this._cameraUbo.destroy();
    this._idReadBuffer.destroy();
    this._posReadBuffer.destroy();
    this._pickTexture?.destroy();
    this._pickDepth?.destroy();
  }

  // ── Internal ─────────────────────────────────────────────────────────────────

  private _invalidateCache(): void {
    this._cache = null;
  }

  /** Change-detector for the cached pick: camera view state. */
  private _cameraFingerprint(): number {
    const cam = this._deps.camera;
    const p   = cam.position;
    const v   = cam.viewMatrix.values;
    let h = 0x811c9dc5;
    h = Math.imul(h ^ Math.fround(p.x), 0x01000193) | 0;
    h = Math.imul(h ^ Math.fround(p.y), 0x01000193) | 0;
    h = Math.imul(h ^ Math.fround(p.z), 0x01000193) | 0;
    for (let i = 0; i < 16; i++) {
      h = Math.imul(h ^ Math.fround(v[i]), 0x01000193) | 0;
    }
    return h;
  }

  /** Change-detector for the cached pick: pickable content + visibility. */
  private _sceneFingerprint(): number {
    // Visibility must participate: toggling prim/globe `.show` without adding
    // or removing anything otherwise keeps serving stale cached picks.
    let visible = this._deps.globe.show ? 1 : 0;
    for (const p of this._deps.primitives) if (p.show) visible += 1;
    for (const p of this._registered) if (p.show) visible += 1;
    return (
      this._deps.primitives.length * 2654435761 +
      this._deps.globe.tilePrimitiveCount * 40503 +
      this._registered.size * 104729 +
      visible * 31337
    );
  }

  /** Ensure the pick pipeline exists (needs the injected opaque pipeline's BGLs). */
  private _ensurePipeline(): void {
    if (this._pipeline) return;
    const opaque = this._deps.getOpaquePipeline();
    if (!opaque) throw new Error('PickPass: scene not initialized yet');

    const device   = this._deps.context.device;
    // Reuse the opaque pipeline's bind group layouts so existing Primitive
    // bind groups (group 0 camera, group 1 model) stay compatible.
    const cameraBGL = opaque.getBindGroupLayout(0);
    const modelBGL  = opaque.getBindGroupLayout(1);

    this._pickIdBGL = device.createBindGroupLayout({
      label: 'PickPass.BGL2',
      entries: [{
        binding:   0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer:    { type: 'uniform' },
      }],
    });

    const shader = device.createShaderModule({ code: PICK_SHADER, label: 'PickShader' });

    this._pipeline = device.createRenderPipeline({
      label:  'PickPassPipeline',
      layout: device.createPipelineLayout({
        bindGroupLayouts: [cameraBGL, modelBGL, this._pickIdBGL],
      }),
      vertex: {
        module:     shader,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 32, // same interleaved layout as the G-Buffer pass
          attributes: [
            { shaderLocation: 0, offset: 0,  format: 'float32x3' }, // position
            { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
            { shaderLocation: 2, offset: 24, format: 'float32x2' }, // uv
          ],
        }],
      },
      fragment: {
        module:     shader,
        entryPoint: 'fs_main',
        targets: [{ format: 'rgba8unorm' }],
      },
      primitive:    { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });

    this._cameraBindGroup = device.createBindGroup({
      label:  'PickPass.CameraBG',
      layout: cameraBGL,
      entries: [{ binding: 0, resource: { buffer: this._cameraUbo } }],
    });

    // F12 instance-id pipeline: same bind group layouts (groups 0-2) as the
    // base pick pipeline, plus the per-vertex instance id buffer at slot 1.
    this._instancePipeline = device.createRenderPipeline({
      label:  'PickPassInstancePipeline',
      layout: device.createPipelineLayout({
        bindGroupLayouts: [cameraBGL, modelBGL, this._pickIdBGL],
      }),
      vertex: {
        module:     shader,
        entryPoint: 'vs_instanced',
        buffers: [
          {
            arrayStride: 32, // same interleaved layout as the G-Buffer pass
            attributes: [
              { shaderLocation: 0, offset: 0,  format: 'float32x3' }, // position
              { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
              { shaderLocation: 2, offset: 24, format: 'float32x2' }, // uv
            ],
          },
          {
            arrayStride: 4,
            attributes: [
              { shaderLocation: 3, offset: 0, format: 'uint32' }, // per-vertex instance id
            ],
          },
        ],
      },
      fragment: {
        module:     shader,
        entryPoint: 'fs_instanced',
        targets: [{ format: 'rgba8unorm' }],
      },
      primitive:    { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });
  }

  /** Lazily create (and cache) the per-id uniform + bind group. */
  private _idBindGroup(pickId: number): GPUBindGroup {
    let bg = this._idBindGroups.get(pickId);
    if (bg) return bg;

    const device = this._deps.context.device;
    let ubo = this._idUbos.get(pickId);
    if (!ubo) {
      ubo = device.createBuffer({
        label: `PickPass.IdUBO#${pickId}`,
        size:  256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this._deps.context.writeBuffer(ubo, 0, new Uint32Array([pickId, 0, 0, 0]));
      this._idUbos.set(pickId, ubo);
    }
    bg = device.createBindGroup({
      label:  'PickPass.IdBG',
      layout: this._pickIdBGL!,
      entries: [{ binding: 0, resource: { buffer: ubo } }],
    });
    this._idBindGroups.set(pickId, bg);
    return bg;
  }

  /** Recreate the pick render targets when the canvas size changes. */
  private _syncTargetSize(): void {
    const ctx = this._deps.context;
    const w = ctx.drawingBufferWidth;
    const h = ctx.drawingBufferHeight;
    if (this._pickTexture && this._targetWidth === w && this._targetHeight === h) return;
    if (w < 1 || h < 1) return;

    this._pickTexture?.destroy();
    this._pickDepth?.destroy();

    const device = ctx.device;
    this._pickTexture = device.createTexture({
      label:  'PickPass.Target',
      size:   { width: w, height: h },
      format: 'rgba8unorm',
      usage:  GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    this._pickDepth = device.createTexture({
      label:  'PickPass.Depth',
      size:   { width: w, height: h },
      format: 'depth32float',
      usage:  GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this._targetWidth  = w;
    this._targetHeight = h;
  }

  /** Upload camera uniforms (layout defined in src/renderer/CameraUniforms.ts). */
  private _uploadCamera(): void {
    const cam  = this._deps.camera;
    const data = new Float32Array(CAMERA_UBO_SIZE / 4);

    // RTC 尾段与 Scene 主相机写入端保持同值（proj·viewRot + 位置 hi/lo）——
    // RTC 图元（瓦片/城市）在 pick 视图里走同款精确路径；缺尾段时 hi/lo 差分
    // 会以 0 为基准产生 ~1.0 量级的伪相对坐标（W-A，2026-10-18）。
    const v = cam.viewMatrix.values;
    const vr = [v[0], v[1], v[2], 0, v[4], v[5], v[6], 0, v[8], v[9], v[10], 0, 0, 0, 0, 1];
    const pj = cam.projectionMatrix.values;
    const m = new Array<number>(16).fill(0);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += pj[k * 4 + r] * vr[c * 4 + k];
        m[c * 4 + r] = s;
      }
    }
    const viewRotProj = new Float32Array(m);
    const fr = (x: number): number => Math.fround(x);
    const positionHi = { x: fr(cam.position.x), y: fr(cam.position.y), z: fr(cam.position.z) };
    const positionLo = {
      x: cam.position.x - positionHi.x,
      y: cam.position.y - positionHi.y,
      z: cam.position.z - positionHi.z,
    };

    writeCameraUniforms(data, {
      viewMatrix:           cam.viewMatrix.values,
      projectionMatrix:     cam.projectionMatrix.values,
      viewProjectionMatrix: cam.viewProjectionMatrix.values,
      position:             cam.position,
      viewportWidth:        this._targetWidth,
      viewportHeight:       this._targetHeight,
      near:                 cam.frustum.near,
      far:                  cam.frustum.far,
      viewRotProjMatrix:    viewRotProj,
      positionHi,
      positionLo,
    });

    this._deps.context.writeBuffer(this._cameraUbo, 0, data);
  }

  /**
   * Draw the exact same opaque set as the G-Buffer pass:
   * globe base mesh → imagery tiles → scene primitives (opaque) → manual.
   * The frustum culling applied here mirrors the G-Buffer pass so the pick
   * set always equals the visible set (a sphere fully outside the frustum
   * rasterises nothing, so results are identical either way).
   */
  private _drawPickSet(pass: GPURenderPassEncoder, cameraBindGroup: GPUBindGroup): void {
    this._idToPrimitive = new Map<number, Primitive>();

    const frustum = this._deps.frustumCullingEnabled()
      ? Frustum.fromViewProjection(this._deps.camera.viewProjectionMatrix)
      : null;

    const drawWithId = (prim: Primitive): void => {
      if (!prim.show) return;
      if (frustum) {
        const bs = prim.boundingSphere;
        if (!frustum.intersectsSphere(bs.center, bs.radius)) return;
      }
      // F12: primitives carrying a per-vertex instance id buffer render through
      // the two-slot pipeline so the id plane holds per-BUILDING ids; everything
      // else keeps the original single-buffer pipeline (unchanged semantics).
      if (prim.hasInstanceIds && prim.instanceIdBuffer && this._instancePipeline) {
        pass.setPipeline(this._instancePipeline);
        pass.setVertexBuffer(1, prim.instanceIdBuffer);
      } else {
        pass.setPipeline(this._pipeline!);
      }
      this._idToPrimitive.set(prim.pickId, prim);
      pass.setBindGroup(2, this._idBindGroup(prim.pickId));
      prim.drawOpaque(pass, cameraBindGroup);
    };

    const globe = this._deps.globe;
    if (globe.show) {
      const base = globe.primitive;
      if (base && !globe.surfaceIsQuadtree) {
        drawWithId(base);
      }
      for (const tile of globe.tilePrimitives) drawWithId(tile);
    }

    for (const prim of this._deps.primitives.opaque) drawWithId(prim);
    for (const prim of this._registered) drawWithId(prim);
  }

  /**
   * Frame hook body: append the pick-view render pass + 1×1 copies to the
   * shared frame encoder. Runs before Scene.submit(); mapping is deferred to
   * a microtask so it strictly follows the submit.
   */
  private _encodePickFrame(encoder: GPUCommandEncoder): void {
    // W-A E1（实验①）：运行时禁用 pick 编制——验证「pick pass 与 GBuffer 同
    // encoder 状态污染」假设（页设 globalThis.__WA_E1_DISABLE_PICK__=true）。
    if ((globalThis as Record<string, unknown>).__WA_E1_DISABLE_PICK__ === true) return;
    const job = this._pending;
    if (!job || job.encoded) return;

    try {
      this._ensurePipeline();
      this._syncTargetSize();
      if (!this._pipeline || !this._pickTexture || !this._pickDepth || !this._cameraBindGroup) return;

      this._uploadCamera();

      const pass = encoder.beginRenderPass({
        label: 'Pick Pass',
        colorAttachments: [{
          view:       this._pickTexture.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 }, // id 0 = background
          loadOp:     'clear',
          storeOp:    'store',
        }],
        depthStencilAttachment: {
          view:            this._pickDepth.createView(),
          depthClearValue: 1.0,
          depthLoadOp:     'clear',
          depthStoreOp:    'store',
        },
      });
      pass.setPipeline(this._pipeline);
      // 采样单一 texel 的正确方式：全尺寸 viewport（保持 NDC→framebuffer
      // 映射，图元只光栅化到自己的屏幕位置）+ 1×1 scissor（片段写入限制到
      // 被 copyTextureToBuffer 读取的那个 texel；WebGPU scissor 恒启用，
      // 无需管线状态；job.x/y 已钳制到 [0, size-1]）。
      //
      // 历史教训（2026-09-21 GPU 实测定案）：曾用 setViewport(x, y, 1, 1)
      // 代替 scissor —— 1×1 viewport 把整个 NDC 平面 [-1,1]² 压缩进那一个
      // texel，任何与视锥相交的图元（无论投影在屏幕何处）都变换到该 texel
      // 内光栅化，深度竞争后恒返回同一图元的 id（含 GLB Model 的场景点哪
      // 里都返回模型 id）。本 pass 独立编码，per-pass 状态不会泄漏到主渲染。
      if (!this.debugFullViewport) {
        pass.setScissorRect(job.x, job.y, 1, 1);
      }
      this._drawPickSet(pass, this._cameraBindGroup);
      pass.end();

      const origin   = { x: job.x, y: job.y, z: 0 };
      const copySize = { width: 1, height: 1 };
      encoder.copyTextureToBuffer(
        { texture: this._pickTexture, origin },
        { buffer: this._idReadBuffer, bytesPerRow: READBACK_BYTES_PER_ROW },
        copySize,
      );
      const worldPos = this._deps.getWorldPosTexture();
      if (worldPos) {
        encoder.copyTextureToBuffer(
          { texture: worldPos, origin },
          { buffer: this._posReadBuffer, bytesPerRow: READBACK_BYTES_PER_ROW },
          copySize,
        );
      }

      job.encoded      = true;
      job.tEncodeStart = performance.now();
      job.tEncodeEnd   = job.tEncodeStart;
      this.debug.encodes += 1;

      // Runs after Scene.render() returns, i.e. strictly after submit.
      queueMicrotask(() => { void this._flushJob(); });
    } catch (err) {
      this._pending = null;
      job.reject(err);
    }
  }

  /** Map the readback buffers (post-submit), decode and resolve the pick. */
  private async _flushJob(): Promise<void> {
    const job = this._pending;
    if (!job || !job.encoded) return;

    try {
      this.debug.flushes += 1;
      const tMapStart = performance.now();
      await Promise.all([
        this._idReadBuffer.mapAsync(GPUMapMode.READ),
        this._posReadBuffer.mapAsync(GPUMapMode.READ),
      ]);

      const idBytes = new Uint8Array(this._idReadBuffer.getMappedRange());
      const id = idBytes[0] | (idBytes[1] << 8) | (idBytes[2] << 16);
      this._idReadBuffer.unmap();

      const posFloats = new Float32Array(this._posReadBuffer.getMappedRange());
      const worldPosition: Cartesian3 | null =
        id !== 0 && Number.isFinite(posFloats[0])
          ? new Cartesian3(posFloats[0], posFloats[1], posFloats[2])
          : null;
      this._posReadBuffer.unmap();

      const tMapEnd = performance.now();
      this._pending = null;

      const instanceId = decodePickInstanceId(id);
      const feature = instanceId !== null
        ? tilesetPickRegistry.resolve(instanceId)
        : null;
      const result: PickResult = {
        id,
        primitive:
          feature?.primitive ??
          (id !== 0 ? (this._idToPrimitive.get(id) ?? null) : null),
        feature:
          feature ??
          (id !== 0
            ? tilesetPickRegistry.featureForPrimitive(this._idToPrimitive.get(id) ?? null)
            : null),
        worldPosition,
        timing: {
          encodeMs:   job.tEncodeEnd - job.tEncodeStart,
          readbackMs: tMapEnd - tMapStart,
          totalMs:    tMapEnd - job.tEncodeStart,
        },
      };

      this._cache = { x: job.x, y: job.y, camFp: job.camFp, sceneFp: job.sceneFp, result };
      job.resolve(result);
    } catch (err) {
      this._pending = null;
      job.reject(err);
    }
  }
}

export default PickPass;
