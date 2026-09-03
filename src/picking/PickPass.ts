import { Scene } from '../scene/Scene';
import { Primitive } from '../scene/Primitive';
import { Cartesian3 } from '../math/Cartesian3';
import { PICK_SHADER, PICK_ID_TEST_SHADER } from '../shaders/PickShader';

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
 * tiles and `scene.primitives`) is automatically pickable: each `Primitive`
 * owns a 24-bit `pickId` assigned at construction. When a pick is pending,
 * the pass re-draws that same set into a canvas-sized rgba8unorm "pick view"
 * using the PickShader, appends 1×1 copies of the pick texture and the
 * G-Buffer worldPos texture to the frame encoder (via `scene.frameHooks`),
 * and decodes them after submit with `mapAsync`.
 *
 * Guarantees:
 *  - at most one readback per rendered frame (concurrent `pick()` calls
 *    share the in-flight promise);
 *  - identical set/order/state as the G-Buffer pass, so pick results match
 *    the visible image;
 *  - the last result is cached and served from cache until the camera or
 *    the scene content changes.
 *
 * Coordinates: `pick(x, y)` takes canvas device-pixel coordinates with a
 * top-left origin — the same convention as DOM `clientX/clientY` scaled to
 * the canvas AND WebGPU texture texels, so no y-flip is required.
 */
export class PickPass {
  private _scene: Scene;

  /** Primitives manually registered in addition to the scene graph. */
  private readonly _registered = new Set<Primitive>();

  /** Lazily created id UBO / bind group per pick id. */
  private readonly _idUbos = new Map<number, GPUBuffer>();
  private readonly _idBindGroups = new Map<number, GPUBindGroup>();

  private _pipeline: GPURenderPipeline | null = null;
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

  /** Registered once into `scene.frameHooks`; encodes the pick view. */
  private readonly _frameHook: (encoder: GPUCommandEncoder) => void;

  constructor(scene: Scene) {
    this._scene = scene;

    const device = scene.context.device;

    this._cameraUbo = device.createBuffer({
      label: 'PickPass.CameraUBO',
      size:  256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bufferUsage = GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ;
    this._idReadBuffer = device.createBuffer({
      label: 'PickPass.IdReadback',       size: READBACK_BYTES_PER_ROW, usage: bufferUsage,
    });
    this._posReadBuffer = device.createBuffer({
      label: 'PickPass.WorldPosReadback', size: READBACK_BYTES_PER_ROW, usage: bufferUsage,
    });

    this._frameHook = (encoder: GPUCommandEncoder) => {
      this.debug.hookCalls += 1;
      this._encodePickFrame(encoder);
    };
    scene.frameHooks.push(this._frameHook);

    device.addEventListener('uncapturederror', (event) => {
      this.debug.gpuErrors += 1;
      console.error('[PickPass] GPU uncaptured error:', (event as GPUUncapturedErrorEvent).error.message);
    });
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
   * Request a pick at canvas device-pixel coordinates (top-left origin).
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
    if (!this._scene.opaquePipeline) return Promise.resolve(null);

    const w = Math.max(1, this._scene.context.drawingBufferWidth);
    const h = Math.max(1, this._scene.context.drawingBufferHeight);
    const cx = Math.min(Math.max(0, Math.round(x)), w - 1);
    const cy = Math.min(Math.max(0, Math.round(y)), h - 1);

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
    const device = this._scene.context.device;

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
    this._scene.context.writeBuffer(ubo, 0, new Uint32Array([testId >>> 0, 0, 0, 0]));
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
    this._scene.context.submit([encoder.finish()]);

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

  destroy(): void {
    const idx = this._scene.frameHooks.indexOf(this._frameHook);
    if (idx !== -1) this._scene.frameHooks.splice(idx, 1);

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
    const cam = this._scene.camera;
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

  /** Change-detector for the cached pick: pickable content. */
  private _sceneFingerprint(): number {
    return (
      this._scene.primitives.length * 2654435761 +
      this._scene.globe.tilePrimitiveCount * 40503 +
      this._registered.size * 104729
    );
  }

  /** Ensure the pick pipeline exists (needs scene.opaquePipeline for BGLs). */
  private _ensurePipeline(): void {
    if (this._pipeline) return;
    const opaque = this._scene.opaquePipeline;
    if (!opaque) throw new Error('PickPass: scene not initialized yet');

    const device   = this._scene.context.device;
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
  }

  /** Lazily create (and cache) the per-id uniform + bind group. */
  private _idBindGroup(pickId: number): GPUBindGroup {
    let bg = this._idBindGroups.get(pickId);
    if (bg) return bg;

    const device = this._scene.context.device;
    let ubo = this._idUbos.get(pickId);
    if (!ubo) {
      ubo = device.createBuffer({
        label: `PickPass.IdUBO#${pickId}`,
        size:  256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this._scene.context.writeBuffer(ubo, 0, new Uint32Array([pickId, 0, 0, 0]));
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
    const ctx = this._scene.context;
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

  /** Upload camera uniforms (same 256-byte layout as Scene._uploadCameraUBO). */
  private _uploadCamera(): void {
    const cam  = this._scene.camera;
    const data = new Float32Array(64); // 256 bytes / 4

    data.set(cam.viewMatrix.values,            0);
    data.set(cam.projectionMatrix.values,     16);
    data.set(cam.viewProjectionMatrix.values, 32);
    data[48] = cam.position.x;
    data[49] = cam.position.y;
    data[50] = cam.position.z;
    data[51] = 1.0;
    data[52] = this._targetWidth;
    data[53] = this._targetHeight;
    data[54] = cam.frustum.near;
    data[55] = cam.frustum.far;

    this._scene.context.writeBuffer(this._cameraUbo, 0, data);
  }

  /**
   * Draw the exact same opaque set as Scene.render()'s G-Buffer pass:
   * globe base mesh → imagery tiles → scene.primitives (opaque) → manual.
   */
  private _drawPickSet(pass: GPURenderPassEncoder, cameraBindGroup: GPUBindGroup): void {
    this._idToPrimitive = new Map<number, Primitive>();

    const drawWithId = (prim: Primitive): void => {
      if (!prim.show) return;
      this._idToPrimitive.set(prim.pickId, prim);
      pass.setBindGroup(2, this._idBindGroup(prim.pickId));
      prim.drawOpaque(pass, cameraBindGroup);
    };

    const globe = this._scene.globe;
    if (globe.show) {
      const base = globe.primitive;
      if (base) {
        drawWithId(base);
        for (const tile of globe.tilePrimitives) drawWithId(tile);
      }
    }

    for (const prim of this._scene.primitives.opaque) drawWithId(prim);
    for (const prim of this._registered) drawWithId(prim);
  }

  /**
   * Frame hook body: append the pick-view render pass + 1×1 copies to the
   * shared frame encoder. Runs before Scene.submit(); mapping is deferred to
   * a microtask so it strictly follows the submit.
   */
  private _encodePickFrame(encoder: GPUCommandEncoder): void {
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
      this._drawPickSet(pass, this._cameraBindGroup);
      pass.end();

      const origin   = { x: job.x, y: job.y, z: 0 };
      const copySize = { width: 1, height: 1 };
      encoder.copyTextureToBuffer(
        { texture: this._pickTexture, origin },
        { buffer: this._idReadBuffer, bytesPerRow: READBACK_BYTES_PER_ROW },
        copySize,
      );
      const gbuffer = this._scene.gbufferPass?.textures;
      if (gbuffer) {
        encoder.copyTextureToBuffer(
          { texture: gbuffer.worldPos, origin },
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

      const result: PickResult = {
        id,
        primitive: id !== 0 ? (this._idToPrimitive.get(id) ?? null) : null,
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
