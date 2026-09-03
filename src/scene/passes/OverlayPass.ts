import { Context } from '../../renderer/Context';
import { Camera } from '../Camera';
import { GBufferPass } from './GBufferPass';
import type { IPass } from './IPass';
import { Color } from '../../math/Color';
import { EntityCollection } from '../../dataSource/EntityCollection';
import { OVERLAY_SHADER } from '../../shaders/OverlayShader';

/** Maximum number of billboards (points + labels) drawn per frame. */
const MAX_QUADS = 4096;
/** Floats per vertex: pixelPos(2) + uv(2) + color(4) + depth(1). */
const FLOATS_PER_VERTEX = 9;
const BYTES_PER_VERTEX  = FLOATS_PER_VERTEX * 4;
const VERTICES_PER_QUAD = 6;

/** Baked label font (canvas pixels == screen pixels at devicePixelRatio 1). */
const LABEL_FONT = '600 24px "Microsoft YaHei", "PingFang SC", Arial, sans-serif';
const LABEL_PAD_X = 8;
const LABEL_PAD_Y = 4;
/** Vertical gap between the projected anchor and the label bottom edge (px). */
const LABEL_LIFT_PX = 12;
/** LRU capacity for baked label textures. */
const LABEL_CACHE_LIMIT = 128;
/**
 * Occlusion bias in normalised depth: billboards whose anchor depth is within
 * this margin of the G-Buffer depth are considered ON the surface and drawn.
 * 1e-4 ≈ a few hundred metres of surface offset at globe viewing distances.
 */
const DEPTH_BIAS = 1e-4;

interface LabelEntry {
  texture: GPUTexture;
  /** Baked size in pixels (== quad size on screen). */
  width: number;
  height: number;
}

interface DrawSegment {
  pipeline: GPURenderPipeline;
  bindGroup: GPUBindGroup;
  firstVertex: number;
  vertexCount: number;
}

/**
 * OverlayPass - Screen-space overlay for entity billboards (points, labels).
 *
 * Unlike {@link Primitive}s these visuals keep a constant pixel size on
 * screen, so they cannot live in the G-Buffer.  Every frame
 * {@link updateCamera} projects each entity anchor (world → clip → pixels)
 * on the CPU, culls anchors behind the camera (w <= 0) or off-screen, and
 * appends a camera-facing quad to a dynamic vertex buffer.
 * {@link execute} then draws the quads straight onto the swap-chain (after
 * Lighting, before the Transparent composite) blending over the lit scene.
 *
 * Depth-vs-globe occlusion is resolved in the fragment stage by loading the
 * G-Buffer depth texture (standard projection depth, 0..1) and discarding
 * fragments where the opaque surface is closer than the billboard anchor.
 */
export class OverlayPass implements IPass {
  readonly label = 'OverlayPass';

  private _context: Context;
  private _entities: EntityCollection;
  private _gbufferPass: GBufferPass;
  private _format: GPUTextureFormat;

  private _width  = 1;
  private _height = 1;

  private _pipelinePoint: GPURenderPipeline | null = null;
  private _pipelineLabel: GPURenderPipeline | null = null;
  private _cameraBGL: GPUBindGroupLayout | null = null;
  private _textureBGL: GPUBindGroupLayout | null = null;

  private _cameraUBO: GPUBuffer | null = null;
  private _cameraBindGroup: GPUBindGroup | null = null;
  private _vertexBuffer: GPUBuffer | null = null;
  private readonly _vertexCapacity = MAX_QUADS * VERTICES_PER_QUAD * BYTES_PER_VERTEX;

  /** 1×1 white texture bound for point segments (label binding is unused). */
  private _whiteTexture: GPUTexture | null = null;
  private _labelSampler: GPUSampler | null = null;

  /** LRU cache of baked label textures, keyed by unique text. */
  private _labelCache = new Map<string, LabelEntry>();
  private _labelCanvas: HTMLCanvasElement | null = null;

  /** CPU-built quad data for the current frame. */
  private readonly _scratch = new Float32Array(MAX_QUADS * VERTICES_PER_QUAD * FLOATS_PER_VERTEX);
  private _quadCount = 0;
  private _segments: DrawSegment[] = [];

  /** Depth view the bind-group cache was built against (invalidated on resize). */
  private _cachedDepthView: GPUTextureView | null = null;
  private _bindGroupCache = new Map<GPUTexture, GPUBindGroup>();

  constructor(
    context: Context,
    entities: EntityCollection,
    gbufferPass: GBufferPass,
    format: GPUTextureFormat
  ) {
    this._context     = context;
    this._entities    = entities;
    this._gbufferPass = gbufferPass;
    this._format      = format;
  }

  /**
   * Create all GPU resources (pipelines, buffers, sampler, bind groups).
   * Call once after context.initialize().
   */
  prepare(): void {
    const device = this._context.device;
    const module = device.createShaderModule({ code: OVERLAY_SHADER, label: 'OverlayShader' });

    this._cameraBGL = device.createBindGroupLayout({
      label: 'Overlay.BGL0',
      // NOTE: VERTEX | FRAGMENT — Chrome validates every entry point in the
      // module against group 0, so a vertex-only visibility fails pipeline
      // validation even though only vs_main statically uses the uniform.
      entries: [{
        binding:    0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer:     { type: 'uniform' },
      }],
    });

    this._textureBGL = device.createBindGroupLayout({
      label: 'Overlay.BGL1',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }, // G-Buffer depth
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },              // label texture
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler:  { type: 'filtering' } },
      ],
    });

    const layout = device.createPipelineLayout({ bindGroupLayouts: [this._cameraBGL, this._textureBGL] });

    const vertexBuffers: GPUVertexBufferLayout[] = [{
      arrayStride: BYTES_PER_VERTEX,
      attributes: [
        { shaderLocation: 0, offset: 0,  format: 'float32x2' }, // pixel position
        { shaderLocation: 1, offset: 8,  format: 'float32x2' }, // uv
        { shaderLocation: 2, offset: 16, format: 'float32x4' }, // tint colour
        { shaderLocation: 3, offset: 32, format: 'float32'   }, // anchor depth (z/w)
      ],
    }];

    const blend: GPUBlendState = {
      color: {
        srcFactor: 'src-alpha',
        dstFactor: 'one-minus-src-alpha',
        operation: 'add',
      },
      alpha: {
        srcFactor: 'one',
        dstFactor: 'one-minus-src-alpha',
        operation: 'add',
      },
    };

    const common = {
      layout,
      vertex:    { module, entryPoint: 'vs_main', buffers: vertexBuffers },
      primitive: { topology: 'triangle-list' as GPUPrimitiveTopology, cullMode: 'none' as GPUCullMode },
    };

    this._pipelinePoint = device.createRenderPipeline({
      ...common,
      label: 'OverlayPointPipeline',
      fragment: {
        module,
        entryPoint: 'fs_point',
        targets: [{ format: this._format, blend }],
      },
    });

    this._pipelineLabel = device.createRenderPipeline({
      ...common,
      label: 'OverlayLabelPipeline',
      fragment: {
        module,
        entryPoint: 'fs_label',
        targets: [{ format: this._format, blend }],
      },
    });

    this._cameraUBO = device.createBuffer({
      label: 'Overlay.CameraUBO',
      size:  80, // viewProj(64) + viewport(8) + depthBias(4) + pad(4)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this._cameraBindGroup = device.createBindGroup({
      label:  'Overlay.CameraBG',
      layout: this._cameraBGL,
      entries: [{ binding: 0, resource: { buffer: this._cameraUBO } }],
    });

    this._vertexBuffer = device.createBuffer({
      label: 'Overlay.VBO',
      size:  this._vertexCapacity,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    this._labelSampler = this._context.createSampler({
      minFilter: 'linear',
      magFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    this._whiteTexture = device.createTexture({
      label:  'Overlay.White',
      size:   { width: 1, height: 1 },
      format: 'rgba8unorm',
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: this._whiteTexture },
      new Uint8Array([255, 255, 255, 255]),
      { bytesPerRow: 4 },
      { width: 1, height: 1 }
    );
  }

  /**
   * Per-frame CPU stage: upload camera uniforms, project every visible
   * point/label entity to pixel space and fill the dynamic vertex buffer.
   * Must be called once per frame before {@link execute}.
   */
  updateCamera(camera: Camera): void {
    if (!this._cameraUBO || !this._cameraBindGroup || !this._vertexBuffer) return;

    this._width  = this._context.drawingBufferWidth  || 1;
    this._height = this._context.drawingBufferHeight || 1;

    const uniform = new Float32Array(20); // 80 bytes
    uniform.set(camera.viewProjectionMatrix.values, 0);
    uniform[16] = this._width;
    uniform[17] = this._height;
    uniform[18] = DEPTH_BIAS;
    this._context.writeBuffer(this._cameraUBO, 0, uniform);

    this._quadCount = 0;
    this._segments  = [];
    // Contiguous run of point quads sharing one bind group → drawn in one call.
    let pointSegment: DrawSegment | null = null;

    const vp = camera.viewProjectionMatrix.values;

    for (const entity of this._entities.values) {
      if (!entity.show || !entity.position) continue;

      // Anchor projection: world → clip → pixels (column-major mat4; NDC y
      // flips to top-left-origin screen y).
      const p  = entity.position;
      const x  = p.x, y = p.y, z = p.z;
      const cw = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
      if (cw <= 1e-6) continue; // behind the camera (w <= 0)
      const sx = (vp[0] * x + vp[4] * y + vp[8]  * z + vp[12]) / cw;
      const sy = (vp[1] * x + vp[5] * y + vp[9]  * z + vp[13]) / cw;
      const sz = (vp[2] * x + vp[6] * y + vp[10] * z + vp[14]) / cw;
      if (sz <= 0 || sz >= 1) continue; // near/far clipped
      const px = (sx * 0.5 + 0.5) * this._width;
      const py = (0.5 - sy * 0.5) * this._height;
      const depth = sz;

      const point = entity.point;
      if (point && point.pixelSize && point.pixelSize > 0) {
        const bg = this._getBindGroup(this._whiteTexture!);
        if (bg && this._appendQuad(px, py, point.pixelSize, point.color ?? Color.WHITE, depth)) {
          if (pointSegment && pointSegment.firstVertex + pointSegment.vertexCount === (this._quadCount - 1) * VERTICES_PER_QUAD) {
            pointSegment.vertexCount += VERTICES_PER_QUAD;
          } else {
            pointSegment = {
              pipeline: this._pipelinePoint!,
              bindGroup: bg,
              firstVertex: (this._quadCount - 1) * VERTICES_PER_QUAD,
              vertexCount: VERTICES_PER_QUAD,
            };
            this._segments.push(pointSegment);
          }
        }
      }

      const label = entity.label;
      if (label && label.text && this._quadCount < MAX_QUADS) {
        const entry = this._getOrCreateLabel(label.text);
        const bg    = this._getBindGroup(entry.texture);
        if (bg && this._appendSizedQuad(
          px - entry.width / 2,
          py - LABEL_LIFT_PX - entry.height,
          entry.width,
          entry.height,
          label.fillColor ?? Color.WHITE,
          depth
        )) {
          // Each label texture owns its segment (own bind group).
          this._segments.push({
            pipeline: this._pipelineLabel!,
            bindGroup: bg,
            firstVertex: (this._quadCount - 1) * VERTICES_PER_QUAD,
            vertexCount: VERTICES_PER_QUAD,
          });
        }
      }
    }

    if (this._quadCount > 0) {
      const view = this._scratch.subarray(0, this._quadCount * VERTICES_PER_QUAD * FLOATS_PER_VERTEX);
      this._context.writeBuffer(this._vertexBuffer, 0, view);
    }
  }

  /**
   * Draw the overlay onto `colorAttachmentView` (the swap-chain).
   * Uses `loadOp: 'load'` so it blends over the lit opaque scene.
   * (`_time` is unused — overlay billboards are stateless per frame; the
   * optional parameter keeps the {@link IPass} signature satisfied while
   * preserving the two-argument call sites in Scene.render.)
   */
  execute(encoder: GPUCommandEncoder, colorAttachmentView: GPUTextureView, _time?: number): void {
    if (this._quadCount === 0 || !this._cameraBindGroup || !this._vertexBuffer) return;

    const pass = encoder.beginRenderPass({
      label: 'Overlay Pass',
      colorAttachments: [{
        view:    colorAttachmentView,
        loadOp:  'load',
        storeOp: 'store',
      }],
    });

    pass.setVertexBuffer(0, this._vertexBuffer);
    pass.setBindGroup(0, this._cameraBindGroup);

    let currentPipeline: GPURenderPipeline | null = null;
    for (const segment of this._segments) {
      if (segment.pipeline !== currentPipeline) {
        pass.setPipeline(segment.pipeline);
        currentPipeline = segment.pipeline;
      }
      pass.setBindGroup(1, segment.bindGroup);
      pass.draw(segment.vertexCount, 1, segment.firstVertex);
    }

    pass.end();
  }

  /** Handle canvas resize (viewport uniform is refreshed every frame anyway). */
  resize(width: number, height: number): void {
    this._width  = width;
    this._height = height;
  }

  destroy(): void {
    for (const entry of this._labelCache.values()) entry.texture.destroy();
    this._labelCache.clear();
    this._bindGroupCache.clear();
    this._vertexBuffer?.destroy();
    this._cameraUBO?.destroy();
    this._whiteTexture?.destroy();
    this._quadCount = 0;
    this._segments  = [];
  }

  // ── Private ─────────────────────────────────────────────────────────────────

  /**
   * Append one axis-aligned quad centred at pixel (px, py) to the CPU scratch
   * buffer.  Corner order (screen space, y down): TL, BL, BR / TL, BR, TR.
   */
  private _appendQuad(
    px: number, py: number, size: number,
    color: Color, depth: number
  ): boolean {
    if (this._quadCount >= MAX_QUADS) return false;

    const half = size * 0.5;
    const left = px - half, top = py - half;
    if (left > this._width || top > this._height || left + size < 0 || top + size < 0) return false;

    const right  = left + size;
    const bottom = top + size;
    const r = color.red, g = color.green, b = color.blue, a = color.alpha;
    const base = this._quadCount * VERTICES_PER_QUAD * FLOATS_PER_VERTEX;
    const s    = this._scratch;

    //             px            py             u              v
    s[base]      = left;  s[base + 1]  = top;    s[base + 2] = 0; s[base + 3] = 0;
    s[base + 9]  = left;  s[base + 10] = bottom; s[base + 11] = 0; s[base + 12] = 1;
    s[base + 18] = right; s[base + 19] = bottom; s[base + 20] = 1; s[base + 21] = 1;
    s[base + 27] = left;  s[base + 28] = top;    s[base + 29] = 0; s[base + 30] = 0;
    s[base + 36] = right; s[base + 37] = bottom; s[base + 38] = 1; s[base + 39] = 1;
    s[base + 45] = right; s[base + 46] = top;    s[base + 47] = 1; s[base + 48] = 0;

    for (let v = 0; v < VERTICES_PER_QUAD; v++) {
      const o = base + v * FLOATS_PER_VERTEX;
      s[o + 4] = r;
      s[o + 5] = g;
      s[o + 6] = b;
      s[o + 7] = a;
      s[o + 8] = depth;
    }

    this._quadCount++;
    return true;
  }

  /**
   * Append a sized (w×h) quad whose top-left corner is at (left, top).
   * Used for labels, which have non-square baked dimensions.
   */
  private _appendSizedQuad(
    left: number, top: number, width: number, height: number,
    color: Color, depth: number
  ): boolean {
    if (this._quadCount >= MAX_QUADS) return false;
    if (left > this._width || top > this._height || left + width < 0 || top + height < 0) return false;

    const right  = left + width;
    const bottom = top + height;
    const r = color.red, g = color.green, b = color.blue, a = color.alpha;
    const base = this._quadCount * VERTICES_PER_QUAD * FLOATS_PER_VERTEX;
    const s    = this._scratch;

    s[base]      = left;  s[base + 1]  = top;    s[base + 2] = 0; s[base + 3] = 0;
    s[base + 9]  = left;  s[base + 10] = bottom; s[base + 11] = 0; s[base + 12] = 1;
    s[base + 18] = right; s[base + 19] = bottom; s[base + 20] = 1; s[base + 21] = 1;
    s[base + 27] = left;  s[base + 28] = top;    s[base + 29] = 0; s[base + 30] = 0;
    s[base + 36] = right; s[base + 37] = bottom; s[base + 38] = 1; s[base + 39] = 1;
    s[base + 45] = right; s[base + 46] = top;    s[base + 47] = 1; s[base + 48] = 0;

    for (let v = 0; v < VERTICES_PER_QUAD; v++) {
      const o = base + v * FLOATS_PER_VERTEX;
      s[o + 4] = r;
      s[o + 5] = g;
      s[o + 6] = b;
      s[o + 7] = a;
      s[o + 8] = depth;
    }

    this._quadCount++;
    return true;
  }

  /** Bind group for a label (or white) texture, sampling the current G-Buffer depth. */
  private _getBindGroup(texture: GPUTexture): GPUBindGroup | null {
    const depthView = this._gbufferPass.textures.depthView;
    if (depthView !== this._cachedDepthView) {
      // G-Buffer resized: cached bind groups reference the dead depth view.
      this._bindGroupCache.clear();
      this._cachedDepthView = depthView;
    }

    const cached = this._bindGroupCache.get(texture);
    if (cached) return cached;

    if (!this._textureBGL || !this._labelSampler) return null;

    const group = this._context.device.createBindGroup({
      label: 'Overlay.TextureBG',
      layout: this._textureBGL,
      entries: [
        { binding: 0, resource: depthView },
        { binding: 1, resource: texture.createView() },
        { binding: 2, resource: this._labelSampler },
      ],
    });
    this._bindGroupCache.set(texture, group);
    return group;
  }

  /** Bake `text` into a GPU texture (Canvas2D), LRU-capped at 128 entries. */
  private _getOrCreateLabel(text: string): LabelEntry {
    const existing = this._labelCache.get(text);
    if (existing) {
      // LRU touch: re-insert to move to the most-recently-used end.
      this._labelCache.delete(text);
      this._labelCache.set(text, existing);
      return existing;
    }

    if (!this._labelCanvas) this._labelCanvas = document.createElement('canvas');
    const canvas = this._labelCanvas;
    const ctx2d  = canvas.getContext('2d');
    ctx2d?.clearRect(0, 0, canvas.width, canvas.height);

    let width  = 1;
    let height = 32;
    if (ctx2d) {
      ctx2d.font = LABEL_FONT;
      const metrics = ctx2d.measureText(text);
      width  = Math.max(1, Math.ceil(metrics.width) + LABEL_PAD_X * 2);
      height = Math.max(1, Math.ceil(metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent) + LABEL_PAD_Y * 2);
    }
    canvas.width  = width;
    canvas.height = height;

    if (ctx2d) {
      ctx2d.font      = LABEL_FONT;
      ctx2d.textAlign = 'center';
      ctx2d.textBaseline = 'middle';
      ctx2d.lineJoin  = 'round';
      // Dark outline for contrast against bright imagery, then white fill
      // (the entity fill colour is applied as the shader tint).
      ctx2d.strokeStyle = 'rgba(0, 0, 0, 0.85)';
      ctx2d.lineWidth   = 4;
      ctx2d.strokeText(text, width / 2, height / 2);
      ctx2d.fillStyle   = '#ffffff';
      ctx2d.fillText(text, width / 2, height / 2);
    }

    const device  = this._context.device;
    const texture = device.createTexture({
      label: `Overlay.Label[${text.slice(0, 16)}]`,
      size:  { width, height },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture(
      { source: canvas },
      { texture },
      { width, height }
    );

    const entry: LabelEntry = { texture, width, height };
    this._labelCache.set(text, entry);

    // LRU eviction.
    while (this._labelCache.size > LABEL_CACHE_LIMIT) {
      const oldestKey = this._labelCache.keys().next().value as string | undefined;
      if (oldestKey === undefined) break;
      const oldest = this._labelCache.get(oldestKey);
      if (oldest) {
        this._bindGroupCache.delete(oldest.texture);
        oldest.texture.destroy();
      }
      this._labelCache.delete(oldestKey);
    }
    return entry;
  }
}

export default OverlayPass;
