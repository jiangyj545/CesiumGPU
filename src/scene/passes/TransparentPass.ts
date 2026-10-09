import { Context } from '../../renderer/Context';
import { Frustum } from '../../renderer/Frustum';
import { Camera } from '../Camera';
import { GBufferPass } from './GBufferPass';
import { Primitive, PrimitiveCollection } from '../Primitive';
import type { IPass } from './IPass';
import {
  TRANSPARENT_ACCUMULATION_SHADER,
  TRANSPARENT_COMPOSITE_SHADER,
} from '../../shaders/TransparentShader';
import { CAMERA_UBO_SIZE, writeCameraUniforms } from '../../renderer/CameraUniforms';

/**
 * TransparentPass - Weighted Blended Order-Independent Transparency (WBOIT).
 *
 * Two sub-passes, both encapsulated in {@link execute}:
 *  1. Accumulation: transparent objects write weighted colour+alpha into two
 *     render targets (accum + reveal), depth-tested read-only against the
 *     G-Buffer depth.
 *  2. Composite: the accum/reveal textures are blended over the opaque scene.
 *
 * No-ops when the scene has no transparent primitives.
 */
export class TransparentPass implements IPass {
  readonly label = 'TransparentPass';

  private _context: Context;
  private _gbufferPass: GBufferPass;
  private _primitives: PrimitiveCollection;
  private _width: number;
  private _height: number;

  private _accumTexture:  GPUTexture | null = null;
  private _revealTexture: GPUTexture | null = null;
  private _accumView:  GPUTextureView | null = null;
  private _revealView: GPUTextureView | null = null;

  private _accumPipeline:     GPURenderPipeline | null = null;
  /** Same layout, depthCompare 'always' — P2-4 穿深可见透明图元。 */
  private _accumPipelineAlways: GPURenderPipeline | null = null;
  private _compositePipeline: GPURenderPipeline | null = null;

  private _compositeBindGroup: GPUBindGroup | null = null;
  private _nearestSampler: GPUSampler | null = null;

  // Shared camera UBO (written by Scene before each frame)
  readonly cameraUniformBuffer: GPUBuffer;
  private _cameraBindGroupLayout: GPUBindGroupLayout | null = null;

  private _outputFormat: GPUTextureFormat;
  private _depthFormat: GPUTextureFormat = 'depth32float';

  /** Camera reference (set each frame via updateCamera) for frustum culling. */
  private _camera: Camera | null = null;
  /** Whether execute() frustum-culls the transparent primitives. */
  private readonly _frustumCullingEnabled: () => boolean;

  constructor(
    context: Context,
    gbufferPass: GBufferPass,
    primitives: PrimitiveCollection,
    width: number,
    height: number,
    outputFormat: GPUTextureFormat,
    frustumCullingEnabled: () => boolean = () => false
  ) {
    this._context      = context;
    this._gbufferPass  = gbufferPass;
    this._primitives   = primitives;
    this._width        = width;
    this._height       = height;
    this._outputFormat = outputFormat;
    this._frustumCullingEnabled = frustumCullingEnabled;

    this.cameraUniformBuffer = context.device.createBuffer({
      label: 'TransparentPass.CameraUBO',
      size:  CAMERA_UBO_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this._nearestSampler = context.createSampler({ minFilter: 'nearest', magFilter: 'nearest' });
  }

  prepare(): void {
    this._createRenderTargets();
    this._createAccumPipeline();
    this._createCompositePipeline();
  }

  resize(width: number, height: number): void {
    if (width === this._width && height === this._height) return;
    this._width  = width;
    this._height = height;
    this._destroyRenderTargets();
    this._createRenderTargets();
    this._rebuildCompositeBindGroup();
  }

  /**
   * T5 真 HDR：目标格式下发（Scene.render 逐帧调用，构造参 =
   * Scene._sceneColorFormat 单一来源）。格式变更时仅重建 composite 管线
   * （其目标 = 场景颜色输出）；accum/reveal 是内部纹理（rgba16float /
   * r8unorm），accumulation 管线不随场景目标变。同格式幂等早退。
   * rgba16float 为 FP16 可混合格式，WBOIT composite 的 src-alpha 混合合法。
   */
  setTargetFormat(format: GPUTextureFormat): void {
    if (format === this._outputFormat) return;
    this._outputFormat = format;
    this._compositePipeline = null;
    this._createCompositePipeline();
  }

  /**
   * Returns the accumulation render pipeline for use by transparent Primitive objects.
   */
  get accumulationPipeline(): GPURenderPipeline | null {
    return this._accumPipeline;
  }

  get cameraBindGroupLayout(): GPUBindGroupLayout | null {
    return this._cameraBindGroupLayout;
  }

  /**
   * Wire a transparent primitive to this pass: build its per-primitive bind
   * groups against the accumulation pipeline and the shared camera UBO.
   *
   * @returns false when the pass is not prepared yet (no accumulation pipeline).
   */
  attachPrimitive(primitive: Primitive): boolean {
    if (!this._accumPipeline || !this._cameraBindGroupLayout) return false;
    primitive.buildTransparentBindGroups(
      this._accumPipeline,
      this.cameraUniformBuffer,
      this._cameraBindGroupLayout
    );
    return true;
  }

  /**
   * Update the camera UBO (called once per frame by Scene). The camera
   * reference is also kept for frustum culling in {@link execute}.
   */
  updateCamera(camera: Camera): void {
    this._camera = camera;
    const data = new Float32Array(CAMERA_UBO_SIZE / 4);
    // Layout lives in src/renderer/CameraUniforms.ts (single source of truth).
    writeCameraUniforms(data, {
      viewMatrix:           camera.viewMatrix.values,
      projectionMatrix:     camera.projectionMatrix.values,
      viewProjectionMatrix: camera.viewProjectionMatrix.values,
      position:             camera.position,
      viewportWidth:        this._context.drawingBufferWidth,
      viewportHeight:       this._context.drawingBufferHeight,
      near:                 camera.frustum.near,
      far:                  camera.frustum.far,
    });
    this._context.writeBuffer(this.cameraUniformBuffer, 0, data);
  }

  /**
   * @inheritdoc IPass
   *
   * Encapsulates both WBOIT sub-passes: accumulation (clears accum/reveal and
   * draws every transparent primitive, reading the G-Buffer depth read-only)
   * followed by composite (blended over `target`, which carries the lit opaque
   * scene).  No-ops when there is nothing transparent to draw or the pass is
   * not prepared yet.
   * (`_time` is unused — WBOIT is stateless per frame; the optional parameter
   * keeps the {@link IPass} signature satisfied.)
   */
  execute(encoder: GPUCommandEncoder, target: GPUTextureView, _time?: number): void {
    const transparents = this._primitives.transparent;
    if (transparents.length === 0 || !this._accumPipeline) return;

    // Frustum culling (disable via Scene.debugFrustumCulling): skip
    // transparent primitives whose bounding sphere is entirely outside.
    const frustum =
      this._frustumCullingEnabled() && this._camera
        ? Frustum.fromViewProjection(this._camera.viewProjectionMatrix)
        : null;
    const drawn = frustum
      ? transparents.filter((prim) => {
          const bs = prim.boundingSphere;
          return frustum.intersectsSphere(bs.center, bs.radius);
        })
      : transparents;
    if (drawn.length === 0) return;

    this._executeAccumulation(encoder, drawn);
    this._executeComposite(encoder, target);
  }

  destroy(): void {
    this.cameraUniformBuffer.destroy();
    this._destroyRenderTargets();
  }

  // ── Private ────────────────────────────────────────────────────────────────

  /**
   * Accumulation sub-pass: clears accum/reveal targets and draws all
   * transparent primitives, depth-tested read-only against the G-Buffer.
   */
  private _executeAccumulation(
    encoder: GPUCommandEncoder,
    transparents: readonly Primitive[]
  ): void {
    const pass = encoder.beginRenderPass({
      label: 'Transparent Accumulation Pass',
      colorAttachments: [
        {
          view:       this._accumView!,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp:  'clear',
          storeOp: 'store',
        },
        {
          view:       this._revealView!,
          clearValue: { r: 1, g: 0, b: 0, a: 0 }, // r8unorm: only the red channel is used; start at 1.0 (fully visible background)
          loadOp:  'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: this._gbufferPass.getDepthView(),
        depthReadOnly: true,
        // depthLoadOp / depthStoreOp must NOT be set when depthReadOnly is true (WebGPU spec)
      },
    });

    let pipeline: GPURenderPipeline | null = null;
    for (const prim of transparents) {
      // P2-4：alwaysPassDepth 图元走「深度恒过」管线（同布局，bind group 兼容）。
      const target = prim.alwaysPassDepth ? this._accumPipelineAlways : this._accumPipeline;
      if (!target) continue;
      if (pipeline !== target) {
        pass.setPipeline(target);
        pipeline = target;
      }
      prim.drawTransparent(pass);
    }
    pass.end();
  }

  /**
   * Composite sub-pass: blends the transparent layer over the opaque output.
   */
  private _executeComposite(encoder: GPUCommandEncoder, outputView: GPUTextureView): void {
    if (!this._compositePipeline || !this._compositeBindGroup) return;

    const pass = encoder.beginRenderPass({
      label: 'Transparent Composite Pass',
      colorAttachments: [{
        view:    outputView,
        loadOp:  'load',   // blend ON TOP of opaque result
        storeOp: 'store',
      }],
    });

    pass.setPipeline(this._compositePipeline);
    pass.setBindGroup(0, this._compositeBindGroup);
    pass.draw(3);
    pass.end();
  }

  private _createRenderTargets(): void {
    const device = this._context.device;
    const w = this._width, h = this._height;

    this._accumTexture = device.createTexture({
      label:  'Transparent.Accum',
      size:   { width: w, height: h },
      format: 'rgba16float',
      usage:  GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this._revealTexture = device.createTexture({
      label:  'Transparent.Reveal',
      size:   { width: w, height: h },
      format: 'r8unorm',
      usage:  GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this._accumView  = this._accumTexture.createView();
    this._revealView = this._revealTexture.createView();
  }

  private _destroyRenderTargets(): void {
    this._accumTexture?.destroy();
    this._revealTexture?.destroy();
    this._accumTexture  = null;
    this._revealTexture = null;
    this._accumView  = null;
    this._revealView = null;
  }

  private _createAccumPipeline(): void {
    const device = this._context.device;
    const shader = device.createShaderModule({ code: TRANSPARENT_ACCUMULATION_SHADER, label: 'TransparentAccumShader' });

    this._cameraBindGroupLayout = device.createBindGroupLayout({
      label: 'Transparent.BGL0',
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });

    const modelBGL = device.createBindGroupLayout({
      label: 'Transparent.BGL1',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    const descriptor: GPURenderPipelineDescriptor = {
      label:  'TransparentAccumPipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this._cameraBindGroupLayout, modelBGL] }),
      vertex: {
        module:     shader,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 32, // 8 floats × 4 bytes
          attributes: [
            { shaderLocation: 0, offset: 0,  format: 'float32x3' }, // position
            { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
            { shaderLocation: 2, offset: 24, format: 'float32x2' }, // uv
          ],
        }],
      },
      fragment: {
        module:     shader,
        entryPoint: 'fs_accum',
        targets: [
          // Accum: additive blending
          {
            format: 'rgba16float',
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            },
          },
          // Reveal: multiplicative (product of (1-alpha))
          {
            format: 'r8unorm',
            blend: {
              color: { srcFactor: 'zero', dstFactor: 'one-minus-src', operation: 'add' },
              alpha: { srcFactor: 'zero', dstFactor: 'one-minus-src', operation: 'add' },
            },
            writeMask: GPUColorWrite.RED,
          },
        ],
      },
      primitive:    { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: this._depthFormat, depthWriteEnabled: false, depthCompare: 'less' },
    };

    this._accumPipeline = device.createRenderPipeline(descriptor);
    this._accumPipelineAlways = device.createRenderPipeline({
      ...descriptor,
      label: 'TransparentAccumPipelineAlways',
      depthStencil: { format: this._depthFormat, depthWriteEnabled: false, depthCompare: 'always' },
    });
  }

  private _createCompositePipeline(): void {
    const device = this._context.device;
    const shader = device.createShaderModule({ code: TRANSPARENT_COMPOSITE_SHADER, label: 'TransparentCompositeShader' });

    const bgl = device.createBindGroupLayout({
      label: 'Composite.BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } },
      ],
    });

    this._compositePipeline = device.createRenderPipeline({
      label:  'TransparentCompositePipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex:   { module: shader, entryPoint: 'vs_main' },
      fragment: {
        module:     shader,
        entryPoint: 'fs_composite',
        targets: [{
          format: this._outputFormat,
          blend: {
            // src = transparent colour; dst = opaque scene
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one',       dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });

    this._rebuildCompositeBindGroup();
  }

  private _rebuildCompositeBindGroup(): void {
    if (!this._compositePipeline || !this._accumView || !this._revealView) return;
    const device = this._context.device;
    this._compositeBindGroup = device.createBindGroup({
      label:  'Composite.BG',
      layout: this._compositePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._accumView },
        { binding: 1, resource: this._revealView },
        { binding: 2, resource: this._nearestSampler! },
      ],
    });
  }
}

export default TransparentPass;
