import { Context } from '../../renderer/Context';
import { Frustum } from '../../renderer/Frustum';
import { Camera } from '../Camera';
import { Globe } from '../Globe';
import { PrimitiveCollection } from '../Primitive';
import type { IPass } from './IPass';

export interface GBufferTextures {
  albedo:   GPUTexture;
  normal:   GPUTexture;
  worldPos: GPUTexture;
  depth:    GPUTexture;

  albedoView:   GPUTextureView;
  normalView:   GPUTextureView;
  worldPosView: GPUTextureView;
  depthView:    GPUTextureView;
}

/**
 * Scene-provided inputs the G-Buffer pass rasterises into its render targets.
 * Injected via the constructor (OverlayPass-style) — Scene builds the shared
 * pipelines and camera bind group before creating its passes, so they are
 * fully initialised by the time the pass is constructed.
 */
export interface GBufferDrawInputs {
  /** Base sphere + per-tile imagery primitives (drawn first, when shown). */
  globe: Globe;
  /** Additional opaque primitives (pipeline switched per `prim.unlit`). */
  primitives: PrimitiveCollection;
  /** Shared per-frame camera bind group (bind group 0). */
  cameraBindGroup: GPUBindGroup;
  /** Opaque G-Buffer pipeline. */
  opaquePipeline: GPURenderPipeline;
  /** Unlit (emissive) G-Buffer pipeline variant, or null when unused. */
  unlitPipeline: GPURenderPipeline | null;
  /** Whether execute() frustum-culls the opaque primitives (Scene.debugFrustumCulling). */
  frustumCullingEnabled: () => boolean;
}

/**
 * GBufferPass - Manages the G-Buffer render targets for deferred shading and
 * rasterises all opaque geometry into them.
 *
 * G-Buffer layout:
 *   Attachment 0 – albedo    (rgba8unorm)
 *   Attachment 1 – normal    (rgba16float) world-space normals
 *   Attachment 2 – worldPos  (rgba32float) world-space positions
 *   Depth        – depth32float
 *
 * Implements {@link IPass}: `execute` opens the G-Buffer render pass and draws
 * the globe plus every opaque primitive (switching to the unlit pipeline for
 * emissive geometry).  Downstream passes consume the targets through
 * {@link getDepthView} / `textures` instead of Scene reaching into internals.
 */
export class GBufferPass implements IPass {
  readonly label = 'GBufferPass';

  private _context: Context;
  private _width: number;
  private _height: number;
  private _inputs: GBufferDrawInputs;
  private _camera: Camera | null = null;
  private _textures: GBufferTextures | null = null;
  private _sampler: GPUSampler | null = null;

  constructor(context: Context, width: number, height: number, inputs: GBufferDrawInputs) {
    this._context = context;
    this._width   = width;
    this._height  = height;
    this._inputs  = inputs;
  }

  /** Idempotent: G-Buffer targets and the sampler are built on first use. */
  prepare(): void {
    if (!this._textures) this._create();
    if (!this._sampler) {
      this._sampler = this._context.createSampler({ minFilter: 'nearest', magFilter: 'nearest' });
    }
  }

  get textures(): GBufferTextures {
    if (!this._textures) this._create();
    return this._textures!;
  }

  get sampler(): GPUSampler {
    if (!this._sampler) {
      this._sampler = this._context.createSampler({ minFilter: 'nearest', magFilter: 'nearest' });
    }
    return this._sampler!;
  }

  /** Depth view for downstream depth-tested passes (WBOIT accumulation, overlay/effects sampling). */
  getDepthView(): GPUTextureView {
    return this.textures.depthView;
  }

  /** World-position texture for GPU-pick readback (carries COPY_SRC usage). */
  getWorldPosTexture(): GPUTexture {
    return this.textures.worldPos;
  }

  /** Resize G-Buffer (e.g., on canvas resize). */
  resize(width: number, height: number): void {
    if (width === this._width && height === this._height) return;
    this._width  = width;
    this._height = height;
    this._destroyTextures();
    this._create();
  }

  /**
   * @inheritdoc IPass
   * Keeps the camera reference for frustum culling in {@link execute}.
   */
  updateCamera(camera: Camera): void {
    this._camera = camera;
  }

  /**
   * @inheritdoc IPass
   * `_target` / `_time` are unused — this pass owns its colour attachments;
   * the optional parameters keep the {@link IPass} signature satisfied.
   */
  execute(encoder: GPUCommandEncoder, _target?: GPUTextureView, _time?: number): void {
    const inputs = this._inputs;
    const pass = this._beginRenderPass(encoder);
    pass.setPipeline(inputs.opaquePipeline);

    // Globe (base sphere + per-tile imagery primitives). Not frustum-culled
    // here: the base mesh is a single planet-wide sphere and tile visibility
    // is already decided by the imagery quadtree selection each frame.
    // Lighting off (CesiumJS `globe.enableLighting=false` default): the whole
    // globe surface draws through the unlit pipeline so imagery shows its raw
    // albedo — with the sun light the back-lit hemisphere drops to ambient
    // (~5% brightness) and reads as holes against the space background.
    const gp = inputs.globe.primitive;
    if (gp && inputs.globe.show) {
      const globePipeline =
        inputs.globe.enableLighting || !inputs.unlitPipeline
          ? inputs.opaquePipeline
          : inputs.unlitPipeline;
      pass.setPipeline(globePipeline);
      gp.drawOpaque(pass, inputs.cameraBindGroup);
      inputs.globe.drawTiles(pass, inputs.cameraBindGroup);
      // Restore the lit pipeline for the primitives loop below (it assumes
      // `currentPipeline` starts at the lit pipeline).
      pass.setPipeline(inputs.opaquePipeline);
    }

    // Frustum culling (disable via Scene.debugFrustumCulling). The frustum is
    // rebuilt once per execute from the camera's cached view-projection
    // matrix; primitives whose bounding sphere is entirely outside are skipped.
    const frustum =
      inputs.frustumCullingEnabled() && this._camera
        ? Frustum.fromViewProjection(this._camera.viewProjectionMatrix)
        : null;

    // Other opaque primitives (switch to the unlit pipeline for emissive
    // geometry — bind groups are layout-compatible with both pipelines)
    const opaquePipeline = inputs.opaquePipeline;
    const unlitPipeline  = inputs.unlitPipeline;
    let currentPipeline  = opaquePipeline;
    for (const prim of inputs.primitives.opaque) {
      if (frustum) {
        const bs = prim.boundingSphere;
        if (!frustum.intersectsSphere(bs.center, bs.radius)) continue;
      }
      const wanted = prim.unlit && unlitPipeline ? unlitPipeline : opaquePipeline;
      if (wanted !== currentPipeline) {
        pass.setPipeline(wanted);
        currentPipeline = wanted;
      }
      prim.drawOpaque(pass, inputs.cameraBindGroup);
    }

    pass.end();
  }

  destroy(): void {
    this._destroyTextures();
  }

  // ── Private ────────────────────────────────────────────────────────────────

  private _create(): void {
    const device = this._context.device;
    const w = this._width;
    const h = this._height;

    const makeRT = (format: GPUTextureFormat, label: string, extraUsage: GPUFlagsConstant = 0) => {
      return device.createTexture({
        label,
        size:   { width: w, height: h },
        format,
        usage:  GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | extraUsage,
      });
    };

    const albedo   = makeRT('rgba8unorm',    'GBuffer.Albedo');
    const normal   = makeRT('rgba16float',   'GBuffer.Normal');
    // worldPos carries COPY_SRC so GPU picking can read back 1×1 texels
    // (depth32float cannot: WebGPU only allows full-subresource copies of
    // depth/stencil formats, so a 1×1 depth readback is not possible)
    const worldPos = makeRT('rgba32float',  'GBuffer.WorldPos', GPUTextureUsage.COPY_SRC);
    const depth    = makeRT('depth32float', 'GBuffer.Depth');

    this._textures = {
      albedo, normal, worldPos, depth,
      albedoView:   albedo.createView(),
      normalView:   normal.createView(),
      worldPosView: worldPos.createView(),
      depthView:    depth.createView(),
    };
  }

  /** Build the render pass descriptor for filling the G-Buffer. */
  private _beginRenderPass(encoder: GPUCommandEncoder): GPURenderPassEncoder {
    const tex = this.textures;
    return encoder.beginRenderPass({
      label: 'GBuffer Pass',
      colorAttachments: [
        {
          view: tex.albedoView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp:  'clear',
          storeOp: 'store',
        },
        {
          view: tex.normalView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp:  'clear',
          storeOp: 'store',
        },
        {
          view: tex.worldPosView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp:  'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view:              tex.depthView,
        depthClearValue:   1.0,
        depthLoadOp:       'clear',
        depthStoreOp:      'store',
      },
    });
  }

  private _destroyTextures(): void {
    if (!this._textures) return;
    this._textures.albedo.destroy();
    this._textures.normal.destroy();
    this._textures.worldPos.destroy();
    this._textures.depth.destroy();
    this._textures = null;
  }
}

export default GBufferPass;
