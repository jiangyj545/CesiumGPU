import { Context } from '../../renderer/Context';

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
 * GBufferPass - Manages G-Buffer render targets for deferred shading.
 *
 * G-Buffer layout:
 *   Attachment 0 – albedo    (rgba8unorm)
 *   Attachment 1 – normal    (rgba16float) world-space normals
 *   Attachment 2 – worldPos  (rgba32float) world-space positions
 *   Depth        – depth32float
 *
 * @future IPass — 资源持有者（`beginRenderPass` 由 Scene.render() 驱动，无独立
 * execute），暂不实现 IPass；待渲染编排重构后渐进迁移。
 */
export class GBufferPass {
  private _context: Context;
  private _width: number;
  private _height: number;
  private _textures: GBufferTextures | null = null;
  private _sampler: GPUSampler | null = null;

  constructor(context: Context, width: number, height: number) {
    this._context = context;
    this._width   = width;
    this._height  = height;
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

  /** Resize G-Buffer (e.g., on canvas resize). */
  resize(width: number, height: number): void {
    if (width === this._width && height === this._height) return;
    this._width  = width;
    this._height = height;
    this._destroyTextures();
    this._create();
  }

  /** Build the render pass descriptor for filling the G-Buffer. */
  beginRenderPass(encoder: GPUCommandEncoder): GPURenderPassEncoder {
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
