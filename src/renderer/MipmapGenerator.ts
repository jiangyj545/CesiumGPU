/**
 * MipmapGenerator - Render-based mip chain generation for WebGPU.
 *
 * WebGPU has no native `generateMipmaps` API, so each mip level (>= 1) is
 * produced by rendering a fullscreen triangle that samples the previous
 * level and blits it into the next one (classic "render-based blit").
 *
 * Requirements on the target texture:
 * - usage must include `RENDER_ATTACHMENT | TEXTURE_BINDING` (plus `COPY_DST`
 *   when level 0 is filled via `queue.copyExternalImageToTexture`);
 * - `mipLevelCount` = floor(log2(max(width, height))) + 1
 *   (see {@link MipmapGenerator.requiredMipLevelCount});
 * - the format must be filterable (e.g. `rgba8unorm`), because the blit
 *   samples with a filtering sampler.
 */

/** WGSL shader: fullscreen triangle vertex stage + simple texture blit. */
const MIPMAP_BLIT_SHADER = /* wgsl */ `
struct VSOutput {
  @builtin(position) position : vec4f,
  @location(0) uv             : vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VSOutput {
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  let p = pos[vertexIndex];
  var out : VSOutput;
  out.position = vec4f(p, 0.0, 1.0);
  out.uv = vec2f((p.x + 1.0) * 0.5, 1.0 - (p.y + 1.0) * 0.5);
  return out;
}

@group(0) @binding(0) var srcTexture : texture_2d<f32>;
@group(0) @binding(1) var srcSampler : sampler;

@fragment
fn fs_main(in : VSOutput) -> @location(0) vec4f {
  return textureSample(srcTexture, srcSampler, in.uv);
}
`;

/**
 * Generates full mip chains for filterable textures via render-to-mip-level
 * blits. Pipelines and samplers are cached per device/format so repeated
 * calls (one per uploaded tile) only pay for command encoding + GPU work.
 */
export class MipmapGenerator {
  /** Cached blit pipelines, keyed per device then per target format. */
  private _pipelines = new Map<GPUDevice, Map<GPUTextureFormat, GPURenderPipeline>>();

  /** Cached filtering samplers, keyed per device. */
  private _samplers = new Map<GPUDevice, GPUSampler>();

  /**
   * Number of mip levels for a texture of the given base dimensions:
   * floor(log2(max(width, height))) + 1.
   */
  static requiredMipLevelCount(width: number, height: number): number {
    return Math.floor(Math.log2(Math.max(width, height))) + 1;
  }

  /**
   * Generate the full mip chain of `texture` (levels 1..mipLevelCount-1).
   * Level 0 is assumed to already contain the image data.
   * A no-op when `texture.mipLevelCount <= 1`.
   */
  generateMips(device: GPUDevice, texture: GPUTexture, format: GPUTextureFormat): void {
    const mipLevelCount = texture.mipLevelCount;
    if (mipLevelCount <= 1) return;

    const pipeline = this._getPipeline(device, format);
    const sampler  = this._getSampler(device);

    const encoder = device.createCommandEncoder({ label: 'MipmapGenerator.Encoder' });
    const width   = texture.width;
    const height  = texture.height;

    for (let level = 1; level < mipLevelCount; level++) {
      const srcView = texture.createView({
        label:         `MipmapGenerator.SrcL${level - 1}`,
        dimension:     '2d',
        baseMipLevel:  level - 1,
        mipLevelCount: 1,
      });
      const dstView = texture.createView({
        label:         `MipmapGenerator.DstL${level}`,
        dimension:     '2d',
        baseMipLevel:  level,
        mipLevelCount: 1,
      });

      const pass = encoder.beginRenderPass({
        label: `MipmapGenerator.BlitL${level}`,
        colorAttachments: [{
          view:       dstView,
          loadOp:     'clear',
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          storeOp:    'store',
        }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: srcView },
          { binding: 1, resource: sampler },
        ],
      }));

      // Non-power-of-two levels floor towards 1×1
      const levelWidth  = Math.max(1, Math.floor(width  / 2 ** level));
      const levelHeight = Math.max(1, Math.floor(height / 2 ** level));
      pass.setViewport(0, 0, levelWidth, levelHeight, 0, 1);
      pass.draw(3);
      pass.end();
    }

    device.queue.submit([encoder.finish()]);
  }

  /** Lazily create (and cache) the blit pipeline for `format`. */
  private _getPipeline(device: GPUDevice, format: GPUTextureFormat): GPURenderPipeline {
    let byFormat = this._pipelines.get(device);
    if (!byFormat) {
      byFormat = new Map<GPUTextureFormat, GPURenderPipeline>();
      this._pipelines.set(device, byFormat);
    }
    const cached = byFormat.get(format);
    if (cached) return cached;

    const module = device.createShaderModule({ code: MIPMAP_BLIT_SHADER, label: 'MipmapBlitShader' });
    const bgl = device.createBindGroupLayout({
      label: 'MipmapGenerator.BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    const pipeline = device.createRenderPipeline({
      label:  `MipmapBlit.${format}`,
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex:   { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
    byFormat.set(format, pipeline);
    return pipeline;
  }

  /** Lazily create (and cache) the filtering sampler used for downsampling. */
  private _getSampler(device: GPUDevice): GPUSampler {
    const cached = this._samplers.get(device);
    if (cached) return cached;
    const sampler = device.createSampler({
      label:        'MipmapGenerator.Sampler',
      minFilter:    'linear',
      magFilter:    'linear',
      mipmapFilter: 'linear',
    });
    this._samplers.set(device, sampler);
    return sampler;
  }
}

export default MipmapGenerator;
