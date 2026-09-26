import { Context } from '../renderer/Context';
import { MipmapGenerator } from '../renderer/MipmapGenerator';
import { createFoamPatternShaderCode } from '../shaders/FoamPatternShader';

/**
 * FoamPatternTexture - 可平铺程序泡沫纹理生成器（一次性 GPU 资源，非 IPass）。
 *
 * 构造后调用一次 {@link generate}：compute kernel 写 rgba16float 纹理的
 * level 0，再由 {@link MipmapGenerator} 补齐 mip 链。产出的纹理供水面
 * shader 以 `foam coverage` 阈值化 R 通道采样（repeat + linear，支持 aniso）。
 * 通道语义（见 FoamPatternShader）：
 *   R 泡沫密度场 / G 细泡细节 / B 大尺度斑驳 / A 条纹。
 *
 * mip 链方案：**复用引擎既有 `MipmapGenerator`**（render-blit 双线性逐级
 * 缩采样），而非自写 compute mip kernel。理由：
 * 1. WebGPU 没有"device 原生 generateMipmaps"API，任何方案都要自己写；
 * 2. MipmapGenerator 是项目验证过的现成路径（ImageryLayer 同款），管线按
 *    device+format 缓存，rgba16float 属核心可过滤格式（16f 过滤无需扩展名）；
 * 3. 泡沫纹理只生成一次，双线性降采样的 mip 质量完全够用，不值得为此
 *    增加第二套 kernel 与验证成本。
 * 要求纹理 usage 含 RENDER_ATTACHMENT（blit 目标），compute 写入用
 * STORAGE_BINDING，采样用 TEXTURE_BINDING。
 *
 * 幂等性：`generate()` 重复调用为空操作；`destroy()` 后所有 getter 返回
 * null，实例不可再用。
 */
export class FoamPatternTexture {
  /** 纹理边长（像素，2 的幂）。 */
  readonly size: number;

  private readonly _context: Context;
  private readonly _seed: number;
  private readonly _mipmapGenerator = new MipmapGenerator();
  private _texture: GPUTexture | null = null;
  private _view: GPUTextureView | null = null;
  private _sampler: GPUSampler | null = null;
  private _destroyed = false;

  constructor(context: Context, options: FoamPatternTextureOptions = {}) {
    if (!context.initialized) {
      throw new Error('FoamPatternTexture: Context.initialize() must complete first.');
    }

    const size = options.size ?? 512;
    // 2 的幂且 ≥ 8（workgroup 8x8 整除 + mip 链下采样到底）。
    if (!Number.isInteger(size) || size < 8 || (size & (size - 1)) !== 0) {
      throw new Error(
        `FoamPatternTexture: size must be a power-of-two integer >= 8, got ${size}.`,
      );
    }

    this._context = context;
    this.size = size;
    this._seed = Math.round(options.seed ?? 1337);
  }

  /**
   * 一次性生成：创建纹理 → compute 写 level 0 → 补齐 mip 链。幂等。
   *
   * 提交顺序即 GPU 执行顺序：compute encoder 先 submit，MipmapGenerator
   * 自建的 mip encoder 后 submit，level 0 就绪后才被逐级降采样。
   */
  generate(): void {
    if (this._destroyed || this._texture) return;

    const device = this._context.device;

    // seed → 正整数 uv 域偏移：周期化噪声下整数偏移 = 格点相位重排，
    // 任意 seed 均严格保持纹理可平铺（见 FoamPatternShader 头注释）。
    const s = Math.abs(this._seed);
    const seedX = s % 997;
    const seedY = (s * 31 + 17) % 449;

    const texture = device.createTexture({
      label: 'FoamPatternTexture',
      size: [this.size, this.size],
      format: 'rgba16float',
      mipLevelCount: MipmapGenerator.requiredMipLevelCount(this.size, this.size),
      usage: GPUTextureUsage.STORAGE_BINDING      // compute textureStore 写 level 0
           | GPUTextureUsage.TEXTURE_BINDING      // 水面 shader 采样
           | GPUTextureUsage.RENDER_ATTACHMENT,   // MipmapGenerator 逐级 blit 目标
    });

    const module = this._context.createShaderModule(
      createFoamPatternShaderCode(this.size, seedX, seedY),
      'FoamPatternShader.Module',
    );
    const bgl = device.createBindGroupLayout({
      label: 'FoamPatternTexture.BGL',
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' },
      }],
    });
    const pipeline = device.createComputePipeline({
      label: 'FoamPatternTexture.Pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module, entryPoint: 'main' },
    });

    // 只绑 level 0（storage view 不允许跨全 mip 链）。
    const level0View = texture.createView({
      label: 'FoamPatternTexture.Level0',
      dimension: '2d',
      baseMipLevel: 0,
      mipLevelCount: 1,
    });
    const bindGroup = device.createBindGroup({
      layout: bgl,
      entries: [{ binding: 0, resource: level0View }],
    });

    const encoder = device.createCommandEncoder({ label: 'FoamPatternTexture.Encoder' });
    const pass = encoder.beginComputePass({ label: 'FoamPatternTexture.Compute' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.size / 8, this.size / 8, 1);
    pass.end();
    this._context.submit([encoder.finish()]);

    // compute 已提交后再补 mip 链（MipmapGenerator 自建 encoder，队列保序）。
    this._mipmapGenerator.generateMips(device, texture, 'rgba16float');

    this._texture = texture;
    this._view = texture.createView({
      label: 'FoamPatternTexture.View',
      dimension: '2d', // 默认覆盖全 mip 链，采样用
    });
    this._sampler = device.createSampler({
      label: 'FoamPatternTexture.Sampler',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      maxAnisotropy: 8,
    });
  }

  /** rgba16float 2d 全 mip 链纹理视图（采样用）。`generate()` 前为 null。 */
  get view(): GPUTextureView | null {
    return this._view;
  }

  /** repeat + linear（含 aniso 8x）采样器。`generate()` 前为 null。 */
  get sampler(): GPUSampler | null {
    return this._sampler;
  }

  /** 释放 GPU 资源。之后实例不可再用（view/sampler 返回 null）。 */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._texture?.destroy();
    this._texture = null;
    this._view = null;
    this._sampler = null;
  }
}

export interface FoamPatternTextureOptions {
  /** 纹理边长（像素，2 的幂且 ≥ 8）。默认 512。 */
  size?: number;
  /** 随机种子（任意整数；仅重排噪声相位，不影响可平铺性）。默认 1337。 */
  seed?: number;
}
