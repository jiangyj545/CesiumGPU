import { BLUR_SHADER } from '../../shaders/postprocess/BlurShader';

/**
 * DownsamplePass - 13-tap 半分辨率下采样（bloom 金字塔逐级收缩）。
 *
 * 管线针对中间 RT 格式构建（Manager 内部 rgba16float），bindGroup 由
 * PostProcessManager 按级构建。pipeline 懒创建，幂等。
 */
export class DownsamplePass {
  readonly label = 'PostProcess.Downsample';

  private _device: GPUDevice;
  private _format: GPUTextureFormat;
  private _pipeline: GPURenderPipeline | null = null;
  private _layout: GPUBindGroupLayout | null = null;
  private _ubo: GPUBuffer | null = null;

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this._device = device;
    this._format = format;
  }

  prepare(): void {
    if (this._pipeline) return;
    const device = this._device;
    const shader = device.createShaderModule({ code: BLUR_SHADER, label: 'PostProcess.BlurShader' });

    this._layout = device.createBindGroupLayout({
      label: 'PostProcess.Down.BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this._pipeline = device.createRenderPipeline({
      label:  'PostProcess.DownPipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this._layout] }),
      vertex:   { module: shader, entryPoint: 'vs_main' },
      fragment: { module: shader, entryPoint: 'fs_main', targets: [{ format: this._format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });

    this._ubo = device.createBuffer({
      label: 'PostProcess.Down.UBO',
      size:  16, // texel(vec2) + pad×2
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  get bindGroupLayout(): GPUBindGroupLayout {
    this.prepare();
    return this._layout!;
  }

  /** 参数 UBO（Manager 构建 bindGroup 时绑定，懒触发 prepare）。 */
  get uniformBuffer(): GPUBuffer {
    this.prepare();
    return this._ubo!;
  }

  /**
   * 执行一级下采样。
   * @param srcWidth/Height 源纹理尺寸（texel 偏移基准）
   */
  execute(
    encoder: GPUCommandEncoder,
    bindGroup: GPUBindGroup,
    outputView: GPUTextureView,
    srcWidth: number,
    srcHeight: number,
  ): void {
    this.prepare();
    if (!this._pipeline || !this._ubo) return;

    const data = new Float32Array(4);
    data[0] = 1 / Math.max(srcWidth, 1);
    data[1] = 1 / Math.max(srcHeight, 1);
    this._device.queue.writeBuffer(this._ubo, 0, data);

    const pass = encoder.beginRenderPass({
      label: this.label,
      colorAttachments: [{
        view:       outputView,
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        loadOp:     'clear',
        storeOp:    'store',
      }],
    });
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }

  destroy(): void {
    this._ubo?.destroy();
    this._ubo = null;
    this._pipeline = null;
    this._layout = null;
  }
}
