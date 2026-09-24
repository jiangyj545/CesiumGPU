import { UPSAMPLE_SHADER } from '../../shaders/postprocess/UpsampleShader';

/**
 * UpsamplePass - 3×3 tent 上采样 + 同尺寸低层线性叠加（bloom 金字塔回升）。
 *
 * bindGroup 绑定 [tentSrc, addSrc, sampler, params]；由
 * PostProcessManager 按级构建。pipeline 懒创建，幂等。
 */
export class UpsamplePass {
  readonly label = 'PostProcess.Upsample';

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
    const shader = device.createShaderModule({ code: UPSAMPLE_SHADER, label: 'PostProcess.UpsampleShader' });

    this._layout = device.createBindGroupLayout({
      label: 'PostProcess.Up.BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this._pipeline = device.createRenderPipeline({
      label:  'PostProcess.UpPipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this._layout] }),
      vertex:   { module: shader, entryPoint: 'vs_main' },
      fragment: { module: shader, entryPoint: 'fs_main', targets: [{ format: this._format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });

    this._ubo = device.createBuffer({
      label: 'PostProcess.Up.UBO',
      size:  16, // texel(vec2) + blend + pad
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
   * 执行一级上采样叠加。
   * @param srcWidth/Height 源（较小那级）尺寸
   * @param blend          tent 结果权重（0.5 = 均等混合）
   */
  execute(
    encoder: GPUCommandEncoder,
    bindGroup: GPUBindGroup,
    outputView: GPUTextureView,
    srcWidth: number,
    srcHeight: number,
    blend: number,
  ): void {
    this.prepare();
    if (!this._pipeline || !this._ubo) return;

    const data = new Float32Array(4);
    data[0] = 1 / Math.max(srcWidth, 1);
    data[1] = 1 / Math.max(srcHeight, 1);
    data[2] = Math.min(Math.max(blend, 0), 1);
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
