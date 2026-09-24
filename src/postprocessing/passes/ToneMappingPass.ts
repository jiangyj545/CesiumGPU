import { TONEMAPPING_SHADER } from '../../shaders/postprocess/ToneMappingShader';

/**
 * ToneMappingPass - 链末输出：none / reinhard / aces + gamma → swapChain。
 *
 * 唯一以交换链格式为输出目标的 pass（其余 pass 针对内部 rgba16float
 * 中间 RT）。bindGroup 绑定 [srcTex, sampler, params]；pipeline 懒创建。
 */
export class ToneMappingPass {
  readonly label = 'PostProcess.ToneMapping';

  private _device: GPUDevice;
  private _format: GPUTextureFormat;   // 交换链格式（preferredFormat）
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
    const shader = device.createShaderModule({ code: TONEMAPPING_SHADER, label: 'PostProcess.ToneMappingShader' });

    this._layout = device.createBindGroupLayout({
      label: 'PostProcess.ToneMap.BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this._pipeline = device.createRenderPipeline({
      label:  'PostProcess.ToneMapPipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this._layout] }),
      vertex:   { module: shader, entryPoint: 'vs_main' },
      fragment: { module: shader, entryPoint: 'fs_main', targets: [{ format: this._format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });

    this._ubo = device.createBuffer({
      label: 'PostProcess.ToneMap.UBO',
      size:  16, // mode(u32) + pad×3
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
   * @param mode 0 = none（直通），1 = reinhard+gamma，2 = aces+gamma
   */
  execute(
    encoder: GPUCommandEncoder,
    bindGroup: GPUBindGroup,
    outputView: GPUTextureView,
    mode: number,
  ): void {
    this.prepare();
    if (!this._pipeline || !this._ubo) return;

    const data = new Uint32Array(4);
    data[0] = mode;
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
