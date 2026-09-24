import { BRIGHT_SHADER } from '../../shaders/postprocess/BrightShader';

/**
 * BrightPass - soft-knee 亮度提取（bloom 链第一级，半分辨率预滤波）。
 *
 * 持有 device + pipeline + bindGroupLayout + UBO；bindGroup 由
 * PostProcessManager 构建（视图随帧管理），execute 画全屏三角 draw(3)
 * （照 LightingPass.execute:191-212 模式）。pipeline 懒创建，幂等。
 */
export class BrightPass {
  readonly label = 'PostProcess.Bright';

  private _device: GPUDevice;
  private _format: GPUTextureFormat;
  private _pipeline: GPURenderPipeline | null = null;
  private _layout: GPUBindGroupLayout | null = null;
  private _ubo: GPUBuffer | null = null;

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this._device = device;
    this._format = format;
  }

  /** 幂等初始化（pipeline + BGL + UBO）。 */
  prepare(): void {
    if (this._pipeline) return;
    const device = this._device;
    const shader = device.createShaderModule({ code: BRIGHT_SHADER, label: 'PostProcess.BrightShader' });

    this._layout = device.createBindGroupLayout({
      label: 'PostProcess.Bright.BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this._pipeline = device.createRenderPipeline({
      label:  'PostProcess.BrightPipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this._layout] }),
      vertex:   { module: shader, entryPoint: 'vs_main' },
      fragment: { module: shader, entryPoint: 'fs_main', targets: [{ format: this._format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });

    this._ubo = device.createBuffer({
      label: 'PostProcess.Bright.UBO',
      size:  16, // threshold, knee, pad, pad
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /** Manager 构建 bindGroup 所需的布局（懒触发 prepare）。 */
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
   * 执行亮度提取。
   * @param bindGroup Manager 构建的 group（srcTex/srcSamp/params）
   */
  execute(
    encoder: GPUCommandEncoder,
    bindGroup: GPUBindGroup,
    outputView: GPUTextureView,
    threshold: number,
    knee: number,
  ): void {
    this.prepare();
    if (!this._pipeline || !this._ubo) return;

    const data = new Float32Array(4);
    data[0] = threshold;
    data[1] = knee;
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
    pass.draw(3); // 全屏三角
    pass.end();
  }

  destroy(): void {
    this._ubo?.destroy();
    this._ubo = null;
    this._pipeline = null;
    this._layout = null;
  }
}
