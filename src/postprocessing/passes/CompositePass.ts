import { COMPOSITE_SHADER } from '../../shaders/postprocess/CompositeShader';

/** CompositePass 每帧参数（Manager 从配置 + 太阳投影计算得出）。 */
export interface CompositeFrameParams {
  bloomIntensity: number;
  flareIntensity: number;   // 屏外太阳 = 0
  sunUv: [number, number];  // uv 域（y 已翻转）
  ghosts: number;
  time: number;
  dirtStrength: number;
  aspect: number;
}

/**
 * CompositePass - sceneColor + bloom + lensFlare（ghost + DirtMask）合成。
 *
 * bindGroup 绑定 [sceneTex, bloomTex, dirtTex, sampler, params]；由
 * PostProcessManager 构建并按帧传入参数。pipeline 懒创建，幂等。
 */
export class CompositePass {
  readonly label = 'PostProcess.Composite';

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
    const shader = device.createShaderModule({ code: COMPOSITE_SHADER, label: 'PostProcess.CompositeShader' });

    this._layout = device.createBindGroupLayout({
      label: 'PostProcess.Composite.BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this._pipeline = device.createRenderPipeline({
      label:  'PostProcess.CompositePipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this._layout] }),
      vertex:   { module: shader, entryPoint: 'vs_main' },
      fragment: { module: shader, entryPoint: 'fs_main', targets: [{ format: this._format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });

    // CompositeParams：7×f32 → 28B → 对齐 32B
    this._ubo = device.createBuffer({
      label: 'PostProcess.Composite.UBO',
      size:  32,
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

  /** 执行合成（输出仍是线性域，ToneMapping 在下一 pass）。 */
  execute(
    encoder: GPUCommandEncoder,
    bindGroup: GPUBindGroup,
    outputView: GPUTextureView,
    p: CompositeFrameParams,
  ): void {
    this.prepare();
    if (!this._pipeline || !this._ubo) return;

    const data = new Float32Array(8);
    data[0] = p.bloomIntensity;
    data[1] = p.flareIntensity;
    data[2] = p.sunUv[0];
    data[3] = p.sunUv[1];
    data[4] = p.ghosts;
    data[5] = p.time;
    data[6] = p.dirtStrength;
    data[7] = p.aspect;
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
