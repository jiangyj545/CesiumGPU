import { GOD_RAYS_SHADER } from '../../shaders/postprocess/GodRaysShader';

/** GodRaysPass 每帧参数（Manager 从配置 + 太阳投影计算得出）。 */
export interface GodRaysFrameParams {
  sunUv: [number, number];  // uv 域（y 已翻转）
  intensity: number;        // 已乘屏外余晖衰减
  decay: number;
  density: number;
  weight: number;
  threshold: number;
  aspect: number;
  tint: [number, number, number];
}

/**
 * GodRaysPass - 屏幕空间光柱（Track G：场景色亮度 mask → 径向 march）。
 *
 * bindGroup 绑定 [sceneTex, sampler, params]；由 PostProcessManager 构建
 * 并按帧传入参数。pipeline 懒创建，幂等（照 BrightPass/CompositePass 范式）。
 * 输出光束纹理（INTERNAL_FORMAT rgba16float），Composite 前执行。
 */
export class GodRaysPass {
  readonly label = 'PostProcess.GodRays';

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
    const shader = device.createShaderModule({ code: GOD_RAYS_SHADER, label: 'PostProcess.GodRaysShader' });

    this._layout = device.createBindGroupLayout({
      label: 'PostProcess.GodRays.BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this._pipeline = device.createRenderPipeline({
      label:  'PostProcess.GodRaysPipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this._layout] }),
      vertex:   { module: shader, entryPoint: 'vs_main' },
      fragment: { module: shader, entryPoint: 'fs_main', targets: [{ format: this._format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });

    // GodRaysParams：12×f32 = 48B（对齐 16 已满足）
    this._ubo = device.createBuffer({
      label: 'PostProcess.GodRays.UBO',
      size:  48,
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
   * 执行光柱 march（输出线性域光束纹理，合成在 CompositePass）。
   * @param bindGroup Manager 构建的 group（sceneTex/samp/params）
   */
  execute(
    encoder: GPUCommandEncoder,
    bindGroup: GPUBindGroup,
    outputView: GPUTextureView,
    p: GodRaysFrameParams,
  ): void {
    this.prepare();
    if (!this._pipeline || !this._ubo) return;

    const data = new Float32Array(12);
    data[0] = p.sunUv[0];
    data[1] = p.sunUv[1];
    data[2] = p.intensity;
    data[3] = p.decay;
    data[4] = p.density;
    data[5] = p.weight;
    data[6] = p.threshold;
    data[7] = p.aspect;
    data[8]  = p.tint[0];
    data[9]  = p.tint[1];
    data[10] = p.tint[2];
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
