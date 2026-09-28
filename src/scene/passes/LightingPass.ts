import { Context } from '../../renderer/Context';
import { GBufferPass } from './GBufferPass';
import { DirectionalLight } from '../DirectionalLight';
import { Camera } from '../Camera';
import type { IPass } from './IPass';
import { CAMERA_UBO_SIZE, writeCameraUniforms } from '../../renderer/CameraUniforms';
import {
  selectLightingShader,
} from '../../shaders/LightingShader';
import { AERIAL_PARAMS_UBO_BYTES } from '../../shaders/AtmosphereSkyShader';
import type { ShadowReceiverBinding } from '../../shadow/ShadowMap';

/** Maximum directional lights supported in the WGSL shader. */
const MAX_LIGHTS = 8;

/**
 * Track C 云影接线（生产方 CloudLayer.cloudShadowSource 的结构契约）。
 * 生产方持有资源并每帧写 ubo（CloudShadowParams 96B，布局单一来源
 * CloudShader.ts）；引用身份稳定 —— 本 pass 以同一性做重建判定
 * （仿 OceanAtmosphereBinding 契约）。
 */
export interface CloudShadowReceiverBinding {
  /** 影贴图视图（rgba8unorm，R 通道 = 云密度）。 */
  view: GPUTextureView;
  /** CloudShadowParams UBO（96B）。 */
  ubo: GPUBuffer;
  /** filtering 线性采样器。 */
  sampler: GPUSampler;
}

/**
 * Track A 大气透视天空资源（生产方 AtmosphereSky getter 的结构契约）。
 * view / atmosphereParamsBuffer getter 身份稳定（勿逐帧重建 bind group）；
 * sunDir 为归一化 ECEF（与 AtmosphereSky.setSunDirection 同源，每帧随工厂
 * 求值取最新）。工厂每帧可返回新包装对象 —— 本 pass 按资源身份做重建判定。
 */
export interface AerialReceiverBinding {
  /** skyView LUT 纹理视图（AtmosphereSky.skyViewTexture）。 */
  view: GPUTextureView;
  /** AtmosphereParams UBO（64B，AtmosphereSky 每帧维护）。 */
  atmosphereParamsBuffer: GPUBuffer;
  /** 归一化 ECEF 太阳方向（AerialParams.sunDir 写入源）。 */
  sunDir: number[];
}

/**
 * LightingPass - Reads the G-Buffer and computes deferred Blinn-Phong lighting.
 * Renders a full-screen triangle, outputting the lit colour to the supplied
 * render target (or the swap-chain texture).
 *
 * Implements {@link IPass}: `resize` rebuilds the G-Buffer bind group (Scene
 * drives passes in order, so GBufferPass has already recreated its textures).
 */
export class LightingPass implements IPass {
  readonly label = 'LightingPass';

  private _context: Context;
  private _pipeline: GPURenderPipeline | null = null;
  private _cameraBindGroup: GPUBindGroup | null = null;
  private _gbufferBindGroup: GPUBindGroup | null = null;

  private _cameraUniformBuffer: GPUBuffer;
  private _lightingUniformBuffer: GPUBuffer;

  private _gbufferPass: GBufferPass;
  /**
   * HDR 路径输出格式（T5 真 HDR：构造参 = Scene._sceneColorFormat 单一来源，
   * rgba16float 离屏 RT）。LDR 路径（交换链）格式见 prepare() 的目标选择。
   */
  private _outputFormat: GPUTextureFormat;

  /**
   * P2 后处理开关：true = 输出线性 HDR（跳过 Reinhard+gamma，交给
   * ToneMappingPass）。默认 false —— 旧 LDR 路径行为逐字节不变。
   */
  private _hdr = false;

  /**
   * A1 CSSM 阴影源（ShadowMap.receiverBinding()）。null = 无阴影管线
   * 变体（LIGHTING_SHADER / _HDR 逐字节原样 —— 关闭时零回归）。
   */
  private _shadowSource: ShadowReceiverBinding | null = null;
  private _shadowActive = false;
  private _shadowBindGroup: GPUBindGroup | null = null;

  /**
   * Track C 云影源（CloudLayer.cloudShadowSource）。null = 无云影管线
   * 变体（原 4 变体逐字节原样 —— 关闭时零回归，90 页不接云不受影响）。
   */
  private _cloudShadowSource: CloudShadowReceiverBinding | null = null;
  private _cloudShadowActive = false;
  private _cloudShadowBindGroup: GPUBindGroup | null = null;

  /**
   * Track A 大气透视源（工厂求值产物）。null = 无 AP 管线变体（原 8 变体
   * 逐字节原样 —— 关闭时零回归，90 页不接 AP 不受影响）。
   */
  private _aerialSource: AerialReceiverBinding | null = null;
  private _aerialActive = false;
  private _aerialBindGroup: GPUBindGroup | null = null;
  /** AerialParams UBO（32B，本 pass 所有，writeAerialParams 每帧写）。 */
  private _aerialUBO: GPUBuffer;
  /** skyView LUT 线性 clamp 采样器（本 pass 所有，T6 契约 b2）。 */
  private _aerialSampler: GPUSampler;

  constructor(context: Context, gbufferPass: GBufferPass, outputFormat: GPUTextureFormat) {
    this._context      = context;
    this._gbufferPass  = gbufferPass;
    this._outputFormat = outputFormat;

    const device = context.device;

    // Camera UBO: 3×mat4 (192) + vec4 (16) + vec2 + vec2 (uniform vec2 aligns
    // to 16 → nearFar @224) = 232 bytes → pad to struct align 64 = 256 bytes
    this._cameraUniformBuffer = device.createBuffer({
      label: 'LightingPass.CameraUBO',
      size:  CAMERA_UBO_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Lighting UBO: MAX_LIGHTS × 48 bytes + 16 bytes header = 400 bytes → pad to 416
    const LIGHT_STRIDE = 48; // 3×vec4 = 48 bytes (direction + color + ambient + pad)
    this._lightingUniformBuffer = device.createBuffer({
      label: 'LightingPass.LightingUBO',
      size:  Math.ceil((MAX_LIGHTS * LIGHT_STRIDE + 16) / 16) * 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // T6 AerialParams UBO（32B，布局单一来源 AERIAL_PARAMS_STRUCT_WGSL）
    this._aerialUBO = device.createBuffer({
      label: 'LightingPass.AerialUBO',
      size:  AERIAL_PARAMS_UBO_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._aerialSampler = device.createSampler({
      label: 'LightingPass.AerialSampler',
      magFilter:     'linear',
      minFilter:     'linear',
      addressModeU:  'clamp-to-edge',
      addressModeV:  'clamp-to-edge',
    });
  }

  /**
   * P2 后处理开关（Scene.render 每帧同步 postProcess.enabled）。
   * 仅在状态变化时重建 pipeline（bindGroup 与 shader 布局无关，沿用）。
   */
  setHdrOutput(hdr: boolean): void {
    if (hdr === this._hdr) return;
    this._hdr = hdr;
    this._pipeline = null;
    this.prepare();
  }

  /**
   * A1 CSSM：接入 / 切换 / 摘除阴影源（Scene.render 每帧传入
   * `shadow.receiverBinding()`；同一对象引用时零开销早退）。
   * 非空 → 管线切到带阴影变体并建组 2；null → 回到原双组管线。
   */
  setShadowSource(source: ShadowReceiverBinding | null): void {
    if (source === this._shadowSource) return;
    this._shadowSource = source;
    this._shadowActive = source !== null;
    this._pipeline = null;
    this.prepare();
  }

  /**
   * Track C：接入 / 摘除云影源（Scene.render 每帧传
   * `cloudLayer.cloudShadowSource`；同一对象引用零开销早退）。
   * 非空 → 管线切到带云影变体并建组 3（uniform + 影纹理 + filtering
   * 采样器）；null → 回到原管线（无云影 4 变体，零回归）。
   * FS 侧 intensity <= 0 时 cloudShadowFactor 短路恒 1，双重门控。
   */
  setCloudShadowSource(source: CloudShadowReceiverBinding | null): void {
    if (source === this._cloudShadowSource) return;
    this._cloudShadowSource = source;
    this._cloudShadowActive = source !== null;
    this._pipeline = null;
    this.prepare();
  }

  /**
   * Track A：接入 / 摘除大气透视源（Scene.render 每帧传 `aerialSource()`）。
   * 工厂每帧返回新包装对象 → 除引用同一性外再按 skyView/atmosphereParams
   * 资源身份判定（两者 getter 身份稳定 → 资源未变时仅换壳不重建管线）。
   * 资源就绪 → 管线切到对应 AP 变体并建组 4；null 或未就绪 → 原 8 变体
   * （零回归）。sunDir 由 writeAerialParams 每帧从最新包装对象读取。
   */
  setAerialSource(source: AerialReceiverBinding | null): void {
    const prev = this._aerialSource;
    if (source === prev) return;
    if (prev && source &&
        prev.view === source.view &&
        prev.atmosphereParamsBuffer === source.atmosphereParamsBuffer) {
      this._aerialSource = source; // 资源身份未变 → 仅换壳（sunDir 取新值）
      return;
    }
    this._aerialSource = source;
    this._aerialActive = source !== null &&
      !!source.view && !!source.atmosphereParamsBuffer;
    this._pipeline = null;
    this.prepare();
  }

  /**
   * Track A：AerialParams UBO 每帧写（strength/distM 由 Scene.aerialPerspective
   * 驱动，sunDir 取当前源的归一化 ECEF）。未接 AP 时 no-op（零开销）。
   */
  writeAerialParams(strength: number, distM: number): void {
    const src = this._aerialSource;
    if (!this._aerialActive || !src) return;
    const data = new Float32Array(AERIAL_PARAMS_UBO_BYTES / 4); // 8 floats
    data[0] = strength; // @0 strength（0 = 恒旧 limb haze，零回归闸）
    data[1] = distM;    // @4 distM（米，参考雾距）
    const sd = src.sunDir;
    data[4] = sd[0]; data[5] = sd[1]; data[6] = sd[2]; // @16 sunDir vec3f
    this._context.writeBuffer(this._aerialUBO, 0, data);
  }

  /** Build or rebuild the render pipeline and bind groups. */
  prepare(): void {
    const device = this._context.device;

    // A1 CSSM：阴影开关 × HDR 开关 → 4 个 shader 变体；Track C 再与
    // 云影开关正交组合 → 8 变体；Track A AP 开关再正交 → 16 变体
    // （任一未就绪恒走原变体，逐字节零回归；选择逻辑单一来源
    // selectLightingShader，LightingShader 单测穷举 16 组合）。
    const src = this._shadowSource;
    const shadowReady = this._shadowActive &&
      src !== null && !!src.ubo && !!src.textureView && !!src.comparisonSampler;
    const cloud = this._cloudShadowSource;
    const cloudReady = this._cloudShadowActive &&
      cloud !== null && !!cloud.view && !!cloud.ubo && !!cloud.sampler;
    const aerial = this._aerialSource;
    const aerialReady = this._aerialActive &&
      aerial !== null && !!aerial.view && !!aerial.atmosphereParamsBuffer;
    const shaderCode = selectLightingShader(
      this._hdr, shadowReady, cloudReady, aerialReady);

    const shader = device.createShaderModule({
      code: shaderCode,
      label: 'LightingShader',
    });

    const bindGroupLayout0 = device.createBindGroupLayout({
      label: 'Lighting.BGL0',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    const bindGroupLayout1 = device.createBindGroupLayout({
      label: 'Lighting.BGL1',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } },
      ],
    });

    // 组 2 仅在阴影激活时存在（无阴影管线保持 2 组，零回归）。
    let bindGroupLayout2: GPUBindGroupLayout | null = null;
    if (shadowReady && src) {
      bindGroupLayout2 = device.createBindGroupLayout({
        label: 'Lighting.BGL2',
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          {
            binding: 1,
            visibility: GPUShaderStage.FRAGMENT,
            texture: { sampleType: 'depth', viewDimension: '2d-array', multisampled: false },
          },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
        ],
      });
    }

    // 组 3 仅在云影激活时存在（独立 bind group，保守不动既有组结构）。
    let bindGroupLayout3: GPUBindGroupLayout | null = null;
    if (cloudReady && cloud) {
      bindGroupLayout3 = device.createBindGroupLayout({
        label: 'Lighting.BGL3',
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        ],
      });
    }

    // 组 4 仅在 AP 激活时存在（T6 契约：uniform + skyView 纹理 + filtering
    // 采样器 + atmosphereParams uniform）。
    let bindGroupLayout4: GPUBindGroupLayout | null = null;
    if (aerialReady && aerial) {
      bindGroupLayout4 = device.createBindGroupLayout({
        label: 'Lighting.BGL4',
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', minBindingSize: 32 } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
          { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', minBindingSize: 64 } },
        ],
      });
    }

    const bindGroupLayouts = [bindGroupLayout0, bindGroupLayout1];
    if (bindGroupLayout2) bindGroupLayouts.push(bindGroupLayout2);
    if (bindGroupLayout3) bindGroupLayouts.push(bindGroupLayout3);
    if (bindGroupLayout4) bindGroupLayouts.push(bindGroupLayout4);

    this._pipeline = device.createRenderPipeline({
      label:  'LightingPipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts }),
      vertex:   { module: shader, entryPoint: 'vs_main' },
      fragment: {
        module:  shader,
        entryPoint: 'fs_main',
        targets: [{
          // T5 真 HDR：HDR 路径写 rgba16float 离屏 RT（构造参 = Scene 单一
          // 来源），LDR 路径写交换链 preferredFormat —— 与 Scene.render 的
          // setHdrOutput 每帧同步严格对应（开关切换即重建本管线）。
          format: this._hdr ? this._outputFormat : this._context.preferredFormat,
        }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });

    // Group 0: camera + lighting uniforms
    this._cameraBindGroup = device.createBindGroup({
      label:  'Lighting.BG0',
      layout: bindGroupLayout0,
      entries: [
        { binding: 0, resource: { buffer: this._cameraUniformBuffer } },
        { binding: 1, resource: { buffer: this._lightingUniformBuffer } },
      ],
    });

    // Group 1: G-Buffer textures
    const gb = this._gbufferPass.textures;
    this._gbufferBindGroup = device.createBindGroup({
      label:  'Lighting.BG1',
      layout: bindGroupLayout1,
      entries: [
        { binding: 0, resource: gb.albedoView },
        { binding: 1, resource: gb.normalView },
        { binding: 2, resource: gb.worldPosView },
        { binding: 3, resource: this._gbufferPass.sampler },
      ],
    });

    // Group 2: CSSM 级联阴影（uniform + depth 数组纹理 + comparison 采样器）。
    this._shadowBindGroup = null;
    if (shadowReady && src && bindGroupLayout2) {
      this._shadowBindGroup = device.createBindGroup({
        label:  'Lighting.BG2',
        layout: bindGroupLayout2,
        entries: [
          { binding: 0, resource: { buffer: src.ubo } },
          { binding: 1, resource: src.textureView },
          { binding: 2, resource: src.comparisonSampler },
        ],
      });
    }

    // Group 3: 云影（uniform + rgba8unorm 影纹理 + filtering 采样器）。
    this._cloudShadowBindGroup = null;
    if (cloudReady && cloud && bindGroupLayout3) {
      this._cloudShadowBindGroup = device.createBindGroup({
        label:  'Lighting.BG3',
        layout: bindGroupLayout3,
        entries: [
          { binding: 0, resource: { buffer: cloud.ubo } },
          { binding: 1, resource: cloud.view },
          { binding: 2, resource: cloud.sampler },
        ],
      });
    }

    // Group 4: 大气透视（AerialParams UBO + skyView LUT + linear/clamp 采样器
    // + AtmosphereParams UBO；本 pass 自有 UBO/采样器，天空资源为外部身份）。
    this._aerialBindGroup = null;
    if (aerialReady && aerial && bindGroupLayout4) {
      this._aerialBindGroup = device.createBindGroup({
        label:  'Lighting.BG4',
        layout: bindGroupLayout4,
        entries: [
          { binding: 0, resource: { buffer: this._aerialUBO } },
          { binding: 1, resource: aerial.view },
          { binding: 2, resource: this._aerialSampler },
          { binding: 3, resource: { buffer: aerial.atmosphereParamsBuffer } },
        ],
      });
    }
  }

  /**
   * Update camera uniform buffer.
   */
  updateCamera(camera: Camera): void {
    const data = new Float32Array(CAMERA_UBO_SIZE / 4);
    // Layout lives in src/renderer/CameraUniforms.ts (single source of truth).
    writeCameraUniforms(data, {
      viewMatrix:           camera.viewMatrix.values,
      projectionMatrix:     camera.projectionMatrix.values,
      viewProjectionMatrix: camera.viewProjectionMatrix.values,
      position:             camera.position,
      viewportWidth:        this._context.drawingBufferWidth,
      viewportHeight:       this._context.drawingBufferHeight,
      near:                 camera.frustum.near,
      far:                  camera.frustum.far,
    });

    this._context.writeBuffer(this._cameraUniformBuffer, 0, data);
  }

  /**
   * Update lighting uniform buffer.
   */
  updateLights(lights: DirectionalLight[]): void {
    const count = Math.min(lights.length, MAX_LIGHTS);
    const LIGHT_FLOATS = 12; // 48 bytes / 4
    const data = new Float32Array(MAX_LIGHTS * LIGHT_FLOATS + 4);

    for (let i = 0; i < count; i++) {
      const ld = lights[i].toFloat32Array();
      data.set(ld, i * LIGHT_FLOATS);
    }

    // lightCount (u32) at offset MAX_LIGHTS * LIGHT_FLOATS
    const countView = new Uint32Array(data.buffer, MAX_LIGHTS * LIGHT_FLOATS * 4, 4);
    countView[0] = count;

    this._context.writeBuffer(this._lightingUniformBuffer, 0, data);
  }

  /**
   * IPass: canvas resize.  GBufferPass recreates its textures first (Scene
   * drives passes in array order), so only the G-Buffer bind group needs
   * rebuilding against the new views.
   */
  resize(_width: number, _height: number): void {
    this._rebuildGBufferBindGroup();
  }

  /**
   * Rebuild G-Buffer bind group (call after GBuffer resize).
   */
  private _rebuildGBufferBindGroup(): void {
    if (!this._pipeline) return;
    const device = this._context.device;
    const layout = this._pipeline.getBindGroupLayout(1);
    const gb = this._gbufferPass.textures;
    this._gbufferBindGroup = device.createBindGroup({
      label:  'Lighting.BG1',
      layout,
      entries: [
        { binding: 0, resource: gb.albedoView },
        { binding: 1, resource: gb.normalView },
        { binding: 2, resource: gb.worldPosView },
        { binding: 3, resource: this._gbufferPass.sampler },
      ],
    });
  }

  /**
   * Execute the lighting pass, drawing a full-screen triangle.
   * (`_time` is unused — deferred shading is stateless per frame; the optional
   * parameter keeps the {@link IPass} signature satisfied.)
   */
  execute(encoder: GPUCommandEncoder, outputView: GPUTextureView, _time?: number): void {
    if (!this._pipeline || !this._cameraBindGroup || !this._gbufferBindGroup) {
      console.warn('LightingPass.execute: not prepared yet');
      return;
    }

    const pass = encoder.beginRenderPass({
      label: 'Lighting Pass',
      colorAttachments: [{
        view: outputView,
        // HDR 路径背景更暗（(0,0,0.02,1)）：后续 bloom 阈值提取不被背景抬亮
        clearValue: this._hdr
          ? { r: 0.0, g: 0.0, b: 0.02, a: 1.0 }
          : { r: 0.0, g: 0.0, b: 0.05, a: 1.0 },
        loadOp:     'clear',
        storeOp:    'store',
      }],
    });

    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, this._cameraBindGroup);
    pass.setBindGroup(1, this._gbufferBindGroup);
    // A1 CSSM：阴影激活时补组 2（管线布局已含 BGL2）。
    if (this._shadowActive && this._shadowBindGroup) {
      pass.setBindGroup(2, this._shadowBindGroup);
    }
    // Track C：云影激活时补组 3（管线布局已含 BGL3）。
    if (this._cloudShadowActive && this._cloudShadowBindGroup) {
      pass.setBindGroup(3, this._cloudShadowBindGroup);
    }
    // Track A：AP 激活时补组 4（管线布局已含 BGL4）。
    if (this._aerialActive && this._aerialBindGroup) {
      pass.setBindGroup(4, this._aerialBindGroup);
    }
    pass.draw(3); // full-screen triangle
    pass.end();
  }

  destroy(): void {
    this._cameraUniformBuffer.destroy();
    this._lightingUniformBuffer.destroy();
    this._aerialUBO.destroy();
  }
}

export default LightingPass;
