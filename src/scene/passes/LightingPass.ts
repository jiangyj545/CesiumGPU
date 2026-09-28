import { Context } from '../../renderer/Context';
import { GBufferPass } from './GBufferPass';
import { DirectionalLight } from '../DirectionalLight';
import { Camera } from '../Camera';
import type { IPass } from './IPass';
import { CAMERA_UBO_SIZE, writeCameraUniforms } from '../../renderer/CameraUniforms';
import {
  LIGHTING_SHADER,
  LIGHTING_SHADER_HDR,
  LIGHTING_SHADER_SHADOW,
  LIGHTING_SHADER_SHADOW_HDR,
} from '../../shaders/LightingShader';
import type { ShadowReceiverBinding } from '../../shadow/ShadowMap';

/** Maximum directional lights supported in the WGSL shader. */
const MAX_LIGHTS = 8;

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

  /** Build or rebuild the render pipeline and bind groups. */
  prepare(): void {
    const device = this._context.device;

    // A1 CSSM：阴影开关 × HDR 开关 → 4 个 shader 变体。
    const src = this._shadowSource;
    const shadowReady = this._shadowActive &&
      src !== null && !!src.ubo && !!src.textureView && !!src.comparisonSampler;
    const shaderCode = this._hdr
      ? (shadowReady ? LIGHTING_SHADER_SHADOW_HDR : LIGHTING_SHADER_HDR)
      : (shadowReady ? LIGHTING_SHADER_SHADOW     : LIGHTING_SHADER);

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

    this._pipeline = device.createRenderPipeline({
      label:  'LightingPipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: bindGroupLayout2
        ? [bindGroupLayout0, bindGroupLayout1, bindGroupLayout2]
        : [bindGroupLayout0, bindGroupLayout1] }),
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
    pass.draw(3); // full-screen triangle
    pass.end();
  }

  destroy(): void {
    this._cameraUniformBuffer.destroy();
    this._lightingUniformBuffer.destroy();
  }
}

export default LightingPass;
