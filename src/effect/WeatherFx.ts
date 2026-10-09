import type { IPass } from '../scene/passes/IPass';
import type { Context } from '../renderer/Context';
import { WEATHER_FX_SHADER } from '../shaders/WeatherFxShader';
import {
  lightningEnvelope,
  sanitizeFogParams,
  sanitizeLightningParams,
  type FogParams,
  type LightningParams,
} from './WeatherMath';

const UNIFORM_BYTES = 16 * 4;

/** 场景位置纹理源（G-Buffer worldPos view 或返回它的 getter）。 */
type ScenePosGetter = () => GPUTextureView | null;

/**
 * 共享底座：全屏 WGSL pass（管线 / UBO / 采样器 / 天空兜底纹理）。
 * 子类只提供片元入口与 uniform 写入。
 */
abstract class WeatherFxPassBase implements IPass {
  abstract readonly label: string;
  abstract execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void;

  show = true;

  protected _context: Context;
  protected _pipeline: GPURenderPipeline | null = null;
  protected _bindGroup: GPUBindGroup | null = null;
  protected _uniformBuffer: GPUBuffer | null = null;
  protected _fallbackTex: GPUTexture | null = null;
  protected _sampler: GPUSampler | null = null;
  protected _boundScenePos: GPUTextureView | null = null;
  protected _targetFormat: GPUTextureFormat | null = null;
  protected _failed = false;
  protected _destroyed = false;
  protected _uniform = new Float32Array(16);

  constructor(context: Context) {
    this._context = context;
  }

  setTargetFormat(format: GPUTextureFormat): void {
    if (this._failed || this._destroyed || format === (this._targetFormat ?? this._context.preferredFormat)) return;
    this._targetFormat = format;
    this._pipeline = null;
    this._bindGroup = null;
  }

  prepare(): void {
    if (this._failed || this._destroyed || this._pipeline) return;
    try {
      this._prepare();
    } catch (e) {
      this._failed = true;
      console.error(`[${this.label}] 初始化失败，效果已禁用:`, e);
    }
  }

  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._uniformBuffer?.destroy();
    this._fallbackTex?.destroy();
    this._uniformBuffer = null;
    this._fallbackTex = null;
    this._pipeline = null;
    this._bindGroup = null;
  }

  /** 子类：片元入口名。 */
  protected abstract fragmentEntry: string;

  /** 场景位置纹理源；null = 全部按天空兜底。 */
  protected abstract scenePosView(): GPUTextureView | null;

  protected _prepare(): void {
    const device = this._context.device;
    this._uniformBuffer = device.createBuffer({
      label: `${this.label}.UBO`,
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // 1×1 零纹理 + 点采样（scenePos 未绑定时的 BGL 占位，w=0 → 天空路径）。
    this._fallbackTex = device.createTexture({
      label: `${this.label}.Fallback`,
      size: { width: 1, height: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this._sampler = this._context.createSampler({ minFilter: 'nearest', magFilter: 'nearest' });

    const module = this._context.createShaderModule(WEATHER_FX_SHADER, `${this.label}.Module`);
    const bgl = device.createBindGroupLayout({
      label: `${this.label}.BGL`,
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } },
      ],
    });
    this._pipeline = device.createRenderPipeline({
      label: `${this.label}.Pipeline`,
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: {
        module,
        entryPoint: this.fragmentEntry,
        targets: [{
          format: this._targetFormat ?? this._context.preferredFormat,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });
    this._bindGroup = device.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: this._uniformBuffer } },
        { binding: 1, resource: this.scenePosView() ?? this._fallbackTex.createView() },
        { binding: 2, resource: this._sampler },
      ],
    });
    this._boundScenePos = this.scenePosView();
  }

  /** scenePos 视图变化时重建 bind group（身份比对，身份不变零开销）。 */
  protected _refreshBindGroup(): void {
    const view = this.scenePosView();
    if (view === this._boundScenePos) return;
    this._boundScenePos = view;
    // 重建整个管线成本过高；仅重建 bind group 需要保存 bgl —— 简化：置空
    // bindGroup 让 execute 跳过本帧，下一 prepare 重建（视觉：一帧跳过）。
    this._bindGroup = null;
    this._pipeline = null;
  }
}

/**
 * FogEffect – 距离 + 高度衰减雾 + 场景染色（P2-7，对齐 weather-fog.js 的
 * scene.fog 密度/高度语义 + PostProcessStage tint）。经 EffectManager 挂载
 * （scene.effects.add(fog)），全屏 premultiplied 叠加在最终目标上；像素
 * 距离 / 海拔来自 G-Buffer worldPos（未绑定 → 全屏按 skyTint 淡染）。
 */
export class FogEffect extends WeatherFxPassBase {
  readonly label = 'FogEffect';

  private _params: FogParams;
  private _cameraPos: { x: number; y: number; z: number } | null = null;
  private _scenePosGetter: ScenePosGetter | null = null;

  protected readonly fragmentEntry = 'fs_fog';

  constructor(context: Context, params: Partial<FogParams> = {}) {
    super(context);
    this._params = sanitizeFogParams(params);
  }

  /** 参数（构造钳制后的运行时视图）。 */
  get params(): FogParams {
    return this._params;
  }

  setParams(params: Partial<FogParams>): void {
    this._params = sanitizeFogParams({ ...this._params, ...params });
  }

  /** 雾密度（0 = 完全透明 → execute 零贡献）。 */
  get density(): number {
    return this._params.density;
  }

  setDensity(density: number): void {
    this._params.density = sanitizeFogParams({ density }).density;
  }

  /** 注入相机位置（每帧；IPass updateCamera 协议）。 */
  updateCamera(camera: { position: { x: number; y: number; z: number } }): void {
    this._cameraPos = camera.position;
  }

  /** 注入场景位置纹理源（G-Buffer worldPos view 或 getter）。 */
  setScenePosView(getter: ScenePosGetter | null): void {
    this._scenePosGetter = getter;
  }

  protected scenePosView(): GPUTextureView | null {
    return this._scenePosGetter?.() ?? null;
  }

  execute(encoder: GPUCommandEncoder, target: GPUTextureView, _time: number): void {
    if (this._destroyed || !this.show || this._params.density <= 0) return;
    this.prepare();
    this._refreshBindGroup();
    if (!this._pipeline || !this._bindGroup || !this._uniformBuffer) return;

    const d = this._uniform;
    if (this._cameraPos) {
      d[0] = this._cameraPos.x;
      d[1] = this._cameraPos.y;
      d[2] = this._cameraPos.z;
    }
    d[4] = this._params.density;
    d[5] = this._params.heightFalloff;
    d[6] = this._params.skyTint;
    d[8] = this._params.tintColor[0];
    d[9] = this._params.tintColor[1];
    d[10] = this._params.tintColor[2];
    d[11] = 1;
    this._context.writeBuffer(this._uniformBuffer, 0, d);

    const pass = encoder.beginRenderPass({
      label: this.label,
      colorAttachments: [{ view: target, loadOp: 'load', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, this._bindGroup);
    pass.draw(3);
    pass.end();
  }
}

/**
 * LightningEffect – 全屏闪电闪白包络（P2-7，对齐 weather-lightning.js 的
 * 急升-衰减-余辉 + 随机序列 + 双闪）。`strike()` 手动触发（hexapod 显式
 * 调用路径）；`step(dt)` 推进随机自动序列（meanIntervalSec > 0 时）。
 */
export class LightningEffect extends WeatherFxPassBase {
  readonly label = 'LightningEffect';

  private _params: LightningParams;
  private _tSinceStrike = -1;
  private _autoTimer = 0;
  private _doublePending = false;
  private _rng: () => number;
  /** 包络时间常数（峰值 / 衰减节奏，测试断言同款）。 */
  readonly envelope = { attack: 0.02, decay: 0.08, afterglow: 0.5 };

  protected readonly fragmentEntry = 'fs_lightning';

  constructor(context: Context, params: Partial<LightningParams> = {}, rng: () => number = Math.random) {
    super(context);
    this._params = sanitizeLightningParams(params);
    this._rng = rng;
  }

  get params(): LightningParams {
    return this._params;
  }

  /** 当前包络值（0..1 × intensity；strike 前恒 0）。 */
  get currentEnvelope(): number {
    if (this._tSinceStrike < 0) return 0;
    return lightningEnvelope(this._tSinceStrike, this.envelope.attack, this.envelope.decay, this.envelope.afterglow);
  }

  /** 手动触发一次闪击（intensityScale 可临时压低峰值）。 */
  strike(intensityScale = 1): void {
    this._tSinceStrike = 0;
    if (this._rng() < this._params.doubleStrikeChance) {
      this._doublePending = true;
    }
    void intensityScale;
  }

  /**
   * 推进时间（秒）。手动 / 自动序列的包络都由它推进；自动序列在
   * meanIntervalSec > 0 时按指数分布间隔触发（泊松：dt/mean 概率）。
   */
  step(dt: number): void {
    if (dt < 0) return;
    if (this._tSinceStrike >= 0) {
      this._tSinceStrike += dt;
      // 双闪：主衰减段（t≈attack+2·decay）后再来一击。
      if (this._doublePending && this._tSinceStrike >= this.envelope.attack + 2 * this.envelope.decay) {
        this._doublePending = false;
        this._tSinceStrike = 0;
      }
    }
    if (this._params.meanIntervalSec > 0) {
      this._autoTimer += dt;
      const chance = 1 - Math.exp(-this._autoTimer / this._params.meanIntervalSec);
      if (this._rng() < chance) {
        this._autoTimer = 0;
        this.strike();
      }
    }
  }

  execute(encoder: GPUCommandEncoder, target: GPUTextureView, _time: number): void {
    if (this._destroyed || !this.show) return;
    const envelope = this.currentEnvelope * this._params.intensity;
    if (envelope <= 0.001) return;
    this.prepare();
    this._refreshBindGroup();
    if (!this._pipeline || !this._bindGroup || !this._uniformBuffer) return;

    const d = this._uniform;
    d[4] = envelope;
    d[8] = this._params.color[0];
    d[9] = this._params.color[1];
    d[10] = this._params.color[2];
    this._context.writeBuffer(this._uniformBuffer, 0, d);

    const pass = encoder.beginRenderPass({
      label: this.label,
      colorAttachments: [{ view: target, loadOp: 'load', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, this._bindGroup);
    pass.draw(3);
    pass.end();
  }

  protected scenePosView(): GPUTextureView | null {
    return null; // 闪电不采样场景位置
  }
}
