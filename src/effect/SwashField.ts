import type { IPass } from '../scene/passes/IPass';
import type { Context } from '../renderer/Context';
import { Cartesian3 } from '../math/Cartesian3';
import { ecefFromLonLatHeight } from '../terrain/TerrainProvider';
import type { SeabedHeightField } from './SeabedHeightField';
import {
  buildSwashShader,
  SWASH_PARAMS_BYTES,
  SWASH_WORKGROUP,
  SWASH_P_GRID,
  SWASH_P_PHYS,
  SWASH_P_WAVE,
  SWASH_P_MISC,
} from '../shaders/SwashComputeShader';

/**
 * SwashField - swash 冲刷带 1D Saint-Venant 浅水场（tidewater 追赶 Track W）。
 *
 * 与 {@link WakeField} 同款 ping-pong compute 架构（双 rgba16float 纹理交替、
 * prepare 失败闸、CPU 步进控制），但模拟域是**固定窗口**（与绑定海床同域同
 * 尺寸，无 WakeField 的跟随滚动）：每 texel 沿本地岸法向解一根 1D 浅水柱
 * （物理与数值格式见 SwashComputeShader 头注释）：
 * - R = 水位 η（米相对静水） G = 法向流速 u（m/s） B = 水深（米） A = 湿度；
 * - 海床 b 从 SeabedHeightField 的 r32float 纹理 textureLoad（不可过滤）；
 * - 入射波从深水带（b < -4m）松弛注入，自动向岸传播；干湿前沿置干。
 *
 * CFL 稳定：CPU 侧按最深水深 cmax = √(g·max(dry, 1.35A - bMin)) 算
 * dtMax = 0.4·cell/cmax，每帧拆 ≤ 4 子步；帧超 4·dtMax 时模拟慢放而非爆炸
 * （dt 先钳 1/30，切后台不掉帧爆）。bMin 在 bindSeabed 时从 CPU 高度数组
 * 取 min（未绑海床回退 -8m 深水假设）。
 *
 * 生命周期：构造即校验；`prepare()` 幂等建管线（失败置 `_failed` 闸一次
 * 报错，execute 重入不重试，WakeField/OceanFFTWater 同款）；`destroy()` 后
 * 不可再用。未绑海床时界内标记恒 0 → 无 forcing、静水维持（不炸但无动画，
 * 接线时必须传 seabed）。
 */

export interface SwashFieldOptions {
  /** 网格边长（向上取整到 8 的倍数，clamp 64..512）。Default: `128` */
  gridN?: number;
  /** 窗口边长（米；bindSeabed 后被 field.sizeMeters 覆盖）。Default: `500` */
  windowSizeM?: number;
  /** 入射主波周期（秒，clamp 2..20）。Default: `7` */
  wavePeriod?: number;
  /** 入射波幅（米，clamp 0.01..2）。Default: `0.35` */
  waveAmplitude?: number;
  /** Manning 简化摩擦系数 cf（clamp 0..1）。Default: `0.06` */
  friction?: number;
  /** 干湿阈值水深（米，clamp 0.001..0.5）。Default: `0.02` */
  dryDepth?: number;
  /** 海床高度场源（湿沙/岸线地形来源；也可后接 {@link bindSeabed}）。 */
  seabed?: SeabedHeightField | null;
  /** 未绑海床时的窗口中心经纬度（度）。Default: `[0, 0]` */
  center?: { lon: number; lat: number };
}

/** 重力（m/s²，与 WakeField 同值）。 */
const GRAVITY = 9.81;
/** CFL 安全上限：c·dt/cell ≤ 0.4（Rusanov 稳定域 ≤ 1，留余量）。 */
const CFL_LIMIT = 0.4;
/** dt 上限（秒）：切后台/掉帧不爆模拟（WakeField 同款）。 */
const MAX_DT = 1 / 30;
/** 子步上限：超出的帧慢放（dtSub 钳 dtMax）而非线性放大 dispatch 数。 */
const MAX_SUBSTEPS = 4;
/** 未绑海床时的 bMin 假设（米，SeabedHeightField 默认 edgeDepth 量级）。 */
const DEFAULT_BMIN = -8;

export class SwashField implements IPass {
  /** 调试标签（compute pass / BindGroup descriptor label）。 */
  readonly label = 'SwashField';

  /** 可见性开关（EffectManager 依据它跳过 execute）。 */
  show = true;

  private _context: Context;
  private _gridN: number;
  private _cellM: number;
  private _sizeM: number;
  private _windowOpt: number;
  private _wavePeriod: number;
  private _waveAmplitude: number;
  private _friction: number;
  private _dryDepth: number;
  private _bMin = DEFAULT_BMIN;
  private _centerRadial: Cartesian3;
  /** 构造期默认中心（未绑海床时 bindSeabed(null) 还原用）。 */
  private _defaultCenterRadial: Cartesian3;

  private _failed = false;
  private _ready = false;
  private _destroyed = false;

  private _seabed: SeabedHeightField | null = null;
  private _textures: GPUTexture[] = [];
  private _views: GPUTextureView[] = [];
  private _bindGroups: GPUBindGroup[] = [];
  /** 海床纹理视图身份缓存，bindSeabed 变更时重建两个 bind group。 */
  private _bgl: GPUBindGroupLayout | null = null;
  private _bgSeabedView: GPUTextureView | null = null;
  private _seabedFallbackTex: GPUTexture | null = null;
  private _pipeline: GPUComputePipeline | null = null;
  private _ubo: GPUBuffer | null = null;
  /** _ping = 最近写入的纹理下标；下一子步从它读、写到 ^1。 */
  private _ping = 0;

  private _params = new Float32Array(SWASH_PARAMS_BYTES / 4);

  /** 模拟累计时间（秒，子步推进；forcing 相位用，与帧 time 解耦防跳变）。 */
  private _time = 0;
  private _reset = 1;
  private _lastTime: number | null = null;

  constructor(context: Context, options: SwashFieldOptions = {}) {
    if (!context.initialized) {
      throw new Error('SwashField: Context.initialize() must complete first.');
    }
    const rawN = Math.round(options.gridN ?? 128);
    this._gridN = Math.max(64, Math.min(512, Math.ceil(rawN / SWASH_WORKGROUP) * SWASH_WORKGROUP));

    this._windowOpt = Number.isFinite(options.windowSizeM ?? NaN) && (options.windowSizeM as number) > 0
      ? (options.windowSizeM as number) : 500;
    this._sizeM = this._windowOpt;
    this._cellM = this._sizeM / this._gridN;

    this._wavePeriod = SwashField._clamp(options.wavePeriod, 2, 20, 7);
    this._waveAmplitude = SwashField._clamp(options.waveAmplitude, 0.01, 2, 0.35);
    this._friction = SwashField._clamp(options.friction, 0, 1, 0.06);
    this._dryDepth = SwashField._clamp(options.dryDepth, 0.001, 0.5, 0.02);

    const center = options.center ?? { lon: 0, lat: 0 };
    const out: [number, number, number] = [0, 0, 0];
    ecefFromLonLatHeight(center.lon, center.lat, 0, out);
    this._defaultCenterRadial = new Cartesian3(out[0], out[1], out[2]);
    this._centerRadial = this._defaultCenterRadial;

    this._context = context;
    if (options.seabed) this.bindSeabed(options.seabed);
  }

  // ── 公开 API ─────────────────────────────────────────────────────────────

  /** 场纹理 view：R=水位η(m) G=流速u(m/s) B=水深(m) A=湿度。prepare 前为 null。 */
  get swashView(): GPUTextureView | null {
    return this._ready ? this._views[this._ping] : null;
  }

  /** 网格边长（格）。 */
  get gridN(): number {
    return this._gridN;
  }

  /** 单格边长（米）。 */
  get cellM(): number {
    return this._cellM;
  }

  /** 窗口边长（米；绑海床后 = field.sizeMeters）。 */
  get windowSizeMeters(): number {
    return this._sizeM;
  }

  /** 窗口中心的归一化 ECEF（海拔 0；绑海床后 = field.centerRadial 同引用）。 */
  get centerRadial(): Cartesian3 {
    return this._centerRadial;
  }

  /** 入射主波周期（秒）。 */
  get wavePeriod(): number {
    return this._wavePeriod;
  }

  /** 入射波幅（米）。 */
  get waveAmplitude(): number {
    return this._waveAmplitude;
  }

  /** Manning 简化摩擦系数。 */
  get friction(): number {
    return this._friction;
  }

  /** 干湿阈值水深（米）。 */
  get dryDepth(): number {
    return this._dryDepth;
  }

  /** 入射主波周期（秒，clamp 2..20；非有限值保持不变）。 */
  setWavePeriod(period: number): void {
    this._wavePeriod = SwashField._clamp(period, 2, 20, this._wavePeriod);
  }

  /** 入射波幅（米，clamp 0.01..2）。 */
  setWaveAmplitude(amp: number): void {
    this._waveAmplitude = SwashField._clamp(amp, 0.01, 2, this._waveAmplitude);
  }

  /** Manning 简化摩擦系数（clamp 0..1）。 */
  setFriction(cf: number): void {
    this._friction = SwashField._clamp(cf, 0, 1, this._friction);
  }

  /** 干湿阈值水深（米，clamp 0.001..0.5）。 */
  setDryDepth(depth: number): void {
    this._dryDepth = SwashField._clamp(depth, 0.001, 0.5, this._dryDepth);
  }

  /**
   * 绑定海床高度场：窗口/中心/bMin 全部跟随 field（bMin 从 CPU 高度数组取
   * min，供 CFL cmax 估计）；传 null 恢复构造选项窗口。同一 field 引用重复
   * 调用为无操作（幂等）。已 prepare 后换绑会重建 bind group（换纹理视图）。
   */
  bindSeabed(field: SeabedHeightField | null): void {
    if (this._seabed === field) return;
    this._seabed = field;
    if (field) {
      this._sizeM = field.sizeMeters;
      this._centerRadial = field.centerRadial;
      const h = field.heights;
      let mn = Infinity;
      for (let i = 0; i < h.length; i++) {
        if (h[i] < mn) mn = h[i];
      }
      this._bMin = Number.isFinite(mn) ? mn : DEFAULT_BMIN;
    } else {
      this._sizeM = this._windowOpt;
      this._centerRadial = this._defaultCenterRadial;
      this._bMin = DEFAULT_BMIN;
    }
    this._cellM = this._sizeM / this._gridN;
    this._bgSeabedView = null; // 强制 execute 重建 bind group（换绑纹理视图）
  }

  /** 幂等初始化（双纹理 / 管线 / bindGroup）。失败置闸一次报错，不再每帧重试。 */
  prepare(): void {
    if (this._failed || this._ready || this._destroyed) return;
    try {
      const device = this._context.device;
      const N = this._gridN;

      // 双缓冲场纹理：本子步读 A 写 B，下子步读 B 写 A（ping-pong 交替）
      for (const suffix of ['A', 'B']) {
        const tex = device.createTexture({
          label: `SwashField.Field${suffix}`,
          size: { width: N, height: N },
          format: 'rgba16float',
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this._textures.push(tex);
        this._views.push(tex.createView({ label: `SwashField.Field${suffix}.View` }));
      }

      this._ubo = device.createBuffer({
        label: 'SwashField.Params',
        size: SWASH_PARAMS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const module = this._context.createShaderModule(
        buildSwashShader(N), 'SwashField.ComputeModule',
      );
      // minBindingSize 须显式，否则管线验证失败（OceanFFTWater 血泪教训）
      const bgl = device.createBindGroupLayout({
        label: 'SwashField.BGL',
        entries: [
          {
            binding: 0, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'uniform', minBindingSize: SWASH_PARAMS_BYTES },
          },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
          {
            binding: 2, visibility: GPUShaderStage.COMPUTE,
            storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' },
          },
          // r32float 不可过滤高度场（textureLoad），CausticsFloor binding 5 同款
          {
            binding: 3, visibility: GPUShaderStage.COMPUTE,
            texture: { sampleType: 'unfilterable-float', viewDimension: '2d' },
          },
        ],
      });
      this._bgl = bgl;
      this._pipeline = device.createComputePipeline({
        label: 'SwashField.Step',
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        compute: { module, entryPoint: 'main' },
      });

      this._ready = true;
      this._ensureBindGroups();
    } catch (e) {
      // execute 每帧重入 prepare：初始化抛错后若不设闸会每帧重试刷屏，
      // 且 GPU 资源已部分创建，重试必然同样失败——一次报错并禁用本效果。
      this._failed = true;
      console.error(`[${this.label}] 初始化失败，效果已禁用:`, e);
    }
  }

  /**
   * 每帧执行：dt 钳制 [0, 1/30] → CFL 子步拆分（≤ 4）→ 每子步 1 dispatch
   * （uniform 打包 + ping-pong 交替）。首帧/`reset()` 后第一子步全场静水。
   */
  execute(encoder: GPUCommandEncoder, _target: GPUTextureView, time: number): void {
    if (this._destroyed) return;
    this.prepare();
    if (this._failed || !this._ready) return;
    this._ensureBindGroups();

    let dt = this._lastTime === null ? 0 : time - this._lastTime;
    this._lastTime = time;
    if (!Number.isFinite(dt) || dt < 0) dt = 0;
    if (dt > MAX_DT) dt = MAX_DT;

    // CFL：cmax 用最深水深（波幅 1.35 倍覆盖次波 + bMin），拆子步保稳定
    const cmax = Math.sqrt(GRAVITY * Math.max(this._dryDepth, 1.35 * this._waveAmplitude - this._bMin));
    const dtMax = CFL_LIMIT * this._cellM / cmax;
    const substeps = Math.max(1, Math.min(MAX_SUBSTEPS, Math.ceil(dt / dtMax)));
    const dtSub = Math.min(dt, MAX_SUBSTEPS * dtMax) / substeps;

    // ---- uniform 打包（4 × vec4，见 SwashComputeShader struct 注释）
    const s = this._params;
    s[SWASH_P_GRID + 0] = this._cellM;
    s[SWASH_P_GRID + 1] = this._sizeM;
    s[SWASH_P_PHYS + 0] = GRAVITY;
    s[SWASH_P_PHYS + 1] = this._friction;
    s[SWASH_P_PHYS + 2] = this._dryDepth;
    s[SWASH_P_WAVE + 0] = this._waveAmplitude;
    s[SWASH_P_WAVE + 1] = (2 * Math.PI) / this._wavePeriod;
    s[SWASH_P_WAVE + 2] = 0.35;   // 次波幅比（确定性噪声，shader 侧 vnoise）
    s[SWASH_P_WAVE + 3] = 0.42;   // 沿岸波数（~15m 波长，与解析路径 0.42 一致）
    s[SWASH_P_MISC + 0] = 1.2;    // forcing 松弛率（1/s，兼吸外行波）
    s[SWASH_P_MISC + 1] = -4;     // forcing 深水阈（米，负）
    s[SWASH_P_MISC + 2] = 6;      // 海绵宽（格）
    s[SWASH_P_MISC + 3] = 0.3;    // 横向涡粘（m²/s）

    const workgroups = this._gridN / SWASH_WORKGROUP;
    const cpass = encoder.beginComputePass({ label: `${this.label}.Step` });
    cpass.setPipeline(this._pipeline!);
    for (let i = 0; i < substeps; i++) {
      s[SWASH_P_GRID + 2] = dtSub;
      s[SWASH_P_GRID + 3] = this._time;
      s[SWASH_P_PHYS + 3] = i === 0 ? this._reset : 0; // 仅首子步消费 reset
      this._context.writeBuffer(this._ubo!, 0, s);
      cpass.setBindGroup(0, this._bindGroups[this._ping]);
      cpass.dispatchWorkgroups(workgroups, workgroups);
      this._ping ^= 1;
      this._time += dtSub;
    }
    cpass.end();
    this._reset = 0;
  }

  /** 全场清静水（下一子步生效；forcing 带随后自然起波）。 */
  reset(): void {
    this._reset = 1;
  }

  /** 释放全部 GPU 资源。调用后实例不可再用。 */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._ready = false;
    for (const t of this._textures) t.destroy();
    this._textures = [];
    this._views = [];
    this._bindGroups = [];
    this._ubo?.destroy();
    this._ubo = null;
    this._pipeline = null;
    this._bgl = null;
    this._bgSeabedView = null;
    this._seabedFallbackTex?.destroy();
    this._seabedFallbackTex = null;
  }

  // ── 内部 ─────────────────────────────────────────────────────────────────

  /** undefined/非有限值回退 fallback，否则夹 [min, max]（构造与 setter 群共用）。 */
  private static _clamp(v: number | undefined, min: number, max: number, fallback: number): number {
    if (v === undefined || !Number.isFinite(v)) return fallback;
    return Math.min(max, Math.max(min, v));
  }

  /**
   * bind group 惰性构建/重建（bg[k]：读纹理 k、写 k^1、binding 3 = 海床）。
   * prepare 首建与 bindSeabed 换绑（_bgSeabedView 置 null）时执行；同视图幂等。
   */
  private _ensureBindGroups(): void {
    const seabedView = this._seabed?.seabedTexture ?? this._seabedFallbackView();
    if (this._bgSeabedView === seabedView && this._bindGroups.length === 2) return;
    const device = this._context.device;
    this._bindGroups = [];
    for (let k = 0; k < 2; k++) {
      this._bindGroups.push(device.createBindGroup({
        label: `SwashField.BG${k}`,
        layout: this._bgl!,
        entries: [
          { binding: 0, resource: { buffer: this._ubo! } },
          { binding: 1, resource: this._views[k] },
          { binding: 2, resource: this._views[k ^ 1] },
          { binding: 3, resource: seabedView },
        ],
      }));
    }
    this._bgSeabedView = seabedView;
  }

  /**
   * binding 3 占位纹理（1×1 r32float 全 -1000 极深 = 无干湿/无摩擦深水；
   * 仅未绑 field 时使用，此时界内标记恒 0，forcing 不会触发，模拟维持静水）。
   */
  private _seabedFallbackView(): GPUTextureView {
    if (!this._seabedFallbackTex) {
      const device = this._context.device;
      const tex = device.createTexture({
        label: 'SwashField.SeabedFallback',
        size: [1, 1],
        format: 'r32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture(
        { texture: tex },
        new Float32Array([-1000]),
        { bytesPerRow: 4, rowsPerImage: 1 },
        [1, 1, 1],
      );
      this._seabedFallbackTex = tex;
    }
    return this._seabedFallbackTex.createView({ label: 'SwashField.SeabedFallback.View' });
  }
}

export default SwashField;
