import type { IPass } from '../scene/passes/IPass';
import type { Scene } from '../scene/Scene';
import { Context } from '../renderer/Context';
import { Camera } from '../scene/Camera';
import type { OceanSeabedSprayBinding } from './OceanFFTWater';
import {
  BREAK_SPRAY_COMPUTE_SHADER,
  BREAK_SPRAY_RENDER_SHADER,
  BREAK_SPRAY_DRAW_PARAMS_BYTES,
  BREAK_SPRAY_FLOATS_PER_PARTICLE,
  BREAK_SPRAY_SIM_PARAMS_BYTES,
  BREAK_SPRAY_WORKGROUP,
} from '../shaders/BreakSprayShader';

/**
 * BreakSpray — 岸线破碎波喷雾粒子特效（海洋 L5/T4，架构照抄 {@link BowSpray}：
 * storage buffer + compute workgroup 64 + instanced billboard + G-Buffer 深度
 * 测试 + 环形发射游标 + CPU 可测的发射预算纯函数）。
 *
 * 与 BowSpray 的差异：
 * - 发射源：岸线破碎带（{@link BreakSprayBand}：centerEcef/alongDir/widthM）
 *   替代船首两角；页面每帧（或一次性）调 {@link BreakSpray.setBand} 喂数，
 *   未调用、已 clearBand 或 intensity=0 时零发射；
 * - 发射预算：emit = min(round(intensity·cap), cap)（intensity ∈ [0,1]）；
 * - 高度吸附：compute 内每粒子 textureLoad 海床高度场 + 位移纹理级联0 波高，
 *   出生高度贴合波面；水深门 smoothstep(farDepth, nearDepth, depth) 概率
 *   薄化（浅水多喷、深水少喷）。吸附源由 options.seabed getter 每帧惰性求值
 *   （通常 `() => ocean.breakSpraySeabed`）；未提供/未就绪时绑占位资源，
 *   粒子不做水深门照常发射。
 *
 * 生命周期：构造即创建 GPU 资源（要求 `Context.initialize()` 已完成），
 * `prepare()` 幂等空实现，`destroy()` 释放全部 buffer。
 */

/** 破碎带状态（归一化 ECEF；alongDir 为带走向切向单位矢量）。 */
export interface BreakSprayBand {
  /** 带中心（归一化 ECEF，通常取岸线中点/沙洲中心的静水面高程）。 */
  centerEcef: { x: number; y: number; z: number };
  /** 带走向（切向单位矢量，发射点沿其 ±widthM/2 撒开）。 */
  alongDir: { x: number; y: number; z: number };
  /** 带宽（米）。 */
  widthM: number;
  /** 水深门近沿（米，≤ 此值全量发射）。 */
  nearDepth: number;
  /** 水深门远沿（米，≥ 此值不发射；须 > nearDepth）。 */
  farDepth: number;
  /** 破碎强度 0..1（发射率系数）。Default: `1` */
  intensity?: number;
}

export interface BreakSprayOptions {
  /**
   * 所属场景：提供相机（billboard 投影）与 G-Buffer 深度（被海床/地形遮挡，
   * BowSpray 同款硬要求）。
   */
  scene: Scene;
  /** 粒子池容量（向上取整到 workgroup 倍数）。默认 2048。 */
  maxParticles?: number;
  /**
   * 吸附源 getter（通常 `() => ocean.breakSpraySeabed`）。每帧惰性求值；
   * null / 场未就绪时绑占位资源（不做水深门，粒子照常发射）。
   */
  seabed?: () => OceanSeabedSprayBinding | null;
}

// ── 发射/粒子常数（CPU 填 uniform + 测试断言共用）───────────────────────────

/** 每帧发射上限。 */
export const BREAK_SPRAY_EMIT_CAP = 100;
/** demand 系数：demand = clamp01(intensity) · cap。 */
export const BREAK_SPRAY_DEMAND_GAIN = 1.0;
/** 重力（m/s²）。 */
export const BREAK_SPRAY_GRAVITY_MPS2 = 9.8;
/** 线性拖拽时间常数（s）。 */
export const BREAK_SPRAY_DRAG_TAU_S = 1.2;
/** 发射点抬升（米）。 */
export const BREAK_SPRAY_RAISE_M = 0.25;
/** 寿命区间（s）：0.8 ± 0.3。 */
export const BREAK_SPRAY_LIFE_S: readonly [number, number] = [0.5, 1.1];
/** 尺寸区间（m）：9 ± 5 cm。 */
export const BREAK_SPRAY_SIZE_M: readonly [number, number] = [0.04, 0.14];
/** 基础 alpha。 */
export const BREAK_SPRAY_ALPHA = 0.5;

/**
 * CPU 侧发射预算（可测处）：emit = min(round(clamp01(intensity)·cap), cap)。
 * intensity=0 → 0；0.5 → 50；1 → 100（cap 触及）；非有限/负值安全归 0。
 */
export function computeEmitDemand(intensity: number): number {
  // 非有限/负强度短路（clamp 前 NaN 比较为 false，须先挡掉）
  if (!Number.isFinite(intensity) || intensity <= 0) return 0;
  const t = Math.min(1, intensity);
  return Math.min(Math.round(t * BREAK_SPRAY_EMIT_CAP * BREAK_SPRAY_DEMAND_GAIN), BREAK_SPRAY_EMIT_CAP);
}

/** uniform f32 索引（与 BREAK_SPRAY_COMPUTE_SHADER 的 SimParams 对齐）。 */
const P_BAND = 0;    // bandCenter.xyz, dt
const P_ALONG = 4;   // along.xyz, EARTH_SCALE
const P_UP = 8;      // up.xyz, emitCount
const P_EAST = 12;   // east.xyz, ringCursor
const P_NORTH = 16;  // north.xyz, frameSeed
const P_OCEANC = 20; // oceanCenter.xyz, widthM
const P_UVAFF = 24;  // uvOffset.xy, uvScale.zw
const P_BANDP = 28;  // nearDepth, farDepth, intensity, raiseM
const P_MISC = 32;   // gravity, dragTau, 预留, count

/** dt 上限（秒），BowSpray 同款：切后台返回不爆模拟。 */
const MAX_DT = 0.1;

export class BreakSpray implements IPass {
  /** RenderPass / BindGroup 调试标签。 */
  readonly label = 'BreakSpray';

  /** 可见性开关（`EffectManager` 依据它跳过 execute）。 */
  show = true;

  /** 粒子池容量（已向上取整到 workgroup 倍数）。 */
  readonly maxParticles: number;

  private _context: Context;
  private _scene: Scene;
  private _band: BreakSprayBand | null = null;
  /** 吸附源 getter（每帧惰性求值）。 */
  private _seabedSource: (() => OceanSeabedSprayBinding | null) | null;
  /** 环形发射游标：下一帧发射窗口起点（mod maxParticles）。 */
  private _cursor = 0;

  private _particleBuffer: GPUBuffer;
  private _simParamsBuffer: GPUBuffer;
  private _drawParamsBuffer: GPUBuffer;
  private _quadBuffer: GPUBuffer;
  /** seabed 未绑定时的 1×1 r32float 占位（R = -1000 极深 = 无吸附无门）。 */
  private _seabedFallbackTex: GPUTexture;
  private _seabedFallbackView: GPUTextureView;
  /** disp 未绑定时的 1×1 rgba16float 2d-array 占位（波高恒 0）。 */
  private _dispFallbackTex: GPUTexture;
  private _dispFallbackView: GPUTextureView;

  private _computePipeline: GPUComputePipeline;
  private _computeBindGroup: GPUBindGroup;
  private _computeBindGroupLayout: GPUBindGroupLayout;
  /** compute 绑定缓存键：吸附源视图身份变化才重建（getter 惰性求值）。 */
  private _cbSeabed: GPUTextureView | null = null;
  private _cbDisp: GPUTextureView | null = null;

  /** 渲染管线共享 BGL（setTargetFormat 重建管线时 bind group 原样复用）。 */
  private _renderBGL: GPUBindGroupLayout;
  /** T5 真 HDR：场景目标格式（null = 未下发跟随 preferredFormat 旧行为）。 */
  private _targetFormat: GPUTextureFormat | null = null;

  private _renderPipeline: GPURenderPipeline;
  private _renderBindGroup: GPUBindGroup;

  private _simData  = new Float32Array(BREAK_SPRAY_SIM_PARAMS_BYTES / 4);
  private _drawData = new Float32Array(BREAK_SPRAY_DRAW_PARAMS_BYTES / 4);
  private _frame = 0;
  private _lastTime: number | null = null;
  private _destroyed = false;
  /** gbuffer 缺失警告只发一次（幂等标志位，不刷屏）。 */
  private _gbufferWarned = false;

  constructor(context: Context, options: BreakSprayOptions) {
    if (!context.initialized) {
      throw new Error('BreakSpray: Context.initialize() must complete first.');
    }
    if (!options.scene.gbufferPass) {
      throw new Error('BreakSpray: scene must be initialized (gbufferPass is required for depth-tested rendering).');
    }

    this._context = context;
    this._scene = options.scene;
    this._seabedSource = options.seabed ?? null;
    // 单帧发射上限为 BREAK_SPRAY_EMIT_CAP：池小于它时环形窗口回绕即覆盖全池。
    const requested = options.maxParticles ?? 2048;
    if (requested < BREAK_SPRAY_EMIT_CAP) {
      throw new Error(
        `BreakSpray: maxParticles must be >= ${BREAK_SPRAY_EMIT_CAP} (single-frame emit cap; ` +
        `a smaller pool lets one frame's emit window wrap and overwrite the whole pool), ` +
        `got ${requested}.`,
      );
    }
    this.maxParticles = Math.max(
      BREAK_SPRAY_WORKGROUP,
      Math.ceil(requested / BREAK_SPRAY_WORKGROUP) * BREAK_SPRAY_WORKGROUP,
    );

    const device = context.device;

    // ── GPU buffers（初始全死亡驻留：age = life，等环形窗口唤醒）────────────
    this._particleBuffer = device.createBuffer({
      label: 'BreakSpray.States',
      size: this.maxParticles * BREAK_SPRAY_FLOATS_PER_PARTICLE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this._simParamsBuffer = device.createBuffer({
      label: 'BreakSpray.SimParams',
      size: BREAK_SPRAY_SIM_PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._drawParamsBuffer = device.createBuffer({
      label: 'BreakSpray.DrawParams',
      size: BREAK_SPRAY_DRAW_PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._quadBuffer = device.createBuffer({
      label: 'BreakSpray.Quad',
      size: 32,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    context.writeBuffer(
      this._quadBuffer, 0,
      new Float32Array([-1, -1,  1, -1,  -1, 1,  1, 1]),
    );

    // ── 吸附占位资源（BGL 恒全绑定：r32float -1000 / 1×1 波高零 2d-array）───
    this._seabedFallbackTex = device.createTexture({
      label: 'BreakSpray.SeabedFallback',
      size: [1, 1],
      format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: this._seabedFallbackTex },
      new Float32Array([-1000]),
      { bytesPerRow: 4, rowsPerImage: 1 },
      [1, 1, 1],
    );
    this._seabedFallbackView = this._seabedFallbackTex.createView();
    this._dispFallbackTex = device.createTexture({
      label: 'BreakSpray.DispFallback',
      size: [1, 1, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });
    this._dispFallbackView = this._dispFallbackTex.createView({ dimension: '2d-array' });

    // ── Compute pipeline（发射 + 吸附 + 积分）───────────────────────────────
    const computeModule = context.createShaderModule(
      BREAK_SPRAY_COMPUTE_SHADER, 'BreakSpray.ComputeModule',
    );
    this._computeBindGroupLayout = device.createBindGroupLayout({
      label: 'BreakSpray.ComputeBGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        // r32float unfilterable → textureLoad only；位移纹理 2d-array float
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      ],
    });
    this._computePipeline = device.createComputePipeline({
      label: 'BreakSpray.ComputePipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this._computeBindGroupLayout] }),
      compute: { module: computeModule, entryPoint: 'update' },
    });
    this._computeBindGroup = this._buildComputeBindGroup(
      this._seabedFallbackView, this._dispFallbackView,
    );

    // ── Render pipeline（premultiplied alpha + G-Buffer 深度测试，BowSpray 同构）─
    this._renderBGL = device.createBindGroupLayout({
      label: 'BreakSpray.RenderBGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this._renderPipeline = this._buildRenderPipeline();
    this._renderBindGroup = device.createBindGroup({
      layout: this._renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this._drawParamsBuffer } },
        { binding: 1, resource: { buffer: this._particleBuffer } },
      ],
    });
  }

  /** 当前环形游标（发射窗口起点；测试/遥测用，恒 mod maxParticles）。 */
  get ringCursor(): number {
    return this._cursor;
  }

  /** 吸附源 getter（测试/遥测探针用）。 */
  get seabedSource(): (() => OceanSeabedSprayBinding | null) | null {
    return this._seabedSource;
  }

  /**
   * 喂破碎带状态（发射几何 + 水深门参数；类内不自推算）。
   * 全字段 Number.isFinite 校验；widthM 必须 > 0、farDepth 必须 > nearDepth。
   * 未调用过或已 clearBand 时 execute 正常运行但零发射。
   */
  setBand(band: BreakSprayBand): void {
    const fields = [
      band.centerEcef.x, band.centerEcef.y, band.centerEcef.z,
      band.alongDir.x, band.alongDir.y, band.alongDir.z,
      band.widthM, band.nearDepth, band.farDepth,
      band.intensity ?? 1,
    ];
    if (fields.some((v) => !Number.isFinite(v))) {
      throw new Error(
        'BreakSpray.setBand: all centerEcef/alongDir/widthM/nearDepth/farDepth/intensity fields must be finite numbers.',
      );
    }
    if (band.widthM <= 0) {
      throw new Error(`BreakSpray.setBand: widthM must be > 0, got ${band.widthM}.`);
    }
    if (band.farDepth <= band.nearDepth) {
      throw new Error(
        `BreakSpray.setBand: farDepth(${band.farDepth}) must be > nearDepth(${band.nearDepth}).`,
      );
    }
    this._band = {
      centerEcef: { ...band.centerEcef },
      alongDir: { ...band.alongDir },
      widthM: band.widthM,
      nearDepth: band.nearDepth,
      farDepth: band.farDepth,
      intensity: band.intensity ?? 1,
    };
  }

  /** 更新已启用破碎带的发射强度（0..1；不改带几何）。 */
  setIntensity(v: number): void {
    if (!Number.isFinite(v) || v < 0) {
      throw new Error(`BreakSpray.setIntensity: 非法 ${v}`);
    }
    if (this._band) this._band.intensity = Math.min(1, v);
  }

  /** 清除内部带状态：此后零发射（直到下一次 setBand）。 */
  clearBand(): void {
    this._band = null;
  }

  /**
   * T5 真 HDR：目标格式下发（EffectManager 可选协议，Scene.render 逐帧调用
   * 当前渲染目标格式）。变更时重建渲染管线（compute 管线/各 buffer/BGL 与
   * 目标格式无关原样复用，renderBindGroup 引用的 BGL 不变同样复用）；同格式
   * 幂等早退。未接线的调用路径保持 preferredFormat 旧行为。
   */
  setTargetFormat(format: GPUTextureFormat): void {
    if (this._destroyed || format === (this._targetFormat ?? this._context.preferredFormat)) return;
    this._targetFormat = format;
    this._renderPipeline = this._buildRenderPipeline();
  }

  /** 按（seabed/disp）视图身份构建 compute bind group。 */
  private _buildComputeBindGroup(
    seabedView: GPUTextureView,
    dispView: GPUTextureView,
  ): GPUBindGroup {
    return this._context.device.createBindGroup({
      layout: this._computeBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this._particleBuffer } },
        { binding: 1, resource: { buffer: this._simParamsBuffer } },
        { binding: 2, resource: seabedView },
        { binding: 3, resource: dispView },
      ],
    });
  }

  /**
   * 建/重建渲染管线（构造与 setTargetFormat 共用；descriptor 与拆出前逐字段
   * 一致，目标格式取 {@link _targetFormat}，null 时回退 preferredFormat）。
   * premultiplied alpha + G-Buffer 深度测试（只读）不变。
   */
  private _buildRenderPipeline(): GPURenderPipeline {
    const device = this._context.device;
    const renderModule = this._context.createShaderModule(
      BREAK_SPRAY_RENDER_SHADER, 'BreakSpray.RenderModule',
    );
    return device.createRenderPipeline({
      label: 'BreakSpray.RenderPipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this._renderBGL] }),
      vertex: {
        module: renderModule,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 8, // vec2<f32> corner
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
        }],
      },
      fragment: {
        module: renderModule,
        entryPoint: 'fs_main',
        targets: [{
          format: this._targetFormat ?? this._context.preferredFormat,
          // premultiplied alpha：FS 输出 rgb 已乘 a → src = 'one'
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-strip' },
      // 深度测试读取 G-Buffer depth32float：喷雾被海床/地形正确遮挡；只读不写
      depthStencil: {
        format: 'depth32float',
        depthWriteEnabled: false,
        depthCompare: 'less',
      },
    });
  }

  /** 幂等初始化。资源在构造函数中创建（Context 已就绪），此处无需再做事。 */
  prepare(): void {
    /* no-op — resources are created eagerly in the constructor. */
  }

  /**
   * 执行一帧：compute（发射 + 吸附 + 积分）+ 深度测试的渲染 pass，追加在当前
   * 帧共享 encoder 上（EffectManager 在场景 composite 之后调用；须排在
   * OceanFFTWater 之后，吸附源读取当帧海床/位移视图）。
   *
   * @param encoder - 当前帧共享 command encoder
   * @param target  - swap-chain 颜色输出（loadOp 'load'）
   * @param time    - 效果统一时间（秒）；内部差分出 dt 并 clamp 到 0.1 s
   */
  execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void {
    if (this._destroyed) return;

    const gbuffer = this._scene.gbufferPass;
    if (!gbuffer) {
      // 场景已销毁（防御；构造时已校验）——静默跳过但发一次警告，不逐帧刷屏。
      if (!this._gbufferWarned) {
        this._gbufferWarned = true;
        console.warn('BreakSpray: scene.gbufferPass is missing — skipping spray render (depth test required).');
      }
      return;
    }

    // ── dt（EffectManager 不传 dt，由时间差分得到）────────────────────────
    let dt = this._lastTime === null ? 0 : time - this._lastTime;
    this._lastTime = time;
    if (!Number.isFinite(dt) || dt < 0) dt = 0;
    if (dt > MAX_DT) dt = MAX_DT;

    this._frame += 1;

    // ── 发射预算（CPU 可测处）+ 环形游标推进 ───────────────────────────────
    // setBand 未调用或已 clearBand → 零发射（compute 仍跑：纯积分已有粒子）。
    const emit = this._band ? computeEmitDemand(this._band.intensity ?? 1) : 0;
    const cursorBefore = this._cursor;
    this._cursor = (this._cursor + emit) % this.maxParticles;

    // ── 吸附源：每帧惰性求值；未就绪 → 占位资源（无吸附无门，照常发射）────
    const src = this._seabedSource?.() ?? null;
    const seabedView = src?.view ?? this._seabedFallbackView;
    const dispView = src?.dispView ?? this._dispFallbackView;
    if (seabedView !== this._cbSeabed || dispView !== this._cbDisp) {
      this._computeBindGroup = this._buildComputeBindGroup(seabedView, dispView);
      this._cbSeabed = seabedView;
      this._cbDisp = dispView;
    }

    // ── SimParams 填充 ──────────────────────────────────────────────────────
    const s = this._simData;
    const b = this._band;
    if (b) {
      // up = 带中心径向；along 投影到切面（⊥ up）后归一化
      const mag = Math.hypot(b.centerEcef.x, b.centerEcef.y, b.centerEcef.z) || 1;
      const ux = b.centerEcef.x / mag, uy = b.centerEcef.y / mag, uz = b.centerEcef.z / mag;
      const dotA = b.alongDir.x * ux + b.alongDir.y * uy + b.alongDir.z * uz;
      let ax = b.alongDir.x - dotA * ux;
      let ay = b.alongDir.y - dotA * uy;
      let az = b.alongDir.z - dotA * uz;
      const am = Math.hypot(ax, ay, az);
      if (am < 1e-9) {
        // alongDir ∥ up 退化：取任一非平行 helper 再投影（BowSpray 同款）
        ax = Math.abs(uz) > 0.9 ? 1 : 0;
        ay = 0;
        az = Math.abs(uz) > 0.9 ? 0 : 1;
      }
      const al = Math.hypot(ax, ay, az) || 1;

      s[P_BAND + 0] = b.centerEcef.x; s[P_BAND + 1] = b.centerEcef.y; s[P_BAND + 2] = b.centerEcef.z;
      s[P_ALONG + 0] = ax / al; s[P_ALONG + 1] = ay / al; s[P_ALONG + 2] = az / al;
      s[P_UP + 0] = ux; s[P_UP + 1] = uy; s[P_UP + 2] = uz;
      s[P_OCEANC + 0] = src?.centerEcef.x ?? 0;
      s[P_OCEANC + 1] = src?.centerEcef.y ?? 0;
      s[P_OCEANC + 2] = src?.centerEcef.z ?? 0;
      s[P_OCEANC + 3] = b.widthM;
      s[P_UVAFF + 0] = src?.uvOffset.x ?? 0.5;
      s[P_UVAFF + 1] = src?.uvOffset.y ?? 0.5;
      s[P_UVAFF + 2] = src?.uvScale.x ?? 0;
      s[P_UVAFF + 3] = src?.uvScale.y ?? 0;
      s[P_BANDP + 0] = b.nearDepth;
      s[P_BANDP + 1] = b.farDepth;
      s[P_BANDP + 2] = Math.min(1, b.intensity ?? 1);
    } else {
      s[P_BAND + 0] = 0; s[P_BAND + 1] = 0; s[P_BAND + 2] = 1;
      s[P_ALONG + 0] = 1; s[P_ALONG + 1] = 0; s[P_ALONG + 2] = 0;
      s[P_UP + 0] = 0; s[P_UP + 1] = 0; s[P_UP + 2] = 1;
      s[P_OCEANC + 0] = 0; s[P_OCEANC + 1] = 0; s[P_OCEANC + 2] = 1;
      s[P_OCEANC + 3] = 0;
      s[P_UVAFF + 0] = 0.5; s[P_UVAFF + 1] = 0.5; s[P_UVAFF + 2] = 0; s[P_UVAFF + 3] = 0;
      s[P_BANDP + 0] = 3; s[P_BANDP + 1] = 8; s[P_BANDP + 2] = 0;
    }
    s[P_BAND + 3] = dt;
    s[P_ALONG + 3] = Camera.EARTH_SCALE;
    s[P_UP + 3] = emit;
    s[P_EAST + 0] = src?.eastDir.x ?? 1;
    s[P_EAST + 1] = src?.eastDir.y ?? 0;
    s[P_EAST + 2] = src?.eastDir.z ?? 0;
    s[P_EAST + 3] = cursorBefore;
    s[P_NORTH + 0] = src?.northDir.x ?? 0;
    s[P_NORTH + 1] = src?.northDir.y ?? 0;
    s[P_NORTH + 2] = src?.northDir.z ?? 1;
    s[P_NORTH + 3] = this._frame; // hash seed — varies every frame
    s[P_BANDP + 3] = BREAK_SPRAY_RAISE_M;
    s[P_MISC + 0] = BREAK_SPRAY_GRAVITY_MPS2;
    s[P_MISC + 1] = BREAK_SPRAY_DRAG_TAU_S;
    s[P_MISC + 3] = this.maxParticles;
    this._context.writeBuffer(this._simParamsBuffer, 0, s);

    // ── Upload draw params ─────────────────────────────────────────────────
    const d = this._drawData;
    const cam = this._scene.camera;
    d.set(cam.viewProjectionMatrix.values, 0);
    const tanHalfFov = Math.tan(cam.frustum.fov * 0.5);
    d[16] = 1 / (tanHalfFov * cam.frustum.aspectRatio);
    d[17] = 1 / tanHalfFov;
    d[18] = Camera.EARTH_SCALE;
    this._context.writeBuffer(this._drawParamsBuffer, 0, d);

    // ── Compute pass（发射 + 吸附 + 积分）──────────────────────────────────
    const cpass = encoder.beginComputePass({ label: `${this.label}.Compute` });
    cpass.setPipeline(this._computePipeline);
    cpass.setBindGroup(0, this._computeBindGroup);
    cpass.dispatchWorkgroups(Math.ceil(this.maxParticles / BREAK_SPRAY_WORKGROUP));
    cpass.end();

    // ── Render pass（premultiplied alpha + G-Buffer 深度测试，BowSpray 同构）─
    // depthReadOnly: true 时不可设 depthLoadOp/depthStoreOp（WebGPU 规范）；
    // 深度 view 每帧现取，resize 自动跟随。
    const pass = encoder.beginRenderPass({
      label: `${this.label}.Draw`,
      colorAttachments: [{
        view: target,
        loadOp:  'load',  // 叠加在已渲染场景之上，不清屏
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: gbuffer.getDepthView(),
        depthReadOnly: true,
      },
    });
    pass.setPipeline(this._renderPipeline);
    pass.setVertexBuffer(0, this._quadBuffer);
    pass.setBindGroup(0, this._renderBindGroup);
    pass.draw(4, this.maxParticles);
    pass.end();
  }

  /** 释放全部 GPU 资源。调用后实例不可再用。 */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._particleBuffer.destroy();
    this._simParamsBuffer.destroy();
    this._drawParamsBuffer.destroy();
    this._quadBuffer.destroy();
    this._seabedFallbackTex.destroy();
    this._dispFallbackTex.destroy();
    this._cbSeabed = null;
    this._cbDisp = null;
  }
}

export default BreakSpray;
