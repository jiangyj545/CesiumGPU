import type { IPass } from '../scene/passes/IPass';
import type { Scene } from '../scene/Scene';
import { Context } from '../renderer/Context';
import { Camera } from '../scene/Camera';
import {
  BOW_SPRAY_COMPUTE_SHADER,
  BOW_SPRAY_RENDER_SHADER,
  BOW_SPRAY_DRAW_PARAMS_BYTES,
  BOW_SPRAY_FLOATS_PER_PARTICLE,
  BOW_SPRAY_SIM_PARAMS_BYTES,
  BOW_SPRAY_WORKGROUP,
} from '../shaders/BowSprayShader';

/**
 * BowSpray — 船首浪喷雾粒子特效（海洋 L4，tidewater BoatSpray 直译裁剪版）。
 *
 * 管线模式照抄 {@link ParticleSystem}（instanced billboard quad + compute 模拟
 * workgroup 64 + G-Buffer 深度测试），差异只在发射模型与粒子状态布局：
 * - 发射：船首两角（bowPos ± sideDir·(beamM/2 + 5 cm)，抬升 0.3 m），
 *   速度门 smoothstep(3.5, 9, speed)，每帧 emit = min(round(u·speed·1.6), 100)；
 * - 两种粒子：90% droplet（0.9±0.4 s / 6±3 cm / α0.55）+ 10% mist
 *   （1.6±0.5 s / 0.35±0.15 m / α0.06 / 重力×0.3 / 初速×0.5）；
 * - 积分：重力 -9.8 m/s² 沿发射帧 -up + 线性拖拽 tau=1.8 s；米制速度、
 *   位置增量除 EARTH_SCALE（模拟空间选型详见 BowSprayShader 文件头）；
 * - 死亡回收：沿 -up 落回发射平面（h < 0，"局部 y < 0"的世界空间等价）或
 *   寿命尽 → 死亡标记，CPU 环形游标推进的发射窗口内重生成（ring buffer）。
 *
 * 船状态来源（与 WakeField 一致）：页面每帧调 {@link BowSpray.setBoat} 喂数据，
 * 类内不自推算；未调用、已 clearBoat 或 speed 未过速度门时零发射。
 *
 * 生命周期：构造即创建 GPU 资源（要求 `Context.initialize()` 已完成），
 * `prepare()` 幂等空实现，`destroy()` 释放全部 buffer。
 */

/** 船首状态（世界坐标为归一化 ECEF，方向/速度为世界单位与米/秒）。 */
export interface BowSprayBoatState {
  /** 船首世界位置（归一化 ECEF，地球半径 = 1）。 */
  bowPos: { x: number; y: number; z: number };
  /** 船前进方向（世界单位向量）。 */
  headingDir: { x: number; y: number; z: number };
  /** 右舷方向（世界单位向量）。 */
  sideDir: { x: number; y: number; z: number };
  /** 航速（米/秒）。 */
  speed: number;
  /** 船宽（米）。 */
  beamM: number;
}

export interface BowSprayOptions {
  /**
   * 所属场景：提供相机（billboard 投影）与 G-Buffer 深度（被船体/地形遮挡，
   * ParticleSystem 同款硬要求）。
   */
  scene: Scene;
  /** 粒子池容量（向上取整到 workgroup 倍数）。默认 2048。 */
  maxParticles?: number;
}

// ── 发射/粒子常数（CPU 填 uniform + 测试断言共用）───────────────────────────
// 注意：mist 初速/重力倍率不在 TS 侧设常数——真源是 BowSprayShader.ts 的 WGSL
// 常数（MIST_VEL_SCALE / MIST_GRAVITY_SCALE），测试直接锁 WGSL 源串，TS 勿另设副本。

/** 速度门 smoothstep 下/上沿（m/s）：低于 ~3.5 m/s 不喷。 */
export const SPRAY_GATE_LO = 3.5;
export const SPRAY_GATE_HI = 9;
/** 每帧发射上限（tidewater 同款钳制）。 */
export const SPRAY_EMIT_CAP = 100;
/** demand 系数：demand = gate · speed · 1.6。 */
export const SPRAY_DEMAND_GAIN = 1.6;
/** 船首角外偏（米）：beamM/2 + 5 cm。 */
export const SPRAY_CORNER_EXTRA_M = 0.05;
/** 发射点抬升（米）。 */
export const SPRAY_RAISE_M = 0.3;
/** 重力（m/s²）。 */
export const SPRAY_GRAVITY_MPS2 = 9.8;
/** 线性拖拽时间常数（s）：v += (-v/tau)·dt。 */
export const SPRAY_DRAG_TAU_S = 1.8;
/** mist 占比（r < MIST_RATIO → mist）。经 uniform misc.w 下发，TS 侧为唯一来源。 */
export const MIST_RATIO = 0.1;
/** droplet 寿命区间（s）：0.9 ± 0.4。 */
export const DROPLET_LIFE_S: readonly [number, number] = [0.5, 1.3];
/** droplet 尺寸区间（m）：6 ± 3 cm。 */
export const DROPLET_SIZE_M: readonly [number, number] = [0.03, 0.09];
/** droplet 基础 alpha。 */
export const DROPLET_ALPHA = 0.55;
/** mist 寿命区间（s）：1.6 ± 0.5。 */
export const MIST_LIFE_S: readonly [number, number] = [1.1, 2.1];
/** mist 尺寸区间（m）：0.35 ± 0.15。 */
export const MIST_SIZE_M: readonly [number, number] = [0.2, 0.5];
/** mist 基础 alpha。 */
export const MIST_ALPHA = 0.06;

/** 速度门 u = smoothstep(3.5, 9, speed)（非有限输入按 0 处理）。 */
export function sprayGate(speed: number): number {
  if (!Number.isFinite(speed)) return 0;
  const t = Math.min(1, Math.max(0, (speed - SPRAY_GATE_LO) / (SPRAY_GATE_HI - SPRAY_GATE_LO)));
  return t * t * (3 - 2 * t);
}

/**
 * CPU 侧发射预算（可测处）：emit = min(round(gate·speed·1.6), 100)。
 * speed=2 → 0；speed=6 → 4；speed≥9 → 14（cap 100 仅超高速时触及）。
 */
export function computeEmitDemand(speed: number): number {
  // 非有限/负速短路（0 * NaN 仍为 NaN，须先挡掉）
  if (!Number.isFinite(speed) || speed <= 0) return 0;
  const demand = sprayGate(speed) * speed * SPRAY_DEMAND_GAIN;
  return Math.min(Math.round(demand), SPRAY_EMIT_CAP);
}

/** uniform f32 索引（与 BOW_SPRAY_COMPUTE_SHADER 的 SimParams 对齐）。 */
const P_BOW = 0;      // bowPos.xyz, dt
const P_BASIS_R = 4;  // right.xyz, EARTH_SCALE
const P_BASIS_U = 8;  // up.xyz, emitCount
const P_BASIS_F = 12; // fwd.xyz, ringCursor
const P_MISC = 16;    // count, frameSeed, cornerOffset, mistRatio
const P_SPAWN = 20;   // speed, raise, dropletAlpha, mistAlpha
const P_DROPLET = 24; // lifeMin, lifeMax, sizeMin, sizeMax
const P_MIST = 28;    // lifeMin, lifeMax, sizeMin, sizeMax
const P_MISC2 = 32;   // gravity, tau

/** dt 上限（秒），ParticleSystem 同款：切后台返回不爆模拟。 */
const MAX_DT = 0.1;

export class BowSpray implements IPass {
  /** RenderPass / BindGroup 调试标签。 */
  readonly label = 'BowSpray';

  /** 可见性开关（`EffectManager` 依据它跳过 execute）。 */
  show = true;

  /** 粒子池容量（已向上取整到 workgroup 倍数）。 */
  readonly maxParticles: number;

  private _context: Context;
  private _scene: Scene;
  private _boat: BowSprayBoatState | null = null;
  /** 环形发射游标：下一帧发射窗口起点（mod maxParticles）。 */
  private _cursor = 0;

  private _particleBuffer: GPUBuffer;
  private _simParamsBuffer: GPUBuffer;
  private _drawParamsBuffer: GPUBuffer;
  private _quadBuffer: GPUBuffer;
  /** 渲染管线共享 BGL（setTargetFormat 重建管线时 bind group 原样复用）。 */
  private _renderBGL: GPUBindGroupLayout;
  /** T5 真 HDR：当前管线目标格式（构造期 = preferredFormat，Scene 下发后跟随）。 */
  private _targetFormat: GPUTextureFormat;

  private _computePipeline: GPUComputePipeline;
  private _computeBindGroup: GPUBindGroup;
  private _renderPipeline: GPURenderPipeline;
  private _renderBindGroup: GPUBindGroup;

  private _simData  = new Float32Array(BOW_SPRAY_SIM_PARAMS_BYTES / 4);
  private _drawData = new Float32Array(BOW_SPRAY_DRAW_PARAMS_BYTES / 4);
  private _frame = 0;
  private _lastTime: number | null = null;
  private _destroyed = false;
  /** gbuffer 缺失警告只发一次（幂等标志位，不刷屏）。 */
  private _gbufferWarned = false;

  constructor(context: Context, options: BowSprayOptions) {
    if (!context.initialized) {
      throw new Error('BowSpray: Context.initialize() must complete first.');
    }
    if (!options.scene.gbufferPass) {
      throw new Error('BowSpray: scene must be initialized (gbufferPass is required for depth-tested rendering).');
    }

    this._context = context;
    this._scene = options.scene;
    // 单帧发射上限为 SPRAY_EMIT_CAP：池小于它时，一帧的发射窗口回绕即可覆盖全池，
    // 全部槽位同帧重生（喷雾瞬间铺满）——故强制下限。
    const requested = options.maxParticles ?? 2048;
    if (requested < SPRAY_EMIT_CAP) {
      throw new Error(
        `BowSpray: maxParticles must be >= ${SPRAY_EMIT_CAP} (single-frame emit cap; ` +
        `a smaller pool lets one frame's emit window wrap and overwrite the whole pool), ` +
        `got ${requested}.`,
      );
    }
    this.maxParticles = Math.max(
      BOW_SPRAY_WORKGROUP,
      Math.ceil(requested / BOW_SPRAY_WORKGROUP) * BOW_SPRAY_WORKGROUP,
    );

    const device = context.device;

    // ── GPU buffers（初始全死亡驻留：age = life，等环形窗口唤醒）────────────
    this._particleBuffer = device.createBuffer({
      label: 'BowSpray.States',
      size: this.maxParticles * BOW_SPRAY_FLOATS_PER_PARTICLE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this._simParamsBuffer = device.createBuffer({
      label: 'BowSpray.SimParams',
      size: BOW_SPRAY_SIM_PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._drawParamsBuffer = device.createBuffer({
      label: 'BowSpray.DrawParams',
      size: BOW_SPRAY_DRAW_PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._quadBuffer = device.createBuffer({
      label: 'BowSpray.Quad',
      size: 32,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    context.writeBuffer(
      this._quadBuffer, 0,
      new Float32Array([-1, -1,  1, -1,  -1, 1,  1, 1]),
    );

    // ── Compute pipeline ───────────────────────────────────────────────────
    const computeModule = context.createShaderModule(
      BOW_SPRAY_COMPUTE_SHADER, 'BowSpray.ComputeModule',
    );
    const computeBGL = device.createBindGroupLayout({
      label: 'BowSpray.ComputeBGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    this._computePipeline = device.createComputePipeline({
      label: 'BowSpray.ComputePipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [computeBGL] }),
      compute: { module: computeModule, entryPoint: 'update' },
    });
    this._computeBindGroup = device.createBindGroup({
      layout: computeBGL,
      entries: [
        { binding: 0, resource: { buffer: this._particleBuffer } },
        { binding: 1, resource: { buffer: this._simParamsBuffer } },
      ],
    });

    // ── Render pipeline（premultiplied alpha + G-Buffer 深度测试）───────────
    this._targetFormat = context.preferredFormat;
    this._renderBGL = device.createBindGroupLayout({
      label: 'BowSpray.RenderBGL',
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

  /**
   * 喂船首状态（每帧调用，与 WakeField 的 WakeSource 约定一致；类内不自推算）。
   * 全字段 Number.isFinite 校验（NaN/Infinity 抛错，对齐 WakeField.updateBoat）；
   * 未调用过或已 clearBoat 时 execute 正常运行但零发射。
   */
  setBoat(state: BowSprayBoatState): void {
    const fields = [
      state.bowPos.x, state.bowPos.y, state.bowPos.z,
      state.headingDir.x, state.headingDir.y, state.headingDir.z,
      state.sideDir.x, state.sideDir.y, state.sideDir.z,
      state.speed, state.beamM,
    ];
    if (fields.some((v) => !Number.isFinite(v))) {
      throw new Error(
        'BowSpray.setBoat: all bowPos/headingDir/sideDir/speed/beamM fields must be finite numbers.',
      );
    }
    this._boat = {
      bowPos: { ...state.bowPos },
      headingDir: { ...state.headingDir },
      sideDir: { ...state.sideDir },
      speed: state.speed,
      beamM: state.beamM,
    };
  }

  /** 清除内部船状态：此后零发射（直到下一次 setBoat）。 */
  clearBoat(): void {
    this._boat = null;
  }

  /**
   * T5 真 HDR：目标格式下发（EffectManager 可选协议，Scene.render 逐帧调用
   * 当前渲染目标格式）。变更时重建渲染管线（compute 管线/各 buffer 与目标
   * 格式无关原样复用；renderBindGroup 引用的 BGL 不变，同样复用）；同格式
   * 幂等早退。未接线的调用路径保持 preferredFormat 旧行为。
   */
  setTargetFormat(format: GPUTextureFormat): void {
    if (this._destroyed || format === this._targetFormat) return;
    this._targetFormat = format;
    this._renderPipeline = this._buildRenderPipeline();
  }

  /** 幂等初始化。资源在构造函数中创建（Context 已就绪），此处无需再做事。 */
  prepare(): void {
    /* no-op — resources are created eagerly in the constructor. */
  }

  /**
   * 执行一帧：compute（发射 + 米制积分）+ 深度测试的渲染 pass，追加在当前帧
   * 共享 encoder 上（EffectManager 在场景 composite 之后调用）。
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
        console.warn('BowSpray: scene.gbufferPass is missing — skipping spray render (depth test required).');
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
    // setBoat 未调用或已 clearBoat → 零发射（compute 仍跑：纯积分已有粒子，无则全驻留）。
    const emit = this._boat ? computeEmitDemand(this._boat.speed) : 0;
    const cursorBefore = this._cursor;
    this._cursor = (this._cursor + emit) % this.maxParticles;

    // ── 发射帧正交基：up = 径向；right = sideDir 投影到切面；fwd ⊥ 双轴 ────
    const s = this._simData;
    if (this._boat) {
      const b = this._boat;
      const mag = Math.hypot(b.bowPos.x, b.bowPos.y, b.bowPos.z) || 1;
      const ux = b.bowPos.x / mag, uy = b.bowPos.y / mag, uz = b.bowPos.z / mag;

      const dotS = b.sideDir.x * ux + b.sideDir.y * uy + b.sideDir.z * uz;
      let rx = b.sideDir.x - dotS * ux;
      let ry = b.sideDir.y - dotS * uy;
      let rz = b.sideDir.z - dotS * uz;
      let rm = Math.hypot(rx, ry, rz);
      if (rm < 1e-9) {
        // sideDir ∥ up 退化：取任一非平行 helper 再投影（ParticleSystem 同款）
        const hx = Math.abs(uz) > 0.9 ? 1 : 0;
        const hz = Math.abs(uz) > 0.9 ? 0 : 1;
        const dh = hx * ux + hz * uz;
        rx = hx - dh * ux;
        ry = 0 - dh * uy;
        rz = hz - dh * uz;
        rm = Math.hypot(rx, ry, rz) || 1;
      }
      rx /= rm; ry /= rm; rz /= rm;

      // fwd = cross(up, right)，按 headingDir 定向（不依赖调用方手性约定）
      let fx = uy * rz - uz * ry;
      let fy = uz * rx - ux * rz;
      let fz = ux * ry - uy * rx;
      if (fx * b.headingDir.x + fy * b.headingDir.y + fz * b.headingDir.z < 0) {
        fx = -fx; fy = -fy; fz = -fz;
      }

      s[P_BOW + 0] = b.bowPos.x; s[P_BOW + 1] = b.bowPos.y; s[P_BOW + 2] = b.bowPos.z;
      s[P_BASIS_R + 0] = rx; s[P_BASIS_R + 1] = ry; s[P_BASIS_R + 2] = rz;
      s[P_BASIS_U + 0] = ux; s[P_BASIS_U + 1] = uy; s[P_BASIS_U + 2] = uz;
      s[P_BASIS_F + 0] = fx; s[P_BASIS_F + 1] = fy; s[P_BASIS_F + 2] = fz;
      s[P_SPAWN + 0] = b.speed;
      s[P_MISC + 2] = b.beamM * 0.5 + SPRAY_CORNER_EXTRA_M;
    } else {
      s[P_BOW + 0] = 0; s[P_BOW + 1] = 0; s[P_BOW + 2] = 1;
      s[P_BASIS_R + 0] = 1; s[P_BASIS_R + 1] = 0; s[P_BASIS_R + 2] = 0;
      s[P_BASIS_U + 0] = 0; s[P_BASIS_U + 1] = 0; s[P_BASIS_U + 2] = 1;
      s[P_BASIS_F + 0] = 0; s[P_BASIS_F + 1] = 1; s[P_BASIS_F + 2] = 0;
      s[P_SPAWN + 0] = 0;
      s[P_MISC + 2] = 0;
    }
    s[P_BOW + 3] = dt;
    s[P_BASIS_R + 3] = Camera.EARTH_SCALE;
    s[P_BASIS_U + 3] = emit;
    s[P_BASIS_F + 3] = cursorBefore;
    s[P_MISC + 0] = this.maxParticles;
    s[P_MISC + 1] = this._frame;           // hash seed — varies every frame
    s[P_MISC + 3] = MIST_RATIO;
    s[P_SPAWN + 1] = SPRAY_RAISE_M;
    s[P_SPAWN + 2] = DROPLET_ALPHA;
    s[P_SPAWN + 3] = MIST_ALPHA;
    s[P_DROPLET + 0] = DROPLET_LIFE_S[0]; s[P_DROPLET + 1] = DROPLET_LIFE_S[1];
    s[P_DROPLET + 2] = DROPLET_SIZE_M[0]; s[P_DROPLET + 3] = DROPLET_SIZE_M[1];
    s[P_MIST + 0] = MIST_LIFE_S[0]; s[P_MIST + 1] = MIST_LIFE_S[1];
    s[P_MIST + 2] = MIST_SIZE_M[0]; s[P_MIST + 3] = MIST_SIZE_M[1];
    s[P_MISC2 + 0] = SPRAY_GRAVITY_MPS2;
    s[P_MISC2 + 1] = SPRAY_DRAG_TAU_S;
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

    // ── Compute pass（发射 + 积分）──────────────────────────────────────────
    const cpass = encoder.beginComputePass({ label: `${this.label}.Compute` });
    cpass.setPipeline(this._computePipeline);
    cpass.setBindGroup(0, this._computeBindGroup);
    cpass.dispatchWorkgroups(Math.ceil(this.maxParticles / BOW_SPRAY_WORKGROUP));
    cpass.end();

    // ── Render pass（premultiplied alpha + G-Buffer 深度测试）───────────────
    // depthReadOnly: true 时不可设 depthLoadOp/depthStoreOp（WebGPU 规范，
    // TransparentPass/ParticleSystem 同款）；深度 view 每帧现取，resize 自动跟随。
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
  }

  // ── 内部 ───────────────────────────────────────────────────────────────────

  /**
   * 建/重建渲染管线（构造与 setTargetFormat 共用；descriptor 与拆出前逐字段
   * 一致，目标格式取 {@link _targetFormat}）。premultiplied alpha + G-Buffer
   * 深度测试（只读）不变。
   */
  private _buildRenderPipeline(): GPURenderPipeline {
    const device = this._context.device;
    const renderModule = device.createShaderModule({
      code: BOW_SPRAY_RENDER_SHADER,
      label: 'BowSpray.RenderModule',
    });
    return device.createRenderPipeline({
      label: 'BowSpray.RenderPipeline',
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
          format: this._targetFormat,
          // premultiplied alpha：FS 输出 rgb 已乘 a → src = 'one'
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-strip' },
      // 深度测试读取 G-Buffer depth32float：喷雾被船体/地形正确遮挡；
      // 只读不写——透明粒子不得污染场景深度（ParticleSystem 同款）。
      depthStencil: {
        format: 'depth32float',
        depthWriteEnabled: false,
        depthCompare: 'less',
      },
    });
  }
}

export default BowSpray;
