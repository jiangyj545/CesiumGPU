import type { IPass } from '../scene/passes/IPass';
import type { Scene } from '../scene/Scene';
import { Context } from '../renderer/Context';
import { Camera } from '../scene/Camera';
import { Cartesian3 } from '../math/Cartesian3';
import {
  PARTICLE_COMPUTE_SHADER,
  PARTICLE_RENDER_SHADER,
} from '../shaders/ParticleCompute';

/**
 * ParticleSystem — GPU compute 粒子系统（T3.4 正式版，对齐 EV-Globe 的
 * `EV_ParticleSystem(viewer)`）。
 *
 * - 模拟：compute pass 就地读写单 storage buffer（32 B/粒子），与场景共用
 *   同一个 command encoder（`EffectManager.execute` 由 `Scene.render()` 在
 *   composite 之后调用，特效画在最后不被覆盖）。
 * - 渲染：instanced screen-aligned billboard，**加法混合 + G-Buffer 深度测试**
 *   （`GBufferPass.textures.depthView` 作 depth attachment，`depthReadOnly` +
 *   `depthCompare: 'less'` 不写深度），地球背面的粒子被遮挡不再透出。
 * - 颜色/尺寸随寿命在 shader 内按预设插值（{@link ParticleSystem.presets}）。
 *
 * 生命周期：构造即创建 GPU 资源（要求 `Context.initialize()` 已完成），
 * `prepare()` 为幂等空实现，`destroy()` 释放全部 buffer，之后实例不可再用。
 *
 * 时间：`execute(encoder, target, time)` 的 `time` 为秒（Scene 统一传入），
 * 内部由相邻帧差分得到 `dt`（clamp 到 0.1 s，防切后台爆模拟）。
 */

export type ParticleMode = 'fire' | 'fountain' | 'fireworks' | 'smoke';

/** Shader 内色带分支编号（与 PARTICLE_RENDER_SHADER 的 colorMode 对应）。 */
export type ParticleColorMode = 'fire' | 'water' | 'multi' | 'smoke';

/** 发射器位置：米制 ECEF `Cartesian3`，或 `{ lon, lat, height }`（度 / 米）。 */
export type ParticleOrigin = Cartesian3 | { lon: number; lat: number; height?: number };

/** 模式预设参数表（world units：1 = 地球半径，见 `Camera.EARTH_SCALE`）。 */
export interface ParticlePreset {
  /** 重力加速度（units/s²，指向地心；0 = 悬浮直线）。 */
  gravity: number;
  /** 寿命区间 [min, max]（秒，每粒子随机）。 */
  lifeRange: [number, number];
  /** 初速区间 [min, max]（units/s）。 */
  speedRange: [number, number];
  /** 锥形半角（弧度，绕地表法线；sphere 形状时忽略）。 */
  coneAngle: number;
  /** Billboard 尺寸区间 [min, max]（米，每粒子随机）。 */
  sizeRange: [number, number];
  /** 色带模式。 */
  colorMode: ParticleColorMode;
  /** 发射形状：cone = 沿地表法线锥形，sphere = 球形四散（烟花爆点）。 */
  shape: 'cone' | 'sphere';
}

export interface ParticleOptions {
  /** 模式预设。默认 `'fire'`。 */
  mode?: ParticleMode;
  /** 粒子池容量（向上取整到 64 的倍数）。默认 10000。 */
  count?: number;
  /** 发射器位置。 */
  origin: ParticleOrigin;
  /** 发射速率（粒子/秒）。默认不限（count 全量激活）。 */
  emissionRate?: number;
  /**
   * 所属场景：提供相机（billboard 投影）与 G-Buffer 深度（地球遮挡剔除，
   * 转正硬要求）。深度 view 每帧从 `scene.gbufferPass` 现取，resize 自动跟随。
   */
  scene: Scene;
}

/** 与 shader `colorMode` 编号的映射。 */
const COLOR_MODE_INDEX: Record<ParticleColorMode, number> = {
  fire: 0,
  water: 1,
  multi: 2,
  smoke: 3,
};

const WORKGROUP_SIZE = 64;
/** pos(3) + age(1) + vel(3) + life(1) floats = 32 bytes per particle. */
const FLOATS_PER_PARTICLE = 8;
const BYTES_PER_PARTICLE  = FLOATS_PER_PARTICLE * 4;
/** SimParams: 4 x vec4<f32>. */
const SIM_PARAMS_BYTES = 64;
/** DrawParams: mat4x4<f32> + 2 x vec4<f32>. */
const DRAW_PARAMS_BYTES = 96;
/** dt 上限（秒），与 spike 一致：切后台返回不爆模拟。 */
const MAX_DT = 0.1;

/** 把发射器位置归一化为内部 ECEF（地球半径 = 1）。 */
function normalizeOrigin(pos: ParticleOrigin): [number, number, number] {
  let ecef: Cartesian3;
  if ('x' in pos && 'y' in pos && 'z' in pos) {
    ecef = Cartesian3.multiplyByScalar(pos, 1 / Camera.EARTH_SCALE, new Cartesian3());
  } else {
    // 与 `Cartesian3.fromDegrees` 同一套 WGS84 换算，保证和场景其它对象一致。
    ecef = Cartesian3.fromDegrees(pos.lon, pos.lat, pos.height ?? 0);
    ecef = Cartesian3.multiplyByScalar(ecef, 1 / Camera.EARTH_SCALE, new Cartesian3());
  }
  return [ecef.x, ecef.y, ecef.z];
}

export class ParticleSystem implements IPass {
  /** RenderPass / BindGroup 调试标签。 */
  readonly label = 'ParticleSystem';

  /** 可见性开关（`EffectManager` 依据它跳过 execute）。 */
  show = true;

  /** 粒子池容量（已向上取整到 workgroup 倍数）。 */
  readonly count: number;

  /** 模式预设参数表。 */
  static presets: Record<ParticleMode, Readonly<ParticlePreset>> = {
    // 火焰：向上锥形喷发，强重力回落，短寿命，白黄→橙红渐变。
    fire: {
      gravity: 0.08, lifeRange: [1.2, 2.0], speedRange: [0.03, 0.09],
      coneAngle: 0.35, sizeRange: [4000, 8000], colorMode: 'fire', shape: 'cone',
    },
    // 喷泉：宽锥高抛物线，水蓝白，中等寿命。
    fountain: {
      gravity: 0.06, lifeRange: [1.8, 2.5], speedRange: [0.02, 0.05],
      coneAngle: 0.60, sizeRange: [3000, 6000], colorMode: 'water', shape: 'cone',
    },
    // 烟花：爆点球形四散，低重力长弧线，多彩；尺寸随寿命衰减（拖尾感）。
    fireworks: {
      gravity: 0.035, lifeRange: [1.5, 2.6], speedRange: [0.05, 0.11],
      coneAngle: Math.PI, sizeRange: [1500, 3500], colorMode: 'multi', shape: 'sphere',
    },
    // 烟雾：近零初速慢速上升，无重力，大尺寸灰白低 alpha。
    smoke: {
      gravity: 0.0, lifeRange: [3.5, 6.0], speedRange: [0.008, 0.02],
      coneAngle: 0.25, sizeRange: [9000, 16000], colorMode: 'smoke', shape: 'cone',
    },
  };

  private _context: Context;
  private _scene: Scene;
  private _preset: ParticlePreset;
  private _origin: [number, number, number];
  /** 发射率预算（Little 定律：rate × 平均寿命 ≈ 稳态活跃数），≤ count。 */
  private _activeCount: number;

  private _particleBuffer: GPUBuffer;
  private _simParamsBuffer: GPUBuffer;
  private _drawParamsBuffer: GPUBuffer;
  private _quadBuffer: GPUBuffer;

  private _computePipeline: GPUComputePipeline;
  private _computeBindGroup: GPUBindGroup;
  private _renderPipeline: GPURenderPipeline;
  private _renderBindGroup: GPUBindGroup;

  private _simData  = new Float32Array(SIM_PARAMS_BYTES  / 4);
  private _drawData = new Float32Array(DRAW_PARAMS_BYTES / 4);
  private _time = 0;
  private _frame = 0;
  private _lastTime: number | null = null;
  private _destroyed = false;

  constructor(context: Context, options: ParticleOptions) {
    if (!context.initialized) {
      throw new Error('ParticleSystem: Context.initialize() must complete first.');
    }
    if (!options.scene.gbufferPass) {
      throw new Error('ParticleSystem: scene must be initialized (gbufferPass is required for depth-tested rendering).');
    }

    this._context     = context;
    this._scene       = options.scene;
    this._preset      = ParticleSystem.presets[options.mode ?? 'fire'];
    this._origin      = normalizeOrigin(options.origin);
    this.count        = Math.max(WORKGROUP_SIZE, Math.ceil(options.count ?? 10_000));

    const meanLife = (this._preset.lifeRange[0] + this._preset.lifeRange[1]) * 0.5;
    this._activeCount = options.emissionRate !== undefined
      ? Math.min(this.count, Math.max(1, Math.round(options.emissionRate * meanLife)))
      : this.count;

    const device = context.device;

    // ── GPU buffers ────────────────────────────────────────────────────────
    // In-place read/write storage: each particle is written only by the one
    // invocation that owns it, so no atomics or double-buffering are needed.
    this._particleBuffer = device.createBuffer({
      label: 'ParticleSystem.States',
      size: this.count * BYTES_PER_PARTICLE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    // CPU 预热到稳态（随机寿命相位），首帧即成稳定形态而非同步脉冲。
    context.writeBuffer(this._particleBuffer, 0, this._initParticleData());

    this._simParamsBuffer = device.createBuffer({
      label: 'ParticleSystem.SimParams',
      size: SIM_PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._drawParamsBuffer = device.createBuffer({
      label: 'ParticleSystem.DrawParams',
      size: DRAW_PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // 4-vertex quad (triangle strip), corner in [-1,1] per vertex.
    this._quadBuffer = device.createBuffer({
      label: 'ParticleSystem.Quad',
      size: 32,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    context.writeBuffer(
      this._quadBuffer, 0,
      new Float32Array([-1, -1,  1, -1,  -1, 1,  1, 1]),
    );

    // ── Compute pipeline ───────────────────────────────────────────────────
    const computeModule = context.createShaderModule(
      PARTICLE_COMPUTE_SHADER, 'ParticleSystem.ComputeModule',
    );
    const computeBGL = device.createBindGroupLayout({
      label: 'ParticleSystem.ComputeBGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    this._computePipeline = device.createComputePipeline({
      label: 'ParticleSystem.ComputePipeline',
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

    // ── Render pipeline (instanced billboards, additive blend, depth test) ─
    const renderModule = context.createShaderModule(
      PARTICLE_RENDER_SHADER, 'ParticleSystem.RenderModule',
    );
    const renderBGL = device.createBindGroupLayout({
      label: 'ParticleSystem.RenderBGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this._renderPipeline = device.createRenderPipeline({
      label: 'ParticleSystem.RenderPipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
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
          format: context.preferredFormat,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one',       dstFactor: 'one', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-strip' },
      // 深度测试读取 G-Buffer depth32float：粒子被地球/地形遮挡的部分被丢弃。
      // 只读不写——粒子是加法发光叠加，不得污染场景深度。
      depthStencil: {
        format: 'depth32float',
        depthWriteEnabled: false,
        depthCompare: 'less',
      },
    });
    this._renderBindGroup = device.createBindGroup({
      layout: renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this._drawParamsBuffer } },
        { binding: 1, resource: { buffer: this._particleBuffer } },
      ],
    });
  }

  /** 当前模式（反查预设表）。 */
  get mode(): ParticleMode {
    const modes = Object.keys(ParticleSystem.presets) as ParticleMode[];
    return modes.find((m) => ParticleSystem.presets[m] === this._preset) ?? 'fire';
  }

  /** 当前发射器位置（归一化 ECEF 的拷贝）。 */
  get origin(): Cartesian3 {
    return new Cartesian3(this._origin[0], this._origin[1], this._origin[2]);
  }

  /**
   * 动态移动发射器。已发射的粒子继续原轨迹飞行，新生的粒子从新位置发射，
   * 无需重建 GPU 资源。
   */
  setOrigin(pos: ParticleOrigin): void {
    this._origin = normalizeOrigin(pos);
  }

  /** 幂等初始化。资源在构造函数中创建（Context 已就绪），此处无需再做事。 */
  prepare(): void {
    /* no-op — resources are created eagerly in the constructor. */
  }

  /**
   * 执行一帧：compute 模拟 pass + 深度测试的渲染 pass，追加在当前帧共享
   * encoder 上（由 `EffectManager` 在场景 composite 之后调用）。
   *
   * @param encoder - 当前帧共享 command encoder
   * @param target  - swap-chain 颜色输出（loadOp 'load'，加法叠加在已渲染场景上）
   * @param time    - 效果统一时间（秒）；内部差分出 dt 并 clamp 到 0.1 s
   */
  execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void {
    if (this._destroyed) return;

    const gbuffer = this._scene.gbufferPass;
    if (!gbuffer) return; // 场景已销毁（防御；构造时已校验）

    // ── dt（EffectManager 不传 dt，由时间差分得到）────────────────────────
    let dt = this._lastTime === null ? 0 : time - this._lastTime;
    this._lastTime = time;
    if (!Number.isFinite(dt) || dt < 0) dt = 0;
    if (dt > MAX_DT) dt = MAX_DT;

    this._time  += dt;
    this._frame += 1;

    // ── Upload sim params ──────────────────────────────────────────────────
    const s = this._simData;
    const p = this._preset;
    s[0]  = this._origin[0];
    s[1]  = this._origin[1];
    s[2]  = this._origin[2];
    s[3]  = dt;
    s[4]  = p.speedRange[0];
    s[5]  = p.speedRange[1];
    s[6]  = p.coneAngle;
    s[7]  = p.gravity;
    s[8]  = this.count;
    s[9]  = this._frame;           // hash seed — varies every frame
    s[10] = this._activeCount;
    s[11] = this._time;
    s[12] = p.shape === 'sphere' ? 1 : 0;
    s[13] = p.lifeRange[0];
    s[14] = p.lifeRange[1];
    this._context.writeBuffer(this._simParamsBuffer, 0, s);

    // ── Upload draw params ─────────────────────────────────────────────────
    const d = this._drawData;
    const cam = this._scene.camera;
    d.set(cam.viewProjectionMatrix.values, 0);
    const tanHalfFov = Math.tan(cam.frustum.fov * 0.5);
    // NDC offset = sizeNorm / (dist * tan(fov/2))：基础系数 CPU 算一次，
    // 每粒子尺寸（sizeRange 米 → 归一化单位）在 vertex 阶段乘上。
    d[16] = 1 / (tanHalfFov * cam.frustum.aspectRatio);
    d[17] = 1 / tanHalfFov;
    d[18] = COLOR_MODE_INDEX[p.colorMode];
    d[20] = p.sizeRange[0];
    d[21] = p.sizeRange[1];
    d[22] = Camera.EARTH_SCALE;
    this._context.writeBuffer(this._drawParamsBuffer, 0, d);

    // ── Compute pass ───────────────────────────────────────────────────────
    const cpass = encoder.beginComputePass({ label: `${this.label}.Compute` });
    cpass.setPipeline(this._computePipeline);
    cpass.setBindGroup(0, this._computeBindGroup);
    cpass.dispatchWorkgroups(Math.ceil(this.count / WORKGROUP_SIZE));
    cpass.end();

    // ── Render pass（加法混合 + G-Buffer 深度测试）────────────────────────
    // depthReadOnly: true 时不可设置 depthLoadOp / depthStoreOp（WebGPU 规范，
    // 与 TransparentPass 同款模式）；深度 view 每帧现取，resize 后自动跟随。
    const pass = encoder.beginRenderPass({
      label: `${this.label}.Draw`,
      colorAttachments: [{
        view: target,
        loadOp:  'load',  // 叠加在已渲染场景之上，不清屏
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: gbuffer.textures.depthView,
        depthReadOnly: true,
      },
    });
    pass.setPipeline(this._renderPipeline);
    pass.setVertexBuffer(0, this._quadBuffer);
    pass.setBindGroup(0, this._renderBindGroup);
    pass.draw(4, this.count);
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

  /**
   * CPU 侧初始状态：每粒子从寿命中随机相位出发并预积分弹道（稳态首帧），
   * 寿命也在 [lifeMin, lifeMax] 随机（与 shader 重生逻辑一致）；
   * 超出发射率预算的粒子直接驻留为死亡状态。
   */
  private _initParticleData(): Float32Array<ArrayBuffer> {
    const data = new Float32Array(this.count * FLOATS_PER_PARTICLE);
    const [ox, oy, oz] = this._origin;
    const mag = Math.hypot(ox, oy, oz) || 1;

    // 地表法线（"up"）与正交基 — 与 compute shader 同一套公式。
    const ux = ox / mag, uy = oy / mag, uz = oz / mag;
    const hx = Math.abs(uz) > 0.9 ? 1 : 0;
    const hz = Math.abs(uz) > 0.9 ? 0 : 1;
    // t1 = normalize(cross(up, helper))
    const c1x = uy * hz;
    const c1y = uz * hx - ux * hz;
    const c1z = -uy * hx;
    const c1m = Math.hypot(c1x, c1y, c1z) || 1;
    const t1x = c1x / c1m, t1y = c1y / c1m, t1z = c1z / c1m;
    // t2 = cross(up, t1)
    const t2x = uy * t1z - uz * t1y;
    const t2y = uz * t1x - ux * t1z;
    const t2z = ux * t1y - uy * t1x;

    const [lifeMin, lifeMax] = this._preset.lifeRange;
    const [speedMin, speedMax] = this._preset.speedRange;
    const { coneAngle, gravity, shape } = this._preset;
    const spherical = shape === 'sphere';

    for (let i = 0; i < this.count; i++) {
      const o = i * FLOATS_PER_PARTICLE;

      if (i >= this._activeCount) {
        // 驻留死亡：age = life，等待被唤醒（与 compute 的 parked 分支一致）。
        data[o + 0] = ox; data[o + 1] = oy; data[o + 2] = oz;
        data[o + 3] = lifeMax;
        data[o + 7] = lifeMax;
        continue;
      }

      const life = lifeMin + Math.random() * (lifeMax - lifeMin);
      const age  = Math.random() * life;

      let dx: number, dy: number, dz: number;
      if (spherical) {
        // 单位球均匀方向（烟花爆点）。
        const z = Math.random() * 2 - 1;
        const a = Math.random() * Math.PI * 2;
        const r = Math.sqrt(Math.max(0, 1 - z * z));
        dx = r * Math.cos(a); dy = r * Math.sin(a); dz = z;
      } else {
        const theta = Math.random() * Math.PI * 2;
        const cosA  = 1 - Math.random() * (1 - Math.cos(coneAngle));
        const sinA  = Math.sqrt(Math.max(0, 1 - cosA * cosA));
        const ct    = Math.cos(theta);
        const st    = Math.sin(theta);
        dx = ux * cosA + (t1x * ct + t2x * st) * sinA;
        dy = uy * cosA + (t1y * ct + t2y * st) * sinA;
        dz = uz * cosA + (t1z * ct + t2z * st) * sinA;
      }

      const speed = speedMin + Math.random() * (speedMax - speedMin);
      const t = age;

      data[o + 0] = ox + dx * speed * t - 0.5 * gravity * ux * t * t;
      data[o + 1] = oy + dy * speed * t - 0.5 * gravity * uy * t * t;
      data[o + 2] = oz + dz * speed * t - 0.5 * gravity * uz * t * t;
      data[o + 3] = age;
      data[o + 4] = dx * speed;
      data[o + 5] = dy * speed;
      data[o + 6] = dz * speed;
      data[o + 7] = life;
    }
    return data;
  }
}

export default ParticleSystem;
