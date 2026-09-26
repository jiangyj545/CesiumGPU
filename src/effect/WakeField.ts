import type { IPass } from '../scene/passes/IPass';
import { Context } from '../renderer/Context';
import {
  buildWakeFieldShader,
  WAKE_FIELD_PARAMS_BYTES,
  WAKE_FIELD_WORKGROUP,
} from '../shaders/WakeFieldShader';

/**
 * WakeField — 交互船迹波场（tidewater WakeSim 的 L2 视觉等价简化版）。
 *
 * ## 保留的 tidewater 三关键设计
 * 1. **环形寻址跟随窗口**：N×N 场纹理即整个跟随窗口，世界格 I 存 texel `I mod N`；
 *    窗口按整格滚动零拷贝（kernel 内做世界格一致性检查，见 WakeFieldShader `wfLoad`）。
 * 2. **Havelock 移动压力**：船是移动压力分布——船头正高斯 + 船尾空穴负高斯，
 *    幅值 K·U²/2g 动压头；压力经 ∇²P 进竖直加速度（平衡解 h ≈ −P），不直接写高度。
 * 3. **开尔文波视觉 + 海绵**：泄漏波动方程（symplectic Euler）从移动偶极压力源
 *    天然长出 V 形尾迹与横波；窗口边缘 ~10 格 sponge 渐变吸收防反射。
 *
 * ## 物理模型与 tidewater 的差异（误差方向）
 * - tidewater：512² 线性自由面**谱方法**（3-dispatch 行列 FFT），精确色散
 *   L = |k|·tanh(|k|·d)（4 深度算子逐格混合），真实型线浸没压力图，涡粘耗散
 *   通量形式，湍流/泡沫/曝气多场。
 * - 本类：N² **网格域**泄漏波动方程 w' = w + dt·c²∇²(h+P)，单一代表相速
 *   c ≈ max(1.5, 0.34U)（CFL clamp 0.45·cell/dt）→ **非色散**波动。差异：
 *   - 波长不遵循深水 λ = 2πU²/g 缩放：短波传播偏快（深水中应更慢），误差方向
 *     是横波偏长、横波/发散波间距比失真；
 *   - 尾迹角：非色散楔半角 = arcsin(c/U)，取 c = 0.34U ≈ sin(19.47°)·U 使
 *     航行态楔角 ≈ 19.9°，视觉对齐开尔文角；U < c（≈4.4 m/s 下）不锁波，
 *     退化为环形涟漪 + 压痕（慢船尾迹本就微弱）；
 *   - 船体压力用双高斯近似代替型线浸没积分（无 trim/heave 调度）；
 *   - 无变深折射、无障碍物、无湍流场；泡沫为 |h| 超阈 + 尾流注入的单场
 *     累积-衰减。
 *
 * ## ping-pong 方案
 * rgba16float 不能 read_write storage：用**两张** rgba16float 纹理
 * （STORAGE_BINDING | TEXTURE_BINDING）ping-pong——kernel 从上一帧纹理
 * `textureLoad`（邻域 wrap 在 mod 域），`textureStore` 到另一张；两个预建
 * bindGroup 交替。选它而非 "storage buffer + 输出纹理" 的理由：省一次输出
 * 拷贝 dispatch、GPU 资源 2 个而非 3 个、B1 直接拿 `fieldView` 采样；半浮点
 * 精度对 ±1 m 波高与 0-1 泡沫足够（CesiumGPU 既有 rgba16float 纹理同款）。
 *
 * 坐标语义：纯局部平面场（x=东向米、z=北向米），与 OceanFFTWater 的 planeXZ
 * 同域；本类不持有经纬度，坐标域对齐由使用方（B1）保证。
 *
 * 生命周期：构造即校验；`prepare()` 幂等建管线（失败置 `_failed` 闸——每帧
 * execute 重入 prepare 不再重试，OceanFFTWater 同款）；`destroy()` 后不可再用。
 */

/** 船源：位置/航向/航速/尺寸（局部平面坐标域，米）。 */
export interface WakeSource {
  /** 船在局部平面坐标（东/北，米，与 OceanFFTWater 的 planeXZ 同域）。 */
  planeX: number;
  /** 北向米（见 {@link WakeSource.planeX}）。 */
  planeZ: number;
  /** 航向（弧度，0=东，逆时针）。 */
  heading: number;
  /** 航速 m/s（0 = 无压力源，仅余波衰减）。 */
  speed: number;
  /** 船长（米，压力分布纵向尺度）。 */
  lengthM: number;
  /** 船宽（米，压力分布横向尺度）。 */
  beamM: number;
}

export interface WakeFieldOptions {
  /** 网格边长（向上取整到 8 的倍数，clamp 64..512）。Default: `256` */
  gridN?: number;
  /** 跟随窗口边长（米）。Default: `200` */
  windowSizeM?: number;
  /** 物理阻尼（1/s，作用于 h 与 w）。Default: `0.02` */
  damping?: number;
  /** 泡沫生成增益。Default: `0.6` */
  foamGain?: number;
}

/** 重力（m/s²，动压头用；与 tidewater GRAVITY 同值量级）。 */
const GRAVITY = 9.81;
/** 动压系数（tidewater WakeSim.dynamicPressure 同款）。 */
const PRESSURE_K = 0.35;
/**
 * 代表相速 = factor·U。非色散模型的行波楔半角 = arcsin(c/U)（U > c 时波锁定），
 * factor = sin(19.47°) ≈ 0.34 → 楔角 ≈ 19.9°，视觉对齐开尔文尾迹角；
 * U < c（≈4.4 m/s 以下）不锁波，退化为环形涟漪 + 船体压痕（慢船尾迹本就微弱）。
 */
const WAVE_SPEED_FACTOR = 0.34;
const MIN_WAVE_SPEED = 1.5;
/** CFL 安全上限：c ≤ 0.45·cell/dt。 */
const CFL_LIMIT = 0.45;
/** dt 上限（秒）：稳定步长，切后台/掉帧不爆模拟。 */
const MAX_DT = 1 / 30;
/** foamThr（米）：|h| 超此值开始生泡沫。 */
const FOAM_THR = 0.06;
/** 涡粘（m²/s，作用于 w 的通量形式）：~k² 阻网格尺度噪声，米级波浪无感。 */
const WAKE_VISC = 0.05;

/** uniform f32 索引（与 WakeFieldShader 的 WakeParams struct 对齐）。 */
const P_ORIGIN = 0;  // origin.xy, prevOrigin.xy
const P_BOAT = 4;    // planeX, planeZ, cos, sin
const P_GRID = 8;    // cellM, dt, reset, dynHead
const P_PHYS = 12;   // c², damping, foamGain, foamThr
const P_HULL = 16;   // lengthM, beamM, speed, 涡粘 nu

export class WakeField implements IPass {
  /** 调试标签（RenderPass/BindGroup descriptor label）。 */
  readonly label = 'WakeField';

  /** 可见性开关（EffectManager 依据它跳过 execute）。 */
  show = true;

  private _context: Context;
  private _gridN: number;
  private _cellM: number;
  private _damping: number;
  private _foamGain: number;

  private _failed = false;
  private _ready = false;
  private _destroyed = false;

  private _textures: GPUTexture[] = [];
  private _views: GPUTextureView[] = [];
  private _bindGroups: GPUBindGroup[] = [];
  private _sampler: GPUSampler | null = null;
  private _pipeline: GPUComputePipeline | null = null;
  private _ubo: GPUBuffer | null = null;
  /** _ping = 最近写入的纹理下标；下一帧从它读、写到 ^1。 */
  private _ping = 0;

  private _params = new Float32Array(WAKE_FIELD_PARAMS_BYTES / 4);

  /** 窗口最小角（世界格，整数）与中心格。 */
  private _origin = { x: 0, y: 0 };
  private _prevOrigin = { x: 0, y: 0 };
  private _center = { x: 0, y: 0 };
  private _hasWindow = false;
  private _reset = 1;

  private _boat: WakeSource | null = null;
  private _lastTime: number | null = null;

  constructor(context: Context, options: WakeFieldOptions = {}) {
    if (!context.initialized) {
      throw new Error('WakeField: Context.initialize() must complete first.');
    }
    const rawN = Math.round(options.gridN ?? 256);
    this._gridN = Math.max(64, Math.min(512, Math.ceil(rawN / WAKE_FIELD_WORKGROUP) * WAKE_FIELD_WORKGROUP));
    const winM = options.windowSizeM ?? 200;
    this._cellM = Number.isFinite(winM) && winM > 0 ? winM / this._gridN : 200 / this._gridN;
    this._damping = Number.isFinite(options.damping ?? NaN) ? (options.damping as number) : 0.02;
    this._foamGain = Number.isFinite(options.foamGain ?? NaN) ? (options.foamGain as number) : 0.6;
    this._context = context;
  }

  /** 波场纹理 view：R=波高(米) G=泡沫(0-1) B=竖直流速 A=压力头。prepare 前为 null。 */
  get fieldView(): GPUTextureView | null {
    return this._ready ? this._views[this._ping] : null;
  }

  /** 线性 clamp 采样器（边缘已是海绵零区，clamp 优于 wrap）。prepare 前为 null。 */
  get fieldSampler(): GPUSampler | null {
    return this._sampler;
  }

  /** 窗口原点（当前滚定格，米）。窗口未初始化时为 {x:0, z:0}。 */
  get windowOrigin(): { x: number; z: number } {
    return { x: this._origin.x * this._cellM, z: this._origin.y * this._cellM };
  }

  /** 网格边长（格）。 */
  get gridN(): number {
    return this._gridN;
  }

  /** 单格边长（米）。 */
  get cellM(): number {
    return this._cellM;
  }

  /**
   * 每帧调用：记录船状态并驱动窗口整格滚动目标；注入在 execute 的 kernel 内完成。
   * 非有限输入被忽略（保持上一帧状态）。
   */
  updateBoat(boat: WakeSource): void {
    if (
      !Number.isFinite(boat.planeX) || !Number.isFinite(boat.planeZ) ||
      !Number.isFinite(boat.heading) || !Number.isFinite(boat.speed) ||
      !Number.isFinite(boat.lengthM) || !Number.isFinite(boat.beamM)
    ) {
      return;
    }
    this._boat = {
      planeX: boat.planeX,
      planeZ: boat.planeZ,
      heading: boat.heading,
      speed: Math.max(0, boat.speed),
      lengthM: boat.lengthM,
      beamM: boat.beamM,
    };
  }

  /** 幂等初始化（管线 / bindGroup）。失败置闸一次报错，不再每帧重试。 */
  prepare(): void {
    if (this._failed || this._ready || this._destroyed) return;
    try {
      const device = this._context.device;
      const N = this._gridN;

      // 双缓冲场纹理：本帧读 A 写 B，下帧读 B 写 A（ping-pong 交替）
      for (const suffix of ['A', 'B']) {
        const tex = device.createTexture({
          label: `WakeField.Field${suffix}`,
          size: { width: N, height: N },
          format: 'rgba16float',
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this._textures.push(tex);
        this._views.push(tex.createView({ label: `WakeField.Field${suffix}.View` }));
      }

      this._sampler = device.createSampler({
        label: 'WakeField.Sampler',
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
      });

      this._ubo = device.createBuffer({
        label: 'WakeField.Params',
        size: WAKE_FIELD_PARAMS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const module = this._context.createShaderModule(
        buildWakeFieldShader(N), 'WakeField.ComputeModule',
      );
      // minBindingSize 须显式，否则管线验证失败（OceanFFTWater 血泪教训）
      const bgl = device.createBindGroupLayout({
        label: 'WakeField.BGL',
        entries: [
          {
            binding: 0, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'uniform', minBindingSize: WAKE_FIELD_PARAMS_BYTES },
          },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
          {
            binding: 2, visibility: GPUShaderStage.COMPUTE,
            storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' },
          },
        ],
      });
      this._pipeline = device.createComputePipeline({
        label: 'WakeField.Step',
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        compute: { module, entryPoint: 'main' },
      });
      // bg[k]：从纹理 k 读、写纹理 k^1
      for (let k = 0; k < 2; k++) {
        this._bindGroups.push(device.createBindGroup({
          label: `WakeField.BG${k}`,
          layout: bgl,
          entries: [
            { binding: 0, resource: { buffer: this._ubo } },
            { binding: 1, resource: this._views[k] },
            { binding: 2, resource: this._views[k ^ 1] },
          ],
        }));
      }

      this._ready = true;
    } catch (e) {
      // execute 每帧重入 prepare：初始化抛错后若不设闸会每帧重试刷屏，
      // 且 GPU 资源已部分创建，重试必然同样失败——一次报错并禁用本效果。
      this._failed = true;
      console.error(`[${this.label}] 初始化失败，效果已禁用:`, e);
    }
  }

  /**
   * 每帧执行：窗口整格滚动检测（updateBoat 的位置驱动）→ 压力注入 + 物理步进
   * （单 compute pass，1 dispatch；dt 由 execute 帧差分取得，clamp [0, 1/30]）。
   */
  execute(encoder: GPUCommandEncoder, _target: GPUTextureView, time: number): void {
    if (this._destroyed) return;
    this.prepare();
    if (this._failed || !this._ready) return;

    let dt = this._lastTime === null ? 0 : time - this._lastTime;
    this._lastTime = time;
    if (!Number.isFinite(dt) || dt < 0) dt = 0;
    if (dt > MAX_DT) dt = MAX_DT;

    const boat = this._boat;
    if (boat) this._scrollWindow(boat);

    // ---- uniform 打包（5 × vec4，见 WakeFieldShader struct 注释）
    const s = this._params;
    const b = boat ?? { planeX: 0, planeZ: 0, heading: 0, speed: 0, lengthM: 2, beamM: 0.8 };
    const cosH = Math.cos(b.heading);
    const sinH = Math.sin(b.heading);
    const speed = b.speed;
    // 代表相速：横波驻相速度近似；CFL clamp 保证显式格式稳定（dt=0 首帧不约束）
    const dtEff = Math.max(dt, 1 / 240);
    const c = Math.min(Math.max(MIN_WAVE_SPEED, WAVE_SPEED_FACTOR * speed), CFL_LIMIT * this._cellM / dtEff);
    const dynHead = PRESSURE_K * speed * speed / (2 * GRAVITY);

    s[P_ORIGIN + 0] = this._origin.x;
    s[P_ORIGIN + 1] = this._origin.y;
    s[P_ORIGIN + 2] = this._prevOrigin.x;
    s[P_ORIGIN + 3] = this._prevOrigin.y;
    s[P_BOAT + 0] = b.planeX;
    s[P_BOAT + 1] = b.planeZ;
    s[P_BOAT + 2] = cosH;
    s[P_BOAT + 3] = sinH;
    s[P_GRID + 0] = this._cellM;
    s[P_GRID + 1] = dt;
    s[P_GRID + 2] = this._reset;
    s[P_GRID + 3] = dynHead;
    s[P_PHYS + 0] = c * c;
    s[P_PHYS + 1] = this._damping;
    s[P_PHYS + 2] = this._foamGain;
    s[P_PHYS + 3] = FOAM_THR;
    s[P_HULL + 0] = b.lengthM;
    s[P_HULL + 1] = b.beamM;
    s[P_HULL + 2] = speed;
    s[P_HULL + 3] = WAKE_VISC;
    this._context.writeBuffer(this._ubo!, 0, s);

    // ---- 物理步进（1 dispatch：压力注入 + 步进 + 海绵 + 泡沫）
    const workgroups = this._gridN / WAKE_FIELD_WORKGROUP;
    const cpass = encoder.beginComputePass({ label: `${this.label}.Step` });
    cpass.setPipeline(this._pipeline!);
    cpass.setBindGroup(0, this._bindGroups[this._ping]);
    cpass.dispatchWorkgroups(workgroups, workgroups);
    cpass.end();

    this._ping ^= 1;
    this._reset = 0;
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
    this._sampler = null;
  }

  /**
   * 局部平面坐标 → 波场 uv（B1 海面采样用）；窗口范围外（含 NaN）返回 null。
   * uv 域 [0,1) 对应窗口，v 轴与纹理行（北向 z）一致。
   */
  planeToUv(planeX: number, planeZ: number): { u: number; v: number } | null {
    if (!this._ready || !this._hasWindow) return null;
    const u = (planeX / this._cellM - this._origin.x) / this._gridN;
    const v = (planeZ / this._cellM - this._origin.y) / this._gridN;
    if (!(u >= 0) || !(u < 1) || !(v >= 0) || !(v < 1)) return null;
    return { u, v };
  }

  /**
   * 窗口整格滚动（tidewater update() 同款）：
   * - 船离窗口中心超过 box 格 → 推动中心（整格），零拷贝；
   * - 距离超过整窗（首次/传送）→ 重定中心 + reset（全场视为新水）。
   */
  private _scrollWindow(boat: WakeSource): void {
    const N = this._gridN;
    const bx = boat.planeX / this._cellM;
    const bz = boat.planeZ / this._cellM;
    const box = Math.round(N * 0.3);
    const c = this._center;
    if (!this._hasWindow || Math.abs(bx - c.x) > N || Math.abs(bz - c.y) > N) {
      c.x = Math.round(bx);
      c.y = Math.round(bz);
      this._hasWindow = true;
      this._reset = 1;
    }
    if (bx - c.x > box) c.x = Math.ceil(bx - box);
    if (c.x - bx > box) c.x = Math.floor(bx + box);
    if (bz - c.y > box) c.y = Math.ceil(bz - box);
    if (c.y - bz > box) c.y = Math.floor(bz + box);
    this._prevOrigin.x = this._origin.x;
    this._prevOrigin.y = this._origin.y;
    this._origin.x = c.x - N / 2;
    this._origin.y = c.y - N / 2;
    if (this._reset > 0.5) {
      this._prevOrigin.x = this._origin.x;
      this._prevOrigin.y = this._origin.y;
    }
  }
}

export default WakeField;
