import type { IPass } from '../scene/passes/IPass';
import type { Camera } from '../scene/Camera';
import type { Context } from '../renderer/Context';
import { ecefFromLonLatHeight } from '../terrain/TerrainProvider';
import {
  OCEAN_FFT_INIT_SPECTRUM_SHADER,
  OCEAN_FFT_CONJUGATE_SHADER,
  OCEAN_FFT_COPY_H0_SHADER,
  OCEAN_FFT_ROW_SHADER,
  OCEAN_FFT_COLUMN_SHADER,
  OCEAN_FFT_MIP_A_DISP_SHADER,
  OCEAN_FFT_MIP_A_DERIV_SHADER,
  OCEAN_FFT_MIP_B_DISP_SHADER,
  OCEAN_FFT_MIP_B_DERIV_SHADER,
  OCEAN_FFT_RENDER_SHADER,
  OCEAN_PARAMS_UBO_BYTES,
  OCEAN_RENDER_PARAMS_UBO_BYTES,
  OCEAN_BREAK_PARAMS_UBO_BYTES,
} from '../shaders/OceanFFTShader';
import { CAMERA_UBO_SIZE, writeCameraUniforms } from '../renderer/CameraUniforms';
import { SHADOW_RECEIVER_UBO_SIZE } from '../shadow/ShadowMap';
import type { ShadowReceiverBinding } from '../shadow/ShadowMap';
import { CLOUD_SHADOW_PARAMS_UBO_BYTES } from '../shaders/CloudShader';
import type { CloudShadowInfo } from './CloudLayer';
import type { FoamPatternTexture } from './FoamPatternTexture';
import type { WakeField } from './WakeField';
import type { SeabedHeightField } from './SeabedHeightField';
import type { FFTDataSource } from './OceanCaustics';

/** FFT 边长（与 shader 内 N 一致）。 */
export const OCEAN_FFT_SIZE = 256;
const N = OCEAN_FFT_SIZE;
/** 级联数。 */
const CASCADES = 4;
/** 级联平铺尺寸（米，非整数比例防重复，tidewater DEFAULT_CASCADE_SIZES）。 */
const CASCADE_SIZES = [733, 157, 33.3, 7.1];
/** 重力加速度（JONSWAP / 色散关系）。 */
const GRAVITY = 9.81;
const TWO_PI = Math.PI * 2;
/** 谱随机种子（tidewater 默认）。 */
const SEED = 1337;
/** 水深（米，色散关系 TMA 修正）。 */
const DEPTH_METERS = 500;
/** 位移/导数纹理 mip 级数（256 → 9 级）。 */
const MIP_LEVELS = 9;
/** 每纬度度数对应米数（小范围线性换算）。 */
const METERS_PER_DEG = 111320;

/** 256 点 IFFT 的 dispatch 尺寸辅助（tidewater update() 同款）。 */
const N_OVER_16 = N / 16;
const N_OVER_32 = N / 32;

/** AtmosphereParams/SkyParams UBO 字节（64B each，布局单一来源 AtmosphereSkyShader.ts）。 */
const ATMO_UBO_BYTES = 64;

/** 归一化 ECEF：1 内部单位 = 6378137 米（Camera.EARTH_SCALE 同值）。 */
const EARTH_RADIUS_M = 6378137;
/**
 * 海面云影强度（Track C：CloudShadowInfo 契约不含强度，海面侧固定——与
 * CloudLayer 默认 shadowIntensity 同值；UBO params.y = 0 时 shader 短路恒 1）。
 */
const CLOUD_SHADOW_SEA_INTENSITY = 0.65;

/** 单个波浪系统参数（tidewater WaveSystem）。 */
interface WaveSystem {
  scale: number;
  windSpeed: number;      // m/s
  windDirection: number;  // 度
  fetch: number;          // km
  spreadBlend: number;
  swell: number;
  peakEnhancement: number;
  shortWavesFade: number;
}

export interface OceanFFTWaterOptions {
  /** WebGPU 上下文（通常 `viewer.scene.context`）。 */
  context: Context;
  /** 海面中心经纬度（度），必填。 */
  center: [number, number];
  /** 海面边长（公里，正方形网格）。Default: `2.5` */
  extentKm?: number;
  /** 每边网格段数（>= 2）。Default: `256` */
  segments?: number;
  /** 水位（米，折入 ECEF 模长）。Default: `0` */
  waterLevel?: number;
  /** 风速 m/s（驱动 local 波浪系统谱）。Default: `8` */
  windSpeed?: number;
  /** 风向（度）。Default: `20` */
  windDirection?: number;
  /** λ 水平位移缩放。Default: `0.9` */
  choppiness?: number;
  /** 效果时间倍率。Default: `1` */
  timeScale?: number;
  /** 太阳方向（指向太阳，归一化 ECEF）。Default: `[0.4, 0.5, 0.77]` */
  sunDirection?: [number, number, number];
  /** 泡沫图案纹理（A3）。绑定后 FS 用图案阈值化替代 hash 噪声调制；null 回退 hash 路径。 */
  foamPattern?: FoamPatternTexture;
  /** 船迹波场（A4）。VS 位移叠加波高、FS 泡沫叠加；null 时不采样。须先于本效果 add 到 effects。 */
  wake?: WakeField;
  /** 折射：FS 扰动采样场景颜色历史混合进水色（需 Scene.sceneColorHistory 提供）。Default: `false` */
  refraction?: boolean;
}

/** 一个 compute pipeline + 配套 bind group（每 kernel 一份）。 */
interface ComputeUnit {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
}

/**
 * L5 物理大气绑定源（AtmosphereSky 四资源；字段任一为 null → fallback +
 * featureFlags bit0=0 短路回旧平色路径）。经 {@link OceanFFTWater.bindAtmosphere}
 * 以 getter 形式每帧惰性求值——AtmosphereSky prepare 后资源才存在，且
 * skyViewTexture 等视图 getter 需身份稳定（bind group 缓存键）。
 */
export interface OceanAtmosphereBinding {
  /** skyView LUT 视图（`AtmosphereSky.skyViewTexture`）。 */
  skyView: GPUTextureView | null;
  /** transmittance LUT 视图（`AtmosphereSky.transmittanceTexture`）。 */
  transmittance: GPUTextureView | null;
  /** AtmosphereParams UBO 64B（`AtmosphereSky.atmosphereParamsBuffer`）。 */
  atmosphereParams: GPUBuffer | null;
  /** SkyParams UBO 64B（`AtmosphereSky.skyParamsBuffer`）。 */
  skyParams: GPUBuffer | null;
}

/** T4 深度感知破碎波参数（{@link OceanFFTWater.setBreakWave}）。 */
export interface OceanBreakWaveConfig {
  /** 破碎带内振幅乘数（disp.y 缩放）。Default: `1.25` */
  ampScale?: number;
  /** 波峰前倾强度（沿梯度水平位移）。Default: `0.55` */
  lean?: number;
  /** 泡沫花边增益。Default: `0.85` */
  laceGain?: number;
  /** 破碎带近沿水深（米，全量破碎）。Default: `3` */
  bandNearM?: number;
  /** 破碎带远沿水深（米，须 > bandNearM）。Default: `8` */
  bandFarM?: number;
  /** 内卷水幕染色强度。Default: `0.55` */
  curlShade?: number;
  /** 花边 smoothstep 阈值（米，更贴岸的细沫带）。Default: `1.6` */
  laceThresholdM?: number;
}

/**
 * T4 破碎喷雾吸附源（{@link OceanFFTWater.breakSpraySeabed} → BreakSpray）：
 * compute 端 textureLoad seabed + disp 级联0 做粒子高度吸附与水深门。
 */
export interface OceanSeabedSprayBinding {
  /** 海床高度场纹理视图（r32float，R = 海拔米）。 */
  view: GPUTextureView;
  /** 位移纹理 2d-array 视图（.y = 波高 Dy 米；compute 吸附消费级联0）。 */
  dispView: GPUTextureView;
  /** 海面中心（归一化 ECEF）。 */
  centerEcef: { x: number; y: number; z: number };
  /** 海面 ENU 基（每米归一化 ECEF 增量）。 */
  eastDir: { x: number; y: number; z: number };
  northDir: { x: number; y: number; z: number };
  /** seabed uv 仿射：uv = planeXZ * uvScale + uvOffset（与渲染 FS 同源）。 */
  uvOffset: { x: number; y: number };
  uvScale: { x: number; y: number };
}

/** OceanParams uniform 的 f32 元素索引（与 OceanFFTShader 的 struct 对齐）。 */
const P_SIZES = 0;
const P_CUTS = 16;
const P_SYSA = 32;
const P_SYSB = 40;
const P_CHOPPINESS = 48;
const P_FOAM_BIAS = 49;
const P_FOAM_GAIN = 50;
const P_FOAM_DECAY = 51;
const P_FOAM_ADD = 52;
const P_TIME = 53;
const P_DEPTH = 54;
const P_DT = 55;
/** seed 是 u32，走同一 ArrayBuffer 的 Uint32Array 视图。 */
const P_SEED_U32 = 56;
const P_EAST = 60;
const P_NORTH = 64;
const P_UP = 68;

/**
 * OceanFFTWater - FFT 谱海洋水体特效（移植 tidewater OceanFFT.js）。
 *
 * JONSWAP×方向谱生成 h0(k)，每帧 GPU 上跑行/列两级打包复数 IFFT，
 * 产出 4 级联位移/导数 rgba16float 2d-array（含 9 级 mip），泡沫由
 * Jacobian 压缩驱动带指数衰减的持久累积。海面为局部平面：CPU 构建
 * 经纬网格（[ecefXYZ, planeXZ_m] 顶点），ENU 基以「每米归一化 ECEF
 * 增量」传入 shader（数值差分自动含极轴压扁）。
 *
 * VS 只位移前 3 个大级联（最小 7.1m 级联留给 FS 法线/泡沫）；clip 位置
 * 用「基准与位移分别过 viewProj 再相加」避免 1m 波幅（~1.6e-7 归一化
 * 单位）在 f32 worldPos 上直接相加的精度损失。
 *
 * B1 集成：本类同时是 {@link FFTDataSource}（A1 焦散数据源），并支持
 * 泡沫图案（A3）/ 船迹波场（A4，须先于本效果 add 到 effects 以取当帧
 * fieldView）/ 折射混色（Scene.sceneColorHistory）三条可选增强路径，
 * 均可空缺、空缺时走原路径（零回归）。
 *
 * @example
 * ```typescript
 * const ocean = scene.effects.add(new OceanFFTWater(scene.context, {
 *   center: [121.5, 31.0],
 * }));
 * ocean.setWindSpeed(10);
 * ```
 */
export class OceanFFTWater implements IPass {
  readonly label = 'OceanFFTWater';

  /** 可见性开关（EffectManager 每帧检查）。 */
  show = true;

  private _context: Context;
  private _camera: Camera | null = null;
  private readonly _extentKm: number;
  private readonly _segments: number;
  private readonly _waterLevel: number;
  private _choppiness: number;
  private _timeScale: number;
  private readonly _sunDir: [number, number, number];
  /** B1 集成：泡沫图案 / 船迹 / 折射（均可空，空 = 原路径零回归）。 */
  private _foamPattern: FoamPatternTexture | null = null;
  private _wake: WakeField | null = null;
  private _refraction: boolean;
  /** L4-A 岸线泡沫：海床高度场（null = 原路径零回归）。 */
  private _seabed: SeabedHeightField | null = null;
  /** 岸线泡沫强度（UBO shoreParams.y，默认 1.0，0 = 等效关闭）。 */
  private _shoreStrength = 1.0;
  /** 海面中心归一化 ECEF（水位折入模长）+ east/north 基——seabedUv 原点差投影用。 */
  private readonly _centerEcef: [number, number, number];
  private readonly _eastDir: [number, number, number];
  private readonly _northDir: [number, number, number];
  /** planeXZ → seabed uv 仿射系数：uv = planeXZ*scale + offset（bindSeabed 时算好）。 */
  private _seabedUvOffset: [number, number] = [0.5, 0.5];
  private _seabedUvScale: [number, number] = [1, 1];
  /** 场景颜色历史来源（Scene.sceneColorHistory），null = 折射不混色。 */
  private _sceneHistorySource: (() => GPUTextureView | null) | null = null;
  /** L3：CSSM 阴影源 getter（`() => scene.shadow.receiverBinding()`）。null = 无阴影。 */
  private _shadowSource: (() => ShadowReceiverBinding | null) | null = null;
  /** Track C：海面云影源 getter（`() => clouds.cloudShadow`）。null = 无云影。 */
  private _cloudShadowSource: (() => CloudShadowInfo | null) | null = null;
  /** L3：水下视角（从下看水面：法线翻转 + 透射青绿 + 高光弱化）。 */
  private _underwaterView = false;
  /** 导数纹理全 mip 2d-array 视图（FFTDataSource 公开 + 渲染绑定共用）。 */
  private _derivTextureView: GPUTextureView | null = null;
  /** 位移纹理全 mip 2d-array 视图（T4 BreakSpray compute 吸附用，缓存身份）。 */
  private _dispArrayView: GPUTextureView | null = null;
  /** 折射/船迹用 linear clamp 采样器（repeat 主采样器不适合非平铺纹理）。 */
  private _clampSampler: GPUSampler | null = null;
  /** 未绑定 foam/wake/history 时的 1×1 零纹理（BGL 恒全绑定的回退资源）。 */
  private _fallbackTex: GPUTexture | null = null;
  private _fallbackView: GPUTextureView | null = null;
  /** L4-A：seabed 未绑定/已 dispose 时的 1×1 r32float 占位（R = -1000 极深 = 无泡沫）。 */
  private _seabedFallbackTex: GPUTexture | null = null;
  private _seabedFallbackView: GPUTextureView | null = null;
  /** 渲染参数暂存（每帧整体写入：静态段 + 动态 flags/wakeUv）。 */
  private readonly _renderParamsF32: Float32Array;
  /** 双波浪系统：local 风浪 + 远场涌浪（tidewater 默认参数）。 */
  private readonly _local: WaveSystem;
  private readonly _swell: WaveSystem;

  /** OceanParams uniform 载荷（288B，F32/U32 双视图写同一 buffer）。 */
  private readonly _paramsBytes: ArrayBuffer;
  private readonly _paramsF32: Float32Array;
  private readonly _paramsU32: Uint32Array;
  /** 谱失效标志：置位后下一帧 execute 重跑 init/conjugate/copyH0。 */
  private _needsSpectrum = true;
  /** 初始化失败闸：GPU 资源创建/绑定抛错后置位，只 console.error 一次并停用本效果。 */
  private _failed = false;

  private _prevTime = -1;
  private _timeAccum = 0;

  private _geometry: { vertices: Float32Array; indices: Uint32Array; indexCount: number };

  // ── 渲染侧 GPU 资源 ──
  private _renderPipeline: GPURenderPipeline | null = null;
  /** T5 真 HDR：Scene 经 TargetFormatAware 协议下发的场景目标格式。
   *  null = 未下发（跟随 preferredFormat 旧行为，postProcess 关闭的 90 页零回归）。 */
  private _targetFormat: GPUTextureFormat | null = null;
  private _renderBindGroup: GPUBindGroup | null = null;
  private _renderBindGroupLayout: GPUBindGroupLayout | null = null;
  /** bind group 缓存键：foam/wake/history 视图身份变化时才重建（wake ping-pong）。 */
  private _bgFoam: GPUTextureView | null = null;
  private _bgWake: GPUTextureView | null = null;
  private _bgHist: GPUTextureView | null = null;
  /** L3 阴影缓存键：源 UBO 身份（ShadowMap 重建 GPU 资源后引用会换）。 */
  private _bgShadow: GPUBuffer | null = null;
  /** L4-A 缓存键：seabed 高度场视图身份（bindSeabed 切换 / dispose 后换占位）。 */
  private _bgSeabed: GPUTextureView | null = null;
  /** L5：物理大气源 getter（`() => sky 四资源快照`）。null = 旧平色路径。 */
  private _atmoSource: (() => OceanAtmosphereBinding | null) | null = null;
  /** L5 特性开关：天空反射（bit0，须已 bindAtmosphere）/ 太阳染色（bit3）。 */
  private _skyReflectOn = true;
  private _sunTintOn = true;
  /** L5 SSS 总强度（RP[22]，0 = 关 / bit1）。 */
  private _sssStrength = 0;
  /** T4 破碎波配置（null = 关闭，gain.x=0 零回归）。 */
  private _breakCfg: OceanBreakWaveConfig | null = null;
  /** T4 破碎波 UBO 载荷（64B，每帧整体写入）。 */
  private readonly _breakParamsF32 = new Float32Array(OCEAN_BREAK_PARAMS_UBO_BYTES / 4);
  /** L5 大气 fallback：全零 UBO（binding 17/18 共用；LUT 复用 _fallbackView 1×1 黑）。 */
  private _atmoFallbackUBO: GPUBuffer | null = null;
  /** T4 破碎波 UBO。 */
  private _breakParamsUBO: GPUBuffer | null = null;
  /** L5 bind group 缓存键（大气 LUT ×2 + UBO ×2，身份变更才重建）。 */
  private _bgAtmoSv: GPUTextureView | null = null;
  private _bgAtmoTr: GPUTextureView | null = null;
  private _bgAtmoUboA: GPUBuffer | null = null;
  private _bgAtmoUboS: GPUBuffer | null = null;
  /** Track C 缓存键：云影贴图视图身份（绑定/解绑、CloudLayer 失效后变化）。 */
  private _bgCloudTex: GPUTextureView | null = null;
  /** L3：阴影未激活时的占位资源（1×1 depth 2d-array + 全零 UBO + comparison）。 */
  private _shadowFallbackTex: GPUTexture | null = null;
  private _shadowFallbackView: GPUTextureView | null = null;
  private _shadowFallbackUBO: GPUBuffer | null = null;
  private _shadowCmpSampler: GPUSampler | null = null;
  /** Track C：云影未激活时的占位（1×1 黑 rgba8unorm + 全零 96B UBO）。 */
  private _cloudFallbackTex: GPUTexture | null = null;
  private _cloudFallbackView: GPUTextureView | null = null;
  private _cloudShadowUBO: GPUBuffer | null = null;
  /** CloudShadowParams UBO 载荷（96B，布局单一来源 CloudShader.ts）。 */
  private readonly _cloudShadowF32 = new Float32Array(CLOUD_SHADOW_PARAMS_UBO_BYTES / 4);
  private _sampler: GPUSampler | null = null;
  private _vertexBuffer: GPUBuffer | null = null;
  private _indexBuffer: GPUBuffer | null = null;
  private _cameraUBO: GPUBuffer | null = null;
  private _oceanUBO: GPUBuffer | null = null;
  private _renderParamsUBO: GPUBuffer | null = null;

  // ── compute 侧 GPU 资源 ──
  private _h0: GPUBuffer | null = null;
  private _waveData: GPUBuffer | null = null;
  private _tmp: GPUBuffer | null = null;
  private _foam: GPUBuffer | null = null;
  private _mipSrc: GPUBuffer | null = null;
  private _mipMid: GPUBuffer | null = null;
  private _dispTex: GPUTexture | null = null;
  private _derivTex: GPUTexture | null = null;
  /** level 0-8 的 write-only storage 视图（column/mip kernel 用）。 */
  private _dispViews: GPUTextureView[] = [];
  private _derivViews: GPUTextureView[] = [];

  private _initK: ComputeUnit | null = null;
  private _conjK: ComputeUnit | null = null;
  private _copyK: ComputeUnit | null = null;
  private _rowK: ComputeUnit | null = null;
  private _colK: ComputeUnit | null = null;
  private _mipADispK: ComputeUnit | null = null;
  private _mipADerivK: ComputeUnit | null = null;
  private _mipBDispK: ComputeUnit | null = null;
  private _mipBDerivK: ComputeUnit | null = null;

  constructor(context: Context, options: OceanFFTWaterOptions) {
    const [lon, lat] = options.center;
    const segments = Math.floor(options.segments ?? 256);
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
      throw new Error(`OceanFFTWater: center 非法 [${lon}, ${lat}]`);
    }
    if (!Number.isFinite(segments) || segments < 2) {
      throw new Error(`OceanFFTWater: segments 必须 >= 2，收到 ${segments}`);
    }

    this._context   = context;
    this._extentKm  = options.extentKm ?? 2.5;
    this._segments  = segments;
    this._waterLevel = options.waterLevel ?? 0;
    this._choppiness = options.choppiness ?? 0.9;
    this._timeScale  = options.timeScale ?? 1;
    this._sunDir = OceanFFTWater._normalize(options.sunDirection ?? [0.4, 0.5, 0.77]);
    this._refraction = options.refraction ?? false;
    this._foamPattern = options.foamPattern ?? null;
    this._wake = options.wake ?? null;
    this._renderParamsF32 = new Float32Array(OCEAN_RENDER_PARAMS_UBO_BYTES / 4);

    this._local = {
      scale: 1, windSpeed: options.windSpeed ?? 8, windDirection: options.windDirection ?? 20,
      fetch: 120, spreadBlend: 0.85, swell: 0.05, peakEnhancement: 3.3, shortWavesFade: 0.01,
    };
    this._swell = {
      scale: 0.48, windSpeed: 6, windDirection: 5,
      fetch: 1200, spreadBlend: 1.0, swell: 0.9, peakEnhancement: 3.3, shortWavesFade: 0.1,
    };

    this._paramsBytes = new ArrayBuffer(OCEAN_PARAMS_UBO_BYTES);
    this._paramsF32 = new Float32Array(this._paramsBytes);
    this._paramsU32 = new Uint32Array(this._paramsBytes);
    this._paramsU32[P_SEED_U32] = SEED;
    this._paramsF32[P_DEPTH] = DEPTH_METERS;
    this._paramsF32[P_CHOPPINESS] = this._choppiness;
    // 泡沫参数（tidewater 默认：bias 0.58 / gain 3.0 / decay 0.35 / add 2.5）
    this._paramsF32[P_FOAM_BIAS] = 0.58;
    this._paramsF32[P_FOAM_GAIN] = 3.0;
    this._paramsF32[P_FOAM_DECAY] = 0.35;
    this._paramsF32[P_FOAM_ADD] = 2.5;

    // ENU 基（每米归一化 ECEF 增量）写入 uniform 静态段
    const enu = OceanFFTWater._enuBasis(lon, lat, this._waterLevel);
    this._centerEcef = OceanFFTWater._ecef(lon, lat, this._waterLevel);
    this._eastDir = enu.east;
    this._northDir = enu.north;
    this._paramsF32.set(enu.east, P_EAST);
    this._paramsF32.set(enu.north, P_NORTH);
    this._paramsF32.set(enu.up, P_UP);

    this._updateSpectrumUniforms();
    this._geometry = OceanFFTWater._buildGeometry(
      lon, lat, this._extentKm, this._segments, this._waterLevel
    );
  }

  // ── 公开 API ─────────────────────────────────────────────────────────────

  /** 风速（m/s）。触发谱重建（下一帧 execute 重跑 spectrum 3 kernel）。 */
  setWindSpeed(mps: number): void {
    if (!Number.isFinite(mps) || mps <= 0) {
      throw new Error(`OceanFFTWater.setWindSpeed: 风速非法 ${mps}`);
    }
    this._local.windSpeed = mps;
    this._updateSpectrumUniforms();
  }

  /** 风向（度）。触发谱重建。 */
  setWindDirection(deg: number): void {
    if (!Number.isFinite(deg)) {
      throw new Error(`OceanFFTWater.setWindDirection: 风向非法 ${deg}`);
    }
    this._local.windDirection = deg;
    this._updateSpectrumUniforms();
  }

  /** λ 水平位移缩放（只更新 uniform，下一帧生效）。 */
  setChoppiness(v: number): void {
    if (!Number.isFinite(v)) {
      throw new Error(`OceanFFTWater.setChoppiness: 非法 ${v}`);
    }
    this._choppiness = v;
    this._paramsF32[P_CHOPPINESS] = v;
  }

  /** 效果时间倍率。 */
  setTimeScale(v: number): void {
    if (!Number.isFinite(v)) {
      throw new Error(`OceanFFTWater.setTimeScale: 非法 ${v}`);
    }
    this._timeScale = v;
  }

  /** 泡沫生成阈值（Jacobian，越小泡沫越少）。 */
  setFoamBias(v: number): void {
    if (!Number.isFinite(v)) {
      throw new Error(`OceanFFTWater.setFoamBias: 非法 ${v}`);
    }
    this._paramsF32[P_FOAM_BIAS] = v;
  }

  /** @inheritdoc IPass - 由 Scene 在 execute 前注入相机（IPass 可选协议）。 */
  updateCamera(camera: Camera): void {
    this._camera = camera;
  }

  // ── B1 集成 API（A1-A4 五件套接线）────────────────────────────────────

  /** 绑定/解绑泡沫图案（A3）。null 回退原 hash 噪声调制路径。 */
  bindFoamPattern(pattern: FoamPatternTexture | null): void {
    this._foamPattern = pattern;
  }

  /** 绑定/解绑船迹波场（A4）。null 时 VS 不叠加波高、FS 不叠加泡沫。 */
  bindWake(wake: WakeField | null): void {
    this._wake = wake;
  }

  /**
   * 绑定/解绑海床高度场（L4-A 岸线泡沫）。null 时 flag=0 走原路径（零回归）。
   *
   * uv 仿射在绑定时刻算好：seabed 局部平面与海面 planeXZ 同为 ENU 米域但
   * 原点不同，须做原点差换算——seabedLocal = oceanLocal + (oceanCenter -
   * seabedCenter) 在共享 ENU 平面基（海面 east/north，每米归一化 ECEF 增量）
   * 下的投影，uv = seabedLocal / sizeMeters + 0.5（与 seabed 纹理行主序
   * j=z / i=x 排布一致）。中心重合时 offset 恰为 (0.5, 0.5)。
   */
  bindSeabed(field: SeabedHeightField | null): void {
    this._seabed = field;
    if (!field) return;
    const c = field.centerRadial;
    const dx = this._centerEcef[0] - c.x;
    const dy = this._centerEcef[1] - c.y;
    const dz = this._centerEcef[2] - c.z;
    // 原点差在共享 ENU 平面基下的投影：center 均为归一化 ECEF（模长 ~1），
    // 基向量为「每米归一化增量」→ 需最小二乘投影 M = dot(diff, basis) /
    // dot(basis, basis) 还原米数（直接点积会差 R 倍，错位测试抓到）。
    const e = this._eastDir;
    const n = this._northDir;
    const eastM = (dx * e[0] + dy * e[1] + dz * e[2]) / (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    const northM = (dx * n[0] + dy * n[1] + dz * n[2]) / (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    const s = 1 / field.sizeMeters;
    this._seabedUvOffset = [eastM * s + 0.5, northM * s + 0.5];
    this._seabedUvScale = [s, s];
  }

  /** 岸线泡沫强度（UBO shoreParams.y；0 = 等效关闭）。 */
  setShoreStrength(v: number): void {
    if (!Number.isFinite(v) || v < 0) {
      throw new Error(`OceanFFTWater.setShoreStrength: 非法 ${v}`);
    }
    this._shoreStrength = v;
  }

  /** 开关折射混色（仍依赖 bindSceneHistory 提供的场景颜色历史）。 */
  setRefraction(v: boolean): void {
    this._refraction = v;
  }

  /**
   * 注入场景颜色历史来源（B1：`() => scene.sceneColorHistory`）。
   * getter 惰性求值——postProcess 未开启时返回 null，折射自动降级为不混色。
   */
  bindSceneHistory(source: (() => GPUTextureView | null) | null): void {
    this._sceneHistorySource = source;
  }

  // ── L5 集成 API（物理大气反射 / 太阳染色 / SSS / 破碎波）──────────────

  /**
   * 注入物理大气源（L5：`() => ({ skyView, transmittance, atmosphereParams,
   * skyParams })`，四 getter 均取自同一 AtmosphereSky 实例）。每帧惰性求值，
   * 任一字段未就绪（prepare 前 / destroy 后）自动换 fallback 资源且
   * featureFlags bit0=0——走旧平色路径（零回归）。
   */
  bindAtmosphere(source: (() => OceanAtmosphereBinding | null) | null): void {
    this._atmoSource = source;
  }

  /**
   * 注入海面云影源（Track C T6：`() => clouds.cloudShadow`）。getter 每帧
   * 惰性求值——null（未绑定 / CloudLayer prepare 前 / destroy 后）时绑定
   * 1×1 黑 rgba8unorm 占位 + 全零 96B UBO（intensity 0 短路恒 1，零回归）。
   * 海面侧强度固定 {@link CLOUD_SHADOW_SEA_INTENSITY}（CloudShadowInfo 契约
   * 不含强度；太阳直射项 = 天光反射 + 高光 glitter 同乘云影因子，水下恒 1）。
   */
  bindCloudShadow(source: (() => CloudShadowInfo | null) | null): void {
    this._cloudShadowSource = source;
  }

  /** 天空反射开关（featureFlags bit0；须先 bindAtmosphere 才生效）。 */
  setSkyReflection(on: boolean): void {
    this._skyReflectOn = on;
  }

  /** 太阳高光染色开关（featureFlags bit3；须先 bindAtmosphere 才生效）。 */
  setSunTint(on: boolean): void {
    this._sunTintOn = on;
  }

  /** 波峰 SSS 透光总强度（RP[22]；0 = 关 / featureFlags bit1）。 */
  setSubsurfaceStrength(v: number): void {
    if (!Number.isFinite(v) || v < 0) {
      throw new Error(`OceanFFTWater.setSubsurfaceStrength: 非法 ${v}`);
    }
    this._sssStrength = v;
  }

  /**
   * 深度感知破碎波（T4：VS 波形变形 + FS 花边/内卷水幕 + featureFlags bit2）。
   * null = 关闭（breakParams 全零 → gain.x=0 短路，零回归）。
   */
  setBreakWave(cfg: OceanBreakWaveConfig | null): void {
    if (cfg) this._validateBreak(cfg);
    this._breakCfg = cfg;
  }

  /** 更新已启用破碎波的参数（不改开关态；null 字段取默认值）。 */
  setBreakParams(cfg: OceanBreakWaveConfig): void {
    this._validateBreak(cfg);
    this._breakCfg = cfg;
  }

  /** bandFarM 必须 > bandNearM（反序 smoothstep 边界重叠会产生 NaN）。 */
  private _validateBreak(cfg: OceanBreakWaveConfig): void {
    const near = cfg.bandNearM ?? 3;
    const far = cfg.bandFarM ?? 8;
    if (!Number.isFinite(near) || !Number.isFinite(far) || far <= near) {
      throw new Error(
        `OceanFFTWater: bandFarM(${far}) 必须 > bandNearM(${near}) 且均为有限数`,
      );
    }
  }

  /**
   * T4 破碎喷雾吸附源（BreakSpray options.seabed 用）：seabed 视图 + 海面
   * ENU 基 + uv 仿射，compute 端 textureLoad 做高度吸附。场未绑定/已 dispose
   * 时返回 null（BreakSpray 换占位资源，粒子不做水深门）。
   */
  get breakSpraySeabed(): OceanSeabedSprayBinding | null {
    if (!this._seabed || !this._seabed.seabedTexture) return null;
    return {
      view: this._seabed.seabedTexture,
      dispView: this._dispArrayView!,
      centerEcef: { x: this._centerEcef[0], y: this._centerEcef[1], z: this._centerEcef[2] },
      eastDir: { x: this._eastDir[0], y: this._eastDir[1], z: this._eastDir[2] },
      northDir: { x: this._northDir[0], y: this._northDir[1], z: this._northDir[2] },
      uvOffset: { x: this._seabedUvOffset[0], y: this._seabedUvOffset[1] },
      uvScale: { x: this._seabedUvScale[0], y: this._seabedUvScale[1] },
    };
  }

  // ── L3 集成 API（海面阴影接收 / 水下双面视角）──────────────────────────

  /**
   * 注入 CSSM 阴影源（L3：`() => scene.shadow.receiverBinding()`）。
   * getter 每帧惰性求值——阴影关闭 / 未就绪 / 失败时返回 null，海面绑定
   * 占位资源（全零 UBO），shader 侧 params.x=0 短路，阴影因子恒 1（零回归）。
   * 级联矩阵在共享 UBO 内由 ShadowMap.execute 每帧更新，相机移动自动跟随。
   */
  setShadowSource(source: (() => ShadowReceiverBinding | null) | null): void {
    this._shadowSource = source;
  }

  /** 水下视角（从下看水面）：法线翻转 + 透射青绿 + 高光弱化 + 菲涅尔反转。 */
  setUnderwaterView(on: boolean): void {
    this._underwaterView = on;
  }

  /** 当前是否处于水下视角（验证探针用）。 */
  get underwaterView(): boolean {
    return this._underwaterView;
  }

  /** 导数纹理视图（rgba16float 2d-array 全 mip）——{@link FFTDataSource}。 */
  get derivativeTexture(): GPUTextureView | null {
    return this._derivTextureView;
  }

  /** 每级联平铺尺寸（米）——{@link FFTDataSource}。 */
  get cascadeSizes(): readonly number[] {
    return CASCADE_SIZES;
  }

  /** FFT 边长——{@link FFTDataSource}。 */
  get fftSize(): number {
    return N;
  }

  /** @inheritdoc TargetFormatAware - 场景目标格式切换（T5 真 HDR）：
   * 置空渲染管线/BGL/bindGroup，prepare() 与 execute() 的惰性守卫段下一帧
   * 全部按新格式重建（同格式幂等零开销；bindGroup 缓存键检查含 null 分支）。 */
  setTargetFormat(format: GPUTextureFormat): void {
    if (this._failed || format === (this._targetFormat ?? this._context.preferredFormat)) return;
    this._targetFormat = format;
    this._renderPipeline = null;
    this._renderBindGroupLayout = null;
    this._renderBindGroup = null;
  }

  /** @inheritdoc IPass - 幂等初始化（渲染管线 + compute 管线族）。 */
  prepare(): void {
    if (this._failed) return;
    try {
      this._prepareCompute();
      this._prepareRender();
    } catch (e) {
      // execute 每帧重入 prepare：初始化抛错后若不设闸会每帧重试刷屏，
      // 且 GPU 资源已部分创建，重试必然同样失败——一次报错并禁用本效果。
      this._failed = true;
      console.error(`[${this.label}] 初始化失败，效果已禁用:`, e);
    }
  }

  /**
   * 每帧执行：一个 compute pass（谱重建[按需] + 行/列 IFFT + mip 链 ×2）
   * + 一个 render pass（网格叠加到 target，loadOp load）。
   */
  execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void {
    this.prepare();
    if (this._failed) return;

    const dt = this._prevTime < 0 ? 0 : Math.min(0.1, Math.max(0, time - this._prevTime));
    this._prevTime = time;
    this._timeAccum += dt * this._timeScale;
    this._paramsF32[P_TIME] = this._timeAccum;
    this._paramsF32[P_DT] = dt;
    this._context.writeBuffer(this._oceanUBO!, 0, this._paramsBytes);

    // ── compute pass ─────────────────────────────────────────────────────
    const cpass = encoder.beginComputePass({ label: `${this.label}.FFT` });

    if (this._needsSpectrum && this._initK && this._conjK && this._copyK) {
      this._needsSpectrum = false;
      for (const k of [this._initK, this._conjK, this._copyK]) {
        cpass.setPipeline(k.pipeline);
        cpass.setBindGroup(0, k.bindGroup);
        cpass.dispatchWorkgroups(N_OVER_16, N_OVER_16, CASCADES);
      }
    }

    if (this._rowK && this._colK && this._mipADispK && this._mipADerivK && this._mipBDispK && this._mipBDerivK) {
      cpass.setPipeline(this._rowK.pipeline);
      cpass.setBindGroup(0, this._rowK.bindGroup);
      cpass.dispatchWorkgroups(N, CASCADES, 1);

      cpass.setPipeline(this._colK.pipeline);
      cpass.setBindGroup(0, this._colK.bindGroup);
      cpass.dispatchWorkgroups(N, CASCADES, 1);

      for (const k of [this._mipADispK, this._mipADerivK]) {
        cpass.setPipeline(k.pipeline);
        cpass.setBindGroup(0, k.bindGroup);
        cpass.dispatchWorkgroups(N_OVER_32, N_OVER_32, CASCADES);
      }
      for (const k of [this._mipBDispK, this._mipBDerivK]) {
        cpass.setPipeline(k.pipeline);
        cpass.setBindGroup(0, k.bindGroup);
        cpass.dispatchWorkgroups(1, 1, CASCADES);
      }
    }
    cpass.end();

    // ── render pass ──────────────────────────────────────────────────────
    const cam = this._camera;
    if (!this._renderPipeline || !this._renderBindGroupLayout || !cam || this._geometry.indexCount === 0) return;

    // ── 渲染参数动态段：flags + 船迹窗口 uv 映射（每帧写）────────────────
    const RP = this._renderParamsF32;
    RP[12] = this._foamPattern && this._foamPattern.view ? 1 : 0;
    RP[13] = this._wake && this._wake.fieldView ? 1 : 0;
    RP[14] = this._refraction ? 1 : 0;
    RP[15] = 0.045; // 折射扰动强度（NDC 域，∝法线水平分量）
    RP[20] = this._underwaterView ? 1 : 0; // L3 水下视角
    // L5：RP[21] 时间秒（反射星光闪烁）/ RP[22] SSS 总强度 / RP[23] featureFlags
    // 位掩码（bit0 天空反射 / bit1 SSS / bit2 破碎波 / bit3 太阳高光染色）。
    // 大气源每帧惰性求值：sky.execute 先于 ocean.execute（effects 顺序），
    // LUT/UBO 同帧已就绪；字段缺失 → fallback + bit0=0（零回归）。
    const atmoSrc = this._atmoSource?.() ?? null;
    const atmoReady = !!(atmoSrc && atmoSrc.skyView && atmoSrc.transmittance
      && atmoSrc.atmosphereParams && atmoSrc.skyParams);
    RP[21] = this._timeAccum;
    RP[22] = this._sssStrength;
    let ff = 0;
    if (atmoReady && this._skyReflectOn) ff |= 1;
    if (this._sssStrength > 0) ff |= 2;
    if (this._breakCfg) ff |= 4;
    if (atmoReady && this._sunTintOn) ff |= 8;
    RP[23] = ff;
    // L4-A 岸线泡沫：RP[24] flag（下一空闲位，flags vec4 已满：x/y/z 开关 + w 强度）
    // + RP[25] 强度 + RP[28-31] seabedUv 仿射（海面局部米 → seabed 纹理 [0,1]）。
    const sbActive = !!(this._seabed && this._seabed.seabedTexture);
    RP[24] = sbActive ? 1 : 0;
    RP[25] = this._shoreStrength;
    if (sbActive) {
      RP[28] = this._seabedUvOffset[0];
      RP[29] = this._seabedUvOffset[1];
      RP[30] = this._seabedUvScale[0];
      RP[31] = this._seabedUvScale[1];
    }
    if (this._wake && this._wake.fieldView) {
      // uv = planeXZ * scale + offset；offset = -windowOrigin / windowSizeM
      const winM = this._wake.gridN * this._wake.cellM;
      const org = this._wake.windowOrigin;
      RP[16] = -org.x / winM;
      RP[17] = -org.z / winM;
      RP[18] = 1 / winM;
      RP[19] = 1 / winM;
    }
    this._context.writeBuffer(this._renderParamsUBO!, 0, OceanFFTWater._asBytes(RP));

    // ── T4 破碎波 UBO：每帧整体写入（未启用全零 → gain.x=0 短路零回归）────
    const BK = this._breakParamsF32;
    if (this._breakCfg) {
      BK[0] = 1;
      BK[1] = this._breakCfg.ampScale ?? 1.25;
      BK[2] = this._breakCfg.lean ?? 0.55;
      BK[3] = this._breakCfg.laceGain ?? 0.85;
      BK[4] = this._breakCfg.bandNearM ?? 3;
      BK[5] = this._breakCfg.bandFarM ?? 8;
      BK[6] = this._breakCfg.curlShade ?? 0.55;
      BK[7] = this._breakCfg.laceThresholdM ?? 1.6;
    } else {
      BK.fill(0);
    }
    this._context.writeBuffer(this._breakParamsUBO!, 0, OceanFFTWater._asBytes(BK));

    // ── Track C 云影 UBO：每帧从 source 惰性读并写入（布局单一来源
    // CloudShader.ts CloudShadowParams 96B；未绑定全零 → intensity 0 短路）──
    const cloudSrc = this._cloudShadowSource?.() ?? null;
    const CS = this._cloudShadowF32;
    if (cloudSrc) {
      CS[0] = cloudSrc.centerEcef[0]; CS[1] = cloudSrc.centerEcef[1];
      CS[2] = cloudSrc.centerEcef[2]; CS[3] = cloudSrc.cloudRadiusNorm;
      CS[4] = cloudSrc.eastDir[0]; CS[5] = cloudSrc.eastDir[1]; CS[6] = cloudSrc.eastDir[2]; CS[7] = 0;
      CS[8] = cloudSrc.northDir[0]; CS[9] = cloudSrc.northDir[1]; CS[10] = cloudSrc.northDir[2]; CS[11] = 0;
      CS[12] = EARTH_RADIUS_M / Math.max(cloudSrc.extentM, 1); // extentScale
      CS[13] = CLOUD_SHADOW_SEA_INTENSITY; CS[14] = 0; CS[15] = 0;
      CS[16] = cloudSrc.sunDirection[0]; CS[17] = cloudSrc.sunDirection[1];
      CS[18] = cloudSrc.sunDirection[2]; CS[19] = 0;
      CS[20] = 0; CS[21] = 0; CS[22] = 0; CS[23] = 0; // windOffset 未消费
    } else {
      CS.fill(0);
    }
    this._context.writeBuffer(this._cloudShadowUBO!, 0, OceanFFTWater._asBytes(CS));

    // ── bind group 缓存：视图身份变化才重建（wake ping-pong / history 就绪）──
    const foamView = this._foamPattern?.view ?? this._fallbackView!;
    const wakeView = (this._wake && RP[13] > 0.5) ? this._wake.fieldView! : this._fallbackView!;
    const histView = this._refraction ? (this._sceneHistorySource?.() ?? this._fallbackView!) : this._fallbackView!;
    // L4-A：seabed 视图（场已 dispose → seabedTexture null → 占位 + flag=0）
    const seabedView = sbActive ? this._seabed!.seabedTexture! : this._seabedFallbackView!;
    // L3：每帧惰性求值阴影源（Scene.render 中 shadow.execute 已先于 effects
    // 更新共享 UBO → 级联矩阵随相机移动自动最新）；null 换占位资源。
    const shadowSrc = this._shadowSource?.() ?? null;
    const shUbo = shadowSrc?.ubo ?? this._shadowFallbackUBO!;
    const shView = shadowSrc?.textureView ?? this._shadowFallbackView!;
    const shSmp = shadowSrc?.comparisonSampler ?? this._shadowCmpSampler!;
    // L5：大气资源（未就绪字段 → fallback；身份参与 bind group 缓存键）
    const atmoSv = atmoReady ? atmoSrc!.skyView! : this._fallbackView!;
    const atmoTr = atmoReady ? atmoSrc!.transmittance! : this._fallbackView!;
    const atmoUbA = atmoReady ? atmoSrc!.atmosphereParams! : this._atmoFallbackUBO!;
    const atmoUbS = atmoReady ? atmoSrc!.skyParams! : this._atmoFallbackUBO!;
    // Track C：云影贴图视图（未激活 → 1×1 黑占位；UBO/采样器身份恒定不进键）
    const cloudView = cloudSrc?.view ?? this._cloudFallbackView!;
    if (!this._renderBindGroup || foamView !== this._bgFoam || wakeView !== this._bgWake
      || histView !== this._bgHist || shUbo !== this._bgShadow || seabedView !== this._bgSeabed
      || atmoSv !== this._bgAtmoSv || atmoTr !== this._bgAtmoTr
      || atmoUbA !== this._bgAtmoUboA || atmoUbS !== this._bgAtmoUboS
      || cloudView !== this._bgCloudTex) {
      this._renderBindGroup = this._context.device.createBindGroup({
        layout: this._renderBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this._cameraUBO! } },
          { binding: 1, resource: { buffer: this._oceanUBO! } },
          { binding: 2, resource: this._dispTex!.createView({ dimension: '2d-array' }) },
          { binding: 3, resource: this._derivTextureView! },
          { binding: 4, resource: this._sampler! },
          { binding: 5, resource: { buffer: this._renderParamsUBO! } },
          { binding: 6, resource: foamView },
          { binding: 7, resource: wakeView },
          { binding: 8, resource: histView },
          { binding: 9, resource: this._clampSampler! },
          { binding: 10, resource: { buffer: shUbo } },
          { binding: 11, resource: shView },
          { binding: 12, resource: shSmp },
          { binding: 13, resource: seabedView },
          // L5/T4：大气 LUT×2 + clamp + UBO×2 + 破碎波 UBO（fallback 见 _prepareRender）
          { binding: 14, resource: atmoSv },
          { binding: 15, resource: this._clampSampler! },
          { binding: 16, resource: atmoTr },
          { binding: 17, resource: { buffer: atmoUbA } },
          { binding: 18, resource: { buffer: atmoUbS } },
          { binding: 19, resource: { buffer: this._breakParamsUBO! } },
          // Track C：云影接收（fallback 见 _prepareRender；UBO 每帧 writeBuffer）
          { binding: 20, resource: cloudView },
          { binding: 21, resource: { buffer: this._cloudShadowUBO! } },
          { binding: 22, resource: this._clampSampler! },
        ],
      });
      this._bgFoam = foamView;
      this._bgWake = wakeView;
      this._bgHist = histView;
      this._bgShadow = shUbo;
      this._bgSeabed = seabedView;
      this._bgAtmoSv = atmoSv;
      this._bgAtmoTr = atmoTr;
      this._bgAtmoUboA = atmoUbA;
      this._bgAtmoUboS = atmoUbS;
      this._bgCloudTex = cloudView;
    }

    const camData = new Float32Array(CAMERA_UBO_SIZE / 4);
    writeCameraUniforms(camData, {
      viewMatrix:           cam.viewMatrix.values,
      projectionMatrix:     cam.projectionMatrix.values,
      viewProjectionMatrix: cam.viewProjectionMatrix.values,
      position:             cam.position,
      viewportWidth:        this._context.drawingBufferWidth,
      viewportHeight:       this._context.drawingBufferHeight,
      near:                 cam.frustum.near,
      far:                  cam.frustum.far,
    });
    this._context.writeBuffer(this._cameraUBO!, 0, camData);

    const pass = encoder.beginRenderPass({
      label: this.label,
      colorAttachments: [
        { view: target, loadOp: 'load', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
      ],
    });
    pass.setPipeline(this._renderPipeline);
    pass.setBindGroup(0, this._renderBindGroup);
    pass.setVertexBuffer(0, this._vertexBuffer!);
    pass.setIndexBuffer(this._indexBuffer!, 'uint32');
    pass.drawIndexed(this._geometry.indexCount);
    pass.end();
  }

  /** 释放全部 GPU 资源；调用后本 pass 不可再用。 */
  destroy(): void {
    for (const b of [this._vertexBuffer, this._indexBuffer, this._cameraUBO, this._oceanUBO,
      this._renderParamsUBO, this._h0, this._waveData, this._tmp, this._foam, this._mipSrc, this._mipMid]) {
      b?.destroy();
    }
    this._dispTex?.destroy();
    this._derivTex?.destroy();
    this._fallbackTex?.destroy();
    this._fallbackTex = null;
    this._fallbackView = null;
    this._seabedFallbackTex?.destroy();
    this._seabedFallbackTex = null;
    this._seabedFallbackView = null;
    this._atmoFallbackUBO?.destroy();
    this._atmoFallbackUBO = null;
    this._breakParamsUBO?.destroy();
    this._breakParamsUBO = null;
    this._derivTextureView = null;
    this._dispArrayView = null;
    this._renderBindGroupLayout = null;
    this._bgFoam = null;
    this._bgWake = null;
    this._bgHist = null;
    this._bgShadow = null;
    this._bgSeabed = null;
    this._bgAtmoSv = null;
    this._bgAtmoTr = null;
    this._bgAtmoUboA = null;
    this._bgAtmoUboS = null;
    this._shadowFallbackTex?.destroy();
    this._shadowFallbackUBO?.destroy();
    this._shadowFallbackTex = null;
    this._shadowFallbackView = null;
    this._shadowFallbackUBO = null;
    this._shadowCmpSampler = null;
    this._cloudFallbackTex?.destroy();
    this._cloudShadowUBO?.destroy();
    this._cloudFallbackTex = null;
    this._cloudFallbackView = null;
    this._cloudShadowUBO = null;
    this._bgCloudTex = null;
    this._clampSampler = null;
    this._vertexBuffer = null;
    this._indexBuffer = null;
    this._cameraUBO = null;
    this._oceanUBO = null;
    this._renderParamsUBO = null;
    this._h0 = null;
    this._waveData = null;
    this._tmp = null;
    this._foam = null;
    this._mipSrc = null;
    this._mipMid = null;
    this._dispTex = null;
    this._derivTex = null;
    this._dispViews = [];
    this._derivViews = [];
    this._renderPipeline = null;
    this._renderBindGroup = null;
    this._sampler = null;
    this._initK = null;
    this._conjK = null;
    this._copyK = null;
    this._rowK = null;
    this._colK = null;
    this._mipADispK = null;
    this._mipADerivK = null;
    this._mipBDispK = null;
    this._mipBDerivK = null;
  }

  // ── GPU 资源准备 ─────────────────────────────────────────────────────────

  /** compute 侧：6 storage buffer + 2 纹理 + 9 kernel（幂等）。 */
  private _prepareCompute(): void {
    if (this._rowK && this._colK && this._h0) return;
    const device = this._context.device;

    // ocean UBO 是 compute/render 两侧 bind group 的公共资源，必须在
    // mk() 之前就绪——原先放在 _prepareRender 创建，首帧 _prepareCompute
    // 绑定时拿到 null，createBindGroup 抛 TypeError（CDP 验证抓到）。
    this._oceanUBO ??= device.createBuffer({
      label: 'OceanFFTOceanUBO',
      size: OCEAN_PARAMS_UBO_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const total = N * N * CASCADES; // vec4f 元素数（foam 为 f32 元素数）
    const VEC4 = 16;
    const mkStorage = (label: string, bytes: number): GPUBuffer =>
      device.createBuffer({ label, size: bytes, usage: GPUBufferUsage.STORAGE });

    this._h0       ??= mkStorage('OceanFFTH0', total * VEC4);
    this._waveData ??= mkStorage('OceanFFTWaveData', total * VEC4);
    this._tmp      ??= mkStorage('OceanFFTTmp', total * 2 * VEC4);
    this._foam     ??= mkStorage('OceanFFTFoam', total * 4);
    this._mipSrc   ??= mkStorage('OceanFFTMipSrc', total * 2 * VEC4);
    this._mipMid   ??= mkStorage('OceanFFTMipMid', 64 * CASCADES * 2 * VEC4);

    this._dispTex ??= this._createArrayTexture('OceanFFTDisplacement');
    this._derivTex ??= this._createArrayTexture('OceanFFTDerivatives');
    this._dispViews = this._mipViews(this._dispTex);
    this._derivViews = this._mipViews(this._derivTex);

    // BGL entry helpers（全部 COMPUTE；uniform 须显式 minBindingSize，否则管线验证失败）
    const uOcean = (binding: number): GPUBindGroupLayoutEntry => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'uniform', minBindingSize: OCEAN_PARAMS_UBO_BYTES },
    });
    const rw = (binding: number): GPUBindGroupLayoutEntry => ({
      binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' },
    });
    const wt = (binding: number): GPUBindGroupLayoutEntry => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d-array' },
    });
    const buf = (binding: number, b: GPUBuffer): GPUBindGroupEntry => ({ binding, resource: { buffer: b } });
    const tex = (binding: number, v: GPUTextureView): GPUBindGroupEntry => ({ binding, resource: v });

    const mk = (label: string, code: string, entries: GPUBindGroupLayoutEntry[], resources: GPUBindGroupEntry[]): ComputeUnit => {
      const bgl = device.createBindGroupLayout({ label: `${label}.BGL`, entries });
      const pipeline = device.createComputePipeline({
        label,
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        compute: { module: this._context.createShaderModule(code, `${label}.Module`), entryPoint: 'main' },
      });
      return { pipeline, bindGroup: device.createBindGroup({ layout: bgl, entries: resources }) };
    };

    // 谱三连（一次性 / 风参数变更时重跑）
    this._initK ??= mk('OceanFFT.InitSpectrum', OCEAN_FFT_INIT_SPECTRUM_SHADER,
      [uOcean(0), rw(1), rw(2)],
      [buf(0, this._oceanUBO!), buf(1, this._h0), buf(2, this._waveData)]);
    this._conjK ??= mk('OceanFFT.Conjugate', OCEAN_FFT_CONJUGATE_SHADER,
      [uOcean(0), rw(1), rw(2)],
      [buf(0, this._oceanUBO!), buf(1, this._h0), buf(2, this._tmp!)]);
    this._copyK ??= mk('OceanFFT.CopyH0', OCEAN_FFT_COPY_H0_SHADER,
      [uOcean(0), rw(1), rw(2), rw(3)],
      [buf(0, this._oceanUBO!), buf(1, this._h0), buf(2, this._tmp!), buf(3, this._foam!)]);

    // 每帧主链
    this._rowK ??= mk('OceanFFT.Rows', OCEAN_FFT_ROW_SHADER,
      [uOcean(0), rw(1), rw(2), rw(3)],
      [buf(0, this._oceanUBO!), buf(1, this._h0), buf(2, this._waveData), buf(3, this._tmp!)]);
    this._colK ??= mk('OceanFFT.Columns', OCEAN_FFT_COLUMN_SHADER,
      [uOcean(0), rw(1), rw(2), rw(3), wt(4), wt(5)],
      [buf(0, this._oceanUBO!), buf(1, this._tmp!), buf(2, this._foam!), buf(3, this._mipSrc!),
        tex(4, this._dispViews[0]), tex(5, this._derivViews[0])]);

    // mip 链 ×2（位移 / 导数）
    this._mipADispK ??= mk('OceanFFT.MipADisp', OCEAN_FFT_MIP_A_DISP_SHADER,
      [uOcean(0), rw(1), rw(2), wt(3), wt(4), wt(5), wt(6), wt(7)],
      [buf(0, this._oceanUBO!), buf(1, this._mipSrc!), buf(2, this._mipMid!),
        tex(3, this._dispViews[1]), tex(4, this._dispViews[2]), tex(5, this._dispViews[3]),
        tex(6, this._dispViews[4]), tex(7, this._dispViews[5])]);
    this._mipADerivK ??= mk('OceanFFT.MipADeriv', OCEAN_FFT_MIP_A_DERIV_SHADER,
      [uOcean(0), rw(1), rw(2), wt(3), wt(4), wt(5), wt(6), wt(7)],
      [buf(0, this._oceanUBO!), buf(1, this._mipSrc!), buf(2, this._mipMid!),
        tex(3, this._derivViews[1]), tex(4, this._derivViews[2]), tex(5, this._derivViews[3]),
        tex(6, this._derivViews[4]), tex(7, this._derivViews[5])]);
    this._mipBDispK ??= mk('OceanFFT.MipBDisp', OCEAN_FFT_MIP_B_DISP_SHADER,
      [uOcean(0), rw(1), wt(2), wt(3), wt(4)],
      [buf(0, this._oceanUBO!), buf(1, this._mipMid!),
        tex(2, this._dispViews[6]), tex(3, this._dispViews[7]), tex(4, this._dispViews[8])]);
    this._mipBDerivK ??= mk('OceanFFT.MipBDeriv', OCEAN_FFT_MIP_B_DERIV_SHADER,
      [uOcean(0), rw(1), wt(2), wt(3), wt(4)],
      [buf(0, this._oceanUBO!), buf(1, this._mipMid!),
        tex(2, this._derivViews[6]), tex(3, this._derivViews[7]), tex(4, this._derivViews[8])]);
  }

  /** 渲染侧：管线 / 网格缓冲 / 相机与参数 UBO（幂等；bindGroup 每帧按视图身份缓存重建）。 */
  private _prepareRender(): void {
    if (this._renderPipeline && this._vertexBuffer && this._indexBuffer) return;
    const device = this._context.device;

    if (!this._renderPipeline) {
      this._oceanUBO ??= device.createBuffer({
        label: 'OceanFFTOceanUBO',
        size: OCEAN_PARAMS_UBO_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this._cameraUBO ??= device.createBuffer({
        label: 'OceanFFTCameraUBO',
        size: CAMERA_UBO_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this._renderParamsUBO ??= device.createBuffer({
        label: 'OceanFFTRenderParamsUBO',
        size: OCEAN_RENDER_PARAMS_UBO_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      // 静态渲染参数：sunDir + 深水色(#04121f) + 天空/雾色（flags/wakeUv 每帧写）
      const RP = this._renderParamsF32;
      RP[0] = this._sunDir[0]; RP[1] = this._sunDir[1]; RP[2] = this._sunDir[2]; RP[3] = 0;
      RP[4] = 0.0157; RP[5] = 0.0706; RP[6] = 0.1216; RP[7] = 1;
      RP[8] = 0.55; RP[9] = 0.72; RP[10] = 0.88; RP[11] = 1;
      this._context.writeBuffer(this._renderParamsUBO, 0, OceanFFTWater._asBytes(RP));

      this._sampler ??= device.createSampler({
        label: 'OceanFFTSampler',
        addressModeU: 'repeat',
        addressModeV: 'repeat',
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: 'linear',
      });

      this._clampSampler ??= device.createSampler({
        label: 'OceanFFTClampSampler',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: 'linear',
      });

      // 1×1 零纹理：foam/wake/history 任一未绑定时占位（BGL 恒全绑定，
      // 采样结果为零 → shader 侧按 flags 走原路径，零回归）
      if (!this._fallbackTex) {
        this._fallbackTex = device.createTexture({
          label: 'OceanFFTFallback1x1',
          size: [1, 1],
          format: 'rgba16float',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        device.queue.writeTexture(
          { texture: this._fallbackTex },
          new Float32Array([0, 0, 0, 0]),
          { bytesPerRow: 16, rowsPerImage: 1 },
          [1, 1, 1],
        );
        this._fallbackView = this._fallbackTex.createView();
      }

      // L4-A 岸线泡沫占位：1×1 r32float 全 -1000（极深 = 无泡沫）。BGL 恒全
      // 绑定；shoreParams.x = 0 时 shader 不采样，走原路径（零回归）。
      if (!this._seabedFallbackTex) {
        this._seabedFallbackTex = device.createTexture({
          label: 'OceanFFTSeabedFallback',
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
      }

      // L3 阴影占位资源：1×1 depth32float 2d-array + 全零 UBO + comparison 采样器。
      // UBO 不写即全零 → params.x=0 → cssmShadowFactor 短路返回 1（阴影零回归）。
      if (!this._shadowFallbackTex) {
        this._shadowFallbackTex = device.createTexture({
          label: 'OceanFFTShadowFallback',
          size: [1, 1, 1],
          format: 'depth32float',
          usage: GPUTextureUsage.TEXTURE_BINDING,
        });
        this._shadowFallbackView = this._shadowFallbackTex.createView({
          label: 'OceanFFTShadowFallbackView',
          dimension: '2d-array',
          arrayLayerCount: 1,
        });
      }
      this._shadowFallbackUBO ??= device.createBuffer({
        label: 'OceanFFTShadowFallbackUBO',
        size: SHADOW_RECEIVER_UBO_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this._shadowCmpSampler ??= this._context.createSampler({ compare: 'less' });

      // Track C 云影占位：1×1 黑 rgba8unorm（dens = 0）+ 全零 96B UBO
      //（intensity = 0 短路）→ cloudShadowFactor 恒 1，零回归。
      if (!this._cloudFallbackTex) {
        this._cloudFallbackTex = device.createTexture({
          label: 'OceanFFTCloudFallback',
          size: [1, 1],
          format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        device.queue.writeTexture(
          { texture: this._cloudFallbackTex },
          new Uint8Array([0, 0, 0, 0]),
          { bytesPerRow: 4, rowsPerImage: 1 },
          [1, 1, 1],
        );
        this._cloudFallbackView = this._cloudFallbackTex.createView();
      }
      this._cloudShadowUBO ??= device.createBuffer({
        label: 'OceanFFTCloudShadowUBO',
        size: CLOUD_SHADOW_PARAMS_UBO_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      // L5 大气 fallback UBO（64B 全零，binding 17/18 共用；LUT 复用上方 1×1 黑
      // rgba16float——不写即全零 → featureFlags 相应位短路，零回归）
      this._atmoFallbackUBO ??= device.createBuffer({
        label: 'OceanFFTAtmoFallbackUBO',
        size: ATMO_UBO_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      // T4 破碎波 UBO（64B）
      this._breakParamsUBO ??= device.createBuffer({
        label: 'OceanFFTBreakParamsUBO',
        size: OCEAN_BREAK_PARAMS_UBO_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      this._derivTextureView ??= this._derivTex!.createView({ dimension: '2d-array' });
      this._dispArrayView ??= this._dispTex!.createView({ dimension: '2d-array' });

      const module = this._context.createShaderModule(OCEAN_FFT_RENDER_SHADER, 'OceanFFTRenderShader');
      const bgl = device.createBindGroupLayout({
        label: 'OceanFFTRenderBGL',
        entries: [
          // visibility 必须按实际着色阶段给全：VS 采样位移纹理/读 ocean.sizes，
          // 缺 VERTEX 会导致管线验证失败并污染整帧（FloodWater binding1 前车之鉴）
          { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', minBindingSize: CAMERA_UBO_SIZE } },
          { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', minBindingSize: OCEAN_PARAMS_UBO_BYTES } },
          { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
          // T4：derivTex 级联0 梯度也供 VS 波峰前倾使用 → visibility 含 VERTEX
          { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
          { binding: 4, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
          // renderParams：VS 读 flags/wakeUv（船迹位移），FS 读全部 → 两阶段都给
          { binding: 5, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', minBindingSize: OCEAN_RENDER_PARAMS_UBO_BYTES } },
          { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },            // foam pattern
          { binding: 7, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }, // wake field（VS 采样高度）
          { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },            // scene color history
          { binding: 9, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } }, // clamp
          // L3 CSSM 阴影接收（FRAGMENT 专用）：级联 UBO + depth2d-array + comparison。
          // 未激活阴影时绑占位资源（BGL 恒全绑定；全零 UBO → shader 因子恒 1）。
          { binding: 10, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', minBindingSize: SHADOW_RECEIVER_UBO_SIZE } },
          { binding: 11, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d-array', multisampled: false } },
          { binding: 12, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
          // L4-A 岸线泡沫：seabed 高度场 r32float（不可过滤 → textureLoad 取最近
          // texel）；T4 起顶点阶段也 textureLoad（破碎波波形变形）→ 双阶段
          { binding: 13, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
          // L5 物大气反射/染色：skyView + transmittance LUT（FRAGMENT）+ 64B UBO×2。
          // 未接 AtmosphereSky 时绑 fallback（1×1 黑 LUT + 全零 UBO）→ 特性位短路。
          { binding: 14, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
          { binding: 15, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
          { binding: 16, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
          { binding: 17, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', minBindingSize: ATMO_UBO_BYTES } },
          { binding: 18, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', minBindingSize: ATMO_UBO_BYTES } },
          // T4 破碎波参数（VS 波形变形 + FS 花边/内卷水幕 → 双阶段）
          { binding: 19, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', minBindingSize: OCEAN_BREAK_PARAMS_UBO_BYTES } },
          // Track C 云影（FRAGMENT 专用）：影贴图 rgba8unorm + 96B
          // CloudShadowParams UBO + filtering 采样器。未激活时绑占位资源
          //（BGL 恒全绑定；全零 UBO → shader 因子恒 1，零回归）。
          { binding: 20, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
          { binding: 21, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', minBindingSize: CLOUD_SHADOW_PARAMS_UBO_BYTES } },
          { binding: 22, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        ],
      });

      this._renderBindGroupLayout = bgl;

      this._renderPipeline = device.createRenderPipeline({
        label: 'OceanFFTPipeline',
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        vertex: {
          module,
          entryPoint: 'vs_main',
          buffers: [{
            arrayStride: 20, // 5 floats：[ecefX, ecefY, ecefZ, planeX_m, planeZ_m]
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },  // ecef
              { shaderLocation: 1, offset: 12, format: 'float32x2' }, // planeXZ（米）
            ],
          }],
        },
        fragment: {
          module,
          entryPoint: 'fs_main',
          targets: [
            {
              // T5 真 HDR：跟随 setTargetFormat 下发的场景目标格式（未下发 = 旧行为）
              format: this._targetFormat ?? this._context.preferredFormat,
              blend: {
                color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              },
            },
          ],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        // 无深度附件：遮挡在片元内用射线-球求交处理（与 FloodWater 一致）
      });
    }

    if (!this._vertexBuffer) {
      const vertices = this._geometry.vertices;
      this._vertexBuffer = device.createBuffer({
        label: 'OceanFFTVertices',
        size: Math.max(4, vertices.byteLength),
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      if (vertices.byteLength > 0) {
        this._context.writeBuffer(this._vertexBuffer, 0, OceanFFTWater._asBytes(vertices));
      }
    }

    if (!this._indexBuffer) {
      const indices = this._geometry.indices;
      this._indexBuffer = device.createBuffer({
        label: 'OceanFFTIndices',
        size: Math.max(4, indices.byteLength),
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      if (indices.byteLength > 0) {
        this._context.writeBuffer(this._indexBuffer, 0, OceanFFTWater._asBytes(indices));
      }
    }
  }

  /** rgba16float 2d-array（N×N×4 层，9 级 mip；write-only storage + 采样 + copyDst）。 */
  private _createArrayTexture(label: string): GPUTexture {
    return this._context.device.createTexture({
      label,
      size: { width: N, height: N, depthOrArrayLayers: CASCADES },
      format: 'rgba16float',
      mipLevelCount: MIP_LEVELS,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
  }

  /** 单 mip 层 write-only storage 视图（column/mip kernel 用）。 */
  private _mipViews(texture: GPUTexture): GPUTextureView[] {
    const views: GPUTextureView[] = [];
    for (let l = 0; l < MIP_LEVELS; l++) {
      views.push(texture.createView({
        label: `${texture.label}.Mip${l}`,
        dimension: '2d-array',
        baseMipLevel: l,
        mipLevelCount: 1,
      }));
    }
    return views;
  }

  // ── 谱 uniform ───────────────────────────────────────────────────────────

  /**
   * 级联带通 + 双波浪系统 JONSWAP 参数 → OceanParams
   * （tidewater updateSpectrumUniforms 同款公式）。
   */
  private _updateSpectrumUniforms(): void {
    const P = this._paramsF32;

    for (let i = 0; i < CASCADES; i++) {
      const low = i === 0 ? 0.0001 : (TWO_PI / CASCADE_SIZES[i]) * 6;
      const high = i === CASCADES - 1 ? 9999 : (TWO_PI / CASCADE_SIZES[i + 1]) * 6;
      P[P_CUTS + i * 4 + 0] = low;
      P[P_CUTS + i * 4 + 1] = high;
      P[P_SIZES + i * 4 + 0] = CASCADE_SIZES[i];
    }

    const sys = [this._local, this._swell];
    for (let i = 0; i < 2; i++) {
      const s = sys[i];
      const fetchM = Math.max(1, s.fetch) * 1000;
      const U = Math.max(0.1, s.windSpeed);
      const alpha = 0.076 * Math.pow((GRAVITY * fetchM) / (U * U), -0.22);
      const peakOmega = 22 * Math.pow((U * fetchM) / (GRAVITY * GRAVITY), -0.33);
      P[P_SYSA + i * 4 + 0] = s.scale;
      P[P_SYSA + i * 4 + 1] = (s.windDirection * Math.PI) / 180;
      P[P_SYSA + i * 4 + 2] = s.spreadBlend;
      P[P_SYSA + i * 4 + 3] = s.swell;
      P[P_SYSB + i * 4 + 0] = alpha;
      P[P_SYSB + i * 4 + 1] = peakOmega;
      P[P_SYSB + i * 4 + 2] = s.peakEnhancement;
      P[P_SYSB + i * 4 + 3] = s.shortWavesFade;
    }

    this._needsSpectrum = true;
  }

  // ── 几何 / 坐标 ──────────────────────────────────────────────────────────

  /**
   * 经纬正方形网格 → 归一化 ECEF 顶点：每顶点 [ecefX,ecefY,ecefZ,planeX,planeZ]。
   * planeX=东向米 / planeZ=北向米（纹理采样坐标，单位与级联尺寸一致）；
   * 经纬增量用小范围线性换算（Δlon = 米 / (111320·cos lat)）。
   */
  private static _buildGeometry(
    centerLon: number,
    centerLat: number,
    extentKm: number,
    segments: number,
    waterLevel: number
  ): { vertices: Float32Array; indices: Uint32Array; indexCount: number } {
    const n = segments + 1;
    const extentM = extentKm * 1000;
    const metersPerDegLon = METERS_PER_DEG * Math.cos((centerLat * Math.PI) / 180);
    const vertices = new Float32Array(n * n * 5);
    const tmp: [number, number, number] = [0, 0, 0];

    for (let iy = 0; iy < n; iy++) {
      const planeZ = (iy / segments - 0.5) * extentM;
      const lat = centerLat + planeZ / METERS_PER_DEG;
      for (let ix = 0; ix < n; ix++) {
        const planeX = (ix / segments - 0.5) * extentM;
        const lon = centerLon + planeX / metersPerDegLon;
        ecefFromLonLatHeight(lon, lat, waterLevel, tmp);
        const v = (iy * n + ix) * 5;
        vertices[v + 0] = tmp[0];
        vertices[v + 1] = tmp[1];
        vertices[v + 2] = tmp[2];
        vertices[v + 3] = planeX;
        vertices[v + 4] = planeZ;
      }
    }

    const quads = segments * segments;
    const indices = new Uint32Array(quads * 6);
    let o = 0;
    for (let iy = 0; iy < segments; iy++) {
      for (let ix = 0; ix < segments; ix++) {
        const r0 = iy * n + ix;
        const r1 = (iy + 1) * n + ix;
        indices[o++] = r0;
        indices[o++] = r1;
        indices[o++] = r0 + 1;
        indices[o++] = r0 + 1;
        indices[o++] = r1;
        indices[o++] = r1 + 1;
      }
    }
    return { vertices, indices, indexCount: indices.length };
  }

  /**
   * 中心点 ENU 基，单位为「每米归一化 ECEF 增量」（数值差分自动含极轴
   * 压扁——不可用 atan2 地心纬度反推重建，项目坐标约定血泪坑）。
   */
  private static _enuBasis(
    lon: number,
    lat: number,
    h: number
  ): { east: [number, number, number]; north: [number, number, number]; up: [number, number, number] } {
    const p0 = OceanFFTWater._ecef(lon, lat, h);

    const dLon = 1e-4; // 度
    const pE = OceanFFTWater._ecef(lon + dLon, lat, h);
    const stepLonM = dLon * METERS_PER_DEG * Math.cos((lat * Math.PI) / 180);
    const east = OceanFFTWater._scale(OceanFFTWater._sub(pE, p0), 1 / stepLonM);

    const dLat = 1e-4; // 度
    const pN = OceanFFTWater._ecef(lon, lat + dLat, h);
    const north = OceanFFTWater._scale(OceanFFTWater._sub(pN, p0), 1 / (dLat * METERS_PER_DEG));

    const pU = OceanFFTWater._ecef(lon, lat, h + 1.0); // +1 米
    const up = OceanFFTWater._scale(OceanFFTWater._sub(pU, p0), 1);

    return { east, north, up };
  }

  private static _ecef(lon: number, lat: number, h: number): [number, number, number] {
    const out: [number, number, number] = [0, 0, 0];
    ecefFromLonLatHeight(lon, lat, h, out);
    return out;
  }

  private static _sub(a: [number, number, number], b: [number, number, number]): [number, number, number] {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  }

  private static _scale(v: [number, number, number], s: number): [number, number, number] {
    return [v[0] * s, v[1] * s, v[2] * s];
  }

  /** 归一化方向向量。 */
  private static _normalize(v: [number, number, number]): [number, number, number] {
    const m = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / m, v[1] / m, v[2] / m];
  }

  /** 以 Uint8Array 视图暴露 typed array 的可拷贝字节（TS 5.7+ BufferSource 兼容）。 */
  private static _asBytes(src: ArrayBufferView): Uint8Array<ArrayBuffer> {
    return new Uint8Array(src.buffer as ArrayBuffer, src.byteOffset, src.byteLength);
  }
}

export default OceanFFTWater;
