/**
 * SensorCam —— 合成传感器通道 v0（F3）。
 *
 * 以给定相机位姿渲染一帧，同一帧导出三通道：
 *  - rgb   ：主 canvas 同管线渲染（临时换相机 → scene.render() → 还原），
 *            `canvas.toDataURL` 导出（WebGPU canvas 2d drawImage 读回全黑的历史坑，
 *            同 canvas 原生 API 可用）；
 *  - depth ：GBufferPass depth32float RT（COPY_SRC）全幅 copyTextureToBuffer →
 *            按引擎投影矩阵约定线性化为**真实米**；
 *  - seg   ：PickPass id 纹理全幅 readback（`debugFullViewport` +
 *            `debugDumpPickTexture`）→ 24bit pickId → 可注入的 pickId→语义
 *            映射表 → 调色板查色 → RGBA Uint8Array。
 *
 * ## 深度线性化（引擎归一化坐标系）
 *
 * 引擎内部使用归一化 ECEF（地球赤道半径 = 1，米 ÷ EARTH_SCALE = 6378137）。
 * `Matrix4.perspective` 产出 WebGPU 风格 [0,1] 深度（非 reversed-Z）：
 *
 *   d = z_ndc = -A + B/z_v，A = f/(n-f)，B = f·n/(n-f)
 *   ⇒ z_v = f·n / (f - d·(f - n))          （z_v 为归一化视空间深度）
 *   ⇒ 米制深度 = z_v × EARTH_SCALE
 *
 * d = 0 → near，d = 1 → far（清屏深度自然落到 far，天空像素即 far 值；
 * 消费方可用 seg=0 掩掉天空）。线性化用的 near/far 取**渲染后**
 * `camera.frustum.near/far` 的实际值（引擎按高度动态收紧，见
 * `Camera._updateProjMatrix`），与当帧投影矩阵/相机 UBO 严格自洽。
 *
 * ## 已知限制（Phase 0）
 *  - rgb/depth/seg 分辨率 = 主 canvas drawing buffer（引擎 G-Buffer/拾取目标
 *    都按 canvas 尺寸分配），params.width/height 仅作记录与预期视场参考；
 *  - 分割粒度（F12 起）：建筑图元为实例级——CityBlockBuilder 每楼唯一实例
 *    id 经 pick 管线第二顶点缓冲渲染进 id 平面（bit23 标记，见
 *    `PICK_INSTANCE_FLAG`），`instanceSegIds` 承载逐像素实例段 id；其余
 *    图元仍为 Primitive 级语义（pickIdToSemantic 查表）；
 *  - near/far 受引擎动态 near/far 策略约束，params.nearM/farM 仅记录。
 */

import { Cartesian3 } from '../math/Cartesian3';
import { CesiumMath } from '../math/CesiumMath';
import { EARTH_SCALE } from '../media/GimbalMath';
import type { Scene } from '../scene/Scene';

/** 调色板版本（对齐 dataset-tools schema `palette-v0.1`，引用式版本化）。 */
export const PALETTE_MAP_VERSION = 'palette-v0.1';

/** 语义类别常量（与默认调色板一一对应）。 */
export const SEMANTIC = {
  /** 天空 / 无物体 */
  EMPTY: 0,
  BUILDING: 1,
  VEGETATION: 2,
  GROUND: 3,
  SKY: 4,
} as const;

/** 默认调色板：semanticId → RGB（A 固定 255）。 */
export const DEFAULT_PALETTE: Readonly<Record<number, readonly [number, number, number]>> = {
  0: [0, 0, 0],          // 天空/无物体
  1: [255, 0, 0],        // building 红
  2: [0, 200, 0],        // vegetation 绿
  3: [139, 90, 43],      // ground 褐
  4: [80, 140, 255],     // sky 蓝
};

/** 传感器光学参数。nearM/farM 为期望值（Phase 0 受引擎动态 near/far 约束，见模块注释）。 */
export interface SensorCamParams {
  /** 垂直 FOV（度）。 */
  fovYDeg: number;
  /** 期望像素宽（Phase 0 实际渲染分辨率 = 主 canvas）。 */
  width: number;
  /** 期望像素高。 */
  height: number;
  /** 期望近距（米）。 */
  nearM: number;
  /** 期望远距（米）。 */
  farM: number;
}

/** 单帧采集位姿：经纬度（度）+ 高度（米）+ 弧度姿态（与 Camera.setView 同约定）。 */
export interface SensorPose {
  lonDeg: number;
  latDeg: number;
  heightM: number;
  /** 0 = 北，正 = 向东（弧度）。 */
  headingRad: number;
  /** 0 = 地平线，负 = 向下，-π/2 = nadir（弧度）。 */
  pitchRad: number;
  /** 绕视线（弧度），Phase 0 恒 0。 */
  rollRad: number;
  /**
   * 米制 ECEF 直传（可选，缺省走 lon/lat 分支）。位姿源持有真值 ECEF 时
   * 应直传：经 fromDegrees(lon,lat) 的往返会把位姿源的 ENU 线性近似误差
   * 带进渲染相机（0.02-0.35m），掠墙像素被 1/sin(夹角) 放大成米级深度残差
   * （W-A 深度残差收尾，2026-10-18）。与 sim-kernel 帧 pos_ecef 同约定。
   */
  posEcefM?: [number, number, number];
}

/**
 * 采集位姿 → setView 米制 ECEF 目的地（真 WGS84 椭球约定）。
 *
 * 背景（febba8d 后的双约定世界）：`Camera.resolveDestination` 的
 * `{lon,lat,height}` 分支已改为引擎线性归一化压扁世界（Globe/海洋/地形
 * 系），而传感器数据链路——CityBlockBuilder 城市顶点、90 页 SSE
 * `pos_ecef` 伴飞盒、`SensorFrameMeta.camPosEcefM`（"与 fromDegrees 同
 * 约定"）、depth 线性化米制换算——全部是真 WGS84 椭球的归一化 ECEF。
 * 采集相机必须与被观测物体同约定：走压扁分支会让视锥整体偏离城市
 * ~11km（31°N 两约定地表模长差），seg/depth 全空（2026-09-26 回归）。
 *
 * 因此 SensorCam 一律显式传米制 ECEF——setView 的 x/y/z 分支
 * （`×1/EARTH_SCALE` 归一化）两个约定共用，未被 febba8d 改动。
 */
export function sensorCamEcefDestination(pose: SensorPose): Cartesian3 {
  // 米制 ECEF 直传优先（见 SensorPose.posEcefM 注释）：setView 的 x/y/z
  // 分支 ×1/EARTH_SCALE 归一化，两约定共用。
  if (pose.posEcefM) {
    return new Cartesian3(pose.posEcefM[0], pose.posEcefM[1], pose.posEcefM[2]);
  }
  return Cartesian3.fromDegrees(pose.lonDeg, pose.latDeg, pose.heightM);
}

export interface SensorFrameMeta {
  intrinsics: { fx: number; fy: number; cx: number; cy: number };
  /** 水平 FOV（度），由 fovY 与实际纵横比推出。 */
  hfovDeg: number;
  /** 采集时刻（仿真时钟，秒）。 */
  tCaptureSim: number;
  /** 相机 ECEF 位置（米，引擎归一化坐标 × EARTH_SCALE，与 fromDegrees 同约定）。 */
  camPosEcefM: [number, number, number];
  /** [w,x,y,z]，body(FLU: 前-左-上)→ENU 旋转（列 = [forward,left,up]）。 */
  camQuat: [number, number, number, number];
  /** 线性化实际使用的 near/far（米，渲染后引擎动态值）。 */
  nearM: number;
  farM: number;
  /** 参数期望值（记录用）。 */
  requestedNearM: number;
  requestedFarM: number;
  fovYDeg: number;
  width: number;
  height: number;
}

/** 一帧合成传感器数据（三通道同帧同视锥）。 */
export interface SensorFrame {
  /** 实际渲染宽（= 主 canvas drawing buffer 宽）。 */
  width: number;
  height: number;
  /** RGB（PNG dataURL，主 canvas 原生导出）。 */
  rgb: string;
  /** 深度（米，行主序、左上原点、length = w*h；天空 = far 值）。 */
  depth: Float32Array;
  /** 语义分割 RGBA（每像素 4 字节，查调色板着色）。 */
  seg: Uint8Array;
  /** 每像素语义 id（0..4，调试/训练侧可直接消费）。 */
  segIds: Uint32Array;
  /**
   * F12 每像素实例段 id（可选；缺省 = 场景无实例图元）。
   * 0 = 背景或非实例图元；bit23 置位值（{@link PICK_INSTANCE_FLAG} | 建筑序号）
   * = 实例像素。全分辨率，与 seg/depth 同尺寸同序——RLE 导出的数据源。
   */
  instanceSegIds?: Uint32Array;
  /**
   * 双目立体通道（可选，SensorCamOptions.stereo 开启才有；Wave1-B1）。
   * 右目 = 左目 pose 沿 body-left 反向平移 baselineM 的同姿态平行光轴渲染，
   * 语义对齐 dataset-tools schema vision.stereo（uri 编码在 SensorRecorder
   * 落 FrameRecord 时组装）。
   */
  stereo?: {
    baselineM: number;
    mode: 'rgb' | 'full';
    /** 右目 RGB（PNG dataURL，与 frame.rgb 同导出路径）。 */
    rightRgb: string;
    /** 右目深度（米，与 depth 同构；mode='full' 才渲染/读回）。 */
    rightDepth?: Float32Array;
    /** 右目 meta（camPosEcefM=平移后光心；camQuat 与左目相同——平行光轴）。 */
    rightMeta: SensorFrameMeta;
  };
  meta: SensorFrameMeta;
}

/** pickId → 语义 id 的可注入映射（0 = 天空/背景）。 */
export type PickIdToSemantic = (pickId: number) => number;

/**
 * F12 实例像素标记位：pick id 平面中 bit23 置位的值为建筑实例
 * （低 23 位 = CityBlockBuilder 建筑序号），与 primitive pickId 空间隔离。
 * 与 PickShader.INSTANCE_FLAG / bbox-from-seg INSTANCE_FLAG 三方对齐。
 */
export const PICK_INSTANCE_FLAG = 0x800000;

/** 双目立体选项（Wave1-B1）。 */
export interface StereoOptions {
  /** 左右目基线（米，>0）。右目 = 左目沿 body-left 反方向平移 baselineM。 */
  baselineM: number;
  /**
   * 'rgb'（缺省）：右目仅 RGB（立体匹配/视差训练默认档，跳过右目深度读回
   * 省带宽）；'full'：右目另含 depth 深度通道（与左目 vision.depth 同构）。
   */
  mode?: 'rgb' | 'full';
}

export interface SensorCamOptions extends SensorCamParams {
  /** pickId→语义映射，缺省全 0（全画面视作背景）。 */
  pickIdToSemantic?: PickIdToSemantic;
  /** 调色板（缺省 {@link DEFAULT_PALETTE}）。 */
  palette?: Readonly<Record<number, readonly [number, number, number]>>;
  /** 双目立体（可选，缺省不渲染右目——单目行为与历史版本逐位一致）。 */
  stereo?: StereoOptions;
}

/**
 * 深度线性化：归一化 [0,1] 深度 → 归一化视空间深度 → 米。
 * 纯函数（单测锚点）：d=0 → nearM，d=1 → farM，中间单调。
 */
export function linearizeDepthMeters(
  d: number,
  nearNorm: number,
  farNorm: number,
  earthScaleM: number = EARTH_SCALE,
): number {
  const n = nearNorm;
  const f = farNorm;
  // z_v = f·n / (f - d·(f - n))（见模块头注释推导）
  const denom = f - d * (f - n);
  return ((f * n) / denom) * earthScaleM;
}

/**
 * 由相机正交基（forward / left / up，ENU 单位向量）求 [w,x,y,z] 四元数
 * （body→ENU）。列 = [forward,left,up] 的右旋矩阵：forward×left = up。
 * Shepperd 方法，纯函数（单测锚点）。
 */
export function quatFromFLU(
  forward: { x: number; y: number; z: number },
  left: { x: number; y: number; z: number },
  up: { x: number; y: number; z: number },
): [number, number, number, number] {
  const m00 = forward.x, m10 = forward.y, m20 = forward.z;
  const m01 = left.x,    m11 = left.y,    m21 = left.z;
  const m02 = up.x,      m12 = up.y,      m22 = up.z;
  const tr = m00 + m11 + m22;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    return [s / 4, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s];
  }
  if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    return [(m21 - m12) / s, s / 4, (m01 + m10) / s, (m02 + m20) / s];
  }
  if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    return [(m02 - m20) / s, (m01 + m10) / s, s / 4, (m12 + m21) / s];
  }
  const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
  return [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, s / 4];
}

/**
 * 双目右目光心（米制 ECEF）：p_right = p_left − baselineM · R(q)·[0,1,0]。
 *
 * body-left = R(q)·[0,1,0]（cam_quat 列 = [forward,left,up]，见
 * {@link quatFromFLU}）；右目在机体坐标 -X 侧（body-left 反向平移），
 * 与 dataset-tools schema `vision.stereo` 约定逐字对齐——两目 cam_quat
 * 相同（同姿态平行光轴），仅光心平移。ECEF 米制直接平移（引擎归一化
 * 世界中 setView x/y/z 分支自行 ×1/EARTH_SCALE，本函数不掺缩放）。
 * 纯函数（单测锚点）：q=[1,0,0,0] 时 FLU 恰对齐 ENU，右目沿 -Y 平移。
 */
export function stereoRightEyeEcefM(
  pLeftEcefM: readonly [number, number, number],
  camQuat: readonly [number, number, number, number],
  baselineM: number,
): [number, number, number] {
  const [qw, qx, qy, qz] = camQuat;
  // R(q)·v 标准式：v' = v + w·t + cross(q.xyz, t)，t = 2·cross(q.xyz, v)
  const vx = 0, vy = 1, vz = 0;
  const tx = 2 * (qy * vz - qz * vy);          // cross(q.xyz,[0,1,0]) ×2
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  const leftX = vx + qw * tx + (qy * tz - qz * ty);
  const leftY = vy + qw * ty + (qz * tx - qx * tz);
  const leftZ = vz + qw * tz + (qx * ty - qy * tx);
  return [
    pLeftEcefM[0] - baselineM * leftX,
    pLeftEcefM[1] - baselineM * leftY,
    pLeftEcefM[2] - baselineM * leftZ,
  ];
}

/** pickId（24bit，RGBA 低 3 字节）→ 语义 id → 调色板查色。纯函数（单测锚点）。 */
export function decodeSegPixel(
  r: number, g: number, b: number,
  pickIdToSemantic: PickIdToSemantic,
  palette: Readonly<Record<number, readonly [number, number, number]>>,
): { semanticId: number; rgba: [number, number, number, number] } {
  const pickId = r | (g << 8) | (b << 16);
  const semanticId = pickId === 0 ? 0 : pickIdToSemantic(pickId);
  const c = palette[semanticId] ?? [0, 0, 0];
  return { semanticId, rgba: [c[0], c[1], c[2], 255] };
}

/** 读回缓冲行 stride 剥离（copyTextureToBuffer bytesPerRow 256 对齐补隙）。 */
function destrideBytes(
  src: Uint8Array, w: number, h: number, bytesPerRow: number, elemBytes: number,
): Uint8Array {
  const out = new Uint8Array(w * h * elemBytes);
  for (let y = 0; y < h; y++) {
    const row = src.subarray(y * bytesPerRow, y * bytesPerRow + w * elemBytes);
    out.set(row, y * w * elemBytes);
  }
  return out;
}

/** Float32Array → base64（分块 btoa，避免 String.fromCharCode 爆栈）。 */
export function f32ToBase64(arr: Float32Array): string {
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let bin = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + CHUNK)));
  }
  return btoa(bin);
}

/**
 * SensorCam —— 绑定 Scene 的合成传感器相机。
 * Phase 0 复用主 canvas 临时换相机方案（渲染→还原），不新增离屏管线。
 */
export class SensorCam {
  private readonly _scene: Scene;
  private readonly _params: SensorCamParams;
  private readonly _pickIdToSemantic: PickIdToSemantic;
  private readonly _palette: Readonly<Record<number, readonly [number, number, number]>>;
  private readonly _stereo: { baselineM: number; mode: 'rgb' | 'full' } | null;

  constructor(scene: Scene, options: SensorCamOptions) {
    this._scene = scene;
    this._params = {
      fovYDeg: options.fovYDeg,
      width: options.width,
      height: options.height,
      nearM: options.nearM,
      farM: options.farM,
    };
    this._pickIdToSemantic = options.pickIdToSemantic ?? (() => 0);
    this._palette = options.palette ?? DEFAULT_PALETTE;
    this._stereo = options.stereo && options.stereo.baselineM > 0
      ? { baselineM: options.stereo.baselineM, mode: options.stereo.mode ?? 'rgb' }
      : null;
  }

  get params(): Readonly<SensorCamParams> { return this._params; }

  /** 场景是否已初始化（G-Buffer 就绪，可 capture）。 */
  isReady(): boolean {
    return this._scene.getDepthTexture() !== null && this._scene.gbufferPass !== null;
  }

  /**
   * 以给定位姿渲染一帧并导出三通道（stereo 开启时额外渲染右目）。
   * 渲染前保存相机状态、渲染后原样还原（position/direction/up/right/frustum
   * 全量），主循环渲染不受污染。单目路径行为与历史版本逐位一致（重构自
   * 原 capture 主体，见 {@link renderEye}）。
   */
  async capture(pose: SensorPose, tCaptureSim?: number): Promise<SensorFrame> {
    const scene = this._scene;
    if (!this.isReady()) throw new Error('SensorCam.capture: scene 尚未 initialize（G-Buffer 不存在）');

    const cam = scene.camera;
    // ── 保存相机（左右目共用一次保存/还原）──
    const saved = {
      position: Cartesian3.clone(cam.position),
      direction: Cartesian3.clone(cam.direction),
      up: Cartesian3.clone(cam.up),
      right: Cartesian3.clone(cam.right),
      fov: cam.frustum.fov,
      aspect: cam.frustum.aspectRatio,
      near: cam.frustum.near,
      far: cam.frustum.far,
    };

    try {
      const tSim = tCaptureSim ?? performance.now() / 1000;

      // ── 左目：原单目全通道路径（depth + seg）──
      const left = await this.renderEye({
        pose,
        tCaptureSim: tSim,
        wantDepth: true,
        wantSeg: true,
      });

      // ── 右目（stereo 开启时）：平行光轴矩形对 ──
      // 光心 = 左目光心沿 body-left 反向平移 baselineM（ECEF 米制，见
      // stereoRightEyeEcefM）；orientation（heading/pitch/roll）不变 →
      // 两目 cam_quat 相同。右目 meta 用右目 render 时刻快照
      // （"渲染用什么记什么"；camQuat 与左目重算值应逐位一致）。
      const st = this._stereo;
      let stereo: SensorFrame['stereo'];
      if (st) {
        const rightPose: SensorPose = {
          ...pose,
          posEcefM: stereoRightEyeEcefM(left.meta.camPosEcefM, left.meta.camQuat, st.baselineM),
        };
        const right = await this.renderEye({
          pose: rightPose,
          tCaptureSim: tSim,
          // mode='rgb'：跳过右目深度读回省带宽；契约 stereo.right 无 seg
          wantDepth: st.mode === 'full',
          wantSeg: false,
          // 两目深度基准同帧：右目锁定左目 render 时刻 near/far（见
          // renderEye lockNearNorm 注释）
          lockNearNorm: left.nearNorm,
          lockFarNorm: left.farNorm,
        });
        stereo = {
          baselineM: st.baselineM,
          mode: st.mode,
          rightRgb: right.rgb,
          rightDepth: right.depth,
          rightMeta: right.meta,
        };
      }

      return {
        width: left.w,
        height: left.h,
        rgb: left.rgb,
        depth: left.depth!,
        seg: left.seg!,
        segIds: left.segIds!,
        instanceSegIds: left.instanceSegIds!,
        ...(stereo ? { stereo } : {}),
        meta: left.meta,
      };
    } finally {
      // ── 还原相机（无论成败；renderEye 自有资源由其自身 finally 清理）──
      cam.position.x = saved.position.x;
      cam.position.y = saved.position.y;
      cam.position.z = saved.position.z;
      cam.direction.x = saved.direction.x;
      cam.direction.y = saved.direction.y;
      cam.direction.z = saved.direction.z;
      cam.up.x = saved.up.x;
      cam.up.y = saved.up.y;
      cam.up.z = saved.up.z;
      cam.right.x = saved.right.x;
      cam.right.y = saved.right.y;
      cam.right.z = saved.right.z;
      // 视矩阵脏标记：位置/基向量已整体回写，标记后主循环按原值重建
      (cam as unknown as { _viewDirty: boolean })._viewDirty = true;
      cam.frustum.fov = saved.fov;
      cam.frustum.near = saved.near;
      cam.frustum.far = saved.far;
      cam.setAspectRatio(saved.aspect);
    }
  }

  /**
   * 单目渲染一轮（capture 的核心，左右目共用）。相机保存/还原由调用方
   * （capture 层）一次成对，本方法只负责：设位姿 → render → render 时刻
   * 快照（W10）→ 读回 rgb/depth/seg → meta。
   *
   * @param opts.wantDepth 建 G-Buffer 深度读回缓冲 + 一次性帧钩子并线性化
   *   （右目 mode='rgb' 关——省带宽）。
   * @param opts.wantSeg 建 pick pass 全幅编码并解码 seg（契约 stereo.right
   *   无 seg 通道，右目恒关）。
   * @param opts.lockNearNorm/lockFarNorm 渲染前锁定动态 near/far（stereo
   *   右目传左目 render 时刻快照值）。背景：引擎 near/far 由海拔动态收紧
   *   （Camera._updateProjMatrix：near=min(0.005, max(alt·0.1, 3e-7))、far 同
   *   式），投影矩阵 getter 按 `_projDirty || 位置快照不匹配` 按需重算并
   *   **覆盖** near/far——右目 setView 改了位置必然触发重算，b=0.12m 基线在
   *   120m 高度下两目 near 差 ~12mm（0.1% 深度基准漂移，立体匹配不可接受）。
   *   锁定做法：写回左目 near/far 后清 _projDirty 并同步 _projPos 位置快照
   *   （与还原路径的 _viewDirty 直写同私有字段惯例），render 的 getter 判定
   *   "无变化"跳过重算——右目复用左目投影矩阵（fov/aspect/near/far 全同 →
   *   矩阵逐位一致），两目深度基准严格同帧。
   */
  private async renderEye(opts: {
    pose: SensorPose;
    tCaptureSim: number;
    wantDepth: boolean;
    wantSeg: boolean;
    lockNearNorm?: number;
    lockFarNorm?: number;
  }): Promise<{
    w: number;
    h: number;
    rgb: string;
    depth?: Float32Array;
    seg?: Uint8Array;
    segIds?: Uint32Array;
    instanceSegIds?: Uint32Array;
    meta: SensorFrameMeta;
    nearNorm: number;
    farNorm: number;
  }> {
    const scene = this._scene;
    const cam = scene.camera;
    const context = scene.context;

    const pickPass = opts.wantSeg ? scene.createPickPass() : null;
    const depthTex = scene.getDepthTexture()!;
    const w = depthTex.width;
    const h = depthTex.height;
    const device = context.device;

    // 深度读回缓冲（bytesPerRow 256 对齐；depth32float 每像素 4B；wantDepth 才建）
    const depthBytesPerRow = Math.ceil((w * 4) / 256) * 256;
    const depthBuf = opts.wantDepth
      ? device.createBuffer({
          label: 'SensorCam.DepthReadback',
          size: depthBytesPerRow * h,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })
      : null;

    // 一次性帧钩子：把 G-Buffer 深度全幅拷进读回缓冲（同帧、同投影）
    const depthHook = depthBuf
      ? (encoder: GPUCommandEncoder): void => {
          encoder.copyTextureToBuffer(
            { texture: depthTex },
            { buffer: depthBuf, bytesPerRow: depthBytesPerRow },
            { width: w, height: h },
          );
        }
      : null;
    if (depthHook) scene.frameHooks.push(depthHook);

    // 全幅 id 纹理模式：本次 render 编码的 pick view 不做 1×1 scissor
    const savedDebugFull = pickPass ? pickPass.debugFullViewport : null;
    if (pickPass) pickPass.debugFullViewport = true;

    try {
      // ── 设传感器位姿（fov/aspect 变更需经 setAspectRatio 触发 proj 重算）──
      // destination 走米制 ECEF（真椭球，城市/SSE pos_ecef 同约定）——
      // 不走 setView 的 {lon,lat,height} 压扁世界分支（见 sensorCamEcefDestination）。
      cam.setView({
        destination: sensorCamEcefDestination(opts.pose),
        orientation: {
          heading: opts.pose.headingRad,
          pitch: opts.pose.pitchRad,
          roll: opts.pose.rollRad,
        },
      });
      cam.frustum.fov = CesiumMath.toRadians(this._params.fovYDeg);
      cam.setAspectRatio(w / h);
      if (opts.lockNearNorm != null && opts.lockFarNorm != null) {
        cam.frustum.near = opts.lockNearNorm;
        cam.frustum.far = opts.lockFarNorm;
        // 抑制 render 时 proj getter 的动态重算（见方法注释 lockNearNorm）
        (cam as unknown as { _projDirty: boolean })._projDirty = false;
        const projPos = (cam as unknown as { _projPos: Cartesian3 })._projPos;
        projPos.x = cam.position.x;
        projPos.y = cam.position.y;
        projPos.z = cam.position.z;
      }

      // 注册 pending pick：本次 render 的帧钩子会全幅编码 id 平面
      const pickPromise = opts.wantSeg ? scene.pick(0, 0) : null;
      // W4 TEMP DEBUG: render-time camera snapshot（诊断后移除；双目各记一条）
      const w4log = (window as unknown as { __W4_CAMLOG?: unknown[] }).__W4_CAMLOG
        ?? ((window as unknown as { __W4_CAMLOG?: unknown[] }).__W4_CAMLOG = []);
      if (Array.isArray(w4log)) {
        w4log.push({
          pos: [cam.position.x, cam.position.y, cam.position.z],
          dir: [cam.direction.x, cam.direction.y, cam.direction.z],
          up: [cam.up.x, cam.up.y, cam.up.z],
          near: cam.frustum.near, far: cam.frustum.far,
          viewDirty: (cam as unknown as { _viewDirty: boolean })._viewDirty,
          projDirty: (cam as unknown as { _projDirty: boolean })._projDirty,
        });
        if (w4log.length > 24) w4log.shift();
      }
      scene.render();

      // W10 渲染时刻相机快照：render() 提交后本帧画面/投影已定格，但下方
      // debugDumpPickTexture/mapAsync 两个 await 挂起期间，页面跟随 tick 的
      // setView（新 SSE 位姿 + W4 动态 near/far）会改写相机——meta（camQuat/
      // fx/near/far）与深度线性化必须用本快照，否则"帧 A 画面配帧 B 投影/
      // 姿态"（W9/F20-2 记录的 0.2-1m 时戳竞态残差源，根因即此）。
      const camSnap = {
        fov: cam.frustum.fov,
        near: cam.frustum.near,
        far: cam.frustum.far,
        dir: Cartesian3.clone(cam.direction),
        up: Cartesian3.clone(cam.up),
        pos: Cartesian3.clone(cam.position),
      };

      // 一次性帧钩子已完成使命：本次 render 的 submit 已包含深度拷贝，
      // 立即摘除（finally 里的移除兜底异常路径）。若留到 finally，主循环
      // RAF 的每一帧都会重复编码 copy → depthBuf，与下方 mapAsync 的
      // mapped 生命周期交错 → "used in submit while mapped" validation
      // error → 整帧 submit 作废 → 录制态渐进黑帧（深度采样归零）。
      if (depthHook) {
        const hookIdx = scene.frameHooks.indexOf(depthHook);
        if (hookIdx !== -1) scene.frameHooks.splice(hookIdx, 1);
      }

      // ── RGB：主 canvas 原生导出（同帧已呈现）──
      const rgb = context.canvas.toDataURL('image/png');

      // ── seg：全幅 id 纹理读回（wantSeg 才有）──
      const dump = pickPass ? await pickPass.debugDumpPickTexture() : null;
      void pickPromise; // pick 结果本身不消费（取整幅 id 纹理）

      // ── depth：map + 剥离行 stride + 线性化（wantDepth 才有）──
      let depthM: Float32Array | undefined;
      if (depthBuf) {
        await depthBuf.mapAsync(GPUMapMode.READ);
        const raw = new Uint8Array(depthBuf.getMappedRange());
        const tightlyPacked = destrideBytes(raw, w, h, depthBytesPerRow, 4);
        depthBuf.unmap();
        const f32 = new Float32Array(tightlyPacked.buffer);
        depthM = new Float32Array(w * h);
        for (let i = 0; i < f32.length; i++) {
          depthM[i] = linearizeDepthMeters(f32[i], camSnap.near, camSnap.far);
        }
      }

      // ── seg 解码：pickId → 语义 → 调色板（F12：同轮填充实例段平面）──
      let seg: Uint8Array | undefined;
      let segIds: Uint32Array | undefined;
      let instanceSegIds: Uint32Array | undefined;
      if (dump) {
        seg = new Uint8Array(w * h * 4);
        segIds = new Uint32Array(w * h);
        instanceSegIds = new Uint32Array(w * h);
        const sw = Math.min(dump.w, w);
        const sh = Math.min(dump.h, h);
        for (let y = 0; y < sh; y++) {
          for (let x = 0; x < sw; x++) {
            const si = (y * dump.w + x) * 4;
            const di = y * w + x;
            const rawId =
              dump.data[si] | (dump.data[si + 1] << 8) | (dump.data[si + 2] << 16);
            const dec = decodeSegPixel(
              dump.data[si], dump.data[si + 1], dump.data[si + 2],
              this._pickIdToSemantic, this._palette,
            );
            seg[di * 4]     = dec.rgba[0];
            seg[di * 4 + 1] = dec.rgba[1];
            seg[di * 4 + 2] = dec.rgba[2];
            seg[di * 4 + 3] = dec.rgba[3];
            segIds[di] = dec.semanticId;
            // F12：bit23 置位 = 建筑实例像素，原值保留（含序号）；其余归 0
            instanceSegIds[di] = (rawId & PICK_INSTANCE_FLAG) !== 0 ? rawId : 0;
          }
        }
      }

      // ── meta：内参 / 位姿真值（W10：全部取 render 时刻快照，见 camSnap 注释）──
      const fx = (h / 2) / Math.tan(camSnap.fov / 2);
      const hfovRad = 2 * Math.atan((w / 2) / fx);
      // FLU 基（forward/left/up）→ body→ENU 四元数
      const fwd = Cartesian3.normalize(camSnap.dir, new Cartesian3());
      const upW = Cartesian3.normalize(camSnap.up, new Cartesian3());
      const left = Cartesian3.normalize(Cartesian3.cross(upW, fwd, new Cartesian3()), new Cartesian3());
      const quat = quatFromFLU(fwd, left, upW);
      const pos = camSnap.pos;

      return {
        w,
        h,
        rgb,
        ...(depthM ? { depth: depthM } : {}),
        ...(seg && segIds && instanceSegIds
          ? { seg, segIds, instanceSegIds }
          : {}),
        meta: {
          intrinsics: { fx, fy: fx, cx: w / 2, cy: h / 2 },
          hfovDeg: CesiumMath.toDegrees(hfovRad),
          tCaptureSim: opts.tCaptureSim,
          camPosEcefM: [pos.x * EARTH_SCALE, pos.y * EARTH_SCALE, pos.z * EARTH_SCALE],
          camQuat: quat,
          nearM: camSnap.near * EARTH_SCALE,
          farM: camSnap.far * EARTH_SCALE,
          requestedNearM: this._params.nearM,
          requestedFarM: this._params.farM,
          fovYDeg: this._params.fovYDeg,
          width: this._params.width,
          height: this._params.height,
        },
        nearNorm: camSnap.near,
        farNorm: camSnap.far,
      };
    } finally {
      // ── 清理 renderEye 自有资源（无论成败）──
      if (depthHook) {
        const hookIdx = scene.frameHooks.indexOf(depthHook);
        if (hookIdx !== -1) scene.frameHooks.splice(hookIdx, 1);
      }
      if (pickPass && savedDebugFull !== null) pickPass.debugFullViewport = savedDebugFull;
      depthBuf?.destroy();
    }
  }
}

// ── 采样调度（不依赖 rAF：headless 后台标签 rAF 会停，setInterval 不会）──────

/** FrameRecord（字段名严格对齐 dataset-tools/schema/frame-record.ts）。 */
export interface FrameRecord {
  frame_id: string;
  scenario_id: string;
  episode_id: string;
  step_index: number;
  sim_time_s: number;
  seed: bigint;
  vision: {
    rgb: {
      uri: string; w: number; h: number;
      fx: number; fy: number; cx: number; cy: number;
      distortion: [number, number, number, number, number];
      hfov_deg: number;
    };
    depth: { uri: string; fmt: 'f32' | 'u16'; near_m: number; far_m: number; unit: 'm' };
    /**
     * seg 通道。旧格式：data = G×G 语义 id 方阵（无 width/height/encoding）。
     * F12 新格式：encoding 'rle-palette' + 全分辨率 RLE（data = [value,run,...]，
     * value 语义：0 背景 / 1..4 调色板 cls / 0x800000|n 建筑实例 n），
     * palette 查表在消费侧；grid32 为旧 32×32 兼容副本（离线重放）。
     */
    seg: {
      uri: string; palette_map_ref: string;
      width?: number; height?: number;
      encoding?: 'rle-palette';
      data?: number[];
      grid32?: number[];
    };
    bbox2d: Array<{ cls: string; track_id: number; x1: number; y1: number; x2: number; y2: number }>;
    /**
     * 双目立体（可选通道，Wave1-B1；字段名严格对齐 dataset-tools/schema
     * frame-record.ts L149-166）：右目 = 左目 pose 沿 body-left 反方向平移
     * baseline_m 的同姿态平行光轴；mode='rgb' 右目仅 RGB，'full' 另含
     * depth（与左目 vision.depth 同构）。未携带 stereo 的旧帧完全向后兼容。
     */
    stereo?: {
      baseline_m: number;
      mode: 'rgb' | 'full';
      right: {
        rgb: FrameRecord['vision']['rgb'];
        depth?: FrameRecord['vision']['depth'];
      };
    };
  };
  command: {
    mode: 'waypoint' | 'velocity' | 'attitude' | 'rc';
    target_enu: [number, number, number];
    setpoint: { vx: number; vy: number; vz: number; yaw_rate: number };
    actuator: [number, number, number, number];
  };
  action: {
    pos_ecef: [number, number, number];
    pos_enu: [number, number, number];
    quat: [number, number, number, number];
    vel_e: [number, number, number];
    ang_rate_e: [number, number, number];
    motor_rpm: [number, number, number, number];
    battery: { v: number; a: number; soc: number };
  };
  environment: {
    wind: { vec: [number, number, number]; gust_model: 'dryden' | 'const' | 'none'; params: Record<string, number> };
    temp_c: number;
    pressure_pa: number;
    gnss: { sv_visible: number; hdop: number; nlos_masked: number[]; noise_std_m: number };
  };
  align: {
    t_capture_ns: number;
    t_command_ns: number;
    latency_model_ms: { sensor: number; control: number; actuator: number };
    cam_T_body: number[];
    imu_T_body: number[];
    /**
     * F19：渲染相机真实姿态 [w,x,y,z]（FLU→ECEF 旋转：列 = [forward,left,up]
     * 均为 ECEF 归一化坐标，光轴 = forward）。capture 时从引擎相机实际基取值
     * （"渲染用什么记什么"），与云台模式无关恒写入。云台稳定（90 页
     * ?gimbal=stab）下相机与机体解耦，action.quat（机体）不再能推出相机朝向
     * ——verify-depth 等解析侧应优先消费本字段重建像素射线：
     * 像素 → 相机 FLU 系 dir=[1,-(px-cx)/fx,-(py-cy)/fy] → qRotate → ECEF 方向。
     * 注意是 ECEF 不是 ENU（引擎 cam.direction/up 原生 ECEF；曾误命名
     * cam_quat_enu 导致消费方双重变换，2026-09-23 勘误改名）。
     */
    cam_quat_ecef: [number, number, number, number];
  };
  events: Array<{ t: number; type: string; payload?: unknown }>;
}

/** buildFrameRecord 的输入（mock 指令/真值由位姿源提供，Phase 0 无闭环）。 */
export interface FrameRecordInput {
  frame: SensorFrame;
  scenarioId: string;
  episodeId: string;
  stepIndex: number;
  /** 仿真时钟（秒）。 */
  simTimeS: number;
  seed: bigint;
  /** RGB 附件 URI（Phase 0 = dataURL / inline base64，不落盘）。 */
  rgbUri: string;
  depthUri: string;
  segUri: string;
  /** 当前指令（mock：航线插值器产出）。 */
  commandTargetEnu: [number, number, number];
  commandSetpoint: { vx: number; vy: number; vz: number; yawRate: number };
  /** 真值速度 ENU（米/秒，来自位姿源）。 */
  velEnu: [number, number, number];
  /** 本机局部 ENU 位置（米，Phase 0 = 位姿源真值）。 */
  posEnu: [number, number, number];
  paletteMapRef?: string;
  /**
   * 双目通道（可选；SensorCam.stereo 开启时由 SensorRecorder._tick 组装）。
   * 右目附件 uri 编码固定走 default 路径（rgb=dataURL、depth=inline
   * f32-base64，与 defaultEncodeUris 同构）——自定义 encodeUris 仅作用于
   * 左目（Phase 0 简化；mode 暂只 'rgb'，rightDepthUri 现实不触发）。
   */
  stereo?: {
    baselineM: number;
    mode: 'rgb' | 'full';
    rightRgbUri: string;
    rightDepthUri?: string;
  };
}

/** 4×4 单位阵（cam_T_body：Phase 0 相机与机体重合）。 */
const IDENTITY4 = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
];

/**
 * 组装 FrameRecord（字段名严格对齐 dataset-tools/schema/frame-record.ts）。
 * 纯函数（单测锚点）：不触 GPU / DOM。
 */
export function buildFrameRecord(input: FrameRecordInput): FrameRecord {
  const frame = input.frame;
  const meta = frame.meta;
  const tNs = Math.round(input.simTimeS * 1e9);
  return {
    frame_id: `${input.scenarioId}/${input.episodeId}/${String(input.stepIndex).padStart(6, '0')}`,
    scenario_id: input.scenarioId,
    episode_id: input.episodeId,
    step_index: input.stepIndex,
    sim_time_s: input.simTimeS,
    seed: input.seed,

    vision: {
      rgb: {
        uri: input.rgbUri,
        w: frame.width,
        h: frame.height,
        fx: meta.intrinsics.fx,
        fy: meta.intrinsics.fy,
        cx: meta.intrinsics.cx,
        cy: meta.intrinsics.cy,
        distortion: [0, 0, 0, 0, 0],
        hfov_deg: meta.hfovDeg,
      },
      depth: {
        uri: input.depthUri,
        fmt: 'f32',
        near_m: meta.nearM,
        far_m: meta.farM,
        unit: 'm',
      },
      seg: { uri: input.segUri, palette_map_ref: input.paletteMapRef ?? PALETTE_MAP_VERSION },
      bbox2d: [],
      // 双目（可选通道）：右目 rgb 内参/近远取右目 render 时刻 meta（平行
      // 光轴下与左目同值，但"渲染用什么记什么"）；mode='rgb' 无 right.depth
      ...(input.stereo && input.frame.stereo ? {
        stereo: {
          baseline_m: input.stereo.baselineM,
          mode: input.stereo.mode,
          right: {
            rgb: {
              uri: input.stereo.rightRgbUri,
              w: frame.width,
              h: frame.height,
              fx: input.frame.stereo.rightMeta.intrinsics.fx,
              fy: input.frame.stereo.rightMeta.intrinsics.fy,
              cx: input.frame.stereo.rightMeta.intrinsics.cx,
              cy: input.frame.stereo.rightMeta.intrinsics.cy,
              distortion: [0, 0, 0, 0, 0],
              hfov_deg: input.frame.stereo.rightMeta.hfovDeg,
            },
            ...(input.stereo.mode === 'full' && input.stereo.rightDepthUri
              ? {
                  depth: {
                    uri: input.stereo.rightDepthUri,
                    fmt: 'f32' as const,
                    near_m: input.frame.stereo.rightMeta.nearM,
                    far_m: input.frame.stereo.rightMeta.farM,
                    unit: 'm' as const,
                  },
                }
              : {}),
          },
        },
      } : {}),
    },

    command: {
      mode: 'waypoint',
      target_enu: input.commandTargetEnu,
      setpoint: {
        vx: input.commandSetpoint.vx,
        vy: input.commandSetpoint.vy,
        vz: input.commandSetpoint.vz,
        yaw_rate: input.commandSetpoint.yawRate,
      },
      actuator: [0.5, 0.5, 0.5, 0.5],
    },

    action: {
      pos_ecef: meta.camPosEcefM,
      pos_enu: input.posEnu,
      quat: meta.camQuat,
      vel_e: input.velEnu,
      ang_rate_e: [0, 0, 0],
      motor_rpm: [0, 0, 0, 0],
      battery: { v: 0, a: 0, soc: 1 },
    },

    environment: {
      wind: { vec: [0, 0, 0], gust_model: 'none', params: {} },
      temp_c: 15,
      pressure_pa: 101325,
      gnss: { sv_visible: 0, hdop: 0, nlos_masked: [], noise_std_m: 0 },
    },

    align: {
      t_capture_ns: tNs,
      t_command_ns: tNs,
      latency_model_ms: { sensor: 0, control: 0, actuator: 0 },
      cam_T_body: IDENTITY4,
      imu_T_body: IDENTITY4,
      // F19：渲染相机姿态真值（FLU→ECEF，拷贝防外部改 meta 污染帧记录）
      cam_quat_ecef: [...meta.camQuat] as [number, number, number, number],
    },

    events: [],
  };
}

export interface EpisodeResult {
  frames: FrameRecord[];
  stats: { nFrames: number; droppedMs: number };
}

/** 位姿源单点输出：位姿 + mock 指令 + 真值。 */
export interface PoseMock {
  pose: SensorPose;
  targetEnu: [number, number, number];
  setpoint: { vx: number; vy: number; vz: number; yawRate: number };
  velEnu: [number, number, number];
  posEnu: [number, number, number];
}

export interface SensorRecorderOptions {
  /** 采样频率（Hz）。 */
  hz: number;
  scenarioId: string;
  episodeId: string;
  /** u64 总种子（JSON 序列化时转十进制字符串）。 */
  seed?: bigint;
  /** 位姿源：t（仿真秒，从 0 计）→ 位姿 + 指令/真值 mock。 */
  poseAt: (tSimS: number) => PoseMock;
  /** 附件编码（缺省 rgb=dataURL；depth/seg=inline base64，不写文件）。 */
  encodeUris?: (frame: SensorFrame, stepIndex: number) => { rgbUri: string; depthUri: string; segUri: string };
  /** 可注入定时器（fake timer 单测锚点）；缺省全局 setInterval/clearInterval。 */
  timers?: {
    set: (fn: () => void, ms: number) => unknown;
    clear: (handle: unknown) => void;
  };
  /** 每帧落盘回调（预览 UI 用；不影响录制主流程）。 */
  onFrame?: (record: FrameRecord, frame: SensorFrame) => void;
}

/** 默认附件编码：rgb 用 dataURL，depth/seg 内联 base64（不写文件）。 */
export function defaultEncodeUris(frame: SensorFrame, _stepIndex: number): {
  rgbUri: string; depthUri: string; segUri: string;
} {
  return {
    rgbUri: frame.rgb,
    depthUri: `inline:f32-base64;${f32ToBase64(frame.depth)}`,
    segUri: 'inline:rgba-base64',
  };
}

/**
 * SensorRecorder —— 按 hz 定时采样（setInterval 驱动，显式调 scene.render()，
 * 不依赖 rAF 回调；后台标签页也能持续出帧）。上一帧未完成时本次触发丢弃，
 * 等效超时部分计入 droppedMs。
 */
export class SensorRecorder {
  private readonly _cam: SensorCam;
  private readonly _opts: SensorRecorderOptions;
  private readonly _timers: NonNullable<SensorRecorderOptions['timers']>;

  private _handle: unknown = null;
  private _inFlight = false;
  private _tStartSim = 0;
  private _stepIndex = 0;
  private readonly _frames: FrameRecord[] = [];
  private _droppedMs = 0;
  private _recording = false;

  constructor(cam: SensorCam, opts: SensorRecorderOptions) {
    this._cam = cam;
    this._opts = opts;
    this._timers = opts.timers ?? {
      set: (fn, ms) => setInterval(fn, ms),
      clear: (h) => clearInterval(h as ReturnType<typeof setInterval>),
    };
  }

  get recording(): boolean { return this._recording; }

  /** capture 在飞（render→copy→mapAsync 异步窗口）——外部相机驱动
   * （90 页跟随 tick 等）应在此期间让路，防 setView 打进 capture 渲染
   * 窗口污染深度帧（SSE 位姿时戳竞态的 0.2-1m 残差源，W9/F20-2 记录）。 */
  get inFlight(): boolean { return this._inFlight; }

  /** 启动采样。间隔 = 1000/hz（setInterval，非 rAF）。 */
  start(): void {
    if (this._recording) return;
    this._recording = true;
    this._frames.length = 0;
    this._droppedMs = 0;
    this._stepIndex = 0;
    this._inFlight = false;
    this._tStartSim = performance.now() / 1000;
    const periodMs = 1000 / this._opts.hz;
    this._handle = this._timers.set(() => { void this._tick(); }, periodMs);
  }

  /** 停止采样并等待在帧落盘，返回 episode 结果。 */
  async stop(): Promise<EpisodeResult> {
    if (this._handle !== null) {
      this._timers.clear(this._handle);
      this._handle = null;
    }
    this._recording = false;
    while (this._inFlight) {
      await new Promise<void>((r) => setTimeout(r, 0));
    }
    return {
      frames: this._frames,
      stats: { nFrames: this._frames.length, droppedMs: this._droppedMs },
    };
  }

  status(): { recording: boolean; nFrames: number; droppedMs: number; hz: number } {
    return {
      recording: this._recording,
      nFrames: this._frames.length,
      droppedMs: this._droppedMs,
      hz: this._opts.hz,
    };
  }

  /** 单次采样触发（定时器回调本体；fake-timer 单测可直接调用）。 */
  async _tick(): Promise<void> {
    if (!this._recording || this._inFlight) return;
    const t0 = performance.now();
    this._inFlight = true;
    const tSim = t0 / 1000 - this._tStartSim;
    try {
      const mock = this._opts.poseAt(tSim);
      const step = this._stepIndex++;
      const frame = await this._cam.capture(mock.pose, tSim);
      const uris = this._opts.encodeUris
        ? this._opts.encodeUris(frame, step)
        : defaultEncodeUris(frame, step);
      this._frames.push(buildFrameRecord({
        frame,
        scenarioId: this._opts.scenarioId,
        episodeId: this._opts.episodeId,
        stepIndex: step,
        simTimeS: tSim,
        seed: this._opts.seed ?? 0n,
        rgbUri: uris.rgbUri,
        depthUri: uris.depthUri,
        segUri: uris.segUri,
        // 双目右目附件编码固定走 default 路径（dataURL / inline f32-base64，
        // 与 defaultEncodeUris 同构——见 FrameRecordInput.stereo 注释）
        stereo: frame.stereo
          ? {
              baselineM: frame.stereo.baselineM,
              mode: frame.stereo.mode,
              rightRgbUri: frame.stereo.rightRgb,
              rightDepthUri: frame.stereo.rightDepth
                ? `inline:f32-base64;${f32ToBase64(frame.stereo.rightDepth)}`
                : undefined,
            }
          : undefined,
        commandTargetEnu: mock.targetEnu,
        commandSetpoint: mock.setpoint,
        velEnu: mock.velEnu,
        posEnu: mock.posEnu,
      }));
      this._opts.onFrame?.(this._frames[this._frames.length - 1]!, frame);
    } catch (err) {
      // 采集中失败按丢帧计（GPU 忙/设备丢失），不中断录制
      console.warn('[SensorRecorder] capture 失败，本帧丢弃:', err);
    } finally {
      const overrun = performance.now() - t0 - 1000 / this._opts.hz;
      if (overrun > 0) this._droppedMs += overrun;
      this._inFlight = false;
    }
  }
}
