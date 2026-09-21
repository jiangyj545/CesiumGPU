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
 *  - 分割粒度 = Primitive（合批 chunk）级：pickId 是 per-Primitive 的，
 *    每顶点 semanticId 尚未接入 GPU 管线（CityBlockBuilder 已预留数据）；
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
  meta: SensorFrameMeta;
}

/** pickId → 语义 id 的可注入映射（0 = 天空/背景）。 */
export type PickIdToSemantic = (pickId: number) => number;

export interface SensorCamOptions extends SensorCamParams {
  /** pickId→语义映射，缺省全 0（全画面视作背景）。 */
  pickIdToSemantic?: PickIdToSemantic;
  /** 调色板（缺省 {@link DEFAULT_PALETTE}）。 */
  palette?: Readonly<Record<number, readonly [number, number, number]>>;
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
  }

  get params(): Readonly<SensorCamParams> { return this._params; }

  /** 场景是否已初始化（G-Buffer 就绪，可 capture）。 */
  isReady(): boolean {
    return this._scene.getDepthTexture() !== null && this._scene.gbufferPass !== null;
  }

  /**
   * 以给定位姿渲染一帧并导出三通道。渲染前保存相机状态、渲染后原样还原
   * （position/direction/up/right/frustum 全量），主循环渲染不受污染。
   */
  async capture(pose: SensorPose, tCaptureSim?: number): Promise<SensorFrame> {
    const scene = this._scene;
    if (!this.isReady()) throw new Error('SensorCam.capture: scene 尚未 initialize（G-Buffer 不存在）');

    const cam = scene.camera;
    const context = scene.context;
    // ── 保存相机 ──
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

    const pickPass = scene.createPickPass();
    const depthTex = scene.getDepthTexture()!;
    const w = depthTex.width;
    const h = depthTex.height;
    const device = context.device;

    // 深度读回缓冲（bytesPerRow 256 对齐；depth32float 每像素 4B）
    const depthBytesPerRow = Math.ceil((w * 4) / 256) * 256;
    const depthBuf = device.createBuffer({
      label: 'SensorCam.DepthReadback',
      size: depthBytesPerRow * h,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // 一次性帧钩子：把 G-Buffer 深度全幅拷进读回缓冲（同帧、同投影）
    const depthHook = (encoder: GPUCommandEncoder): void => {
      encoder.copyTextureToBuffer(
        { texture: depthTex },
        { buffer: depthBuf, bytesPerRow: depthBytesPerRow },
        { width: w, height: h },
      );
    };
    scene.frameHooks.push(depthHook);

    // 全幅 id 纹理模式：本次 render 编码的 pick view 不做 1×1 scissor
    const savedDebugFull = pickPass.debugFullViewport;
    pickPass.debugFullViewport = true;

    try {
      // ── 设传感器位姿（fov/aspect 变更需经 setAspectRatio 触发 proj 重算）──
      cam.setView({
        destination: { lon: pose.lonDeg, lat: pose.latDeg, height: pose.heightM },
        orientation: { heading: pose.headingRad, pitch: pose.pitchRad, roll: pose.rollRad },
      });
      cam.frustum.fov = CesiumMath.toRadians(this._params.fovYDeg);
      cam.setAspectRatio(w / h);

      // 注册 pending pick：本次 render 的帧钩子会全幅编码 id 平面
      const pickPromise = scene.pick(0, 0);
      // W4 TEMP DEBUG: render-time camera snapshot（诊断后移除）
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

      // 一次性帧钩子已完成使命：本次 render 的 submit 已包含深度拷贝，
      // 立即摘除（finally 里的移除兜底异常路径）。若留到 finally，主循环
      // RAF 的每一帧都会重复编码 copy → depthBuf，与下方 mapAsync 的
      // mapped 生命周期交错 → "used in submit while mapped" validation
      // error → 整帧 submit 作废 → 录制态渐进黑帧（深度采样归零）。
      const hookIdx = scene.frameHooks.indexOf(depthHook);
      if (hookIdx !== -1) scene.frameHooks.splice(hookIdx, 1);

      // ── RGB：主 canvas 原生导出（同帧已呈现）──
      const rgb = context.canvas.toDataURL('image/png');

      // ── seg：全幅 id 纹理读回 ──
      const dump = await pickPass.debugDumpPickTexture();
      void pickPromise; // pick 结果本身不消费（取整幅 id 纹理）

      // ── depth：map + 剥离行 stride + 线性化 ──
      await depthBuf.mapAsync(GPUMapMode.READ);
      const raw = new Uint8Array(depthBuf.getMappedRange());
      const tightlyPacked = destrideBytes(raw, w, h, depthBytesPerRow, 4);
      depthBuf.unmap();
      const f32 = new Float32Array(tightlyPacked.buffer);
      const nearNorm = cam.frustum.near;
      const farNorm = cam.frustum.far;
      const depthM = new Float32Array(w * h);
      for (let i = 0; i < f32.length; i++) {
        depthM[i] = linearizeDepthMeters(f32[i], nearNorm, farNorm);
      }

      // ── seg 解码：pickId → 语义 → 调色板 ──
      const segData = dump ?? { w, h, data: new Uint8Array(w * h * 4) };
      const seg = new Uint8Array(w * h * 4);
      const segIds = new Uint32Array(w * h);
      const sw = Math.min(segData.w, w);
      const sh = Math.min(segData.h, h);
      for (let y = 0; y < sh; y++) {
        for (let x = 0; x < sw; x++) {
          const si = (y * segData.w + x) * 4;
          const di = y * w + x;
          const dec = decodeSegPixel(
            segData.data[si], segData.data[si + 1], segData.data[si + 2],
            this._pickIdToSemantic, this._palette,
          );
          seg[di * 4]     = dec.rgba[0];
          seg[di * 4 + 1] = dec.rgba[1];
          seg[di * 4 + 2] = dec.rgba[2];
          seg[di * 4 + 3] = dec.rgba[3];
          segIds[di] = dec.semanticId;
        }
      }

      // ── meta：内参 / 位姿真值 ──
      const fx = (h / 2) / Math.tan(cam.frustum.fov / 2);
      const hfovRad = 2 * Math.atan((w / 2) / fx);
      // FLU 基（forward/left/up）→ body→ENU 四元数
      const fwd = Cartesian3.normalize(cam.direction, new Cartesian3());
      const upW = Cartesian3.normalize(cam.up, new Cartesian3());
      const left = Cartesian3.normalize(Cartesian3.cross(upW, fwd, new Cartesian3()), new Cartesian3());
      const quat = quatFromFLU(fwd, left, upW);
      const pos = cam.position;

      return {
        width: w,
        height: h,
        rgb,
        depth: depthM,
        seg,
        segIds,
        meta: {
          intrinsics: { fx, fy: fx, cx: w / 2, cy: h / 2 },
          hfovDeg: CesiumMath.toDegrees(hfovRad),
          tCaptureSim: tCaptureSim ?? performance.now() / 1000,
          camPosEcefM: [pos.x * EARTH_SCALE, pos.y * EARTH_SCALE, pos.z * EARTH_SCALE],
          camQuat: quat,
          nearM: nearNorm * EARTH_SCALE,
          farM: farNorm * EARTH_SCALE,
          requestedNearM: this._params.nearM,
          requestedFarM: this._params.farM,
          fovYDeg: this._params.fovYDeg,
          width: this._params.width,
          height: this._params.height,
        },
      };
    } finally {
      // ── 还原相机与全局状态（无论成败）──
      const hookIdx = scene.frameHooks.indexOf(depthHook);
      if (hookIdx !== -1) scene.frameHooks.splice(hookIdx, 1);
      pickPass.debugFullViewport = savedDebugFull;

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
      depthBuf.destroy();
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
    seg: { uri: string; palette_map_ref: string };
    bbox2d: Array<{ cls: string; track_id: number; x1: number; y1: number; x2: number; y2: number }>;
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
