/**
 * Telemetry - 云台遥测数据模型与处理。
 *
 * 对接 DJI 上云 API `device_osd` 报文：解析、插值、按时间缓冲。
 * 引擎内部状态约定：角度一律弧度、距离一律米。
 */
import { CesiumMath } from "../math/CesiumMath";

/** 引擎内部遥测状态：角度全弧度、距离米。 */
export interface GimbalState {
  timestampMs: number;
  lonRad: number;
  latRad: number;
  heightMeters: number;
  headingRad: number;
  pitchRad: number;
  rollRad: number;
  fovYRad: number;
  aspect: number;
  zoomFactor: number;
}

/** DeviceOsdParser 可选项。 */
export interface DeviceOsdParseOptions {
  /** 无变焦时的垂直视场角（度），默认 45。 */
  fovYDeg?: number;
  /** 视口宽高比，默认 16/9。 */
  aspect?: number;
  /** 由 zoom_factor 计算 fovY（度）的回调；提供时优先于 fovYDeg。 */
  fovFromZoom?: (zoom: number) => number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function num(value: unknown, fallback: number): number {
  return finiteNumber(value) ? value : fallback;
}

/**
 * DJI 上云 API device_osd 报文解析器。
 *
 * 校验失败（biz_code 不符 / data.host 缺失 / payload[0] 缺失 / 经纬高缺失）返回 null；
 * 其余字段缺省 0，zoom_factor 缺省 1。
 */
export class DeviceOsdParser {
  static parse(raw: unknown, opts: DeviceOsdParseOptions = {}): GimbalState | null {
    if (!isRecord(raw) || raw.biz_code !== "device_osd") return null;

    const data = raw.data;
    if (!isRecord(data) || !isRecord(data.host)) return null;
    const host = data.host;

    const payload = host.payload;
    if (!Array.isArray(payload) || payload.length === 0 || !isRecord(payload[0])) return null;
    const gimbal = payload[0];

    // 经纬高必须有，否则报文无效
    if (!finiteNumber(host.longitude) || !finiteNumber(host.latitude) || !finiteNumber(host.height)) {
      return null;
    }

    const zoomFactor = num(gimbal.zoom_factor, 1.0);
    const fovYDeg =
      typeof opts.fovFromZoom === "function" ? opts.fovFromZoom(zoomFactor) : (opts.fovYDeg ?? 45.0);

    return {
      timestampMs: num(raw.timestamp, 0),
      lonRad: CesiumMath.toRadians(host.longitude),
      latRad: CesiumMath.toRadians(host.latitude),
      heightMeters: host.height,
      headingRad: CesiumMath.toRadians(num(gimbal.gimbal_yaw, 0.0)),
      pitchRad: CesiumMath.toRadians(num(gimbal.gimbal_pitch, 0.0)),
      rollRad: CesiumMath.toRadians(num(gimbal.gimbal_roll, 0.0)),
      fovYRad: CesiumMath.toRadians(fovYDeg),
      aspect: opts.aspect ?? 16 / 9,
      zoomFactor,
    };
  }
}

/** 最短角差路径插值，结果 wrap 到 [-PI, PI]。 */
function lerpAngle(a: number, b: number, t: number): number {
  const diff = CesiumMath.negativePiToPi(b - a);
  return CesiumMath.negativePiToPi(a + diff * t);
}

/** GimbalState 线性插值，写入 out 并返回。朝向角走最短角差路径。 */
export function lerpGimbal(a: GimbalState, b: GimbalState, t: number, out: GimbalState): GimbalState {
  out.timestampMs = CesiumMath.lerp(a.timestampMs, b.timestampMs, t);
  out.lonRad = CesiumMath.lerp(a.lonRad, b.lonRad, t);
  out.latRad = CesiumMath.lerp(a.latRad, b.latRad, t);
  out.heightMeters = CesiumMath.lerp(a.heightMeters, b.heightMeters, t);
  out.headingRad = lerpAngle(a.headingRad, b.headingRad, t);
  out.pitchRad = lerpAngle(a.pitchRad, b.pitchRad, t);
  out.rollRad = lerpAngle(a.rollRad, b.rollRad, t);
  out.fovYRad = CesiumMath.lerp(a.fovYRad, b.fovYRad, t);
  out.aspect = CesiumMath.lerp(a.aspect, b.aspect, t);
  out.zoomFactor = CesiumMath.lerp(a.zoomFactor, b.zoomFactor, t);
  return out;
}

/** 按时间戳排序的遥测帧缓冲，支持带延迟补偿的插值采样。 */
export class TelemetryBuffer {
  private states: GimbalState[] = [];
  private readonly scratch: GimbalState = {
    timestampMs: 0,
    lonRad: 0,
    latRad: 0,
    heightMeters: 0,
    headingRad: 0,
    pitchRad: 0,
    rollRad: 0,
    fovYRad: 0,
    aspect: 0,
    zoomFactor: 0,
  };

  constructor(private readonly capacity: number = 500) {}

  /** 按时间戳升序插入（稳定），超过容量丢最旧。 */
  push(state: GimbalState): void {
    let i = this.states.length;
    while (i > 0 && this.states[i - 1].timestampMs > state.timestampMs) i--;
    this.states.splice(i, 0, state);
    if (this.states.length > this.capacity) this.states.shift();
  }

  /**
   * 采样 atMs - latencyMs 时刻的遥测状态（两帧插值）。
   * 越界取最近帧；缓冲为空返回 null。
   */
  sample(atMs: number, latencyMs: number = 0): GimbalState | null {
    const states = this.states;
    const n = states.length;
    if (n === 0) return null;

    const target = atMs - latencyMs;
    if (target <= states[0].timestampMs) return states[0];
    if (target >= states[n - 1].timestampMs) return states[n - 1];

    // 二分找最后一个 timestampMs <= target 的帧
    let lo = 0;
    let hi = n - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (states[mid].timestampMs <= target) lo = mid;
      else hi = mid - 1;
    }
    const a = states[lo];
    const b = states[lo + 1];
    const t = (target - a.timestampMs) / (b.timestampMs - a.timestampMs);
    return lerpGimbal(a, b, t, this.scratch);
  }

  clear(): void {
    this.states.length = 0;
  }
}
