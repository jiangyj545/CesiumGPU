/**
 * TelemetrySource - 遥测数据源抽象。
 *
 * 数据源把原始报文（device_osd 形状的 unknown）推给回调，
 * 解析统一交给 DeviceOsdParser（src/media/Telemetry.ts）。
 */
import type { DeviceOsdParseOptions } from "./Telemetry";
import { CesiumMath } from "../math/CesiumMath";

export interface TelemetrySource {
  start(onMessage: (raw: unknown) => void): void;
  stop(): void;
}

export interface MockTelemetryOptions {
  /** 轨道圆心（度）。缺失时构造抛 Error。 */
  center: { lon: number; lat: number };
  /** 轨道半径（米），默认 800。 */
  orbitRadiusMeters?: number;
  /** 固定高度（米），默认 400。 */
  altitudeMeters?: number;
  /** 角速度（度/秒），默认 6（60s 一圈，逆时针）。 */
  angularSpeedDegPerSec?: number;
  /** 发送频率（Hz），默认 10。 */
  rateHz?: number;
  /** 视场角（度），默认 45。 */
  fovYDeg?: number;
  /** 时钟源（毫秒），默认 performance.now；测试注入虚拟时钟。 */
  clock?: () => number;
}

const METERS_PER_DEG_LAT = 110540;
const METERS_PER_DEG_LON_EQUATOR = 111320;

/** 模拟遥测源：飞机绕 center 逆时针匀速绕圈，镜头指向圆心。 */
export class MockTelemetrySource implements TelemetrySource {
  private readonly center: { lon: number; lat: number };
  private readonly radiusMeters: number;
  private readonly altitudeMeters: number;
  private readonly omegaRadPerSec: number;
  private readonly rateHz: number;
  /** 视场角（度），默认 45。供消费方（如 VideoOverlay）读取。 */
  readonly fovYDeg: number;
  private readonly clock: () => number;

  private intervalId: ReturnType<typeof setInterval> | null = null;
  private startClockMs = 0;

  constructor(options: MockTelemetryOptions) {
    if (
      typeof options !== "object" ||
      options === null ||
      typeof options.center !== "object" ||
      options.center === null ||
      !Number.isFinite(options.center.lon) ||
      !Number.isFinite(options.center.lat)
    ) {
      throw new Error("MockTelemetrySource: center { lon, lat } is required");
    }
    this.center = { lon: options.center.lon, lat: options.center.lat };
    this.radiusMeters = options.orbitRadiusMeters ?? 800;
    this.altitudeMeters = options.altitudeMeters ?? 400;
    this.omegaRadPerSec = CesiumMath.toRadians(options.angularSpeedDegPerSec ?? 6.0);
    this.rateHz = options.rateHz ?? 10;
    this.fovYDeg = options.fovYDeg ?? 45;
    this.clock = options.clock ?? performance.now.bind(performance);
  }

  start(onMessage: (raw: unknown) => void): void {
    this.stop(); // 幂等：重复 start 先清理旧 interval
    this.startClockMs = this.clock();
    this.intervalId = setInterval(() => {
      onMessage(this.buildMessage(this.clock()));
    }, 1000 / this.rateHz);
  }

  stop(): void {
    if (this.intervalId !== null) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  private buildMessage(nowMs: number): unknown {
    const tSec = (nowMs - this.startClockMs) / 1000;
    const theta = this.omegaRadPerSec * tSec; // 逆时针绕圈相位

    const lon =
      this.center.lon +
      (this.radiusMeters * Math.cos(theta)) /
        (METERS_PER_DEG_LON_EQUATOR * Math.cos(CesiumMath.toRadians(this.center.lat)));
    const lat = this.center.lat + (this.radiusMeters * Math.sin(theta)) / METERS_PER_DEG_LAT;

    const gimbalYawDeg = CesiumMath.toDegrees(theta + CesiumMath.PI_OVER_TWO); // 镜头指向圆心
    const gimbalPitchDeg = CesiumMath.toDegrees(-Math.atan2(this.altitudeMeters, this.radiusMeters));

    return {
      biz_code: "device_osd",
      timestamp: nowMs,
      data: {
        host: {
          longitude: lon,
          latitude: lat,
          height: this.altitudeMeters,
          elevation: 0,
          payload: [
            {
              payload_index: "39-0-0",
              gimbal_yaw: gimbalYawDeg,
              gimbal_pitch: gimbalPitchDeg,
              gimbal_roll: 0,
              zoom_factor: 1,
            },
          ],
        },
      },
    };
  }
}

/**
 * WebSocket 遥测源：onmessage JSON.parse 后透传原始报文；
 * 意外断开自动重连（默认 5s 起指数退避），stop 后不再重连。
 */
export class WebSocketTelemetrySource implements TelemetrySource {
  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private onMessage: ((raw: unknown) => void) | null = null;
  private stopped = true;
  private attempts = 0;

  constructor(
    private readonly url: string,
    /** 透传给 DeviceOsdParser 的解析选项，由消费方读取使用。 */
    public readonly parseOptions: DeviceOsdParseOptions = {},
    private readonly reconnectDelayMs: number = 5000,
  ) {}

  start(onMessage: (raw: unknown) => void): void {
    this.stop();
    this.onMessage = onMessage;
    this.stopped = false;
    this.attempts = 0;
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws !== null) {
      this.ws.onclose = null;
      this.ws.onmessage = null;
      this.ws.close();
      this.ws = null;
    }
    this.onMessage = null;
  }

  private connect(): void {
    const ws = new WebSocket(this.url);
    this.ws = ws;

    ws.onmessage = (ev: MessageEvent) => {
      try {
        this.onMessage?.(JSON.parse(String(ev.data)));
      } catch (e) {
        // 坏报文只告警不断流
        console.error("WebSocketTelemetrySource: invalid JSON message", e);
      }
    };

    ws.onclose = () => {
      if (this.stopped || this.ws !== ws) return;
      this.ws = null;
      const delay = this.reconnectDelayMs * Math.pow(2, Math.min(this.attempts++, 5));
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null;
        if (!this.stopped) this.connect();
      }, delay);
    };
  }
}
