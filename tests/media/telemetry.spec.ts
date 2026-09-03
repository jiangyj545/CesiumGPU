import { describe, it, expect, vi, afterEach } from "vitest";
import { DeviceOsdParser, lerpGimbal, TelemetryBuffer, GimbalState } from "../../src/media/Telemetry";
import { MockTelemetrySource } from "../../src/media/TelemetrySource";
import type { MockTelemetryOptions } from "../../src/media/TelemetrySource";
import { CesiumMath } from "../../src/math/CesiumMath";

/** 任务说明中的 device_osd 合成样例报文 */
const SAMPLE_OSD = {
  biz_code: "device_osd",
  timestamp: 1725345600123,
  data: {
    host: {
      longitude: 116.397,
      latitude: 39.908,
      height: 120.5,
      elevation: 43.2,
      payload: [
        {
          payload_index: "39-0-0",
          gimbal_yaw: 45.0,
          gimbal_pitch: -30.0,
          gimbal_roll: 0.0,
          zoom_factor: 1.0,
        },
      ],
    },
  },
};

function makeState(timestampMs: number, heightMeters: number): GimbalState {
  return {
    timestampMs,
    lonRad: CesiumMath.toRadians(116),
    latRad: CesiumMath.toRadians(39.9),
    heightMeters,
    headingRad: 0,
    pitchRad: 0,
    rollRad: 0,
    fovYRad: CesiumMath.PI_OVER_FOUR,
    aspect: 16 / 9,
    zoomFactor: 1,
  };
}

/** 等距圆柱近似地面距离（米），仅用于测试量级验证 */
function groundDistanceMeters(lon: number, lat: number, centerLon: number, centerLat: number): number {
  const dLonM = (lon - centerLon) * 111320 * Math.cos(((lat + centerLat) / 2) * (Math.PI / 180));
  const dLatM = (lat - centerLat) * 110540;
  return Math.hypot(dLonM, dLatM);
}

describe("DeviceOsdParser", () => {
  it("解析样例报文：度转弧度、timestampMs 透传", () => {
    const s = DeviceOsdParser.parse(SAMPLE_OSD);
    expect(s).not.toBeNull();
    expect(s!.lonRad).toBeCloseTo(CesiumMath.toRadians(116.397), 12);
    expect(s!.latRad).toBeCloseTo(CesiumMath.toRadians(39.908), 12);
    expect(s!.heightMeters).toBe(120.5);
    expect(s!.headingRad).toBeCloseTo(CesiumMath.toRadians(45), 12);
    expect(s!.pitchRad).toBeCloseTo(-Math.PI / 6, 12);
    expect(s!.rollRad).toBeCloseTo(0, 12);
    expect(s!.timestampMs).toBe(1725345600123);
    expect(s!.zoomFactor).toBe(1.0);
  });

  it("默认 fovY=45°、aspect=16/9；可用 fovYDeg/aspect 覆盖", () => {
    const s = DeviceOsdParser.parse(SAMPLE_OSD);
    expect(s!.fovYRad).toBeCloseTo(Math.PI / 4, 12);
    expect(s!.aspect).toBeCloseTo(16 / 9, 12);

    const custom = DeviceOsdParser.parse(SAMPLE_OSD, { fovYDeg: 60, aspect: 2 });
    expect(custom!.fovYRad).toBeCloseTo(CesiumMath.toRadians(60), 12);
    expect(custom!.aspect).toBe(2);
  });

  it("biz_code 不符返回 null", () => {
    const bad = { ...SAMPLE_OSD, biz_code: "device_hms" };
    expect(DeviceOsdParser.parse(bad)).toBeNull();
  });

  it("缺 payload / 缺经纬高返回 null", () => {
    const noPayload = structuredClone(SAMPLE_OSD) as { data: { host: Record<string, unknown> } };
    delete noPayload.data.host.payload;
    expect(DeviceOsdParser.parse(noPayload)).toBeNull();

    const noLon = structuredClone(SAMPLE_OSD) as { data: { host: Record<string, unknown> } };
    delete noLon.data.host.latitude;
    expect(DeviceOsdParser.parse(noLon)).toBeNull();
  });

  it("zoom_factor 传给 fovFromZoom 并以返回值计算 fovY", () => {
    const received: number[] = [];
    const zoomed = {
      ...structuredClone(SAMPLE_OSD),
      data: {
        host: {
          ...SAMPLE_OSD.data.host,
          payload: [{ ...SAMPLE_OSD.data.host.payload[0], zoom_factor: 2.0 }],
        },
      },
    };
    const s = DeviceOsdParser.parse(zoomed, {
      fovFromZoom: (z: number) => {
        received.push(z);
        return 30 / z;
      },
    });
    expect(received).toEqual([2.0]);
    expect(s!.zoomFactor).toBe(2.0);
    expect(s!.fovYRad).toBeCloseTo(CesiumMath.toRadians(15), 12);
  });
});

describe("lerpGimbal", () => {
  const a = makeState(1000, 10);
  const b = makeState(2000, 20);

  it("heading 走最短角差路径：350°→10° t=0.5 过 0 得 0°，不过 180°", () => {
    const out: GimbalState = { ...a };
    a.headingRad = CesiumMath.toRadians(350);
    b.headingRad = CesiumMath.toRadians(10);
    lerpGimbal(a, b, 0.5, out);
    expect(out.headingRad).toBeCloseTo(0, 9);
    lerpGimbal(a, b, 1, out);
    expect(out.headingRad).toBeCloseTo(CesiumMath.toRadians(10), 9);
  });

  it("lon / pitch 普通线性插值", () => {
    const out: GimbalState = { ...a };
    a.lonRad = CesiumMath.toRadians(116);
    b.lonRad = CesiumMath.toRadians(117);
    a.pitchRad = CesiumMath.toRadians(-30);
    b.pitchRad = CesiumMath.toRadians(-40);
    lerpGimbal(a, b, 0.25, out);
    expect(out.lonRad).toBeCloseTo(CesiumMath.toRadians(116.25), 12);
    lerpGimbal(a, b, 0.5, out);
    expect(out.pitchRad).toBeCloseTo(CesiumMath.toRadians(-35), 12);
  });

  it("timestamp / height 线性，结果写入 out 并返回同一引用", () => {
    const out: GimbalState = { ...a };
    const ret = lerpGimbal(a, b, 0.5, out);
    expect(ret).toBe(out);
    expect(out.timestampMs).toBeCloseTo(1500, 9);
    expect(out.heightMeters).toBeCloseTo(15, 9);
  });
});

describe("TelemetryBuffer", () => {
  it("乱序 push 后按时间排序，采样值随时间单调", () => {
    const buf = new TelemetryBuffer();
    buf.push(makeState(1000, 10));
    buf.push(makeState(3000, 30));
    buf.push(makeState(2000, 20));

    const s1 = buf.sample(1000)!;
    const s2 = buf.sample(2000)!;
    const s3 = buf.sample(3000)!;
    expect(s1.heightMeters).toBeLessThanOrEqual(s2.heightMeters);
    expect(s2.heightMeters).toBeLessThanOrEqual(s3.heightMeters);
    // 中点落在 2000-3000 两帧之间插值
    expect(buf.sample(2500)!.heightMeters).toBeCloseTo(25, 9);
  });

  it("latencyMs 补偿：sample(1000, latency 200) 使用 800ms 附近帧", () => {
    const buf = new TelemetryBuffer();
    buf.push(makeState(700, 7));
    buf.push(makeState(900, 9));
    expect(buf.sample(1000, 200)!.heightMeters).toBeCloseTo(8, 9);
  });

  it("空 buffer 返回 null", () => {
    expect(new TelemetryBuffer().sample(0)).toBeNull();
  });

  it("越界取最近帧", () => {
    const buf = new TelemetryBuffer();
    buf.push(makeState(1000, 10));
    buf.push(makeState(2000, 20));
    expect(buf.sample(500)!.timestampMs).toBe(1000);
    expect(buf.sample(9999)!.timestampMs).toBe(2000);
  });

  it("超过默认容量 500 丢最旧", () => {
    const buf = new TelemetryBuffer();
    for (let ts = 0; ts <= 500; ts++) buf.push(makeState(ts, ts));
    expect(buf.sample(0)!.timestampMs).toBe(1); // ts=0 已被丢弃
    expect(buf.sample(9999)!.timestampMs).toBe(500);
  });

  it("clear 后采样返回 null", () => {
    const buf = new TelemetryBuffer();
    buf.push(makeState(1000, 10));
    buf.clear();
    expect(buf.sample(1000)).toBeNull();
  });
});

describe("MockTelemetrySource", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("构造时 center 缺失抛 Error", () => {
    expect(() => new MockTelemetrySource({} as unknown as MockTelemetryOptions)).toThrow();
    expect(() => new MockTelemetrySource({ center: { lon: NaN, lat: 39.9 } })).toThrow();
  });

  it("推进 1s 收到 10 条：位置在圆上、pitch 对准圆心、timestamp 来自虚拟时钟", () => {
    vi.useFakeTimers();
    let now = 1_725_345_600_000;
    const clock = () => now;
    const center = { lon: 116, lat: 39.9 };
    const src = new MockTelemetrySource({
      center,
      clock,
      rateHz: 10,
      orbitRadiusMeters: 800,
      altitudeMeters: 400,
    });

    const msgs: Array<Record<string, any>> = [];
    src.start((raw) => msgs.push(raw as Record<string, any>));
    for (let i = 0; i < 10; i++) {
      now += 100;
      vi.advanceTimersByTime(100);
    }

    expect(msgs).toHaveLength(10);
    // timestampMs 来自注入的虚拟时钟且严格递增
    expect(msgs[0].timestamp).toBe(1_725_345_600_100);
    expect(msgs[9].timestamp).toBe(1_725_345_601_000);
    for (let i = 1; i < msgs.length; i++) {
      expect(msgs[i].timestamp).toBeGreaterThan(msgs[i - 1].timestamp);
    }

    for (const m of msgs) {
      expect(m.biz_code).toBe("device_osd");
      const host = m.data.host;
      // 地面距离 ≈ 轨道半径
      expect(groundDistanceMeters(host.longitude, host.latitude, center.lon, center.lat)).toBeCloseTo(800, 0);
      expect(host.height).toBe(400);

      const p = host.payload[0];
      expect(p.payload_index).toBe("39-0-0");
      expect(p.gimbal_pitch).toBeCloseTo((-Math.atan2(400, 800) * 180) / Math.PI, 6);
      expect(p.gimbal_roll).toBe(0);
      expect(p.zoom_factor).toBe(1);
    }

    src.stop();
  });

  it("stop 后不再发送", () => {
    vi.useFakeTimers();
    const src = new MockTelemetrySource({ center: { lon: 116, lat: 39.9 }, rateHz: 10 });
    const msgs: unknown[] = [];
    src.start((m) => msgs.push(m));

    vi.advanceTimersByTime(500);
    expect(msgs).toHaveLength(5);

    src.stop();
    vi.advanceTimersByTime(1000);
    expect(msgs).toHaveLength(5);
  });

  it("重复 start 幂等（先 stop 旧 interval，不重复发送）", () => {
    vi.useFakeTimers();
    const src = new MockTelemetrySource({ center: { lon: 116, lat: 39.9 }, rateHz: 10 });
    const msgs: unknown[] = [];
    src.start(() => undefined);
    src.start((m) => msgs.push(m)); // 二次 start 覆盖第一次

    vi.advanceTimersByTime(1000);
    expect(msgs).toHaveLength(10);
    src.stop();
  });
});
