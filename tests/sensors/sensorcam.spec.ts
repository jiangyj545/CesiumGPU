import { describe, expect, it } from 'vitest';
import {
  DEFAULT_PALETTE,
  PALETTE_MAP_VERSION,
  SensorRecorder,
  buildFrameRecord,
  decodeSegPixel,
  linearizeDepthMeters,
  quatFromFLU,
} from '../../src/sensors/SensorCam';
import type {
  FrameRecordInput,
  SensorCam,
  SensorFrame,
} from '../../src/sensors/SensorCam';

const EARTH_SCALE = 6378137;

// ── 深度线性化数学 ─────────────────────────────────────────────────────────────

describe('linearizeDepthMeters（引擎归一化 [0,1] 深度 → 真实米）', () => {
  // 引擎低空典型 near/far（120m 高度时动态策略产出）
  const nearNorm = 1e-5;
  const farNorm = 0.0228;

  it('d=0 → near，d=1 → far（米制端点自洽）', () => {
    expect(linearizeDepthMeters(0, nearNorm, farNorm)).toBeCloseTo(nearNorm * EARTH_SCALE, 6);
    expect(linearizeDepthMeters(1, nearNorm, farNorm)).toBeCloseTo(farNorm * EARTH_SCALE, 4);
  });

  it('投影-反投影往返：z → d → z（已知深度值验证米制换算）', () => {
    // Matrix4.perspective 约定：d = -A + B/z，A=f/(n-f)，B=f·n/(n-f)
    const n = nearNorm, f = farNorm;
    const A = f / (n - f);
    const B = (f * n) / (n - f);
    for (const zMeters of [64, 120, 300, 1500, 20000]) {
      const zNorm = zMeters / EARTH_SCALE;
      const d = -A + B / zNorm;
      expect(d).toBeGreaterThan(0);
      expect(d).toBeLessThan(1);
      const backM = linearizeDepthMeters(d, n, f);
      expect(backM).toBeCloseTo(zMeters, 4);
    }
  });

  it('单调递增：深度值越大越远', () => {
    let prev = -Infinity;
    for (let i = 0; i <= 20; i++) {
      const m = linearizeDepthMeters(i / 20, nearNorm, farNorm);
      expect(m).toBeGreaterThan(prev);
      prev = m;
    }
  });

  it('120m 高度 nadir：地面像素（d 对应视深 120m）换算回 120m', () => {
    const n = nearNorm, f = farNorm;
    const A = f / (n - f);
    const B = (f * n) / (n - f);
    const d120 = -A + B / (120 / EARTH_SCALE);
    expect(linearizeDepthMeters(d120, n, f)).toBeCloseTo(120, 3);
  });
});

// ── 调色板映射 ────────────────────────────────────────────────────────────────

describe('decodeSegPixel（pickId → 语义 id → RGBA 查表）', () => {
  const map = (id: number): number => (id === 100 ? 1 : id === 200 ? 3 : 0);

  it('pickId 24bit 解码（RGB 低 3 字节小端）→ 注入映射 → 调色板查色', () => {
    // pickId = 100 = r:100,g:0,b:0 → building 红
    const b = decodeSegPixel(100, 0, 0, map, DEFAULT_PALETTE);
    expect(b.semanticId).toBe(1);
    expect(b.rgba).toEqual([255, 0, 0, 255]);
    // pickId = 200 = r:200,g:0,b:0 → ground 褐
    const g = decodeSegPixel(200, 0, 0, map, DEFAULT_PALETTE);
    expect(g.semanticId).toBe(3);
    expect(g.rgba).toEqual([139, 90, 43, 255]);
  });

  it('pickId=0（背景/天空）恒语义 0，不触发注入映射', () => {
    let called = false;
    const zero = decodeSegPixel(0, 0, 0, () => { called = true; return 1; }, DEFAULT_PALETTE);
    expect(zero.semanticId).toBe(0);
    expect(zero.rgba).toEqual([0, 0, 0, 255]);
    expect(called).toBe(false);
  });

  it('高字节组合：pickId = 1 | 2<<8 | 3<<16 = 197121', () => {
    let seen = -1;
    const dec = decodeSegPixel(1, 2, 3, (id) => { seen = id; return 4; }, DEFAULT_PALETTE);
    expect(seen).toBe(1 | (2 << 8) | (3 << 16));
    expect(dec.semanticId).toBe(4);
    expect(dec.rgba).toEqual([80, 140, 255, 255]);
  });

  it('未知语义 id 兜底为黑色（调色板缺项不抛错）', () => {
    const dec = decodeSegPixel(50, 0, 0, () => 99, DEFAULT_PALETTE);
    expect(dec.rgba).toEqual([0, 0, 0, 255]);
  });
});

// ── FLU 四元数 ───────────────────────────────────────────────────────────────

describe('quatFromFLU（body(FLU)→ENU，[w,x,y,z]）', () => {
  it('单位基 → 单位四元数', () => {
    expect(quatFromFLU({ x: 1, y: 0, z: 0 }, { x: 0, y: 1, z: 0 }, { x: 0, y: 0, z: 1 }))
      .toEqual([1, 0, 0, 0]);
  });

  it('绕 up 轴 90°（东向飞行）→ [√2/2, 0, 0, √2/2]', () => {
    const q = quatFromFLU(
      { x: 0, y: 1, z: 0 },   // forward = 东
      { x: -1, y: 0, z: 0 },  // left = 西
      { x: 0, y: 0, z: 1 },   // up = 天
    );
    const s = Math.SQRT1_2;
    expect(q[0]).toBeCloseTo(s, 12);
    expect(q[1]).toBeCloseTo(0, 12);
    expect(q[2]).toBeCloseTo(0, 12);
    expect(q[3]).toBeCloseTo(s, 12);
  });

  it('任意正交基下模长恒 1', () => {
    const q = quatFromFLU(
      { x: 0, y: -1, z: 0 }, { x: 1, y: 0, z: 0 }, { x: 0, y: 0, z: 1 },
    );
    expect(Math.hypot(q[0], q[1], q[2], q[3])).toBeCloseTo(1, 12);
  });
});

// ── 采样调度器时序（fake timer）──────────────────────────────────────────────

/** 无 GPU 的 mock 传感器相机（capture 立即产出最小合法帧）。 */
function makeMockCam(): { cam: SensorCam; captureCalls: number[] } {
  const captureCalls: number[] = [];
  const cam = {
    capture: async (_pose: unknown, t: number) => {
      captureCalls.push(t);
      return makeMockFrame(t);
    },
  } as unknown as SensorCam;
  return { cam, captureCalls };
}

function makeMockFrame(t = 0): SensorFrame {
  return {
    width: 4,
    height: 3,
    rgb: 'data:image/png;base64,MOCK',
    depth: new Float32Array(12).fill(120),
    seg: new Uint8Array(48),
    segIds: new Uint32Array(12),
    meta: {
      intrinsics: { fx: 3.621, fy: 3.621, cx: 2, cy: 1.5 },
      hfovDeg: 95.0,
      tCaptureSim: t,
      camPosEcefM: [-2165334.1, 4733056.5, 3575300.2],
      camQuat: [0.7071, 0, 0, 0.7071],
      nearM: 63.78137,
      farM: 145421.5236,
      requestedNearM: 1,
      requestedFarM: 500,
      fovYDeg: 45,
      width: 4,
      height: 3,
    },
  };
}

/** fake timer：记录注册的回调与周期，测试手动驱动（node 无 WebGPU 同款先例）。 */
function makeFakeTimers() {
  const pending: Array<{ fn: () => void; ms: number; cleared: boolean }> = [];
  return {
    pending,
    timers: {
      set: (fn: () => void, ms: number) => {
        const entry = { fn, ms, cleared: false };
        pending.push(entry);
        return entry;
      },
      clear: (h: unknown) => { (h as { cleared: boolean }).cleared = true; },
    },
  };
}

const poseMock = () => ({
  pose: { lonDeg: 114.393, latDeg: 30.5, heightM: 120, headingRad: 0, pitchRad: -Math.PI / 2, rollRad: 0 },
  targetEnu: [500, 0, -120] as [number, number, number],
  setpoint: { vx: 8, vy: 0, vz: 0, yawRate: 0 },
  velEnu: [8, 0, 0] as [number, number, number],
  posEnu: [0, 0, 120] as [number, number, number],
});

/** 驱动一次采样（等价定时器回调本体，返回可等待的 _tick promise）。 */
const tickOf = (rec: SensorRecorder) =>
  (rec as unknown as { _tick(): Promise<void> })._tick.bind(rec)();

describe('SensorRecorder 调度时序（setInterval 驱动，非 rAF）', () => {
  it('start 按 1000/hz ms 注册定时器（5Hz → 200ms）', () => {
    const { cam } = makeMockCam();
    const { pending, timers } = makeFakeTimers();
    const rec = new SensorRecorder(cam, {
      hz: 5, scenarioId: 's', episodeId: 'e', poseAt: poseMock, timers,
    });
    rec.start();
    expect(rec.recording).toBe(true);
    expect(pending).toHaveLength(1);
    expect(pending[0]!.ms).toBe(200);
    expect(pending[0]!.cleared).toBe(false);
  });

  it('inFlight 期间再触发不重叠采样（去重），完成后恢复', async () => {
    const { cam, captureCalls } = makeMockCam();
    const { timers } = makeFakeTimers();
    const rec = new SensorRecorder(cam, {
      hz: 10, scenarioId: 's', episodeId: 'e', poseAt: poseMock, timers,
    });
    rec.start();
    const p1 = tickOf(rec);   // 第 1 tick：同步置 inFlight
    await tickOf(rec);        // 第 2 tick：应被去重跳过
    await p1;
    expect(captureCalls).toHaveLength(1);
    await tickOf(rec);        // inFlight 清除后可继续采样
    expect(captureCalls).toHaveLength(2);
    await rec.stop();
  });

  it('stop 清除定时器并返回 EpisodeResult（nFrames 与帧数一致，step 递增）', async () => {
    const { cam } = makeMockCam();
    const { pending, timers } = makeFakeTimers();
    const rec = new SensorRecorder(cam, {
      hz: 10, scenarioId: 's', episodeId: 'e', poseAt: poseMock, timers,
    });
    rec.start();
    await tickOf(rec);
    await tickOf(rec);
    await tickOf(rec);
    const result = await rec.stop();
    expect(pending[0]!.cleared).toBe(true);
    expect(result.stats.nFrames).toBe(3);
    expect(result.frames).toHaveLength(3);
    expect(result.frames[0]!.step_index).toBe(0);
    expect(result.frames[2]!.step_index).toBe(2);
    expect(rec.status().recording).toBe(false);
    expect(rec.status().nFrames).toBe(3);
  });

  it('tSim 从 0 起单调递增并写入帧 sim_time_s / meta.tCaptureSim', async () => {
    const { cam, captureCalls } = makeMockCam();
    const { timers } = makeFakeTimers();
    const rec = new SensorRecorder(cam, {
      hz: 10, scenarioId: 's', episodeId: 'e', poseAt: poseMock, timers,
    });
    rec.start();
    await tickOf(rec);
    await tickOf(rec);
    const { frames } = await rec.stop();
    expect(frames[0]!.sim_time_s).toBeLessThanOrEqual(frames[1]!.sim_time_s);
    expect(captureCalls[0]).toBeCloseTo(frames[0]!.sim_time_s, 9);
  });
});

// ── FrameRecord 字段完整性（对齐 dataset-tools/schema/frame-record.ts）───────

describe('buildFrameRecord 字段完整性', () => {
  const baseInput = (): FrameRecordInput => ({
    frame: makeMockFrame(12.5),
    scenarioId: 'guanggu',
    episodeId: 'ep001',
    stepIndex: 7,
    simTimeS: 12.5,
    seed: 42n,
    rgbUri: 'data:image/png;base64,MOCK',
    depthUri: 'inline:f32-base64;AAAA',
    segUri: 'inline:rgba-base64',
    commandTargetEnu: [500, 0, -120],
    commandSetpoint: { vx: 8, vy: 0, vz: 0, yawRate: 0 },
    velEnu: [8, 0, 0],
    posEnu: [100, 50, 120],
  });

  it('顶层与嵌套字段名与 schema 一一对应', () => {
    const rec = buildFrameRecord(baseInput());

    // 顶层字段（schema §9.2 三元对齐：视觉-指令-动作）
    expect(Object.keys(rec).sort()).toEqual([
      'action', 'align', 'command', 'environment', 'episode_id', 'events',
      'frame_id', 'scenario_id', 'seed', 'sim_time_s', 'step_index', 'vision',
    ]);
    expect(rec.frame_id).toBe('guanggu/ep001/000007');
    expect(rec.seed).toBe(42n); // JSON 序列化侧转十进制字符串
    expect(rec.step_index).toBe(7);
    expect(rec.sim_time_s).toBe(12.5);

    // vision.rgb（内参 / 畸变 / hfov）
    expect(Object.keys(rec.vision.rgb).sort()).toEqual([
      'cx', 'cy', 'distortion', 'fx', 'fy', 'h', 'hfov_deg', 'uri', 'w',
    ]);
    expect(rec.vision.rgb.w).toBe(4);
    expect(rec.vision.rgb.h).toBe(3);
    expect(rec.vision.rgb.fx).toBeCloseTo(3.621, 2); // (h/2)/tan(fovY/2)
    expect(rec.vision.rgb.cx).toBe(2);
    expect(rec.vision.rgb.cy).toBe(1.5);
    expect(rec.vision.rgb.distortion).toEqual([0, 0, 0, 0, 0]);

    // vision.depth（near/far 来自 meta 实际线性化值）
    expect(Object.keys(rec.vision.depth).sort()).toEqual(['far_m', 'fmt', 'near_m', 'unit', 'uri']);
    expect(rec.vision.depth.fmt).toBe('f32');
    expect(rec.vision.depth.unit).toBe('m');
    expect(rec.vision.depth.near_m).toBeCloseTo(63.78137, 4);
    expect(rec.vision.depth.far_m).toBeCloseTo(145421.5236, 3);

    // vision.seg（调色板版本化引用）
    expect(rec.vision.seg.palette_map_ref).toBe(PALETTE_MAP_VERSION);
    expect(rec.vision.bbox2d).toEqual([]);

    // command（Phase 0 mock：waypoint 模式 + 航点 target + 速度 setpoint）
    expect(rec.command.mode).toBe('waypoint');
    expect(rec.command.target_enu).toEqual([500, 0, -120]);
    expect(rec.command.setpoint).toEqual({ vx: 8, vy: 0, vz: 0, yaw_rate: 0 });
    expect(rec.command.actuator).toEqual([0.5, 0.5, 0.5, 0.5]);

    // action（位姿源真值）
    expect(Object.keys(rec.action).sort()).toEqual([
      'ang_rate_e', 'battery', 'motor_rpm', 'pos_ecef', 'pos_enu', 'quat', 'vel_e',
    ]);
    expect(rec.action.pos_ecef).toEqual([-2165334.1, 4733056.5, 3575300.2]);
    expect(rec.action.pos_enu).toEqual([100, 50, 120]);
    expect(rec.action.quat).toEqual([0.7071, 0, 0, 0.7071]);
    expect(rec.action.vel_e).toEqual([8, 0, 0]);
    expect(rec.action.battery).toEqual({ v: 0, a: 0, soc: 1 });

    // environment / align / events
    expect(rec.environment.wind.gust_model).toBe('none');
    expect(rec.environment.temp_c).toBe(15);
    expect(rec.environment.pressure_pa).toBe(101325);
    expect(rec.align.t_capture_ns).toBe(12_500_000_000);
    expect(rec.align.t_command_ns).toBe(12_500_000_000);
    expect(rec.align.latency_model_ms).toEqual({ sensor: 0, control: 0, actuator: 0 });
    expect(rec.align.cam_T_body).toHaveLength(16);
    expect(rec.align.imu_T_body).toHaveLength(16);
    expect(rec.events).toEqual([]);
  });

  it('JSON 序列化约定：seed → 十进制字符串（u64 bigint 不可直接 JSON）', () => {
    const rec = buildFrameRecord(baseInput());
    const json = JSON.stringify(rec, (_k, v) => (typeof v === 'bigint' ? v.toString() : v));
    const parsed = JSON.parse(json) as { seed: string; frame_id: string };
    expect(parsed.seed).toBe('42');
    expect(parsed.frame_id).toBe('guanggu/ep001/000007');
  });
});
