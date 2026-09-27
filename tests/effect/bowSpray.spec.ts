import { describe, expect, it } from 'vitest';
import {
  BowSpray,
  computeEmitDemand,
  sprayGate,
  DROPLET_LIFE_S,
  DROPLET_SIZE_M,
  DROPLET_ALPHA,
  MIST_LIFE_S,
  MIST_SIZE_M,
  MIST_ALPHA,
  MIST_RATIO,
  SPRAY_GATE_LO,
  SPRAY_GATE_HI,
  SPRAY_EMIT_CAP,
} from '../../src/effect/BowSpray';
import {
  BOW_SPRAY_FLOATS_PER_PARTICLE,
  BOW_SPRAY_COMPUTE_SHADER,
} from '../../src/shaders/BowSprayShader';
import type { Context } from '../../src/renderer/Context';
import type { BowSprayBoatState } from '../../src/effect/BowSpray';

// node 环境无 WebGPU 全局枚举 —— pickInstance.spec.ts / cssm.spec.ts 同款补桩
const g = globalThis as unknown as Record<string, unknown>;
g.GPUBufferUsage ??= { STORAGE: 128, UNIFORM: 64, COPY_DST: 8, COPY_SRC: 4, VERTEX: 32 };
g.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };

/**
 * 海洋 L4 船首浪喷雾 — 记录型 mock（无 GPU）：
 * CPU 侧速度门/发射预算、常数规格范围、管线创建参数、环形复用。
 */

/** 合成船首状态：赤道一点向东航行。 */
function boatState(speed: number, beamM = 8): BowSprayBoatState {
  const r = 1 / 6378137; // 1 m 弧长对应的归一化角度
  return {
    bowPos: { x: 1, y: 0, z: 0 },
    headingDir: { x: 0, y: 0, z: 1 },
    sideDir: { x: -r * beamM * 0.5, y: 0, z: 0 }, // 近似东向航行的右舷（-x 切向）
    speed,
    beamM,
  };
}

interface Harness {
  spray: BowSpray;
  buffers: Array<{ label: string; size: number }>;
  renderPipelines: Array<Record<string, unknown>>;
  renderPasses: Array<Record<string, unknown>>;
  uniforms: Map<string, Float32Array>;
  drawArgs: number[];
  encoder: GPUCommandEncoder;
  target: GPUTextureView;
}

function makeHarness(maxParticles?: number): Harness {
  const buffers: Array<{ label: string; size: number }> = [];
  const renderPipelines: Array<Record<string, unknown>> = [];
  const renderPasses: Array<Record<string, unknown>> = [];
  const uniforms = new Map<string, Float32Array>();
  let drawArgs: number[] = [];

  const device = {
    createBuffer: (d: { label: string; size: number }) => {
      buffers.push({ label: d.label, size: d.size });
      return { label: d.label, destroy() { /* mock */ } };
    },
    createShaderModule: () => ({}),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createComputePipeline: (d: { label: string }) => ({ label: d.label }),
    createRenderPipeline: (d: Record<string, unknown>) => {
      renderPipelines.push(d);
      return { label: d.label };
    },
    createBindGroup: () => ({}),
  };
  const context = {
    initialized: true,
    device,
    preferredFormat: 'bgra8unorm',
    createShaderModule: () => ({}),
    writeBuffer: (buffer: { label: string }, _offset: number, data: ArrayBufferLike) => {
      uniforms.set(buffer.label, new Float32Array(data.slice(0) as ArrayBuffer));
    },
  };
  const scene = {
    camera: {
      viewProjectionMatrix: { values: new Float32Array(16) },
      frustum: { fov: Math.PI / 3, aspectRatio: 1 },
    },
    gbufferPass: { getDepthView: () => ({}) },
  };
  const cpass = {
    setPipeline() { /* mock */ },
    setBindGroup() { /* mock */ },
    dispatchWorkgroups() { /* mock */ },
    end() { /* mock */ },
  };
  const rpass = {
    setPipeline() { /* mock */ },
    setVertexBuffer() { /* mock */ },
    setBindGroup() { /* mock */ },
    draw(...args: number[]) { drawArgs.push(...args); },
    end() { /* mock */ },
  };
  const encoder = {
    beginComputePass: () => cpass,
    beginRenderPass: (d: Record<string, unknown>) => {
      renderPasses.push(d);
      return rpass;
    },
  } as unknown as GPUCommandEncoder;
  const target = {} as GPUTextureView;

  const spray = new BowSpray(context as unknown as Context, {
    scene: scene as never,
    ...(maxParticles !== undefined ? { maxParticles } : {}),
  });
  return { spray, buffers, renderPipelines, renderPasses, uniforms, drawArgs, encoder, target };
}

describe('computeEmitDemand 速度门（CPU 侧发射预算）', () => {
  it('speed=2 低于速度门 → 零发射', () => {
    expect(sprayGate(2)).toBe(0);
    expect(computeEmitDemand(2)).toBe(0);
  });

  it('speed=6 过门 → 发射数 > 0', () => {
    expect(sprayGate(6)).toBeGreaterThan(0);
    expect(computeEmitDemand(6)).toBeGreaterThan(0);
  });

  it('边界：gate 在 3.5 处为 0、9 处为 1，单调不减', () => {
    expect(sprayGate(SPRAY_GATE_LO)).toBe(0);
    expect(sprayGate(SPRAY_GATE_HI)).toBe(1);
    let prev = -1;
    for (let s = 0; s <= 12; s += 0.25) {
      const v = sprayGate(s);
      expect(v).toBeGreaterThanOrEqual(prev);
      prev = v;
    }
  });

  it('负速/非有限输入安全，且发射数不超过 cap 100', () => {
    expect(computeEmitDemand(-3)).toBe(0);
    expect(computeEmitDemand(NaN)).toBe(0);
    expect(computeEmitDemand(Infinity)).toBe(0); // 非有限输入按无效处理（sprayGate 同语义）
    expect(computeEmitDemand(1000)).toBe(SPRAY_EMIT_CAP); // cap 触及
  });

  it('speed=6 的发射数等于 round(gate·6·1.6)', () => {
    const expected = Math.min(Math.round(sprayGate(6) * 6 * 1.6), SPRAY_EMIT_CAP);
    expect(computeEmitDemand(6)).toBe(expected);
    expect(expected).toBeGreaterThan(0);
  });
});

describe('setBoat 未调用时 execute', () => {
  it('不崩且零发射（uniform emit 槽 = 0）', () => {
    const h = makeHarness();
    expect(() => h.spray.execute(h.encoder, h.target, 1.0)).not.toThrow();
    const sim = h.uniforms.get('BowSpray.SimParams');
    expect(sim).toBeDefined();
    expect(sim![11]).toBe(0); // basisU.w = emitCount
  });

  it('setBoat(speed=6) 后同帧发射 > 0', () => {
    const h = makeHarness();
    h.spray.setBoat(boatState(6));
    h.spray.execute(h.encoder, h.target, 1.0);
    const sim = h.uniforms.get('BowSpray.SimParams')!;
    expect(sim[11]).toBe(computeEmitDemand(6));
    expect(sim[11]).toBeGreaterThan(0);
  });

  it('speed=2 的船也零发射（CPU 预算已为 0）', () => {
    const h = makeHarness();
    h.spray.setBoat(boatState(2));
    h.spray.execute(h.encoder, h.target, 1.0);
    expect(h.uniforms.get('BowSpray.SimParams')![11]).toBe(0);
  });

  it('clearBoat 后恢复零发射（emit 槽归 0，computeEmitDemand 本身不受影响）', () => {
    const h = makeHarness();
    h.spray.setBoat(boatState(6));
    h.spray.execute(h.encoder, h.target, 1.0);
    expect(h.uniforms.get('BowSpray.SimParams')![11]).toBeGreaterThan(0);
    h.spray.clearBoat();
    expect(computeEmitDemand(6)).toBeGreaterThan(0); // 纯函数仍按速度算预算
    h.spray.execute(h.encoder, h.target, 2.0);
    expect(h.uniforms.get('BowSpray.SimParams')![11]).toBe(0);
  });

  it('setBoat 传 NaN/Infinity 抛错（全字段有限校验，对齐 WakeField）', () => {
    const h = makeHarness();
    expect(() => h.spray.setBoat({ ...boatState(6), speed: NaN })).toThrow(/finite/);
    expect(() => h.spray.setBoat({ ...boatState(6), bowPos: { x: NaN, y: 0, z: 0 } })).toThrow(/finite/);
    expect(() => h.spray.setBoat({ ...boatState(6), sideDir: { x: 0, y: 0, z: Infinity } })).toThrow(/finite/);
    expect(() => h.spray.setBoat({ ...boatState(6), beamM: NaN })).toThrow(/finite/);
    // 合法状态仍可通过
    expect(() => h.spray.setBoat(boatState(6))).not.toThrow();
  });
});

describe('两种粒子常数（规格范围）', () => {
  it('90% droplet / 10% mist 比例', () => {
    expect(MIST_RATIO).toBeGreaterThan(0.05);
    expect(MIST_RATIO).toBeLessThan(0.15);
    expect(MIST_RATIO).toBe(0.1);
  });

  it('droplet：寿命 0.9±0.4 s、尺寸 6±3 cm、alpha 0.55', () => {
    expect(DROPLET_LIFE_S[0]).toBeGreaterThanOrEqual(0.5);
    expect(DROPLET_LIFE_S[1]).toBeLessThanOrEqual(1.3);
    expect(DROPLET_SIZE_M[0] * 100).toBeGreaterThanOrEqual(3);   // cm
    expect(DROPLET_SIZE_M[1] * 100).toBeLessThanOrEqual(9);
    expect(DROPLET_ALPHA).toBeCloseTo(0.55, 5);
  });

  it('mist：寿命 1.6±0.5 s、尺寸 0.35±0.15 m、alpha 0.06', () => {
    expect(MIST_LIFE_S[0]).toBeGreaterThanOrEqual(1.1);
    expect(MIST_LIFE_S[1]).toBeLessThanOrEqual(2.1);
    expect(MIST_SIZE_M[0]).toBeGreaterThanOrEqual(0.2);
    expect(MIST_SIZE_M[1]).toBeLessThanOrEqual(0.5);
    expect(MIST_ALPHA).toBeCloseTo(0.06, 5);
  });

  it('mist 物理倍率：初速 ×0.5、重力 ×0.3（真源在 WGSL 常数，锁源串）', () => {
    expect(BOW_SPRAY_COMPUTE_SHADER).toMatch(/const\s+MIST_VEL_SCALE\s*=\s*0\.5\s*;/);
    expect(BOW_SPRAY_COMPUTE_SHADER).toMatch(/const\s+MIST_GRAVITY_SCALE\s*=\s*0\.3\s*;/);
  });
});

describe('管线创建参数（mock device 断言）', () => {
  it('blend premultiplied + depthCompare less + 只读深度', () => {
    const h = makeHarness();
    expect(h.renderPipelines).toHaveLength(1);
    const rp = h.renderPipelines[0] as {
      fragment: { targets: Array<{ format: string; blend: {
        color: { srcFactor: string; dstFactor: string; operation: string };
        alpha: { srcFactor: string; dstFactor: string; operation: string };
      } }> };
      primitive: { topology: string };
      depthStencil: { format: string; depthWriteEnabled: boolean; depthCompare: string };
    };
    const target0 = rp.fragment.targets[0];
    expect(target0.format).toBe('bgra8unorm');
    // premultiplied alpha：src = one
    expect(target0.blend.color.srcFactor).toBe('one');
    expect(target0.blend.color.dstFactor).toBe('one-minus-src-alpha');
    expect(target0.blend.color.operation).toBe('add');
    expect(target0.blend.alpha.srcFactor).toBe('one');
    // G-Buffer 深度测试：less + 不写深度
    expect(rp.primitive.topology).toBe('triangle-strip');
    expect(rp.depthStencil.format).toBe('depth32float');
    expect(rp.depthStencil.depthWriteEnabled).toBe(false);
    expect(rp.depthStencil.depthCompare).toBe('less');
  });

  it('execute 的 render pass 用只读 G-Buffer 深度 + loadOp load', () => {
    const h = makeHarness();
    h.spray.execute(h.encoder, h.target, 1.0);
    const desc = h.renderPasses[0] as {
      colorAttachments: Array<{ loadOp: string }>;
      depthStencilAttachment: { depthReadOnly: boolean };
    };
    expect(desc.colorAttachments[0].loadOp).toBe('load');
    expect(desc.depthStencilAttachment.depthReadOnly).toBe(true);
  });

  it('draw(4, maxParticles) 全池 instanced billboard', () => {
    const h = makeHarness();
    h.spray.execute(h.encoder, h.target, 1.0);
    expect(h.drawArgs).toEqual([4, 2048]);
  });
});

describe('环形复用（ring buffer）', () => {
  it('默认池 2048、buffer 尺寸 = 2048 × 64 B', () => {
    const h = makeHarness();
    expect(h.spray.maxParticles).toBe(2048);
    const states = h.buffers.find((b) => b.label === 'BowSpray.States');
    expect(states!.size).toBe(2048 * BOW_SPRAY_FLOATS_PER_PARTICLE * 4);
  });

  it('maxParticles=100 向上取整到 workgroup 倍数 128', () => {
    const h = makeHarness(100);
    expect(h.spray.maxParticles).toBe(128);
  });

  it('maxParticles=50 低于单帧发射上限 100 → 构造抛错', () => {
    expect(() => makeHarness(50)).toThrow(/maxParticles/);
  });

  it('长跑游标恒 mod maxParticles，总发射可超池容量', () => {
    const h = makeHarness();
    h.spray.setBoat(boatState(12));
    const perFrame = computeEmitDemand(12); // 19
    expect(perFrame).toBeGreaterThan(0);
    let t = 0;
    for (let i = 0; i < 120; i++) {   // 120 × 19 = 2280 > 2048 → 必然回绕
      t += 1 / 60;
      h.spray.execute(h.encoder, h.target, t);
    }
    const total = perFrame * 120;
    expect(total).toBeGreaterThan(h.spray.maxParticles);
    expect(h.spray.ringCursor).toBe(total % h.spray.maxParticles);
    expect(h.spray.ringCursor).toBeGreaterThanOrEqual(0);
    expect(h.spray.ringCursor).toBeLessThan(h.spray.maxParticles);
  });

  it('uniform 里的 cursor 槽始终落在 [0, maxParticles) 的整数', () => {
    const h = makeHarness();
    h.spray.setBoat(boatState(12));
    let t = 0;
    for (let i = 0; i < 40; i++) {
      t += 1 / 60;
      h.spray.execute(h.encoder, h.target, t);
      const cursor = h.uniforms.get('BowSpray.SimParams')![15];
      expect(cursor).toBeGreaterThanOrEqual(0);
      expect(cursor).toBeLessThan(h.spray.maxParticles);
      expect(Number.isInteger(cursor)).toBe(true);
    }
  });
});
