import { describe, expect, it } from 'vitest';
import {
  BreakSpray,
  computeEmitDemand,
  BREAK_SPRAY_EMIT_CAP,
  BREAK_SPRAY_LIFE_S,
  BREAK_SPRAY_SIZE_M,
  BREAK_SPRAY_ALPHA,
} from '../../src/effect/BreakSpray';
import type { BreakSprayBand } from '../../src/effect/BreakSpray';
import type { OceanSeabedSprayBinding } from '../../src/effect/OceanFFTWater';
import {
  BREAK_SPRAY_FLOATS_PER_PARTICLE,
  BREAK_SPRAY_SIM_PARAMS_BYTES,
  BREAK_SPRAY_COMPUTE_SHADER,
} from '../../src/shaders/BreakSprayShader';
import type { Context } from '../../src/renderer/Context';

// node 环境无 WebGPU 全局枚举 —— bowSpray.spec.ts 同款补桩
const g = globalThis as unknown as Record<string, unknown>;
g.GPUBufferUsage ??= { STORAGE: 128, UNIFORM: 64, COPY_DST: 8, COPY_SRC: 4, VERTEX: 32 };
g.GPUTextureUsage ??= { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 };
g.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };

/**
 * 岸线破碎波喷雾（T4）— 记录型 mock（无 GPU）：
 * CPU 侧发射预算、常数规格、管线创建参数、环形复用、零 band 零发射、
 * 吸附源 fallback / 真实视图身份切换。
 */

/** 合成破碎带：赤道一点，沿 +z 走向 420 m 宽。 */
function bandState(intensity = 1): BreakSprayBand {
  return {
    centerEcef: { x: 1, y: 0, z: 0 },
    alongDir: { x: 0, y: 0, z: 1 },
    widthM: 420,
    nearDepth: 3,
    farDepth: 8,
    intensity,
  };
}

function seabedSource(tag = 'sb'): OceanSeabedSprayBinding {
  return {
    view: { textureLabel: `${tag}.seabed` } as unknown as GPUTextureView,
    dispView: { textureLabel: `${tag}.disp` } as unknown as GPUTextureView,
    centerEcef: { x: 1, y: 0, z: 0 },
    eastDir: { x: 1, y: 0, z: 0 },
    northDir: { x: 0, y: 0, z: 1 },
    uvOffset: { x: 0.5, y: 0.5 },
    uvScale: { x: 0.002, y: 0.002 },
  };
}

interface Harness {
  spray: BreakSpray;
  buffers: Array<{ label: string; size: number }>;
  renderPipelines: Array<Record<string, unknown>>;
  renderPasses: Array<Record<string, unknown>>;
  computeBindGroups: Array<{ entries: { binding: number; resource: unknown }[] }>;
  uniforms: Map<string, Float32Array>;
  drawArgs: number[];
  encoder: GPUCommandEncoder;
  target: GPUTextureView;
}

function makeHarness(maxParticles?: number, seabed?: () => OceanSeabedSprayBinding | null): Harness {
  const buffers: Array<{ label: string; size: number }> = [];
  const renderPipelines: Array<Record<string, unknown>> = [];
  const renderPasses: Array<Record<string, unknown>> = [];
  const computeBindGroups: Array<{ entries: { binding: number; resource: unknown }[] }> = [];
  const uniforms = new Map<string, Float32Array>();
  let drawArgs: number[] = [];

  const device = {
    createBuffer: (d: { label: string; size: number }) => {
      buffers.push({ label: d.label, size: d.size });
      return { label: d.label, destroy() { /* mock */ } };
    },
    createTexture: (d: { label: string }) => ({
      label: d.label,
      createView: (opts?: { label?: string; dimension?: string }) => ({
        label: opts?.label ?? `${d.label}.View`,
        textureLabel: d.label,
        dimension: opts?.dimension ?? '2d',
      }),
      destroy() { /* mock */ },
    }),
    createSampler: () => ({}),
    createShaderModule: () => ({}),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createComputePipeline: (d: { label: string }) => ({ label: d.label }),
    createRenderPipeline: (d: Record<string, unknown>) => {
      renderPipelines.push(d);
      return { label: d.label };
    },
    createBindGroup: (d: { entries: { binding: number; resource: unknown }[] }) => {
      const bg = { entries: d.entries };
      // binding 2/3（吸附纹理）仅出现在 compute bind group
      if (d.entries.some((e) => e.binding === 2)) computeBindGroups.push(bg);
      return bg;
    },
    queue: { writeTexture: () => { /* mock */ } },
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

  const spray = new BreakSpray(context as unknown as Context, {
    scene: scene as never,
    ...(seabed ? { seabed } : {}),
    ...(maxParticles !== undefined ? { maxParticles } : {}),
  });
  return { spray, buffers, renderPipelines, renderPasses, computeBindGroups, uniforms, drawArgs, encoder, target };
}

describe('computeEmitDemand 强度预算（CPU 侧）', () => {
  it('intensity=0 → 零发射', () => {
    expect(computeEmitDemand(0)).toBe(0);
  });

  it('intensity=0.5 → 50；1 → 100（cap 触及）', () => {
    expect(computeEmitDemand(0.5)).toBe(50);
    expect(computeEmitDemand(1)).toBe(BREAK_SPRAY_EMIT_CAP);
  });

  it('超 1 强度钳制到 cap 100；负值/非有限安全归 0', () => {
    expect(computeEmitDemand(2)).toBe(BREAK_SPRAY_EMIT_CAP);
    expect(computeEmitDemand(-0.5)).toBe(0);
    expect(computeEmitDemand(NaN)).toBe(0);
    expect(computeEmitDemand(Infinity)).toBe(0); // 非有限输入按无效处理（BowSpray 同语义）
  });
});

describe('setBand 未调用 / clearBand 时 execute', () => {
  it('零 band：不崩且零发射（uniform emit 槽 = 0）', () => {
    const h = makeHarness();
    expect(() => h.spray.execute(h.encoder, h.target, 1.0)).not.toThrow();
    const sim = h.uniforms.get('BreakSpray.SimParams');
    expect(sim).toBeDefined();
    expect(sim![11]).toBe(0); // up.w = emitCount
  });

  it('setBand(intensity=1) 后同帧发射 = cap 100', () => {
    const h = makeHarness();
    h.spray.setBand(bandState(1));
    h.spray.execute(h.encoder, h.target, 1.0);
    const sim = h.uniforms.get('BreakSpray.SimParams')!;
    expect(sim[11]).toBe(computeEmitDemand(1));
    expect(sim[11]).toBe(100);
  });

  it('setBand(intensity=0.5) → 发射 50；intensity=0 → 0', () => {
    const h = makeHarness();
    h.spray.setBand(bandState(0.5));
    h.spray.execute(h.encoder, h.target, 1.0);
    expect(h.uniforms.get('BreakSpray.SimParams')![11]).toBe(50);

    const h2 = makeHarness();
    h2.spray.setBand(bandState(0));
    h2.spray.execute(h2.encoder, h2.target, 1.0);
    expect(h2.uniforms.get('BreakSpray.SimParams')![11]).toBe(0);
  });

  it('clearBand 后恢复零发射', () => {
    const h = makeHarness();
    h.spray.setBand(bandState(1));
    h.spray.execute(h.encoder, h.target, 1.0);
    expect(h.uniforms.get('BreakSpray.SimParams')![11]).toBeGreaterThan(0);
    h.spray.clearBand();
    h.spray.execute(h.encoder, h.target, 2.0);
    expect(h.uniforms.get('BreakSpray.SimParams')![11]).toBe(0);
  });

  it('setBand 非法输入抛错（NaN / 零宽 / far <= near）', () => {
    const h = makeHarness();
    expect(() => h.spray.setBand({ ...bandState(1), widthM: NaN })).toThrow(/finite/);
    expect(() => h.spray.setBand({ ...bandState(1), centerEcef: { x: Infinity, y: 0, z: 0 } })).toThrow(/finite/);
    expect(() => h.spray.setBand({ ...bandState(1), widthM: 0 })).toThrow(/widthM/);
    expect(() => h.spray.setBand({ ...bandState(1), nearDepth: 8, farDepth: 3 })).toThrow(/farDepth/);
    // 合法状态仍可通过
    expect(() => h.spray.setBand(bandState(1))).not.toThrow();
  });
});

describe('吸附源（seabed getter）', () => {
  it('未提供 getter：compute binding 2/3 挂占位资源（-1000 / DispFallback）', () => {
    const h = makeHarness();
    h.spray.execute(h.encoder, h.target, 1.0);
    expect(h.computeBindGroups.length).toBeGreaterThanOrEqual(1);
    const bg = h.computeBindGroups[h.computeBindGroups.length - 1];
    const e2 = bg.entries.find((e) => e.binding === 2) as { resource: { textureLabel: string } };
    const e3 = bg.entries.find((e) => e.binding === 3) as { resource: { textureLabel: string } };
    expect(e2.resource.textureLabel).toBe('BreakSpray.SeabedFallback');
    expect(e3.resource.textureLabel).toBe('BreakSpray.DispFallback');
  });

  it('提供 getter：换真实视图；身份稳定不重建，换源才重建', () => {
    // 身份稳定源：两帧同一视图对象 → compute bind group 不再重建
    const fixed = seabedSource('fixed');
    const h = makeHarness(undefined, () => fixed);
    h.spray.execute(h.encoder, h.target, 1.0);
    const n1 = h.computeBindGroups.length;
    expect((h.computeBindGroups[n1 - 1].entries.find((e) => e.binding === 2) as { resource: unknown }).resource)
      .toBe(fixed.view);
    h.spray.execute(h.encoder, h.target, 2.0);
    expect(h.computeBindGroups.length).toBe(n1);

    // 换新源（新身份）→ 再重建一次
    const fixedB = seabedSource('b');
    let src = fixed as OceanSeabedSprayBinding;
    const h2 = makeHarness(undefined, () => src);
    h2.spray.execute(h2.encoder, h2.target, 1.0);
    const m1 = h2.computeBindGroups.length;
    src = fixedB;
    h2.spray.execute(h2.encoder, h2.target, 2.0);
    expect(h2.computeBindGroups.length).toBe(m1 + 1);
  });

  it('getter 返回 null → 回占位资源', () => {
    const h = makeHarness(undefined, () => null);
    h.spray.execute(h.encoder, h.target, 1.0);
    const bg = h.computeBindGroups[h.computeBindGroups.length - 1];
    const e2 = bg.entries.find((e) => e.binding === 2) as { resource: { textureLabel: string } };
    expect(e2.resource.textureLabel).toBe('BreakSpray.SeabedFallback');
  });
});

describe('粒子常数与池规格', () => {
  it('寿命 0.5-1.1s、尺寸 4-14cm、alpha 0.5', () => {
    expect(BREAK_SPRAY_LIFE_S[0]).toBeGreaterThanOrEqual(0.4);
    expect(BREAK_SPRAY_LIFE_S[1]).toBeLessThanOrEqual(1.2);
    expect(BREAK_SPRAY_SIZE_M[0]).toBeGreaterThanOrEqual(0.02);
    expect(BREAK_SPRAY_SIZE_M[1]).toBeLessThanOrEqual(0.2);
    expect(BREAK_SPRAY_ALPHA).toBeCloseTo(0.5, 5);
  });

  it('速度门/吸附 WGSL 源串锁定（smoothstep(far, near, depth) 概率薄化）', () => {
    expect(BREAK_SPRAY_COMPUTE_SHADER).toContain('smoothstep(params.bandp.y, params.bandp.x, depth)');
    expect(BREAK_SPRAY_COMPUTE_SHADER).toContain('textureLoad(seabedTex, tc, 0).r');
    expect(BREAK_SPRAY_COMPUTE_SHADER).toContain('textureLoad(dispTex, dtc, 0, 0).y');
  });

  it('默认池 2048、buffer 尺寸 = 2048 × 64B、SimParams 144B', () => {
    const h = makeHarness();
    expect(h.spray.maxParticles).toBe(2048);
    expect(h.buffers.find((b) => b.label === 'BreakSpray.States')!.size)
      .toBe(2048 * BREAK_SPRAY_FLOATS_PER_PARTICLE * 4);
    expect(h.buffers.find((b) => b.label === 'BreakSpray.SimParams')!.size)
      .toBe(BREAK_SPRAY_SIM_PARAMS_BYTES);
  });

  it('maxParticles=100 向上取整到 workgroup 倍数 128', () => {
    const h = makeHarness(100);
    expect(h.spray.maxParticles).toBe(128);
  });

  it('maxParticles=50 低于单帧发射上限 100 → 构造抛错', () => {
    expect(() => makeHarness(50)).toThrow(/maxParticles/);
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
    expect(target0.blend.color.srcFactor).toBe('one');
    expect(target0.blend.color.dstFactor).toBe('one-minus-src-alpha');
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
  it('长跑游标恒 mod maxParticles，总发射可超池容量', () => {
    const h = makeHarness();
    h.spray.setBand(bandState(1));
    const perFrame = computeEmitDemand(1); // 100
    let t = 0;
    for (let i = 0; i < 30; i++) {   // 30 × 100 = 3000 > 2048 → 必然回绕
      t += 1 / 60;
      h.spray.execute(h.encoder, h.target, t);
    }
    const total = perFrame * 30;
    expect(total).toBeGreaterThan(h.spray.maxParticles);
    expect(h.spray.ringCursor).toBe(total % h.spray.maxParticles);
    expect(h.spray.ringCursor).toBeGreaterThanOrEqual(0);
    expect(h.spray.ringCursor).toBeLessThan(h.spray.maxParticles);
  });

  it('uniform 里的 cursor 槽始终落在 [0, maxParticles) 的整数', () => {
    const h = makeHarness();
    h.spray.setBand(bandState(1));
    let t = 0;
    for (let i = 0; i < 40; i++) {
      t += 1 / 60;
      h.spray.execute(h.encoder, h.target, t);
      const cursor = h.uniforms.get('BreakSpray.SimParams')![15];
      expect(cursor).toBeGreaterThanOrEqual(0);
      expect(cursor).toBeLessThan(h.spray.maxParticles);
      expect(Number.isInteger(cursor)).toBe(true);
    }
  });
});
