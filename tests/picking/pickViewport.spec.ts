import { describe, expect, it } from 'vitest';
import { PickPass } from '../../src/picking/PickPass';
import type { PickPassDeps } from '../../src/picking/PickPass';
import { PrimitiveCollection } from '../../src/scene/Primitive';
import type { Primitive } from '../../src/scene/Primitive';

// node 环境无 WebGPU 全局枚举 —— PickPass 构造/编码路径引用它们
// （先例：其他 spec 直接绕开 GPU 路径；本测试必须走 encode，故补桩）
const g = globalThis as unknown as Record<string, unknown>;
g.GPUBufferUsage ??= { UNIFORM: 1, COPY_DST: 2, MAP_READ: 4 };
g.GPUTextureUsage ??= { RENDER_ATTACHMENT: 4, COPY_SRC: 8 };
g.GPUMapMode ??= { READ: 1 };
g.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };

/**
 * PickPass 采样策略回归测试。
 *
 * 背景 bug（2026-09-21 GPU 实测定案）：生产路径曾用
 * `setViewport(x, y, 1, 1)` 光栅化单个 texel。1×1 viewport 把整个 NDC
 * 平面 [-1,1]² 压缩进那 1 个 texel —— 任何与视锥相交的图元（无论投影
 * 在屏幕何处）都变换到该 texel 内光栅化，深度竞争后恒返回同一图元的
 * pickId（含 GLB Model 的场景恒返回模型 id，点击太空也如此）。
 *
 * 正确语义：全尺寸 viewport 保持 NDC→framebuffer 映射（图元只光栅化
 * 到自己的屏幕位置），1×1 scissor 把片段写入限制到被 copyTextureToBuffer
 * 读取的那个 texel。
 *
 * node 环境无 WebGPU：记录型 mock encoder 断言 encode 序列（先例：
 * globeLighting.spec.ts / globePickSet.spec.ts）。
 */

interface EncodeCalls {
  setViewport: number[][];
  setScissorRect: number[][];
  drawn: string[];
  copies: number;
}

function mockPrimitive(label: string, pickId: number, show = true): Primitive {
  return {
    label,
    show,
    pickId,
    translucent: false,
    drawOpaque(pass: GPURenderPassEncoder) {
      (pass as unknown as { draw(l: string): void }).draw(label);
    },
  } as unknown as Primitive;
}

function makeHarness(opts: { quadtree?: boolean } = {}) {
  const calls: EncodeCalls = { setViewport: [], setScissorRect: [], drawn: [], copies: 0 };
  const pass = {
    setPipeline() {},
    setViewport(...args: number[]) { calls.setViewport.push(args); },
    setScissorRect(...args: number[]) { calls.setScissorRect.push(args); },
    setBindGroup() {},
    setVertexBuffer() {},
    setIndexBuffer() {},
    drawIndexed() {},
    draw(label: string) { calls.drawn.push(label); },
    end() {},
  };
  const encoder = {
    beginRenderPass() { return pass; },
    copyTextureToBuffer() { calls.copies += 1; },
    finish() { return {}; },
  };
  const device = {
    createBuffer: () => ({ destroy() {} }),
    createShaderModule: () => ({}),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createRenderPipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup: () => ({}),
    createTexture: () => ({ createView: () => ({}), destroy() {} }),
    addEventListener() {},
    removeEventListener() {},
    queue: { writeBuffer() {}, writeTexture() {} },
  };
  const context = {
    device,
    canvas: { clientWidth: 100, clientHeight: 100 },
    drawingBufferWidth: 100,
    drawingBufferHeight: 100,
    writeBuffer() {},
    submit() {},
    createSampler: () => ({}),
  };
  const mat = () => ({ values: new Float32Array(16) });
  const camera = {
    position: { x: 0, y: 0, z: 2 },
    viewMatrix: mat(),
    projectionMatrix: mat(),
    viewProjectionMatrix: mat(),
    frustum: { near: 0.1, far: 10 },
  };
  const base = mockPrimitive('base-sphere', 1);
  const tile = mockPrimitive('tile-a', 2);
  const coll = new PrimitiveCollection();
  const deps: PickPassDeps = {
    context: context as never,
    camera: camera as never,
    primitives: coll,
    globe: {
      show: true,
      primitive: base,
      tilePrimitives: [tile],
      tilePrimitiveCount: 1,
      surfaceIsQuadtree: opts.quadtree ?? true,
    },
    getOpaquePipeline: () => ({ getBindGroupLayout: () => ({}) }) as never,
    getWorldPosTexture: () => null,
    frustumCullingEnabled: () => false,
    addFrameHook() {},
    removeFrameHook() {},
  };
  const pp = new PickPass(deps);
  let jobError: unknown = null;
  const job = {
    x: 10, y: 20, camFp: 1, sceneFp: 2, encoded: false,
    tEncodeStart: 0, tEncodeEnd: 0,
    promise: Promise.resolve({} as never),
    resolve: () => {},
    reject: (e: unknown) => { jobError = e; },
  };
  return { pp, calls, encoder, job, deps, coll, jobError: () => jobError };
}

describe('PickPass 采样策略：全视口 + 1×1 scissor（禁止 1×1 viewport）', () => {
  it('生产路径不调用 setViewport（1×1 viewport 会把全 NDC 压进一个 texel）', () => {
    const { pp, calls, encoder, job } = makeHarness();
    (pp as unknown as { _pending: unknown })._pending = job;
    (pp as unknown as { _encodePickFrame(e: unknown): void })._encodePickFrame(encoder);
    expect(calls.setViewport).toEqual([]);
  });

  it('1×1 scissor 精确限制在被回读的 texel (job.x, job.y)', () => {
    const { pp, calls, encoder, job, jobError } = makeHarness();
    (pp as unknown as { _pending: unknown })._pending = job;
    (pp as unknown as { _encodePickFrame(e: unknown): void })._encodePickFrame(encoder);
    expect(jobError(), 'encode 不应抛错: ' + String(jobError())).toBeNull();
    expect(calls.setScissorRect).toEqual([[10, 20, 1, 1]]);
  });

  it('debugFullViewport 模式不加 scissor（整平面导出诊断用）', () => {
    const { pp, calls, encoder, job } = makeHarness();
    pp.debugFullViewport = true;
    (pp as unknown as { _pending: unknown })._pending = job;
    (pp as unknown as { _encodePickFrame(e: unknown): void })._encodePickFrame(encoder);
    expect(calls.setScissorRect).toEqual([]);
    expect(calls.setViewport).toEqual([]);
  });

  it('quadtree 模式跳过底球（与 G-Buffer 可见集合同源），瓦片仍可拾取', () => {
    const { pp, calls, encoder, job } = makeHarness({ quadtree: true });
    (pp as unknown as { _pending: unknown })._pending = job;
    (pp as unknown as { _encodePickFrame(e: unknown): void })._encodePickFrame(encoder);
    expect(calls.drawn).toContain('tile-a');
    expect(calls.drawn).not.toContain('base-sphere');
  });

  it('legacy 单纹理模式（surfaceIsQuadtree=false）底球仍可拾取', () => {
    const { pp, calls, encoder, job } = makeHarness({ quadtree: false });
    (pp as unknown as { _pending: unknown })._pending = job;
    (pp as unknown as { _encodePickFrame(e: unknown): void })._encodePickFrame(encoder);
    expect(calls.drawn).toContain('base-sphere');
    expect(calls.drawn).toContain('tile-a');
  });
});

describe('PickPass sceneFingerprint 包含可见性状态（防 show 切换假缓存）', () => {
  it('图元 show 切换改变 fingerprint（同 x/y 不再命中缓存）', () => {
    const { pp, coll } = makeHarness();
    const prim = mockPrimitive('p', 9);
    (coll as unknown as { _primitives: Primitive[] })._primitives.push(prim);

    const fp = (p: PickPass) =>
      (p as unknown as { _sceneFingerprint(): number })._sceneFingerprint();

    const shown = fp(pp);
    prim.show = false;
    const hidden = fp(pp);
    expect(shown).not.toBe(hidden);

    prim.show = true;
    expect(fp(pp)).toBe(shown);
  });

  it('globe.show 切换改变 fingerprint', () => {
    const { pp, deps } = makeHarness();
    const fp = (p: PickPass) =>
      (p as unknown as { _sceneFingerprint(): number })._sceneFingerprint();
    const on = fp(pp);
    (deps.globe as { show: boolean }).show = false;
    const off = fp(pp);
    expect(on).not.toBe(off);
  });
});
