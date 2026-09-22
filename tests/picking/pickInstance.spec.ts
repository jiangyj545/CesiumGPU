import { describe, expect, it } from 'vitest';
import { PickPass } from '../../src/picking/PickPass';
import type { PickPassDeps } from '../../src/picking/PickPass';
import { PrimitiveCollection } from '../../src/scene/Primitive';
import type { Primitive } from '../../src/scene/Primitive';

// node 环境无 WebGPU 全局枚举 —— 与 pickViewport.spec.ts 同款补桩
const g = globalThis as unknown as Record<string, unknown>;
g.GPUBufferUsage ??= { UNIFORM: 1, COPY_DST: 2, MAP_READ: 4, VERTEX: 32 };
g.GPUTextureUsage ??= { RENDER_ATTACHMENT: 4, COPY_SRC: 8 };
g.GPUMapMode ??= { READ: 1 };
g.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };

/**
 * F12 实例级拾取：per-vertex instance id 顶点缓冲的 pick 渲染路径。
 *
 * 契约（完全向后兼容）：
 *  - 携带 instanceIdBuffer 的图元 → 切实例管线 + setVertexBuffer(1, buf)；
 *  - 不携带的图元 → 原 pick 管线，从不触碰顶点槽 1（G-Buffer 管线只有
 *    一个顶点槽，setVertexBuffer(1) 会触发 WebGPU 校验错误——回归红线）。
 *
 * node 环境无 WebGPU：记录型 mock（先例：pickViewport.spec.ts）。
 */

interface PipelineCalls {
  /** pass.setPipeline 收到的管线 label 序列 */
  pipelines: string[];
  /** pass.setVertexBuffer(slot, bufferLabel) 序列 */
  vertexBuffers: Array<[number, string]>;
  drawn: string[];
}

function mockPrimitive(
  label: string,
  pickId: number,
  opts: { show?: boolean; instanceBufferLabel?: string } = {},
): Primitive {
  return {
    label,
    show: opts.show ?? true,
    pickId,
    translucent: false,
    hasInstanceIds: opts.instanceBufferLabel !== undefined,
    instanceIdBuffer: opts.instanceBufferLabel !== undefined
      ? { label: opts.instanceBufferLabel }
      : null,
    drawOpaque(pass: GPURenderPassEncoder) {
      (pass as unknown as { draw(l: string): void }).draw(label);
    },
  } as unknown as Primitive;
}

function makeHarness() {
  const calls: PipelineCalls = { pipelines: [], vertexBuffers: [], drawn: [] };
  const pass = {
    setPipeline(p: { label: string }) { calls.pipelines.push(p.label); },
    setViewport() {},
    setScissorRect() {},
    setBindGroup() {},
    setVertexBuffer(slot: number, buf: { label: string }) {
      calls.vertexBuffers.push([slot, buf.label]);
    },
    setIndexBuffer() {},
    drawIndexed() {},
    draw(label: string) { calls.drawn.push(label); },
    end() {},
  };
  const encoder = {
    beginRenderPass() { return pass; },
    copyTextureToBuffer() {},
    finish() { return {}; },
  };
  const device = {
    createBuffer: () => ({ destroy() {} }),
    createShaderModule: () => ({}),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    // 按调用序区分两条管线：第 1 条 = 基础 pick 管线，第 2 条 = 实例管线
    createRenderPipeline: (() => {
      let n = 0;
      return () => {
        n += 1;
        return { label: n === 1 ? 'pick-base' : 'pick-instance', getBindGroupLayout: () => ({}) };
      };
    })(),
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
  const deps: PickPassDeps = {
    context: context as never,
    camera: camera as never,
    primitives: new PrimitiveCollection(),
    globe: {
      show: true,
      primitive: null,
      tilePrimitives: [],
      tilePrimitiveCount: 0,
      surfaceIsQuadtree: true,
    },
    getOpaquePipeline: () => ({ getBindGroupLayout: () => ({}) }) as never,
    getWorldPosTexture: () => null,
    frustumCullingEnabled: () => false,
    addFrameHook() {},
    removeFrameHook() {},
  };
  const pp = new PickPass(deps);
  const job = {
    x: 10, y: 20, camFp: 1, sceneFp: 2, encoded: false,
    tEncodeStart: 0, tEncodeEnd: 0,
    promise: Promise.resolve({} as never),
    resolve: () => {},
    reject: () => {},
  };
  return { pp, calls, encoder, job, deps };
}

describe('PickPass F12 实例 id 渲染路径（per-vertex override）', () => {
  it('携带实例缓冲的图元走实例管线并绑定顶点槽 1', () => {
    const { pp, calls, encoder, job } = makeHarness();
    const instanced = mockPrimitive('city-chunk', 7, { instanceBufferLabel: 'InstanceIdVBO' });
    (pp as unknown as { _deps: PickPassDeps })._deps.primitives.add(instanced);
    (pp as unknown as { _pending: unknown })._pending = job;
    (pp as unknown as { _encodePickFrame(e: unknown): void })._encodePickFrame(encoder);

    expect(calls.drawn).toContain('city-chunk');
    expect(calls.pipelines).toContain('pick-instance');
    expect(calls.vertexBuffers).toContainEqual([1, 'InstanceIdVBO']);
  });

  it('不携带实例缓冲的图元维持基础管线，从不触碰顶点槽 1（向后兼容）', () => {
    const { pp, calls, encoder, job } = makeHarness();
    const plain = mockPrimitive('tile-a', 3);
    (pp as unknown as { _deps: PickPassDeps })._deps.primitives.add(plain);
    (pp as unknown as { _pending: unknown })._pending = job;
    (pp as unknown as { _encodePickFrame(e: unknown): void })._encodePickFrame(encoder);

    expect(calls.drawn).toContain('tile-a');
    expect(calls.pipelines).toContain('pick-base');
    expect(calls.pipelines).not.toContain('pick-instance');
    expect(calls.vertexBuffers).toEqual([]);
  });

  it('混排（实例 + 普通图元）各自走对应管线，互不污染', () => {
    const { pp, calls, encoder, job } = makeHarness();
    const coll = (pp as unknown as { _deps: PickPassDeps })._deps.primitives;
    coll.add(mockPrimitive('city-chunk', 7, { instanceBufferLabel: 'inst-1' }));
    coll.add(mockPrimitive('tile-a', 3));
    coll.add(mockPrimitive('city-chunk-2', 8, { instanceBufferLabel: 'inst-2' }));
    (pp as unknown as { _pending: unknown })._pending = job;
    (pp as unknown as { _encodePickFrame(e: unknown): void })._encodePickFrame(encoder);

    expect(calls.drawn).toEqual(['city-chunk', 'tile-a', 'city-chunk-2']);
    const instanced = calls.vertexBuffers.filter(([slot]) => slot === 1);
    expect(instanced).toEqual([[1, 'inst-1'], [1, 'inst-2']]);
    // 基础管线出现（普通图元）且实例管线出现（两个实例图元）
    expect(calls.pipelines).toContain('pick-base');
    expect(calls.pipelines.filter((l) => l === 'pick-instance')).toHaveLength(2);
  });

  it('hide 的实例图元不触发任何管线/顶点槽调用', () => {
    const { pp, calls, encoder, job } = makeHarness();
    const hidden = mockPrimitive('city-hidden', 9, { show: false, instanceBufferLabel: 'inst-x' });
    (pp as unknown as { _deps: PickPassDeps })._deps.primitives.add(hidden);
    (pp as unknown as { _pending: unknown })._pending = job;
    (pp as unknown as { _encodePickFrame(e: unknown): void })._encodePickFrame(encoder);

    expect(calls.drawn).toEqual([]);
    expect(calls.vertexBuffers).toEqual([]);
  });
});
