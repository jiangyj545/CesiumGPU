import { describe, expect, it } from 'vitest';
import { GltfLoader } from '../../src/loader/GltfLoader';
import { buildGlb } from './helpers';

interface PrimSpec { min: number[]; max: number[]; translation: number[] }

/** 构造带 accessor min/max 与节点平移的最小 GLB（每个 spec → 独立 mesh/node）。 */
function buildBoundsGlb(prims: PrimSpec[]): ArrayBuffer {
  const bin = new Uint8Array(new Float32Array([0, 0, 0]).buffer);
  const json = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: prims.map((_, i) => i) }],
    nodes: prims.map(p => ({ mesh: prims.indexOf(p), translation: p.translation })),
    meshes: prims.map((_, i) => ({ primitives: [{ attributes: { POSITION: i }, mode: 4 }] })),
    accessors: prims.map(p => ({
      bufferView: 0,
      componentType: 5126,
      count: 1,
      type: 'VEC3',
      min: p.min,
      max: p.max,
    })),
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 12 }],
    buffers: [{ byteLength: 12 }],
  };
  return buildGlb(json, bin);
}

function load(prims: PrimSpec[]) {
  return GltfLoader.load(URL.createObjectURL(new Blob([buildBoundsGlb(prims)])));
}

describe('GltfLoader model bounds（POSITION accessor min/max → 模型空间 AABB）', () => {
  it('恒等节点下按 accessor min/max 收集', async () => {
    const model = await load([{ min: [-1, -1, -1], max: [1, 1, 1], translation: [0, 0, 0] }]);
    expect(model.bounds).toBeDefined();
    expect(model.bounds!.min).toEqual([-1, -1, -1]);
    expect(model.bounds!.max).toEqual([1, 1, 1]);
  });

  it('角点经节点平移后并入（平移 10）', async () => {
    const model = await load([{ min: [0, 0, 0], max: [2, 2, 2], translation: [10, 0, 0] }]);
    expect(model.bounds!.min).toEqual([10, 0, 0]);
    expect(model.bounds!.max).toEqual([12, 2, 2]);
  });

  it('多 mesh/node 取并集', async () => {
    const model = await load([
      { min: [-1, -1, -1], max: [0, 0, 0], translation: [0, 0, 0] },
      { min: [5, 5, 5], max: [6, 6, 6], translation: [0, 0, 0] },
    ]);
    expect(model.bounds!.min).toEqual([-1, -1, -1]);
    expect(model.bounds!.max).toEqual([6, 6, 6]);
  });

  it('accessor 无 min/max 时 bounds 不存在（诚实缺省）', async () => {
    const bin = new Uint8Array(new Float32Array([0, 0, 0]).buffer);
    const json = {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, mode: 4 }] }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 1, type: 'VEC3' }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 12 }],
      buffers: [{ byteLength: 12 }],
    };
    const model = await GltfLoader.load(URL.createObjectURL(new Blob([buildGlb(json, bin)])));
    expect(model.bounds).toBeUndefined();
  });
});
