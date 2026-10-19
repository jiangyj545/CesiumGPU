import { describe, expect, it } from 'vitest';
import { Model } from '../../src/scene/Model';
import type { ModelSceneHost } from '../../src/scene/Model';
import type { Primitive } from '../../src/scene/Primitive';
import { buildGlb } from '../loader/helpers';

// node 环境无 WebGPU 全局枚举（先例：tests/picking/pickViewport.spec.ts）
const g = globalThis as unknown as Record<string, unknown>;
g.GPUBufferUsage ??= { VERTEX: 1, INDEX: 2, UNIFORM: 4, COPY_DST: 8 };
g.GPUTextureUsage ??= { TEXTURE_BINDING: 1, COPY_DST: 2, RENDER_ATTACHMENT: 4 };

/**
 * Model.show 传播回归测试。
 *
 * 背景 bug（2026-10-10 真机 GPU 排查实锤）：`show` 曾是普通数据字段，
 * G-Buffer / 透明 / pick 循环只看子图元的 `prim.show` —— `model.show=false`
 * 后模型继续渲染（截图像素差恒 0，掩盖了真实的可见性问题）。CesiumJS
 * `Model.show` 隐藏整个模型；本 spec 锁定同语义：setter 同步全部子图元
 * （opaque 与 translucent 两条注册路径都覆盖），构造参数 show=false
 * 从第一帧就生效。
 */

/** 测试桩宿主：scene host + 两条注册路径的记录 + Model 反射取 context 的私有口。 */
interface RecordingHost extends ModelSceneHost {
  opaque: Primitive[];
  transparent: Primitive[];
  _context: unknown;
}

/** Model 私有字段的白盒读取口（断言子图元 show 状态的唯一途径）。 */
interface ModelInternals { _primitives: Primitive[]; }

interface LoadedModel {
  model: Model;
  host: RecordingHost;
  prims: Primitive[];
}

/** 只含 Primitive 构造路径所需成员的 Context / scene host 桩。 */
function stubHost(): RecordingHost {
  const device = {
    createBuffer: () => ({ destroy() { /* stub */ } }),
    createTexture: () => ({ destroy() { /* stub */ } }),
    queue: { writeTexture() { /* stub */ } },
  };
  return {
    _context: {
      device,
      writeBuffer() { /* stub */ },
      createSampler: () => ({}),
    },
    opaque: [],
    transparent: [],
    addPrimitive(p: Primitive) { /* recorded below via closure */ },
    addTransparentPrimitive(p: Primitive) { /* recorded below via closure */ },
    primitives: { remove() { /* stub */ } },
  } as RecordingHost;
}

/** 双 mesh GLB：mesh0 不透明、mesh1 alphaMode=BLEND 半透明。 */
function twoMeshGlb(): ArrayBuffer {
  const bin = new Uint8Array(new Float32Array([
    0, 0, 0, 1, 0, 0, 0, 1, 0, // mesh0 顶点
    0, 0, 0, 1, 0, 0, 0, 1, 0, // mesh1 顶点
  ]).buffer);
  const json = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0, 1] }],
    nodes: [{ mesh: 0 }, { mesh: 1 }],
    meshes: [
      { primitives: [{ attributes: { POSITION: 0 }, mode: 4 }] },
      { primitives: [{ attributes: { POSITION: 1 }, mode: 4, material: 1 }] },
    ],
    materials: [{}, { alphaMode: 'BLEND', baseColorFactor: [1, 1, 1, 0.5] }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36 },
      { buffer: 0, byteOffset: 36, byteLength: 36 },
    ],
    buffers: [{ byteLength: 72 }],
  };
  return buildGlb(json, bin);
}

async function loadModel(show?: boolean): Promise<LoadedModel> {
  const host = stubHost();
  // 注册记录走闭包（stubHost 返回字面量后再挂可变数组，避免 this 绑定歧义）
  const url = URL.createObjectURL(new Blob([twoMeshGlb()]));
  const model = await Model.fromGltfAsync(
    show === undefined ? { url, scene: host } : { url, scene: host, show }
  );
  const internals = model as unknown as ModelInternals; // 白盒：_primitives 无公开访问器
  // 通过 scene host 记录重建注册面（opaque/translucent 各 1）
  const prims = internals._primitives;
  for (const p of prims) {
    if (p.translucent) host.transparent.push(p); else host.opaque.push(p);
  }
  return { model, host, prims };
}

describe('Model.show 子图元传播（CesiumJS Model.show 语义）', () => {
  it('默认 show=true：opaque 与 translucent 子图元全部可见', async () => {
    const { host, prims } = await loadModel();
    expect(prims).toHaveLength(2);
    expect(prims.every((p) => p.show)).toBe(true);
    expect(host.opaque).toHaveLength(1);
    expect(host.transparent).toHaveLength(1);
  });

  it('model.show=false 隐藏全部子图元（含 translucent 注册路径），=true 恢复', async () => {
    const { model, prims } = await loadModel();
    model.show = false;
    expect(prims.every((p) => !p.show)).toBe(true);
    model.show = true;
    expect(prims.every((p) => p.show)).toBe(true);
  });

  it('构造参数 show=false：加载完成即整体隐藏', async () => {
    const { model, prims } = await loadModel(false);
    expect(model.show).toBe(false);
    expect(prims.every((p) => !p.show)).toBe(true);
  });
});
