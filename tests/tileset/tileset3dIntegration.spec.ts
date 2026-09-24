/**
 * tileset3dIntegration.spec.ts – Tileset3D 调度状态机集成测试。
 *
 * 桩策略（先例：tests/picking/pickViewport.spec.ts 的记录型假 Primitive）：
 * - scene = 记录型桩（真实 PrimitiveCollection 收集 add/remove/destroy）；
 * - content 加载经 `_contentLoader` 测试缝注入：返回手动 resolve 的
 *   Promise（精确控制「L0 已上屏 / L1 在途 / L1 就绪」时序）；
 * - tileset.json 经 global.fetch 桩供给（合成树，无网络）；
 * - 相机用真 `Camera`（纯数学，无 GPU）俯视原点瓦片球。
 *
 * 覆盖：L0 先上屏 → L1 就绪后 REPLACE 父隐藏；LRU 驱逐 → 下一帧 remove；
 * 并发上限 8；destroy 清场（含在途请求不自注册）。
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Tileset3D } from '../../src/tileset/Tileset3D';
import { PrimitiveCollection } from '../../src/scene/Primitive';
import type { Primitive } from '../../src/scene/Primitive';
import type { Scene } from '../../src/scene/Scene';
import { Camera } from '../../src/scene/Camera';
import { Matrix4 } from '../../src/math/Matrix4';
import type { Context } from '../../src/renderer/Context';

// ── 桩 ───────────────────────────────────────────────────────────────────────

class FakePrimitive {
  show = false; // Tileset3D 注册时先置 false，由渲染集决定显隐
  translucent = false;
  destroyed = false;
  modelMatrix: Matrix4 = Matrix4.identity();
  destroy(): void { this.destroyed = true; }
}

function makeFakeScene(): { scene: Scene; collection: PrimitiveCollection } {
  const collection = new PrimitiveCollection();
  const scene = {
    primitives: collection,
    addPrimitive: (p: Primitive) => { collection.add(p); },
    addTransparentPrimitive: (p: Primitive) => { collection.add(p); },
    _context: {} as Context,
  } as unknown as Scene;
  return { scene, collection };
}

interface ManualLoader {
  loader: ReturnType<typeof vi.fn>;
  requestedUrls: string[];
  resolve(url: string): void;
}

/** 手动 resolve 的假 content loader：url → 未决 Promise（逐 Primitive 1 个假件）。 */
function makeManualLoader(): ManualLoader {
  const resolvers = new Map<string, () => void>();
  const requestedUrls: string[] = [];
  const loader = vi.fn(
    (_ctx: Context, content: { resolvedUrl: string }, _t: Matrix4) => {
      requestedUrls.push(content.resolvedUrl);
      return new Promise<{ primitives: Primitive[]; textures: GPUTexture[] }>((resolve) => {
        resolvers.set(content.resolvedUrl, () =>
          resolve({ primitives: [new FakePrimitive() as unknown as Primitive], textures: [] })
        );
      });
    }
  );
  return {
    loader,
    requestedUrls,
    resolve(url: string): void {
      const r = resolvers.get(url);
      if (!r) throw new Error(`no pending request for ${url}`);
      r();
    },
  };
}

/** tileset.json 的 fetch 桩（content 加载走测试缝，不会 fetch glb）。 */
function stubTilesetFetch(payload: object): void {
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    arrayBuffer: async () =>
      new TextEncoder().encode(JSON.stringify(payload)).buffer as ArrayBuffer,
  })));
}

/** 俯视原点的相机（位置模长 2.0，nadir 朝 (0,0,0)）。 */
function makeNadirCamera(): Camera {
  const camera = new Camera();
  camera.setView({
    destination: { lon: 0, lat: 0, height: Camera.EARTH_SCALE }, // 归一化后 |p| = 2
    orientation: { heading: 0, pitch: -Math.PI / 2 },
  });
  return camera;
}

/** 让微任务/宏任务链（注册、缓存插入）全部落定。 */
async function settle(): Promise<void> {
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0));
}

/** 合成树骨架：单位 transform、原点球（半径 0.001 已是归一化尺度）。 */
function treeWith(root: object): object {
  return {
    root: {
      boundingVolume: { sphere: [0, 0, 0, 0.001] },
      ...root,
    },
  };
}

async function makeTileset(
  payload: object,
  loader: ManualLoader['loader'],
  extra?: { cacheCapacity?: number }
): Promise<{ tileset: Tileset3D; collection: PrimitiveCollection }> {
  const { scene, collection } = makeFakeScene();
  stubTilesetFetch(payload);
  const tileset = new Tileset3D({
    url: 'http://test/tileset.json',
    scene,
    maximumSseError: 16,
    ...extra,
    _contentLoader: loader,
  });
  await tileset.ready();
  return { tileset, collection };
}

// SSE 数值背景（d = 2，vh = 1080，fov 60°）：factor = 1080/(2×1.1547) ≈ 467.7
// g = 1e6 → sse ≈ 73（细分）；g = 100 → sse ≈ 0.007（选中自身）

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('Tileset3D – REPLACE 状态机（L0 上屏 → L1 就绪后父隐藏）', () => {
  it('父可见回退 → 子就绪后父 show=false', async () => {
    const manual = makeManualLoader();
    const { tileset, collection } = await makeTileset(treeWith({
      geometricError: 1e6,                       // sse≈73 → 细分
      content: { uri: 'root.glb' },              // REPLACE（缺省）
      children: [{
        geometricError: 100,                     // sse≈0.007 → 选中
        boundingVolume: { sphere: [0, 0, 0, 0.0005] },
        content: { uri: 'child.glb' },
      }],
    }), manual.loader);

    const camera = makeNadirCamera();

    // 帧 1：child 选中；祖先 root 优先入队 → 两个请求都在途
    tileset.update(camera);
    expect(tileset.stats.selected).toBe(1);
    expect(tileset.stats.requested).toBe(2);
    expect(collection.length).toBe(0); // 尚无 Primitive 上屏

    // L0 就绪 → 上屏（回退显示）
    manual.resolve('http://test/root.glb');
    await settle();
    expect(collection.length).toBe(1);

    tileset.update(camera);
    expect(tileset.stats.drawn).toBe(1);
    expect(collection.get(0).show).toBe(true); // L0 先上屏 ✓

    // L1 就绪 → REPLACE 父隐藏、子显示
    manual.resolve('http://test/child.glb');
    await settle();
    expect(collection.length).toBe(2);

    tileset.update(camera);
    expect(tileset.stats.drawn).toBe(1);            // 仍只有一个瓦片可见
    expect(collection.get(0).show).toBe(false);     // REPLACE 父已隐藏 ✓
    expect(collection.get(1).show).toBe(true);
  });
});

describe('Tileset3D – LRU 驱逐（两段式延迟销毁）', () => {
  it('容量 1：载入第二个子瓦片 → 驱逐第一个 → 下一帧真正 remove', async () => {
    const manual = makeManualLoader();
    const { tileset, collection } = await makeTileset(treeWith({
      geometricError: 1e6,
      children: [
        { geometricError: 100, boundingVolume: { sphere: [0, 0, 0, 0.0005] }, content: { uri: 'a.glb' } },
        { geometricError: 100, boundingVolume: { sphere: [0, 0, 0, 0.0006] }, content: { uri: 'b.glb' } },
      ],
    }), manual.loader, { cacheCapacity: 1 });

    const camera = makeNadirCamera();
    tileset.update(camera);
    expect(tileset.stats.requested).toBe(2);

    manual.resolve('http://test/a.glb');
    await settle();
    tileset.update(camera);
    const primA = collection.get(0);
    expect(primA.show).toBe(true);
    expect(tileset.stats.cached).toBe(1);

    // b 插入 → a 被驱逐（墓碑）：延迟销毁第 1 段——尚未 update，
    // a 的 GPU 资源必须还活着（不能在驱逐当刻立即 destroy）
    manual.resolve('http://test/b.glb');
    await settle();
    expect(tileset.stats.cached).toBe(1);        // 驻留只剩 b
    expect(primA.destroyed).toBe(false);         // 墓碑期：未销毁
    expect(collection.contains(primA)).toBe(true);

    // 下一次 update 开头 flush → 真正 remove/destroy（第 2 段，
    // 期间已跨过一次完整 render/submit 边界，满足 WebGPU 约束）
    tileset.update(camera);
    expect(primA.destroyed).toBe(true);
    expect(collection.contains(primA)).toBe(false);
    expect(collection.length).toBe(1); // 只剩 b
  });
});

describe('Tileset3D – 并发上限', () => {
  it('10 个选中瓦片 + 请求永不完成 → 在途恒 ≤ 8', async () => {
    const manual = makeManualLoader();
    const children = Array.from({ length: 10 }, (_, i) => ({
      geometricError: 100,
      boundingVolume: { sphere: [i * 1e-6, 0, 0, 0.0001] },
      content: { uri: `leaf_${i}.glb` },
    }));
    const { tileset } = await makeTileset(
      treeWith({ geometricError: 1e6, children }), manual.loader
    );

    const camera = makeNadirCamera();
    tileset.update(camera);
    expect(tileset.stats.requested).toBe(8); // 上限 8（root 无 content）
    expect(manual.requestedUrls.length).toBe(8);

    tileset.update(camera); // 在途去重：不追加请求
    expect(tileset.stats.requested).toBe(8);
    expect(manual.requestedUrls.length).toBe(8);
  });
});

describe('Tileset3D – destroy 与生命周期', () => {
  it('ready 幂等（同一 Promise）；destroy 卸载已上屏 Primitive', async () => {
    const manual = makeManualLoader();
    const { tileset, collection } = await makeTileset(treeWith({
      geometricError: 100, // 不细分：root 自身选中
      content: { uri: 'root.glb' },
    }), manual.loader);

    expect(tileset.ready()).toBe(tileset.ready()); // 幂等

    const camera = makeNadirCamera();
    tileset.update(camera);
    manual.resolve('http://test/root.glb');
    await settle();
    expect(collection.length).toBe(1);
    tileset.update(camera); // 上屏后跑一次渲染集，统计 drawn
    expect(tileset.stats.drawn).toBe(1);

    tileset.destroy();
    expect(collection.length).toBe(0);
    expect((collection.get(0) as FakePrimitive | undefined)).toBeUndefined();
  });

  it('在途请求于 destroy 后完成 → 不得自注册（防御性释放）', async () => {
    const manual = makeManualLoader();
    const { tileset, collection } = await makeTileset(treeWith({
      geometricError: 100,
      content: { uri: 'root.glb' },
    }), manual.loader);

    const camera = makeNadirCamera();
    tileset.update(camera);
    expect(tileset.stats.requested).toBe(1);

    // 先销毁，再让请求完成 → Primitive 不得进入场景
    tileset.destroy();
    manual.resolve('http://test/root.glb');
    await settle();
    expect(collection.length).toBe(0);
    expect(tileset.stats.cached).toBe(0);
  });
});
