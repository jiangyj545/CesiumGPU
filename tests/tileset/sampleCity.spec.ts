/**
 * sampleCity.spec.ts – 样板城市 tileset（buildSampleCityTileset.mjs 产物）
 * × 读取器端到端验收（Phase 1「倾斜摄影样板城市入库」收官测试）。
 *
 * 桩策略：tileset.json 与 b3dm 从磁盘（public/data/tilesets/sample-city/）
 * 经 global.fetch 桩供给——读取器网络路径真实走 fetch；Primitive 组装经
 * `_contentLoader` 测试缝注入（无 GPU，先例 tileset3dIntegration.spec）。
 *
 * 验收断言：
 * ① 数据合法性：40 个 b3dm 全部通过 28B 头/GLB 剥壳/parseGlbBuffer，
 *    BATCH_LENGTH 与 BatchTable 列对齐，Σ BATCH_LENGTH = 1280（≥1000）；
 * ② 树结构：root REPLACE、20×L0(GE 600)→20×L1(GE 100) 两级 LOD；
 * ③ 调度行为：远 12km 只请求 L0（远低）；近 1km 触发 L1 细分（近高）；
 *    L1 就绪后 REPLACE 隐藏同瓦片 L0；drawn>0。
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Tileset3D } from '../../src/tileset/Tileset3D';
import {
  computeGlbOffset,
  extractGlb,
  parseFeatureTableJson,
  parseBatchTable,
} from '../../src/tileset/B3dmDecoder';
import { GltfLoader } from '../../src/loader/GltfLoader';
import { PrimitiveCollection } from '../../src/scene/Primitive';
import type { Primitive } from '../../src/scene/Primitive';
import type { Scene } from '../../src/scene/Scene';
import { Camera } from '../../src/scene/Camera';
import { Cartesian3 } from '../../src/math/Cartesian3';
import { Matrix4 } from '../../src/math/Matrix4';
import type { Context } from '../../src/renderer/Context';

const CITY_DIR = join(process.cwd(), 'public/data/tilesets/sample-city');
const TILESET_URL = 'http://test/sample-city/tileset.json';
const LON = 114.4057;
const LAT = 30.5070;

// ── 磁盘数据加载 ─────────────────────────────────────────────────────────────

const tilesetJson = JSON.parse(
  readFileSync(join(CITY_DIR, 'tileset.json'), 'utf-8')
) as {
  asset: { version: string };
  root: { refine: string; geometricError: number; children: Array<{
    geometricError: number; content: { uri: string };
    children: Array<{ geometricError: number; content: { uri: string } }>;
  }> };
};

const TILE_COUNT = tilesetJson.root.children.length;   // 20

function readTile(relUri: string): ArrayBuffer {
  const buf = readFileSync(join(CITY_DIR, relUri));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

/** fetch 桩：URL pathname → 磁盘文件（tileset.json / b3dm）。 */
function stubDiskFetch(): void {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    const path = url.replace(/^https?:\/\/[^/]+/, '').split('?')[0]
      .replace(/^\/sample-city/, '');   // http://test/sample-city/… → CITY_DIR 相对
    return {
      ok: true,
      arrayBuffer: async () => readTile(path),
    };
  }));
}

// ── 桩（先例：tileset3dIntegration.spec）─────────────────────────────────────

class FakePrimitive {
  show = false;
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

/** 立即 resolve 的 content 缝：记录请求 URL，逐 Primitive 1 假件。 */
function makeInstantLoader() {
  const requestedUrls: string[] = [];
  const primByUrl = new Map<string, Primitive>();
  const loader = vi.fn(
    (_ctx: Context, content: { resolvedUrl: string }, _t: Matrix4) => {
      requestedUrls.push(content.resolvedUrl);
      const prim = new FakePrimitive() as unknown as Primitive;
      primByUrl.set(content.resolvedUrl, prim);
      return Promise.resolve({ primitives: [prim], textures: [] });
    }
  );
  return { loader, requestedUrls, primByUrl };
}

function makeNadirCamera(heightM: number): Camera {
  const camera = new Camera();
  camera.setView({
    destination: Cartesian3.fromDegrees(LON, LAT, heightM), // 真 WGS84 ECEF（同 94 页警示）
    orientation: { heading: 0, pitch: -Math.PI / 2 },
  });
  return camera;
}

async function settle(): Promise<void> {
  for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));
}

/**
 * 泵调度：请求队列靠 update() 推进（并发 8 上限），循环到请求集合
 * 稳定（连续两轮无新增）或轮数耗尽；再跑 1 帧 REPLACE 渲染集稳定。
 */
async function pump(tileset: Tileset3D, camera: Camera, maxRounds = 12): Promise<void> {
  let prev = -1;
  for (let i = 0; i < maxRounds; i++) {
    tileset.update(camera);
    await settle();
    if (tileset.stats.requested === prev) break;
    prev = tileset.stats.requested;
  }
  tileset.update(camera); // 最后一帧：注册/REPLACE 状态机收敛
  await settle();
}

afterEach(() => {
  vi.unstubAllGlobals();
});

// ── ① 数据合法性（真解析层，无桩缝）──────────────────────────────────────────

describe('样板城市数据 – b3dm/GLB 全量真解析', () => {
  const leaves = tilesetJson.root.children;

  it('树结构：root REPLACE，20×L0(GE600) → 20×L1(GE100) 两级 LOD', () => {
    expect(tilesetJson.asset.version).toBe('1.1');
    expect(tilesetJson.root.refine).toBe('REPLACE');
    expect(TILE_COUNT).toBe(20);
    for (const leaf of leaves) {
      expect(leaf.geometricError).toBe(600);
      expect(leaf.content.uri).toMatch(/^tiles\/l0_\d+_\d+\.b3dm$/);
      expect(leaf.children.length).toBe(1);
      expect(leaf.children[0].geometricError).toBe(100);
      expect(leaf.children[0].content.uri).toMatch(/^tiles\/l1_\d+_\d+\.b3dm$/);
    }
  });

  it('40 个 b3dm：28B 头 / BATCH_LENGTH=64 / BatchTable 列对齐 / GLB 可解析', async () => {
    let totalBatches = 0;
    for (const leaf of leaves) {
      for (const tile of [leaf, leaf.children[0]]) {
        const buf = readTile(tile.content.uri);
        expect(computeGlbOffset(buf)).toBeGreaterThan(28);

        const ft = parseFeatureTableJson(buf);
        expect(ft.BATCH_LENGTH).toBe(64);

        const table = parseBatchTable(buf, 64);
        expect(table).not.toBeNull();
        expect(table!['name']).toHaveLength(64);
        expect(table!['height']).toHaveLength(64);
        expect(table!['floors']).toHaveLength(64);

        const parsed = await GltfLoader.parseGlbBuffer(extractGlb(buf));
        expect(parsed.primitives.length).toBe(1);
        const prim = parsed.primitives[0];
        expect(prim.indices.length).toBeGreaterThan(0);
        // 每顶点 _BATCHID（展平后与索引等长）
        expect(prim.batchIds).toBeDefined();
        expect(prim.batchIds!.length).toBe(prim.indices.length);

        totalBatches += Number(ft.BATCH_LENGTH);
      }
    }
    expect(totalBatches).toBe(20 * 64 * 2); // L0+L1 各 1280
    expect(totalBatches / 2).toBeGreaterThanOrEqual(1000);
  });

  it('LOD 面数差：同瓦片 L1 三角数 > L0（近档几何更细）', () => {
    for (const leaf of leaves) {
      const idx0 = triCount(readTile(leaf.content.uri));
      const idx1 = triCount(readTile(leaf.children[0].content.uri));
      expect(idx1).toBeGreaterThan(idx0);
    }
  });

  /** 从 b3dm 里直接读 indices accessor count（避免 async 断言膨胀）。 */
  function triCount(buf: ArrayBuffer): number {
    const dv = new DataView(buf);
    // GLB 头 12B：magic(0) ver(4) total(8)；首 chunk：len(12) type(16) data(20)
    const glbOffset = computeGlbOffset(buf);
    const jsonLen = dv.getUint32(glbOffset + 12, true);
    const json = JSON.parse(new TextDecoder().decode(
      new Uint8Array(buf, glbOffset + 20, jsonLen)
    )) as { accessors: Array<{ count: number }> };
    return json.accessors[1].count / 3; // accessor 1 = indices
  }
});

// ── ② 读取器调度（fetch 桩 + content 缝）─────────────────────────────────────

describe('样板城市 × Tileset3D – LOD 切换与 REPLACE', () => {
  it('远 40km：只请求 20 个 L0（远低，L1 不进入请求集）', async () => {
    // SSE 阈值（computeTileSse：sse = GE·vh/(d·2tan(fov/2))，vh1080/fov60°
    // → factor 935.3/m）：leaf GE 600 细分需 d < 35km；根 GE 6000 细分需
    // d < 350km。40km 落在「L0 自身选中、L1 不被细分请求」窗。
    stubDiskFetch();
    const { scene } = makeFakeScene();
    const seam = makeInstantLoader();
    const tileset = new Tileset3D({
      url: TILESET_URL, scene, maximumSseError: 16, _contentLoader: seam.loader,
    });
    await tileset.ready();

    await pump(tileset, makeNadirCamera(40000));

    expect(seam.requestedUrls.length).toBe(20);
    expect(seam.requestedUrls.every((u) => u.includes('/l0_'))).toBe(true);
    expect(seam.requestedUrls.some((u) => u.includes('/l1_'))).toBe(false);
    tileset.destroy();
  });

  it('近 1km：触发 L1 细分（近高）；L1 就绪后 REPLACE 隐藏同瓦片 L0', async () => {
    stubDiskFetch();
    const { scene, collection } = makeFakeScene();
    const seam = makeInstantLoader();
    const tileset = new Tileset3D({
      url: TILESET_URL, scene, maximumSseError: 16, _contentLoader: seam.loader,
    });
    await tileset.ready();

    await pump(tileset, makeNadirCamera(1000));

    // L0 与 L1 同时被请求（祖先降级 + 子细分）
    expect(seam.requestedUrls.some((u) => u.includes('/l0_0_0'))).toBe(true);
    expect(seam.requestedUrls.some((u) => u.includes('/l1_0_0'))).toBe(true);

    expect(tileset.stats.drawn).toBeGreaterThan(0);

    // REPLACE 语义：同瓦片 L0 隐藏、L1 显示（双份 Primitive 均在场景）
    const base = TILESET_URL.replace('tileset.json', '');
    const l0 = seam.primByUrl.get(`${base}tiles/l0_0_0.b3dm`);
    const l1 = seam.primByUrl.get(`${base}tiles/l1_0_0.b3dm`);
    expect(l0).toBeDefined();
    expect(l1).toBeDefined();
    expect(collection.contains(l0!)).toBe(true);
    expect(collection.contains(l1!)).toBe(true);
    expect(l1!.show).toBe(true);
    expect(l0!.show).toBe(false); // 子就绪后父隐藏（近档替换远档的显示面）

    tileset.destroy();
    expect(collection.length).toBe(0);
  });

  it('加载零报错：failed=0、请求数为正', async () => {
    stubDiskFetch();
    const { scene } = makeFakeScene();
    const seam = makeInstantLoader();
    const tileset = new Tileset3D({
      url: TILESET_URL, scene, maximumSseError: 16, _contentLoader: seam.loader,
    });
    await tileset.ready();
    await pump(tileset, makeNadirCamera(1000));
    expect(tileset.stats.failed).toBe(0);
    expect(seam.requestedUrls.length).toBeGreaterThan(0);
    tileset.destroy();
  });
});
