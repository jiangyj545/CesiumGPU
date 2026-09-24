/**
 * TilesetContentLoader.spec.ts – tileset.json 树解析单测。
 *
 * 覆盖：transform 归一化数值断言（米制 [6378137,0,0] → 归一化 [1,0,0]）、
 * 子级累计、refine 继承、boundingVolume sphere/box/region → 世界球、
 * content uri 解析 + urlDecorator 注入、fetchTilesetJson 取数（fetch 桩）。
 * 全部使用合成数据。
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  fetchTilesetJson,
  parseTileset,
  TILESET_EARTH_SCALE,
} from '../../src/tileset/TilesetContentLoader';
import type { TileNode } from '../../src/tileset/TilesetContentLoader';

const R = TILESET_EARTH_SCALE; // 6378137

/** 米制 ENU 平移矩阵（列主序 16 元）。 */
function translationMetre(x: number, y: number, z: number): number[] {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];
}

function flattenTree(node: TileNode): TileNode[] {
  return [node, ...node.children.flatMap(flattenTree)];
}

/** 相对误差断言：Matrix4.values 是 Float32Array（f32 舍入 ~6e-8 相对误差，
 * 见 loader-layer-learnings：期望值不能按 f64 直接比）。 */
function expectCloseRel(actual: number, expected: number, rel = 1e-5): void {
  const tolerance = Math.max(Math.abs(expected) * rel, 1e-12);
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);
}

describe('parseTileset – transform 归一化', () => {
  it('米制平移 [6378137,0,0] → 归一化世界球中心 [1,0,0]', () => {
    const node = parseTileset({
      root: {
        transform: translationMetre(R, 0, 0),
        geometricError: 100,
        boundingVolume: { sphere: [0, 0, 0, 500] },
      },
    }, 'http://test/tileset.json');

    expectCloseRel(node.sphere.center.x, 1);
    expect(node.sphere.center.y).toBeCloseTo(0, 12);
    expect(node.sphere.center.z).toBeCloseTo(0, 12);
    // 半径按旋转列最大模（= 1/R）缩放：500m → 500/R
    expectCloseRel(node.sphere.radius, 500 / R);
  });

  it('子级累计：N子 = N父×T子（等效整体 S×T_acc）', () => {
    // 父平移 (R,0,0)、子局部再平移 (0,R,0) → 归一化后 (1,1,0)
    const node = parseTileset({
      root: {
        transform: translationMetre(R, 0, 0),
        geometricError: 100,
        boundingVolume: { sphere: [0, 0, 0, 10] },
        children: [
          {
            transform: translationMetre(0, R, 0),
            geometricError: 10,
            boundingVolume: { sphere: [0, 0, 0, 10] },
          },
        ],
      },
    }, 'http://test/tileset.json');

    const child = node.children[0];
    expectCloseRel(child.sphere.center.x, 1);
    expectCloseRel(child.sphere.center.y, 1);
    expect(child.sphere.center.z).toBeCloseTo(0, 12);
    expect(child.depth).toBe(1);
    expect(child.parent).toBe(node);
  });

  it('无 transform 的瓦片继承父矩阵（根缺省单位阵 → 世界原点）', () => {
    const node = parseTileset({
      root: {
        geometricError: 100,
        boundingVolume: { sphere: [0, 0, 0, 1] },
      },
    }, 'http://test/tileset.json');
    expect(node.sphere.center.x).toBeCloseTo(0, 12);
    expectCloseRel(node.sphere.radius, 1 / R);
  });

  it('缺 root → 抛错', () => {
    expect(() => parseTileset({}, 'http://test/tileset.json')).toThrow(/root/);
  });
});

describe('parseTileset – refine 继承', () => {
  it('子缺省继承父 refine；显式声明优先', () => {
    const node = parseTileset({
      root: {
        refine: 'REPLACE',
        geometricError: 100,
        boundingVolume: { sphere: [0, 0, 0, 1] },
        children: [
          { geometricError: 10, boundingVolume: { sphere: [0, 0, 0, 1] } },                                 // 继承 REPLACE
          { refine: 'ADD', geometricError: 10, boundingVolume: { sphere: [0, 0, 0, 1] } },                  // 显式 ADD
        ],
      },
    }, 'http://test/tileset.json');
    expect(node.refine).toBe('REPLACE');
    expect(node.children[0].refine).toBe('REPLACE');
    expect(node.children[1].refine).toBe('ADD');
  });

  it('根未声明 refine → 缺省 REPLACE', () => {
    const node = parseTileset({
      root: { geometricError: 1, boundingVolume: { sphere: [0, 0, 0, 1] } },
    }, 'http://test/tileset.json');
    expect(node.refine).toBe('REPLACE');
  });
});

describe('parseTileset – boundingVolume → 世界球', () => {
  it('box（轴对齐半轴）→ 外接世界球', () => {
    const node = parseTileset({
      root: {
        transform: translationMetre(R, 0, 0),
        geometricError: 100,
        boundingVolume: { box: [0, 0, 0, 100, 0, 0, 0, 50, 0, 0, 0, 20] },
      },
    }, 'http://test/tileset.json');
    // 保守半径 = 最大半轴 100m
    expectCloseRel(node.sphere.radius, 100 / R);
    expectCloseRel(node.sphere.center.x, 1);
  });

  it('region（弧度，绝对 WGS84，与 transform 无关）', () => {
    const node = parseTileset({
      root: {
        transform: translationMetre(R, 0, 0),
        geometricError: 100,
        boundingVolume: { region: [0, 0, 0.01, 0.01, 0, 100] },
      },
    }, 'http://test/tileset.json');
    // 中心应落在地球表面附近（归一化模长 ~1 + 高度项）
    const d = Math.hypot(node.sphere.center.x, node.sphere.center.y, node.sphere.center.z);
    expect(d).toBeGreaterThan(0.999);
    expect(d).toBeLessThan(1.01);
    expect(node.sphere.radius).toBeGreaterThan(0);
  });

  it('缺失 boundingVolume → 单位球兜底（warn 不抛）', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const node = parseTileset({
        root: { geometricError: 1, transform: translationMetre(R, 0, 0) },
      }, 'http://test/tileset.json');
      expect(node.sphere.radius).toBe(1);
      expect(warnSpy).toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });
});

describe('parseTileset – content 解析与 urlDecorator', () => {
  it('相对 uri 基于 tileset.json URL 解析；扩展名判 kind', () => {
    const nodes = flattenTree(parseTileset({
      root: {
        geometricError: 100,
        boundingVolume: { sphere: [0, 0, 0, 1] },
        content: { uri: 'root.b3dm' },
        children: [
          { geometricError: 10, boundingVolume: { sphere: [0, 0, 0, 1] }, content: { uri: 'sub/model.glb' } },
          { geometricError: 10, boundingVolume: { sphere: [0, 0, 0, 1] }, content: { uri: 'cloud.pnts' } },
        ],
      },
    }, 'http://test/dir/tileset.json'));

    expect(nodes[0].content?.resolvedUrl).toBe('http://test/dir/root.b3dm');
    expect(nodes[0].content?.kind).toBe('b3dm');
    expect(nodes[1].content?.resolvedUrl).toBe('http://test/dir/sub/model.glb');
    expect(nodes[1].content?.kind).toBe('glb');
    expect(nodes[2].content?.kind).toBe('pnts');
  });

  it('urlDecorator 逐请求注入（content uri 同样被加工）', () => {
    const node = parseTileset({
      root: {
        geometricError: 100,
        boundingVolume: { sphere: [0, 0, 0, 1] },
        content: { uri: 'root.b3dm' },
      },
    }, 'http://test/dir/tileset.json', {
      urlDecorator: (u) => `${u}${u.includes('?') ? '&' : '?'}key=K`,
    });
    expect(node.content?.resolvedUrl).toBe('http://test/dir/root.b3dm?key=K');
  });
});

describe('fetchTilesetJson', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('fetch → JSON.parse；urlDecorator 作用于请求 URL', async () => {
    const fetchStub = vi.fn(async () => ({
      ok: true,
      arrayBuffer: async () =>
        new TextEncoder().encode(JSON.stringify({
          root: { geometricError: 1, boundingVolume: { sphere: [0, 0, 0, 1] } },
        })).buffer as ArrayBuffer,
    }));
    vi.stubGlobal('fetch', fetchStub);

    const { json, finalUrl } = await fetchTilesetJson('http://test/tileset.json', {
      urlDecorator: (u) => `${u}?key=K`,
    });
    expect(finalUrl).toBe('http://test/tileset.json?key=K');
    expect(fetchStub).toHaveBeenCalledWith('http://test/tileset.json?key=K', undefined);
    expect(json.root).toBeDefined();
  });

  it('HTTP 失败 / 缺 root → 抛错（不静默）', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 404, statusText: 'Not Found' })));
    await expect(fetchTilesetJson('http://test/missing.json')).rejects.toThrow(/404/);

    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      arrayBuffer: async () => new TextEncoder().encode('{"asset":{}}').buffer as ArrayBuffer,
    })));
    await expect(fetchTilesetJson('http://test/empty.json')).rejects.toThrow(/root/);
  });
});
