import { describe, it, expect, vi } from 'vitest';
import { TerrainLayer } from '../../src/terrain/TerrainLayer';
import type { TerrainProvider, TerrainMeshData, TerrainTilingScheme } from '../../src/terrain/TerrainProvider';

/**
 * TerrainLayer.ensureRegion / pinRegion / unpinAll 回归测试。
 *
 * ensureRegion 绕过视锥选择按 bounds 反算瓦片范围直接预载（淹没/水深
 * 分析用途）；pin 让 LRU 驱逐跳过指定瓦片。覆盖：
 * - bounds → z/x/y 反算集合正确（geographic TMS 南起 + webmercator 北起）
 * - 请求失败收集后抛带清单的错（不静默）；provider null（无数据）不抛
 * - 幂等（缓存命中/已失败不再请求）
 * - pin 前置校验（level <= 10、总数 <= 256）与全 pinned 时缓存超限保护
 */

/** 记录请求、按键返回预置网格/失败注入的桩 provider。 */
class RecordingProvider implements TerrainProvider {
  readonly ready = true;
  readonly maximumLevel = 14;
  readonly tilingScheme: TerrainTilingScheme;
  readonly meshes = new Map<string, TerrainMeshData>();
  readonly requests: string[] = [];
  readonly failKeys = new Set<string>();

  constructor(tilingScheme: TerrainTilingScheme = 'geographic') {
    this.tilingScheme = tilingScheme;
  }

  async requestTileGeometry(z: number, x: number, y: number): Promise<TerrainMeshData | null> {
    const key = `${z}/${x}/${y}`;
    this.requests.push(key);
    if (this.failKeys.has(key)) throw new Error(`模拟网络失败 ${key}`);
    return this.meshes.get(key) ?? null;
  }

  getTileDataAvailable(z: number, x: number, y: number): boolean {
    return z <= this.maximumLevel;
  }

  destroy(): void { /* nothing */ }
}

/** 占位最小网格（ensureRegion 只关心 _putMesh 入缓存，不采样）。 */
function stubMesh(): TerrainMeshData {
  return {
    positions: new Float32Array([1, 0, 0, 0, 0.6, 0, 0, 0, 0.6]),
    indices: new Uint16Array([0, 1, 2]),
    minHeight: 0,
    maxHeight: 0,
    center: [0, 0, 0],
  };
}

function sorted(provider: RecordingProvider): string[] {
  return [...provider.requests].sort();
}

describe('TerrainLayer.ensureRegion（geographic TMS）', () => {
  it('按 bounds 反算瓦片范围并发起恰好覆盖的请求（z3 3×3 = 9 块）', async () => {
    const provider = new RecordingProvider('geographic');
    const layer = new TerrainLayer({ provider, maximumLevel: 14 });

    // z3：cols=16, rows=8。bounds (0..45°E, 0..45°N) → x 8..10, y 4..6
    await layer.ensureRegion({ west: 0, east: 45, south: 0, north: 45 }, 3);

    expect(sorted(provider)).toEqual([
      '3/10/4', '3/10/5', '3/10/6',
      '3/8/4',  '3/8/5',  '3/8/6',
      '3/9/4',  '3/9/5',  '3/9/6',
    ]);
  });

  it('provider 无数据（null）不算失败：不抛错且不再重复请求', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const provider = new RecordingProvider('geographic'); // meshes 空 → 全 null
    const layer = new TerrainLayer({ provider, maximumLevel: 14 });
    const bounds = { west: 0, east: 45, south: 0, north: 45 };

    await expect(layer.ensureRegion(bounds, 3)).resolves.toBeUndefined();
    const first = provider.requests.length;
    expect(first).toBe(9);

    await layer.ensureRegion(bounds, 3); // 失败缓存生效：零重复请求
    expect(provider.requests.length).toBe(first);
    expect(warn).toHaveBeenCalled(); // 缺失清单有告警（不静默）
    warn.mockRestore();
  });

  it('请求 reject 收集后抛带完整清单的错', async () => {
    const provider = new RecordingProvider('geographic');
    provider.failKeys.add('3/8/4');
    provider.failKeys.add('3/9/6');
    provider.meshes.set('3/8/5', stubMesh());
    const layer = new TerrainLayer({ provider, maximumLevel: 14 });

    await expect(
      layer.ensureRegion({ west: 0, east: 45, south: 0, north: 45 }, 3)
    ).rejects.toThrow(/2\/9 瓦片请求失败.*3\/8\/4.*3\/9\/6/s);
    // 成功的瓦片已入缓存
    expect(layer.getMesh(3, 8, 5)).not.toBeNull();
  });

  it('幂等：缓存命中的瓦片不重复请求', async () => {
    const provider = new RecordingProvider('geographic');
    provider.meshes.set('3/8/4', stubMesh());
    const layer = new TerrainLayer({ provider, maximumLevel: 14 });
    const bounds = { west: 0, east: 11.25, south: 0, north: 11.25 }; // 恰 z3 瓦片 (8,4)

    await layer.ensureRegion(bounds, 3);
    expect(provider.requests).toEqual(['3/8/4']);
    await layer.ensureRegion(bounds, 3);
    expect(provider.requests).toEqual(['3/8/4']);
  });

  it('层级 clamp：请求 level 高于 provider.maximumLevel 时落到可用层级', async () => {
    const provider = new RecordingProvider('geographic');
    (provider as { maximumLevel: number }).maximumLevel = 5;
    const layer = new TerrainLayer({ provider, maximumLevel: 14 });

    // z5 瓦片 (32,16) = 0..5.625°；内缩避开瓦片边界（floor 落界会多含邻块）
    await layer.ensureRegion({ west: 0.1, east: 5.5, south: 0.1, north: 5.5 }, 8);
    expect(provider.requests).toEqual(['5/32/16']);
  });
});

describe('TerrainLayer.ensureRegion（webmercator XYZ）', () => {
  it('北起 y 反算正确（lat 0..45° → y 1..2 @z2）', async () => {
    const provider = new RecordingProvider('webmercator');
    const layer = new TerrainLayer({ provider, maximumLevel: 14 });

    await layer.ensureRegion({ west: 0, east: 90, south: 0, north: 45 }, 2);

    expect(sorted(provider)).toEqual(['2/2/1', '2/2/2', '2/3/1', '2/3/2']);
  });
});

describe('TerrainLayer.pinRegion / unpinAll', () => {
  it('前置校验：level > 10 抛错；瓦片数 > 256 抛错', () => {
    const provider = new RecordingProvider('geographic');
    const layer = new TerrainLayer({ provider, maximumLevel: 14 });

    expect(() => layer.pinRegion({ west: 0, east: 1, south: 0, north: 1 }, 11))
      .toThrow(/超过上限/);
    // z8 bounds (0..90°E, 0..45°N) → 129×65 = 8385 块 >> 256
    expect(() => layer.pinRegion({ west: 0, east: 90, south: 0, north: 45 }, 8))
      .toThrow(/超过上限 256/);
    expect(layer.unpinAll()).toBeUndefined();
  });

  it('pin 幂等：重复登记同一瓦片不重复计数', () => {
    const provider = new RecordingProvider('geographic');
    const layer = new TerrainLayer({ provider, maximumLevel: 14 });
    // z5 瓦片 (32,16) = 0..5.625°，内缩避开边界
    const bounds = { west: 0.1, east: 5.5, south: 0.1, north: 5.5 };

    expect(layer.pinRegion(bounds, 5)).toBe(1);
    expect(layer.pinRegion(bounds, 5)).toBe(1); // Set 去重
  });

  it('LRU 驱逐跳过 pinned；全 pinned 时允许缓存超限（保护分析数据）', async () => {
    const provider = new RecordingProvider('geographic');
    const layer = new TerrainLayer({ provider, maximumLevel: 14, cacheCapacity: 4 });

    // z3 瓦片宽 22.5°；第 i 块 bounds 内缩 0.1° 恰覆盖瓦片 (8+i, 4)
    const tileIBounds = (i: number) => ({
      west: i * 22.5 + 0.1, east: (i + 1) * 22.5 - 0.1, south: 0.1, north: 11.15,
    });
    const load = async (i: number): Promise<void> => {
      provider.meshes.set(`3/${8 + i}/4`, stubMesh());
      await layer.ensureRegion(tileIBounds(i), 3);
    };

    // ① 灌 4 块（=容量）后 pin 其中 2 块（8、9），10、11 留作驱逐候选
    for (let i = 0; i <= 3; i++) await load(i);
    layer.pinRegion(tileIBounds(0), 3); // (8,4)
    layer.pinRegion(tileIBounds(1), 3); // (9,4)

    // ② 新块挤入：驱逐最老的未 pin 瓦片 (10,4)，pinned 的 8/9 存活
    await load(4);
    expect(layer.getMesh(3, 10, 4)).toBeNull();          // 未 pin 被逐（驱逐在生效）
    expect(layer.getMesh(3, 8, 4)).not.toBeNull();       // pinned 存活
    expect(layer.getMesh(3, 9, 4)).not.toBeNull();

    // ③ 全部 pinned 后再挤入：无未 pin 可逐 → 缓存超限放行（5 > 4）
    layer.pinRegion(tileIBounds(2), 3);
    layer.pinRegion(tileIBounds(3), 3);
    layer.pinRegion(tileIBounds(4), 3);
    await load(5);
    expect(layer.stats.cachedMeshes).toBe(5);
    expect(layer.getMesh(3, 11, 4)).not.toBeNull();
    expect(layer.getMesh(3, 12, 4)).not.toBeNull();

    layer.unpinAll();
  });
});
