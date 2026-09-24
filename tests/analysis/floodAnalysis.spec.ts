import { describe, it, expect } from 'vitest';
import { TerrainLayer } from '../../src/terrain/TerrainLayer';
import { FloodAnalysis } from '../../src/analysis/FloodAnalysis';
import type { RectangleDeg } from '../../src/analysis/FloodAnalysis';
import type { TerrainProvider, TerrainMeshData } from '../../src/terrain/TerrainProvider';
import {
  geographicTileBounds,
  ecefFromLonLatHeight,
} from '../../src/terrain/TerrainProvider';

/**
 * FloodAnalysis 淹没分析回归测试。
 *
 * 手法照 tests/analysis/clampChain.spec.ts：RecordingProvider + gridMesh
 * 构造**解析高度场的真实 TerrainLayer 网格**（非 mock sampleHeight），
 * 让 prepareRegion → ensureRegion → sampleHeight 整条链路吃同一份数据。
 *
 * 断言覆盖（计划 P1 验收项）：
 * ① computeFlooded 淹没面积随水位单调增
 * ② marching squares 输出多边形环闭合（首尾点相同）
 * ③ 碗型盆地水位=沿口高时面积 ≈ 全格
 * ④ prepareRegion 触发的瓦片请求集合正确（stub provider 记录请求）
 * ⑤ pin 后 LRU 驱逐不逐 pinned
 */

/** 按键返回预置合成网格、并记录全部请求的桩 provider。 */
class RecordingProvider implements TerrainProvider {
  readonly ready = true;
  readonly maximumLevel = 14;
  readonly tilingScheme = 'geographic' as const;
  readonly meshes = new Map<string, TerrainMeshData>();
  /** 收到的全部 `"{z}/{x}/{y}"` 请求（含重复）。 */
  readonly requests: string[] = [];
  /** 命中即 reject 的键（模拟网络/解码失败）。 */
  readonly failKeys = new Set<string>();

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

/**
 * n×n 规则网格 mesh（无裙边）：顶点经纬度均匀覆盖瓦片，高度由
 * heightFn(lonDeg, latDeg) 给出，位置按引擎约定（归一化 ECEF，高度折入模长）。
 */
function gridMesh(
  z: number,
  x: number,
  y: number,
  heightFn: (lon: number, lat: number) => number,
  n: number
): TerrainMeshData {
  const b = geographicTileBounds(z, x, y);
  const positions = new Float32Array(n * n * 3);
  const tmp: [number, number, number] = [0, 0, 0];
  let minH = Infinity;
  let maxH = -Infinity;
  for (let iy = 0; iy < n; iy++) {
    for (let ix = 0; ix < n; ix++) {
      const lon = b.west + (ix / (n - 1)) * (b.east - b.west);
      const lat = b.south + (iy / (n - 1)) * (b.north - b.south);
      const h = heightFn(lon, lat);
      ecefFromLonLatHeight(lon, lat, h, tmp);
      const i = (iy * n + ix) * 3;
      positions[i] = tmp[0];
      positions[i + 1] = tmp[1];
      positions[i + 2] = tmp[2];
      minH = Math.min(minH, h);
      maxH = Math.max(maxH, h);
    }
  }
  const indices = new Uint32Array((n - 1) * (n - 1) * 6);
  let o = 0;
  for (let iy = 0; iy < n - 1; iy++) {
    for (let ix = 0; ix < n - 1; ix++) {
      const r0 = iy * n;
      const r1 = (iy + 1) * n;
      indices[o++] = r0 + ix;
      indices[o++] = r1 + ix;
      indices[o++] = r0 + ix + 1;
      indices[o++] = r0 + ix + 1;
      indices[o++] = r1 + ix;
      indices[o++] = r1 + ix + 1;
    }
  }
  return { positions, indices, minHeight: minH, maxHeight: maxH, center: [0, 0, 0] };
}

// ── 夹具：z6 单瓦片内的分析区（四边内缩 2%，恰好覆盖 1 个瓦片）──────────────

const TILE = { z: 6, x: 64, y: 32 }; // geographic：west=0, east=2.8125, south=0, north=2.8125
const TILE_B = geographicTileBounds(TILE.z, TILE.x, TILE.y);
const PAD = (TILE_B.east - TILE_B.west) * 0.02;
/** 分析区（瓦片内缩，bounds→瓦片反算恰命中 TILE 一块）。 */
const BOUNDS: RectangleDeg = {
  west:  TILE_B.west + PAD,
  east:  TILE_B.east - PAD,
  south: TILE_B.south + PAD,
  north: TILE_B.north - PAD,
};

/** 南北斜坡：h = 100 + 2200·(lat-south)/span → 100 ~ 2300 m，水位等值线为纬线。 */
const SLOPE_FN = (lon: number, lat: number): number =>
  100 + 2200 * ((lat - TILE_B.south) / (TILE_B.north - TILE_B.south));

/** 锥形坑（碗型盆地）：中心 0 m，四角 ≈ 495 m（< 500 沿口水位 → 全格淹没）。 */
const BOWL_FN = (lon: number, lat: number): number => {
  const dx = (lon - (BOUNDS.west + BOUNDS.east) / 2) / (BOUNDS.east - BOUNDS.west);
  const dy = (lat - (BOUNDS.south + BOUNDS.north) / 2) / (BOUNDS.north - BOUNDS.south);
  return 700 * Math.hypot(dx, dy);
};

/** 键排序后的请求集合（便于断言）。 */
function sortedRequests(provider: RecordingProvider): string[] {
  return [...provider.requests].sort();
}

/** 建 FloodAnalysis（真实 TerrainLayer + 桩 provider，全程无网络）。 */
function makeFlood(heightFn: (lon: number, lat: number) => number, gridN = 24): {
  layer: TerrainLayer;
  provider: RecordingProvider;
  flood: FloodAnalysis;
} {
  const provider = new RecordingProvider();
  provider.meshes.set(
    `${TILE.z}/${TILE.x}/${TILE.y}`,
    gridMesh(TILE.z, TILE.x, TILE.y, heightFn, 33)
  );
  const layer = new TerrainLayer({ provider, maximumLevel: TILE.z, cacheCapacity: 8 });
  const flood = new FloodAnalysis(layer, {
    gridN,
    heightSampler: (lon, lat) => layer.sampleHeight(lon, lat),
  });
  return { layer, provider, flood };
}

// ── 用例 ─────────────────────────────────────────────────────────────────────

describe('FloodAnalysis.computeFlooded（斜坡 DEM）', () => {
  it('① 淹没面积随水位单调增', async () => {
    const { flood } = makeFlood(SLOPE_FN);
    await flood.prepareRegion(BOUNDS, TILE.z);

    let prev = -1;
    for (const level of [700, 1200, 1700, 2200]) {
      const res = flood.computeFlooded(BOUNDS, level);
      expect(res.areaKm2).toBeGreaterThan(prev); // 严格递增（斜坡连续）
      expect(res.level).toBe(level);
      prev = res.areaKm2;
    }
    // 水位低于最低高程（100 m）→ 零淹没
    const dry = flood.computeFlooded(BOUNDS, 50);
    expect(dry.areaKm2).toBeCloseTo(0, 6);
    expect(dry.polygons.length).toBe(0);
  });

  it('② 多边形环闭合（首尾点严格相同）且落在分析区内', async () => {
    const { flood } = makeFlood(SLOPE_FN);
    await flood.prepareRegion(BOUNDS, TILE.z);

    const res = flood.computeFlooded(BOUNDS, 1200); // 等值线横穿网格
    expect(res.polygons.length).toBeGreaterThan(0);
    for (const ring of res.polygons) {
      expect(ring.length).toBeGreaterThanOrEqual(4);
      expect(ring[0][0]).toBe(ring[ring.length - 1][0]); // lon 首尾相同
      expect(ring[0][1]).toBe(ring[ring.length - 1][1]); // lat 首尾相同
      for (const [lon, lat] of ring) {
        expect(lon).toBeGreaterThanOrEqual(BOUNDS.west - 1e-9);
        expect(lon).toBeLessThanOrEqual(BOUNDS.east + 1e-9);
        expect(lat).toBeGreaterThanOrEqual(BOUNDS.south - 1e-9);
        expect(lat).toBeLessThanOrEqual(BOUNDS.north + 1e-9);
      }
    }
  });

  it('③ 碗型盆地水位=沿口高时面积≈全格（等值线消失）', async () => {
    const { flood } = makeFlood(BOWL_FN);
    await flood.prepareRegion(BOUNDS, TILE.z);

    const res = flood.computeFlooded(BOUNDS, 500);
    const total =
      (BOUNDS.east - BOUNDS.west) * 111.32 * Math.cos(((BOUNDS.south + BOUNDS.north) / 2) * Math.PI / 180) *
      (BOUNDS.north - BOUNDS.south) * 110.86;
    expect(res.areaKm2).toBeGreaterThan(total * 0.99); // 全格淹没（角点最高 ≈495 m < 500）
    expect(res.polygons.length).toBe(0);               // 全 case15，无等值线
  });
});

describe('FloodAnalysis.prepareRegion（真实 TerrainLayer + 桩 provider）', () => {
  it('④ 触发的瓦片请求集合正确，且幂等（重复调用不再请求）', async () => {
    const { provider, flood } = makeFlood(SLOPE_FN);

    await flood.prepareRegion(BOUNDS, TILE.z);
    expect(sortedRequests(provider)).toEqual([`${TILE.z}/${TILE.x}/${TILE.y}`]);

    const countAfterFirst = provider.requests.length;
    await flood.prepareRegion(BOUNDS, TILE.z); // 幂等：缓存命中 + pin Set 去重
    expect(provider.requests.length).toBe(countAfterFirst);
  });

  it('heightGrid 返回 gridN×gridN 行主序网格，值经 sampleHeight 链路', async () => {
    const { flood } = makeFlood(SLOPE_FN, 16);
    await flood.prepareRegion(BOUNDS, TILE.z); // 先装载网格，sampleHeight 才有覆盖
    const grid = flood.heightGrid(BOUNDS, 16);
    expect(grid.length).toBe(16 * 16);
    // 南端 ≈ 100 + 一点内缩偏移，北端 ≈ 2300 - 一点内缩偏移（斜坡解析值域内）
    expect(Math.min(...grid)).toBeGreaterThan(90);
    expect(Math.max(...grid)).toBeLessThan(2310);
  });

  it('⑤ pin 后 LRU 驱逐不逐 pinned 瓦片', async () => {
    const { layer, provider } = makeFlood(SLOPE_FN);
    await layer.ensureRegion(BOUNDS, TILE.z);
    layer.pinRegion(BOUNDS, TILE.z);
    expect(layer.getMesh(TILE.z, TILE.x, TILE.y)).not.toBeNull();

    // 容量 8：再灌 12 个邻接瓦片，未 pin 时最早的分析瓦片必被逐出
    for (let dx = 1; dx <= 12; dx++) {
      const x = TILE.x + dx;
      provider.meshes.set(
        `${TILE.z}/${x}/${TILE.y}`,
        gridMesh(TILE.z, x, TILE.y, SLOPE_FN, 9)
      );
      await layer.ensureRegion(geographicTileBounds(TILE.z, x, TILE.y), TILE.z);
    }
    expect(layer.getMesh(TILE.z, TILE.x, TILE.y)).not.toBeNull(); // pinned 存活
  });

  it('release 后 unpin 生效且实例不可再用；pin 校验仍拦截超限 level', async () => {
    const { layer, flood } = makeFlood(SLOPE_FN);
    await flood.prepareRegion(BOUNDS, TILE.z);
    flood.release();
    expect(() => flood.computeFlooded(BOUNDS, 100)).toThrow(/已释放/);
    await expect(flood.prepareRegion(BOUNDS, TILE.z)).rejects.toThrow(/已释放/);
    // pin 的 level 校验独立于 release 状态（TerrainLayer 自身约定）
    expect(() => layer.pinRegion(BOUNDS, 11)).toThrow(/超过上限/);
  });
});
