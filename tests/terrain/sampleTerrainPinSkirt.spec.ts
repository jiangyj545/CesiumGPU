/**
 * sampleTerrainPinSkirt.spec.ts – 两项引擎级地形修复的回归测试：
 *
 * 1. pin 保护：sampleTerrain 预载 await 期间渲染流插入瓦片不得把已到手的
 *    采样瓦片从 LRU 挤出（Cesium sampleTerrainMostDetailed 语义——结果
 *    不受缓存压力影响；实测 18_dem 恩施 0 / 19-80 剖面中段半数 0）。
 * 2. 裙边排除：贴边（瓦片 rim）采样不得命中裙边墙三角形得到裙深伪负高
 *    （实测 -388 ~ -1854 m）。mesh 带 skirtVertexStart 时采样跳过含裙边
 *    顶点的三角形（几何级排除）。
 */
import { describe, it, expect, vi } from 'vitest';
import { TerrainLayer } from '../../src/terrain/TerrainLayer';
import {
  geographicTileBounds,
  ecefFromLonLatHeight,
  type TerrainProvider,
  type TerrainMeshData,
} from '../../src/terrain/TerrainProvider';
import { Cartographic } from '../../src/math/Cartographic';
import { sampleTerrain } from '../../src/terrain/sampleTerrain';

/**
 * 门控 provider：`gate(key)` 后该瓦片请求挂起直到测试 `release(key)`，
 * 精确控制“采样瓦片先落缓存、渲染瓦片后插入挤占”的时序。
 */
class GatedProvider implements TerrainProvider {
  readonly ready = true;
  readonly maximumLevel = 14;
  readonly tilingScheme = 'geographic' as const;
  readonly requests: string[] = [];
  private readonly gates = new Map<string, (() => void)[]>();

  /** 挂起指定瓦片（"z/x/y"），直到 release。 */
  gate(key: string): void {
    this.gates.set(key, []);
  }

  release(key: string): void {
    for (const r of this.gates.get(key) ?? []) r();
    this.gates.delete(key);
  }

  async requestTileGeometry(z: number, x: number, y: number): Promise<TerrainMeshData | null> {
    const key = `${z}/${x}/${y}`;
    this.requests.push(key);
    if (this.gates.has(key)) {
      await new Promise<void>((resolve) => this.gates.get(key)!.push(resolve));
    }
    return constantQuad(z, x, y, TILE_HEIGHTS[key] ?? 1000);
  }

  getTileDataAvailable(): boolean {
    return true;
  }

  destroy(): void { /* nothing */ }
}

/** 每瓦片恒定高程（米）：S1=472（恩施参考）、S2=300、S3=100。 */
const TILE_HEIGHTS: Record<string, number> = {
  '3/8/4': 472,
  '3/9/4': 300,
  '3/10/4': 100,
};

/**
 * 常高程四边形网格（2 三角形）；`withSkirt` 时在西边追加裙边墙（rim 顶点
 * 沿半径下压 `drop` 米），并标记 skirtVertexStart。
 */
function constantQuad(
  z: number,
  x: number,
  y: number,
  heightM: number,
  withSkirt = false,
  drop = 2000
): TerrainMeshData {
  const b = geographicTileBounds(z, x, y);
  const rim: Array<[number, number]> = [
    [b.west, b.south],
    [b.east, b.south],
    [b.east, b.north],
    [b.west, b.north],
  ];
  const skirtCount = withSkirt ? 2 : 0;
  const positions = new Float32Array((4 + skirtCount) * 3);
  const xyz: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < 4; i++) {
    ecefFromLonLatHeight(rim[i][0], rim[i][1], heightM, xyz);
    positions.set(xyz, i * 3);
  }
  // 裙边三角形排在基础网格之前：_sampleMeshHeight 取首个命中的三角形，
  // 这样“未排除裙边”的实现会先命中裙边墙拿到负高（回归可检出）。
  const indicesArr: number[] = withSkirt ? [3, 4, 5, 0, 4, 3] : [0, 1, 2, 0, 2, 3];
  if (withSkirt) {
    // 西边裙边墙：rim 0（西南）/ 3（西北）复制下沉 drop 米。经度再向瓦片
    // 外偏 0.2% 瓦片宽——真实网格的 float32 精度抖动让裙边三角形在
    // (lon,lat) 投影下不是精确退化窄条，宽松重心容差会把贴边查询点吸进
    // 去并插值出裙深负高（复现 19/80 面板的 -388 ~ -1854 m）。
    const skirtLon = b.west - (b.east - b.west) * 0.002;
    ecefFromLonLatHeight(skirtLon, rim[0][1], heightM - drop, xyz);
    positions.set(xyz, 12);
    ecefFromLonLatHeight(skirtLon, rim[3][1], heightM - drop, xyz);
    positions.set(xyz, 15);
    indicesArr.push(0, 1, 2, 0, 2, 3);
  }
  const indices = new Uint32Array(indicesArr);
  return {
    positions,
    indices,
    minHeight: withSkirt ? heightM - drop : heightM,
    maxHeight: heightM,
    skirtVertexStart: withSkirt ? 4 : undefined,
    center: [0, 0, 0],
  };
}

/** 瓦片内部（避开边界）的一个采样点。 */
function pointIn(z: number, x: number, y: number): Cartographic {
  const b = geographicTileBounds(z, x, y);
  const lon = b.west + (b.east - b.west) * 0.5;
  const lat = b.south + (b.north - b.south) * 0.5;
  return Cartographic.fromDegrees(lon, lat);
}

describe('sampleTerrain：采样期间 pin（结果不受缓存压力影响）', () => {
  it('预载期间渲染流插入瓦片不驱逐已到手的采样瓦片（S1 先落、R 后插）', async () => {
    const provider = new GatedProvider();
    // 容量 4：S1 落缓存后，4 块渲染瓦片足以把“最老”的它逐出（未 pin 时）。
    const layer = new TerrainLayer({ provider, cacheCapacity: 4 });

    // 包络覆盖 S1(3/8/4)、S2(3/9/4)、S3(3/10/4) 三块；全部挂起。
    const pts = [pointIn(3, 8, 4), pointIn(3, 9, 4), pointIn(3, 10, 4)];
    for (const key of ['3/8/4', '3/9/4', '3/10/4']) provider.gate(key);

    const sampling = sampleTerrain(layer, pts, 3);
    await vi.waitFor(() => {
      expect(provider.requests).toEqual(
        expect.arrayContaining(['3/8/4', '3/9/4', '3/10/4'])
      );
    });

    // ① S1 先到手落缓存（此时是全缓存最老条目——未 pin 的世界它就是牺牲品）。
    provider.release('3/8/4');
    await vi.waitFor(() => {
      expect(layer.getMesh(3, 8, 4)).not.toBeNull();
    });

    // ② 预载仍在 await（S2/S3 未回）——渲染流插入 4 块瓦片（x11..14, y4）
    //    占满容量 4：未 pin 的世界里最老条目 S1 必被逐出。
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await layer.ensureRegion({ west: 78.1, south: 1, east: 146, north: 20 }, 3);
    warn.mockRestore();
    expect(layer.getMesh(3, 14, 4)).not.toBeNull();

    // ③ S2/S3 到手 → ensureRegion 全部完成 → 回读。
    provider.release('3/9/4');
    provider.release('3/10/4');
    const result = await sampling;

    expect(result[0].height).toBeCloseTo(472, 0);
    expect(result[1].height).toBeCloseTo(300, 0);
    expect(result[2].height).toBeCloseTo(100, 0);
    // 回读完成后 pin 已解除（不留长驻 pin，LRU 恢复正常）。
  }, 20_000);

  it('pin 总数超限（包络 L10 祖先 >256 块）时退化为不 pin 且只 console.info 一次', async () => {
    const provider = new GatedProvider();
    const layer = new TerrainLayer({ provider, cacheCapacity: 512 });
    const info = vi.spyOn(console, 'info').mockImplementation(() => {});

    // 约 3° × 3° 的包络在 L10 反算 18×18 = 324 块 > 256（z10 = 2048×1024）。
    const wide = [Cartographic.fromDegrees(100, 29), Cartographic.fromDegrees(103, 32)];
    await sampleTerrain(layer, wide, 10);
    await sampleTerrain(layer, wide, 10);
    expect(info).toHaveBeenCalledTimes(1);
    info.mockRestore();
  });
});

describe('sampleHeight：裙边几何排除（贴边不再负值）', () => {
  it('贴边点命中裙边墙三角形时返回真实地形高程而非裙深负高', () => {
    const mesh = constantQuad(3, 8, 4, 500, true, 2000); // 裙底 -1500 m
    const provider: TerrainProvider = {
      ready: true,
      maximumLevel: 14,
      tilingScheme: 'geographic',
      requestTileGeometry: async () => mesh,
      getTileDataAvailable: () => true,
      destroy: () => {},
    };
    const layer = new TerrainLayer({ provider, cacheCapacity: 4 });
    return layer.ensureRegion(geographicTileBounds(3, 8, 4), 3).then(() => {
      const b = geographicTileBounds(3, 8, 4);
      // 西边 rim 中点：落在裙边墙与基础三角形的公共边上。
      const h = layer.sampleHeight(b.west, (b.south + b.north) / 2);
      expect(h).toBeGreaterThanOrEqual(mesh.minHeight);
      expect(h).not.toBeLessThan(0); // 不再是 -1500 一带的裙深伪高
      expect(h).toBeGreaterThan(400); // 基础网格常高程 500 m
    });
  });
});
