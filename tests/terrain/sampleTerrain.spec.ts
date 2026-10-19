/**
 * sampleTerrain.spec.ts – P1-H sampleTerrainMostDetailed 异步语义：
 * 包络矩形预载 → sampleHeight 回填 → 新数组不改入参；空数组 / 无地形回退。
 * 采样读回用 spy 注入高度（插值本身有 ensureRegion.spec / terrainProvider
 * 回归覆盖），预载范围正确性由 RecordingProvider 的请求瓦片集合验证。
 */
import { describe, it, expect, vi } from 'vitest';
import { TerrainLayer } from '../../src/terrain/TerrainLayer';
import type { TerrainProvider, TerrainMeshData, TerrainTilingScheme } from '../../src/terrain/TerrainProvider';
import { Cartographic } from '../../src/math/Cartographic';
import { sampleTerrain, sampleTerrainMostDetailed } from '../../src/terrain/sampleTerrain';

class RecordingProvider implements TerrainProvider {
  readonly ready = true;
  readonly maximumLevel: number;
  readonly tilingScheme: TerrainTilingScheme = 'geographic';
  readonly requests: string[] = [];

  constructor(maximumLevel = 3) {
    this.maximumLevel = maximumLevel;
  }

  async requestTileGeometry(z: number, x: number, y: number): Promise<TerrainMeshData | null> {
    this.requests.push(`${z}/${x}/${y}`);
    return null; // 无数据：只验证请求集合，不构造网格
  }

  getTileDataAvailable(): boolean {
    return true;
  }

  destroy(): void { /* nothing */ }
}

function makeLayer(provider: TerrainProvider) {
  const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
  return { layer: new TerrainLayer({ provider }), warn };
}

describe('sampleTerrainMostDetailed', () => {
  it('预载包络矩形覆盖全部点的瓦片并回填高度（新数组、不改入参）', async () => {
    const provider = new RecordingProvider(3);
    const { layer, warn } = makeLayer(provider);
    let call = 0;
    const heightsByCall = [111, 222];
    const sample = vi
      .spyOn(layer, 'sampleHeight')
      .mockImplementation(() => heightsByCall[call++] ?? 0);

    const input = [
      Cartographic.fromDegrees(10, 20),
      Cartographic.fromDegrees(11, 21),
    ];
    const inputSnapshot = input.map((p) => p.height);
    const result = await sampleTerrainMostDetailed(layer, input);

    // 预载范围：z3（provider 最高层），bounds (10..11°E, 20..21°N)
    // geographic z3 cols=16 → x = floor((lon+180)/22.5) = 8；rows=8 →
    // y = floor((lat+90)/22.5) = 4（TMS 南起）。两格点同瓦片 → 恰好 1 块。
    expect(provider.requests).toEqual(['3/8/4']);
    // 高度回填（sampleTerrainMostDetailed 用 provider 最高层采样；
    // sampleTerrain 内部把弧度转成度传给 sampleHeight）。
    expect(sample).toHaveBeenCalledWith(10, 20, 3);
    expect(result.map((p) => p.height)).toEqual([111, 222]);
    // 新数组：入参对象未被改写。
    expect(input.map((p) => p.height)).toEqual(inputSnapshot);
    expect(result[0]).not.toBe(input[0]);
    expect(result[0].longitude).toBeCloseTo(input[0].longitude, 12);
    warn.mockRestore();
  });

  it('无地形数据时回退 height=0（椭球短路，同 waypointEditor）', async () => {
    const provider = new RecordingProvider(3);
    const { layer, warn } = makeLayer(provider); // 真实 sampleHeight：无网格 → 0
    const result = await sampleTerrainMostDetailed(layer, [Cartographic.fromDegrees(5, 5)]);
    expect(result).toHaveLength(1);
    expect(result[0].height).toBe(0);
    warn.mockRestore();
  });

  it('空数组 → 空 Promise，不发起任何请求', async () => {
    const provider = new RecordingProvider(3);
    const { layer } = makeLayer(provider);
    const result = await sampleTerrainMostDetailed(layer, []);
    expect(result).toEqual([]);
    expect(provider.requests).toEqual([]);
  });
});

describe('sampleTerrain（level 钉定）', () => {
  it('按显式 level 预载与采样', async () => {
    const provider = new RecordingProvider(5);
    const { layer, warn } = makeLayer(provider);
    const sample = vi.spyOn(layer, 'sampleHeight').mockReturnValue(42);
    const result = await sampleTerrain(layer, [Cartographic.fromDegrees(10, 20)], 2);
    expect(sample).toHaveBeenCalledWith(10, 20, 2);
    expect(result[0].height).toBe(42);
    warn.mockRestore();
  });

  it('单位约定：弧度入参 → 度采样，返回高度为米（湖北 110.67°E/31.74°N）', async () => {

    const provider = new RecordingProvider(5);
    const { layer, warn } = makeLayer(provider);
    // 引擎 sampleHeight 返回米（神农架山区数百~三千米），sampleTerrain 须原样透传，
    // 不得按 Cartographic.height 归一化约定（1 = EARTH_SCALE 米）换算。
    const ensure = vi.spyOn(layer, 'ensureRegion').mockResolvedValue(undefined);
    const sample = vi.spyOn(layer, 'sampleHeight').mockReturnValue(1723.5);
    const result = await sampleTerrain(
      layer,
      [Cartographic.fromDegrees(110.67, 31.74)],
      5
    );
    // 预载 bounds 与采样入参均为度，而非弧度（0.1745…/0.3490… 那种值）。
    expect(ensure).toHaveBeenCalledWith(
      { west: 110.67, south: 31.74, east: 110.67, north: 31.74 },
      5
    );
    expect(sample).toHaveBeenCalledWith(110.67, 31.74, 5);
    expect(result[0].height).toBeCloseTo(1723.5, 6);
    expect(result[0].longitude).toBeCloseTo((110.67 * Math.PI) / 180, 12);
    expect(result[0].latitude).toBeCloseTo((31.74 * Math.PI) / 180, 12);
    ensure.mockRestore();
    sample.mockRestore();
    warn.mockRestore();
  });
});
