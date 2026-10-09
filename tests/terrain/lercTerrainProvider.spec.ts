/**
 * lercTerrainProvider.spec.ts – P2-8 provider 层：
 * 默认 URL 模板（y/x 顺序对齐 ArcGIS）、headers 透传（P1-F 模式）、
 * stub fetch + 测试编码器合成瓦片 → decodeTile → TerrainMeshData 网格
 * 语义（瓦片边界 / 行主序 / 高程还原）。不 mock 解码器。
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Camera } from '../../src/scene/Camera';
import { Cartesian3 } from '../../src/math/Cartesian3';
import {
  LercTerrainProvider,
  DEFAULT_LERC_URL,
  webmercatorTileOfLonLat,
  lercTileBounds,
} from '../../src/terrain/LercTerrainProvider';
import { encodeLerc1TestTile } from '../../src/terrain/LercDecoder';

const AUTH = { Authorization: 'Bearer lerc-token' };

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('LercTerrainProvider', () => {
  it('默认 URL 为 ArcGIS Terrain3D 模板（tile/{z}/{y}/{x}，y 在前）', () => {
    expect(DEFAULT_LERC_URL).toContain('elevation3d.arcgis.com');
    expect(DEFAULT_LERC_URL).toContain('WorldElevation3D/Terrain3D/ImageServer');
    const p = new LercTerrainProvider();
    expect(p.tilingScheme).toBe('webmercator');
    expect(p.maximumLevel).toBe(15);
    expect(p.headers).toBeUndefined();
  });

  it('瓦片 URL 按模板解析（y/x 顺序）且 headers 逐请求透传', async () => {
    const calls: Array<{ url: string; init: RequestInit | undefined }> = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string | URL, init?: RequestInit) => {
        calls.push({ url: String(url), init });
        return new Response(encodeLerc1TestTile({ width: 2, height: 2, constant: 5 }), {
          status: 200,
        });
      })
    );
    const p = new LercTerrainProvider({ headers: AUTH });
    const mesh = await p.requestTileGeometry(3, 2, 1);
    expect(mesh).not.toBeNull();
    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe(
      DEFAULT_LERC_URL.replace('{z}', '3').replace('{y}', '1').replace('{x}', '2')
    );
    expect(calls[0].init?.headers).toEqual(AUTH);
    p.destroy();
  });

  it('合成 LERC 瓦片 → 网格语义：顶点/索引数 / 高程归一化半径', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(
            encodeLerc1TestTile({
              width: 4,
              height: 4,
              value: (col, row) => 100 + col * 10 + row, // 100..133
            }),
            { status: 200 }
          )
      )
    );
    const p = new LercTerrainProvider();
    const mesh = await p.requestTileGeometry(1, 1, 1);
    expect(mesh).not.toBeNull();
    // 4×4 网格 → 16 顶点、(4-1)²×6 = 54 索引。
    expect(mesh!.positions.length).toBe(16 * 3);
    expect(mesh!.indices.length).toBe(3 * 3 * 6);
    expect(mesh!.minHeight).toBeCloseTo(100, 3);
    expect(mesh!.maxHeight).toBeCloseTo(133, 3);
    // 高程折算：|pos| − 引擎地表矢长（Camera.surfaceRadius，压扁系）
    // ≈ h/6378137（全部顶点，任一纬度成立）。
    for (let i = 0; i < 16; i++) {
      const p3 = new Cartesian3(
        mesh!.positions[i * 3],
        mesh!.positions[i * 3 + 1],
        mesh!.positions[i * 3 + 2]
      );
      const h = (Cartesian3.magnitude(p3) - Camera.surfaceRadius(p3)) * 6378137;
      expect(h).toBeGreaterThanOrEqual(99.9);
      expect(h).toBeLessThanOrEqual(133.1);
    }
    p.destroy();
  });

  it('无效瓦片（非 LERC 载荷）→ 失败缓存（不炸树、不重试）', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(new Uint8Array([1, 2, 3]).buffer, { status: 200 }))
    );
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const p = new LercTerrainProvider();
    expect(await p.requestTileGeometry(1, 0, 0)).toBeNull();
    expect(await p.requestTileGeometry(1, 0, 0)).toBeNull(); // 失败缓存：不再请求
    warn.mockRestore();
    p.destroy();
  });
});

describe('webmercator 瓦片映射（某经纬度 → 正确 z/x/y）', () => {
  it('0°/0° 在 z0 → 0/0；武汉（114.39, 30.5）在 z3 → x6 y3 且被瓦片覆盖', () => {
    expect(webmercatorTileOfLonLat(0, 0, 0)).toEqual({ z: 0, x: 0, y: 0 });
    const t = webmercatorTileOfLonLat(114.393, 30.5, 3);
    expect(t).toEqual({ z: 3, x: 6, y: 3 });
    const b = lercTileBounds(t.z, t.x, t.y);
    expect(114.393).toBeGreaterThanOrEqual(b.west);
    expect(114.393).toBeLessThanOrEqual(b.east);
    expect(30.5).toBeGreaterThanOrEqual(b.south);
    expect(30.5).toBeLessThanOrEqual(b.north);
  });

  it('跨日界线与近极点钳制不越界', () => {
    expect(webmercatorTileOfLonLat(179.9, 0, 2).x).toBe(3);
    expect(webmercatorTileOfLonLat(-179.9, 0, 2).x).toBe(0);
    expect(webmercatorTileOfLonLat(0, 89.9, 1).y).toBe(0);
  });
});
