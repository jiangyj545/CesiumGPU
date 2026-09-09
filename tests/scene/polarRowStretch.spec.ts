import { describe, expect, it } from 'vitest';
import { EllipsoidGeometry } from '../../src/scene/EllipsoidGeometry';
import { TileKey } from '../../src/imagery/TileKey';

const MERC_EDGE = Math.atan(Math.sinh(Math.PI)); // ±85.0511°

/**
 * 极区架构（CesiumJS 语义对齐）：
 * 1. Web Mercator 瓦片矩形止步于 ±85.0511°（CesiumJS WebMercatorTilingScheme
 *    同款——tileXYToNativeRectangle 无极行特判，北极不在瓦片网格内）；
 * 2. 极冠由 Globe._syncPolarFills 的平面冰白填充段覆盖（CesiumJS 对无影像
 *    区域的 fill 语义）——顶行瓦片顶部黑边不再被 smear 成黑色极冠 + 85.05°
 *    黑圈；
 * 3. mercatorV 采样对越界纬度钳制进 Mercator 域，不产生 NaN。
 */
describe('polar region: mercator bounds + mercatorV domain clamp', () => {
  it('顶/底行 rectangle 止步于 Mercator 边界（极区不属于瓦片网格）', () => {
    const top = new TileKey(2, 5, 0).rectangle();
    expect(top.north).toBeCloseTo(MERC_EDGE, 10);
    expect(top.south).toBeCloseTo(Math.atan(Math.sinh(Math.PI / 2)), 10);

    const bottom = new TileKey(2, 5, 3).rectangle();
    expect(bottom.south).toBeCloseTo(-MERC_EDGE, 10);
    expect(bottom.north).toBeCloseTo(-Math.atan(Math.sinh(Math.PI / 2)), 10);

    const mid = new TileKey(2, 5, 1).rectangle();
    expect(mid.north).toBeCloseTo(Math.atan(Math.sinh(Math.PI / 2)), 10);
    expect(mid.south).toBeCloseTo(Math.atan(Math.sinh(0)), 10); // 赤道
  });

  it('相邻行共享纬圈边界（无几何缝隙）', () => {
    const top = new TileKey(2, 0, 0).rectangle();
    const next = new TileKey(2, 0, 1).rectangle();
    expect(top.south).toBeCloseTo(next.north, 10);

    const bottom = new TileKey(2, 0, 3).rectangle();
    const above = new TileKey(2, 0, 2).rectangle();
    expect(bottom.north).toBeCloseTo(above.south, 10);
  });

  it('mercatorV：越界纬度钳制进 Mercator 域，v 有界且无 NaN', () => {
    const g = EllipsoidGeometry.createRegion(-Math.PI / 2, MERC_EDGE, 0, Math.PI / 2, 4, {
      mercatorV: true,
      uvRect: { u0: 0, v0: 0.25, u1: 0.5, v1: 0.5 },
    });
    expect(g.vertexCount).toBe(5 * 5);
    for (let i = 0; i < g.vertexCount; i++) {
      const u = g.vertices[i * 8 + 6];
      const v = g.vertices[i * 8 + 7];
      expect(Number.isFinite(u)).toBe(true);
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0.25 - 1e-9);
      expect(v).toBeLessThanOrEqual(0.5 + 1e-9);
    }
  });
});
