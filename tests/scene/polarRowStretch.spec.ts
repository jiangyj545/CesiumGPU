import { describe, expect, it } from 'vitest';
import { EllipsoidGeometry } from '../../src/scene/EllipsoidGeometry';
import { TileKey } from '../../src/imagery/TileKey';

/**
 * 极行拉伸（CesiumJS WebMercatorTilingScheme 语义）取代独立极冠贴片：
 * 每级顶行矩形北缘拉伸到 +90°、底行南缘拉伸到 -90°，极区表面与相邻行
 * 是同一连续网格——不存在悬浮的 cap mesh，85.05° 纬圈处的 cap/tile
 * z-fighting 黑圈在结构上不可能出现。此处锁定该行为。
 */
describe('polar row stretch (CesiumJS semantics)', () => {
  it('顶/底行 rectangle 拉伸到极点，中间行保持 Mercator 边界', () => {
    const top = new TileKey(2, 5, 0).rectangle();
    expect(top.north).toBeCloseTo(Math.PI / 2, 10);
    // 南缘仍是该行 Mercator 边界（L2 第 0 行 = atan(sinh(π/2)) ≈ 66.51°）
    expect(top.south).toBeCloseTo(Math.atan(Math.sinh(Math.PI / 2)), 10);

    const bottom = new TileKey(2, 5, 3).rectangle();
    expect(bottom.south).toBeCloseTo(-Math.PI / 2, 10);
    expect(bottom.north).toBeCloseTo(-Math.atan(Math.sinh(Math.PI / 2)), 10);

    const mid = new TileKey(2, 5, 1).rectangle();
    expect(mid.north).toBeCloseTo(Math.atan(Math.sinh(Math.PI / 2)), 10);
    expect(mid.south).toBeCloseTo(Math.atan(Math.sinh(0)), 10); // 赤道
  });

  it('极行与相邻行共享纬圈边界（无几何缝隙）', () => {
    const top = new TileKey(2, 0, 0).rectangle();
    const next = new TileKey(2, 0, 1).rectangle();
    expect(top.south).toBeCloseTo(next.north, 10);

    const bottom = new TileKey(2, 0, 3).rectangle();
    const above = new TileKey(2, 0, 2).rectangle();
    expect(bottom.north).toBeCloseTo(above.south, 10);
  });

  it('mercatorV：极点纬度钳制进 Mercator 域，v 单调且无 NaN（边缘行影像拉伸铺满极区）', () => {
    const rect = new TileKey(2, 0, 0).rectangle();
    const P = 8;
    const g = EllipsoidGeometry.createRegion(
      rect.west, rect.south, rect.east, rect.north, P, { mercatorV: true },
    );

    // 顶点布局 [pos(3), normal(3), uv(2)]，stride 8；v 在 offset 7。
    // 每行首个顶点（slice 0）代表该 stack 的 v。
    let prev = -Infinity;
    for (let stack = 0; stack <= P; stack++) {
      const v = g.vertices[stack * (P + 1) * 8 + 7];
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0.0 - 1e-9);
      expect(v).toBeLessThanOrEqual(1.0 + 1e-9);
      // stack 0 = 北缘（极点）：钳制到域边 → v=0（整行采样纹理最北像素）
      if (stack === 0) expect(v).toBeCloseTo(0.0, 10);
      // v 从北到南单调不减：85.05°→90° 段全部钳制到 v=0（CesiumJS 式
      // 边缘行 smear，相邻 stack 允许相等），Mercator 域内严格递增（无折叠）
      expect(v).toBeGreaterThanOrEqual(prev);
      prev = v;
    }
    // 南缘（66.51°）v=1：与相邻行的 v=0 精确相接
    const lastRow = P * (P + 1) * 8 + 7;
    expect(g.vertices[lastRow]).toBeCloseTo(1.0, 10);
  });

  it('全拉伸行（南北缘均越过域边）退化为纯边缘采样，不产生 NaN', () => {
    const g = EllipsoidGeometry.createRegion(-Math.PI / 2, 1.55, 0, Math.PI / 2, 4, {
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
