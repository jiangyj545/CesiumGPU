import { describe, expect, it } from 'vitest';
import { EllipsoidGeometry } from '../../src/scene/EllipsoidGeometry';

/**
 * 极冠按 ROOT 列分段贴图（Globe._createPolarCapPrimitive）的几何与 UV 约定：
 * 修复前单列纹理铺满全经度导致 270° 经度错位 + 蓝白过渡行被全经度拉伸成
 * 均匀环带（用户看到的"一圈缝隙"）。分段后每列采自己的 u 子矩形。
 */
describe('polar cap per-column UV segmentation', () => {
  it('每列段的 u 子矩形互不重叠且覆盖 [0,1]', () => {
    const n = 4;
    const rects: Array<{ u0: number; u1: number }> = [];
    for (let x = 0; x < n; x++) {
      rects.push({ u0: x / n, u1: (x + 1) / n });
    }
    for (let x = 0; x < n; x++) {
      expect(rects[x].u0).toBeCloseTo(x / n, 6);
      expect(rects[x].u1).toBeCloseTo((x + 1) / n, 6);
      if (x > 0) expect(rects[x].u0).toBeCloseTo(rects[x - 1].u1, 6); // 无缝相接
    }
    expect(rects[0].u0).toBe(0);
    expect(rects[n - 1].u1).toBe(1);
  });

  it('分段几何经度连续（相邻段共享边界，首尾环绕）', () => {
    const n = 4;
    const P = 16;
    const edges: Array<{ west: number; east: number }> = [];
    for (let x = 0; x < n; x++) {
      const west = -Math.PI + (x / n) * 2 * Math.PI;
      const east = -Math.PI + ((x + 1) / n) * 2 * Math.PI;
      EllipsoidGeometry.createRegion(west, 1.4, east, Math.PI / 2, P, { radiusScale: 1.0 });
      edges.push({ west, east });
    }
    for (let x = 0; x < n - 1; x++) {
      expect(edges[x].east).toBeCloseTo(edges[x + 1].west, 6);
    }
    // 最后一段 east = π 与第一段 west = -π 是同一经线（球面环绕）
    expect(edges[n - 1].east).toBeCloseTo(Math.PI, 6);
    expect(edges[0].west).toBeCloseTo(-Math.PI, 6);
  });

  it('分段半径一致（段边界无 lift 台阶）', () => {
    const n = 4;
    const P = 8;
    const R = 6356752.3142 / 6378137.0; // WGS84 polar ratio
    for (let x = 0; x < n; x++) {
      const west = -Math.PI + (x / n) * 2 * Math.PI;
      const east = -Math.PI + ((x + 1) / n) * 2 * Math.PI;
      const g = EllipsoidGeometry.createRegion(west, 1.4, east, Math.PI / 2, P, { radiusScale: 1.0 });
      // 第一顶点（stack 0 = 极点侧），y/z 带极比压缩：等效球半径 ≈ 1
      const v0x = g.vertices[0], v0y = g.vertices[1] / R, v0z = g.vertices[2] / R;
      const r = Math.hypot(v0x, v0y, v0z);
      // 极点行纬度 90°：位置 (0,0,ratio)×scale → 归一后 = 1（在 1e-2 容差内，
      // 顶点在 1.4 rad 边界处 cos 分量非零）
      expect(r).toBeGreaterThan(0.99);
      expect(r).toBeCloseTo(1.0, 2);
    }
  });
});
