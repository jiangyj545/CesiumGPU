import { describe, expect, it } from 'vitest';
import { EllipsoidGeometry } from '../../src/scene/EllipsoidGeometry';

/**
 * edgeLift 接缝连续性（CreateRegionOptions.edgeLift）：
 * 粗/细瓦片共边的纬圈两侧必须同半径，否则 (1.02 − 1.008) ≈ 76 km 径向台阶
 * 在掠射视角下读作沿纬圈的缝隙环。
 *
 * 机制：细瓦片把北/南边缘行钉到粗邻居的 lift（Math.max 保护，不会低于自身
 * lift），在 edgeLiftRamp 个 stack 内线性渐回自身 lift。极冠恒 1.02，顶/底
 * 行瓦片边界因此无缝；L3/L4 层界同理。
 *
 * 纯几何测试：直接对 createRegion 顶点做断言（无 GPU 依赖）。
 */

const R = 6356752.3142 / 6378137.0; // WGS84 polar ratio

/** 顶点数组中取 (stack, slice) 的 xyz。 */
function vertexAt(
  vertices: Float32Array,
  partitions: number,
  stack: number,
  slice: number
): [number, number, number] {
  const i = (stack * (partitions + 1) + slice) * 8;
  return [vertices[i], vertices[i + 1], vertices[i + 2]];
}

/** 顶点的径向缩放因子（消去 polar ratio 后的等效球半径）。 */
function radialOf(v: [number, number, number]): number {
  return Math.hypot(v[0], v[1] / R, v[2] / R);
}

describe('createRegion edgeLift seam continuity', () => {
  const P = 8;

  it('无 edgeLift 时行为不变（所有行同半径）', () => {
    const g = EllipsoidGeometry.createRegion(0, 0, 0.1, 0.1, P, { radiusScale: 1.008, mercatorV: true });
    const r0 = radialOf(vertexAt(g.vertices, P, 0, 0));
    const rMid = radialOf(vertexAt(g.vertices, P, 4, 0));
    const rN = radialOf(vertexAt(g.vertices, P, P, 0));
    expect(r0).toBeCloseTo(1.008, 5);
    expect(rMid).toBeCloseTo(1.008, 5);
    expect(rN).toBeCloseTo(1.008, 5);
  });

  it('north edgeLift 钉住北边缘行到粗 lift 并渐回自身 lift', () => {
    const g = EllipsoidGeometry.createRegion(0, 0, 0.1, 0.1, P, {
      radiusScale: 1.008,
      edgeLift: { north: 1.02 },
      mercatorV: true,
    });
    const rNorth = radialOf(vertexAt(g.vertices, P, 0, 0));
    const rSouth = radialOf(vertexAt(g.vertices, P, P, 0));
    expect(rNorth).toBeCloseTo(1.02, 5); // 钉到邻居
    expect(rSouth).toBeCloseTo(1.008, 5); // 对侧不受影响
    // ramp 默认 2：stack 2 已回到自身 lift，stack 1 在两者之间
    const r1 = radialOf(vertexAt(g.vertices, P, 1, 0));
    const r2 = radialOf(vertexAt(g.vertices, P, 2, 0));
    expect(r2).toBeCloseTo(1.008, 5);
    expect(r1).toBeGreaterThan(1.008);
    expect(r1).toBeLessThan(1.02);
  });

  it('south edgeLift 钉住南边缘行', () => {
    const g = EllipsoidGeometry.createRegion(0, 0, 0.1, 0.1, P, {
      radiusScale: 1.008,
      edgeLift: { south: 1.02 },
      mercatorV: true,
    });
    const rSouth = radialOf(vertexAt(g.vertices, P, P, 0));
    const rNorth = radialOf(vertexAt(g.vertices, P, 0, 0));
    expect(rSouth).toBeCloseTo(1.02, 5);
    expect(rNorth).toBeCloseTo(1.008, 5);
  });

  it('edgeLift 低于自身 lift 时被 max() 忽略（不压低）', () => {
    const g = EllipsoidGeometry.createRegion(0, 0, 0.1, 0.1, P, {
      radiusScale: 1.02, // 粗瓦片
      edgeLift: { north: 1.008 }, // 更细的邻居
      mercatorV: true,
    });
    const rNorth = radialOf(vertexAt(g.vertices, P, 0, 0));
    expect(rNorth).toBeCloseTo(1.02, 5);
  });

  it('共边连续性：粗瓦片南行 == 细瓦片钉边北行（同纬圈同半径）', () => {
    // 粗 L3 瓦片（自身 1.02）与南侧细 L4 瓦片（1.008 + edgeLift.north=1.02）
    const coarse = EllipsoidGeometry.createRegion(0, 0, 0.1, 0.1, P, {
      radiusScale: 1.02, mercatorV: true,
    });
    const fine = EllipsoidGeometry.createRegion(0, 0, 0.1, 0.1, P, {
      radiusScale: 1.008, edgeLift: { north: 1.02 }, mercatorV: true,
    });
    const rCoarseSouth = radialOf(vertexAt(coarse.vertices, P, P, 0));
    const rFineNorth = radialOf(vertexAt(fine.vertices, P, 0, 0));
    expect(rFineNorth).toBeCloseTo(rCoarseSouth, 6); // 接缝两侧同半径
  });
});
