import { describe, expect, it } from 'vitest';
import { EllipsoidGeometry } from '../../src/scene/EllipsoidGeometry';

const PARTS = 8; // 与 Globe.TILE_PARTITIONS 一致

/** 平铺顶点布局：[pos(3), normal(3), uv(2)]，步长 8。 */
const STRIDE = 8;

/** 裙边顶点区起点与每条边的偏移（生成顺序：北、南、西、东，按边遍历序复制）。 */
const skirtLayout = (parts: number) => {
  const base = (parts + 1) * (parts + 1);
  const perCol = parts + 1; // 北/南缘各有 slices+1 个副本
  return {
    base,
    northStart: base,
    southStart: base + perCol,
    westStart: base + 2 * perCol,
    eastStart: base + 3 * perCol,
  };
};

/** 面法向 (b-a)×(c-a) 与参考向量 r 的点积。 */
const faceDot = (
  g: ReturnType<typeof EllipsoidGeometry.createRegion>,
  t: number,
  r: [number, number, number]
): number => {
  const p = (i: number): [number, number, number] => [
    g.vertices[i * STRIDE],
    g.vertices[i * STRIDE + 1],
    g.vertices[i * STRIDE + 2],
  ];
  const [ax, ay, az] = p(g.indices[t]);
  const [bx, by, bz] = p(g.indices[t + 1]);
  const [cx, cy, cz] = p(g.indices[t + 2]);
  const nx = (by - ay) * (cz - az) - (bz - az) * (cy - ay);
  const ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az);
  const nz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  return nx * r[0] + ny * r[1] + nz * r[2];
};

/**
 * 影像瓦片裙边（CesiumJS skirt 语义）：
 * 跨 level 相邻瓦片共边时，粗层边缘是弦线、细层中点顶点浮在弧上，楔形缝在
 * 掠射角下透出太空背景成黑线。skirtDepth>0 时四条边各复制一份顶点、半径乘
 * (1-skirtDepth) 向地心压，连成向下垂的裙边面从外侧遮缝。
 */
describe('tile skirt: T-junction 遮盖几何', () => {
  it('skirtDepth 缺省：顶点/索引数与现状一致（回归）', () => {
    const g = EllipsoidGeometry.createRegion(-Math.PI / 4, 0, 0, Math.PI / 4, PARTS);
    expect(g.vertexCount).toBe((PARTS + 1) * (PARTS + 1));
    expect(g.indexCount).toBe(PARTS * PARTS * 6);
  });

  it('skirtDepth=0.005：新增 4 条边环顶点（角顶点按边重复），位置 = 表面 × (1-depth)', () => {
    const depth = 0.005;
    const g = EllipsoidGeometry.createRegion(-Math.PI / 4, 0.2, 0, 0.8, PARTS, {
      skirtDepth: depth,
    });
    const { base, northStart, southStart, westStart, eastStart } = skirtLayout(PARTS);
    // 2*(slices+1) + 2*(stacks+1) 个副本（不去重）
    expect(g.vertexCount).toBe(base + 4 * (PARTS + 1));

    // 每条边：裙边顶点与源边缘顶点逐分量比较（float32 精度放宽到 1e-6）
    const checkEdge = (sourceOf: (i: number) => number, skirtStart: number): void => {
      for (let i = 0; i <= PARTS; i++) {
        const v = sourceOf(i) * STRIDE;
        const s = (skirtStart + i) * STRIDE;
        for (let f = 0; f < 3; f++) {
          // 位置成 (1-skirtDepth) 比例（同 lat/lon、半径向地心压）
          expect(g.vertices[s + f]).toBeCloseTo(g.vertices[v + f] * (1 - depth), 6);
        }
        for (let f = 3; f < STRIDE; f++) {
          // 法线沿用表面法线、uv 沿用边缘值
          expect(g.vertices[s + f]).toBeCloseTo(g.vertices[v + f], 12);
        }
      }
    };
    const stride = PARTS + 1;
    checkEdge((i) => i, northStart);                     // 北缘 stack 0，西→东
    checkEdge((i) => PARTS * stride + i, southStart);    // 南缘，西→东
    checkEdge((i) => i * stride, westStart);             // 西缘 slice 0，北→南
    checkEdge((i) => i * stride + PARTS, eastStart);     // 东缘，北→南
  });

  it('裙边三角形数 = 2*(2*slices + 2*stacks)，索引全部有效', () => {
    const g = EllipsoidGeometry.createRegion(-Math.PI / 4, 0.2, 0, 0.8, PARTS, {
      skirtDepth: 0.005,
    });
    const baseIndexCount = PARTS * PARTS * 6;
    const skirtTriangles = 2 * (2 * PARTS + 2 * PARTS);
    expect(g.indexCount).toBe(baseIndexCount + 3 * skirtTriangles);
    for (let i = 0; i < g.indexCount; i++) {
      expect(g.indices[i]).toBeGreaterThanOrEqual(0);
      expect(g.indices[i]).toBeLessThan(g.vertexCount);
    }
  });

  it('winding 朝外：表面法向离地心、裙边法向背离瓦片中心（南北半球各一）', () => {
    for (const south of [0.2, -0.8]) {
      const g = EllipsoidGeometry.createRegion(-Math.PI / 4, south, 0, south + 0.6, PARTS, {
        skirtDepth: 0.005,
      });
      const { base } = skirtLayout(PARTS);
      // 瓦片矩形中点的表面位置（裙边面的"背离"参考点）。
      const latMid = south + 0.3;
      const lonMid = -Math.PI / 8;
      const tileCenter: [number, number, number] = [
        Math.cos(latMid) * Math.cos(lonMid),
        Math.cos(latMid) * Math.sin(lonMid),
        Math.sin(latMid),
      ];
      for (let t = 0; t < g.indexCount; t += 3) {
        const isSkirt =
          g.indices[t] >= base || g.indices[t + 1] >= base || g.indices[t + 2] >= base;
        if (isSkirt) {
          // 裙边面法向是水平向（背离瓦片内部），用对瓦片中心的点积判定。
          const p = (i: number): [number, number, number] => [
            g.vertices[i * STRIDE],
            g.vertices[i * STRIDE + 1],
            g.vertices[i * STRIDE + 2],
          ];
          const rel: [number, number, number] = [
            (p(g.indices[t])[0] + p(g.indices[t + 1])[0] + p(g.indices[t + 2])[0]) / 3 - tileCenter[0],
            (p(g.indices[t])[1] + p(g.indices[t + 1])[1] + p(g.indices[t + 2])[1]) / 3 - tileCenter[1],
            (p(g.indices[t])[2] + p(g.indices[t + 1])[2] + p(g.indices[t + 2])[2]) / 3 - tileCenter[2],
          ];
          expect(faceDot(g, t, rel)).toBeGreaterThan(0);
        } else {
          // 表面三角形法向离地心（与 (a,b,d)/(b,c,d) 表面约定一致）。
          const centroid = p3(g, g.indices[t], g.indices[t + 1], g.indices[t + 2]);
          expect(faceDot(g, t, centroid)).toBeGreaterThan(0);
        }
      }
    }
  });
});

/** 三个顶点位置之和（作径向参考向量，符号等价于质心方向）。 */
const p3 = (
  g: ReturnType<typeof EllipsoidGeometry.createRegion>,
  a: number,
  b: number,
  c: number
): [number, number, number] => [
  g.vertices[a * STRIDE] + g.vertices[b * STRIDE] + g.vertices[c * STRIDE],
  g.vertices[a * STRIDE + 1] + g.vertices[b * STRIDE + 1] + g.vertices[c * STRIDE + 1],
  g.vertices[a * STRIDE + 2] + g.vertices[b * STRIDE + 2] + g.vertices[c * STRIDE + 2],
];
