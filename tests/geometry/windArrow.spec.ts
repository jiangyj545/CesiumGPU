import { describe, expect, it } from 'vitest';
import {
  WindArrowGeometry, WIND_ARROW_VERTEX_COUNT, WIND_ARROW_INDEX_COUNT,
} from '../../src/geometry/WindArrowGeometry';

/**
 * WindArrowGeometry 纯数据单测（无 GPU 依赖，cityBlockInstanceIds 同模式）：
 * 单箭头 = 4 面锥杆 + 4 面锥头 + 基座封口，沿局部 +Y 轴，8-float 交错
 * 顶点（pos3+normal3+uv2，Primitive.ts:431 契约）。顶点/索引数只由
 * 拓扑决定（恒 32/42），几何规模只由 opts 缩放。
 */

const STRIDE = 8;

/** 从交错缓冲取出三角形 (三个顶点的 pos/normal)。 */
function* triangles(vertices: Float32Array, indices: Uint32Array) {
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const vs = [0, 1, 2].map((k) => {
      const s = indices[t + k]! * STRIDE;
      return {
        pos: [vertices[s]!, vertices[s + 1]!, vertices[s + 2]!] as const,
        n:   [vertices[s + 3]!, vertices[s + 4]!, vertices[s + 5]!] as const,
      };
    });
    yield vs;
  }
}

/** 三角形几何法向（未归一化）。 */
function geoNormal(tri: { pos: readonly number[] }[]): [number, number, number] {
  const [a, b, c] = tri;
  const e1 = [b.pos[0] - a.pos[0], b.pos[1] - a.pos[1], b.pos[2] - a.pos[2]];
  const e2 = [c.pos[0] - a.pos[0], c.pos[1] - a.pos[1], c.pos[2] - a.pos[2]];
  return [
    e1[1] * e2[2] - e1[2] * e2[1],
    e1[2] * e2[0] - e1[0] * e2[2],
    e1[0] * e2[1] - e1[1] * e2[0],
  ];
}

describe('WindArrowGeometry', () => {
  it('顶点数/索引数恒定（与 lengthM、自定义比例无关）', () => {
    for (const opts of [
      { lengthM: 8 },
      { lengthM: 20 },
      { lengthM: 45 },
      { lengthM: 30, shaftRadiusM: 2, headRadiusM: 4, headLenM: 9 },
    ]) {
      const g = WindArrowGeometry.create(opts);
      expect(g.vertexCount).toBe(WIND_ARROW_VERTEX_COUNT);
      expect(g.vertexCount).toBe(32);
      expect(g.indexCount).toBe(WIND_ARROW_INDEX_COUNT);
      expect(g.indexCount).toBe(42);
      expect(g.vertices.length).toBe(32 * STRIDE);
      expect(g.indices.length).toBe(42);
    }
  });

  it('法线全部归一化，uv 全部在 [0,1]', () => {
    const g = WindArrowGeometry.create({ lengthM: 25 });
    for (let v = 0; v < g.vertexCount; v++) {
      const s = v * STRIDE;
      const nl = Math.hypot(g.vertices[s + 3]!, g.vertices[s + 4]!, g.vertices[s + 5]!);
      expect(nl).toBeCloseTo(1, 6);
      expect(g.vertices[s + 6]!).toBeGreaterThanOrEqual(0);
      expect(g.vertices[s + 6]!).toBeLessThanOrEqual(1);
      expect(g.vertices[s + 7]!).toBeGreaterThanOrEqual(0);
      expect(g.vertices[s + 7]!).toBeLessThanOrEqual(1);
    }
  });

  it('沿 +Y 轴：y ∈ [0, lengthM]，尖端在 lengthM、基座在 0，横向不出锥头半径', () => {
    const L = 40, headR = 0.12 * L;
    const g = WindArrowGeometry.create({ lengthM: L });
    let minY = Infinity, maxY = -Infinity, maxR = 0;
    let hasTip = false, hasBase = false;
    for (let v = 0; v < g.vertexCount; v++) {
      const s = v * STRIDE;
      const x = g.vertices[s]!, y = g.vertices[s + 1]!, z = g.vertices[s + 2]!;
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      maxR = Math.max(maxR, Math.hypot(x, z));
      if (Math.abs(y - L) < 1e-4) hasTip = true;
      if (y === 0) hasBase = true;
    }
    expect(minY).toBe(0);
    expect(maxY).toBeCloseTo(L, 5);
    expect(hasTip).toBe(true);
    expect(hasBase).toBe(true);
    expect(maxR).toBeLessThanOrEqual(headR * (1 + 1e-6));
  });

  it('winding：锥杆/锥头三角形法向朝外（离心分量 > 0），基座封口法向 = −Y', () => {
    const L = 30;
    const g = WindArrowGeometry.create({ lengthM: L });
    let sideTris = 0, headTris = 0, capTris = 0;
    for (const tri of triangles(g.vertices, g.indices)) {
      const n = geoNormal(tri);
      const nl = Math.hypot(n[0], n[1], n[2]);
      expect(nl).toBeGreaterThan(0);
      const cy = (tri[0].pos[1] + tri[1].pos[1] + tri[2].pos[1]) / 3;
      if (cy < 1e-9 && Math.abs(n[1]) > 0.99 * nl && n[1] < 0) {
        capTris++;   // 基座封口：全部顶点 y=0、法向 −Y
        continue;
      }
      const cx = (tri[0].pos[0] + tri[1].pos[0] + tri[2].pos[0]) / 3;
      const cz = (tri[0].pos[2] + tri[1].pos[2] + tri[2].pos[2]) / 3;
      expect(cx * n[0] + cz * n[2]).toBeGreaterThan(0);   // 外向（远离 Y 轴）
      if (cy > L * 0.69) headTris++; else sideTris++;
    }
    expect(sideTris).toBe(8);   // 4 面 × 2 三角
    expect(headTris).toBe(4);
    expect(capTris).toBe(2);
  });

  it('退化输入（≤0 / 非有限长度）→ 空缓冲（emptyBuffers 约定）', () => {
    for (const L of [0, -5, NaN, Infinity]) {
      const g = WindArrowGeometry.create({ lengthM: L });
      expect(g.vertexCount).toBe(0);
      expect(g.indexCount).toBe(0);
      expect(g.vertices.length).toBe(0);
    }
  });
});
