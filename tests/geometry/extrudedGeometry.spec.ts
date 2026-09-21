import { describe, expect, it } from 'vitest';
import { ExtrudedGeometry } from '../../src/geometry/ExtrudedGeometry';
import type { ExtrudedGeometryBuffers, LonLatPoint } from '../../src/geometry/ExtrudedGeometry';
import { CityBlockBuilder } from '../../src/geometry/CityBlockBuilder';
import { GeometryPipeline } from '../../src/geometry/GeometryPipeline';

/**
 * 建筑拉伸体（白模）几何与合批构建器回归测试。
 *
 * 锁定 F2b 的四个关键不变量（纯函数路径，node 无 WebGPU 全局，无需 mock）：
 * 1. 顶点/索引数公式：侧面每边独立 4 顶点 + 顶面 n 顶点（底面省略）；
 * 2. 法向朝外：dot(顶点法向, 椭球径向) >= 0（小 footprint 下径向即外向）；
 * 3. winding 与存储法向一致：三角形几何法向 (b-a)×(c-a) 与顶点法向同向
 *    （Primitive frontFace ccw + cullMode back，见 polarFillWinding 历史坑）；
 * 4. semanticId 透传：单楼填充 / 合批后按楼分段保持。
 */

// 武汉光谷附近合成 footprint：约 100m × 100m 方形
const BASE_LON = 114.393;
const BASE_LAT = 30.5;
const M_PER_DEG_LON = 111320 * Math.cos((BASE_LAT * Math.PI) / 180); // ≈ 95916
const M_PER_DEG_LAT = 110574;

/** CCW 方形环：(min,min)→(max,min)→(max,max)→(min,max)，经度向东、纬度向北。 */
function squareRingCCW(sizeM = 100): LonLatPoint[] {
  const dLon = sizeM / 2 / M_PER_DEG_LON;
  const dLat = sizeM / 2 / M_PER_DEG_LAT;
  return [
    { lon: BASE_LON - dLon, lat: BASE_LAT - dLat },
    { lon: BASE_LON + dLon, lat: BASE_LAT - dLat },
    { lon: BASE_LON + dLon, lat: BASE_LAT + dLat },
    { lon: BASE_LON - dLon, lat: BASE_LAT + dLat },
  ];
}

/** L 形六点环（CCW），覆盖多边形顶点数 ≠ 4 的合批场景。 */
function lShapeRingCCW(): LonLatPoint[] {
  const u = 40 / M_PER_DEG_LON;
  const v = 40 / M_PER_DEG_LAT;
  return [
    { lon: BASE_LON,         lat: BASE_LAT },
    { lon: BASE_LON + 2 * u, lat: BASE_LAT },
    { lon: BASE_LON + 2 * u, lat: BASE_LAT + v },
    { lon: BASE_LON + u,     lat: BASE_LAT + v },
    { lon: BASE_LON + u,     lat: BASE_LAT + 2 * v },
    { lon: BASE_LON,         lat: BASE_LAT + 2 * v },
  ];
}

const STRIDE = GeometryPipeline.VERTEX_STRIDE;

/** 读取第 v 个顶点的存储法向。 */
function normalAt(b: ExtrudedGeometryBuffers, v: number): [number, number, number] {
  const o = v * STRIDE + 3;
  return [b.vertices[o], b.vertices[o + 1], b.vertices[o + 2]];
}

/** 读取第 v 个顶点的位置。 */
function positionAt(b: ExtrudedGeometryBuffers, v: number): [number, number, number] {
  const o = v * STRIDE;
  return [b.vertices[o], b.vertices[o + 1], b.vertices[o + 2]];
}

function dot3(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * 全量逐三角形判定：
 * - 存储法向与椭球径向（归一化顶点位置）点积 >= 0（朝外）；
 * - 三角形几何法向 (b-a)×(c-a) 与存储法向点积 > 0（winding 与法向一致，
 *   即正面朝外，可被 Primitive 默认 cullMode:'back' 正确保留）。
 */
function expectAllFacesOutward(b: ExtrudedGeometryBuffers, label: string): void {
  for (let v = 0; v < b.vertexCount; v++) {
    const p = positionAt(b, v);
    const len = Math.hypot(p[0], p[1], p[2]);
    expect(len, `${label} 顶点 ${v} 应在归一化 ECEF（|p|≈1）`).toBeGreaterThan(0.99);
    expect(len).toBeLessThan(1.01);
    const radial = [p[0] / len, p[1] / len, p[2] / len];
    const n = normalAt(b, v);
    // 垂直侧墙与径向的点积随朝向变化：北向墙 ≈ +cos(纬度)，东西向墙
    // 与径向水平分量（指向正北）正交 → 理论值 0；叠加地心纬度与大地
    // 纬度的微倾后实测 |dot| < 1e-5。容差 -1e-4 距内翻面（≈ -0.5）尚有
    // 3 个量级余量，判据不失真；严格朝外性由下方 winding 判定保证。
    expect(dot3(n, radial), `${label} 顶点 ${v} 法向应朝外`).toBeGreaterThanOrEqual(-1e-4);
  }
  let checked = 0;
  for (let t = 0; t + 2 < b.indexCount; t += 3) {
    const i0 = b.indices[t], i1 = b.indices[t + 1], i2 = b.indices[t + 2];
    const a = positionAt(b, i0), c = positionAt(b, i1), d = positionAt(b, i2);
    const e1 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const e2 = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    const fn = [
      e1[1] * e2[2] - e1[2] * e2[1],
      e1[2] * e2[0] - e1[0] * e2[2],
      e1[0] * e2[1] - e1[1] * e2[0],
    ];
    const fnLen = Math.hypot(fn[0], fn[1], fn[2]);
    if (fnLen < 1e-12) continue; // 退化三角形（footprint 极小时数值收敛），无片元
    const n0 = normalAt(b, i0);
    const cos = dot3(fn, n0) / fnLen;
    expect(cos, `${label} 三角形 #${t / 3} winding 与法向不一致 (cos=${cos})`).toBeGreaterThan(0);
    checked++;
  }
  expect(checked, `${label} 应有非退化三角形`).toBeGreaterThan(0);
}

describe('ExtrudedGeometry 顶点/索引数', () => {
  it('方形 4 点环：侧面 4×4 + 顶面 4 = 20 顶点，索引 4×6 + (4-2)×3 = 30', () => {
    const b = ExtrudedGeometry.create({ ring: squareRingCCW(), heightM: 30 });
    expect(b.vertexCount).toBe(20);
    expect(b.vertices.length).toBe(20 * STRIDE);
    expect(b.indexCount).toBe(30);
    expect(b.indices.length).toBe(30);
  });

  it('L 形 6 点环：顶点 6×5 = 30，索引 6×6 + 4×3 = 48', () => {
    const b = ExtrudedGeometry.create({ ring: lShapeRingCCW(), heightM: 20 });
    expect(b.vertexCount).toBe(30);
    expect(b.indexCount).toBe(48);
  });

  it('底面省略：无任何朝下法向顶点', () => {
    const b = ExtrudedGeometry.create({ ring: squareRingCCW(), heightM: 30 });
    expect(b.indexCount / 3).toBe(4 * 2 + (4 - 2));
    for (let v = 0; v < b.vertexCount; v++) {
      const n = normalAt(b, v);
      const p = positionAt(b, v);
      // 底面若存在会有一批法向 ≈ -up 的顶点；侧面/顶面与径向点积均 >= 0
      expect(dot3(n, p)).toBeGreaterThanOrEqual(0);
    }
  });

  it('heightM = 0 退化为纯平顶面：仅 n 顶点 + 3(n-2) 索引', () => {
    const b = ExtrudedGeometry.create({ ring: squareRingCCW(), heightM: 0 });
    expect(b.vertexCount).toBe(4);
    expect(b.indexCount).toBe(6);
  });

  it('退化输入：少于 3 点返回空缓冲', () => {
    const b = ExtrudedGeometry.create({
      ring: [{ lon: BASE_LON, lat: BASE_LAT }, { lon: BASE_LON + 0.001, lat: BASE_LAT }],
      heightM: 10,
    });
    expect(b.vertexCount).toBe(0);
    expect(b.indexCount).toBe(0);
    expect(b.semanticIds.length).toBe(0);
  });

  it('GeoJSON 闭合环（首末重复）自动去重，不产生多余顶点', () => {
    const closed = [...squareRingCCW(), squareRingCCW()[0]];
    const b = ExtrudedGeometry.create({ ring: closed, heightM: 30 });
    expect(b.vertexCount).toBe(20);
    expect(b.indexCount).toBe(30);
  });
});

describe('ExtrudedGeometry 法向朝外与 winding', () => {
  it('CCW 输入：全部三角形法向朝外且 winding 一致', () => {
    expectAllFacesOutward(ExtrudedGeometry.create({ ring: squareRingCCW(), heightM: 30 }), 'CCW 方形');
  });

  it('CW 输入：自动翻转为 CCW，法向同样朝外（顶点序无关性）', () => {
    const cw = squareRingCCW().reverse();
    const b = ExtrudedGeometry.create({ ring: cw, heightM: 30 });
    expect(b.vertexCount).toBe(20); // 仅顶点序变化，数量不变
    expectAllFacesOutward(b, 'CW 方形');
  });

  it('L 形多边形同样朝外', () => {
    expectAllFacesOutward(ExtrudedGeometry.create({ ring: lShapeRingCCW(), heightM: 45 }), 'L 形');
  });

  it('baseHeightM 抬升：顶面位于 base + height，法向判定不受影响', () => {
    expectAllFacesOutward(
      ExtrudedGeometry.create({ ring: squareRingCCW(), heightM: 30, baseHeightM: 50 }),
      '抬升 50m',
    );
    // 顶面顶点（第 16 个起）相对地面抬升约 80m（归一化半径差 × EARTH_SCALE）
    const ground = ExtrudedGeometry.create({ ring: squareRingCCW(), heightM: 0 });
    const rGround = Math.hypot(...positionAt(ground, 0));
    const lifted = ExtrudedGeometry.create({ ring: squareRingCCW(), heightM: 30, baseHeightM: 50 });
    const pTop = positionAt(lifted, 16);
    const liftM = (Math.hypot(pTop[0], pTop[1], pTop[2]) - rGround) * 6378137;
    expect(liftM, '顶面应抬升约 80m').toBeGreaterThan(60);
    expect(liftM).toBeLessThan(100);
  });
});

describe('semanticId 透传', () => {
  it('create 显式 semanticId 填充全部顶点', () => {
    const b = ExtrudedGeometry.create({ ring: squareRingCCW(), heightM: 30, semanticId: 42 });
    expect(b.semanticIds.length).toBe(b.vertexCount);
    for (let i = 0; i < b.semanticIds.length; i++) expect(b.semanticIds[i]).toBe(42);
  });

  it('缺省 semanticId 为 0', () => {
    const b = ExtrudedGeometry.create({ ring: squareRingCCW(), heightM: 30 });
    for (let i = 0; i < b.semanticIds.length; i++) expect(b.semanticIds[i]).toBe(0);
  });
});

describe('CityBlockBuilder 合批', () => {
  it('两栋楼合并为一块：顶点拼接 + 索引偏移连续 + semanticId 分段保持', () => {
    const builder = new CityBlockBuilder();
    builder.addBuilding(squareRingCCW(), 30, 7);
    builder.addBuilding(lShapeRingCCW(), 20, 9);
    const chunks = builder.buildChunks();

    expect(chunks.length).toBe(1);
    const g = chunks[0].geometry;
    expect(chunks[0].buildingCount).toBe(2);
    expect(builder.buildingCount).toBe(2);
    expect(builder.triangleCount).toBe((30 + 48) / 3);
    expect(g.vertexCount).toBe(20 + 30);
    expect(g.indexCount).toBe(30 + 48);

    // 索引全部落在合并顶点范围内
    for (let i = 0; i < g.indexCount; i++) {
      expect(g.indices[i]).toBeLessThan(g.vertexCount);
    }

    // semanticId 按楼分段：前 20 顶点 = 7，后 30 顶点 = 9
    for (let v = 0; v < 20; v++) expect(g.semanticIds[v]).toBe(7);
    for (let v = 20; v < 50; v++) expect(g.semanticIds[v]).toBe(9);

    // 合批不改变逐三角形 winding，整体仍朝外
    expectAllFacesOutward(g, '合批块');
  });

  it('超阈值自动分块：块数正确、每块索引自洽、楼栋总数不丢', () => {
    const builder = new CityBlockBuilder({ maxChunkBytes: 2000 });
    // 方形楼一块 ≈ 20×32 + 30×4 = 760B → 2000B 阈值下每块 2 栋
    for (let i = 0; i < 5; i++) builder.addBuilding(squareRingCCW(), 20 + i, i + 1);
    const chunks = builder.buildChunks();

    expect(chunks.length).toBe(3);
    expect(chunks[0].buildingCount).toBe(2);
    expect(chunks[1].buildingCount).toBe(2);
    expect(chunks[2].buildingCount).toBe(1);
    let totalVerts = 0;
    for (const c of chunks) {
      for (let i = 0; i < c.geometry.indexCount; i++) {
        expect(c.geometry.indices[i]).toBeLessThan(c.geometry.vertexCount);
      }
      totalVerts += c.geometry.vertexCount;
    }
    expect(totalVerts).toBe(5 * 20);
    expect(builder.buildingCount).toBe(5);
    expect(builder.triangleCount).toBe(5 * 10);
  });

  it('退化 footprint 静默跳过，不计入统计', () => {
    const builder = new CityBlockBuilder();
    builder.addBuilding([{ lon: BASE_LON, lat: BASE_LAT }], 10, 1); // 单点
    builder.addBuilding(squareRingCCW(), 10, 2);
    expect(builder.buildingCount).toBe(1);
    expect(builder.triangleCount).toBe(10);
  });
});
