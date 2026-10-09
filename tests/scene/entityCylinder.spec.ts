import { describe, expect, it } from 'vitest';
import { HeadingPitchRoll } from '../../src/math/HeadingPitchRoll';
import { Transforms } from '../../src/math/Transforms';
import { Quaternion } from '../../src/math/Quaternion';
import { CylinderGeometry } from '../../src/geometry/CylinderGeometry';
import { composeEntityFrameMatrix } from '../../src/scene/EntityModelMatrix';
import { Cartesian3 } from '../../src/math/Cartesian3';
const EARTH = 6378137.0;
const surface = new Cartesian3(1, 0, 0); // 赤道 lon0：east=(0,1,0) north=(0,0,1) up=(1,0,0)

describe('HeadingPitchRoll', () => {
  it('value semantics: fromRadians / clone / equals', () => {
    const a = HeadingPitchRoll.fromRadians(0.1, -0.2, 0.3);
    expect(a.heading).toBeCloseTo(0.1, 12);
    expect(a.pitch).toBeCloseTo(-0.2, 12);
    expect(a.roll).toBeCloseTo(0.3, 12);
    const b = a.clone();
    expect(b).not.toBe(a);
    expect(b.equals(a)).toBe(true);
    b.pitch += 1e-3;
    expect(b.equals(a)).toBe(false);
    const into = new HeadingPitchRoll();
    a.clone(into);
    expect(into.equals(a)).toBe(true);
  });
});

describe('Transforms.headingPitchRollQuaternion 重载', () => {
  const origin = Cartesian3.fromDegrees(116, 39, 100);

  it('HeadingPitchRoll 对象与三 number 形式等价', () => {
    const viaObject = Transforms.headingPitchRollQuaternion(
      origin,
      HeadingPitchRoll.fromRadians(0.4, -0.5, 0.6)
    );
    const viaNumbers = Transforms.headingPitchRollQuaternion(origin, 0.4, -0.5, 0.6);
    for (const k of ['x', 'y', 'z', 'w'] as const) {
      expect(viaObject[k]).toBeCloseTo(viaNumbers[k], 12);
    }
  });

  it('结果可写入传入的 result 四元数', () => {
    const out = new Quaternion();
    const q = Transforms.headingPitchRollQuaternion(origin, new HeadingPitchRoll(0, 0, Math.PI), out);
    expect(q).toBe(out);
    expect(Math.hypot(q.x, q.y, q.z, q.w)).toBeCloseTo(1, 9);
  });
});

describe('CylinderGeometry', () => {
  it('顶点半径/高度换算到归一化单位（轴 = 局部 +Z）', () => {
    const geo = CylinderGeometry.createGeometry({
      length: 1000,
      topRadius: 0,
      bottomRadius: 200,
      slices: 8,
    });
    const v = geo.vertices;
    // 8+1 底环 + 8+1 顶环（尖顶半径 0，无顶盖），每顶点 8 floats。
    const bottomRing = 9;
    const topRingStart = bottomRing;
    // 底环第一个顶点 (r, 0, -h/2)。
    expect(v[0]).toBeCloseTo(200 / EARTH, 9);
    expect(v[1]).toBeCloseTo(0, 9);
    expect(v[2]).toBeCloseTo(-500 / EARTH, 9);
    // 顶环（锥尖）第一个顶点 (0, 0, +h/2)。
    const t = topRingStart * 8;
    expect(v[t]).toBeCloseTo(0, 9);
    expect(v[t + 1]).toBeCloseTo(0, 9);
    expect(v[t + 2]).toBeCloseTo(500 / EARTH, 9);
    expect(geo.indexCount).toBeGreaterThan(0);
  });

  it('锥面法线沿斜面（dz = (rBottom - rTop)/length）', () => {
    const length = 1000, top = 0, bottom = 200;
    const geo = CylinderGeometry.createGeometry({ length, topRadius: top, bottomRadius: bottom, slices: 4 });
    const dz = (bottom - top) / length;
    const nl = Math.hypot(1, dz);
    // 底环第一个顶点法线 (1, 0, dz)/|·|（法线在第一个 float 三元组之后）。
    expect(geo.vertices[3]).toBeCloseTo(1 / nl, 7);
    expect(geo.vertices[4]).toBeCloseTo(0, 7);
    expect(geo.vertices[5]).toBeCloseTo(dz / nl, 7);
  });

  it('signature 随 length/topRadius 变化（重建判据）', () => {
    const a = CylinderGeometry.signature({ length: 10, topRadius: 0, bottomRadius: 2 });
    // 1e-5 量级抖动不触发重建。
    expect(CylinderGeometry.signature({ length: 10.00001, topRadius: 0, bottomRadius: 2 })).toBe(a);
    expect(CylinderGeometry.signature({ length: 11, topRadius: 0, bottomRadius: 2 })).not.toBe(a);
    expect(CylinderGeometry.signature({ length: 10, topRadius: 1, bottomRadius: 2 })).not.toBe(a);
  });
  it('与 Cesium 逐轴组合公式严格一致（ground truth 对拍）', () => {
    const h = 0.7, p = -1.1, r = 0.3;
    const expected = Quaternion.multiply(
      Quaternion.fromAxisAngle(new Cartesian3(0, 0, 1), -h),
      Quaternion.multiply(
        Quaternion.fromAxisAngle(new Cartesian3(0, 1, 0), -p),
        Quaternion.fromAxisAngle(new Cartesian3(1, 0, 0), r),
        new Quaternion()
      ),
      new Quaternion()
    );
    const q = Quaternion.fromHeadingPitchRoll(h, p, r);
    for (const k of ['x', 'y', 'z', 'w'] as const) {
      expect(q[k]).toBeCloseTo(expected[k], 12);
    }
  });

  it('pitch −90°：轴倒向东（机头锥姿态，Cesium 语义）', () => {
    const q = Transforms.headingPitchRollQuaternion(
      Cartesian3.multiplyByScalar(surface, EARTH, new Cartesian3()),
      HeadingPitchRoll.fromRadians(0, -Math.PI / 2, 0)
    );
    const m = composeEntityFrameMatrix(surface, q);
    // 局部 +Z 的像 = east (0,1,0)。
    expect(m.values[8]).toBeCloseTo(0, 9);
    expect(m.values[9]).toBeCloseTo(1, 9);
    expect(m.values[10]).toBeCloseTo(0, 9);
  });

  it('pitch −90° + heading 90°：轴顺时针转到南', () => {
    const q = Transforms.headingPitchRollQuaternion(
      Cartesian3.multiplyByScalar(surface, EARTH, new Cartesian3()),
      HeadingPitchRoll.fromRadians(Math.PI / 2, -Math.PI / 2, 0)
    );
    const m = composeEntityFrameMatrix(surface, q);
    // 局部 +Z 的像 = south (0,0,-1)。
    expect(m.values[8]).toBeCloseTo(0, 9);
    expect(m.values[9]).toBeCloseTo(0, 9);
    expect(m.values[10]).toBeCloseTo(-1, 9);
  });
});
