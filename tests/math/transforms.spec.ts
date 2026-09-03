import { describe, it, expect } from "vitest";
import { Transforms } from "../../src/math/Transforms";
import { Cartesian3 } from "../../src/math/Cartesian3";
import { Matrix4 } from "../../src/math/Matrix4";

const PRECISION = 1e-6;

/** 从 ENU 矩阵提取第 index 列的方向向量（前 3 个分量） */
function column(enu: Matrix4, index: number): Cartesian3 {
  const v = enu.values;
  return new Cartesian3(v[index * 4], v[index * 4 + 1], v[index * 4 + 2]);
}

describe("Transforms.eastNorthUpToFixedFrame", () => {
  it("赤道/本初子午线处：East=+Y, North=+Z, Up=+X", () => {
    const origin = Cartesian3.fromDegrees(0, 0); // (a, 0, 0) 米
    const enu = Transforms.eastNorthUpToFixedFrame(origin);

    const east = column(enu, 0);
    const north = column(enu, 1);
    const up = column(enu, 2);

    // East ≈ (0, 1, 0)
    expect(east.x).toBeCloseTo(0, PRECISION);
    expect(east.y).toBeCloseTo(1, PRECISION);
    expect(east.z).toBeCloseTo(0, PRECISION);
    // North ≈ (0, 0, 1)
    expect(north.x).toBeCloseTo(0, PRECISION);
    expect(north.y).toBeCloseTo(0, PRECISION);
    expect(north.z).toBeCloseTo(1, PRECISION);
    // Up ≈ (1, 0, 0)（球面近似下的表面法线）
    expect(up.x).toBeCloseTo(1, PRECISION);
    expect(up.y).toBeCloseTo(0, PRECISION);
    expect(up.z).toBeCloseTo(0, PRECISION);
  });

  it("平移列：ECEF 米坐标被归一化到引擎单位（逐分量除以 EARTH_RADIUS）", () => {
    const origin = Cartesian3.fromDegrees(116.4, 39.9, 0);
    const enu = Transforms.eastNorthUpToFixedFrame(origin);
    const t = enu.getTranslation();
    expect(t.x).toBeCloseTo(origin.x / Cartesian3.EARTH_RADIUS, 6);
    expect(t.y).toBeCloseTo(origin.y / Cartesian3.EARTH_RADIUS, 6);
    expect(t.z).toBeCloseTo(origin.z / Cartesian3.EARTH_RADIUS, 6);
  });

  it("北京上空：三个基向量正交且为单位长度", () => {
    const origin = Cartesian3.fromDegrees(116.4, 39.9);
    const enu = Transforms.eastNorthUpToFixedFrame(origin);
    const east = column(enu, 0);
    const north = column(enu, 1);
    const up = column(enu, 2);

    expect(Cartesian3.magnitude(east)).toBeCloseTo(1, PRECISION);
    expect(Cartesian3.magnitude(north)).toBeCloseTo(1, PRECISION);
    expect(Cartesian3.magnitude(up)).toBeCloseTo(1, PRECISION);
    expect(Cartesian3.dot(east, north)).toBeCloseTo(0, PRECISION);
    expect(Cartesian3.dot(east, up)).toBeCloseTo(0, PRECISION);
    expect(Cartesian3.dot(north, up)).toBeCloseTo(0, PRECISION);
  });

  it("Up 列等于位置方向的单位向量（球面近似）", () => {
    const origin = Cartesian3.fromDegrees(116.4, 39.9, 500);
    const enu = Transforms.eastNorthUpToFixedFrame(origin);
    const up = column(enu, 2);
    const expected = Cartesian3.normalize(
      new Cartesian3(origin.x, origin.y, origin.z),
      new Cartesian3()
    );
    expect(up.x).toBeCloseTo(expected.x, 5);
    expect(up.y).toBeCloseTo(expected.y, 5);
    expect(up.z).toBeCloseTo(expected.z, 5);
  });

  it("右手系：East × North = Up", () => {
    const origin = Cartesian3.fromDegrees(-73.5, 12.3);
    const enu = Transforms.eastNorthUpToFixedFrame(origin);
    const cross = Cartesian3.cross(column(enu, 0), column(enu, 1), new Cartesian3());
    const up = column(enu, 2);
    expect(cross.x).toBeCloseTo(up.x, 5);
    expect(cross.y).toBeCloseTo(up.y, 5);
    expect(cross.z).toBeCloseTo(up.z, 5);
  });

  it("与 fromDegrees 组合：ENU 矩阵把本地原点映射到归一化 ECEF", () => {
    const origin = Cartesian3.fromDegrees(116.4, 39.9, 1000);
    const enu = Transforms.eastNorthUpToFixedFrame(origin);
    // ENU * (0,0,0,1) = 平移列
    const t = enu.getTranslation();
    const expectedMagnitude = Cartesian3.magnitude(origin) / Cartesian3.EARTH_RADIUS;
    expect(Cartesian3.magnitude(t)).toBeCloseTo(expectedMagnitude, 5);
  });

  it("result 参数被复用", () => {
    const out = new Matrix4();
    const r = Transforms.eastNorthUpToFixedFrame(Cartesian3.fromDegrees(1, 2), out);
    expect(r).toBe(out);
  });
});
