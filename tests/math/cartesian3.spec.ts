import { describe, it, expect } from "vitest";
import { Cartesian3 } from "../../src/math/Cartesian3";
import { CesiumMath } from "../../src/math/CesiumMath";

describe("Cartesian3", () => {
  describe("fromDegrees", () => {
    it("(0,0) 赤道与本初子午线交点：x = 半长轴 a", () => {
      const c = Cartesian3.fromDegrees(0, 0);
      expect(c.x).toBeCloseTo(6378137.0, 3);
      expect(c.y).toBeCloseTo(0, 6);
      expect(c.z).toBeCloseTo(0, 6);
    });

    it("(90,0)：落在 +Y 轴", () => {
      const c = Cartesian3.fromDegrees(90, 0);
      expect(c.x).toBeCloseTo(0, 3);
      expect(c.y).toBeCloseTo(6378137.0, 3);
      expect(c.z).toBeCloseTo(0, 6);
    });

    it("(0,90)：北极点，z = 半短轴 b", () => {
      const c = Cartesian3.fromDegrees(0, 90);
      expect(c.x).toBeCloseTo(0, 3);
      expect(c.y).toBeCloseTo(0, 3);
      expect(c.z).toBeCloseTo(6356752.3142, 3);
    });

    it("带高度：沿法线方向抬升", () => {
      const c = Cartesian3.fromDegrees(0, 0, 1000);
      expect(c.x).toBeCloseTo(6378137.0 + 1000, 3);
    });

    it("负经度/纬度方向正确", () => {
      const c = Cartesian3.fromDegrees(-90, -90);
      // 纬度 ±90° 时 cosLat ≈ 0，水平分量趋于 0
      expect(c.x).toBeCloseTo(0, 3);
      expect(c.y).toBeCloseTo(0, 3);
      expect(c.z).toBeCloseTo(-6356752.3142, 3);
    });
  });

  describe("normalize", () => {
    it("(3,4,0) 归一化 → (0.6, 0.8, 0)", () => {
      const r = Cartesian3.normalize(new Cartesian3(3, 4, 0), new Cartesian3());
      expect(r.x).toBeCloseTo(0.6, 12);
      expect(r.y).toBeCloseTo(0.8, 12);
      expect(r.z).toBeCloseTo(0, 12);
      expect(Cartesian3.magnitude(r)).toBeCloseTo(1, 12);
    });

    it("单位向量归一化不变", () => {
      const r = Cartesian3.normalize(Cartesian3.UNIT_Z, new Cartesian3());
      expect(r.x).toBeCloseTo(0, 12);
      expect(r.y).toBeCloseTo(0, 12);
      expect(r.z).toBeCloseTo(1, 12);
    });

    it("零向量：按当前实现除以 0，各分量为 NaN", () => {
      const r = Cartesian3.normalize(Cartesian3.ZERO, new Cartesian3());
      expect(Number.isNaN(r.x)).toBe(true);
      expect(Number.isNaN(r.y)).toBe(true);
      expect(Number.isNaN(r.z)).toBe(true);
    });
  });

  describe("dot", () => {
    it("垂直向量点积为 0", () => {
      expect(Cartesian3.dot(Cartesian3.UNIT_X, Cartesian3.UNIT_Y)).toBe(0);
    });

    it("平行向量点积为模长平方", () => {
      expect(Cartesian3.dot(Cartesian3.UNIT_X, Cartesian3.UNIT_X)).toBe(1);
      const a = new Cartesian3(1, 2, 2);
      expect(Cartesian3.dot(a, a)).toBeCloseTo(9, 12);
    });
  });

  describe("cross", () => {
    it("UNIT_X × UNIT_Y = UNIT_Z", () => {
      const r = Cartesian3.cross(Cartesian3.UNIT_X, Cartesian3.UNIT_Y, new Cartesian3());
      expect(r.x).toBeCloseTo(0, 12);
      expect(r.y).toBeCloseTo(0, 12);
      expect(r.z).toBeCloseTo(1, 12);
    });

    it("反交换律：UNIT_Y × UNIT_X = -UNIT_Z", () => {
      const r = Cartesian3.cross(Cartesian3.UNIT_Y, Cartesian3.UNIT_X, new Cartesian3());
      expect(r.x).toBeCloseTo(0, 12);
      expect(r.y).toBeCloseTo(0, 12);
      expect(r.z).toBeCloseTo(-1, 12);
    });

    it("叉积垂直于两个输入向量", () => {
      const a = new Cartesian3(1, 2, 3);
      const b = new Cartesian3(-4, 5, 6);
      const r = Cartesian3.cross(a, b, new Cartesian3());
      expect(Cartesian3.dot(r, a)).toBeCloseTo(0, 9);
      expect(Cartesian3.dot(r, b)).toBeCloseTo(0, 9);
    });
  });

  describe("magnitude / magnitudeSquared", () => {
    it("(1,2,2) 模长为 3", () => {
      const a = new Cartesian3(1, 2, 2);
      expect(Cartesian3.magnitude(a)).toBeCloseTo(3, 12);
      expect(Cartesian3.magnitudeSquared(a)).toBeCloseTo(9, 12);
    });

    it("零向量模长为 0", () => {
      expect(Cartesian3.magnitude(Cartesian3.ZERO)).toBe(0);
    });
  });

  describe("distance", () => {
    it("原点到 (1,2,2) 距离为 3", () => {
      expect(Cartesian3.distance(Cartesian3.ZERO, new Cartesian3(1, 2, 2)))
        .toBeCloseTo(3, 12);
    });

    it("同一点距离为 0", () => {
      const a = new Cartesian3(5, 5, 5);
      expect(Cartesian3.distance(a, a)).toBe(0);
    });
  });

  describe("lerp", () => {
    it("t=0 返回起点，t=1 返回终点", () => {
      const start = new Cartesian3(0, 0, 0);
      const end = new Cartesian3(10, -10, 4);
      const r0 = Cartesian3.lerp(start, end, 0, new Cartesian3());
      const r1 = Cartesian3.lerp(start, end, 1, new Cartesian3());
      expect(Cartesian3.equals(r0, start)).toBe(true);
      expect(Cartesian3.equals(r1, end)).toBe(true);
    });

    it("t=0.5 取中点", () => {
      const r = Cartesian3.lerp(
        new Cartesian3(0, 0, 0),
        new Cartesian3(10, 20, 30),
        0.5,
        new Cartesian3()
      );
      expect(r.x).toBeCloseTo(5, 12);
      expect(r.y).toBeCloseTo(10, 12);
      expect(r.z).toBeCloseTo(15, 12);
    });

    it("t=0.25 加权插值", () => {
      const r = Cartesian3.lerp(
        new Cartesian3(0, 4, 8),
        new Cartesian3(8, 0, 0),
        0.25,
        new Cartesian3()
      );
      expect(r.x).toBeCloseTo(2, 12);
      expect(r.y).toBeCloseTo(3, 12);
      expect(r.z).toBeCloseTo(6, 12);
    });
  });

  describe("add / subtract / multiplyByScalar", () => {
    it("add 逐分量相加", () => {
      const r = Cartesian3.add(
        new Cartesian3(1, 2, 3),
        new Cartesian3(4, 5, 6),
        new Cartesian3()
      );
      expect(Cartesian3.equals(r, new Cartesian3(5, 7, 9))).toBe(true);
    });

    it("subtract 逐分量相减", () => {
      const r = Cartesian3.subtract(
        new Cartesian3(4, 5, 6),
        new Cartesian3(1, 2, 3),
        new Cartesian3()
      );
      expect(Cartesian3.equals(r, new Cartesian3(3, 3, 3))).toBe(true);
    });

    it("multiplyByScalar 逐分量缩放", () => {
      const r = Cartesian3.multiplyByScalar(
        new Cartesian3(1, -2, 3),
        2,
        new Cartesian3()
      );
      expect(Cartesian3.equals(r, new Cartesian3(2, -4, 6))).toBe(true);
    });
  });

  describe("WGS84 常数", () => {
    it("EARTH_RADIUS 为 WGS84 半长轴", () => {
      expect(Cartesian3.EARTH_RADIUS).toBe(6378137.0);
    });

    it("fromRadians 与 fromDegrees 一致（经 180° 验证）", () => {
      const deg = Cartesian3.fromDegrees(180, 0);
      const rad = Cartesian3.fromRadians(Math.PI, 0);
      expect(Cartesian3.equalsEpsilon(deg, rad, CesiumMath.EPSILON14)).toBe(true);
    });
  });
});
