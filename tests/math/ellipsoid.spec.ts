import { describe, it, expect } from "vitest";
import { Ellipsoid } from "../../src/math/Ellipsoid";
import { Cartesian3 } from "../../src/math/Cartesian3";
import { CesiumMath } from "../../src/math/CesiumMath";

describe("Ellipsoid", () => {
  describe("WGS84 常数", () => {
    it("半轴长：a=6378137.0, b=6356752.3142", () => {
      const e = Ellipsoid.WGS84;
      expect(e.radii.x).toBe(6378137.0);
      expect(e.radii.y).toBe(6378137.0);
      expect(e.radii.z).toBe(6356752.3142);
    });

    it("radiiSquared 为各半轴平方", () => {
      const e = Ellipsoid.WGS84;
      expect(e.radiiSquared.x).toBeCloseTo(6378137.0 ** 2, 0);
      expect(e.radiiSquared.z).toBeCloseTo(6356752.3142 ** 2, 0);
    });

    it("oneOverRadiiSquared 为各半轴平方的倒数", () => {
      const e = Ellipsoid.WGS84;
      expect(e.oneOverRadiiSquared.x).toBeCloseTo(1 / (6378137.0 ** 2), 20);
      expect(e.oneOverRadiiSquared.z).toBeCloseTo(1 / (6356752.3142 ** 2), 20);
    });

    it("UNIT_SPHERE 三轴均为 1", () => {
      const e = Ellipsoid.UNIT_SPHERE;
      expect(e.radii.x).toBe(1);
      expect(e.radii.y).toBe(1);
      expect(e.radii.z).toBe(1);
    });
  });

  describe("geodeticSurfaceNormal", () => {
    it("赤道点 (a,0,0) 法线为 +X", () => {
      const n = Ellipsoid.WGS84.geodeticSurfaceNormal(
        new Cartesian3(6378137.0, 0, 0)
      );
      expect(n.x).toBeCloseTo(1, 9);
      expect(n.y).toBeCloseTo(0, 9);
      expect(n.z).toBeCloseTo(0, 9);
    });

    it("北极点 (0,0,b) 法线为 +Z", () => {
      const n = Ellipsoid.WGS84.geodeticSurfaceNormal(
        new Cartesian3(0, 0, 6356752.3142)
      );
      expect(n.x).toBeCloseTo(0, 9);
      expect(n.y).toBeCloseTo(0, 9);
      expect(n.z).toBeCloseTo(1, 9);
    });

    it("单位球面上法线等于位置方向的单位向量", () => {
      const p = new Cartesian3(1, 1, 1);
      const n = Ellipsoid.UNIT_SPHERE.geodeticSurfaceNormal(p);
      const expected = 1 / Math.sqrt(3);
      expect(n.x).toBeCloseTo(expected, 12);
      expect(n.y).toBeCloseTo(expected, 12);
      expect(n.z).toBeCloseTo(expected, 12);
    });

    it("结果为单位向量", () => {
      const n = Ellipsoid.WGS84.geodeticSurfaceNormal(
        new Cartesian3(3000000, 3000000, 4000000)
      );
      const mag = Cartesian3.magnitude(n);
      expect(mag).toBeCloseTo(1, 9);
    });
  });

  describe("cartographicToCartesian / cartesianToCartographic", () => {
    it("WGS84 表面点往返转换保持经纬度（1e-6 精度）", () => {
      const lon = CesiumMath.toRadians(116.4);
      const lat = CesiumMath.toRadians(39.9);
      const height = 1234.5;
      const c = Ellipsoid.WGS84.cartographicToCartesian(lon, lat, height);
      const back = Ellipsoid.WGS84.cartesianToCartographic(c);
      expect(back.longitude).toBeCloseTo(lon, 6);
      expect(back.latitude).toBeCloseTo(lat, 6);
      expect(back.height).toBeCloseTo(height, 3);
    });

    it("赤道表面点到原点距离为半长轴 a", () => {
      const c = Ellipsoid.WGS84.cartographicToCartesian(0, 0, 0);
      expect(Cartesian3.magnitude(c)).toBeCloseTo(6378137.0, 3);
    });

    it("极点表面点到原点距离为半短轴 b", () => {
      const c = Ellipsoid.WGS84.cartographicToCartesian(0, Math.PI / 2, 0);
      expect(Cartesian3.magnitude(c)).toBeCloseTo(6356752.3142, 3);
    });
  });
});
