import { describe, it, expect } from "vitest";
import { Quaternion } from "../../src/math/Quaternion";
import { Cartesian3 } from "../../src/math/Cartesian3";

const PRECISION = 1e-12;

/** 用 v' = q * v * q⁻¹ 计算四元数旋转向量 */
function rotateVector(q: Quaternion, v: Cartesian3): Cartesian3 {
  const p = new Quaternion(v.x, v.y, v.z, 0);
  const tmp = Quaternion.multiply(q, p, new Quaternion());
  const inverse = Quaternion.conjugate(q, new Quaternion());
  const r = Quaternion.multiply(tmp, inverse, new Quaternion());
  return new Cartesian3(r.x, r.y, r.z);
}

describe("Quaternion", () => {
  describe("IDENTITY", () => {
    it("单位四元数为 (0,0,0,1)", () => {
      const q = Quaternion.IDENTITY;
      expect(q.x).toBe(0);
      expect(q.y).toBe(0);
      expect(q.z).toBe(0);
      expect(q.w).toBe(1);
    });

    it("IDENTITY 乘任意四元数保持不变", () => {
      const q = Quaternion.fromAxisAngle(Cartesian3.UNIT_Z, Math.PI / 3);
      const r = Quaternion.multiply(Quaternion.IDENTITY, q, new Quaternion());
      expect(r.x).toBeCloseTo(q.x, 12);
      expect(r.y).toBeCloseTo(q.y, 12);
      expect(r.z).toBeCloseTo(q.z, 12);
      expect(r.w).toBeCloseTo(q.w, 12);
    });
  });

  describe("fromAxisAngle", () => {
    it("绕 Z 轴 90°：(0, 0, √2/2, √2/2)", () => {
      const q = Quaternion.fromAxisAngle(Cartesian3.UNIT_Z, Math.PI / 2);
      const half = Math.SQRT1_2;
      expect(q.x).toBeCloseTo(0, 12);
      expect(q.y).toBeCloseTo(0, 12);
      expect(q.z).toBeCloseTo(half, 12);
      expect(q.w).toBeCloseTo(half, 12);
    });

    it("0° 角为单位四元数", () => {
      const q = Quaternion.fromAxisAngle(Cartesian3.UNIT_X, 0);
      expect(q.x).toBeCloseTo(0, 12);
      expect(q.y).toBeCloseTo(0, 12);
      expect(q.z).toBeCloseTo(0, 12);
      expect(q.w).toBeCloseTo(1, 12);
    });

    it("绕 Z 轴 90° 将 UNIT_X 旋转为 UNIT_Y", () => {
      const q = Quaternion.fromAxisAngle(Cartesian3.UNIT_Z, Math.PI / 2);
      const r = rotateVector(q, Cartesian3.UNIT_X);
      expect(r.x).toBeCloseTo(0, 9);
      expect(r.y).toBeCloseTo(1, 9);
      expect(r.z).toBeCloseTo(0, 9);
    });

    it("绕 X 轴 90° 将 UNIT_Y 旋转为 UNIT_Z", () => {
      const q = Quaternion.fromAxisAngle(Cartesian3.UNIT_X, Math.PI / 2);
      const r = rotateVector(q, Cartesian3.UNIT_Y);
      expect(r.x).toBeCloseTo(0, 9);
      expect(r.y).toBeCloseTo(0, 9);
      expect(r.z).toBeCloseTo(1, 9);
    });

    it("单位轴（需预先归一化）生成单位四元数", () => {
      // API 约定：axis 必须为单位向量（与 Cesium 一致，不内部归一化）
      const invSqrt3 = 1 / Math.sqrt(3);
      const axis = new Cartesian3(invSqrt3, invSqrt3, invSqrt3);
      const q = Quaternion.fromAxisAngle(axis, Math.PI / 5);
      const mag = Math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
      expect(mag).toBeCloseTo(1, 9);
    });
  });

  describe("multiply", () => {
    it("两个 45° 绕 Z 旋转复合为 90° 绕 Z 旋转", () => {
      const half = Quaternion.fromAxisAngle(Cartesian3.UNIT_Z, Math.PI / 4);
      const full = Quaternion.fromAxisAngle(Cartesian3.UNIT_Z, Math.PI / 2);
      const r = Quaternion.multiply(half, half, new Quaternion());
      expect(r.x).toBeCloseTo(full.x, 9);
      expect(r.y).toBeCloseTo(full.y, 9);
      expect(r.z).toBeCloseTo(full.z, 9);
      expect(r.w).toBeCloseTo(full.w, 9);
    });

    it("乘法不满足交换律（不同轴）", () => {
      const qx = Quaternion.fromAxisAngle(Cartesian3.UNIT_X, Math.PI / 2);
      const qy = Quaternion.fromAxisAngle(Cartesian3.UNIT_Y, Math.PI / 2);
      const xy = Quaternion.multiply(qx, qy, new Quaternion());
      const yx = Quaternion.multiply(qy, qx, new Quaternion());
      const differs =
        Math.abs(xy.x - yx.x) > 1e-9 ||
        Math.abs(xy.y - yx.y) > 1e-9 ||
        Math.abs(xy.z - yx.z) > 1e-9 ||
        Math.abs(xy.w - yx.w) > 1e-9;
      expect(differs).toBe(true);
    });

    it("q * conjugate(q) = IDENTITY", () => {
      const invSqrt3 = 1 / Math.sqrt(3);
      const q = Quaternion.fromAxisAngle(
        new Cartesian3(invSqrt3, invSqrt3, invSqrt3),
        1.2
      );
      const c = Quaternion.conjugate(q, new Quaternion());
      const r = Quaternion.multiply(q, c, new Quaternion());
      expect(r.x).toBeCloseTo(0, 9);
      expect(r.y).toBeCloseTo(0, 9);
      expect(r.z).toBeCloseTo(0, 9);
      expect(r.w).toBeCloseTo(1, 9);
    });
  });

  describe("slerp", () => {
    const start = Quaternion.IDENTITY;
    const end = Quaternion.fromAxisAngle(Cartesian3.UNIT_Z, Math.PI / 2);

    it("t=0 返回 start，t=1 返回 end", () => {
      const r0 = Quaternion.slerp(start, end, 0, new Quaternion());
      const r1 = Quaternion.slerp(start, end, 1, new Quaternion());
      expect(r0.x).toBeCloseTo(start.x, 9);
      expect(r0.w).toBeCloseTo(start.w, 9);
      expect(r1.z).toBeCloseTo(end.z, 9);
      expect(r1.w).toBeCloseTo(end.w, 9);
    });

    it("t=0.5 为 45° 绕 Z 旋转", () => {
      const mid = Quaternion.fromAxisAngle(Cartesian3.UNIT_Z, Math.PI / 4);
      const r = Quaternion.slerp(start, end, 0.5, new Quaternion());
      expect(r.x).toBeCloseTo(mid.x, 9);
      expect(r.y).toBeCloseTo(mid.y, 9);
      expect(r.z).toBeCloseTo(mid.z, 9);
      expect(r.w).toBeCloseTo(mid.w, 9);
    });

    it("球面插值结果保持单位模长", () => {
      const r = Quaternion.slerp(start, end, 0.3, new Quaternion());
      const mag = Math.sqrt(r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w);
      expect(mag).toBeCloseTo(1, 9);
    });

    it("对跖四元数（dot<0 分支）插值结果保持单位模长", () => {
      const negated = new Quaternion(-end.x, -end.y, -end.z, -end.w);
      const r = Quaternion.slerp(start, negated, 1, new Quaternion());
      const mag = Math.sqrt(r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w);
      expect(mag).toBeCloseTo(1, 9);
    });
  });

  describe("normalize", () => {
    it("非单位四元数归一化为单位模长", () => {
      const q = new Quaternion(0, 0, 2, 2);
      const r = Quaternion.normalize(q, new Quaternion());
      const mag = Math.sqrt(r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w);
      expect(mag).toBeCloseTo(1, 12);
      expect(r.z).toBeCloseTo(Math.SQRT1_2, 12);
      expect(r.w).toBeCloseTo(Math.SQRT1_2, 12);
    });
  });

  describe("conjugate", () => {
    it("共轭仅翻转向量部分", () => {
      const q = Quaternion.fromAxisAngle(Cartesian3.UNIT_Y, Math.PI / 3);
      const c = Quaternion.conjugate(q, new Quaternion());
      expect(c.x).toBeCloseTo(-q.x, 12);
      expect(c.y).toBeCloseTo(-q.y, 12);
      expect(c.z).toBeCloseTo(-q.z, 12);
      expect(c.w).toBeCloseTo(q.w, 12);
    });
  });
});
