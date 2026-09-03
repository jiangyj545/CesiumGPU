import { describe, it, expect } from "vitest";
import { CesiumMath } from "../../src/math/CesiumMath";

describe("CesiumMath", () => {
  describe("toRadians / toDegrees", () => {
    it("toRadians(180) = PI", () => {
      expect(CesiumMath.toRadians(180)).toBeCloseTo(Math.PI, 12);
    });

    it("toRadians(90) = PI/2", () => {
      expect(CesiumMath.toRadians(90)).toBeCloseTo(Math.PI / 2, 12);
    });

    it("toDegrees(PI) = 180", () => {
      expect(CesiumMath.toDegrees(Math.PI)).toBeCloseTo(180, 12);
    });

    it("往返转换恒等", () => {
      const angle = 37.7749;
      expect(CesiumMath.toDegrees(CesiumMath.toRadians(angle)))
        .toBeCloseTo(angle, 12);
    });

    it("常数 RADIANS_PER_DEGREE = PI/180", () => {
      expect(CesiumMath.RADIANS_PER_DEGREE).toBeCloseTo(Math.PI / 180, 15);
    });
  });

  describe("clamp", () => {
    it("值在区间内保持不变", () => {
      expect(CesiumMath.clamp(2, 0, 3)).toBe(2);
    });

    it("低于下界返回 min", () => {
      expect(CesiumMath.clamp(-1, 0, 3)).toBe(0);
    });

    it("高于上界返回 max", () => {
      expect(CesiumMath.clamp(5, 0, 3)).toBe(3);
    });

    it("边界值原样返回", () => {
      expect(CesiumMath.clamp(0, 0, 3)).toBe(0);
      expect(CesiumMath.clamp(3, 0, 3)).toBe(3);
    });
  });

  describe("lerp", () => {
    it("t=0 返回 p，t=1 返回 q", () => {
      expect(CesiumMath.lerp(2, 10, 0)).toBe(2);
      expect(CesiumMath.lerp(2, 10, 1)).toBe(10);
    });

    it("t=0.5 取中点", () => {
      expect(CesiumMath.lerp(0, 10, 0.5)).toBeCloseTo(5, 12);
    });

    it("t=0.25 加权插值", () => {
      expect(CesiumMath.lerp(0, 8, 0.25)).toBeCloseTo(2, 12);
    });

    it("负区间插值", () => {
      expect(CesiumMath.lerp(-4, 4, 0.5)).toBeCloseTo(0, 12);
    });
  });

  describe("smoothstep", () => {
    it("区间中点 x=(edge0+edge1)/2 结果为 0.5", () => {
      expect(CesiumMath.smoothstep(0, 2, 1)).toBeCloseTo(0.5, 12);
    });

    it("x=edge0 结果为 0，x=edge1 结果为 1", () => {
      expect(CesiumMath.smoothstep(0, 1, 0)).toBeCloseTo(0, 12);
      expect(CesiumMath.smoothstep(0, 1, 1)).toBeCloseTo(1, 12);
    });

    it("低于 edge0 夹取为 0，高于 edge1 夹取为 1", () => {
      expect(CesiumMath.smoothstep(0, 2, -1)).toBeCloseTo(0, 12);
      expect(CesiumMath.smoothstep(0, 2, 3)).toBeCloseTo(1, 12);
    });

    it("平滑性：x=1.5（[0,2] 区间 75% 处）≈ 0.84375", () => {
      // t=0.75 → t²(3-2t) = 0.5625 * 1.5 = 0.84375
      expect(CesiumMath.smoothstep(0, 2, 1.5)).toBeCloseTo(0.84375, 12);
    });

    it("edge0 === edge1 且 x === edge：按当前实现除零得 NaN", () => {
      const r = CesiumMath.smoothstep(1, 1, 1);
      expect(Number.isNaN(r)).toBe(true);
    });

    it("edge0 === edge1 且 x < edge：-Infinity 被夹取为 0（当前实现行为）", () => {
      expect(CesiumMath.smoothstep(1, 1, 0)).toBe(0);
    });

    it("edge0 === edge1 且 x > edge：+Infinity 被夹取为 1（当前实现行为）", () => {
      expect(CesiumMath.smoothstep(1, 1, 2)).toBe(1);
    });
  });

  describe("zeroToTwoPi / negativePiToPi", () => {
    it("zeroToTwoPi 把负角映射到 [0, 2PI)", () => {
      expect(CesiumMath.zeroToTwoPi(-Math.PI / 2)).toBeCloseTo((3 * Math.PI) / 2, 12);
      expect(CesiumMath.zeroToTwoPi(Math.PI / 2)).toBeCloseTo(Math.PI / 2, 12);
    });

    it("negativePiToPi 把 3PI/2 映射为 -PI/2", () => {
      expect(CesiumMath.negativePiToPi((3 * Math.PI) / 2)).toBeCloseTo(-Math.PI / 2, 12);
    });

    it("negativePiToPi 区间内的角保持不变", () => {
      expect(CesiumMath.negativePiToPi(0.5)).toBeCloseTo(0.5, 12);
      expect(CesiumMath.negativePiToPi(-0.5)).toBeCloseTo(-0.5, 12);
    });
  });

  describe("equalsEpsilon", () => {
    it("绝对误差判定", () => {
      expect(CesiumMath.equalsEpsilon(1.0, 1.0 + 1e-7, 0, 1e-6)).toBe(true);
      expect(CesiumMath.equalsEpsilon(1.0, 1.0 + 1e-5, 0, 1e-6)).toBe(false);
    });

    it("相对误差判定", () => {
      // 大数值之间的小差异由相对 epsilon 判定
      expect(CesiumMath.equalsEpsilon(1e9, 1e9 + 1, 1e-9)).toBe(true);
    });
  });

  describe("isPowerOfTwo / nextPowerOfTwo", () => {
    it("isPowerOfTwo 判定", () => {
      expect(CesiumMath.isPowerOfTwo(1)).toBe(true);
      expect(CesiumMath.isPowerOfTwo(4)).toBe(true);
      expect(CesiumMath.isPowerOfTwo(1024)).toBe(true);
      expect(CesiumMath.isPowerOfTwo(3)).toBe(false);
      expect(CesiumMath.isPowerOfTwo(0)).toBe(false);
    });

    it("nextPowerOfTwo 向上取整到 2 的幂", () => {
      expect(CesiumMath.nextPowerOfTwo(1)).toBe(1);
      expect(CesiumMath.nextPowerOfTwo(5)).toBe(8);
      expect(CesiumMath.nextPowerOfTwo(8)).toBe(8);
      expect(CesiumMath.nextPowerOfTwo(1023)).toBe(1024);
    });
  });
});
