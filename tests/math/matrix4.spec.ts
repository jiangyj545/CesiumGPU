import { describe, it, expect } from "vitest";
import { Matrix4 } from "../../src/math/Matrix4";
import { Cartesian3 } from "../../src/math/Cartesian3";

const PRECISION = 1e-6;

function expectMatrixClose(actual: Float32Array, expected: number[]): void {
  expect(actual.length).toBe(16);
  for (let i = 0; i < 16; i++) {
    expect(actual[i]).toBeCloseTo(expected[i], 6);
  }
}

function isIdentity(m: Matrix4): boolean {
  const v = m.values;
  return (
    v[0] === 1 && v[5] === 1 && v[10] === 1 && v[15] === 1 &&
    v[1] === 0 && v[2] === 0 && v[3] === 0 && v[4] === 0 &&
    v[6] === 0 && v[7] === 0 && v[8] === 0 && v[9] === 0 &&
    v[11] === 0 && v[12] === 0 && v[13] === 0 && v[14] === 0
  );
}

describe("Matrix4", () => {
  describe("multiply", () => {
    it("单位阵乘任意矩阵保持不变", () => {
      const m = Matrix4.fromTranslation(new Cartesian3(1, 2, 3));
      const r = Matrix4.multiply(Matrix4.IDENTITY, m, new Matrix4());
      expectMatrixClose(r.values, Array.from(m.values));
      const r2 = Matrix4.multiply(m, Matrix4.IDENTITY, new Matrix4());
      expectMatrixClose(r2.values, Array.from(m.values));
    });

    it("零阵乘任意矩阵得零阵", () => {
      const m = Matrix4.fromUniformScale(3);
      const r = Matrix4.multiply(Matrix4.ZERO, m, new Matrix4());
      expectMatrixClose(r.values, new Array(16).fill(0));
    });

    it("已知值：平移×缩放（列主序）", () => {
      const t = Matrix4.fromTranslation(new Cartesian3(1, 2, 3));
      const s = Matrix4.fromScale(new Cartesian3(2, 3, 4));
      const ts = Matrix4.multiply(t, s, new Matrix4());
      // 列0=(2,0,0,0) 列1=(0,3,0,0) 列2=(0,0,4,0) 列3=(1,2,3,1)
      expectMatrixClose(ts.values, [
        2, 0, 0, 0,
        0, 3, 0, 0,
        0, 0, 4, 0,
        1, 2, 3, 1,
      ]);
      const st = Matrix4.multiply(s, t, new Matrix4());
      expectMatrixClose(st.values, [
        2, 0, 0, 0,
        0, 3, 0, 0,
        0, 0, 4, 0,
        2, 6, 12, 1,
      ]);
    });

    it("不满足交换律：t*s 与 s*t 平移分量不同", () => {
      const t = Matrix4.fromTranslation(new Cartesian3(1, 2, 3));
      const s = Matrix4.fromUniformScale(2);
      const ts = Matrix4.multiply(t, s, new Matrix4());
      const st = Matrix4.multiply(s, t, new Matrix4());
      expect(ts.values[12]).not.toBe(st.values[12]);
    });

    it("满足结合律：(A*B)*C === A*(B*C)", () => {
      const a = Matrix4.fromRotationX(0.3);
      const b = Matrix4.fromRotationY(-1.1);
      const c = Matrix4.fromTranslation(new Cartesian3(5, -2, 7));
      const abC = Matrix4.multiply(Matrix4.multiply(a, b, new Matrix4()), c, new Matrix4());
      const aBC = Matrix4.multiply(a, Matrix4.multiply(b, c, new Matrix4()), new Matrix4());
      for (let i = 0; i < 16; i++) {
        expect(abC.values[i]).toBeCloseTo(aBC.values[i], 6);
      }
    });

    it("结果写入传入的 result 参数", () => {
      const out = new Matrix4();
      const r = Matrix4.multiply(Matrix4.IDENTITY, Matrix4.IDENTITY, out);
      expect(r).toBe(out);
      expect(isIdentity(r)).toBe(true);
    });
  });

  describe("inverse", () => {
    it("单位阵的逆为单位阵", () => {
      const inv = Matrix4.inverse(Matrix4.IDENTITY, new Matrix4());
      expect(isIdentity(inv)).toBe(true);
    });

    it("可逆矩阵：M * inverse(M) ≈ 单位阵", () => {
      const m = Matrix4.multiply(
        Matrix4.fromRotationZ(Math.PI / 6),
        Matrix4.fromTranslation(new Cartesian3(10, -4, 2.5)),
        new Matrix4()
      );
      const inv = Matrix4.inverse(m, new Matrix4());
      const product = Matrix4.multiply(m, inv, new Matrix4());
      expectMatrixClose(product.values, [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
      ]);
    });

    it("已知值：平移矩阵的逆为反向平移", () => {
      const t = Matrix4.fromTranslation(new Cartesian3(3, -7, 0.5));
      const inv = Matrix4.inverse(t, new Matrix4());
      expectMatrixClose(inv.values, [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        -3, 7, -0.5, 1,
      ]);
    });

    it("奇异矩阵（零阵）：按当前实现返回单位阵（非 undefined）", () => {
      const inv = Matrix4.inverse(Matrix4.ZERO, new Matrix4());
      expect(isIdentity(inv)).toBe(true);
    });

    it("奇异矩阵（行重复）：按当前实现返回单位阵", () => {
      const singular = Matrix4.fromArray([
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
      ]);
      const inv = Matrix4.inverse(singular, new Matrix4());
      expect(isIdentity(inv)).toBe(true);
    });
  });

  describe("perspective", () => {
    it("已知值：fov=90°, aspect=2, near=1, far=3", () => {
      const p = Matrix4.perspective(Math.PI / 2, 2, 1, 3);
      const f = 1 / Math.tan(Math.PI / 4); // = 1
      expectMatrixClose(p.values, [
        f / 2, 0, 0, 0,
        0, f, 0, 0,
        0, 0, 3 / (1 - 3), -1,
        0, 0, (3 * 1) / (1 - 3), 0,
      ]);
      expect(p.values[0]).toBeCloseTo(0.5, 6);
      expect(p.values[5]).toBeCloseTo(1, 6);
      expect(p.values[10]).toBeCloseTo(-1.5, 6);
      expect(p.values[11]).toBe(-1);
      expect(p.values[14]).toBeCloseTo(-1.5, 6);
      [1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 15].forEach((i) => {
        expect(p.values[i]).toBe(0);
      });
    });

    it("非 90° 视锥角：fov=60° 时 f=√3", () => {
      const p = Matrix4.perspective(Math.PI / 3, 1, 0.5, 100);
      const f = 1 / Math.tan(Math.PI / 6); // = √3
      expect(p.values[0]).toBeCloseTo(f, 6);
      expect(p.values[5]).toBeCloseTo(f, 6);
    });

    it("perspectiveFieldOfView 使用 reversed-Z：[10] = near/(near-far)", () => {
      const p = Matrix4.perspectiveFieldOfView(Math.PI / 2, 2, 1, 3);
      expect(p.values[10]).toBeCloseTo(1 / (1 - 3), 6); // -0.5
      expect(p.values[14]).toBeCloseTo((3 * 1) / (1 - 3), 6); // -1.5
      expect(p.values[11]).toBe(-1);
    });

    it("near 平面映射：标准 perspective 下 z=-near → NDC z=0", () => {
      const p = Matrix4.perspective(Math.PI / 2, 1, 1, 3);
      const zEye = -1;
      const zClip = p.values[10] * zEye + p.values[14];
      const wClip = -zEye;
      expect(zClip / wClip).toBeCloseTo(0, PRECISION);
    });

    it("far 平面映射：标准 perspective 下 z=-far → NDC z=1", () => {
      const p = Matrix4.perspective(Math.PI / 2, 1, 1, 3);
      const zEye = -3;
      const zClip = p.values[10] * zEye + p.values[14];
      const wClip = -zEye;
      expect(zClip / wClip).toBeCloseTo(1, PRECISION);
    });
  });

  describe("lookAt", () => {
    it("eye=(0,0,5) 看向原点：结果为标准视图矩阵", () => {
      const v = Matrix4.lookAt(
        new Cartesian3(0, 0, 5),
        Cartesian3.ZERO,
        Cartesian3.UNIT_Y
      );
      expectMatrixClose(v.values, [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, -5, 1,
      ]);
    });

    it("旋转部分正交：右/上/前轴两两垂直且为单位长度", () => {
      const v = Matrix4.lookAt(
        new Cartesian3(10, 8, 6),
        new Cartesian3(1, 2, 3),
        Cartesian3.UNIT_Y
      );
      const right = new Cartesian3(v.values[0], v.values[1], v.values[2]);
      const up = new Cartesian3(v.values[4], v.values[5], v.values[6]);
      const back = new Cartesian3(v.values[8], v.values[9], v.values[10]);
      expect(Cartesian3.magnitude(right)).toBeCloseTo(1, 6);
      expect(Cartesian3.magnitude(up)).toBeCloseTo(1, 6);
      expect(Cartesian3.magnitude(back)).toBeCloseTo(1, 6);
      expect(Cartesian3.dot(right, up)).toBeCloseTo(0, 6);
      expect(Cartesian3.dot(right, back)).toBeCloseTo(0, 6);
      expect(Cartesian3.dot(up, back)).toBeCloseTo(0, 6);
    });

    it("eye 经视矩阵变换后映射到原点", () => {
      const eye = new Cartesian3(4, -2, 9);
      const v = Matrix4.lookAt(eye, Cartesian3.ZERO, Cartesian3.UNIT_Y);
      // 列主序：第 row 行 = v[row] + v[row+4] + v[row+8] + v[row+12]
      for (let row = 0; row < 4; row++) {
        const expected = row === 3 ? 1 : 0;
        const r0 = v.values[row], r1 = v.values[row + 4];
        const r2 = v.values[row + 8], r3 = v.values[row + 12];
        expect(r0 * eye.x + r1 * eye.y + r2 * eye.z + r3)
          .toBeCloseTo(expected, 6);
      }
    });
  });

  describe("fromRotationZ", () => {
    it("旋转 90°：已知元素抽查", () => {
      const r = Matrix4.fromRotationZ(Math.PI / 2);
      expectMatrixClose(r.values, [
        0, 1, 0, 0,
        -1, 0, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
      ]);
    });

    it("0° 旋转为单位阵", () => {
      expect(isIdentity(Matrix4.fromRotationZ(0))).toBe(true);
    });
  });

  describe("transpose", () => {
    it("转置后行列互换", () => {
      const m = Matrix4.fromArray([
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
      ]);
      const t = Matrix4.transpose(m, new Matrix4());
      expectMatrixClose(t.values, [
        1, 5, 9, 13,
        2, 6, 10, 14,
        3, 7, 11, 15,
        4, 8, 12, 16,
      ]);
    });

    it("转置两次还原", () => {
      const m = Matrix4.lookAt(
        new Cartesian3(1, 2, 3),
        new Cartesian3(4, 5, 6),
        Cartesian3.UNIT_Y
      );
      const tt = Matrix4.transpose(Matrix4.transpose(m, new Matrix4()), new Matrix4());
      for (let i = 0; i < 16; i++) {
        expect(tt.values[i]).toBeCloseTo(m.values[i], 6);
      }
    });
  });

  describe("getTranslation / fromTranslation / fromScale / fromArray", () => {
    it("fromTranslation 生成正确矩阵", () => {
      const m = Matrix4.fromTranslation(new Cartesian3(1, 2, 3));
      expectMatrixClose(m.values, [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        1, 2, 3, 1,
      ]);
    });

    it("fromScale 生成对角阵", () => {
      const m = Matrix4.fromScale(new Cartesian3(2, 3, 4));
      expectMatrixClose(m.values, [
        2, 0, 0, 0,
        0, 3, 0, 0,
        0, 0, 4, 0,
        0, 0, 0, 1,
      ]);
    });

    it("fromArray 按列主序读取 16 个元素", () => {
      const m = Matrix4.fromArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
      expect(m.values.length).toBe(16);
      expect(m.values[0]).toBe(1);
      expect(m.values[4]).toBe(5);
      expect(m.values[15]).toBe(16);
    });

    it("getTranslation 提取平移分量", () => {
      const m = Matrix4.fromTranslation(new Cartesian3(7, -8, 9));
      const t = m.getTranslation();
      expect(t.x).toBe(7);
      expect(t.y).toBe(-8);
      expect(t.z).toBe(9);
    });
  });
});
