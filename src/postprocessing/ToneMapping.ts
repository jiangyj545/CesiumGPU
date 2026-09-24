/**
 * ToneMapping - 色调映射纯函数（P2，CPU 参考实现）。
 *
 * WGSL 对应实现见 `src/shaders/postprocess/ToneMappingShader.ts`
 * （同一组系数，两处必须同步修改）。本文件供 node 测试直接驱动
 * （acesCurve.spec.ts），不依赖 GPU。
 *
 * 曲线设计（soft-knee 有理多项式拟合，对齐 ACES filmic 的中灰响应）：
 *   f(x) = x·(A·x + B) / (x·(C·x + D) + E)，x ≥ X_LIN
 *   f(x) = x，x < X_LIN（线性段，斜率 B/E = 1 → knee 处 C0 连续）
 *
 * 已验证性质（见 tests/postprocess/acesCurve.spec.ts）：
 *   f(0) = 0；f(1) ≈ 0.807 ∈ (0.8, 1.0)；f(0.18) ≈ 0.180（感知中灰 ±10%）；
 *   全域单调递增（f′ 分子三项系数全正，无驻点）。
 */

/** 拟合系数（与 ToneMappingShader.ts 中 WGSL 常量保持一致）。 */
export const ACES_FIT = {
  A: 1.4,
  B: 1.0,
  C: 0.7,
  D: 1.274,
  E: 1.0,
  /** 线性段上界（此处曲线斜率 ≈ 1，与线性段无缝衔接）。 */
  X_LIN: 0.0301,
} as const;

/** sRGB gamma（显示 ODT 近似）。 */
export function applyGamma(c: number, gamma = 1 / 2.2): number {
  return Math.pow(Math.max(c, 0), gamma);
}

/**
 * ACES filmic 拟合曲线（单通道，线性域输入）。
 * 高段不回卷：>1 的输入持续变亮，由 shader 输出端 clamp 到 1
 * （"shoulder to white"的简化形态）。
 */
export function acesFilmic(x: number): number {
  if (!(x >= ACES_FIT.X_LIN)) {
    // 线性段直通；NaN 在此处被过滤（Math.max(NaN,0)=NaN 会穿透，须显式处理）
    const v = Math.max(x, 0);
    return Number.isFinite(v) ? v : 0;
  }
  const num = x * (ACES_FIT.A * x + ACES_FIT.B);
  const den = x * (ACES_FIT.C * x + ACES_FIT.D) + ACES_FIT.E;
  return num / den;
}

/** Reinhard（与旧 LightingShader :135-136 内嵌公式一致）。 */
export function reinhard(x: number): number {
  return Math.max(x, 0) / (Math.max(x, 0) + 1);
}

export type ToneMappingFn = (x: number) => number;

/** 按模式返回单通道映射函数（不含 gamma —— gamma 由调用方决定是否叠加）。 */
export function toneMapFunction(mode: 'none' | 'reinhard' | 'aces'): ToneMappingFn {
  switch (mode) {
    case 'reinhard': return reinhard;
    case 'aces':     return acesFilmic;
    case 'none':
    default:         return (x) => Math.max(x, 0);
  }
}
