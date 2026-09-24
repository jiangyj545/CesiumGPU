import { describe, expect, it } from 'vitest';
import { ACES_FIT, acesFilmic, reinhard, toneMapFunction, applyGamma } from '../../src/postprocessing/ToneMapping';

/**
 * ACES 拟合曲线性质测试（P2 ToneMappingPass 的 CPU 参考实现）。
 *
 * 曲线（与 ToneMappingShader.ts WGSL 同系数）：
 *   f(x) = x(1.4x+1)/(0.7x²+1.274x+1)，x ≥ 0.0301；线性段 f=x。
 */
describe('ToneMapping ACES 曲线', () => {
  it('f(0) = 0', () => {
    expect(acesFilmic(0)).toBe(0);
  });

  it('f(1) ∈ (0.8, 1.0) —— 白点落入显示域', () => {
    const f1 = acesFilmic(1);
    expect(f1).toBeGreaterThan(0.8);
    expect(f1).toBeLessThan(1.0);
  });

  it('f(0.18) ≈ 0.18 ±10% —— 感知中灰保持', () => {
    const f = acesFilmic(0.18);
    expect(f).toBeGreaterThanOrEqual(0.18 * 0.9);
    expect(f).toBeLessThanOrEqual(0.18 * 1.1);
  });

  it('全域单调递增（[0, 16] 细步进）', () => {
    let prev = acesFilmic(0);
    for (let x = 0.0005; x <= 16; x += 0.0005) {
      const cur = acesFilmic(x);
      expect(cur).toBeGreaterThanOrEqual(prev);
      prev = cur;
    }
  });

  it('负值钳到 0，NaN 安全', () => {
    expect(acesFilmic(-1)).toBe(0);
    expect(Number.isFinite(acesFilmic(NaN))).toBe(true);
    expect(acesFilmic(NaN)).toBe(0);
  });

  it('线性段 x < 0.0301 直通，knee 处 C0 连续', () => {
    const below = acesFilmic(ACES_FIT.X_LIN - 1e-4);
    expect(below).toBeCloseTo(ACES_FIT.X_LIN - 1e-4, 4);
    const jump = Math.abs(
      acesFilmic(ACES_FIT.X_LIN + 1e-6) - acesFilmic(ACES_FIT.X_LIN - 1e-6),
    );
    expect(jump).toBeLessThan(1e-3);
  });

  it('reinhard 与旧 LightingShader 内嵌公式一致', () => {
    expect(reinhard(1)).toBeCloseTo(0.5, 12);
    expect(reinhard(0)).toBe(0);
  });

  it('toneMapFunction 按模式返回对应映射', () => {
    expect(toneMapFunction('aces')(0.5)).toBeCloseTo(acesFilmic(0.5), 12);
    expect(toneMapFunction('reinhard')(0.5)).toBeCloseTo(reinhard(0.5), 12);
    expect(toneMapFunction('none')(0.5)).toBe(0.5);
    expect(toneMapFunction('none')(-0.5)).toBe(0);
  });

  it('applyGamma 即 sRGB 近似', () => {
    expect(applyGamma(1)).toBeCloseTo(1, 12);
    expect(applyGamma(0)).toBeCloseTo(0, 12);
    expect(applyGamma(0.5)).toBeGreaterThan(0.5); // gamma 提亮
  });
});
