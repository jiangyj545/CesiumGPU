/**
 * weatherFx.spec.ts – P2-7 纯逻辑：
 * 雾因子随距离/高度单调、闪电包络峰值/衰减时间常数、参数钳制校验。
 */
import { describe, expect, it } from 'vitest';
import {
  computeFogFactor,
  lightningEnvelope,
  sanitizeFogParams,
  sanitizeLightningParams,
} from '../../src/effect/WeatherMath';
import { FogEffect, LightningEffect } from '../../src/effect/WeatherFx';
import { Context } from '../../src/renderer/Context';

describe('computeFogFactor（指数距离 + 高度衰减雾）', () => {
  it('距离单调递增（同高度），值域 [0,1)」，远距 → 1', () => {
    let prev = -1;
    for (const dist of [0, 1000, 10_000, 100_000, 1e6, 1e8]) {
      const f = computeFogFactor(dist, 0, 0.5, 0.12);
      expect(f).toBeGreaterThanOrEqual(prev);
      expect(f).toBeLessThanOrEqual(1);
      prev = f;
    }
    expect(computeFogFactor(0, 0, 0.5, 0.12)).toBe(0);
    expect(computeFogFactor(1e9, 0, 1, 0)).toBeCloseTo(1, 6);
  });

  it('高度单调递减（逆温层：像素海拔越高雾越薄）', () => {
    let prev = 2;
    for (const h of [0, 500, 2000, 8000, 20_000]) {
      const f = computeFogFactor(50_000, h, 0.6, 0.3);
      expect(f).toBeLessThanOrEqual(prev);
      prev = f;
    }
  });

  it('density=0 恒 0；负距离/负高度钳制安全', () => {
    expect(computeFogFactor(1e6, 100, 0, 0.5)).toBe(0);
    expect(computeFogFactor(-5, -100, 1, 1)).toBe(0);
  });
});

describe('lightningEnvelope（急升-衰减-余辉）', () => {
  const attack = 0.02, decay = 0.08, afterglow = 0.5;

  it('t<0 恒 0（未触发）；攻击段线性升至峰值 1', () => {
    expect(lightningEnvelope(-1, attack, decay, afterglow)).toBe(0);
    expect(lightningEnvelope(0, attack, decay, afterglow)).toBe(0);
    expect(lightningEnvelope(attack / 2, attack, decay, afterglow)).toBeCloseTo(0.5, 9);
    expect(lightningEnvelope(attack, attack, decay, afterglow)).toBeCloseTo(1, 9);
  });

  it('峰值恰在 t=attack；之后单调衰减；时间常数数量级正确', () => {
    const peak = lightningEnvelope(attack, attack, decay, afterglow);
    for (const t of [attack + 0.01, attack + decay, attack + 5 * decay]) {
      expect(lightningEnvelope(t, attack, decay, afterglow)).toBeLessThan(peak);
    }
    // 攻击后 1 个 decay 主衰减项 0.75→0.276 + 余辉 0.25→0.245。
    const v1 = lightningEnvelope(attack + decay, attack, decay, afterglow);
    expect(v1).toBeCloseTo(0.75 * Math.exp(-1) + 0.25 * Math.exp(-decay / afterglow), 9);
    // 长尾由 afterglow 主导：t=attack+2s 仍有可感知余辉。
    // 长尾由 afterglow 主导：t=attack+1s 仍有可感知余辉（0.25·e^-2 ≈ 0.034）。
    expect(lightningEnvelope(attack + 1.0, attack, decay, afterglow)).toBeGreaterThan(0.01);
  });
});

describe('参数钳制', () => {
  it('Fog：负 density/heightFalloff 钳 0，skyTint 钳 [0,1]，默认填充', () => {
    const p = sanitizeFogParams({ density: -1, heightFalloff: -3, skyTint: 5 });
    expect(p.density).toBe(0);
    expect(p.heightFalloff).toBe(0);
    expect(p.skyTint).toBe(1);
    expect(p.tintColor).toEqual([0.78, 0.82, 0.88, 1]);
  });

  it('Lightning：intensity/chance 钳 [0,1]，间隔钳非负', () => {
    const p = sanitizeLightningParams({ intensity: 2, doubleStrikeChance: -1, meanIntervalSec: -5 });
    expect(p.intensity).toBe(1);
    expect(p.doubleStrikeChance).toBe(0);
    expect(p.meanIntervalSec).toBe(0);
    expect(p.color).toEqual([0.85, 0.9, 1.0]);
  });
});

describe('LightningEffect 时序（GPU-free：包络/序列状态机）', () => {
  const makeEffect = (rng: () => number) =>
    new LightningEffect({} as Context, { intensity: 0.8, meanIntervalSec: 0 }, rng);

  it('strike → 包络升起衰减；未 strike 恒 0', () => {
    const fx = makeEffect(() => 0.99);
    expect(fx.currentEnvelope).toBe(0);
    fx.strike();
    expect(fx.currentEnvelope).toBe(0); // t=0 恰为 0（attack 起点）
    fx.step(0.01);
    expect(fx.currentEnvelope).toBeGreaterThan(0.4);
    fx.step(0.5);
    expect(fx.currentEnvelope).toBeLessThan(0.2);
    fx.step(5);
    expect(fx.currentEnvelope).toBeLessThan(0.01);
  });

  it('双闪：第二次 strike 在主衰减后重置包络', () => {
    const fx = makeEffect(() => 0); // rng=0 → 不触发双闪
    fx.strike();
    fx.step(1.0);
    const before = fx.currentEnvelope;
    expect(before).toBeLessThan(0.1);
    fx.strike();
    fx.step(0.01);
    expect(fx.currentEnvelope).toBeGreaterThan(before);
    expect(fx.currentEnvelope).toBeGreaterThan(0.4);
  });

  it('随机自动序列：chance 越过 rng 阈值时触发（泊松步进）', () => {
    const fx = new LightningEffect(
      {} as Context,
      { intensity: 0.8, meanIntervalSec: 0.01 },
      () => 0.5 // chance > 0.5 即触发：timer > mean·ln2 ≈ 0.0069s
    );
    fx.step(0.001); // chance ≈ 0.095 < 0.5：不触发
    expect(fx.currentEnvelope).toBe(0);
    fx.step(0.01); // 累计 0.011s → chance ≈ 0.67 > 0.5：触发（t=0）
    expect(fx.currentEnvelope).toBe(0);
    fx.step(0.005); // 不再触发（timer 重置）；包络推进到 0.005s
    expect(fx.currentEnvelope).toBeGreaterThan(0.1);
  });

  it('手动模式（meanInterval=0）step 不自动触发', () => {
    const fx = makeEffect(() => 0);
    fx.step(100);
    expect(fx.currentEnvelope).toBe(0);
  });
});

describe('FogEffect 参数面', () => {
  it('setDensity / setParams 钳制生效', () => {
    const fx = new FogEffect({} as Context, { density: 0.4 });
    fx.setDensity(3);
    expect(fx.density).toBe(1);
    fx.setParams({ skyTint: -1 });
    expect(fx.params.skyTint).toBe(0);
    expect(fx.params.density).toBe(1); // 未给出的字段保留
  });
});
