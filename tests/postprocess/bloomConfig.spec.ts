import { describe, expect, it } from 'vitest';
import {
  BLOOM_LEVELS_MAX,
  BLOOM_LEVELS_MIN,
  defaultPostProcessConfig,
  sanitizeBloomParams,
  sanitizeGhostCount,
  validatePostProcessConfig,
} from '../../src/postprocessing/PostProcessConfig';

/**
 * 后处理配置夹取/校验测试（P2）。
 * 约定：非法值不抛异常打崩渲染循环 —— sanitize 夹取到合法域，
 * validate 返回告警列表（空 = 合法）。
 */
describe('PostProcessConfig 校验与夹取', () => {
  const base = () => defaultPostProcessConfig();

  it('levels clamp 到 [4, 6]', () => {
    expect(sanitizeBloomParams({ enabled: true, threshold: 1, knee: 0.5, intensity: 1, levels: 2 }).levels)
      .toBe(BLOOM_LEVELS_MIN);
    expect(sanitizeBloomParams({ enabled: true, threshold: 1, knee: 0.5, intensity: 1, levels: 99 }).levels)
      .toBe(BLOOM_LEVELS_MAX);
    expect(sanitizeBloomParams({ enabled: true, threshold: 1, knee: 0.5, intensity: 1, levels: 5.6 }).levels)
      .toBe(6); // 四舍五入后仍在域内
    expect(sanitizeBloomParams({ enabled: true, threshold: 1, knee: 0.5, intensity: 1, levels: 4.4 }).levels)
      .toBe(4);
  });

  it('threshold 边界 [0, 4]，负值/NaN 安全', () => {
    expect(sanitizeBloomParams({ enabled: true, threshold: -1, knee: 0.5, intensity: 1, levels: 5 }).threshold).toBe(0);
    expect(sanitizeBloomParams({ enabled: true, threshold: 9, knee: 0.5, intensity: 1, levels: 5 }).threshold).toBe(4);
    expect(sanitizeBloomParams({ enabled: true, threshold: NaN, knee: 0.5, intensity: 1, levels: 5 }).threshold).toBe(0);
  });

  it('knee 边界 [0, 1]', () => {
    expect(sanitizeBloomParams({ enabled: true, threshold: 1, knee: -0.5, intensity: 1, levels: 5 }).knee).toBe(0);
    expect(sanitizeBloomParams({ enabled: true, threshold: 1, knee: 3, intensity: 1, levels: 5 }).knee).toBe(1);
  });

  it('intensity 边界 [0, 10]', () => {
    expect(sanitizeBloomParams({ enabled: true, threshold: 1, knee: 0.5, intensity: -2, levels: 5 }).intensity).toBe(0);
    expect(sanitizeBloomParams({ enabled: true, threshold: 1, knee: 0.5, intensity: 50, levels: 5 }).intensity).toBe(10);
  });

  it('ghosts clamp 到 [4, 8]', () => {
    expect(sanitizeGhostCount(1)).toBe(4);
    expect(sanitizeGhostCount(6)).toBe(6);
    expect(sanitizeGhostCount(20)).toBe(8);
    expect(sanitizeGhostCount(NaN)).toBe(4);
  });

  it('默认配置合法（enabled=false 走旧路径）', () => {
    const cfg = base();
    expect(validatePostProcessConfig(cfg)).toEqual([]);
    expect(cfg.enabled).toBe(false);
  });

  it('validate 拒绝非法值并逐项告警', () => {
    const cfg = base();
    cfg.enabled = true;
    cfg.bloom.threshold = -1;
    cfg.bloom.knee = 2;
    cfg.bloom.intensity = NaN;
    cfg.bloom.levels = 3;
    cfg.lensFlare.ghosts = 99;
    (cfg.toneMapping as unknown as string) = 'filmic'; // 非法模式
    const problems = validatePostProcessConfig(cfg);
    expect(problems.length).toBeGreaterThanOrEqual(6);
    expect(problems.some((p) => p.includes('threshold'))).toBe(true);
    expect(problems.some((p) => p.includes('knee'))).toBe(true);
    expect(problems.some((p) => p.includes('intensity'))).toBe(true);
    expect(problems.some((p) => p.includes('levels'))).toBe(true);
    expect(problems.some((p) => p.includes('ghosts'))).toBe(true);
    expect(problems.some((p) => p.includes('toneMapping'))).toBe(true);
  });

  it('Manager.getConfig/validate/getLevelSizes 与纯函数一致', async () => {
    const { PostProcessManager } = await import('../../src/postprocessing/PostProcessManager');
    // 桩 device：构造路径不触碰 GPU
    const manager = new PostProcessManager({} as unknown as GPUDevice, 'bgra8unorm');
    expect(manager.validate()).toEqual([]);
    manager.bloom.levels = 10;
    expect(manager.validate().some((p) => p.includes('levels'))).toBe(true);
    const sizes = manager.getLevelSizes();
    expect(sizes.down.length).toBe(6); // clamp 后 6 级
    expect(sizes.up.length).toBe(5);
  });
});
