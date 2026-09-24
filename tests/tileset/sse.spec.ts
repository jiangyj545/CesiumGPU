/**
 * sse.spec.ts – SSE 纯函数（computeTileSse / shouldRefine）单测。
 *
 * 公式（与 ImageryLayer 同构，worldSize = geometricError / 6378137）：
 *   sse = (geometricError/R) × viewportHeight / (distance × 2·tan(fov/2))
 */
import { describe, expect, it } from 'vitest';
import { computeTileSse, shouldRefine } from '../../src/tileset/Tileset3D';
import { TILESET_EARTH_SCALE } from '../../src/tileset/TilesetContentLoader';

const R = TILESET_EARTH_SCALE;
const FOV_60 = Math.PI / 3;          // tan(30°) ≈ 0.57735
const VH = 1080;

describe('computeTileSse', () => {
  it('手工数值：g=100m、d=0.001 → sse ≈ 14.67（低于 16 阈值）', () => {
    const sse = computeTileSse(100, 0.001, VH, FOV_60);
    const expected = (100 / R) * (VH / (0.001 * 2 * Math.tan(FOV_60 * 0.5)));
    expect(sse).toBeCloseTo(expected, 9);
    expect(sse).toBeGreaterThan(14);
    expect(sse).toBeLessThan(15);
  });

  it('距离减半 → sse 翻倍（线性反比）', () => {
    const far  = computeTileSse(100, 0.001, VH, FOV_60);
    const near = computeTileSse(100, 0.0005, VH, FOV_60);
    expect(near / far).toBeCloseTo(2, 9);
    expect(near).toBeGreaterThan(16); // 进入细分区
  });

  it('viewportHeight 线性影响；fov 半角增大 → sse 减小', () => {
    const s1080 = computeTileSse(100, 0.001, 1080, FOV_60);
    const s2160 = computeTileSse(100, 0.001, 2160, FOV_60);
    expect(s2160 / s1080).toBeCloseTo(2, 9);

    const wide = computeTileSse(100, 0.001, VH, Math.PI / 2); // tan45° = 1
    expect(wide).toBeLessThan(s1080);
  });

  it('earthScale 可注入（默认 6378137）；distance 下限防除零', () => {
    expect(computeTileSse(6378137, 1, 1000, FOV_60))
      .toBeCloseTo(1000 / (2 * Math.tan(FOV_60 / 2)), 6);
    // 0 距离被钳到 1e-9 → 结果巨大但有限（不产生 Infinity/NaN）
    expect(Number.isFinite(computeTileSse(100, 0, VH, FOV_60))).toBe(true);
  });
});

describe('shouldRefine – 细分决策与边界', () => {
  it('sse 超阈值且有子级 → 细分', () => {
    expect(shouldRefine(29, 16, 0, 16, true)).toBe(true);
  });

  it('sse 低于阈值 → 不细分（选中自身）', () => {
    expect(shouldRefine(14, 16, 0, 16, true)).toBe(false);
  });

  it('maxLevel 边界：level ≥ maximumLevel → 永不细分', () => {
    expect(shouldRefine(9999, 16, 16, 16, true)).toBe(false);
    expect(shouldRefine(9999, 16, 15, 16, true)).toBe(true);
  });

  it('叶子（无子级）→ 不细分', () => {
    expect(shouldRefine(9999, 16, 0, 16, false)).toBe(false);
  });
});
