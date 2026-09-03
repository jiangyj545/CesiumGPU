import { describe, it, expect } from 'vitest';
import { decodeTerrariumPixel } from '../../src/terrain/TerrariumProvider';

describe('decodeTerrariumPixel', () => {
  it('海平面 (128,0,0) → 0m', () => {
    expect(decodeTerrariumPixel(128, 0, 0)).toBe(0);
  });

  it('正高程 (128,100,0) → 100m', () => {
    expect(decodeTerrariumPixel(128, 100, 0)).toBe(100);
  });

  it('任务给定值 (127,255,253) → -0.01171875m（-3/256）', () => {
    // 127*256 + 255 + 253/256 - 32768 = -3/256
    expect(decodeTerrariumPixel(127, 255, 253)).toBeCloseTo(-0.01171875, 10);
  });

  it('小数部分来自 B 通道 (130,64,128) → 576.5m', () => {
    // 130*256 + 64 + 128/256 - 32768 = 576.5
    expect(decodeTerrariumPixel(130, 64, 128)).toBe(576.5);
  });

  it('海洋负值保留 (127,255,255) → -0.00390625m', () => {
    expect(decodeTerrariumPixel(127, 255, 255)).toBeCloseTo(-0.00390625, 10);
  });

  it('千米级高程 (141,164,0) → 3492m', () => {
    // 141*256 + 164 - 32768 = 36096 + 164 - 32768 = 3492
    expect(decodeTerrariumPixel(141, 164, 0)).toBe(3492);
  });
});
