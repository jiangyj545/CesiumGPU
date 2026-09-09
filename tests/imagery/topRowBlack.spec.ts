import { describe, expect, it } from 'vitest';
import {
  analyseTopRows,
  computeRowMeans,
  BLACK_ROW_THRESHOLD,
  SAMPLE_STRIDE_X,
} from '../../src/imagery/TopRowBlackFill';

/**
 * 顶行（y === 0）"深蓝黑填充数据"检测纯函数单测（node 环境，不依赖 canvas）。
 *
 * 数据事实（curl 实测天地图/高德卫星影像，见 TopRowBlackFill.ts 文件头）：
 * 黑行平均亮度 10~15，有效影像 ≥ 60，阈值 20 居中分割；
 * z2 顶行黑带 43~104 行不等，z5 顶行整张黑。
 */

/** 构造一批"行平均亮度"序列：顶部 `lead` 行黑（亮度 12），其余为有效影像（亮度 90）。 */
function rowMeansWithLead(totalRows: number, lead: number): number[] {
  return Array.from({ length: totalRows }, (_, y) => (y < lead ? 12 : 90));
}

describe('analyseTopRows：顶部连续黑行计数与整张黑判定', () => {
  it('正常影像（无黑行）→ lead = 0，不触发任何治理', () => {
    const r = analyseTopRows(rowMeansWithLead(256, 0));
    expect(r.leadBlackRows).toBe(0);
    expect(r.allBlack).toBe(false);
  });

  it('z2 顶行实测最坏情形：顶部 104 行黑 → lead = 104，未整张黑（不弃瓦片）', () => {
    const r = analyseTopRows(rowMeansWithLead(256, 104));
    expect(r.leadBlackRows).toBe(104);
    expect(r.allBlack).toBe(false);
  });

  it('整张黑（256 行全黑）→ allBlack = true（走祖先回退链）', () => {
    const r = analyseTopRows(rowMeansWithLead(256, 256));
    expect(r.leadBlackRows).toBe(256);
    expect(r.allBlack).toBe(true);
  });

  it('整张黑但底部残留 ≤ 4 行暗色过渡 → 仍判整张黑（尾部容差）', () => {
    const rows = rowMeansWithLead(256, 253); // 前 253 行黑，最后 3 行亮度 90
    rows[255] = 14; // 底部残留暗色过渡行，不应推翻整张黑判定
    const r = analyseTopRows(rows);
    expect(r.leadBlackRows).toBe(253);
    expect(r.allBlack).toBe(true);
  });

  it('顶部 4 行黑 → lead = 4，allBlack = false（低于 MIN_CROP_LEAD_ROWS = 8 不裁剪）', () => {
    const r = analyseTopRows(rowMeansWithLead(256, 4));
    expect(r.leadBlackRows).toBe(4);
    expect(r.allBlack).toBe(false);
    expect(r.leadBlackRows).toBeLessThan(8);
  });

  it('默认阈值 BLACK_ROW_THRESHOLD = 20；亮度 15 判黑、60 判有效（实测数据分离带）', () => {
    expect(BLACK_ROW_THRESHOLD).toBe(20);
    expect(analyseTopRows([15, 15, 90]).leadBlackRows).toBe(2);
    expect(analyseTopRows([60, 15]).leadBlackRows).toBe(0);
  });

  it('阈值可传入覆盖', () => {
    expect(analyseTopRows([25, 25], 30).leadBlackRows).toBe(2);
    expect(analyseTopRows([25, 25], 20).leadBlackRows).toBe(0);
  });

  it('空序列不误判整张黑（length - 4 为负的边界保护）', () => {
    const r = analyseTopRows([]);
    expect(r.leadBlackRows).toBe(0);
    expect(r.allBlack).toBe(false);
  });
});

describe('computeRowMeans：RGBA 像素缓冲逐行平均亮度（隔列采样）', () => {
  it('亮度公式 0.299R + 0.587G + 0.114B：纯色行均值与手算一致', () => {
    // 深蓝黑填充行实测 RGB ≈ (7, 6, 40)：0.299·7 + 0.587·6 + 0.114·40 ≈ 10.96
    const w = 8;
    const h = 2;
    const px = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      px[i * 4] = 7; px[i * 4 + 1] = 6; px[i * 4 + 2] = 40; px[i * 4 + 3] = 255;
    }
    const means = computeRowMeans(px, w, h);
    expect(means).toHaveLength(2);
    expect(means[0]).toBeCloseTo(0.299 * 7 + 0.587 * 6 + 0.114 * 40, 6);
    expect(means[0]).toBeLessThan(BLACK_ROW_THRESHOLD);
  });

  it('逐行独立：上半黑（(9,13,40)）、下半亮（200,200,200）→ 两行均值分跨阈值两侧', () => {
    const w = 8;
    const px = new Uint8ClampedArray(2 * w * 4);
    const fill = (y: number, r: number, g: number, b: number): void => {
      for (let x = 0; x < w; x++) {
        const o = (y * w + x) * 4;
        px[o] = r; px[o + 1] = g; px[o + 2] = b; px[o + 3] = 255;
      }
    };
    fill(0, 9, 13, 40);
    fill(1, 200, 200, 200);
    const means = computeRowMeans(px, w, 2);
    expect(means[0]).toBeCloseTo(0.299 * 9 + 0.587 * 13 + 0.114 * 40, 6);
    expect(means[1]).toBeCloseTo(200, 6);
    expect(means[0]).toBeLessThan(BLACK_ROW_THRESHOLD);
    expect(means[1]).toBeGreaterThan(60); // 实测有效影像行下界
  });

  it('隔列采样（步长 4）：均匀行均值不受采样影响；列向不均匀行仍取到中间列', () => {
    // 均匀行：采样步长不改变均值
    const w = 16;
    const px = new Uint8ClampedArray(w * 4).fill(0);
    for (let x = 0; x < w; x++) { px[x * 4] = 100; px[x * 4 + 3] = 255; }
    expect(computeRowMeans(px, w, 1)[0]).toBeCloseTo(0.299 * 100, 6);

    // 不均匀行：只有第 8 列（采样点之一，8 % 4 === 0）为亮，其余黑
    const px2 = new Uint8ClampedArray(w * 4).fill(0);
    px2[8 * 4] = 255; px2[8 * 4 + 1] = 255; px2[8 * 4 + 2] = 255;
    expect(computeRowMeans(px2, w, 1)[0]).toBeCloseTo(255 / (w / SAMPLE_STRIDE_X), 6);
  });

  it('analyseTopRows × computeRowMeans 端到端（合成瓦片）：顶 43 黑行、其余有效 → lead = 43', () => {
    const w = 8;
    const h = 64;
    const px = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const o = (y * w + x) * 4;
        if (y < 43) { px[o] = 7; px[o + 1] = 6; px[o + 2] = 40; }
        else        { px[o] = 180; px[o + 1] = 170; px[o + 2] = 150; }
        px[o + 3] = 255;
      }
    }
    const r = analyseTopRows(computeRowMeans(px, w, h));
    expect(r.leadBlackRows).toBe(43);
    expect(r.allBlack).toBe(false);
  });
});
