import { describe, expect, it } from 'vitest';
import { imageryTileFullyInside } from '../../src/scene/Globe';

/**
 * 单层绘制闸门（2026-10-23 拉远闪烁回归）：地形覆盖区内的影像 flat 瓦片
 * 被 drawTiles 跳过（draping 已让地形自带纹理；两壳深度仅差数十米，拉远
 * 视距下深度裁决退化为逐帧交替=「一层起伏一层平坦」闪）。本测试钉住
 * 「完全在覆盖区内才跳过」的边界语义：跨界瓦片保留（兜底覆盖区外缘）。
 *
 * 参考瓦片坐标（z8，Google 行序 y=0 北）：
 * - 8/205/104 → [108.28,109.69]×[30.74,32.0] 完全在湖北 [108,116]×[29,34] 内
 * - 8/210/104 → east=116.72 跨东界；8/204/104 → west=106.88 跨西界
 * - 8/205/101 → north=35.4 跨北界；8/213/105 → 上海 121°E 在外
 */
describe('imageryTileFullyInside（单层绘制覆盖判定）', () => {
  const HUBEI = { w: 108.0, s: 29.0, e: 116.0, n: 34.0 };

  it('完全在覆盖区内的影像瓦片 → true（跳过 flat 绘制）', () => {
    expect(imageryTileFullyInside('8/205/104', HUBEI)).toBe(true);
  });

  it('跨界瓦片 → false（保留绘制兜底覆盖区外缘）', () => {
    expect(imageryTileFullyInside('8/210/104', HUBEI)).toBe(false); // 跨 116°E
    expect(imageryTileFullyInside('8/204/104', HUBEI)).toBe(false); // 跨 108°E
    expect(imageryTileFullyInside('8/205/101', HUBEI)).toBe(false); // 跨 34°N
  });

  it('完全在覆盖区外的瓦片 → false', () => {
    expect(imageryTileFullyInside('8/213/105', HUBEI)).toBe(false); // 上海
  });

  it('非法 key → false（保守：不跳过）', () => {
    expect(imageryTileFullyInside('terrain:12/6614/2775', HUBEI)).toBe(false);
    expect(imageryTileFullyInside('', HUBEI)).toBe(false);
    expect(imageryTileFullyInside('a/b/c', HUBEI)).toBe(false);
  });
});
