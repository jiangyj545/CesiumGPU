import { describe, expect, it } from 'vitest';
import { computeImagerySkipKeys } from '../../src/scene/Globe';

/**
 * Tile-exact 影像压制（闪烁修复 v2，2026-10-23 回滚重做版）：
 * 影像 flat 壳在「地形 tile 并集完全覆盖」区域被跳过（拉远视距下两壳
 * 深度裁决退化为逐帧交替闪）；tile 级精确判定保证 mesh 空隙处的影像
 * 保留——v1 外接矩形版开出破洞的教训。测试覆盖集用与
 * geographicTileBounds 互逆的公式动态构造（测并集判定逻辑本身）。
 */

/** 影像瓦片（mercator）的经纬矩形。 */
function mercRect(key: string): { w: number; s: number; e: number; n: number } {
  const [z, x, y] = key.split('/').map(Number);
  const n2 = 2 ** z;
  return {
    w: (x / n2) * 360 - 180,
    e: ((x + 1) / n2) * 360 - 180,
    n: (Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / n2))) * 180) / Math.PI,
    s: (Math.atan(Math.sinh(Math.PI * (1 - (2 * (y + 1)) / n2))) * 180) / Math.PI,
  };
}

/** 完全覆盖影像瓦片矩形的地理 z 级 tile key 集（可剔除若干块模拟空隙）。 */
function coveringTerrainKeys(imgKey: string, z: number, drop: string[] = []): string[] {
  const r = mercRect(imgKey);
  const cols = 2 ** (z + 1);
  const rows = 2 ** z;
  const x0 = Math.floor(((r.w + 180) / 360) * cols);
  const x1 = Math.ceil(((r.e + 180) / 360) * cols) - 1;
  const y0 = Math.floor(((r.s + 90) / 180) * rows);
  const y1 = Math.ceil(((r.n + 90) / 180) * rows) - 1;
  const out: string[] = [];
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const k = `${z}/${x}/${y}`;
      if (!drop.includes(k)) out.push(k);
    }
  }
  return out;
}

const IMG = '8/205/104'; // ~[108.28,109.69]×[30.74,32.0]（湖北内）

describe('computeImagerySkipKeys（tile-exact 影像压制 v2）', () => {
  it('地形并集完全覆盖 → 影像瓦片被跳过', () => {
    const terrain = new Set(coveringTerrainKeys(IMG, 8));
    expect(terrain.size).toBeGreaterThan(4); // 3×3 邻域
    const skip = computeImagerySkipKeys(terrain, [IMG, '8/213/105']);
    expect(skip.has(IMG)).toBe(true);   // 完全覆盖 → 压制
    expect(skip.has('8/213/105')).toBe(false); // 上海（覆盖区外）→ 保留
  });

  it('L 形空隙（缺角块）→ 该影像瓦片保留（防破洞回归闸门）', () => {
    const full = coveringTerrainKeys(IMG, 8);
    const corner = full[full.length - 1]; // 东北角块
    const lShaped = new Set(coveringTerrainKeys(IMG, 8, [corner]));
    expect(lShaped.size).toBe(full.length - 1);
    const skip = computeImagerySkipKeys(lShaped, [IMG]);
    expect(skip.has(IMG)).toBe(false); // 空角未覆盖 → 不压制（v1 破洞教训）
  });

  it('部分重叠（仅单块地形）→ 不压制', () => {
    const one = new Set([coveringTerrainKeys(IMG, 8)[0]]);
    const skip = computeImagerySkipKeys(one, [IMG]);
    expect(skip.has(IMG)).toBe(false);
  });

  it('无地形 / 非法地形 key → 全部保留', () => {
    expect(computeImagerySkipKeys(new Set(), [IMG]).size).toBe(0);
    expect(computeImagerySkipKeys(new Set(['junk']), [IMG]).size).toBe(0);
  });

  it('多 z 层混合（粗层补空隙 → 跨层并集覆盖成立）→ 压制', () => {
    // 细层挖角 + 粗层全铺：并集仍完全覆盖
    const fine = coveringTerrainKeys(IMG, 10);
    const corner = fine[fine.length - 1];
    const mixed = new Set([...coveringTerrainKeys(IMG, 10, [corner]), ...coveringTerrainKeys(IMG, 8)]);
    const skip = computeImagerySkipKeys(mixed, [IMG]);
    expect(skip.has(IMG)).toBe(true);
  });

  it('非法影像 key → 忽略（不炸）', () => {
    const terrain = new Set(coveringTerrainKeys(IMG, 8));
    const skip = computeImagerySkipKeys(terrain, ['terrain:12/6614/2775', '']);
    expect(skip.size).toBe(0);
  });
});
