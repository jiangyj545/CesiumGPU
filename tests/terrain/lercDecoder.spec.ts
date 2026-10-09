/**
 * lercDecoder.spec.ts – P2-8 LERC1 编解码：
 * 测试专用最小编码器（encodeLerc1TestTile：常值 / 梯度 / 掩码洞三档）
 * 与 decodeLerc1 的闭环，断言像素值、行列数、无效值掩码、量化还原。
 * （解码器语义移植自 Esri lerc.js Apache-2.0，见 LercDecoder.ts 头注。）
 */
import { describe, expect, it } from 'vitest';
import { decodeLerc1, encodeLerc1TestTile } from '../../src/terrain/LercDecoder';

describe('decodeLerc1（合成瓦片闭环）', () => {
  it('常值瓦片：全像素 = 常值，无掩码（全有效）', () => {
    const tile = decodeLerc1(encodeLerc1TestTile({ width: 4, height: 4, constant: 12.5 }));
    expect(tile.width).toBe(4);
    expect(tile.height).toBe(4);
    expect(tile.mask).toBeNull();
    for (const v of tile.pixels) expect(v).toBeCloseTo(12.5, 6);
    expect(tile.minValue).toBeCloseTo(12.5, 6);
    expect(tile.maxValue).toBeCloseTo(12.5, 6);
  });

  it('常 0 瓦片走 encoding-2 短路径', () => {
    const tile = decodeLerc1(encodeLerc1TestTile({ width: 3, height: 2, constant: 0 }));
    expect(Array.from(tile.pixels)).toEqual(Array.from(new Float32Array(6)));
    expect(tile.mask).toBeNull();
  });

  it('简单梯度（行主序 + 非整字节对齐 bpp）：精确位流还原', () => {
    // 5×3 = 15 值，bpp=6：跨字边界 + 尾字非整字节——位序最险路径。
    const tile = decodeLerc1(
      encodeLerc1TestTile({ width: 5, height: 3, value: (col, row) => col * 10 + row })
    );
    const expected: number[] = [];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 5; c++) expected.push(c * 10 + r);
    expect(Array.from(tile.pixels)).toEqual(expected);
    expect(tile.minValue).toBe(0);
    expect(tile.maxValue).toBe(42);
  });

  it('掩码洞：洞像素值 0 且 mask 位为 0，min/max 只统计有效像素', () => {
    const tile = decodeLerc1(
      encodeLerc1TestTile({
        width: 4,
        height: 2,
        value: (col) => col * 5, // 0,5,10,15 / 0,5,10,15
        hole: (col) => col === 1, // 每行第 2 列为洞
      })
    );
    expect(tile.mask).not.toBeNull();
    // 位图（bitset，8 像素/字节，MSB 先）：洞在 col 1 → 索引 1 与 5。
    const on = (i: number) => (tile.mask![i >> 3] & (0x80 >> (i & 7))) !== 0;
    expect(on(0)).toBe(true);
    expect(on(1)).toBe(false); // 洞
    expect(on(5)).toBe(false); // 洞
    expect(on(7)).toBe(true);
    // 行主序 [0,H,10,15, 0,H,10,15]，H=洞（值置 0）。
    expect(tile.pixels[1]).toBe(0);
    expect(tile.pixels[5]).toBe(0);
    expect(tile.pixels[2]).toBeCloseTo(10, 6);
    expect(tile.pixels[6]).toBeCloseTo(10, 6);
    expect(tile.minValue).toBeCloseTo(0, 6);
    expect(tile.maxValue).toBeCloseTo(15, 6);
  });

  it('量化还原：非网格值在 maxZError 内还原', () => {
    const tile = decodeLerc1(
      encodeLerc1TestTile({
        width: 8,
        height: 8,
        value: (col, row) => 100 + (col * 7 + row * 3) / 3, // 非整数值
        maxZError: 0.01,
      })
    );
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const expected = 100 + (c * 7 + r * 3) / 3;
        expect(Math.abs(tile.pixels[r * 8 + c] - expected)).toBeLessThanOrEqual(0.01);
      }
    }
  });

  it('非法 magic / 截断缓冲抛错（不静默）', () => {
    const bad = new ArrayBuffer(64);
    new Uint8Array(bad).set(new TextEncoder().encode('NotZImage!'), 0);
    expect(() => decodeLerc1(bad)).toThrow(/CntZImage/);
    expect(() => decodeLerc1(new ArrayBuffer(8))).toThrow(/too small/);
  });

  it('解码器消费真实 lerc.js 风格字段：maxZError=0 时常值块原样', () => {
    const tile = decodeLerc1(
      encodeLerc1TestTile({ width: 2, height: 2, constant: -33.25, maxZError: 0 })
    );
    for (const v of tile.pixels) expect(v).toBeCloseTo(-33.25, 6);
  });
});
