import { describe, expect, it } from 'vitest';
import { ImageryLayer } from '../../src/imagery/ImageryLayer';
import { TileKey } from '../../src/imagery/TileKey';

/**
 * 祖先链逐出保护（第四案 2026-10-23 回归闸门）：
 * 近视角下影像 selected 深化到 z13+ 后，地形 draping 绑定的粗层
 * （地形瓦片自身层级）不再是 selected——纹理压力下这些源被持续逐出
 * （真机实测 ~0.5 evictions/帧），draping 在每次重取窗口内闪 1×1
 * 占位。保护集必须覆盖 selected ∪ 每块到树根（z2）的完整祖先链。
 */
describe('ImageryLayer.renderProtectionKeys（selected ∪ 祖先链）', () => {
  it('z13 叶块 → 集含 z13..z2 全链（12 个 key）', () => {
    const keys = ImageryLayer.renderProtectionKeys([new TileKey(13, 3000, 1500)]);
    expect(keys.size).toBe(12); // levels 13 down to 2
    for (let z = 2; z <= 13; z++) expect(keys.has(`${z}/${3000 >> (13 - z)}/${1500 >> (13 - z)}`)).toBe(true);
  });

  it('兄弟两块 → 共享祖先去重（2 叶 + 11 共同祖先 = 13 key）', () => {
    const keys = ImageryLayer.renderProtectionKeys([
      new TileKey(13, 3000, 1500),
      new TileKey(13, 3001, 1500),
    ]);
    expect(keys.size).toBe(13);
  });

  it('z2 根块 → 仅自身（parent 链到根即止）', () => {
    const keys = ImageryLayer.renderProtectionKeys([new TileKey(2, 1, 1)]);
    expect(keys.size).toBe(1);
    expect(keys.has('2/1/1')).toBe(true);
  });

  it('空集 → 空集（保护帧替换语义不变）', () => {
    expect(ImageryLayer.renderProtectionKeys([]).size).toBe(0);
  });
});
