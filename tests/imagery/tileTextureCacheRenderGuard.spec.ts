import { describe, expect, it } from 'vitest';
import { TileTextureCache } from '../../src/imagery/TileTextureCache';
import type { GPUTexture } from '../../src/imagery/TileTextureCache';

/**
 * 渲染集逐出保护（2026-10-22 闪烁根因回归闸门）：
 * 宽视野 z14 selected ~259 块 > 默认纹理缓存容量 200 → 每次落地逐出
 * 4–6 块「选中」瓦片 → primitive 被删 / draping 退化 1×1 占位 → 重取
 * → 重建：evict→refetch→rebuild 永动循环 = 用户可见的持续闪烁。
 * protectRenderSet 让 selected 集预算豁免（同 pin 语义）。
 * key 用 z13/z14（高于 PIN_LEVEL=4，避开粗层固有豁免，单测保护语义）。
 */
const fakeTexture = (): GPUTexture =>
  ({ destroyed: false } as unknown as GPUTexture);

function makeCache(capacity: number): TileTextureCache {
  return new TileTextureCache({ capacity });
}

describe('TileTextureCache 渲染集逐出保护', () => {
  it('容量压力下逐出未保护项、selected 存活（预算豁免语义）', () => {
    const cache = makeCache(2);
    cache.put('13/1/1', fakeTexture());
    cache.put('13/2/2', fakeTexture());
    // selected = 13/1/1（保护 1 块 → effective = 2+1 = 3）
    cache.protectRenderSet(['13/1/1']);
    // 第三块落地：size 3 ≤ effective 3 → 尚无压力，全员驻留
    cache.put('13/3/3', fakeTexture());
    expect(cache.has('13/3/3')).toBe(true);
    // 第四块落地：size 4 > effective 3 → 逐出最旧未保护 13/2/2，selected 13/1/1 存活
    cache.put('14/9/9', fakeTexture());
    expect(cache.has('13/1/1')).toBe(true);
    expect(cache.has('13/2/2')).toBe(false);
    expect(cache.has('13/3/3')).toBe(true);
    expect(cache.has('14/9/9')).toBe(true);
  });

  it('全 selected 保护时允许缓存临时超限（选中集完整性优先于名义容量）', () => {
    const cache = makeCache(2);
    cache.protectRenderSet(['13/a', '13/b', '13/c', '13/d']);
    cache.put('13/a', fakeTexture());
    cache.put('13/b', fakeTexture());
    cache.put('13/c', fakeTexture());
    cache.put('13/d', fakeTexture());
    expect(cache.has('13/a')).toBe(true);
    expect(cache.has('13/b')).toBe(true);
    expect(cache.has('13/c')).toBe(true);
    expect(cache.has('13/d')).toBe(true);
    // 名义容量 2 + 保护 4 → effective 6，全部驻留（不逐任何 selected）
  });

  it('保护集随帧整体替换：取消保护后恢复可逐性', () => {
    const cache = makeCache(2);
    cache.put('13/x', fakeTexture());
    cache.put('13/y', fakeTexture());
    // selected = {13/x}（effective 3）：put z 后 size 3 ≤ 3，y 仍驻留
    cache.protectRenderSet(['13/x']);
    cache.put('13/z', fakeTexture());
    expect(cache.has('13/x')).toBe(true);
    // 下一帧 selected 变化：x 不再选中 → 保护空 → put w 触发逐出最旧（x）
    cache.protectRenderSet(['13/z']);
    cache.put('13/w', fakeTexture());
    expect(cache.has('13/x')).toBe(false);
    expect(cache.has('13/z')).toBe(true);
    expect(cache.has('13/w')).toBe(true);
  });
});
