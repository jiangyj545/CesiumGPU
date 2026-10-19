/**
 * compositeImagery.spec.ts – P3 多图层合成：
 * ① blendSourceOver 纯像素数学（不透明底 / 半透明覆盖 / 全透明跳过 / 层 alpha）；
 * ② fetchChildTiles 单子层失败跳过不阻塞（reject / null / 全失败）；
 * ③ ImageryLayerCollection.renderLayer：单层零改动路径 / 双层合成层 /
 *    show 翻转重建 / remove 重建 / 栈序重建；
 * ④ CompositeImageryProvider 元数据（maximumLevel=min、tileSize、ready）。
 * node 无 OffscreenCanvas：requestTile 的 canvas 合成由 Main 浏览器复验，
 * 本文件覆盖全部 canvas-free 逻辑与像素数学（不 mock 合成本身）。
 */
import { describe, expect, it } from 'vitest';
import {
  blendSourceOver,
  blendBufferSourceOver,
  fetchChildTiles,
  CompositeImageryProvider,
} from '../../src/imagery/CompositeImageryProvider';
import { ImageryLayerCollection } from '../../src/imagery/ImageryLayerCollection';
import { ImageryLayer } from '../../src/imagery/ImageryLayer';
import type { ImageryProvider } from '../../src/imagery/ImageryProvider';
import { TileKey } from '../../src/imagery/TileKey';

/** 彩色桩 provider（Main 指定的「两个彩色 stub 源」；网络边界桩，合成数学不桩）。 */
function colorProvider(color: string, opts: Partial<ImageryProvider> = {}): ImageryProvider {
  return {
    url: `stub://${color}`,
    maximumLevel: opts.maximumLevel ?? 18,
    tileSize: 256,
    subdomains: [],
    ready: opts.ready ?? true,
    resolveUrl: (key: TileKey) => `stub://${color}/${key.key}`,
    requestTile: opts.requestTile ?? (async () => null),
  };
}

/** 直通 RGBA 像素（4 字节）。 */
const px = (r: number, g: number, b: number, a: number) => new Uint8ClampedArray([r, g, b, a]);

describe('blendSourceOver（纯像素 source-over）', () => {
  it('不透明底 + 不透明覆盖 → 覆盖色原样落地', () => {
    const dst = px(30, 60, 120, 255); // 蓝底
    blendSourceOver(dst, px(200, 40, 40, 255)); // 不透明红
    expect([...dst]).toEqual([200, 40, 40, 255]);
  });

  it('不透明底 + 半透明覆盖 → 经典 alpha 混合', () => {
    const dst = px(0, 0, 255, 255); // 蓝底
    blendSourceOver(dst, px(255, 0, 0, 128)); // 50% 红
    // outA=1；outC = 0.502·255 + 0.498·C_blue
    expect(dst[0]).toBeCloseTo(128, 0);
    expect(dst[1]).toBeCloseTo(0, 0);
    expect(dst[2]).toBeCloseTo(127, 0);
    expect(dst[3]).toBe(255);
  });

  it('透明底 + 半透明覆盖 → 输出保持覆盖色的半透明（注记层直觉）', () => {
    const dst = px(0, 0, 0, 0); // 透明洞（底图该瓦片缺失）
    blendSourceOver(dst, px(255, 255, 255, 102)); // 40% 白注记
    expect(dst[0]).toBe(255);
    expect(dst[1]).toBe(255);
    expect(dst[2]).toBe(255);
    expect(dst[3]).toBe(102); // alpha 不被底图放大
  });

  it('全透明覆盖（alpha=0）→ 目标原样（跳过优化不改语义）', () => {
    const dst = px(10, 20, 30, 200);
    blendSourceOver(dst, px(255, 0, 0, 0));
    expect([...dst]).toEqual([10, 20, 30, 200]);
  });

  it('层 alpha 缩放：alpha=0.5 的不透明红 ≈ 50% 红', () => {
    const dst = px(0, 0, 255, 255);
    blendSourceOver(dst, px(255, 0, 0, 255), 0.5);
    // Uint8ClampedArray 四舍五入：127.5 → 128。
    expect(dst[0]).toBe(128);
    expect(dst[2]).toBe(128);
  });

  it('三层叠序：底→顶顺序合成的最终值与逐层一致（底图→注记→高亮）', () => {
    // 底：不透明蓝；中：50% 红；顶：50% 白。
    const acc = px(0, 0, 255, 255);
    blendSourceOver(acc, px(255, 0, 0, 128));
    const midR = acc[0], midB = acc[2];
    blendSourceOver(acc, px(255, 255, 255, 128));
    // 顶白把中间色向白拉一半（四舍五入）。
    expect(acc[0]).toBe(Math.round((midR + 255) / 2));
    expect(acc[2]).toBe(Math.round((midB + 255) / 2));
    expect(acc[3]).toBe(255);
  });
});

describe('blendBufferSourceOver（整块缓冲逐像素混合——requestTile 实际路径）', () => {
  it('2×2：第二层仅 (1,1) 不透明红 → 只该像素被覆盖，其余保持底层', () => {
    // 回归闸门：旧实现 requestTile 只调一次单像素 blendSourceOver，
    // 注记层只影响 (0,0)，瓦片其余像素全部丢弃（线上天地图注记丢失）。
    const dst = new Uint8ClampedArray([
      0, 0, 255, 255, 0, 0, 255, 255,
      0, 0, 255, 255, 0, 0, 255, 255,
    ]);
    const src = new Uint8ClampedArray([
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 255, 0, 0, 255,
    ]);
    blendBufferSourceOver(dst, src, 1, 4);
    expect([...dst]).toEqual([
      0, 0, 255, 255, 0, 0, 255, 255,
      0, 0, 255, 255, 255, 0, 0, 255,
    ]);
  });

  it('2×2：层 alpha=0.5 的不透明覆盖 → 每像素 50% 混合（不只 (0,0)）', () => {
    const dst = new Uint8ClampedArray([
      0, 0, 255, 255, 0, 0, 255, 255,
      0, 0, 255, 255, 0, 0, 255, 255,
    ]);
    const src = new Uint8ClampedArray([
      255, 0, 0, 255, 255, 0, 0, 255,
      255, 0, 0, 255, 255, 0, 0, 255,
    ]);
    blendBufferSourceOver(dst, src, 0.5, 4);
    for (let p = 0; p < 4; p++) {
      const o = p * 4;
      expect(dst[o]).toBe(128);                 // r：127.5 经 Uint8ClampedArray 取整
      expect(dst[o + 1]).toBe(0);               // g
      expect(dst[o + 2]).toBe(128);             // b：同上取整
      expect(dst[o + 3]).toBe(255);             // 不透明
    }
  });
});

describe('fetchChildTiles（单子层失败跳过不阻塞）', () => {
  const key = new TileKey(3, 2, 1);
  const bitmap = { width: 4 } as ImageBitmap;

  const layerWith = (requestTile: ImageryProvider['requestTile']) => {
    const layer = new ImageryLayer({ provider: colorProvider('x', { requestTile }) });
    return layer;
  };
  it('全部成功 → 栈序保持（底→顶）', async () => {
    const bottom = layerWith(async () => bitmap);
    const top = layerWith(async () => ({ width: 5 }) as ImageBitmap);
    const out = await fetchChildTiles([bottom, top], key);
    expect(out).toHaveLength(2);
    expect(out[0].ok).toBe(true);
    expect(out[1].ok).toBe(true);
    if (out[0].ok && out[1].ok) {
      expect(out[0].layer).toBe(bottom);
      expect(out[1].layer).toBe(top);
    }
  });

  it('单子层 null（该瓦片无数据）→ ok:false，其余子层照常', async () => {
    const bottom = layerWith(async () => bitmap);
    const top = layerWith(async () => null);
    const out = await fetchChildTiles([bottom, top], key);
    expect(out[0].ok).toBe(true);
    expect(out[1].ok).toBe(false);
  });

  it('单子层 reject → ok:false（不阻塞、不抛出）', async () => {
    const bottom = layerWith(async () => {
      throw new Error('network down');
    });
    const top = layerWith(async () => bitmap);
    const out = await fetchChildTiles([bottom, top], key);
    expect(out[0].ok).toBe(false);
    expect(out[1].ok).toBe(true);
  });

  it('全部失败 → 全 ok:false（上层 requestTile 返回 null 计瓦片失败）', async () => {
    const a = layerWith(async () => null);
    const b = layerWith(async () => {
      throw new Error('x');
    });
    const out = await fetchChildTiles([a, b], key);
    expect(out.every((c) => !c.ok)).toBe(true);
  });
});

describe('ImageryLayerCollection.renderLayer（合成路径解析）', () => {
  it('单可见层：零改动路径——renderLayer 就是该层，members 为空', () => {
    const c = new ImageryLayerCollection();
    const base = c.addImageryProvider(colorProvider('base'));
    const hidden = c.addImageryProvider(colorProvider('labels'));
    hidden.show = false;
    expect(c.renderLayer).toBe(base); // 原样引用（无包装）
    expect(c.compositedMembers()).toEqual([]);
  });

  it('双可见层：renderLayer 为合成层（非任一成员），members 栈序', () => {
    const c = new ImageryLayerCollection();
    const base = c.addImageryProvider(colorProvider('base'));
    const labels = c.addImageryProvider(colorProvider('labels'));
    const render = c.renderLayer;
    expect(render).not.toBe(base);
    expect(render).not.toBe(labels);
    expect(render?.provider).toBeInstanceOf(CompositeImageryProvider);
    expect(c.compositedMembers()).toEqual([base, labels]); // 底→顶
  });

  it('show 翻转：隐藏顶层 → 回单层原样路径；再显示 → 新合成层实例', () => {
    const c = new ImageryLayerCollection();
    const base = c.addImageryProvider(colorProvider('base'));
    const labels = c.addImageryProvider(colorProvider('labels'));
    const composite1 = c.renderLayer;

    labels.show = false; // 只剩底图
    expect(c.renderLayer).toBe(base);
    expect(c.compositedMembers()).toEqual([]);

    labels.show = true; // 双层回归
    const composite2 = c.renderLayer;
    expect(composite2).not.toBe(composite1); // 重建（缓存失效）
    expect(composite2).not.toBe(base);
  });

  it('remove 成员 → 合成层失效重建；全部移除 → null', () => {
    const c = new ImageryLayerCollection();
    const base = c.addImageryProvider(colorProvider('base'));
    const labels = c.addImageryProvider(colorProvider('labels'));
    void c.renderLayer;
    c.remove(labels, false);
    expect(c.renderLayer).toBe(base); // 单层原样
    c.remove(base, false);
    expect(c.renderLayer).toBeNull();
  });

  it('raiseToTop 改变合成顺序 → 重建（签名捕获栈序）', () => {
    const c = new ImageryLayerCollection();
    const base = c.addImageryProvider(colorProvider('base'));
    const overlay = c.addImageryProvider(colorProvider('overlay'));
    const first = c.renderLayer;
    c.raiseToTop(base); // 栈序反转
    const second = c.renderLayer;
    expect(second).not.toBe(first); // 顺序变化触发重建
    expect(c.compositedMembers()).toEqual([overlay, base]); // 新栈序
  });
});

describe('CompositeImageryProvider 元数据', () => {
  it('maximumLevel = min(子层)；tileSize = 首子层；ready = 全部就绪', () => {
    const c = new ImageryLayerCollection();
    const base = c.addImageryProvider(colorProvider('base', { maximumLevel: 12 }));
    c.addImageryProvider(colorProvider('labels', { maximumLevel: 18 }));
    const provider = c.renderLayer!.provider as CompositeImageryProvider;
    expect(provider.maximumLevel).toBe(12);
    expect(provider.tileSize).toBe(256);
    expect(provider.ready).toBe(true);
    expect(provider.layers).toEqual([base, c.get(1)]);
    void base;
  });

  it('任一子层未就绪 → ready=false；空栈构造抛错', () => {
    const c = new ImageryLayerCollection();
    c.addImageryProvider(colorProvider('a', { ready: true }));
    c.addImageryProvider(colorProvider('b', { ready: false }));
    expect((c.renderLayer!.provider as CompositeImageryProvider).ready).toBe(false);
    expect(() => new CompositeImageryProvider([])).toThrow(/empty/);
  });
});
