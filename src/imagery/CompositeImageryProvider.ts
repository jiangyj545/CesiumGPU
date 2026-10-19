import { TileKey } from './TileKey';
import type { ImageryProvider } from './ImageryProvider';
import type { ImageryLayer } from './ImageryLayer';

/**
 * CompositeImageryProvider – 多图层逐瓦片合成（P3：真实消费方踩到的
 * 「底图+注记只画顶层」缺陷的引擎侧修复）。
 *
 * 集合内可见层 ≥2 时，Globe 不再只渲染最顶层，而是经本 provider 按
 * TileKey 逐瓦片拉取全部子层并自底向上 source-over 合成：
 * - 子层失败（reject 或 null）→ 该子层该瓦片跳过（透明洞），其余子层
 *   照常叠加；全部失败才返回 null（计瓦片失败，走父级回退）。
 * - maximumLevel = min(子层)；tileSize = 首子层（服务约定一致）；
 *   ready = 全部就绪。
 *   抽成纯函数 {@link blendSourceOver}/{@link blendBufferSourceOver}
 *   （node 可测）；requestTile 对整块瓦片逐像素循环混合；子层各自保留缓存
 *   （composite 直连 provider.requestTile 不经子层 LRU——重复占一份网络
 *   可接受，注释存档）。
 */

/**
 * 纯像素 source-over：把 `src`（直通 alpha RGBA）叠到 `dst` 上，就地更新
 * `dst`。`layerAlpha` 为子层自身透明度缩放（ImageryLayer.alpha）。
 *
 * 公式（straight alpha）：
 *   outA = srcA' + dstA·(1−srcA')；outC = (srcC·srcA' + dstC·dstA·(1−srcA'))/outA
 *   （outA = 0 时输出 0，画布合成结果与 drawImage 一致。）
 */
export function blendSourceOver(
  dst: Uint8ClampedArray,
  src: Uint8ClampedArray,
  layerAlpha = 1
): void {
  const srcA = (src[3] / 255) * layerAlpha;
  if (srcA <= 0) return; // 全透明覆盖：目标原样
  const inv = 1 - srcA;
  const dstA = dst[3] / 255;
  const outA = srcA + dstA * inv;
  if (outA <= 0) {
    dst[0] = dst[1] = dst[2] = dst[3] = 0;
    return;
  }
  dst[0] = (src[0] * srcA + dst[0] * dstA * inv) / outA;
  dst[1] = (src[1] * srcA + dst[1] * dstA * inv) / outA;
  dst[2] = (src[2] * srcA + dst[2] * dstA * inv) / outA;
  dst[3] = outA * 255;
}

/**
 * 整块 RGBA 缓冲逐像素 source-over（{@link blendSourceOver} 的循环版）：
 * requestTile 的实际合成路径——对 `pixelCount` 个像素逐个执行单像素
 * 混合（内联同一公式，避免每像素分配）。语义 = 对每个像素各调用一次
 * blendSourceOver。
 */
export function blendBufferSourceOver(
  dst: Uint8ClampedArray,
  src: Uint8ClampedArray,
  layerAlpha = 1,
  pixelCount: number
): void {
  for (let p = 0; p < pixelCount; p++) {
    const o = p * 4;
    const srcA = (src[o + 3] / 255) * layerAlpha;
    if (srcA <= 0) continue; // 全透明覆盖：目标原样
    const inv = 1 - srcA;
    const dstA = dst[o + 3] / 255;
    const outA = srcA + dstA * inv;
    if (outA <= 0) {
      dst[o] = dst[o + 1] = dst[o + 2] = dst[o + 3] = 0;
      continue;
    }
    dst[o] = (src[o] * srcA + dst[o] * dstA * inv) / outA;
    dst[o + 1] = (src[o + 1] * srcA + dst[o + 1] * dstA * inv) / outA;
    dst[o + 2] = (src[o + 2] * srcA + dst[o + 2] * dstA * inv) / outA;
    dst[o + 3] = outA * 255;
  }
}

/** 单个子层的取瓦片结局（fetchChildTiles 产物，canvas-free 便于 node 测试）。 */
export type ChildTileResult =
  | { ok: true; layer: ImageryLayer; bitmap: ImageBitmap }
  | { ok: false; layer: ImageryLayer };

/**
 * 并发拉取全部子层的同一瓦片：`Promise.allSettled` 语义——单子层 reject /
 * null 只记 skip（`ok:false`），不阻塞其余子层。顺序保持栈序（底→顶）。
 */
export async function fetchChildTiles(
  layers: readonly ImageryLayer[],
  key: TileKey
): Promise<ChildTileResult[]> {
  const settled = await Promise.allSettled(
    layers.map((layer) => layer.provider.requestTile(key))
  );
  const out: ChildTileResult[] = [];
  for (let i = 0; i < layers.length; i++) {
    const r = settled[i];
    if (r.status === 'fulfilled' && r.value !== null) {
      out.push({ ok: true, layer: layers[i], bitmap: r.value });
    } else {
      out.push({ ok: false, layer: layers[i] });
    }
  }
  return out;
}

/** 合成 provider：子层栈（底→顶）→ 逐瓦片合成的单一 ImageryProvider。 */
export class CompositeImageryProvider implements ImageryProvider {
  readonly url: string;
  readonly maximumLevel: number;
  readonly tileSize: number;
  readonly subdomains: readonly string[] = [];
  readonly ready: boolean;

  /** 参与合成的子层（底→顶，只在构造时快照；集合重建即换新实例）。 */
  readonly layers: readonly ImageryLayer[];

  constructor(layers: readonly ImageryLayer[]) {
    if (layers.length === 0) throw new Error('CompositeImageryProvider: empty layer stack');
    this.layers = layers;
    this.url = `composite://${layers.length}`;
    this.maximumLevel = Math.min(...layers.map((l) => l.provider.maximumLevel));
    this.tileSize = layers[0].provider.tileSize;
    this.ready = layers.every((l) => l.provider.ready);
  }

  /** 合成层无网络请求本体；URL 交给首子层（诊断用途）。 */
  resolveUrl(key: TileKey): string {
    return this.layers[0].provider.resolveUrl(key);
  }

  /**
   * 拉取并合成一块瓦片：任一子层失败跳过；全部失败 → null（调用方按
   * 父级回退处理）。合成主线程 OffscreenCanvas + 纯函数像素数学。
   */
  async requestTile(key: TileKey): Promise<ImageBitmap | null> {
    const children = await fetchChildTiles(this.layers, key);
    const hits = children.filter((c): c is Extract<ChildTileResult, { ok: true }> => c.ok);
    if (hits.length === 0) return null;

    if (typeof OffscreenCanvas === 'undefined' || typeof createImageBitmap !== 'function') {
      // 无 canvas 运行时（node）：降级为最底层子层瓦片（浏览器主路径不受
      // 影响；node 侧合成数学由 blendSourceOver 纯函数测试覆盖）。
      return hits[0].bitmap;
    }

    const size = this.tileSize;
    const canvas = new OffscreenCanvas(size, size);
    const ctx = canvas.getContext('2d');
    if (!ctx) return hits[0].bitmap;

    // 自底向上：逐层取像素 → blendBufferSourceOver 逐像素混合（含层
    // alpha）→ 回写。getImageData/putImageData 直通 RGBA，与 drawImage
    // 的 source-over 数值一致（见 blendSourceOver 公式注）。
    let acc: Uint8ClampedArray | null = null;
    for (const hit of hits) {
      const layerCanvas = new OffscreenCanvas(size, size);
      const layerCtx = layerCanvas.getContext('2d');
      if (!layerCtx) continue;
      layerCtx.drawImage(hit.bitmap, 0, 0, size, size);
      const rgba = layerCtx.getImageData(0, 0, size, size).data;
      if (acc === null) {
        acc = new Uint8ClampedArray(rgba); // 底层：整层落地
      } else {
        blendBufferSourceOver(acc, rgba, hit.layer.alpha, size * size);
      }
    }
    if (!acc) return null;
    ctx.putImageData(new ImageData(new Uint8ClampedArray(acc), size, size), 0, 0);
    return createImageBitmap(canvas);
  }
}
