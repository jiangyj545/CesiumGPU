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
  | { ok: true; layer: ImageryLayer; bitmap: ImageBitmap | HTMLCanvasElement | OffscreenCanvas }
  | { ok: false; layer: ImageryLayer };

/**
 * 子层瓦片请求的超时上限（ms）。慢源截断为子层失败（透明洞），避免拖死
 * 整块合成瓦片：`Promise.allSettled` 要等全部子层 settle，大陆网络不可达
 * 的 OSM 官方瓦片 TCP 超时 ~45s，会把每个合成瓦片拖到 45s 才出图（真机
 * 现象：叠加模式地球长时间空白）。正常源 p99 < 5s，10s 上限只截真正
 * 不可达的源。
 */
const CHILD_TILE_TIMEOUT_MS = 10_000;

/**
 * 源级冷却：子层超时一次即在冷却窗口内直接跳过（不再发请求、不再陪跑
 * race）。只靠 race 不够——合成层每块瓦片都要陪慢源跑满 10s，叠加
 * ImageryLayer 的并发上限（10 in-flight），吞吐被钳制在 ~10 块/10s，
 * 一个视口 29 块瓦片要 30-50s 才齐（真机实测：叠加模式黑屏冻结 45s+，
 * OSM 自然超时后集体解冻）。冷却后后续瓦片立即出图（剩余可达子层
 * p99 < 5s），不可达源留透明洞、窗口过后自动重试（对齐 ImageryLayer
 * FAILED_TILE_RETRY_MS 语义）。正常完成（含快速失败 404/reject）不计入
 * ——瓦片级失败不是源级故障。
 */
const SOURCE_COOLDOWN_MS = 30_000;

interface CooldownState {
  /** 冷却截止时刻（performance.now() 基准）；0 = 未冷却。 */
  until: number;
}

/** 源冷却表：provider 身份键，页面/单测试模块实例各自独立。 */
const sourceCooldowns = new WeakMap<ImageryProvider, CooldownState>();

/** 该 provider 是否处于冷却窗口（源级慢/不可达）内。 */
function isCoolingDown(provider: ImageryProvider, now: number): boolean {
  const state = sourceCooldowns.get(provider);
  return state !== undefined && now < state.until;
}

/** 记一次源级超时 → 进入冷却窗口。 */
function markTimedOut(provider: ImageryProvider, now: number): void {
  sourceCooldowns.set(provider, { until: now + SOURCE_COOLDOWN_MS });
}

/** 正常完成 → 解除冷却（源恢复即回队）。 */
function markHealthy(provider: ImageryProvider): void {
  if (sourceCooldowns.has(provider)) sourceCooldowns.delete(provider);
}

/**
 * 单子层请求加超时：超时按 resolve(null) 处理并标记源冷却，走
 * fetchChildTiles 既有的「子层失败跳过」语义。底层请求不 abort（挂到
 * 自然超时，无渲染影响），正常完成时立即透传，不引入额外延迟。
 */
function withChildTimeout(
  layer: ImageryLayer,
  key: TileKey,
  timeoutMs: number
): Promise<ImageBitmap | HTMLCanvasElement | OffscreenCanvas | null> {
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      markTimedOut(layer.provider, performance.now());
      resolve(null);
    }, timeoutMs);
    layer.provider.requestTile(key).then(
      (value) => {
        clearTimeout(timer);
        markHealthy(layer.provider);
        resolve(value);
      },
      () => {
        clearTimeout(timer);
        resolve(null);
      }
    );
  });
}

/**
 * 并发拉取全部子层的同一瓦片：`Promise.allSettled` 语义——单子层 reject /
 * null / 超时 / 冷却只记 skip（`ok:false`），不阻塞其余子层。顺序保持栈序
 * （底→顶）。冷却中的子层不发请求（慢源不陪跑、不占并发）。
 */
export async function fetchChildTiles(
  layers: readonly ImageryLayer[],
  key: TileKey,
  timeoutMs: number = CHILD_TILE_TIMEOUT_MS
): Promise<ChildTileResult[]> {
  const now = performance.now();
  const settled = await Promise.allSettled(
    layers.map((layer) => {
      if (isCoolingDown(layer.provider, now)) return Promise.resolve(null);
      return withChildTimeout(layer, key, timeoutMs);
    })
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
   *
   * 产物直接返回合成 canvas（CanvasSurface）而非 createImageBitmap：
   * Chrome 的 copyExternalImageToTexture 对「OffscreenCanvas 再创建的
   * ImageBitmap」静默拷出全零纹理（真机 GPU 回读实证：合成瓦片上传后
   * mip0 全 (0,0,0,0)，同一 bitmap 经 2D canvas 读回却是健康像素——
   * 叠加模式地球深黑无纹理的根因）；canvas 源上传路径与 fetch 解码
   * bitmap 同 API 且已被顶行裁剪（cropTopRows）路径验证。
   */
  async requestTile(key: TileKey): Promise<ImageBitmap | HTMLCanvasElement | OffscreenCanvas | null> {
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
    return canvas;
  }
}
