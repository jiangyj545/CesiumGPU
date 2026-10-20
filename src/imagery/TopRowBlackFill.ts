/**
 * TopRowBlackFill - 影像顶行（y === 0）"深蓝黑填充数据"检测与治理。
 *
 * 数据事实（curl 实测天地图/高德 Web Mercator 卫星影像，y=0 为最北行）：
 * - 顶行瓦片顶部带连续"深蓝黑"行（行平均亮度 10~15，RGB ≈ (7,6,40)/(9,13,40)），
 *   是 85.05°N 以上无卫星影像被填充成深蓝黑的结果：
 *   · z2 顶行 43~104 行（张而异）；z3 部分瓦片整张黑；
 *   · z4 顶行 11/16 张整张黑；z5 顶行 32/32 全部整张黑（83°N 以上无卫星影像）。
 * - 有效影像行平均亮度 ≥ 60，与黑行（≤ 15）分离开，阈值 20 居中分割。
 * - 南极底行（y === 2^z - 1）数据完全干净，不需要处理。
 *
 * 治理语义（与天地图官网"极区不可见"观感一致）：
 * - 整张黑（allBlack）：不建纹理、不进缓存 → resolveTexture 的祖先回退链
 *   自动用上级瓦片影像显示（83°~85.05°N 区间呈现上级内容的放大）。
 * - 顶部连续黑行 ≥ MIN_CROP_LEAD_ROWS：裁剪重绘——把有效区域拉伸到整幅
 *   （drawImage 源矩形 [0, lead, W, H-lead] → 目标 [0, 0, W, H]），
 *   消除渲染上的黑带。
 *
 * 性能：检测只在 y === 0 瓦片上传时做一次（每级 2^z 张），不是每帧。
 * 像素采样隔列（每行每隔 SAMPLE_STRIDE_X 取 1 个样本）；行本身逐行统计，
 * 因为 leadBlackRows 需要按行粒度精确计数。256×256 瓦片 ≈ 16K 样本，
 * 单瓦片一次性开销 < 0.5 ms（相对其网络解码可忽略）。
 */

/** 行平均亮度（0.299R + 0.587G + 0.114B）低于该值视为"无数据"黑行。实测黑行 10~15、有效影像 ≥ 60，取 20 居中。 */
export const BLACK_ROW_THRESHOLD = 20;

/**
 * 顶部连续黑行达到该行数才触发"裁剪重绘"。黑带太薄不值得重绘
 * （重绘引入一次整幅拉伸重采样）；实测 z2 顶行黑带 43 行起，8 行以下极少见。
 */
export const MIN_CROP_LEAD_ROWS = 8;

/**
 * "整张黑"判定的尾部容差：leadBlackRows ≥ rowMeans.length - 4 即判整张黑。
 * 实测整张黑瓦片并非绝对纯色，底部可能残留极少数暗色过渡行，留 4 行容差
 * 避免误判为"部分黑"而走裁剪路径。
 */
export const ALL_BLACK_TAIL_TOLERANCE = 4;

/** 列向采样步长：每行每隔 4 像素取 1 个样本（256 宽 → 64 样本/行）。 */
export const SAMPLE_STRIDE_X = 4;

export interface TopRowAnalysis {
  /** 从顶部起连续"黑行"（行平均亮度 < blackThreshold）的行数。 */
  leadBlackRows: number;
  /** 是否整张黑（leadBlackRows ≥ 总行数 - 4）：走祖先回退链，不建纹理。 */
  allBlack: boolean;
}

/**
 * 纯函数：给定逐行平均亮度序列，从顶部数连续黑行数，并判定是否整张黑。
 * 不依赖 canvas/DOM，可在 node（vitest）下直接单测。
 *
 * 空序列返回 lead = 0 / allBlack = false（"没有数据"不等于"全黑"，
 * 否则 length - 4 为负会让空输入误判为整张黑）。
 */
export function analyseTopRows(rowMeans: readonly number[], blackThreshold: number = BLACK_ROW_THRESHOLD): TopRowAnalysis {
  let lead = 0;
  while (lead < rowMeans.length && rowMeans[lead] < blackThreshold) lead++;

  const allBlack =
    rowMeans.length > 0 &&
    lead >= rowMeans.length - ALL_BLACK_TAIL_TOLERANCE;

  return { leadBlackRows: lead, allBlack };
}

/**
 * 纯函数：从 RGBA 像素缓冲（getImageData 输出，行主序、每像素 4 字节）
 * 按列向步长 `sampleStep` 采样，计算每行平均亮度（0.299R + 0.587G + 0.114B）。
 * 忽略 alpha（不透明影像瓦片）；不依赖 canvas/DOM，可在 node 下单测。
 */
export function computeRowMeans(
  pixels: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
  sampleStep: number = SAMPLE_STRIDE_X,
): number[] {
  const rowMeans = new Array<number>(height);
  for (let y = 0; y < height; y++) {
    const rowStart = y * width * 4;
    let sum = 0;
    let count = 0;
    for (let x = 0; x < width; x += sampleStep) {
      const o = rowStart + x * 4;
      sum += 0.299 * pixels[o] + 0.587 * pixels[o + 1] + 0.114 * pixels[o + 2];
      count++;
    }
    rowMeans[y] = count > 0 ? sum / count : 0;
  }
  return rowMeans;
}

/** 可作为 copyExternalImageToTexture 源的 2D 画布表面。 */
export type CanvasSurface = HTMLCanvasElement | OffscreenCanvas;

/**
 * 创建一张 2D 画布表面：浏览器环境优先 OffscreenCanvas（worker-safe、
 * 无 DOM 耦合），退化 document.createElement('canvas')；node（vitest）
 * 两者皆无时返回 null。与 VideoSource 的画布创建约定一致。
 */
export function createCanvasSurface(width: number, height: number): CanvasSurface | null {
  if (typeof OffscreenCanvas !== 'undefined') {
    return new OffscreenCanvas(width, height);
  }
  if (typeof document !== 'undefined') {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }
  return null;
}

/**
 * 读取一张位图的逐行平均亮度（canvas 路径：drawImage → getImageData →
 * computeRowMeans）。环境不支持 canvas 时返回 null（调用方按现状路径放行，
 * 不阻塞上传）。注意：这会把位图整幅读回 CPU，仅对 y === 0 顶行瓦片调用
 * （每级 2^z 张、每张一次），见文件头性能注释。
 */
export function readBitmapRowMeans(bitmap: ImageBitmap | CanvasSurface): number[] | null {
  const surface = createCanvasSurface(bitmap.width, bitmap.height);
  if (surface === null) return null;
  // OffscreenCanvasRenderingContext2D 与 CanvasRenderingContext2D 的
  // drawImage/getImageData 是同一 API 子集，视图收窄到 2D context。
  const ctx = surface.getContext('2d') as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null;
  if (ctx === null) return null;
  ctx.drawImage(bitmap, 0, 0);
  const data = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
  return computeRowMeans(data, bitmap.width, bitmap.height);
}

/**
 * 裁剪重绘：把位图顶部 `leadRows` 行之后的区域拉伸到整幅画布
 * （源矩形 [0, lead, W, H - lead] → 目标矩形 [0, 0, W, H]）。
 * 语义：83°~85.05°N 显示 83° 处影像内容的拉伸（消除黑带）。
 * 返回 null 表示环境不支持 canvas（调用方回退现状路径）。
 */
export function cropTopRows(bitmap: ImageBitmap | CanvasSurface, leadRows: number): CanvasSurface | null {
  const width  = bitmap.width;
  const height = bitmap.height;
  const lead   = Math.min(leadRows, height - 1); // 至少保留 1 行有效源，避免 0 高源矩形
  const surface = createCanvasSurface(width, height);
  if (surface === null) return null;
  const ctx = surface.getContext('2d') as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null;
  if (ctx === null) return null;
  ctx.drawImage(bitmap, 0, lead, width, height - lead, 0, 0, width, height);
  return surface;
}
