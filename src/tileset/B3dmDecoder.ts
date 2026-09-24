/**
 * B3dmDecoder – 3DTiles b3dm 容器剥壳（Batched 3D Model）。
 *
 * b3dm 二进制布局（spec：3D Tiles 1.0 / b3dm）：
 * ```
 * | 偏移 | 长度 | 字段                        |
 * |------|------|-----------------------------|
 * | 0    | 4    | magic = 'b3dm' (0x6D643362) |
 * | 4    | 4    | version = 1 (u32)           |
 * | 8    | 4    | featureTableJsonLength      |
 * | 12   | 4    | featureTableBinLength       |
 * | 16   | 4    | batchTableJsonLength        |
 * | 20   | 4    | batchTableBinLength         |
 * | 24   | 4    | gltfFormat = 0 (u32)        |
 * | 28   | ..   | featureTableJson + featureTableBin
 *              + batchTableJson + batchTableBin + GLB        |
 * ```
 * 28 字节头后依次是四段表，GLB 紧随其后。本模块只负责
 * ① 剥出 GLB 切片（交给 `GltfLoader.parseGlbBuffer`）；
 * ② 解析特征表里的 RTC_CENTER（b3dm 局部平移，glTF 顶点需整体偏移）。
 */

/** RTC_CENTER 结果（b3dm 局部坐标系，单位米）。 */
export interface RtcCenter {
  x: number;
  y: number;
  z: number;
}

const HEADER_LENGTH = 28;
const B3DM_MAGIC = 0x6d643362; // 'b3dm' 小端
const FLOAT32_SIZE = 4;

/**
 * 解析 28B 头并返回 GLB 起始偏移（= 28 + 四段表长度之和）。
 * magic / version / gltfFormat 非法时抛错（调用方捕获后计入 failed，
 * 单瓦片失败不炸整树）。
 */
export function computeGlbOffset(buffer: ArrayBuffer): number {
  if (buffer.byteLength < HEADER_LENGTH) {
    throw new Error(`B3dmDecoder: buffer too small (${buffer.byteLength}B) for a b3dm header`);
  }
  const view = new DataView(buffer);
  const magic = view.getUint32(0, true);
  if (magic !== B3DM_MAGIC) {
    throw new Error(`B3dmDecoder: bad magic 0x${magic.toString(16)} (expected 'b3dm')`);
  }
  const version = view.getUint32(4, true);
  if (version !== 1) {
    throw new Error(`B3dmDecoder: unsupported b3dm version ${version}`);
  }
  const featureTableJsonLength = view.getUint32(8, true);
  const featureTableBinLength  = view.getUint32(12, true);
  const batchTableJsonLength   = view.getUint32(16, true);
  const batchTableBinLength    = view.getUint32(20, true);
  const gltfFormat = view.getUint32(24, true);
  if (gltfFormat !== 0) {
    throw new Error(`B3dmDecoder: unsupported gltfFormat ${gltfFormat} (only embedded GLB = 0)`);
  }

  const glbOffset =
    HEADER_LENGTH +
    featureTableJsonLength + featureTableBinLength +
    batchTableJsonLength + batchTableBinLength;
  if (glbOffset > buffer.byteLength) {
    throw new Error(
      `B3dmDecoder: GLB offset ${glbOffset} exceeds buffer size ${buffer.byteLength} (truncated b3dm?)`
    );
  }
  return glbOffset;
}

/**
 * 剥壳：返回 b3dm 内嵌 GLB 的 ArrayBuffer 拷贝切片。
 * 切片交给 `GltfLoader.parseGlbBuffer` 直接解析（无 URL 场景）。
 */
export function extractGlb(buffer: ArrayBuffer): ArrayBuffer {
  const glbOffset = computeGlbOffset(buffer);
  return buffer.slice(glbOffset);
}

/** 解析特征表 JSON（空格填充对齐 4 —— trim 后 JSON.parse）。 */
export function parseFeatureTableJson(buffer: ArrayBuffer): Record<string, unknown> {
  const view = new DataView(buffer);
  const length = view.getUint32(8, true);
  if (length === 0) return {};
  const bytes = new Uint8Array(buffer, HEADER_LENGTH, length);
  const text = new TextDecoder().decode(bytes).trim();
  return JSON.parse(text) as Record<string, unknown>;
}

/**
 * 解析 RTC_CENTER（b3dm 局部平移，单位米）；不存在时返回 null。
 *
 * 特征表二进制区布局按 **JSON 键插入顺序** 累积偏移（b3dm spec：
 * 二进制属性依其在 featureTableJson 中出现的先后排列）。二进制属性在
 * JSON 中写为 `[分量数]`（如 `RTC_CENTER: [1]` = 1 个 VEC3 FLOAT32），
 * 纯 JSON 标量（如 `BATCH_LENGTH: 3`）不占二进制区。
 *
 * 已知简化：非 RTC_CENTER 的二进制属性按 VEC3 FLOAT32（12B × 分量数）
 * 保守推进偏移——真实数据集里 RTC_CENTER 几乎总在首位；遇更多二进制
 * 属性的瓦片时偏移可能不准，届时该瓦片仅平移有误差（TODO：完整特征
 * 表 layout 查表）。
 */
export function parseRtcCenter(buffer: ArrayBuffer): RtcCenter | null {
  let json: Record<string, unknown>;
  try {
    json = parseFeatureTableJson(buffer);
  } catch (error) {
    console.error('B3dmDecoder: feature table JSON parse failed', error);
    return null;
  }

  const view = new DataView(buffer);
  const featureTableBinLength = view.getUint32(12, true);
  const binStart = HEADER_LENGTH + view.getUint32(8, true);

  // 按 JSON 插入顺序累积二进制偏移，直到命中 RTC_CENTER
  let offset = 0;
  for (const [key, value] of Object.entries(json)) {
    if (!Array.isArray(value)) continue; // 纯 JSON 标量不占二进制区
    const count = Number(value[0]) || 0;
    if (key === 'RTC_CENTER') {
      const byteLength = 3 * FLOAT32_SIZE;
      if (offset + byteLength > featureTableBinLength) {
        console.error('B3dmDecoder: RTC_CENTER exceeds feature table binary');
        return null;
      }
      const floats = new Float32Array(buffer, binStart + offset, 3);
      return { x: floats[0], y: floats[1], z: floats[2] };
    }
    offset += count * 3 * FLOAT32_SIZE; // 保守按 VEC3 FLOAT32 推进（见注释）
  }
  return null;
}
