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

/** 展开后的批次表：属性名 → 每批次值数组。 */
export type ExpandedBatchTable = Record<string, unknown[]>;

/**
 * 解析并展开 batch table（P1-E feature 挑拣）：
 *
 * - JSON 数组属性 → 原样取列；
 * - 二进制属性 `{byteOffset, type, componentType}` → 从 batchTableBinary
 *   按分量类型读出 batchCount 个值（SCALAR→number、VECN→number[]、
 *   STRING→UTF-8 字符串，对齐 b3dm spec 的二进制批次表编码）；
 * - `batchCount = 0` / JSON 解析失败 / 无表 → null（tile 级降级）。
 *
 * @param buffer   完整 b3dm ArrayBuffer
 * @param batchCount BATCH_LENGTH（0 或缺失时返回 null）
 */
export function parseBatchTable(
  buffer: ArrayBuffer,
  batchCount: number
): ExpandedBatchTable | null {
  if (!Number.isFinite(batchCount) || batchCount <= 0) return null;
  const view = new DataView(buffer);
  const jsonLength = view.getUint32(16, true);
  const binLength  = view.getUint32(20, true);
  if (jsonLength === 0) return null;

  const jsonStart = HEADER_LENGTH + view.getUint32(8, true) + view.getUint32(12, true);
  let json: Record<string, unknown>;
  try {
    json = JSON.parse(
      new TextDecoder().decode(new Uint8Array(buffer, jsonStart, jsonLength)).trim()
    ) as Record<string, unknown>;
  } catch (error) {
    console.error('B3dmDecoder: batch table JSON parse failed', error);
    return null;
  }

  const bin = binLength > 0 ? new DataView(buffer, jsonStart + jsonLength, binLength) : null;
  const table: ExpandedBatchTable = {};
  for (const [name, value] of Object.entries(json)) {
    if (Array.isArray(value)) {
      table[name] = value;
      continue;
    }
    if (value && typeof value === 'object' && bin) {
      const column = readBinaryBatchProperty(value as Record<string, unknown>, bin, batchCount);
      if (column) table[name] = column;
    }
  }
  return table;
}

const COMPONENT_BYTES: Record<number, number> = {
  5120: 1, // BYTE
  5121: 1, // UNSIGNED_BYTE
  5122: 2, // SHORT
  5123: 2, // UNSIGNED_SHORT
  5125: 4, // UNSIGNED_INT
  5126: 4, // FLOAT
};

const TYPE_COUNT: Record<string, number> = {
  SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4,
};

/** 读一个二进制批次属性列为 batchCount 个 JS 值；非法定义返回 null。 */
function readBinaryBatchProperty(
  def: Record<string, unknown>,
  bin: DataView,
  batchCount: number
): unknown[] | null {
  const byteOffset = Number(def.byteOffset);
  const type = typeof def.type === 'string' ? def.type : 'SCALAR';
  const componentType = def.componentType === undefined ? 5126 : Number(def.componentType);
  if (!Number.isFinite(byteOffset) || !(componentType in COMPONENT_BYTES)) return null;
  if (type === 'STRING') {
    // 字符串列：每批一段 NUL 结尾 UTF-8（spec 允许定长/变长，取 NUL 终止读法）
    const out: string[] = [];
    for (let b = 0; b < batchCount; b++) {
      let end = byteOffset;
      while (end < bin.byteLength && bin.getUint8(end) !== 0) end++;
      out.push(new TextDecoder().decode(new Uint8Array(bin.buffer, bin.byteOffset + byteOffset, end - byteOffset)));
    }
    return out;
  }
  const components = TYPE_COUNT[type];
  if (!components) return null;
  const bytes = COMPONENT_BYTES[componentType];
  const out: unknown[] = [];
  for (let b = 0; b < batchCount; b++) {
    const values: number[] = [];
    for (let c = 0; c < components; c++) {
      const o = byteOffset + (b * components + c) * bytes;
      if (o + bytes > bin.byteLength) return null; // 越界：整列丢弃
      values.push(readComponent(bin, o, componentType));
    }
    out.push(components === 1 ? values[0] : values);
  }
  return out;
}

function readComponent(bin: DataView, offset: number, componentType: number): number {
  switch (componentType) {
    case 5120: return bin.getInt8(offset);
    case 5121: return bin.getUint8(offset);
    case 5122: return bin.getInt16(offset, true);
    case 5123: return bin.getUint16(offset, true);
    case 5125: return bin.getUint32(offset, true);
    default:   return bin.getFloat32(offset, true);
  }
}
