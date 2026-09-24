/**
 * tests/loader/helpers.ts – 手工构造最小 GLB / KTX2 二进制的辅助函数。
 *
 * GLB 结构：12B 头（magic 0x46546C67 + version + totalLength）
 *           + JSON chunk（type 0x4E4F534A，空格填充对齐 4）
 *           + BIN chunk（type 0x004E4942，零填充对齐 4）。
 */

/** 构造一个最小 GLB ArrayBuffer（JSON 必填，BIN 可选）。 */
export function buildGlb(json: object, bin?: Uint8Array): ArrayBuffer {
  const jsonBytes = new TextEncoder().encode(JSON.stringify(json));
  const jsonPad = (4 - (jsonBytes.length % 4)) % 4;
  const binPad = bin ? (4 - (bin.length % 4)) % 4 : 0;

  const total = 12 + 8 + jsonBytes.length + jsonPad + (bin ? 8 + bin.length + binPad : 0);
  const out = new ArrayBuffer(total);
  const view = new DataView(out);
  const bytes = new Uint8Array(out);

  // GLB 头：magic + version(2) + total length
  view.setUint32(0, 0x46546C67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, total, true);

  let off = 12;

  // JSON chunk
  view.setUint32(off, jsonBytes.length + jsonPad, true);
  view.setUint32(off + 4, 0x4E4F534A, true);
  bytes.set(jsonBytes, off + 8);
  bytes.fill(0x20, off + 8 + jsonBytes.length, off + 8 + jsonBytes.length + jsonPad);
  off += 8 + jsonBytes.length + jsonPad;

  // BIN chunk
  if (bin) {
    view.setUint32(off, bin.length + binPad, true);
    view.setUint32(off + 4, 0x004E4942, true);
    bytes.set(bin, off + 8);
  }

  return out;
}

export interface MinimalGltfOptions {
  extensionsUsed?: string[];
  extensionsRequired?: string[];
}

/**
 * 构造最小可解析 glTF JSON：1 mesh（1 primitive，POSITION=accessor 0）+ 1 node + 1 scene。
 * accessor 数据由调用方通过 buildGlb 的 bin 参数提供。
 */
export function buildMinimalGltfJson(opts: MinimalGltfOptions = {}): object {
  return {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, mode: 4 }] }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 1, type: 'VEC3' }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 12 }],
    buffers: [{ byteLength: 12 }],
    ...(opts.extensionsUsed ? { extensionsUsed: opts.extensionsUsed } : {}),
    ...(opts.extensionsRequired ? { extensionsRequired: opts.extensionsRequired } : {}),
  };
}

/** 默认 POSITION 数据：单个顶点 (0, 0, 0)。 */
export function defaultPositionBin(): Uint8Array {
  return new Uint8Array(new Float32Array([0, 0, 0]).buffer);
}

/** 一个带 POSITION 的最小可解析 GLB。 */
export function buildMinimalGlb(opts: MinimalGltfOptions = {}): ArrayBuffer {
  return buildGlb(buildMinimalGltfJson(opts), defaultPositionBin());
}

export interface AccessorCase {
  /** glTF 组件类型：5120 BYTE / 5121 UBYTE / 5122 SHORT / 5123 USHORT / 5126 FLOAT */
  componentType: number;
  componentSize: number;
  /** DataView 写入函数（小端）。 */
  write(view: DataView, offset: number, value: number): void;
  values: number[];
  type: string;
  normalized?: boolean;
}

/** 构造只含一个 accessor 的 GLB，accessor 即 POSITION（VEC3），便于断言顶点值。 */
export function buildAccessorGlb(c: AccessorCase): ArrayBuffer {
  // values 是逐分量的原始值；VEC3 时每 3 个值构成 1 个元素
  const comps = c.type === 'VEC3' ? 3 : 1;
  const count = c.values.length / comps;
  const byteLength = c.values.length * c.componentSize;
  const bin = new Uint8Array(byteLength);
  const dv = new DataView(bin.buffer);
  for (let i = 0; i < c.values.length; i++) {
    c.write(dv, i * c.componentSize, c.values[i]);
  }

  const json = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, mode: 4 }] }],
    accessors: [{
      bufferView: 0,
      componentType: c.componentType,
      count,
      type: c.type,
      ...(c.normalized !== undefined ? { normalized: c.normalized } : {}),
    }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength }],
    buffers: [{ byteLength }],
  };

  return buildGlb(json, bin);
}

export interface Ktx2BuildOptions {
  width: number;
  height: number;
  levelCount?: number;
  supercompressionScheme?: number;
  vkFormat?: number;
  faces?: number;
  depth?: number;
  layers?: number;
  /** 每级 mip 的 byteLength（默认取 [1024, 256, 64, 16]，不足补 8）。 */
  levelByteLengths?: number[];
}

/**
 * 构造一个最小 KTX2 容器（合法 magic + 80B 头 + levelIndex + 零填充载荷），
 * 只测头解析，不覆盖转码行为。
 */
export function buildKtx2(opts: Ktx2BuildOptions): ArrayBuffer {
  const levelCount = opts.levelCount ?? 1;
  const defaultLengths = [1024, 256, 64, 16];
  const lengths = opts.levelByteLengths ?? Array.from({ length: levelCount }, (_, i) => defaultLengths[i] ?? 8);
  const payloadLength = lengths.reduce((a, b) => a + b, 0);

  const total = 80 + levelCount * 24 + payloadLength;
  const out = new ArrayBuffer(total);
  const view = new DataView(out);
  const bytes = new Uint8Array(out);

  // magic：«ABKTX 20»\r\n\x1A\n
  bytes.set([0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A], 0);

  view.setUint32(12, opts.vkFormat ?? 0, true);                 // vkFormat（0 = UNDEFINED）
  view.setUint32(16, 1, true);                                  // typeSize
  view.setUint32(20, opts.width, true);                         // pixelWidth
  view.setUint32(24, opts.height, true);                        // pixelHeight
  view.setUint32(28, opts.depth ?? 0, true);                    // pixelDepth
  view.setUint32(32, opts.layers ?? 0, true);                   // layerCount
  view.setUint32(36, opts.faces ?? 1, true);                    // faceCount
  view.setUint32(40, levelCount, true);                         // levelCount
  view.setUint32(44, opts.supercompressionScheme ?? 0, true);   // supercompressionScheme
  // DFD/KVD/SGD 偏移长度保持 0（头解析测试不涉及）

  // 载荷紧跟 levelIndex 数组之后：80B 头 + levelCount×24B 条目
  let offset = 80 + levelCount * 24;
  for (let i = 0; i < levelCount; i++) {
    const entry = 80 + i * 24;
    view.setBigUint64(entry, BigInt(offset), true);             // byteOffset
    view.setBigUint64(entry + 8, BigInt(lengths[i]), true);     // byteLength
    view.setBigUint64(entry + 16, 0n, true);                    // uncompressedByteLength
    offset += lengths[i];
  }

  return out;
}
