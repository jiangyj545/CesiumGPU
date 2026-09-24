/**
 * b3dmDecoder.spec.ts – b3dm 28B 头解析 / GLB 剥壳 / RTC_CENTER 单测。
 *
 * 全部使用手工合成的 b3dm 字节（无生产数据）：
 *   [28B 头][featureTableJson][featureTableBin][batchTableJson][batchTableBin][GLB]
 * GLB 复用 tests/loader/helpers.ts 的 buildMinimalGlb（1 顶点最小模型）。
 */
import { describe, expect, it } from 'vitest';
import {
  computeGlbOffset,
  extractGlb,
  parseFeatureTableJson,
  parseRtcCenter,
} from '../../src/tileset/B3dmDecoder';
import { GltfLoader } from '../../src/loader/GltfLoader';
import { buildMinimalGlb } from '../loader/helpers';

const B3DM_MAGIC = 0x6d643362; // 'b3dm' 小端

interface B3dmPart {
  featureTableJson?: object;
  /** 特征表二进制区原始字节（RTC 等 float 载荷）。 */
  featureTableBin?: Uint8Array;
  batchTableJson?: object;
  batchTableBin?: Uint8Array;
}

function pad4(n: number): number {
  return (4 - (n % 4)) % 4;
}

function encodeTable(payload: object | undefined): { bytes: Uint8Array; paddedLen: number } {
  if (!payload) return { bytes: new Uint8Array(0), paddedLen: 0 };
  const raw = new TextEncoder().encode(JSON.stringify(payload));
  const pad = pad4(raw.length);
  const bytes = new Uint8Array(raw.length + pad);
  bytes.set(raw, 0);
  bytes.fill(0x20, raw.length); // 表 JSON 用空格填充对齐 4（b3dm spec）
  return { bytes, paddedLen: bytes.length };
}

/** 构造一个合成 b3dm ArrayBuffer。 */
function buildB3dm(parts: B3dmPart, glb: ArrayBuffer = buildMinimalGlb()): ArrayBuffer {
  const ftj = encodeTable(parts.featureTableJson);
  const btj = encodeTable(parts.batchTableJson);
  const ftb = parts.featureTableBin ?? new Uint8Array(0);
  const btb = parts.batchTableBin ?? new Uint8Array(0);

  const glbBytes = new Uint8Array(glb);
  const total = 28 + ftj.paddedLen + ftb.length + btj.paddedLen + btb.length + glbBytes.length;
  const out = new ArrayBuffer(total);
  const view = new DataView(out);
  const bytes = new Uint8Array(out);

  view.setUint32(0, B3DM_MAGIC, true);
  view.setUint32(4, 1, true);                       // version
  view.setUint32(8, ftj.paddedLen, true);           // featureTableJsonLength
  view.setUint32(12, ftb.length, true);             // featureTableBinLength
  view.setUint32(16, btj.paddedLen, true);          // batchTableJsonLength
  view.setUint32(20, btb.length, true);             // batchTableBinLength
  view.setUint32(24, 0, true);                      // gltfFormat = 0（内嵌 GLB）

  let off = 28;
  bytes.set(ftj.bytes, off); off += ftj.paddedLen;
  bytes.set(ftb, off);       off += ftb.length;
  bytes.set(btj.bytes, off); off += btj.paddedLen;
  bytes.set(btb, off);       off += btb.length;
  bytes.set(glbBytes, off);
  return out;
}

describe('B3dmDecoder.computeGlbOffset / extractGlb', () => {
  it('28B 头 + 无表：GLB 偏移 = 28', () => {
    const buffer = buildB3dm({});
    expect(computeGlbOffset(buffer)).toBe(28);
  });

  it('四段表长度累加进 GLB 偏移（28 + 各段对齐长度）', () => {
    const buffer = buildB3dm({
      featureTableJson: { RTC_CENTER: [1] },
      featureTableBin: new Uint8Array(12),
      batchTableJson: { name: 'x' },
      batchTableBin: new Uint8Array(8),
    });
    // '{"RTC_CENTER":[1]}' 19B → pad 到 20；'{"name":"x"}' 12B → 已对齐
    expect(computeGlbOffset(buffer)).toBe(28 + 20 + 12 + 12 + 8);
  });

  it('extractGlb 返回的切片以 glTF magic 开头且可被 parseGlbBuffer 解析', async () => {
    const buffer = buildB3dm({});
    const glb = extractGlb(buffer);
    const view = new DataView(glb);
    expect(view.getUint32(0, true)).toBe(0x46546C67);
    // 端到端：剥出的 GLB 交 GltfLoader（生产路径）
    const parsed = await GltfLoader.parseGlbBuffer(glb);
    expect(parsed.primitives.length).toBe(1);
  });

  it('magic 非法 → 抛错（不静默）', () => {
    const buffer = buildB3dm({});
    new DataView(buffer).setUint32(0, 0x12345678, true);
    expect(() => computeGlbOffset(buffer)).toThrow(/magic/);
  });

  it('buffer 过小 / 声称长度超界 → 抛错', () => {
    const buffer = buildB3dm({});
    const tiny = buffer.slice(0, 20);
    expect(() => computeGlbOffset(tiny)).toThrow(/too small/);
    // 头声称 GLB 在 28 起，但 buffer 只有 30B —— 偏移仍合法（28 ≤ 30）
    const clipped = buffer.slice(0, 30);
    expect(() => computeGlbOffset(clipped)).not.toThrow();
  });
});

describe('B3dmDecoder.parseFeatureTableJson / parseRtcCenter', () => {
  it('空特征表 → 空对象；RTC_CENTER 不存在 → null', () => {
    const buffer = buildB3dm({ featureTableJson: {} });
    expect(parseFeatureTableJson(buffer)).toEqual({});
    expect(parseRtcCenter(buffer)).toBeNull();
  });

  it('RTC_CENTER 为唯一二进制属性：偏移 0 处读 float32[3]', () => {
    const rtc = new Float32Array([10, 20, 30]);
    const buffer = buildB3dm({
      featureTableJson: { RTC_CENTER: [1] },
      featureTableBin: new Uint8Array(rtc.buffer),
    });
    expect(parseRtcCenter(buffer)).toEqual({ x: 10, y: 20, z: 30 });
  });

  it('纯 JSON 标量属性不占二进制区（RTC_CENTER 仍在偏移 0）', () => {
    const rtc = new Float32Array([1, 2, 3]);
    const buffer = buildB3dm({
      featureTableJson: { BATCH_LENGTH: 5, RTC_CENTER: [1] },
      featureTableBin: new Uint8Array(rtc.buffer),
    });
    expect(parseRtcCenter(buffer)).toEqual({ x: 1, y: 2, z: 3 });
  });

  it('RTC_CENTER 之前的二进制属性按 VEC3 FLOAT32 保守推进偏移', () => {
    // 已知简化：非 RTC 二进制属性按 12B×分量数推进（见 B3dmDecoder 注释）。
    // FIRST 为 2 个 VEC3 → 占 24B；RTC_CENTER 从 24B 处读。
    const bin = new Float32Array([9, 9, 9, 8, 8, 8, 1, 2, 3]);
    const buffer = buildB3dm({
      featureTableJson: { FIRST: [2], RTC_CENTER: [1] },
      featureTableBin: new Uint8Array(bin.buffer),
    });
    expect(parseRtcCenter(buffer)).toEqual({ x: 1, y: 2, z: 3 });
  });

  it('特征表 JSON 非法 → console.error 并返回 null（不抛炸）', () => {
    const buffer = buildB3dm({ featureTableJson: {} });
    // 手工把特征表 JSON 段改坏（内容换成非法 JSON）
    new Uint8Array(buffer, 28, 4).set(new TextEncoder().encode('!!!!'));
    expect(parseRtcCenter(buffer)).toBeNull();
  });
});
