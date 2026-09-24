import { describe, expect, it } from 'vitest';
import {
  TranscoderFormat,
  isKtx2,
  parseKtx2Header,
  selectTranscodeFormat,
} from '../../src/loader/Ktx2Decoder';
import { buildKtx2 } from './helpers';

/** KTX2 magic：«ABKTX 20»\r\n\x1A\n */
const MAGIC = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A];

// ── KTX2 magic 校验 ──────────────────────────────────────────────────────────

describe('isKtx2 / parseKtx2Header – 容器 magic 校验', () => {
  it('合法 magic 返回 true', () => {
    const data = buildKtx2({ width: 64, height: 64 });
    expect(isKtx2(data)).toBe(true);
  });

  it('篡改任一 magic 字节返回 false', () => {
    const data = buildKtx2({ width: 64, height: 64 });
    const bytes = new Uint8Array(data);
    for (let i = 0; i < MAGIC.length; i++) {
      const corrupted = bytes.slice();
      corrupted[i] = (corrupted[i] + 1) & 0xFF;
      expect(isKtx2(corrupted)).toBe(false);
    }
  });

  it('长度不足 12 字节返回 false', () => {
    expect(isKtx2(new Uint8Array(8))).toBe(false);
    expect(isKtx2(new ArrayBuffer(0))).toBe(false);
  });

  it('parseKtx2Header 对非法 magic 抛错', () => {
    const data = buildKtx2({ width: 64, height: 64 });
    const bytes = new Uint8Array(data);
    bytes[1] = 0x00; // 破坏 'K'
    expect(() => parseKtx2Header(bytes.buffer)).toThrow(/invalid KTX2 container/);
  });
});

// ── KTX2 头字段解析 ──────────────────────────────────────────────────────────

describe('parseKtx2Header – 头字段', () => {
  it('解析 width/height/levelCount/scheme 与各级 levelIndex', () => {
    const data = buildKtx2({
      width: 256,
      height: 128,
      levelCount: 3,
      supercompressionScheme: 2,
      vkFormat: 0,
    });
    const header = parseKtx2Header(data);
    expect(header.width).toBe(256);
    expect(header.height).toBe(128);
    expect(header.levelCount).toBe(3);
    expect(header.supercompressionScheme).toBe(2);
    expect(header.vkFormat).toBe(0);
    expect(header.levels).toHaveLength(3);
    // 级偏移从 80 + 3×24 = 152 开始，按构造逐级累加
    expect(header.levels[0]).toEqual({ offset: 152, byteLength: 1024 });
    expect(header.levels[1]).toEqual({ offset: 1176, byteLength: 256 });
    expect(header.levels[2]).toEqual({ offset: 1432, byteLength: 64 });
  });

  it('非法尺寸（0×0）抛错', () => {
    const data = buildKtx2({ width: 0, height: 0 });
    expect(() => parseKtx2Header(data)).toThrow(/invalid KTX2 dimensions/);
  });

  it('3D / 立方体 / 图层布局抛不支持', () => {
    expect(() => parseKtx2Header(buildKtx2({ width: 8, height: 8, depth: 4 })))
      .toThrow(/unsupported KTX2 layout/);
    expect(() => parseKtx2Header(buildKtx2({ width: 8, height: 8, faces: 6 })))
      .toThrow(/unsupported KTX2 layout/);
    expect(() => parseKtx2Header(buildKtx2({ width: 8, height: 8, layers: 2 })))
      .toThrow(/unsupported KTX2 layout/);
  });

  it('levelCount 为 0 抛错', () => {
    const data = buildKtx2({ width: 8, height: 8, levelCount: 0 });
    expect(() => parseKtx2Header(data)).toThrow(/invalid KTX2 levelCount/);
  });
});

// ── 转码目标格式探测（stub device.features 三分支） ─────────────────────────

describe('selectTranscodeFormat – 设备特性三分支', () => {
  it('分支1：texture-compression-bc + 有 alpha → BC7', () => {
    const fmt = selectTranscodeFormat(new Set(['texture-compression-bc']), true);
    expect(fmt.transcoderFormat).toBe(TranscoderFormat.cTFBC7_RGBA);
    expect(fmt.gpuFormat).toBe('bc7-rgba-unorm');
    expect(fmt.bytesPerBlock).toBe(16);
    expect(fmt.compressed).toBe(true);
  });

  it('分支1备选：texture-compression-bc + 无 alpha → BC1（更省）', () => {
    const fmt = selectTranscodeFormat(new Set(['texture-compression-bc']), false);
    expect(fmt.transcoderFormat).toBe(TranscoderFormat.cTFBC1_RGB);
    expect(fmt.gpuFormat).toBe('bc1-rgba-unorm');
    expect(fmt.bytesPerBlock).toBe(8);
    expect(fmt.compressed).toBe(true);
  });

  it('分支2：etc2-rgba8unorm（无 BC）→ ETC2_RGBA', () => {
    const fmt = selectTranscodeFormat(new Set(['etc2-rgba8unorm']), true);
    expect(fmt.transcoderFormat).toBe(TranscoderFormat.cTFETC2_RGBA);
    expect(fmt.gpuFormat).toBe('etc2-rgba8unorm');
    expect(fmt.bytesPerBlock).toBe(16);
    expect(fmt.compressed).toBe(true);
  });

  it('分支3：都不支持 → CPU RGBA8（全平台可用）', () => {
    const fmt = selectTranscodeFormat(new Set<string>(), true);
    expect(fmt.transcoderFormat).toBe(TranscoderFormat.cTFRGBA32);
    expect(fmt.gpuFormat).toBe('rgba8unorm');
    expect(fmt.bytesPerBlock).toBe(4);
    expect(fmt.blockWidth).toBe(1);
    expect(fmt.compressed).toBe(false);
  });

  it('BC 特性优先于 ETC2（桌面设备选 BC）', () => {
    const fmt = selectTranscodeFormat(new Set(['texture-compression-bc', 'etc2-rgba8unorm']), true);
    expect(fmt.gpuFormat).toBe('bc7-rgba-unorm');
  });

  it('接受任意 Iterable（如 GPUSupportedFeatures），不必是 Set', () => {
    const fmt = selectTranscodeFormat(['texture-compression-bc'], true);
    expect(fmt.transcoderFormat).toBe(TranscoderFormat.cTFBC7_RGBA);
  });
});
