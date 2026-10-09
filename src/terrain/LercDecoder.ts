/**
 * LercDecoder – LERC1（"CntZImage"）瓦片编解码（P2-8）。
 *
 * 格式与算法参照 Esri/Apache-2.0 的 lerc.js（npm `lerc` 2.0.0，
 * Copyright 2015-2018 Esri；CesiumJS 1.135 同源打包为 Workers 内嵌）移植
 * 裁剪：只保留 F32 地形路径与 constant-0 / constant-offset / bitstuffed
 * 块编码，头 / 掩码 RLE / 位填充位序逐字段对齐原实现。
 *
 * 布局（小端）：
 * | 段 | 字段 |
 * |----|------|
 * | 文件头 | 10B magic "CntZImage"+2 空格 · i32 version · i32 imageType · u32 height · u32 width · f64 maxZError |
 * | 掩码头 | u32 numBlocksY · u32 numBlocksX · u32 numBytes · f32 maxValue |
 * | 掩码体 | numBytes>0：int16 计数 RLE（正=字面字节，负=重复）展开 ceil(w·h/8) 位图（MSB=有效），哨兵 cnt=-32768 |
 * | 像素头 | u32 numBlocksY · u32 numBlocksX · u32 numBytes · f32 maxValue |
 * | 块×(格+边缘) | headerByte：encoding=b&63（0 raw / 1 bitstuff / 2 const-0 / 3 const-offset），offsetType=b>>6（0 f32 / 1 i16 / 2 i8） |
 *
 * bitstuffed 值 = offset + n·(2·maxZError)；n 为 MSB-first 位流整数（按
 * 小端 Uint32 字消费、字内自高位取位，尾字按「无效尾字节」左移对齐），
 * n ≥ (maxValue−offset)/scale 时钳到 maxValue（量化越界保护）。块内数据
 * 按**有效像素紧凑序**存放（掩码洞不占位）。
 */

/** 解码结果（行主序，row 0 = 图像顶行）。 */
export interface Lerc1Tile {
  width: number;
  height: number;
  /** F32 像素值；无效像素为 0 且 mask 对应位为 0。 */
  pixels: Float32Array;
  /** 逐像素有效位（全有效时 null）。 */
  mask: Uint8Array | null;
  /** 仅统计有效像素。 */
  minValue: number;
  maxValue: number;
}

/** 解码一块 LERC1 瓦片；magic / 结构非法抛错。 */
export function decodeLerc1(buffer: ArrayBuffer): Lerc1Tile {
  if (buffer.byteLength < 50) throw new Error('LercDecoder: buffer too small for a LERC1 header');
  const view = new DataView(buffer);

  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 10));
  if (magic.trim() !== 'CntZImage') {
    throw new Error(`LercDecoder: unexpected file identifier "${magic.slice(0, 10)}" (expect CntZImage)`);
  }
  let fp = 10;
  fp += 4; // fileVersion（信息字段）
  fp += 4; // imageType（LERC1 中为信息字段，解码固定按 F32）
  const height = view.getUint32(fp, true);
  const width = view.getUint32(fp + 4, true);
  const maxZError = view.getFloat64(fp + 8, true);
  fp += 16;
  if (width === 0 || height === 0) throw new Error('LercDecoder: zero-sized tile');
  if (!(maxZError >= 0)) throw new Error('LercDecoder: negative maxZError');

  // ── 掩码 ──────────────────────────────────────────────────────────────────
  const maskNumBlocksY = view.getUint32(fp, true);
  const maskNumBytes = view.getUint32(fp + 8, true);
  const maskMaxValue = view.getFloat32(fp + 12, true);
  fp += 16;

  const pixelCount = width * height;
  let mask: Uint8Array | null = null;
  if (maskNumBytes > 0) {
    mask = unpackMaskRle(new DataView(buffer, fp, maskNumBytes), pixelCount);
    fp += maskNumBytes;
  } else if ((maskNumBytes | maskNumBlocksY | maskMaxValue) === 0) {
    mask = new Uint8Array(Math.ceil(pixelCount / 8)); // 全部无效
  } // 否则：无掩码体 → 全部有效。

  // ── 像素块 ────────────────────────────────────────────────────────────────
  const numBlocksY = view.getUint32(fp, true);
  const numBlocksX = view.getUint32(fp + 4, true);
  const pixelsNumBytes = view.getUint32(fp + 8, true);
  const globalMax = view.getFloat32(fp + 12, true);
  fp += 16;
  if (numBlocksX === 0 || numBlocksY === 0) throw new Error('LercDecoder: zero pixel block grid');

  const pixels = new Float32Array(pixelCount);
  let minValue = Number.POSITIVE_INFINITY;
  const scale = 2 * maxZError;
  const blockWidth = Math.floor(width / numBlocksX);
  const blockHeight = Math.floor(height / numBlocksY);
  const blockData = new Float32Array(blockWidth * blockHeight || 1);

  for (let by = 0; by <= numBlocksY; by++) {
    const bh = by !== numBlocksY ? blockHeight : height % numBlocksY;
    if (bh === 0) continue;
    for (let bx = 0; bx <= numBlocksX; bx++) {
      const bw = bx !== numBlocksX ? blockWidth : width % numBlocksX;
      if (bw === 0) continue;

      const headerByte = view.getUint8(fp);
      const encoding = headerByte & 63;
      if (encoding > 3) throw new Error(`LercDecoder: invalid block encoding ${encoding}`);
      let size = 1;
      let offset = 0;
      let bitsPerPixel = 0;
      let numValidPixels = 0;

      if (encoding !== 2) {
        const offsetType = headerByte >> 6;
        if (offsetType === 2) {
          offset = view.getInt8(fp + 1);
          size += 1;
        } else if (offsetType === 1) {
          offset = view.getInt16(fp + 1, true);
          size += 2;
        } else if (offsetType === 0) {
          offset = view.getFloat32(fp + 1, true);
          size += 4;
        } else {
          throw new Error('LercDecoder: invalid block offset type');
        }
        if (encoding === 1) {
          const b = view.getUint8(fp + size);
          bitsPerPixel = b & 63;
          const countType = b >> 6;
          size += 1;
          if (countType === 2) {
            numValidPixels = view.getUint8(fp + size);
            size += 1;
          } else if (countType === 1) {
            numValidPixels = view.getUint16(fp + size, true);
            size += 2;
          } else if (countType === 0) {
            numValidPixels = view.getUint32(fp + size, true);
            size += 4;
          } else {
            throw new Error('LercDecoder: invalid valid pixel count type');
          }
        }
      }
      fp += size;

      let data: Float32Array | undefined;
      let constValue = 0;
      if (encoding === 0) {
        const numPixels = (pixelsNumBytes - 1) / 4;
        if (numPixels !== Math.floor(numPixels)) {
          throw new Error('LercDecoder: uncompressed block has invalid length');
        }
        data = new Float32Array(buffer, fp, numPixels);
        fp += numPixels * 4;
      } else if (encoding === 1) {
        const dataBytes = Math.ceil((numValidPixels * bitsPerPixel) / 8);
        const words = new Uint32Array(Math.ceil(dataBytes / 4));
        new Uint8Array(words.buffer).set(new Uint8Array(buffer, fp, dataBytes));
        fp += dataBytes;
        data = unstuff(words, bitsPerPixel, numValidPixels, offset, scale, blockData, globalMax);
      } else if (encoding === 3) {
        constValue = offset;
      }
      // encoding === 2 → constValue = 0。

      // 回填：数据按有效像素紧凑序（掩码洞不占位）。
      let outPtr = by * width * blockHeight + bx * blockWidth;
      let blockPtr = 0;
      for (let yy = 0; yy < bh; yy++) {
        for (let xx = 0; xx < bw; xx++) {
          const valid = mask === null || bitAt(mask, outPtr);
          if (valid) {
            const value = encoding < 2 && data ? data[blockPtr++] : constValue;
            pixels[outPtr] = value;
            if (value < minValue) minValue = value;
          }
          outPtr++;
        }
        outPtr += width - bw;
      }
    }
  }

  if (minValue === Number.POSITIVE_INFINITY) minValue = 0;
  let maxValue = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < pixelCount; i++) maxValue = Math.max(maxValue, pixels[i]);
  return { width, height, pixels, mask, minValue, maxValue };
}

/**
 * 位流解包（对齐 lerc.js unstuff）：MSB-first 消费小端 Uint32 字流；尾字
 * 按「无效尾字节数」左移对齐；n ≥ nmax 的量化越界值钳到 maxValue。
 */
function unstuff(
  src: Uint32Array,
  bitsPerPixel: number,
  numPixels: number,
  offset: number,
  scale: number,
  dest: Float32Array,
  maxValue: number
): Float32Array {
  if (bitsPerPixel === 0) {
    dest.fill(Math.min(offset, maxValue));
    return dest;
  }
  const bitMask = (1 << bitsPerPixel) - 1;
  const nmax = scale > 0 ? Math.ceil((maxValue - offset) / scale) : Number.POSITIVE_INFINITY;
  const numInvalidTailBytes = src.length * 4 - Math.ceil((bitsPerPixel * numPixels) / 8);
  if (numInvalidTailBytes > 0) {
    src[src.length - 1] = (src[src.length - 1] << (8 * numInvalidTailBytes)) >>> 0;
  }

  let i = 0;
  let bitsLeft = 0;
  let buffer = 0;
  for (let o = 0; o < numPixels; o++) {
    if (bitsLeft === 0) {
      buffer = src[i++];
      bitsLeft = 32;
    }
    let n: number;
    if (bitsLeft >= bitsPerPixel) {
      n = (buffer >>> (bitsLeft - bitsPerPixel)) & bitMask;
      bitsLeft -= bitsPerPixel;
    } else {
      const missingBits = bitsPerPixel - bitsLeft;
      n = ((buffer & bitMask) << missingBits) & bitMask;
      buffer = src[i++];
      bitsLeft = 32 - missingBits;
      n += buffer >>> bitsLeft;
    }
    dest[o] = n < nmax ? offset + n * scale : maxValue;
  }
  return dest;
}

/** 掩码 RLE：int16 计数（正=字面字节，负=重复下一字节），哨兵 -32768。 */
function unpackMaskRle(view: DataView, pixelCount: number): Uint8Array {
  const bitset = new Uint8Array(Math.ceil(pixelCount / 8));
  let ip = 0;
  let op = 0;
  let cnt = view.getInt16(ip, true);
  ip += 2;
  do {
    if (cnt > 0) {
      while (cnt--) bitset[op++] = view.getUint8(ip++);
    } else {
      const val = view.getUint8(ip++);
      cnt = -cnt;
      while (cnt--) bitset[op++] = val;
    }
    cnt = view.getInt16(ip, true);
    ip += 2;
  } while (ip < view.byteLength);
  if (cnt !== -32768 || op < bitset.length) {
    throw new Error('LercDecoder: unexpected end of mask RLE encoding');
  }
  return bitset;
}

function bitAt(bitset: Uint8Array, index: number): boolean {
  return (bitset[index >> 3] & (0x80 >> (index & 7))) !== 0;
}

// ── 测试专用编码器 ───────────────────────────────────────────────────────────

/**
 * LERC1 编码器（**测试专用**：与解码器共享位序语义，供 vitest 合成瓦片；
 * 覆盖常值 / 简单梯度 / 掩码洞三档，仅产出解码器消费的字段子集）。
 */
export function encodeLerc1TestTile(options: {
  width: number;
  height: number;
  /** 值 = f(col, row)；缺省用 constant。 */
  value?: (col: number, row: number) => number;
  constant?: number;
  /** true = 无效像素（掩码洞）。 */
  hole?: (col: number, row: number) => boolean;
  maxZError?: number;
}): ArrayBuffer {
  const { width, height } = options;
  const maxZError = options.maxZError ?? 0.001;
  const scale = 2 * maxZError;
  const pixelCount = width * height;

  const values = new Float32Array(pixelCount);
  const bitset = options.hole ? new Uint8Array(Math.ceil(pixelCount / 8)) : null;
  let min = Infinity;
  let max = -Infinity;
  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      const i = r * width + c;
      const v = options.value ? options.value(c, r) : (options.constant ?? 0);
      values[i] = v;
      if (options.hole?.(c, r)) continue;
      if (bitset) bitset[i >> 3] |= 0x80 >> (i & 7);
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
  }
  if (!Number.isFinite(min)) { min = 0; max = 0; }

  const out = new ByteWriter();

  // 文件头
  out.ascii('CntZImage '); // 10B：9 字符 + 1 空格（解码端 trim 比对）
  out.i32(1);   // version
  out.i32(6);   // imageType = F32（信息字段）
  out.u32(height);
  out.u32(width);
  out.f64(maxZError);

  // 掩码：全有效 → numBytes=0 / maxValue=1；有洞 → 字面 RLE 位图 + 哨兵。
  if (bitset) {
    out.u32(1); // numBlocksY（解码器未消费具体值，非零避免「全无效」歧义）
    out.u32(1); // numBlocksX
    const body = new ByteWriter();
    let idx = 0;
    while (idx < bitset.length) {
      const take = Math.min(32767, bitset.length - idx);
      body.i16(take);
      for (let k = 0; k < take; k++) body.u8(bitset[idx + k]);
      idx += take;
    }
    body.i16(-32768); // 哨兵
    out.u32(body.length);
    out.f32(1);
    out.bytes(body);
  } else {
    out.u32(0);
    out.u32(0);
    out.u32(0);   // numBytes = 0
    out.f32(1);   // maxValue ≠ 0 → 全部有效
  }

  // 像素：单块（numBlocks 1×1）。常值 → encoding 3（0 时 encoding 2）；
  // 梯度 → encoding 1 bitstuffed（offset = min 对齐 scale 网格）。
  const validValues: number[] = [];
  if (bitset) {
    for (let i = 0; i < pixelCount; i++) {
      if (bitAt(bitset, i)) validValues.push(values[i]);
    }
  } else {
    for (let i = 0; i < pixelCount; i++) validValues.push(values[i]);
  }
  const numValid = validValues.length;

  out.u32(1); // numBlocksY
  out.u32(1); // numBlocksX
  out.u32(0); // numBytes（仅 raw 块消费；本编码器不产 raw）
  out.f32(max + scale); // maxValue：保证 nmax 严格大于最大 n，不触发钳制

  const allSame = numValid === 0 || validValues.every((v) => v === validValues[0]);
  if (allSame) {
    const v = numValid === 0 ? 0 : validValues[0];
    if (v === 0) {
      out.u8(2); // encoding 2：常 0 块（仅头字节）
    } else {
      out.u8(3 | (0 << 6)); // encoding 3 + offsetType 0（f32）
      out.f32(v);
    }
  } else {
    const offset = Math.floor(min / scale) * scale;
    const ints = validValues.map((v) => Math.round((v - offset) / scale));
    const maxInt = Math.max(...ints);
    const bitsPerPixel = Math.max(1, 32 - Math.clz32(maxInt));
    out.u8(1 | (0 << 6)); // encoding 1 + offsetType 0（f32 offset）
    out.f32(offset);
    out.u8(bitsPerPixel | (2 << 6)); // bpp + numValidPixelsType 2（u8 计数）
    out.u8(numValid);
    out.stuffedBits(ints, bitsPerPixel);
  }

  return out.finish();
}

/** 小端字节流写入器（测试编码器专用）。 */
class ByteWriter {
  private _bytes: number[] = [];

  get length(): number { return this._bytes.length; }

  u8(v: number): void { this._bytes.push(v & 0xff); }
  i16(v: number): void {
    this._bytes.push(v & 0xff, (v >> 8) & 0xff);
  }
  u32(v: number): void { this.writeNum_(v, 4, false); }
  i32(v: number): void { this.writeNum_(v, 4, true); }
  f32(v: number): void {
    const d = new DataView(new ArrayBuffer(4));
    d.setFloat32(0, v, true);
    this.bytes({ _bytes: [...new Uint8Array(d.buffer)] } as unknown as ByteWriter);
  }
  f64(v: number): void {
    const d = new DataView(new ArrayBuffer(8));
    d.setFloat64(0, v, true);
    this.bytes({ _bytes: [...new Uint8Array(d.buffer)] } as unknown as ByteWriter);
  }
  ascii(s: string): void { for (const ch of s) this.u8(ch.charCodeAt(0)); }
  bytes(w: ByteWriter): void { this._bytes.push(...w._bytes); }

  /**
   * 位填充：values 的 bitsPerPixel 位整数以 MSB-first 流写入；字（32 位）
   * 按小端序列化，尾字只落 ⌈bits/8⌉ 个「最高有效字节」（对齐解码器的
   * 无效尾字节左移约定）。
   */
  stuffedBits(values: number[], bitsPerPixel: number): void {
    const totalBits = values.length * bitsPerPixel;
    const words: number[] = [];
    let word = 0;
    let filled = 0;
    for (const v of values) {
      for (let b = bitsPerPixel - 1; b >= 0; b--) {
        if ((v >>> b) & 1) word |= 1 << (31 - filled);
        if (++filled === 32) { words.push(word); word = 0; filled = 0; }
      }
    }
    if (filled > 0) words.push(word);

    for (let w = 0; w < words.length; w++) {
      const isLast = w === words.length - 1;
      const meaningfulBits = isLast ? ((totalBits - 1) % 32) + 1 : 32;
      const byteCount = isLast ? Math.ceil(meaningfulBits / 8) : 4;
      // 顶 byteCount 个字节按「大端序」发射：disk byte j = lane (4-byteCount+j)。
      for (let j = 0; j < byteCount; j++) {
        const lane = 4 - byteCount + j;
        this.u8((words[w] >>> (8 * lane)) & 0xff);
      }
    }
  }

  finish(): ArrayBuffer {
    const buf = new ArrayBuffer(this._bytes.length);
    new Uint8Array(buf).set(this._bytes);
    return buf;
  }

  private writeNum_(v: number, bytes: number, signed: boolean): void {
    const d = new DataView(new ArrayBuffer(bytes));
    if (bytes === 4) (signed ? d.setInt32(0, v, true) : d.setUint32(0, v, true));
    for (const b of new Uint8Array(d.buffer)) this._bytes.push(b);
  }
}
