import { describe, it, expect } from 'vitest';
import { gzipSync } from 'node:zlib';
import {
  zigzagDecode,
  decompressGzip,
  decodeQuantizedMesh,
} from '../../src/terrain/QuantizedMeshDecoder';
import { geographicTileBounds } from '../../src/terrain/TerrainProvider';

/** 与解码器互逆的 zigzag 编码（delta 编码前的单值变换）。 */
function zigzagEncode(value: number): number {
  return (value << 1) ^ (value >> 31);
}

interface SyntheticTile {
  center: [number, number, number];
  /** 量化 u/v/h 值（u/v ∈ [0,32767]，h 为量化高度码）。 */
  vertices: Array<{ u: number; v: number; h: number }>;
  /** 实际索引值（写入前会减去 watermark）。 */
  indices: number[];
  minH: number;
  maxH: number;
  /** 默认 65536（无 HiLo 偏移）。 */
  watermark?: number;
  /** 高程范围写法：f32 对（默认）或 f64 对回退布局。 */
  heightEncoding?: 'f32' | 'f64';
}

/** 按小端变体布局手工打包一个合成瓦片（未压缩）。 */
function packTile(tile: SyntheticTile): Uint8Array {
  const watermark = tile.watermark ?? 65536;
  const vCount = tile.vertices.length;
  const size = 92 + vCount * 6 + 8 + tile.indices.length * 2;
  const buffer = new ArrayBuffer(size);
  const view = new DataView(buffer);

  // +0  center 3 x f64
  view.setFloat64(0, tile.center[0], true);
  view.setFloat64(8, tile.center[1], true);
  view.setFloat64(16, tile.center[2], true);
  // +24 horizon occlusion 2 x f32（跳过区，填占位值）
  view.setFloat32(24, 1.5, true);
  view.setFloat32(28, -2.5, true);
  // +32 center 重复 3 x f64
  view.setFloat64(32, tile.center[0], true);
  view.setFloat64(40, tile.center[1], true);
  view.setFloat64(48, tile.center[2], true);
  // +56 高程范围
  if ((tile.heightEncoding ?? 'f32') === 'f32') {
    view.setFloat32(56, tile.minH, true);
    view.setFloat32(60, tile.maxH, true);
  } else {
    // 构造「f32 读数不合理、f64 对合理」的字节布局：
    // +56 的 f64 = 0xC0230000477FFF00 ≈ -9.5，其低 4 字节按 f32 重读
    // = 0x477FFF00 ≈ 65535 → 超出合理范围 → 触发 f64 回退
    view.setUint32(56, 0x477fff00, true);
    view.setUint32(60, 0xc0230000, true);
    view.setFloat64(64, tile.maxH, true);
  }
  // +72 杂项 f64（跳过区）
  view.setFloat64(72, 9.0, true);
  view.setFloat64(80, 10.0, true);
  // +88 vertexCount
  view.setUint32(88, vCount, true);

  // +92 zigzag-delta u / v / h 三通道
  let off = 92;
  for (const channel of ['u', 'v', 'h'] as const) {
    let prev = 0;
    for (const vertex of tile.vertices) {
      const value = vertex[channel];
      view.setUint16(off, zigzagEncode(value - prev), true);
      prev = value;
      off += 2;
    }
  }

  // 索引块：indexCount u32 + watermark u32 + (actual - watermark) u16 数组
  view.setUint32(off, tile.indices.length, true);
  view.setUint32(off + 4, watermark, true);
  off += 8;
  for (const index of tile.indices) {
    view.setUint16(off, index - watermark, true);
    off += 2;
  }
  return new Uint8Array(buffer);
}

describe('quantizedMeshDecoder', () => {
  describe('zigzag-delta 往返', () => {
    it('单值往返：decode(encode(v)) === v（含负数与边界）', () => {
      const values = [0, 1, -1, 2, -2, 32767, -32768, 65535, -65535, 123456];
      for (const value of values) {
        expect(zigzagDecode(zigzagEncode(value))).toBe(value);
      }
    });

    it('delta 序列往返：累积解码还原原始 u/v/h 通道', () => {
      // 模拟一条真实顶点轨迹（含负增量）
      const raw = [100, 200, 150, 140, 300];
      const deltas: number[] = [];
      let prev = 0;
      for (const value of raw) {
        deltas.push(value - prev);
        prev = value;
      }
      let acc = 0;
      const decoded = deltas.map((d) => (acc += zigzagDecode(zigzagEncode(d))));
      expect(decoded).toEqual(raw);
    });
  });

  describe('合成小端变体瓦片解码', () => {
    // 四角顶点瓦片：minH=100, maxH=1100（跨度 1000m）
    const cornerTile: SyntheticTile = {
      center: [1.5, 2.5, 3.5],
      vertices: [
        { u: 0, v: 0, h: 0 }, // 西南角，h = 100m
        { u: 32767, v: 0, h: 16383 }, // 东南角
        { u: 0, v: 32767, h: 32767 }, // 西北角，h = 1100m
        { u: 32767, v: 32767, h: 16383 },
      ],
      indices: [0, 1, 2, 1, 3, 2],
      minH: 100,
      maxH: 1100,
    };

    it('gzip 解压 + 解码：positions/minMax/center 正确', async () => {
      const plain = packTile(cornerTile);
      const gz = gzipSync(Buffer.from(plain));
      const raw = await decompressGzip(new Uint8Array(gz));
      expect(Array.from(raw)).toEqual(Array.from(plain));

      // z=10, x=100, y=200（geographic TMS south-up）
      const mesh = decodeQuantizedMesh(raw, 10, 100, 200);
      expect(mesh.minHeight).toBeCloseTo(100, 3);
      expect(mesh.maxHeight).toBeCloseTo(1100, 3);
      expect(mesh.center).toEqual([1.5, 2.5, 3.5]);

      // 4 顶点全在边缘 → 每个复制一个裙边顶点：8 顶点 24 floats
      expect(mesh.positions.length).toBe(8 * 3);

      // 主网格顶点位置 = ECEF(lon(u 边界), lat(v 边界), h)
      const bounds = geographicTileBounds(10, 100, 200);
      const rad = Math.PI / 180;
      const ratio = 6356752.3142 / 6378137;
      const scale0 = 1 + 100 / 6378137; // hq=0 → h = minH = 100m
      const nx = Math.cos(bounds.south * rad) * Math.cos(bounds.west * rad);
      const ny = Math.cos(bounds.south * rad) * Math.sin(bounds.west * rad);
      const nz = Math.sin(bounds.south * rad);
      expect(mesh.positions[0]).toBeCloseTo(nx * scale0, 5);
      expect(mesh.positions[1]).toBeCloseTo(ny * ratio * scale0, 5);
      expect(mesh.positions[2]).toBeCloseTo(nz * ratio * scale0, 5);

      // 顶点 2（u=0, v=32767, hq=32767）→ 西北角、最高处
      const scale2 = 1 + 1100 / 6378137;
      const nx2 = Math.cos(bounds.north * rad) * Math.cos(bounds.west * rad);
      const ny2 = Math.cos(bounds.north * rad) * Math.sin(bounds.west * rad);
      const nz2 = Math.sin(bounds.north * rad);
      const p2 = Array.from(mesh.positions.slice(2 * 3, 2 * 3 + 3));
      expect(p2[0]).toBeCloseTo(nx2 * scale2, 5);
      expect(p2[1]).toBeCloseTo(ny2 * ratio * scale2, 5);
      expect(p2[2]).toBeCloseTo(nz2 * ratio * scale2, 5);
    });

    it('基础索引原样解码（watermark=65536 不偏移）', () => {
      const mesh = decodeQuantizedMesh(packTile(cornerTile), 10, 100, 200);
      // 6 个基础索引 + 4 条边 × 2 三角形 × 3 索引 = 30
      expect(mesh.indices.length).toBe(30);
      expect(Array.from(mesh.indices.slice(0, 6))).toEqual([0, 1, 2, 1, 3, 2]);
    });

    it('watermark != 65536 时索引值加上偏移', () => {
      const tile: SyntheticTile = {
        center: [0, 0, 0],
        vertices: [
          { u: 100, v: 100, h: 0 },
          { u: 200, v: 200, h: 16383 },
          { u: 300, v: 300, h: 32767 },
        ],
        indices: [5, 6, 7], // 存储为 [0,1,2]
        minH: 0,
        maxH: 1000,
        watermark: 5,
      };
      const mesh = decodeQuantizedMesh(packTile(tile), 10, 100, 200);
      expect(Array.from(mesh.indices.slice(0, 3))).toEqual([5, 6, 7]);
      // 全内部顶点 → 无裙边，仅 3 顶点 + 3 索引
      expect(mesh.positions.length).toBe(9);
      expect(mesh.indices.length).toBe(3);
    });

    it('高程 f32 对非法时回退 +56/+64 的 f64 对', () => {
      const tile: SyntheticTile = {
        ...cornerTile,
        heightEncoding: 'f64',
        minH: -9.5, // 实际值由字节模式决定：f64 ≈ -9.5000004
        maxH: 3373.004,
      };
      const mesh = decodeQuantizedMesh(packTile(tile), 10, 100, 200);
      // f64 对读数：minH ≈ -9.5（0xC0230000477FFF00），maxH = 3373.004
      expect(mesh.minHeight).toBeGreaterThan(-9.6);
      expect(mesh.minHeight).toBeLessThan(-9.4);
      expect(mesh.maxHeight).toBeCloseTo(3373.004, 3);
    });
  });

  describe('skirt 裙边生成', () => {
    it('边缘顶点被复制且索引数增加；裙边顶点半径更小', () => {
      const tile: SyntheticTile = {
        center: [0, 0, 0],
        vertices: [
          { u: 0, v: 0, h: 0 },
          { u: 32767, v: 0, h: 16383 },
          { u: 0, v: 32767, h: 32767 },
          { u: 32767, v: 32767, h: 16383 },
        ],
        indices: [0, 1, 2, 1, 3, 2],
        minH: 100,
        maxH: 1100, // skirtDepth = max(50, 1000) = 1000m
      };
      const mesh = decodeQuantizedMesh(packTile(tile), 10, 100, 200);

      // 4 个边缘顶点 → 4 个裙边副本：顶点数 8
      expect(mesh.positions.length).toBe(8 * 3);
      // 索引：基础 6 + 裙边 4 条边 × 2 三角形 × 3 = 30（索引数增加）
      expect(mesh.indices.length).toBe(6 + 24);

      // 裙边顶点 4（顶点 0 的副本）半径 = 原半径 × (1 - 1000/6378137)
      const norm = (i: number): number =>
        Math.hypot(
          mesh.positions[i * 3],
          mesh.positions[i * 3 + 1],
          mesh.positions[i * 3 + 2]
        );
      const ratio = norm(4) / norm(0);
      expect(ratio).toBeCloseTo(1 - 1000 / 6378137, 5);
    });

    it('近平瓦片使用最小裙深 50m', () => {
      const tile: SyntheticTile = {
        center: [0, 0, 0],
        vertices: [
          { u: 0, v: 0, h: 0 },
          { u: 32767, v: 32767, h: 3 },
        ],
        indices: [],
        minH: 0,
        maxH: 10, // 跨度 10 < 50 → skirtDepth = 50
      };
      const mesh = decodeQuantizedMesh(packTile(tile), 10, 100, 200);
      const norm = (i: number): number =>
        Math.hypot(
          mesh.positions[i * 3],
          mesh.positions[i * 3 + 1],
          mesh.positions[i * 3 + 2]
        );
      expect(norm(2) / norm(0)).toBeCloseTo(1 - 50 / 6378137, 5);
    });
  });

  describe('v 方向结论（south-up TMS）自检', () => {
    it('z12/x6553/y2707 中心落在湖北（lat≈+28.98, lon≈108）', () => {
      const b = geographicTileBounds(12, 6553, 2707);
      const centerLon = (b.west + b.east) / 2;
      const centerLat = (b.south + b.north) / 2;
      // south-up：lat = (2707.5)/4096*180-90 = +28.98193359375（湖北边界）
      expect(centerLat).toBeCloseTo(28.98193359375, 6);
      // north-up 解释会得到 -28.98（南半球，非湖北）——排除
      expect(centerLat).toBeGreaterThan(28.9);
      expect(centerLat).toBeLessThan(34);
      // lon = (6553.5)/8192*360-180 = 107.99560546875
      expect(centerLon).toBeCloseTo(107.99560546875, 6);
    });

    it('z0 = 2×1（EPSG:4326 geographic TMS）', () => {
      // z0 每瓦 180° 经度 × 180° 纬度（2 列 × 1 行，行跨满南北方）
      const b = geographicTileBounds(0, 0, 0);
      expect(b.west).toBe(-180);
      expect(b.east).toBe(0);
      expect(b.south).toBe(-90);
      expect(b.north).toBe(90);
      const b1 = geographicTileBounds(0, 1, 0);
      expect(b1.west).toBe(0);
      expect(b1.east).toBe(180);
      expect(b1.north).toBe(90);
    });
  });
});
