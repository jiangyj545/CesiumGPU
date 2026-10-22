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
  /** 实际索引值（写入时会按高水位规则编码）。 */
  indices: number[];
  minH: number;
  maxH: number;
  /** 宣称的索引数（默认 = indices.length；真实服务只写第一段）。 */
  declaredIndexCount?: number;
  /** 高程范围写法：官方 f32 对 @+24（默认）或 f64 对 @+56/+64 回退布局。 */
  heightEncoding?: 'f32' | 'f64';
}

/**
 * 将索引按 CesiumJS 高水位规则编码：
 * idx == highest 时写 0 并推进 highest；否则写 highest - idx。
 */
function hwmEncode(indices: number[]): number[] {
  let highest = 0;
  const out: number[] = [];
  for (const idx of indices) {
    if (idx === highest) {
      out.push(0);
      highest++;
    } else {
      out.push(highest - idx);
    }
  }
  return out;
}

/** 按官方 CesiumJS 88 字节头布局手工打包一个合成瓦片（未压缩）。 */
function packTile(tile: SyntheticTile): Uint8Array {
  const vCount = tile.vertices.length;
  const enc = hwmEncode(tile.indices);
  const declared = tile.declaredIndexCount ?? tile.indices.length;
  // 88 头 + 顶点 + u32 declared + 索引 + 边缘区(4×u32 计数 1..200 占位)
  const size = 92 + vCount * 6 + 4 + enc.length * 2 + 16;
  const buffer = new ArrayBuffer(size);
  const view = new DataView(buffer);

  // +0  center 3 x f64
  view.setFloat64(0, tile.center[0], true);
  view.setFloat64(8, tile.center[1], true);
  view.setFloat64(16, tile.center[2], true);
  // +24 minHeight / maxHeight（官方规范的 f32 对）
  if ((tile.heightEncoding ?? 'f32') === 'f32') {
    view.setFloat32(24, tile.minH, true);
    view.setFloat32(28, tile.maxH, true);
  } else {
    // 官方位置写入非法值触发回退；真实值写在 +56/+64 的 f64 对
    view.setFloat32(24, 99999, true);
    view.setFloat32(28, 99999, true);
    view.setFloat64(56, tile.maxH, true);
    view.setFloat64(64, tile.minH, true);
  }
  // +32 boundingSphere center 3 x f64 + radius f64
  view.setFloat64(32, tile.center[0], true);
  view.setFloat64(40, tile.center[1], true);
  view.setFloat64(48, tile.center[2], true);
  view.setFloat64(56, tile.heightEncoding === 'f64' ? tile.maxH : 1234.5, true);
  // +64 horizonOcclusion 3 x f64（f64 布局时 +64 被复用为 minH）
  view.setFloat64(64, tile.heightEncoding === 'f64' ? tile.minH : 0.5, true);
  view.setFloat64(72, 0.5, true);
  view.setFloat64(80, 0.5, true);
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

  // u32 declaredIndexCount + 连续 HWM 编码 u16 流
  view.setUint32(off, declared, true);
  off += 4;
  for (const code of enc) {
    view.setUint16(off, code, true);
    off += 2;
  }
  // 边缘区特征：4 个 u32 计数（1..200），供边界探测识别
  for (let e = 0; e < 4; e++) {
    view.setUint32(off, 4, true);
    off += 4;
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

      // 主网格顶点位置 = ECEF(lon(u 边界), lat(v 边界), h)——真椭球约定
      // （ecefFromLonLatHeight，2026-10-22）：x=n·nx, y=n·ny, z=n(1−e²)·nz。
      const bounds = geographicTileBounds(10, 100, 200);
      const rad = Math.PI / 180;
      const ratio = 6356752.3142 / 6378137;
      const e2 = 1 - ratio * ratio;
      const ell = (latDeg: number) => {
        const sinLat = Math.sin(latDeg * rad);
        return 1 / Math.sqrt(1 - e2 * sinLat * sinLat);
      };
      const scale0 = 1 + 100 / 6378137; // hq=0 → h = minH = 100m
      const nS = ell(bounds.south);
      const nx = Math.cos(bounds.south * rad) * Math.cos(bounds.west * rad);
      const ny = Math.cos(bounds.south * rad) * Math.sin(bounds.west * rad);
      const nz = Math.sin(bounds.south * rad);
      expect(mesh.positions[0]).toBeCloseTo(nS * nx * scale0, 5);
      expect(mesh.positions[1]).toBeCloseTo(nS * ny * scale0, 5);
      expect(mesh.positions[2]).toBeCloseTo(nS * (1 - e2) * nz * scale0, 5);

      // 顶点 2（u=0, v=32767, hq=32767）→ 西北角、最高处
      const scale2 = 1 + 1100 / 6378137;
      const nN = ell(bounds.north);
      const nx2 = Math.cos(bounds.north * rad) * Math.cos(bounds.west * rad);
      const ny2 = Math.cos(bounds.north * rad) * Math.sin(bounds.west * rad);
      const nz2 = Math.sin(bounds.north * rad);
      const p2 = Array.from(mesh.positions.slice(2 * 3, 2 * 3 + 3));
      expect(p2[0]).toBeCloseTo(nN * nx2 * scale2, 5);
      expect(p2[1]).toBeCloseTo(nN * ny2 * scale2, 5);
      expect(p2[2]).toBeCloseTo(nN * (1 - e2) * nz2 * scale2, 5);
    });

    it('基础索引高水位解码还原原始索引', () => {
      const mesh = decodeQuantizedMesh(packTile(cornerTile), 10, 100, 200);
      // 6 个基础索引 + 4 条边 × 2 三角形 × 3 索引 = 30
      expect(mesh.indices.length).toBe(30);
      expect(Array.from(mesh.indices.slice(0, 6))).toEqual([0, 1, 2, 1, 3, 2]);
    });

    it('declaredIndexCount 只覆盖第一段时仍解出全量索引（边界探测）', () => {
      const tile: SyntheticTile = {
        center: [0, 0, 0],
        vertices: [
          { u: 0, v: 0, h: 0 },
          { u: 32767, v: 0, h: 16383 },
          { u: 0, v: 32767, h: 32767 },
          { u: 32767, v: 32767, h: 16383 },
        ],
        indices: [0, 1, 2, 1, 3, 2],
        minH: 0,
        maxH: 1000,
        declaredIndexCount: 3, // 真实服务只声明第一段
      };
      const mesh = decodeQuantizedMesh(packTile(tile), 10, 100, 200);
      expect(Array.from(mesh.indices.slice(0, 6))).toEqual([0, 1, 2, 1, 3, 2]);
    });

    it('declared 为三角形数（真实服务布局）：解出 3×declared 全量索引', () => {
      // 3×3 网格 8 个三角形；索引流按「首次出现顺序」编号（高水位可编码）。
      // 真实 8888 服务在顶点区后写的是**三角形数**（如 10/1650/690：
      // declared=5518 → 16554 个 u16 值）。旧读法只取前 1/3，瓦片近半
      // 无三角形 → sampleHeight 覆盖 ~50%（63 页实测 33%）。
      const stream = [
        0, 1, 2, 0, 2, 3, // cellA
        1, 4, 5, 1, 5, 2, // cellB
        3, 2, 6, 3, 6, 7, // cellC
        2, 5, 8, 2, 8, 6, // cellD
      ];
      const tile: SyntheticTile = {
        center: [0, 0, 0],
        vertices: [
          { u: 0, v: 0, h: 0 },
          { u: 16383, v: 0, h: 4095 },
          { u: 16383, v: 16383, h: 8191 },
          { u: 0, v: 16383, h: 2047 },
          { u: 32767, v: 0, h: 6143 },
          { u: 32767, v: 16383, h: 10239 },
          { u: 16383, v: 32767, h: 14335 },
          { u: 0, v: 32767, h: 3071 },
          { u: 32767, v: 32767, h: 12287 },
        ],
        indices: stream,
        minH: 0,
        maxH: 2000,
        declaredIndexCount: 8, // 三角形数（非值数）
      };
      const mesh = decodeQuantizedMesh(packTile(tile), 10, 1650, 690);
      // 基础索引 = 3×declared = 24，全部 < vertexCount 且还原原始流
      const base = Array.from(mesh.indices.slice(0, 24));
      expect(base).toEqual(stream);
      expect(Math.max(...base)).toBeLessThan(9);
    });

    it('高程官方 f32 对非法时回退 +56/+64 的 f64 对', () => {
      const tile: SyntheticTile = {
        ...cornerTile,
        heightEncoding: 'f64',
        minH: -9.5,
        maxH: 3373.004,
      };
      const mesh = decodeQuantizedMesh(packTile(tile), 10, 100, 200);
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
