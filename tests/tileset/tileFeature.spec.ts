/**
 * tileFeature.spec.ts – P1-E 3DTiles feature 挑拣：
 * b3dm _BATCHID 解析 → 全局 instance id → TilesetPickRegistry → feature。
 * 全部合成数据（buildB3dm 复用 b3dmDecoder.spec 的布局构造方式）。
 */
import { describe, expect, it } from 'vitest';
import { parseBatchTable } from '../../src/tileset/B3dmDecoder';
import { GltfLoader } from '../../src/loader/GltfLoader';
import { buildGlb, buildMinimalGltfJson, defaultPositionBin } from '../loader/helpers';
import {
  Cesium3DTileFeature,
  TilesetPickRegistry,
  buildTileInstanceIds,
  decodePickInstanceId,
} from '../../src/tileset/Cesium3DTileFeature';
import type { Primitive } from '../../src/scene/Primitive';

const B3DM_MAGIC = 0x6d643362; // 'b3dm' 小端

function pad4(n: number): number {
  return (4 - (n % 4)) % 4;
}

const fakePrimitive = (n: number) => ({ label: `p${n}` }) as unknown as Primitive;

function encodeTable(payload: object | undefined): { bytes: Uint8Array; paddedLen: number } {
  if (!payload) return { bytes: new Uint8Array(0), paddedLen: 0 };
  const raw = new TextEncoder().encode(JSON.stringify(payload));
  const pad = pad4(raw.length);
  const bytes = new Uint8Array(raw.length + pad);
  bytes.set(raw, 0);
  bytes.fill(0x20, raw.length);
  return { bytes, paddedLen: bytes.length };
}

interface B3dmPart {
  featureTableJson?: object;
  featureTableBin?: Uint8Array;
  batchTableJson?: object;
  batchTableBin?: Uint8Array;
}

function buildB3dm(parts: B3dmPart, glb: ArrayBuffer): ArrayBuffer {
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
  view.setUint32(4, 1, true);
  view.setUint32(8, ftj.paddedLen, true);
  view.setUint32(12, ftb.length, true);
  view.setUint32(16, btj.paddedLen, true);
  view.setUint32(20, btb.length, true);
  view.setUint32(24, 0, true);
  let off = 28;
  bytes.set(ftj.bytes, off); off += ftj.paddedLen;
  bytes.set(ftb, off);       off += ftb.length;
  bytes.set(btj.bytes, off); off += btj.paddedLen;
  bytes.set(btb, off);       off += btb.length;
  bytes.set(glbBytes, off);
  return out;
}

/** 3 顶点非索引三角形 + 每 顶点 _BATCHID [0,1,0] 的 GLB。 */
function buildBatchedGlb(): ArrayBuffer {
  const json = {
    ...buildMinimalGltfJson(),
    meshes: [
      {
        primitives: [
          { attributes: { POSITION: 0, _BATCHID: 1 }, mode: 4 },
        ],
      },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5126, count: 3, type: 'SCALAR' },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36 },
      { buffer: 0, byteOffset: 36, byteLength: 12 },
    ],
    buffers: [{ byteLength: 48 }],
  };
  const bin = new Uint8Array(48);
  bin.set(new Uint8Array(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]).buffer), 0);
  bin.set(new Uint8Array(new Float32Array([0, 1, 0]).buffer), 36);
  return buildGlb(json, bin);
}

describe('GltfLoader._BATCHID', () => {
  it('解析 _BATCHID 为逐顶点 Uint32Array（对齐最终顶点布局）', async () => {
    const parsed = await GltfLoader.parseGlbBuffer(buildBatchedGlb());
    expect(parsed.primitives).toHaveLength(1);
    expect(Array.from(parsed.primitives[0].batchIds ?? [])).toEqual([0, 1, 0]);
  });

  it('无 _BATCHID 时 batchIds 为 undefined', async () => {
    const parsed = await GltfLoader.parseGlbBuffer(
      buildGlb(buildMinimalGltfJson(), defaultPositionBin())
    );
    expect(parsed.primitives[0].batchIds).toBeUndefined();
  });
});

describe('B3dmDecoder.parseBatchTable', () => {
  it('展开 JSON 数组列与二进制列', () => {
    const batchTableBin = new Uint8Array(new Float32Array([12.5, 40]).buffer);
    const buffer = buildB3dm(
      {
        featureTableJson: { BATCH_LENGTH: 2 },
        batchTableJson: {
          name: ['tower-a', 'tower-b'],
          height: { byteOffset: 0, type: 'SCALAR', componentType: 5126 },
        },
        batchTableBin,
      },
      buildBatchedGlb()
    );
    const table = parseBatchTable(buffer, 2);
    expect(table).not.toBeNull();
    expect(table!['name']).toEqual(['tower-a', 'tower-b']);
    expect(table!['height']).toEqual([12.5, 40]);
  });

  it('BATCH_LENGTH=0 / 缺表 → null（tile 级降级）', () => {
    const buffer = buildB3dm({ featureTableJson: { BATCH_LENGTH: 0 } }, buildBatchedGlb());
    expect(parseBatchTable(buffer, 0)).toBeNull();
    expect(parseBatchTable(buffer, 2)).toBeNull(); // 无 batchTableJson
  });
});

describe('Cesium3DTileFeature + TilesetPickRegistry', () => {
  const fakePrimitive = (n: number) => ({ label: `p${n}` }) as unknown as Primitive;

  it('buildTileInstanceIds 偏移到全局空间（base + id + 1，0 保留）', () => {
    const ids = buildTileInstanceIds(new Uint32Array([0, 1, 0]), 100);
    expect(Array.from(ids)).toEqual([101, 102, 101]);
  });

  it('decodePickInstanceId 提取实例位', () => {
    expect(decodePickInstanceId(0x800005)).toBe(5);
    expect(decodePickInstanceId(42)).toBeNull();
  });

  it('allocate → register → resolve 全链路：batchId 回读 + 属性访问', () => {
    const registry = new TilesetPickRegistry();
    const table = { name: ['a', 'b'], height: [10, 20] };
    const prim = fakePrimitive(1);
    const base = registry.allocate(2);
    registry.registerInstanceRange(prim, base, 2, new Cesium3DTileFeature({}, undefined, table, prim));

    const feature = registry.resolve(base + 1 + 1)!; // batchId 1
    expect(feature).not.toBeNull();
    expect(feature.batchId).toBe(1);
    expect(feature.getProperty('name')).toBe('b');
    expect(feature.getProperty('height')).toBe(20);
    expect(feature.getProperty('missing')).toBeUndefined();
    expect(feature.hasProperty('name')).toBe(true);
    expect(feature.hasProperty('missing')).toBe(false);
    expect(feature.getPropertyNames().sort()).toEqual(['height', 'name']);

    // 相邻瓦片第二个 allocate 不重叠
    const base2 = registry.allocate(3);
    expect(base2).toBeGreaterThan(base + 2);
    expect(registry.resolve(base2 + 1)).toBeNull(); // batchId 0 未注册
  });

  it('unregister 后 instance 解析与 tile 级降级均失效', () => {
    const registry = new TilesetPickRegistry();
    const prim = fakePrimitive(2);
    const base = registry.allocate(1);
    registry.registerInstanceRange(prim, base, 1, new Cesium3DTileFeature({}, undefined, null, prim));
    registry.registerTilePrimitive(prim, new Cesium3DTileFeature({}, undefined, null, prim));
    registry.unregister(prim);
    expect(registry.resolve(base + 1)).toBeNull();
    expect(registry.featureForPrimitive(prim)).toBeNull();
  });
  it('tile 级 feature（无 batchId）：getProperty 恒 undefined', () => {
    const registry = new TilesetPickRegistry();
    const prim = fakePrimitive(3);
    registry.registerTilePrimitive(prim, new Cesium3DTileFeature({}, undefined, null, prim));
    const feature = registry.featureForPrimitive(prim)!;
    expect(feature.batchId).toBeUndefined();
    expect(feature.getProperty('name')).toBeUndefined();
  });
});

describe('合成 b3dm 端到端（解析层）', () => {
  it('b3dm → batchTable + batchIds → registry feature', async () => {
    const buffer = buildB3dm(
      {
        featureTableJson: { BATCH_LENGTH: 2 },
        batchTableJson: { name: ['x', 'y'] },
      },
      buildBatchedGlb()
    );
    const table = parseBatchTable(buffer, 2)!;
    const parsed = await GltfLoader.parseGlbBuffer(buildBatchedGlb());
    const batchIds = parsed.primitives[0].batchIds!;
    expect(Array.from(batchIds)).toEqual([0, 1, 0]);

    const registry = new TilesetPickRegistry();
    const prim = fakePrimitive(9);
    const base = registry.allocate(2);
    registry.registerInstanceRange(prim, base, 2, new Cesium3DTileFeature({}, undefined, table, prim));
    const feature = registry.resolve(base + 1)!;
    expect(feature.getProperty('name')).toBe('x');
  });
});
