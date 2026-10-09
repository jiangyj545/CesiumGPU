/**
 * instanceIdSpace.spec.ts – P1-E 打回回归：F12 CityBlock 实例拾取与
 * P1-E tileset 批次拾取共用 `0x800000 | id` 编码，实例 id 必须由同一
 * 进程级分配器（picking/InstanceIdPool.ts）发放，两系统互不串扰。
 */
import { describe, expect, it } from 'vitest';
import { CityBlockBuilder } from '../../src/geometry/CityBlockBuilder';
import {
  TilesetPickRegistry,
  Cesium3DTileFeature,
  decodePickInstanceId,
  PICK_INSTANCE_FLAG,
} from '../../src/tileset/Cesium3DTileFeature';
import {
  allocatePickInstanceIds,
  peekNextPickInstanceId,
} from '../../src/picking/InstanceIdPool';
import type { Primitive } from '../../src/scene/Primitive';

const fakePrimitive = () => ({ ts: Math.random() }) as unknown as Primitive;

function squareRing(lon: number, lat: number, half = 0.00002) {
  return [
    { lon: lon - half, lat: lat - half },
    { lon: lon + half, lat: lat - half },
    { lon: lon + half, lat: lat + half },
    { lon: lon - half, lat: lat + half },
  ];
}

describe('实例 id 共享分配器（CityBlock × Tileset 共存）', () => {
  it('两系统领取的区间互不相交且单调推进', () => {
    const before = peekNextPickInstanceId();
    const cityIds = allocatePickInstanceIds(3); // 模拟 CityBlock 3 栋楼
    const registry = new TilesetPickRegistry();
    const tileBase = registry.allocate(5); // 覆盖 batchId 0..4（count+1 槽）
    const after = peekNextPickInstanceId();

    expect(cityIds).toBe(before); // 首段从当前游标起
    expect(tileBase).toBeGreaterThanOrEqual(cityIds + 3); // 紧随其后，不重叠
    expect(after).toBe(tileBase + 6);
  });

  it('CityBlock 实例 id 解析不出瓦片 feature（打回场景复现）', () => {
    // 先盖 3 栋楼（真实 builder 路径），再注册瓦片批次。
    const builder = new CityBlockBuilder();
    builder.addBuilding(squareRing(114.0, 30.0), 20, 1);
    builder.addBuilding(squareRing(114.001, 30.0), 35, 1);
    builder.addBuilding(squareRing(114.002, 30.0), 50, 1);
    const chunks = builder.buildChunks();
    const cityIds = [...new Set(chunks[0]!.instanceIds)];
    expect(cityIds).toHaveLength(3);

    const registry = new TilesetPickRegistry();
    const prim = fakePrimitive();
    const base = registry.allocate(2);
    registry.registerInstanceRange(
      prim,
      base,
      2,
      new Cesium3DTileFeature({}, undefined, { name: ['a', 'b'] }, prim)
    );

    for (const cityId of cityIds) {
      // 解码正常（实例位 + 本楼序号）……
      expect(decodePickInstanceId(PICK_INSTANCE_FLAG | cityId)).toBe(cityId);
      // ……但绝不是瓦片批次：registry 解析为 null（PickPass 由此走非瓦片回退）。
      expect(registry.resolve(cityId)).toBeNull();
    }

    // 反向：瓦片 batch id 解析出 feature，且不落在任何 CityBlock id 上。
    const batchId = 1;
    const tileInstanceId = base + batchId + 1;
    const feature = registry.resolve(tileInstanceId);
    expect(feature?.batchId).toBe(batchId);
    expect(feature?.getProperty('name')).toBe('b');
    for (const cityId of cityIds) {
      expect(tileInstanceId).not.toBe(cityId);
    }
  });

  it('多个 registry 实例 / 多个 builder 共用同一进程级游标', () => {
    const r1 = new TilesetPickRegistry();
    const b1 = r1.allocate(1);
    const city = allocatePickInstanceIds(1);
    const r2 = new TilesetPickRegistry();
    const b2 = r2.allocate(1);
    // 三个区间两两不同（严格递增）。
    expect(new Set([b1, city, b2]).size).toBe(3);
    expect(r1.resolve(b2 + 1)).toBeNull(); // r2 的 id 在 r1 里解析不出
  });
});
