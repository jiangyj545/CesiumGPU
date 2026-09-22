import { describe, expect, it } from 'vitest';
import { CityBlockBuilder } from '../../src/geometry/CityBlockBuilder';

/**
 * F12 实例 id 数据层单测（纯数据，无 GPU 依赖）：
 * CityBlockBuilder 为每栋建筑分配 builder 内全局唯一的实例 id（1 起），
 * 逐顶点填充进 `CityBlockChunk.instanceIds`；类别 id（semanticIds）保持
 * 原语义不变——类别给调色板、实例给 bbox。
 */

/** 4 点矩形 footprint（经纬度，度） */
function squareRing(lon: number, lat: number, half = 0.00002) {
  return [
    { lon: lon - half, lat: lat - half },
    { lon: lon + half, lat: lat - half },
    { lon: lon + half, lat: lat + half },
    { lon: lon - half, lat: lat + half },
  ];
}

describe('CityBlockBuilder F12 实例 id', () => {
  it('两栋建筑 → 两个互不相同的实例 id，逐顶点填充', () => {
    const b = new CityBlockBuilder();
    b.addBuilding(squareRing(114.0, 30.0), 20, 1);
    b.addBuilding(squareRing(114.001, 30.0), 35, 1);

    const chunks = b.buildChunks();
    expect(chunks).toHaveLength(1);
    const chunk = chunks[0]!;
    const vc = chunk.geometry.vertexCount;

    expect(chunk.instanceIds).toBeInstanceOf(Uint32Array);
    expect(chunk.instanceIds.length).toBe(vc);

    const unique = new Set(chunk.instanceIds);
    expect(unique.size).toBe(2);
    expect(unique.has(1)).toBe(true);
    expect(unique.has(2)).toBe(true);
    // id 全为正整数（建筑序号）
    for (let i = 0; i < vc; i++) {
      expect(Number.isInteger(chunk.instanceIds[i])).toBe(true);
      expect(chunk.instanceIds[i]).toBeGreaterThan(0);
    }
    expect(chunk.buildingCount).toBe(2);
  });

  it('semanticIds 类别语义不受影响（同类别多实例）', () => {
    const b = new CityBlockBuilder();
    b.addBuilding(squareRing(114.0, 30.0), 20, 1);
    b.addBuilding(squareRing(114.001, 30.0), 35, 1);

    const chunk = b.buildChunks()[0]!;
    const clsSet = new Set(chunk.geometry.semanticIds);
    expect(clsSet.size).toBe(1);
    expect(clsSet.has(1)).toBe(true);
    // 实例分离 ≠ 类别合并：两楼同类但实例 id 不同
    expect(new Set(chunk.instanceIds).size).toBe(2);
  });

  it('跨分块：实例 id 全 builder 单调递增不重复', () => {
    const b = new CityBlockBuilder({ maxChunkBytes: 1 }); // 极小预算 → 每楼一 chunk
    for (let i = 0; i < 3; i++) {
      b.addBuilding(squareRing(114.0 + i * 0.001, 30.0), 20, 1);
    }
    const chunks = b.buildChunks();
    expect(chunks.length).toBeGreaterThanOrEqual(2);
    const all: number[] = [];
    for (const c of chunks) all.push(...c.instanceIds);
    // 逐顶点值可重复（同一楼内恒定），但不同楼的 id 值域 = {1,2,3} 互不串块
    expect(new Set(all)).toEqual(new Set([1, 2, 3]));
    expect(Math.min(...all)).toBe(1);
    expect(Math.max(...all)).toBe(3);
  });

  it('id 上限防护：MAX_INSTANCE_ID = 0x7FFFFF（23-bit，bit23 留作实例标记位）', () => {
    expect(CityBlockBuilder.MAX_INSTANCE_ID).toBe(0x7FFFFF);
  });
});
