import { describe, expect, it } from 'vitest';
import { Entity } from '../../src/dataSource/Entity';
import { Camera } from '../../src/scene/Camera';
import {
  LabelStyle,
  HorizontalOrigin,
  VerticalOrigin,
  HeightReference,
  DistanceDisplayCondition,
  billboardAnchorOffset,
  depthTestDisabled,
  applyHeightReference,
} from '../../src/dataSource/BillboardStyle';
import { Cartesian3 } from '../../src/math/Cartesian3';

const EARTH = 6378137.0;

describe('P2-4 枚举值（Cesium 对齐）', () => {
  it('LabelStyle / HorizontalOrigin / VerticalOrigin / HeightReference', () => {
    expect(LabelStyle).toMatchObject({ FILL: 0, OUTLINE: 1, FILL_AND_OUTLINE: 2 });
    expect(HorizontalOrigin).toMatchObject({ LEFT: 0, CENTER: 1, RIGHT: 2 });
    expect(VerticalOrigin).toMatchObject({ BASELINE: 0, BOTTOM: 1, CENTER: 2, TOP: 3 });
    expect(HeightReference).toMatchObject({ NONE: 0, CLAMP_TO_GROUND: 1, RELATIVE_TO_GROUND: 2 });
  });
});

describe('DistanceDisplayCondition', () => {
  it('near/far 双端闭区间（临界值可见）', () => {
    const ddc = new DistanceDisplayCondition(100, 1000);
    expect(ddc.isVisible(99.9)).toBe(false);
    expect(ddc.isVisible(100)).toBe(true);   // near 临界
    expect(ddc.isVisible(1000)).toBe(true);  // far 临界
    expect(ddc.isVisible(1000.1)).toBe(false);
    expect(ddc.isVisible(500)).toBe(true);
  });

  it('默认全域可见', () => {
    const ddc = new DistanceDisplayCondition();
    expect(ddc.isVisible(0)).toBe(true);
    expect(ddc.isVisible(Number.MAX_VALUE)).toBe(true);
  });
});

describe('billboardAnchorOffset', () => {
  it('水平三档：LEFT/CENTER/RIGHT → ∓w/2 / 0 / +w/2', () => {
    const w = 40, h = 20;
    expect(billboardAnchorOffset(HorizontalOrigin.LEFT, VerticalOrigin.CENTER, w, h).dx).toBe(-20);
    expect(billboardAnchorOffset(HorizontalOrigin.CENTER, VerticalOrigin.CENTER, w, h).dx).toBe(0);
    expect(billboardAnchorOffset(HorizontalOrigin.RIGHT, VerticalOrigin.CENTER, w, h).dx).toBe(20);
  });

  it('垂直四档（屏幕 y 向下）：TOP +h/2、CENTER 0、BOTTOM/BASELINE −h/2', () => {
    const w = 40, h = 20;
    expect(billboardAnchorOffset(HorizontalOrigin.CENTER, VerticalOrigin.TOP, w, h).dy).toBe(10);
    expect(billboardAnchorOffset(HorizontalOrigin.CENTER, VerticalOrigin.CENTER, w, h).dy).toBe(0);
    expect(billboardAnchorOffset(HorizontalOrigin.CENTER, VerticalOrigin.BOTTOM, w, h).dy).toBe(-10);
    expect(billboardAnchorOffset(HorizontalOrigin.CENTER, VerticalOrigin.BASELINE, w, h).dy).toBe(-10);
  });

  it('dx/dy 独立组合（LEFT + BOTTOM）', () => {
    const { dx, dy } = billboardAnchorOffset(HorizontalOrigin.LEFT, VerticalOrigin.BOTTOM, 40, 20);
    expect(dx).toBe(-20);
    expect(dy).toBe(-10);
  });
});

describe('depthTestDisabled（disableDepthTestDistance 语义）', () => {
  it('未设置 → 恒 false；距离内 → true；距离外 → false；含临界', () => {
    expect(depthTestDisabled(undefined, 1e9)).toBe(false);
    expect(depthTestDisabled(500, 500)).toBe(true);   // <= 临界
    expect(depthTestDisabled(500, 500.1)).toBe(false);
    expect(depthTestDisabled(Number.POSITIVE_INFINITY, 1e12)).toBe(true); // Infinity = 始终可见
  });
});

describe('applyHeightReference（mock terrain 采样）', () => {
  // 赤道 lon0 表面点：surfaceRadius = 1。
  const ground = new Cartesian3(1, 0, 0);

  it('NONE / 无采样器：原样返回，不调采样', () => {
    const calls: number[] = [];
    const sampler = { sampleHeight: () => { calls.push(1); return 50; } };
    expect(applyHeightReference(ground, HeightReference.NONE, sampler)).toBe(ground);
    expect(applyHeightReference(ground, HeightReference.CLAMP_TO_GROUND, null)).toBe(ground);
    expect(calls).toHaveLength(0);
  });

  it('CLAMP_TO_GROUND：海拔 = 地形高（径向重定标），采样收到正确经纬度', () => {
    const seen: Array<[number, number]> = [];
    const sampler = {
      sampleHeight: (lon: number, lat: number) => {
        seen.push([lon, lat]);
        return 60;
      },
    };
    const out = applyHeightReference(ground, HeightReference.CLAMP_TO_GROUND, sampler);
    expect(Cartesian3.magnitude(out) * EARTH).toBeCloseTo(1 * EARTH + 60, 6);
    // 位置在 lon 0/lat 0（方向不变）。
    expect(out.y).toBeCloseTo(0, 12);
    expect(out.z).toBeCloseTo(0, 12);
    expect(seen).toHaveLength(1);
    expect(seen[0][0]).toBeCloseTo(0, 6);
    expect(seen[0][1]).toBeCloseTo(0, 6);
  });

  it('RELATIVE_TO_GROUND：海拔 = 当前海拔 + 地形高', () => {
    const aloft = new Cartesian3(1 + 120 / EARTH, 0, 0); // 120 m
    const sampler = { sampleHeight: () => 30 };
    const out = applyHeightReference(aloft, HeightReference.RELATIVE_TO_GROUND, sampler);
    expect(Cartesian3.magnitude(out) * EARTH).toBeCloseTo(EARTH + 150, 6);
  });

  it('非零经纬度位置：采样经纬度正确（武汉 lon114/lat30.5）', () => {
    const seen: Array<[number, number]> = [];
    const sampler = {
      sampleHeight: (lon: number, lat: number) => {
        seen.push([lon, lat]);
        return 0;
      },
    };
    // 压扁世界约定（与 Globe/引擎地表同系）：Camera.resolveDestination。
    const p = Camera.resolveDestination({ lon: 114, lat: 30.5, height: 0 });
    applyHeightReference(p, HeightReference.CLAMP_TO_GROUND, sampler);
    expect(seen[0][0]).toBeCloseTo(114, 4);
    expect(seen[0][1]).toBeCloseTo(30.5, 4);
  });
});

describe('Entity P2-4 字段接线', () => {
  it('heightReference / label / point / polyline 新字段可读写并触发 definitionChanged', () => {
    const e = new Entity({
      heightReference: HeightReference.CLAMP_TO_GROUND,
      point: {
        pixelSize: 6,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
        verticalOrigin: VerticalOrigin.BOTTOM,
      },
      label: {
        text: 'x',
        style: LabelStyle.FILL_AND_OUTLINE,
        outlineWidth: 3,
        distanceDisplayCondition: new DistanceDisplayCondition(0, 5000),
      },
      polyline: { positions: [new Cartesian3(1, 0, 0), new Cartesian3(0, 1, 0)], disableDepthTestDistance: 1000 },
    });
    expect(e.heightReference).toBe(HeightReference.CLAMP_TO_GROUND);
    expect(e.point?.disableDepthTestDistance).toBe(Number.POSITIVE_INFINITY);
    expect(e.label?.style).toBe(LabelStyle.FILL_AND_OUTLINE);
    expect(e.label?.distanceDisplayCondition?.isVisible(4999)).toBe(true);
    expect(e.polyline?.disableDepthTestDistance).toBe(1000);

    let raised = 0;
    e.definitionChanged.addListener(() => raised++);
    e.heightReference = HeightReference.RELATIVE_TO_GROUND;
    expect(raised).toBe(1);
    expect(e.heightReference).toBe(HeightReference.RELATIVE_TO_GROUND);
  });
});
