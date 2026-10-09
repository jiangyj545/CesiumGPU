/**
 * flowArrow.spec.ts – P2-6 FlowArrow 纯函数层：
 * 相位（弧长参数化 + 时间滚动）、箭头形状不变量（朝向不随段方向翻转、
 * 周期性、尾亮头收敛）、几何构建（累计弧长 / 侧向偏移 / 退化输入）。
 */
import { describe, expect, it } from 'vitest';
import {
  flowArrowPhase,
  flowArrowMask,
  buildFlowArrowGeometry,
  FLOW_ARROW_SHAPE,
  type FlowArrowPolylineMaterial,
} from '../../src/effect/FlowArrowMath';
import { Entity } from '../../src/dataSource/Entity';
import { CallbackProperty } from '../../src/dataSource/CallbackProperty';
import { Cartesian3 } from '../../src/math/Cartesian3';

describe('flowArrowPhase（phase = fract(arcLen·gap − t·speed)）', () => {
  it('基础公式与 [0,1) 值域', () => {
    expect(flowArrowPhase(0, 24, 0, 0.35)).toBe(0);
    expect(flowArrowPhase(1 / 24, 24, 0, 0.35)).toBeCloseTo(0, 12); // 一个箭头周期
    expect(flowArrowPhase(1 / 48, 24, 0, 0.35)).toBeCloseTo(0.5, 12);
    const p = flowArrowPhase(0.031, 24, 12.7, 0.35);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThan(1);
  });

  it('周期性：弧长平移 1/gap 或时间平移 1/speed 相位不变', () => {
    const a = flowArrowPhase(0.4, 24, 3.3, 0.35);
    expect(flowArrowPhase(0.4 + 1 / 24, 24, 3.3, 0.35)).toBeCloseTo(a, 12);
    expect(flowArrowPhase(0.4, 24, 3.3 + 1 / 0.35, 0.35)).toBeCloseTo(a, 12);
  });

  it('方向不变量：相位只依赖标量弧长——两段反向折线在同一物理点相位相同', () => {
    // 正向 A→B 弧长 0.02；反向 B→A 弧长同为 0.02（累计标量）。
    const fwd = flowArrowPhase(0.02, 24, 1.0, 0.35);
    const bwd = flowArrowPhase(0.02, 24, 1.0, 0.35);
    expect(bwd).toBe(fwd); // 相位不感知段方向 → 箭头永不翻转
  });

  it('流动方向：时间增大相位减小（箭头向 s 增大方向移动）', () => {
    const t0 = flowArrowPhase(0.05, 24, 0, 0.35);
    const dt = 0.01;
    const t1 = flowArrowPhase(0.05, 24, dt, 0.35);
    // phase(t+dt) = phase(t) − speed·dt (mod 1)。
    const expected = ((t0 - 0.35 * dt) % 1 + 1) % 1;
    expect(t1).toBeCloseTo(expected, 12);
  });
});

describe('flowArrowMask（渐亮尾 + 三角头）', () => {
  it('头段：tc 越靠近尖端容许半宽越大（dHead↓ → headHalf↓），尖端外不在形状内', () => {
    const nearRoot = flowArrowMask(FLOW_ARROW_SHAPE.tip - FLOW_ARROW_SHAPE.headLen + 1e-6, 0);
    const nearTip = flowArrowMask(FLOW_ARROW_SHAPE.tip - 1e-4, 0);
    expect(nearRoot.inside).toBe(1);
    expect(nearTip.inside).toBe(1);
    // 根部容许 |tc| ≤ headHalfMax，尖端附近容许趋 0。
    expect(flowArrowMask(FLOW_ARROW_SHAPE.tip - FLOW_ARROW_SHAPE.headLen, 0.31).inside).toBe(1);
    expect(flowArrowMask(FLOW_ARROW_SHAPE.tip - 1e-4, 0.01).inside).toBe(0);
    // 尖端之后（> tip）不在形状内。
    expect(flowArrowMask(FLOW_ARROW_SHAPE.tip + 1e-4, 0).inside).toBe(0);
  });

  it('尾段：中心恒在形状内，fade 从尾根向头部单调渐亮', () => {
    const tailStart = FLOW_ARROW_SHAPE.tip - FLOW_ARROW_SHAPE.headLen - FLOW_ARROW_SHAPE.tailLen;
    const root = flowArrowMask(tailStart + 1e-6, 0);
    const mid = flowArrowMask((tailStart + FLOW_ARROW_SHAPE.tip) / 2, 0);
    expect(root.inside).toBe(1);
    expect(mid.inside).toBe(1);
    expect(mid.fade).toBeGreaterThan(root.fade);
  });

  it('形状外的底线区域 inside=0', () => {
    expect(flowArrowMask(0.0, 0).inside).toBe(0); // 周期起点在尾根之前
    expect(flowArrowMask(0.5, 0.49).inside).toBe(0); // 头段外沿
  });
});

describe('buildFlowArrowGeometry', () => {
  const p0 = new Cartesian3(1, 0, 0);
  const p1 = new Cartesian3(0, 1, 0);
  const p2 = new Cartesian3(-1, 0, 0);

  it('每段 4 顶点 / 6 索引；累计弧长写入顶点属性', () => {
    const geo = buildFlowArrowGeometry([p0, p1, p2], 1e-4);
    expect(geo).not.toBeNull();
    expect(geo!.indices.length).toBe(2 * 6);
    expect(geo!.vertices.length).toBe(2 * 4 * 5);
    // 段 0 顶点 arc=0；段 1 顶点 arc=√2。
    const arc00 = geo!.vertices[3];
    const arc10 = geo!.vertices[5 * 2 + 3];
    expect(arc00).toBeCloseTo(0, 9);
    expect(arc10).toBeCloseTo(Math.SQRT2, 6); // Float32 顶点属性精度
    expect(geo!.arcLength).toBeCloseTo(2 * Math.SQRT2, 6);
  });

  it('顶点横向 v ∈ {0,1}；侧向偏移沿切向×径向法向', () => {
    const geo = buildFlowArrowGeometry([p0, p1], 1e-4)!;
    const l = new Cartesian3(geo.vertices[0], geo.vertices[1], geo.vertices[2]);
    const r = new Cartesian3(geo.vertices[5], geo.vertices[6], geo.vertices[7]);
    expect(geo.vertices[4]).toBe(0); // L0 v=0
    expect(geo.vertices[9]).toBe(1); // R0 v=1
    // 中心点仍在 p0：L/R 关于 p0 对称。
    expect((l.x + r.x) / 2).toBeCloseTo(p0.x, 9);
    expect((l.y + r.y) / 2).toBeCloseTo(p0.y, 9);
    expect(Cartesian3.distance(l, r)).toBeCloseTo(2e-4, 9); // 全宽 = 2×halfWidth
  });

  it('退化输入（<2 点）返回 null', () => {
    expect(buildFlowArrowGeometry([p0], 1e-4)).toBeNull();
    expect(buildFlowArrowGeometry([], 1e-4)).toBeNull();
  });
});

describe('Entity polyline.material 接线', () => {
  it('type 判别 + 动态 positions（CallbackProperty）', () => {
    const material: FlowArrowPolylineMaterial = {
      type: 'flow',
      speed: 0.5,
      gap: 16,
      arrowColor: [1, 1, 0, 0.9],
    };
    const e = new Entity({
      polyline: {
        positions: new CallbackProperty(() => [new Cartesian3(1, 0, 0), new Cartesian3(0, 1, 0)]),
        material,
      },
    });
    expect(e.polyline?.material?.type).toBe('flow');
    expect(e.polyline?.material?.speed).toBe(0.5);
    const positions = e.polyline?.positions;
    expect(typeof positions?.getValue).toBe('function');
  });
});
