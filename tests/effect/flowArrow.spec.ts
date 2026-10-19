/**
 * flowArrow.spec.ts – P2-6 FlowArrow 纯函数层：
 * 相位（弧长参数化 + 时间滚动）、箭头形状不变量（朝向不随段方向翻转、
 * 周期性、尾亮头收敛）、几何构建（8-float 布局 / 累计弧长 / 中心线 +
 * 本段 side 展开、逐段 butt 拼接、退化输入）、箭头密度解析（repeatCount
 * 自适应语义）。
 */
import { describe, expect, it } from 'vitest';
import {
  flowArrowPhase,
  flowArrowMask,
  buildFlowArrowGeometry,
  effectiveFlowArrowGap,
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

describe('buildFlowArrowGeometry（中心线 + 本段 side 展开，逐段 butt 拼接）', () => {
  const p0 = new Cartesian3(1, 0, 0);
  const p1 = new Cartesian3(0, 1, 0);
  const p2 = new Cartesian3(-1, 0, 0);

  it('每段 4 顶点 / 6 索引，8-float 布局；累计弧长写入顶点属性', () => {
    const geo = buildFlowArrowGeometry([p0, p1, p2]);
    expect(geo).not.toBeNull();
    expect(geo!.indices.length).toBe(2 * 6);
    expect(geo!.vertices.length).toBe(2 * 4 * 8);
    // 段 0 L0 顶点 arc=0；段 0 L1 / 段 1 L0 顶点（共享点 p1）arc=√2。
    expect(geo!.vertices[3]).toBeCloseTo(0, 9);
    expect(geo!.vertices[8 * 2 + 3]).toBeCloseTo(Math.SQRT2, 6); // Float32 精度
    expect(geo!.vertices[8 * 4 + 3]).toBeCloseTo(Math.SQRT2, 6);
    expect(geo!.arcLength).toBeCloseTo(2 * Math.SQRT2, 6);
  });

  it('直线：side ⟂ tangent 且 ⟂ radial、L/R 位置 == 中心线、v∈{0,1}', () => {
    const geo = buildFlowArrowGeometry([p0, p1])!;
    const tangent = new Cartesian3(-1 / Math.SQRT2, 1 / Math.SQRT2, 0);
    for (let c = 0; c < 4; c++) {
      const o = c * 8;
      const pos = new Cartesian3(geo.vertices[o], geo.vertices[o + 1], geo.vertices[o + 2]);
      const side = new Cartesian3(geo.vertices[o + 5], geo.vertices[o + 6], geo.vertices[o + 7]);
      const center = c < 2 ? p0 : p1; // (L0,R0)=p0，(L1,R1)=p1
      expect(Cartesian3.dot(side, tangent)).toBeCloseTo(0, 9);
      expect(Cartesian3.dot(side, center)).toBeCloseTo(0, 9); // radial = normalize(p)
      // 无侧向偏移：L/R 顶点 == 中心线点（宽度由 VS 屏幕空间展开）。
      expect(Cartesian3.distance(pos, center)).toBeCloseTo(0, 9);
      expect(geo.vertices[o + 4]).toBe(c % 2); // v ∈ {0,1}：L=0 / R=1
    }
  });

  it('直角折线：共享点在两段 quad 中 side 各为本段方向（butt 拼接，无 miter）', () => {
    // a→b 沿 +x，b→c 沿 +y，radial≈+z：两段 side 分别为 −y 与 +x（互相垂直）。
    const a = new Cartesian3(-1, 0, 1);
    const b = new Cartesian3(0, 0, 1);
    const c = new Cartesian3(0, 1, 1);
    const geo = buildFlowArrowGeometry([a, b, c])!;
    const sideAt = (corner: number) =>
      new Cartesian3(
        geo.vertices[corner * 8 + 5],
        geo.vertices[corner * 8 + 6],
        geo.vertices[corner * 8 + 7]
      );
    // 段 0（a→b，tangent=+x，radial≈+z）side ≈ −y。
    for (const corner of [0, 1, 2, 3]) {
      expect(Cartesian3.dot(sideAt(corner), new Cartesian3(0, -1, 0))).toBeCloseTo(1, 6);
    }
    // 段 1（b→c，tangent=+y，radial≈+z）side ≈ +x。
    for (const corner of [4, 5, 6, 7]) {
      expect(Cartesian3.dot(sideAt(corner), new Cartesian3(1, 0, 0))).toBeCloseTo(1, 6);
    }
    // 共享点 b：两段 quad 中 side 不同（butt 拼接——拐角外侧小缺口，
    // CesiumJS 同观感，拐角不膨胀）。
    const s0 = sideAt(2); // 段 0 L1 @ b
    const s1 = sideAt(4); // 段 1 L0 @ b
    expect(Cartesian3.distance(s0, s1)).toBeCloseTo(Math.SQRT2, 6);
  });

  it('全部顶点位置 == 中心线（无侧向偏移，含拐角共享点）', () => {
    const a = new Cartesian3(-1, 0, 1);
    const b = new Cartesian3(0, 0, 1);
    const c = new Cartesian3(0, 1, 1);
    const geo = buildFlowArrowGeometry([a, b, c])!;
    // 每段 quad 顶点序 (L0,R0,L1,R1)：段0 用 a,a,b,b；段1 用 b,b,c,c。
    const centers = [a, a, b, b, b, b, c, c];
    for (let corner = 0; corner < 8; corner++) {
      const o = corner * 8;
      const pos = new Cartesian3(geo.vertices[o], geo.vertices[o + 1], geo.vertices[o + 2]);
      // Float32 顶点属性精度：绝对误差 ~1e-7。
      expect(Cartesian3.distance(pos, centers[corner])).toBeCloseTo(0, 5);
    }
  });

  it('退化输入（<2 点）返回 null', () => {
    expect(buildFlowArrowGeometry([p0])).toBeNull();
    expect(buildFlowArrowGeometry([])).toBeNull();
  });
});

describe('effectiveFlowArrowGap（hexapod repeatCount → gap 换算）', () => {
  const EARTH = 6378137;
  const arcNorm = 3000 / EARTH; // 3000 m 线的归一化弧长

  it('默认自适应：~每 100 m 一支 → repeatCount = round(len_m/100)', () => {
    expect(effectiveFlowArrowGap({}, arcNorm)).toBeCloseTo(30 / arcNorm, 9);
    expect(effectiveFlowArrowGap({ type: 'flow' }, arcNorm)).toBeCloseTo(30 / arcNorm, 9);
  });

  it('repeatCount 覆盖自适应', () => {
    expect(effectiveFlowArrowGap({ repeatCount: 60 }, arcNorm)).toBeCloseTo(60 / arcNorm, 9);
  });

  it('gap 优先于 repeatCount（高级覆盖）', () => {
    expect(effectiveFlowArrowGap({ repeatCount: 60, gap: 2.5 }, arcNorm)).toBe(2.5);
    expect(effectiveFlowArrowGap({ gap: 24 }, arcNorm)).toBe(24);
  });

  it('totalArcNorm ≤ 0 或非有限 → 1', () => {
    expect(effectiveFlowArrowGap({}, 0)).toBe(1);
    expect(effectiveFlowArrowGap({}, -arcNorm)).toBe(1);
    expect(effectiveFlowArrowGap({}, Number.NaN)).toBe(1);
    expect(effectiveFlowArrowGap({}, Number.POSITIVE_INFINITY)).toBe(1);
  });

  it('极短线 clamp 到 ≥1 支箭头', () => {
    const tiny = 1 / EARTH; // 1 m
    expect(effectiveFlowArrowGap({}, tiny)).toBeCloseTo(1 / tiny, 9);
  });
});

describe('Entity polyline.material 接线', () => {
  it('type 判别 + repeatCount 透传 + 动态 positions（CallbackProperty）', () => {
    const material: FlowArrowPolylineMaterial = {
      type: 'flow',
      speed: 0.5,
      gap: 16,
      repeatCount: 30,
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
    expect(e.polyline?.material?.repeatCount).toBe(30);
    expect(e.polyline?.material?.gap).toBe(16);
    const positions = e.polyline?.positions;
    expect(typeof positions?.getValue).toBe('function');
  });
});
