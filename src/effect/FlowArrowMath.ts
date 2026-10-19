import { Cartesian3 } from '../math/Cartesian3';
import { Measure } from '../analysis/Measure';

/**
 * FlowArrowMath – 流动箭头线的纯函数层（P2-6，对齐 hexapod flowArrowMaterial
 * 的 "渐亮尾 + 三角头" 程序化箭头）。
 *
 * 概念：折线顶点带**累计弧长** arcLen（沿 s 增大方向单调），箭头相位
 * `phase = fract(arcLen·gap − t·speed)` 在 GPU 上滚动——几何不随时间重算，
 * 箭头朝向恒指 s 增大方向（相位只依赖标量弧长，与段方向无关，天然不翻转）。
 */

/** 箭头形状常数（与 hexapod FLOW_ARROW_SHADER 同值，尖端朝 s 增大方向）。 */
export const FLOW_ARROW_SHAPE = {
  tip: 0.78,          // 尖端位置（周期内 s 坐标）
  headLen: 0.20,      // 三角头长
  tailLen: 0.48,      // 渐亮尾长
  headHalfMax: 0.32,  // 头根最大半宽（横向 t 半幅）
} as const;

/**
 * 箭头相位 ∈ [0, 1)：`fract(arcLen·gap − t·speed)`。
 * gap = 每单位弧长的箭头数；speed = 周期/秒（正值向 s 增大方向流动）。
 */
export function flowArrowPhase(
  arcLength: number,
  gap: number,
  t: number,
  speed: number
): number {
  const raw = arcLength * gap - t * speed;
  return raw - Math.floor(raw);
}

/**
 * 周期内箭头形状（对齐 hexapop shader 语义）：
 * @param sPos 周期内 s 坐标 [0,1)
 * @param tc   横向坐标 (−0.5, 0.5)，0 = 线中心
 * @returns inside ∈ {0,1} 是否在箭头形状内；fade ∈ (0,1] 尾部渐亮（头恒 1）
 */
export function flowArrowMask(sPos: number, tc: number): { inside: number; fade: number } {
  const { tip, headLen, tailLen, headHalfMax } = FLOW_ARROW_SHAPE;
  const tailHalf = headHalfMax * 0.3;
  const tailStart = tip - headLen - tailLen;

  const dHead = tip - sPos;
  const inHead = dHead >= 0 && dHead <= headLen ? 1 : 0;
  const inTail = sPos >= tailStart && sPos <= tip - headLen ? 1 : 0;

  const headHalf = headHalfMax * Math.min(Math.max(dHead / headLen, 0), 1);
  const halfW = Math.max(inHead * headHalf, inTail * tailHalf);
  const inside = (Math.abs(tc) < halfW ? 1 : 0) * Math.max(inHead, inTail);

  const dTail = sPos - tailStart;
  const tailFade = 0.15 + 0.85 * Math.min(Math.max(dTail / (tailLen * 0.75), 0), 1);
  const fade = inTail ? tailFade : 1;
  return { inside, fade };
}

/** FlowArrow 材质参数（PolylineGraphics.material，type 判别）。 */
export interface FlowArrowPolylineMaterial {
  type: 'flow';
  /** 底线颜色（半透明带）。 */
  color?: [number, number, number, number];
  /** 箭头颜色（默认青色，hexapod waypointEditor 航线）。 */
  arrowColor?: [number, number, number, number];
  /** 流速：周期/秒（hexapod duration=3.0 s → 默认 1/3）。 */
  speed?: number;
  /**
   * 箭头密度（hexapod 语义）：整条线上的箭头数（≥1）；默认自适应
   * ~每 100 m 一支（ARROW_INTERVAL=100，见 effectiveFlowArrowGap）。
   */
  repeatCount?: number;
  /**
   * 高级覆盖：每归一化弧长单位的箭头数（优先级高于 repeatCount）。
   * 引擎归一化单位 = 地球赤道半径，直接给定 gap 时注意量级（1/单位 ≈
   * 每 6378 km 一支）。
   */
  gap?: number;
}

export const FLOW_ARROW_DEFAULTS = {
  color: [0, 1, 1, 0.25] as [number, number, number, number],
  arrowColor: [0, 1, 1, 0.95] as [number, number, number, number],
  /** hexapod duration = 3.0 s → 每秒 1/3 周期。 */
  speed: 1 / 3,
};

/**
 * 解析实际生效的箭头密度 gap（每归一化弧长单位箭头数）：
 * `material.gap` 显式给定 → 原样；否则按 hexapod repeatCount 语义换算
 * `gap = repeatCount / totalArcNorm`，repeatCount 默认自适应
 * `max(1, round(线长_m / 100))`（约每 100 m 一支箭头）。
 */
export function effectiveFlowArrowGap(
  material: Partial<FlowArrowPolylineMaterial>,
  totalArcNorm: number
): number {
  if (!(totalArcNorm > 0) || !Number.isFinite(totalArcNorm)) return 1;
  if (material.gap !== undefined) return material.gap;
  const repeatCount =
    material.repeatCount ??
    Math.max(1, Math.round((totalArcNorm * Measure.EARTH_SCALE) / 100));
  return repeatCount / totalArcNorm;
}

/**
 * 构建流动箭头线几何（CPU，重建仅当 positions 引用变化）：
 * 每段独立四边形 [L0,R0,L1,R1]，顶点布局 [x,y,z, arc, v, side.x, side.y, side.z]
 * （8 floats）。位置 = **纯中心线点**（无侧向偏移）——宽度由顶点着色器按
 * 屏幕空间恒定像素展开（side 即展开方向）。side 单位向量 =
 * normalize(cross(tangent, radial))（radial = normalize(p)，PolylineGeometry
 * 同款惯例），tangent 取**本段两端点连线方向**：同一折点在相邻两段各自的
 * quad 中使用各自段的 side（CesiumJS polyline 的逐段 butt 拼接语义——
 * 拐角外侧留小缺口、拐角不膨胀；实测 CesiumJS 1.120 拐角无 miter 膨胀，
 * 直段线宽与本引擎一致）。返回 null 当 positions 退化（<2 点）。
 */
export function buildFlowArrowGeometry(
  positions: readonly Cartesian3[]
): { vertices: Float32Array; indices: Uint32Array; arcLength: number } | null {
  const n = positions.length;
  if (n < 2) return null;

  // 每段切向 + 累计弧长。
  const segTangents: Cartesian3[] = new Array(n - 1);
  const arc = new Float64Array(n);
  for (let i = 0; i < n - 1; i++) {
    const a = positions[i];
    const b = positions[i + 1];
    const t = new Cartesian3(b.x - a.x, b.y - a.y, b.z - a.z);
    const len = Cartesian3.magnitude(t);
    segTangents[i] =
      len < 1e-12
        ? new Cartesian3(0, 0, 1)
        : new Cartesian3(t.x / len, t.y / len, t.z / len);
    arc[i + 1] = arc[i] + Cartesian3.distance(a, b);
  }

  // 每段展开方向 side（本段切向求值；同一折点在相邻两段 quad 中 side 不同
  // → butt 拼接，拐角外侧小缺口，CesiumJS 同观感）。
  const segSides: Cartesian3[] = new Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    const p = positions[i];
    const pr = Cartesian3.magnitude(p) || 1;
    const radial = new Cartesian3(p.x / pr, p.y / pr, p.z / pr);
    const t = segTangents[i];
    const s = new Cartesian3(
      t.y * radial.z - t.z * radial.y,
      t.z * radial.x - t.x * radial.z,
      t.x * radial.y - t.y * radial.x
    );
    const sLen = Cartesian3.magnitude(s);
    if (sLen > 1e-12) {
      segSides[i] = new Cartesian3(s.x / sLen, s.y / sLen, s.z / sLen);
    } else {
      // 切向 ∥ 径向（径向竖线）：改用与切向正交的坐标轴再叉积。
      const ax = Math.abs(t.z) < 0.9;
      const axis = ax ? new Cartesian3(0, 0, 1) : new Cartesian3(1, 0, 0);
      const f = new Cartesian3(
        t.y * axis.z - t.z * axis.y,
        t.z * axis.x - t.x * axis.z,
        t.x * axis.y - t.y * axis.x
      );
      const fLen = Cartesian3.magnitude(f);
      segSides[i] =
        fLen > 1e-12
          ? new Cartesian3(f.x / fLen, f.y / fLen, f.z / fLen)
          : new Cartesian3(1, 0, 0);
    }
  }

  const segmentCount = n - 1;
  const FLOATS = 8;
  const vertices = new Float32Array(segmentCount * 4 * FLOATS);
  const indices = new Uint32Array(segmentCount * 6);

  let vi = 0;
  let ii = 0;
  const writeCorner = (pi: number, side: Cartesian3, v: number): void => {
    const p = positions[pi];
    vertices[vi] = p.x;
    vertices[vi + 1] = p.y;
    vertices[vi + 2] = p.z;
    vertices[vi + 3] = arc[pi];
    vertices[vi + 4] = v;
    vertices[vi + 5] = side.x;
    vertices[vi + 6] = side.y;
    vertices[vi + 7] = side.z;
    vi += FLOATS;
  };
  for (let seg = 0; seg < segmentCount; seg++) {
    const side = segSides[seg];
    const base = seg * 4;
    writeCorner(seg, side, 0); // L0
    writeCorner(seg, side, 1); // R0
    writeCorner(seg + 1, side, 0); // L1
    writeCorner(seg + 1, side, 1); // R1
    // (L0,R0,L1) + (R0,R1,L1)。
    indices[ii++] = base;
    indices[ii++] = base + 1;
    indices[ii++] = base + 2;
    indices[ii++] = base + 1;
    indices[ii++] = base + 3;
    indices[ii++] = base + 2;
  }

  return { vertices, indices, arcLength: arc[n - 1] };
}
