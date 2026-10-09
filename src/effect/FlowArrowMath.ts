import { Cartesian3 } from '../math/Cartesian3';

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
  /** 箭头颜色（默认亮青）。 */
  arrowColor?: [number, number, number, number];
  /** 流速：周期/秒（默认 0.35，对齐 hexapod 默认观感）。 */
  speed?: number;
  /** 箭头密度：每单位弧长箭头数（默认 24，配合引擎归一化弧长 ≈ 每 266 km 一支）。 */
  gap?: number;
}

export const FLOW_ARROW_DEFAULTS = {
  color: [0.1, 0.45, 0.85, 0.35] as [number, number, number, number],
  arrowColor: [0.35, 0.95, 1.0, 0.95] as [number, number, number, number],
  speed: 0.35,
  gap: 24,
};

/**
 * 构建流动箭头线几何（CPU，重建仅当 positions 引用变化）：
 * 每段四角 [L0,R0,L1,R1]，顶点布局 [x,y,z, arcLen, v]（v ∈ {0,1} 横向）。
 * 侧向偏移 = normalize(cross(tangent, radial)) × halfWidth（PolylineGeometry
 * 同款惯例）。返回 null 当 positions 退化（<2 点）。
 */
export function buildFlowArrowGeometry(
  positions: readonly Cartesian3[],
  halfWidth: number
): { vertices: Float32Array; indices: Uint32Array; arcLength: number } | null {
  const n = positions.length;
  if (n < 2) return null;

  // 每点切向（内点取相邻段平均）与累计弧长。
  const tangents: Cartesian3[] = new Array(n);
  const arc = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const prev = positions[Math.max(0, i - 1)];
    const next = positions[Math.min(n - 1, i + 1)];
    const t = new Cartesian3(next.x - prev.x, next.y - prev.y, next.z - prev.z);
    const len = Cartesian3.magnitude(t);
    if (len < 1e-12) {
      tangents[i] = i > 0 ? tangents[i - 1] : new Cartesian3(0, 0, 1);
    } else {
      tangents[i] = new Cartesian3(t.x / len, t.y / len, t.z / len);
    }
    if (i > 0) arc[i] = arc[i - 1] + Cartesian3.distance(positions[i - 1], positions[i]);
  }

  const segmentCount = n - 1;
  const FLOATS = 5;
  const vertices = new Float32Array(segmentCount * 4 * FLOATS);
  const indices = new Uint32Array(segmentCount * 6);

  let vi = 0;
  let ii = 0;
  for (let seg = 0; seg < segmentCount; seg++) {
    const i0 = seg;
    const i1 = seg + 1;
    const base = seg * 4;
    const corners: Array<[number, number]> = []; // [顶点索引 0/1, side −1/+1]
    corners.push([i0, -1], [i0, 1], [i1, -1], [i1, 1]);
    for (let c = 0; c < 4; c++) {
      const [pi, side] = corners[c];
      const p = positions[pi];
      // radial = normalize(p)（球面外法向）；side 向量 = normalize(cross(t, radial))。
      const pr = Cartesian3.magnitude(p) || 1;
      const radial = new Cartesian3(p.x / pr, p.y / pr, p.z / pr);
      const t = tangents[pi];
      const s = new Cartesian3(
        t.y * radial.z - t.z * radial.y,
        t.z * radial.x - t.x * radial.z,
        t.x * radial.y - t.y * radial.x
      );
      const sLen = Cartesian3.magnitude(s);
      if (sLen > 1e-12) {
        const k = (side * halfWidth) / sLen;
        vertices[vi] = p.x + s.x * k;
        vertices[vi + 1] = p.y + s.y * k;
        vertices[vi + 2] = p.z + s.z * k;
      } else {
        vertices[vi] = p.x;
        vertices[vi + 1] = p.y;
        vertices[vi + 2] = p.z;
      }
      vertices[vi + 3] = arc[pi];
      vertices[vi + 4] = side > 0 ? 1 : 0;
      vi += FLOATS;
    }
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
