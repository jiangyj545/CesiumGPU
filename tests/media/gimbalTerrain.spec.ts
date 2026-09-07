import { describe, it, expect } from "vitest";
import {
  WGS84_POLAR_RATIO,
  EARTH_SCALE,
  computeFootprint,
  buildFootprintGrid,
  buildTextureMatrix,
  buildFrustumLineVertices,
  projectToSurface,
  type Footprint,
  type GimbalPose,
  type HeightSampler,
} from "../../src/media/GimbalMath";
import { Cartesian3 } from "../../src/math/Cartesian3";
import { Matrix4 } from "../../src/math/Matrix4";

const S = EARTH_SCALE;
const DEG = Math.PI / 180;

/** 归一化 ECEF 位置：经纬度（度）+ 高度（米）。 */
function normPos(lonDeg: number, latDeg: number, heightMeters: number): Cartesian3 {
  const m = Cartesian3.fromDegrees(lonDeg, latDeg, heightMeters);
  return new Cartesian3(m.x / S, m.y / S, m.z / S);
}

/** 默认位姿：(0,0) 上空 500m，nadir，60° FOV，aspect 1。 */
function makePose(opts: Partial<GimbalPose> = {}): GimbalPose {
  return {
    position: normPos(0, 0, 500),
    heading: 0,
    pitch: -Math.PI / 2,
    roll: 0,
    fovY: 60 * DEG,
    aspect: 1,
    ...opts,
  };
}

/** 手写 4x4（列主序）× 点，返回 clip 坐标（Matrix4 无点变换 API）。 */
function xformPoint(m: Matrix4, p: Cartesian3): { x: number; y: number; z: number; w: number } {
  const v = m.values;
  const x = p.x, y = p.y, z = p.z;
  return {
    x: v[0] * x + v[4] * y + v[8] * z + v[12],
    y: v[1] * x + v[5] * y + v[9] * z + v[13],
    z: v[2] * x + v[6] * y + v[10] * z + v[14],
    w: v[3] * x + v[7] * y + v[11] * z + v[15],
  };
}

/** 压缩椭球度量 |(x, y, z/rho)|：贴地点应为 1（+ 抬升量）。 */
function squashedRadius(p: Cartesian3): number {
  const zr = p.z / WGS84_POLAR_RATIO;
  return Math.sqrt(p.x * p.x + p.y * p.y + zr * zr);
}

/** 点相对基准椭球面的径向高度（米）。 */
function radialHeightMeters(p: Cartesian3): number {
  return (squashedRadius(p) - 1) * S;
}

/** 归一化约定互逆：由点反解经纬度（度）——ecefFromLonLatHeight 的精确逆
 *  （z/rho = sin(lat)、hypot(x,y) = cos(lat)），与引擎地形网格同一约定。 */
function toLonLatDeg(p: Cartesian3): { lonDeg: number; latDeg: number } {
  return {
    lonDeg: Math.atan2(p.y, p.x) / DEG,
    latDeg: Math.atan2(p.z / WGS84_POLAR_RATIO, Math.hypot(p.x, p.y)) / DEG,
  };
}

/** 解析起伏地形（米）：跨足迹（~±288m）呈 ±5m 起伏，足迹内同时含"山峰"与"谷地"。 */
const steepTerrain: HeightSampler = (lonDeg, latDeg) =>
  600 * Math.sin(lonDeg * 200 * DEG) * Math.cos(latDeg * 200 * DEG);

/** 恒定高度 sampler。 */
const H0 = 250;
const constTerrain: HeightSampler = () => H0;

/** Footprint → number[]（便于整值精确比较）。 */
function fpFloats(fp: Footprint): number[] {
  return [
    ...fp.corners.flatMap((c) => [c.x, c.y, c.z]),
    fp.center.x,
    fp.center.y,
    fp.center.z,
    fp.centerRangeMeters,
    fp.valid ? 1 : 0,
  ];
}

describe("无 sampler 回归（现行为不变）", () => {
  it("所有函数：显式传 undefined 与不传参输出完全一致", () => {
    const pose = makePose({ pitch: -Math.PI / 4 });
    expect(fpFloats(computeFootprint(pose, undefined))).toEqual(fpFloats(computeFootprint(pose)));

    const n = 4;
    const size = 3 * (n + 1) * (n + 1);
    const p1 = new Float32Array(size);
    const p2 = new Float32Array(size);
    const idx = new Uint32Array(6 * n * n);
    buildFootprintGrid(computeFootprint(pose), n, 30, p1, idx, undefined);
    buildFootprintGrid(computeFootprint(pose), n, 30, p2, idx);
    expect(Array.from(p1)).toEqual(Array.from(p2));

    const fp = computeFootprint(pose);
    const l1 = new Float32Array(48);
    const l2 = new Float32Array(48);
    expect(buildFrustumLineVertices(pose, fp, 10, l1, undefined)).toBe(
      buildFrustumLineVertices(pose, fp, 10, l2),
    );
    expect(Array.from(l1)).toEqual(Array.from(l2));

    const q = normPos(117, 31, 1000);
    const s1 = projectToSurface(q, 20, new Cartesian3(), undefined);
    const s2 = projectToSurface(q, 20, new Cartesian3());
    expect([s1.x, s1.y, s1.z]).toEqual([s2.x, s2.y, s2.z]);
  });

  it("nadir 足迹中心落在 (1,0,0)（椭球面基准锚点不变）", () => {
    const fp = computeFootprint(makePose());
    expect(fp.valid).toBe(true);
    expect(fp.center.x).toBeCloseTo(1, 9);
    expect(fp.center.y).toBeCloseTo(0, 9);
    expect(fp.center.z).toBeCloseTo(0, 9);
  });
});

describe("恒定高度 sampler", () => {
  it("computeFootprint 角点/中心径向高度 = h0（±0.1m），与基准点同向（纯径向抬升）", () => {
    const base = computeFootprint(makePose());
    const lifted = computeFootprint(makePose(), undefined, constTerrain);
    expect(lifted.valid).toBe(true);
    for (let i = 0; i < 4; i++) {
      const b = base.corners[i];
      const l = lifted.corners[i];
      // 同向：叉积长度 ~ 0（只沿地心方向缩放，未横向偏移）
      const cx = b.y * l.z - b.z * l.y;
      const cy = b.z * l.x - b.x * l.z;
      const cz = b.x * l.y - b.y * l.x;
      expect(Math.hypot(cx, cy, cz)).toBeLessThan(1e-12);
      // 相对无 sampler 沿径向抬高 h0 米（±0.1m）
      expect(Math.abs((squashedRadius(l) - squashedRadius(b)) * S - H0)).toBeLessThan(0.1);
      expect(Math.abs(radialHeightMeters(l) - H0)).toBeLessThan(0.1);
    }
    expect(Math.abs(radialHeightMeters(lifted.center) - H0)).toBeLessThan(0.1);
    // centerRangeMeters 为到抬升后地形点的斜距
    const expectedRange = Cartesian3.distance(makePose().position, lifted.center) * S;
    expect(Math.abs(lifted.centerRangeMeters - expectedRange)).toBeLessThan(0.01);
  });

  it("buildFootprintGrid 全顶点径向高度 = h0 + heightOffset（f32 精度内），offset 叠加生效", () => {
    const fp = computeFootprint(makePose(), undefined, constTerrain);
    const n = 4;
    const positions = new Float32Array(3 * (n + 1) * (n + 1));
    const { vertexCount } = buildFootprintGrid(fp, n, 40, positions, new Uint32Array(6 * n * n), constTerrain);
    expect(vertexCount).toBe(25);
    // Float32Array 存储精度 1 ULP ≈ 6e-8（相对）≈ 0.4m，±0.1m 仅对 float64
    // （computeFootprint 输出）成立；网格断言放宽到 1m（与 gimbalMath.spec
    // 网格半径 1e-6 归一化容差同量级）。
    for (let v = 0; v < vertexCount; v++) {
      const p = new Cartesian3(positions[3 * v], positions[3 * v + 1], positions[3 * v + 2]);
      expect(Math.abs(radialHeightMeters(p) - (H0 + 40))).toBeLessThan(1);
    }
  });

  it("projectToSurface：半径 = 1 + h0/S（±0.1m），采样经纬度 = p 方向的反解（径向缩放不改方向）", () => {
    const p = normPos(117, 31, 1000);
    const out = projectToSurface(p, 0, new Cartesian3(), constTerrain);
    expect(Math.abs(radialHeightMeters(out) - H0)).toBeLessThan(0.1);
    // sampler 坐标约定 = ecefFromLonLatHeight 的精确逆（引擎地形管线约定）。
    // 与 fromDegrees（大地纬度公式）构造的点存在 ~0.1° 的引擎既有约定差，
    // 故与"输入点自身的反解"比较（径向缩放保方向 → 应完全一致）。
    const inLL = toLonLatDeg(p);
    const ll = toLonLatDeg(out);
    expect(ll.lonDeg).toBeCloseTo(inLL.lonDeg, 9);
    expect(ll.latDeg).toBeCloseTo(inLL.latDeg, 9);
    expect(ll.lonDeg).toBeCloseTo(117, 3); // 经度两约定一致
    // 纬度约定差（大地 vs 球向+压缩）在 31° 处 ≈ 0.085°，引擎既有现象
    expect(Math.abs(ll.latDeg - 31)).toBeLessThan(0.2);
  });
});

describe("解析起伏 sampler（模拟 DEM）", () => {
  it("computeFootprint：山顶点高于椭球面、谷地点低于椭球面，高度差 ≈ sampler 值（±1m）", () => {
    const fp = computeFootprint(makePose(), undefined, steepTerrain);
    expect(fp.valid).toBe(true);
    const hs = fp.corners.map((c) => radialHeightMeters(c));
    expect(Math.max(...hs)).toBeGreaterThan(0); // 山顶（h>0）
    expect(Math.min(...hs)).toBeLessThan(0);    // 谷地（h<0）
    fp.corners.forEach((c, i) => {
      const { lonDeg, latDeg } = toLonLatDeg(c);
      expect(Math.abs(hs[i] - steepTerrain(lonDeg, latDeg))).toBeLessThan(1);
    });
    // 中心点同样满足（该 sampler 在 (0,0) 处 h=0 → 贴椭球面）
    const cc = toLonLatDeg(fp.center);
    expect(Math.abs(radialHeightMeters(fp.center) - steepTerrain(cc.lonDeg, cc.latDeg))).toBeLessThan(1);
  });

  it("buildFootprintGrid 逐顶点贴地形：每顶点独立采样（非角点抬升的双线性混合）", () => {
    const n = 4;
    const positions = new Float32Array(3 * (n + 1) * (n + 1));
    const flat = new Float32Array(3 * (n + 1) * (n + 1));
    const idx = new Uint32Array(6 * n * n);
    buildFootprintGrid(computeFootprint(makePose(), undefined, steepTerrain), n, 0, positions, idx, steepTerrain);
    buildFootprintGrid(computeFootprint(makePose()), n, 0, flat, idx);

    let maxDeform = 0;
    for (let v = 0; v < (n + 1) * (n + 1); v++) {
      const p = new Cartesian3(positions[3 * v], positions[3 * v + 1], positions[3 * v + 2]);
      const { lonDeg, latDeg } = toLonLatDeg(p);
      expect(Math.abs(radialHeightMeters(p) - steepTerrain(lonDeg, latDeg))).toBeLessThan(1);
      maxDeform = Math.max(maxDeform, Math.abs(radialHeightMeters(p) - radialHeightMeters(
        new Cartesian3(flat[3 * v], flat[3 * v + 1], flat[3 * v + 2]),
      )));
    }
    // 网格确实被地形改变（相邻顶点间起伏 > 1m，双线性混合无法满足逐点采样）
    expect(maxDeform).toBeGreaterThan(1);
  });

  it("buildFrustumLineVertices：地面端点抬升到地形面（f32 精度内），对未抬升足迹重采样幂等", () => {
    const pose = makePose({ pitch: -Math.PI / 4 });
    const fpLifted = computeFootprint(pose, undefined, steepTerrain);
    const out = new Float32Array(48);
    expect(buildFrustumLineVertices(pose, fpLifted, 10, out, steepTerrain)).toBe(48);
    // Float32 存储精度 1 ULP ≈ 0.4m，故 0.5m 容差（float64 语义下 <1e-6m）
    for (let i = 0; i < 4; i++) {
      // 每条线段 floats [24+i*6 .. 24+i*6+5] = [drone(3), ground(3)]
      const g = new Cartesian3(out[24 + i * 6 + 3], out[24 + i * 6 + 4], out[24 + i * 6 + 5]);
      const { lonDeg, latDeg } = toLonLatDeg(g);
      expect(Math.abs(radialHeightMeters(g) - steepTerrain(lonDeg, latDeg))).toBeLessThan(0.5);
    }

    // 幂等：projectToSurface 先归一回基准椭球再采样抬升 → 基准足迹同样重采样
    // 结果一致（f32 舍入允许 1 ULP 差）
    const outBase = new Float32Array(48);
    buildFrustumLineVertices(pose, computeFootprint(pose), 10, outBase, steepTerrain);
    for (let k = 24; k < 48; k++) {
      expect(outBase[k]).toBeCloseTo(out[k], 6);
    }
  });
});

describe("buildTextureMatrix 两模式有效性 + degraded 路径", () => {
  it("有/无 sampler 时 texMatrix 对（位移后）角点均有效：w>0、uv 有界", () => {
    // 2000m 高度 + 50m 高程：位移角点仍在视锥内。位移使 uv 随 h/斜距轻微
    // 平移——这正是地形点在视频画面中的正确投影位置（透视投影保持有效）。
    const pose = makePose({ position: normPos(0, 0, 2000) });
    const smallTerrain: HeightSampler = () => 50;
    const samplers: (HeightSampler | undefined)[] = [undefined, smallTerrain];
    for (const sampler of samplers) {
      const fp = computeFootprint(pose, undefined, sampler);
      expect(fp.valid).toBe(true);
      const tex = buildTextureMatrix(pose, 1, 100000);
      fp.corners.forEach((c) => {
        const clip = xformPoint(tex, c);
        expect(clip.w).toBeGreaterThan(0);
        const u = clip.x / clip.w;
        const v = clip.y / clip.w;
        expect(Number.isFinite(u) && Number.isFinite(v)).toBe(true);
        expect(u).toBeGreaterThan(-0.1);
        expect(u).toBeLessThan(1.1);
        expect(v).toBeGreaterThan(-0.1);
        expect(v).toBeLessThan(1.1);
      });
    }
  });

  it("射线未命中椭球：degraded 行为与无 sampler 完全一致", () => {
    // 仰视 45°：全部角射线/中心射线均高于地平（俯角 ≤ -15°），无一命中
    const pose = makePose({ pitch: Math.PI / 4 });
    const base = computeFootprint(pose);
    const lifted = computeFootprint(pose, undefined, constTerrain);
    expect(base.valid).toBe(false);
    expect(lifted.valid).toBe(false);
    expect(lifted.centerRangeMeters).toBe(0);
    // 占位角点（position + dir）不受 sampler 影响
    expect(fpFloats(lifted)).toEqual(fpFloats(base));
  });
});
