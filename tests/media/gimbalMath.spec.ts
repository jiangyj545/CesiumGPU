import { describe, it, expect } from "vitest";
import {
  WGS84_POLAR_RATIO,
  EARTH_SCALE,
  GLOBE_OCCLUDE_R2,
  NDC_TO_UV_Y_SIGN,
  enuBasisAt,
  rotateAroundAxis,
  gimbalBasis,
  gimbalCornerRay,
  intersectRayEllipsoid,
  computeFootprint,
  buildFootprintGrid,
  buildTextureMatrix,
  buildFrustumLineVertices,
  projectToSurface,
  type GimbalPose,
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

describe("GimbalMath 常量", () => {
  it("WGS84_POLAR_RATIO = b/a ≈ 0.9966472", () => {
    expect(WGS84_POLAR_RATIO).toBeCloseTo(0.9966472, 6);
    expect(WGS84_POLAR_RATIO).toBeCloseTo(6356752.3142 / 6378137, 12);
  });

  it("EARTH_SCALE = WGS84 赤道半径", () => {
    expect(EARTH_SCALE).toBe(6378137);
  });

  it("GLOBE_OCCLUDE_R2 = 0.998²", () => {
    expect(GLOBE_OCCLUDE_R2).toBeCloseTo(0.996004, 12);
  });

  it("NDC_TO_UV_Y_SIGN = -1（v 轴向下）", () => {
    expect(NDC_TO_UV_Y_SIGN).toBe(-1);
  });
});

describe("enuBasisAt", () => {
  it("(0,0) 处：up=+X，east=+Y，north=+Z（引擎以 Z 为极轴）", () => {
    const { east, north, up } = enuBasisAt(normPos(0, 0, 500));
    expect(up.x).toBeCloseTo(1, 9);
    expect(up.y).toBeCloseTo(0, 9);
    expect(up.z).toBeCloseTo(0, 9);
    expect(east.x).toBeCloseTo(0, 9);
    expect(east.y).toBeCloseTo(1, 9);
    expect(east.z).toBeCloseTo(0, 9);
    expect(north.x).toBeCloseTo(0, 9);
    expect(north.y).toBeCloseTo(0, 9);
    expect(north.z).toBeCloseTo(1, 9);
  });

  it("任意点三轴正交且为单位向量", () => {
    const { east, north, up } = enuBasisAt(normPos(117, 31, 300));
    expect(Cartesian3.dot(east, north)).toBeCloseTo(0, 9);
    expect(Cartesian3.dot(east, up)).toBeCloseTo(0, 9);
    expect(Cartesian3.dot(north, up)).toBeCloseTo(0, 9);
    expect(Cartesian3.magnitude(east)).toBeCloseTo(1, 9);
    expect(Cartesian3.magnitude(north)).toBeCloseTo(1, 9);
    expect(Cartesian3.magnitude(up)).toBeCloseTo(1, 9);
  });
});

describe("rotateAroundAxis", () => {
  it("UNIT_X 绕 UNIT_Z 转 90° → UNIT_Y（右手系）", () => {
    const r = rotateAroundAxis(Cartesian3.UNIT_X, Cartesian3.UNIT_Z, Math.PI / 2, new Cartesian3());
    expect(r.x).toBeCloseTo(0, 9);
    expect(r.y).toBeCloseTo(1, 9);
    expect(r.z).toBeCloseTo(0, 9);
  });

  it("result 可与 v 别名（原位旋转）", () => {
    const v = new Cartesian3(1, 0, 0);
    rotateAroundAxis(v, Cartesian3.UNIT_Z, Math.PI / 2, v);
    expect(v.x).toBeCloseTo(0, 9);
    expect(v.y).toBeCloseTo(1, 9);
  });
});

describe("intersectRayEllipsoid", () => {
  it("赤道上空 500m 垂直下视：t = 500/EARTH_SCALE，命中点 |x| ≈ 1", () => {
    const origin = normPos(0, 0, 500);
    const dir = Cartesian3.normalize(new Cartesian3(-1, 0, 0), new Cartesian3());
    const t = intersectRayEllipsoid(origin, dir);
    expect(t).not.toBeNull();
    expect(t!).toBeCloseTo(500 / S, 9);
    const hit = Cartesian3.add(origin, Cartesian3.multiplyByScalar(dir, t!, new Cartesian3()), new Cartesian3());
    expect(Math.abs(hit.x)).toBeCloseTo(1, 9);
  });

  it("水平掠射（高于地平线）：null", () => {
    const origin = normPos(0, 0, 500);
    const dir = Cartesian3.normalize(new Cartesian3(0, 0, 1), new Cartesian3());
    expect(intersectRayEllipsoid(origin, dir)).toBeNull();
  });

  it("椭球内朝外：命中远根", () => {
    const origin = new Cartesian3(0.5, 0, 0);
    const dir = Cartesian3.normalize(new Cartesian3(1, 0, 0), new Cartesian3());
    const t = intersectRayEllipsoid(origin, dir);
    expect(t).not.toBeNull();
    expect(t!).toBeCloseTo(0.5, 9);
    const hit = Cartesian3.add(origin, Cartesian3.multiplyByScalar(dir, t!, new Cartesian3()), new Cartesian3());
    expect(hit.x).toBeCloseTo(1, 9);
  });

  it("切线情形：判别式 ≈ 0，命中点在椭球面上", () => {
    // 过 (1,1,0) 方向 (-1,0,0) 的直线与单位球相切于 (0,1,0)，各量可精确表示
    const origin = new Cartesian3(1, 1, 0);
    const dir = new Cartesian3(-1, 0, 0);
    const t = intersectRayEllipsoid(origin, dir);
    expect(t).not.toBeNull();
    expect(t!).toBeCloseTo(1, 9);
    const hit = Cartesian3.add(origin, Cartesian3.multiplyByScalar(dir, t!, new Cartesian3()), new Cartesian3());
    expect(squashedRadius(hit)).toBeCloseTo(1, 9);
  });
});

describe("gimbalBasis（镜像 Camera._applyOrientation 旋转链）", () => {
  it("heading=0, pitch=0 → dir = 北（+Z），right = 东（+Y）", () => {
    const { dir, up, right } = gimbalBasis(makePose({ heading: 0, pitch: 0 }));
    expect(dir.x).toBeCloseTo(0, 9);
    expect(dir.y).toBeCloseTo(0, 9);
    expect(dir.z).toBeCloseTo(1, 9);
    expect(right.x).toBeCloseTo(0, 9);
    expect(right.y).toBeCloseTo(1, 9);
    expect(right.z).toBeCloseTo(0, 9);
    expect(up.x).toBeCloseTo(1, 9);
  });

  it("heading=π/2 → dir 转向东（+Y）", () => {
    const { dir } = gimbalBasis(makePose({ heading: Math.PI / 2, pitch: 0 }));
    expect(dir.x).toBeCloseTo(0, 9);
    expect(dir.y).toBeCloseTo(1, 9);
    expect(dir.z).toBeCloseTo(0, 9);
  });

  it("pitch=-π/2 → dir = -up（nadir）", () => {
    const pose = makePose({ pitch: -Math.PI / 2 });
    const { dir, up } = gimbalBasis(pose);
    expect(dir.x).toBeCloseTo(-1, 9);
    expect(dir.y).toBeCloseTo(0, 9);
    expect(dir.z).toBeCloseTo(0, 9);
    // 相机 up 与 dir 正交；dir 与地理 up 的点积 = -1（正对地心）
    expect(Cartesian3.dot(dir, up)).toBeCloseTo(0, 9);
    const geoUp = Cartesian3.normalize(pose.position, new Cartesian3());
    expect(Cartesian3.dot(dir, geoUp)).toBeCloseTo(-1, 9);
  });

  it("带 roll 的非平凡位姿：三轴正交、单位长度", () => {
    const basis = gimbalBasis(
      makePose({ heading: 0.7, pitch: -0.5, roll: 0.3, position: normPos(117, 31, 300) }),
    );
    expect(Cartesian3.dot(basis.dir, basis.right)).toBeCloseTo(0, 9);
    expect(Cartesian3.dot(basis.dir, basis.up)).toBeCloseTo(0, 9);
    expect(Cartesian3.dot(basis.right, basis.up)).toBeCloseTo(0, 9);
    expect(Cartesian3.magnitude(basis.dir)).toBeCloseTo(1, 9);
    expect(Cartesian3.magnitude(basis.up)).toBeCloseTo(1, 9);
    expect(Cartesian3.magnitude(basis.right)).toBeCloseTo(1, 9);
  });
});

describe("gimbalCornerRay", () => {
  it("NDC (0,0) 即视线方向；NDC 角与 pickGlobe 公式一致", () => {
    const pose = makePose({ heading: 0.4, pitch: -0.6 });
    const center = new Cartesian3();
    gimbalCornerRay(pose, 0, 0, center);
    const basis = gimbalBasis(pose);
    expect(center.x).toBeCloseTo(basis.dir.x, 9);
    expect(center.y).toBeCloseTo(basis.dir.y, 9);
    expect(center.z).toBeCloseTo(basis.dir.z, 9);

    const tanH = Math.tan(pose.fovY / 2);
    const expected = Cartesian3.normalize(
      new Cartesian3(
        basis.dir.x + 1 * tanH * pose.aspect * basis.right.x + 1 * tanH * basis.up.x,
        basis.dir.y + 1 * tanH * pose.aspect * basis.right.y + 1 * tanH * basis.up.y,
        basis.dir.z + 1 * tanH * pose.aspect * basis.right.z + 1 * tanH * basis.up.z,
      ),
      new Cartesian3(),
    );
    const corner = new Cartesian3();
    gimbalCornerRay(pose, 1, 1, corner);
    expect(corner.x).toBeCloseTo(expected.x, 9);
    expect(corner.y).toBeCloseTo(expected.y, 9);
    expect(corner.z).toBeCloseTo(expected.z, 9);
  });
});

describe("computeFootprint", () => {
  it("500m nadir、fovY=60°、aspect=1：中心≈星下点，边长≈2·500·tan30°（5%）", () => {
    const pose = makePose();
    const fp = computeFootprint(pose);
    expect(fp.valid).toBe(true);
    expect(fp.centerRangeMeters).toBeCloseTo(500, 1);

    // 足迹中心 ≈ 星下点：center 相对位置的 ENU 水平分量 ≈ 0
    const enu = enuBasisAt(pose.position);
    const d = Cartesian3.subtract(fp.center, pose.position, new Cartesian3());
    expect(Math.abs(Cartesian3.dot(d, enu.east)) * S).toBeLessThan(1);
    expect(Math.abs(Cartesian3.dot(d, enu.north)) * S).toBeLessThan(1);

    // 角点平均 ≈ 视线中心落点
    const avg = new Cartesian3();
    for (const c of fp.corners) Cartesian3.add(avg, c, avg);
    Cartesian3.multiplyByScalar(avg, 0.25, avg);
    expect(Cartesian3.distance(avg, fp.center) * S).toBeLessThan(5);

    // 角点间距（TL-TR 边）≈ 577.35m
    const side = Cartesian3.distance(fp.corners[0], fp.corners[1]) * S;
    const expected = 2 * 500 * Math.tan(Math.PI / 6);
    expect(Math.abs(side - expected) / expected).toBeLessThan(0.05);
  });

  it("result 出参复用：返回同一对象、角点对象不变", () => {
    const pose = makePose();
    const fp = computeFootprint(pose);
    const fp2 = computeFootprint(pose, fp);
    expect(fp2).toBe(fp);
    expect(fp2.corners[0]).toBe(fp.corners[0]);
    expect(fp2.valid).toBe(true);
  });

  it("pitch=-π/4、heading=0：中心沿北前移 ≈500m（15%），东西分量 ≈0", () => {
    const pose = makePose({ pitch: -Math.PI / 4 });
    const fp = computeFootprint(pose);
    expect(fp.valid).toBe(true);
    const enu = enuBasisAt(pose.position);
    const d = Cartesian3.subtract(fp.center, pose.position, new Cartesian3());
    const northM = Cartesian3.dot(d, enu.north) * S;
    const eastM = Cartesian3.dot(d, enu.east) * S;
    expect(Math.abs(northM - 500) / 500).toBeLessThan(0.15);
    expect(Math.abs(eastM)).toBeLessThan(1);
  });

  it("heading=π/2：前移方向偏东", () => {
    const pose = makePose({ pitch: -Math.PI / 4, heading: Math.PI / 2 });
    const fp = computeFootprint(pose);
    expect(fp.valid).toBe(true);
    const enu = enuBasisAt(pose.position);
    const d = Cartesian3.subtract(fp.center, pose.position, new Cartesian3());
    const northM = Cartesian3.dot(d, enu.north) * S;
    const eastM = Cartesian3.dot(d, enu.east) * S;
    expect(Math.abs(eastM - 500) / 500).toBeLessThan(0.15);
    expect(Math.abs(northM)).toBeLessThan(1);
  });

  it("pitch=0 平视：上方角射线越出地平线 → valid=false", () => {
    const fp = computeFootprint(makePose({ pitch: 0 }));
    expect(fp.valid).toBe(false);
  });
});

describe("buildFootprintGrid", () => {
  it("顶点数=(n+1)²、索引数=6n²，全部顶点满足压缩半径 ≈ 1 + heightOffset/EARTH_SCALE", () => {
    const fp = computeFootprint(makePose());
    const n = 4;
    const positions = new Float32Array(3 * (n + 1) * (n + 1));
    const indices = new Uint32Array(6 * n * n);
    const { vertexCount, indexCount } = buildFootprintGrid(fp, n, 50, positions, indices);
    expect(vertexCount).toBe(25);
    expect(indexCount).toBe(96);

    const expectedR = 1 + 50 / S;
    for (let v = 0; v < vertexCount; v++) {
      const p = new Cartesian3(positions[3 * v], positions[3 * v + 1], positions[3 * v + 2]);
      expect(Math.abs(squashedRadius(p) - expectedR)).toBeLessThan(1e-6);
    }

    // 索引引用均落在顶点范围内，首个 quad 的三角形拓扑正确
    for (let i = 0; i < indexCount; i++) {
      expect(indices[i]).toBeLessThan(vertexCount);
    }
    expect(indices[0]).toBe(0);
    expect(indices[1]).toBe(n + 1);
    expect(indices[2]).toBe(1);
  });

  it("网格四角 = 足迹角点沿径向抬升", () => {
    const fp = computeFootprint(makePose());
    const n = 4;
    const lift = 50 / S;
    const positions = new Float32Array(3 * (n + 1) * (n + 1));
    buildFootprintGrid(fp, n, 50, positions, new Uint32Array(6 * n * n));
    const gridIndex = (i: number, j: number) => (j * (n + 1) + i) * 3;
    const corners = [
      { idx: gridIndex(0, 0), c: fp.corners[0] }, // TL
      { idx: gridIndex(n, 0), c: fp.corners[1] }, // TR
      { idx: gridIndex(n, n), c: fp.corners[2] }, // BR
      { idx: gridIndex(0, n), c: fp.corners[3] }, // BL
    ];
    for (const { idx, c } of corners) {
      expect(positions[idx]).toBeCloseTo(c.x * (1 + lift), 5);
      expect(positions[idx + 1]).toBeCloseTo(c.y * (1 + lift), 5);
      expect(positions[idx + 2]).toBeCloseTo(c.z * (1 + lift), 5);
    }
  });

  it("fp.valid=false：返回 {0,0} 且不写出", () => {
    const fp = computeFootprint(makePose({ pitch: 0 })); // 平视 → invalid
    const positions = new Float32Array(75);
    const indices = new Uint32Array(96);
    const r = buildFootprintGrid(fp, 4, 0, positions, indices);
    expect(r.vertexCount).toBe(0);
    expect(r.indexCount).toBe(0);
    expect(positions.every((v) => v === 0)).toBe(true);
  });
});

describe("buildTextureMatrix", () => {
  // 角点顺序 TL/TR/BR/BL ↔ 期望 UV（NDC_TO_UV_Y_SIGN=-1 翻转 v 轴）
  const uvExpected: Array<[number, number]> = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ];

  it("nadir 足迹 4 角投影到 (0,0),(1,0),(1,1),(0,1)，w>0 且对称相等", () => {
    const pose = makePose();
    const fp = computeFootprint(pose);
    const tex = buildTextureMatrix(pose, 1, 100000);
    const ws: number[] = [];
    fp.corners.forEach((c, i) => {
      const clip = xformPoint(tex, c);
      expect(clip.w).toBeGreaterThan(0);
      ws.push(clip.w);
      expect(clip.x / clip.w).toBeCloseTo(uvExpected[i][0], 2);
      expect(clip.y / clip.w).toBeCloseTo(uvExpected[i][1], 2);
    });
    // nadir 对称：4 角斜距一致 → w 一致（1%）
    for (const w of ws) expect(Math.abs(w - ws[0]) / ws[0]).toBeLessThan(0.01);
  });

  it("倾斜位姿（pitch=-π/4, heading=0.3）同样精确映射", () => {
    const pose = makePose({ pitch: -Math.PI / 4, heading: 0.3 });
    const fp = computeFootprint(pose);
    expect(fp.valid).toBe(true);
    const tex = buildTextureMatrix(pose, 1, 100000);
    fp.corners.forEach((c, i) => {
      const clip = xformPoint(tex, c);
      expect(clip.w).toBeGreaterThan(0);
      expect(clip.x / clip.w).toBeCloseTo(uvExpected[i][0], 2);
      expect(clip.y / clip.w).toBeCloseTo(uvExpected[i][1], 2);
    });
  });

  it("result 出参复用", () => {
    const pose = makePose();
    const m = new Matrix4();
    const out = buildTextureMatrix(pose, 1, 100000, m);
    expect(out).toBe(m);
  });
});

describe("buildFrustumLineVertices", () => {
  it("valid：近平面 4 边 + 位置到 4 地面角连线 = 48 floats", () => {
    const pose = makePose({ pitch: -Math.PI / 4 });
    const fp = computeFootprint(pose);
    const out = new Float32Array(48);
    const count = buildFrustumLineVertices(pose, fp, 10, out);
    expect(count).toBe(48);
    // 第 5 条线段起点 = 无人机位置（floats 24..26）
    expect(out[24]).toBeCloseTo(pose.position.x, 6);
    expect(out[25]).toBeCloseTo(pose.position.y, 6);
    expect(out[26]).toBeCloseTo(pose.position.z, 6);
    expect(Number.isFinite(out[0]) && Number.isFinite(out[23])).toBe(true);
  });

  it("invalid：只画近平面部分 = 24 floats", () => {
    const pose = makePose({ pitch: 0 });
    const fp = computeFootprint(pose);
    const out = new Float32Array(48);
    const count = buildFrustumLineVertices(pose, fp, 10, out);
    expect(count).toBe(24);
    expect(out.slice(24).every((v) => v === 0)).toBe(true);
  });
});

describe("projectToSurface", () => {
  it("椭球外一点径向回投后压缩半径 ≈ 1", () => {
    const p = normPos(117, 31, 1000);
    expect(squashedRadius(p)).toBeGreaterThan(1); // 先确认在椭球外
    const out = projectToSurface(p, 0, new Cartesian3());
    expect(squashedRadius(out)).toBeCloseTo(1, 9);
  });

  it("带 heightOffsetMeters 时半径 ≈ 1 + offset/EARTH_SCALE", () => {
    const p = normPos(0, 0, 500);
    const out = projectToSurface(p, 200, new Cartesian3());
    expect(squashedRadius(out)).toBeCloseTo(1 + 200 / S, 9);
  });
});
