import { describe, it, expect } from 'vitest';
import { Cartesian3 } from '../../src/math/Cartesian3';
import { Camera } from '../../src/scene/Camera';
import { Measure } from '../../src/analysis/Measure';
import { PolylineGeometry } from '../../src/geometry/PolylineGeometry';
import { PolygonGeometry } from '../../src/geometry/PolygonGeometry';

const SCALE = 6378137.0;

/** Normalised ECEF (Earth radius = 1) from degrees. */
function fromDegrees(lonDeg: number, latDeg: number): Cartesian3 {
  const metres = Cartesian3.fromDegrees(lonDeg, latDeg);
  return new Cartesian3(
    metres.x / SCALE,
    metres.y / SCALE,
    metres.z / SCALE,
  );
}

/** Magnitude of every interleaved position vertex. */
function vertexMagnitudes(buffers: {
  vertices: Float32Array;
  vertexCount: number;
}): number[] {
  const stride = buffers.vertices.length / buffers.vertexCount;
  const mags: number[] = [];
  for (let i = 0; i < buffers.vertexCount; i++) {
    const o = i * stride;
    mags.push(Math.hypot(buffers.vertices[o], buffers.vertices[o + 1], buffers.vertices[o + 2]));
  }
  return mags;
}

describe('Measure.clampToTerrain', () => {
  it('零向量原样返回', () => {
    const out = Measure.clampToTerrain(new Cartesian3(0, 0, 0), () => 100);
    expect(out.x).toBe(0);
    expect(out.y).toBe(0);
    expect(out.z).toBe(0);
  });

  it('常量采样 h=1000m：赤道点径向半径 = 1 + h/6378137', () => {
    const p = fromDegrees(0, 0); // (1, 0, 0)
    const clamped = Measure.clampToTerrain(p, () => 1000);
    expect(clamped.x).toBeCloseTo(1 + 1000 / SCALE, 9);
    expect(clamped.y).toBeCloseTo(0, 12);
    expect(clamped.z).toBeCloseTo(0, 12);
  });

  it('clamp 幂等：贴地点再次贴地不变', () => {
    const p = fromDegrees(111, 31);
    const clamped = Measure.clampToTerrain(p, () => 500);
    const round = Measure.clampToTerrain(clamped, () => 500);
    expect(Cartesian3.distance(clamped, round) * SCALE).toBeLessThan(1e-3);
    // 方向仍指向原经纬度附近（< 0.01° 角误差）
    const dot = Cartesian3.dot(
      Cartesian3.normalize(p, new Cartesian3()),
      Cartesian3.normalize(clamped, new Cartesian3()),
    );
    expect(dot).toBeGreaterThan(Math.cos((0.01 * Math.PI) / 180));
  });

  it('sampleHeight 返回 0 → 退化为基础面（h=0）', () => {
    const p = fromDegrees(111, 31);
    const clamped = Measure.clampToTerrain(p, () => 0);
    const raised = Measure.clampToTerrain(p, () => 100);
    expect(Cartesian3.magnitude(clamped)).toBeLessThan(Cartesian3.magnitude(raised));
    expect(clamped.x).toBeCloseTo(raised.x / (1 + 100 / SCALE), 9);
  });

  it('非有限采样值（NaN）按 0 处理', () => {
    const p = fromDegrees(111, 31);
    const clamped = Measure.clampToTerrain(p, () => NaN);
    const zero = Measure.clampToTerrain(p, () => 0);
    expect(clamped.x).toBe(zero.x);
    expect(clamped.y).toBe(zero.y);
    expect(clamped.z).toBe(zero.z);
  });
});

describe('Measure.distance / area 可选 clampToTerrain 钩子', () => {
  const points = [fromDegrees(110, 30), fromDegrees(111, 30), fromDegrees(112, 31)];
  /** 假贴地：每点抬升 1e-4 归一化单位（~638 m）。 */
  const fakeClamp = (p: Cartesian3): Cartesian3 =>
    new Cartesian3(p.x * (1 + 1e-4), p.y * (1 + 1e-4), p.z * (1 + 1e-4));

  it('distance：注入钩子 === 预先贴地后计算', () => {
    const viaOptions = Measure.distance(points, false, { clampToTerrain: fakeClamp });
    expect(viaOptions).toBeCloseTo(Measure.distance(points.map(fakeClamp)), 6);
  });

  it('distance：默认行为不变（options 缺省 / 空对象）', () => {
    const base = Measure.distance(points);
    expect(Measure.distance(points, false)).toBe(base);
    expect(Measure.distance(points, false, {})).toBe(base);
    expect(Measure.distance(points, false, { clampToTerrain: fakeClamp })).not.toBe(base);
  });

  it('area：注入钩子 === 预先贴地后计算；默认行为不变', () => {
    // area 内部把顶点归一化到单位球，均匀缩放不可见——用非均匀假贴地验证
    const stretchZ = (p: Cartesian3): Cartesian3 =>
      new Cartesian3(p.x, p.y, p.z * 1.001);
    const base = Measure.area(points);
    expect(Measure.area(points)).toBe(base);
    expect(Measure.area(points, {})).toBe(base);
    const viaOptions = Measure.area(points, { clampToTerrain: stretchZ });
    expect(viaOptions).toBeCloseTo(Measure.area(points.map(stretchZ)), 3);
    expect(viaOptions).not.toBe(base);
  });
});

describe('Camera.pickGlobe 地形贴合', () => {
  it('无 globe 引用 → 保持椭球命中（回归）', () => {
    const camera = new Camera();
    camera.setView({ destination: { lon: 0, lat: 0, height: 1_000_000 } });
    const hit = camera.pickGlobe(0, 0);
    expect(hit).not.toBeNull();
    expect(Cartesian3.magnitude(hit!)).toBeCloseTo(1, 9);
  });

  it('常量采样 1000m → 命中点抬升到 1 + h/6378137', () => {
    const camera = new Camera();
    camera.setView({ destination: { lon: 0, lat: 0, height: 1_000_000 } });
    camera.globe = { terrainLayer: { sampleHeight: () => 1000 } };
    const hit = camera.pickGlobe(0, 0)!;
    // 赤道 lon=0：重构点 x = 1 + h/SCALE（y/z 为 0）
    expect(hit.x).toBeCloseTo(1 + 1000 / SCALE, 8);
    expect(Math.abs(hit.y)).toBeLessThan(1e-9);
    expect(Math.abs(hit.z)).toBeLessThan(1e-9);
  });

  it('采样 0（无地形数据）→ 保持椭球命中', () => {
    const camera = new Camera();
    camera.setView({ destination: { lon: 0, lat: 0, height: 1_000_000 } });
    camera.globe = { terrainLayer: { sampleHeight: () => 0 } };
    const hit = camera.pickGlobe(0, 0)!;
    expect(Cartesian3.magnitude(hit)).toBeCloseTo(1, 9);
  });

  it('迭代收敛：随纬度缓变的采样场下命中点稳定且高于椭球面', () => {
    const camera = new Camera();
    camera.setView({ destination: { lon: 0, lat: 0, height: 1_000_000 } });
    // 高程随采样纬度缓变：迭代中纬度微移 → 高程微移，验证 3 次内收敛
    camera.globe = {
      terrainLayer: {
        sampleHeight: (_lon: number, lat: number) => 63781.37 * (1 + lat / 90),
      },
    };
    const hit1 = camera.pickGlobe(0, 0)!;
    const hit2 = camera.pickGlobe(0, 0)!;
    expect(Number.isFinite(hit1.x)).toBe(true);
    expect(hit1.x).toBeGreaterThan(1.0);
    // pick 幂等：同屏同一像素重复拾取结果一致
    expect(hit1.x).toBeCloseTo(hit2.x, 12);
  });

  it('orbitAroundPivot 复用贴地 pick 结果（pivot 不被修改）', () => {
    const camera = new Camera();
    camera.setView({ destination: { lon: 0, lat: 0, height: 500_000 } });
    camera.globe = { terrainLayer: { sampleHeight: () => 3000 } };
    const pivot = camera.pickGlobe(0, 0)!;
    const before = pivot.clone();
    camera.orbitAroundPivot(pivot, 0.1, 0.05);
    expect(pivot.x).toBeCloseTo(before.x, 12);
    expect(pivot.y).toBeCloseTo(before.y, 12);
    expect(pivot.z).toBeCloseTo(before.z, 12);
  });
});

describe('PolylineGeometry heightReference: clampToGround', () => {
  const a = fromDegrees(110, 30);
  const b = fromDegrees(111, 30); // ~96 km 间距

  it("默认 'none'：行为不变（2 点 → 4 顶点）", () => {
    const buffers = PolylineGeometry.fromCartesianPositions([a, b]);
    expect(buffers.vertexCount).toBe(4);
  });

  it('clampToGround 缺 sampleHeight → 等效 none', () => {
    const buffers = PolylineGeometry.fromCartesianPositions([a, b], {
      heightReference: 'clampToGround',
    });
    expect(buffers.vertexCount).toBe(4);
  });

  it('clampToGround：加密（32 细分/段）且全部顶点高于 0 高基线', () => {
    const base = PolylineGeometry.fromCartesianPositions([a, b], {
      heightReference: 'clampToGround',
      sampleHeight: () => 0,
    });
    const clamped = PolylineGeometry.fromCartesianPositions([a, b], {
      heightReference: 'clampToGround',
      sampleHeight: () => 1000,
    });
    // 10° 间距 → steps = max(ceil(10/2), 32) = 32 → 32 段 × 4 顶点
    expect(clamped.vertexCount).toBe(32 * 4);
    const baseMags = vertexMagnitudes(base);
    const clampedMags = vertexMagnitudes(clamped);
    expect(Math.min(...clampedMags)).toBeGreaterThan(Math.max(...baseMags));
    // 抬升量接近 1000 m / 6378137（同纬度带上模长近似线性）
    const lift = Math.min(...clampedMags) - Math.max(...baseMags);
    expect(lift).toBeGreaterThan(0.0001);
    expect(lift).toBeLessThan((1000 / SCALE) * 1.01);
  });

  it('多段折线：顶点按段数缩放，输入数组不被修改', () => {
    const c = fromDegrees(112, 31);
    const input = [a, b, c];
    const snapshot = input.map((p) => p.clone());
    const buffers = PolylineGeometry.fromCartesianPositions(input, {
      heightReference: 'clampToGround',
      sampleHeight: () => 500,
    });
    expect(buffers.vertexCount).toBe(2 * 32 * 4);
    expect(input[0].x).toBe(snapshot[0].x);
    expect(input[2].z).toBe(snapshot[2].z);
  });
});

describe('PolygonGeometry heightReference: clampToGround', () => {
  const ring = [
    fromDegrees(110, 30),
    fromDegrees(111, 30),
    fromDegrees(111, 31),
    fromDegrees(110, 31),
  ];

  it("默认 'none'：行为不变", () => {
    const buffers = PolygonGeometry.fromCartesianPositions(ring);
    expect(buffers.vertexCount).toBe(4);
    expect(Math.max(...vertexMagnitudes(buffers))).toBeLessThan(1.001);
  });

  it('clampToGround：顶点抬升至采样高度，索引数不变', () => {
    const base = PolygonGeometry.fromCartesianPositions(ring, {
      heightReference: 'clampToGround',
      sampleHeight: () => 0,
    });
    const clamped = PolygonGeometry.fromCartesianPositions(ring, {
      heightReference: 'clampToGround',
      sampleHeight: () => 1500,
    });
    expect(clamped.indexCount).toBe(base.indexCount);
    expect(clamped.vertexCount).toBe(base.vertexCount);
    const baseMags = vertexMagnitudes(base);
    const clampedMags = vertexMagnitudes(clamped);
    expect(Math.min(...clampedMags)).toBeGreaterThan(Math.max(...baseMags));
    expect(Math.max(...baseMags)).toBeLessThan(1.001);
  });

  it('输入数组不被修改', () => {
    const input = ring.map((p) => p.clone());
    PolygonGeometry.fromCartesianPositions(input, {
      heightReference: 'clampToGround',
      sampleHeight: () => 800,
    });
    expect(input[0].x).toBeCloseTo(ring[0].x, 15);
  });
});
