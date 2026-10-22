import { describe, it, expect, vi, beforeAll } from 'vitest';
import { Cartesian3 } from '../../src/math/Cartesian3';
import { Camera } from '../../src/scene/Camera';
import { Measure } from '../../src/analysis/Measure';
import { TerrainLayer } from '../../src/terrain/TerrainLayer';
import { PolylineGeometry } from '../../src/geometry/PolylineGeometry';
import { PolygonGeometry } from '../../src/geometry/PolygonGeometry';
import type { TerrainProvider, TerrainMeshData } from '../../src/terrain/TerrainProvider';
import {
  geographicTileBounds,
  ecefFromLonLatHeight,
} from '../../src/terrain/TerrainProvider';

/**
 * 地形贴地消费端集成测试（clamp 链路贯通）。
 *
 * 真实链路：TerrainLayer.sampleHeight（三角形重心插值，读 LRU 网格）
 *   → Measure.clampToTerrain（ECEF ↔ 经纬度往返 + 高度折入模长）
 *   → PolylineGeometry / PolygonGeometry 的 heightReference:'clampToGround'
 *     （经 options.sampleHeight 回调注入，几何模块不直接依赖 terrain 层）。
 *
 * 与 measureTerrain.spec.ts 的区别：那边用**合成常量采样回调**验证贴地数学；
 * 这里构造**解析高度场的真实 TerrainLayer 网格**（stitching.spec.ts 的
 * ScriptedProvider + gridMesh 手法），验证整条链路对同一份网格数据自洽。
 *
 * 期望值构造遵循「实现自洽」原则：sampleHeight 是网格三角形上的线性
 * （重心）插值，因此期望值 = 同一网格离散化后的线性插值（非解析函数本身）。
 *
 * 边缘节点历史缺陷（sampleHeight 返回 0）已由 SAMPLE_BARYCENTRIC_EPSILON
 * 修复，见「瓦片边缘网格节点」用例注释。
 */

/** WGS84 polar / equatorial ratio（与引擎约定一致）。 */
const POLAR_RATIO = 6356752.3142 / 6378137.0;
/** WGS84 semi-major axis（metres）。 */
const SEMI_MAJOR = 6378137.0;
const DEG2RAD = Math.PI / 180.0;
const RAD2DEG = 180.0 / Math.PI;

// ── 合成场景：单个 z8 geographic 瓦片，65×65 规则网格 ────────────────────────

const TILE = { z: 8, x: 256, y: 128 }; // bounds = [0, 0.703125]° × [0, 0.703125]°
const N = 65; // 65×65 顶点网格（量化网格标准分辨率）
const bounds = geographicTileBounds(TILE.z, TILE.x, TILE.y);

/** 网格第 ix 列 / 第 iy 行的经纬度（与 gridMesh 顶点公式一致）。 */
const lonAt = (ix: number): number => bounds.west + (ix / (N - 1)) * (bounds.east - bounds.west);
const latAt = (iy: number): number => bounds.south + (iy / (N - 1)) * (bounds.north - bounds.south);

/**
 * 解析高度场：500 ± 300 m，波长与瓦片跨度同量级（梯度峰值 ~2600 m/°），
 * 足以让任何网格索引 off-by-one 产生 ~30 m 级误差而被断言捕获。
 */
function heightFn(lon: number, lat: number): number {
  return 500 + 300 * Math.sin(lon * DEG2RAD * 500) * Math.cos(lat * DEG2RAD * 450);
}

/** 按键返回预置合成网格的桩 provider（同 stitching.spec.ts 手法）。 */
class ScriptedProvider implements TerrainProvider {
  readonly ready = true;
  readonly maximumLevel = 14;
  readonly tilingScheme = 'geographic' as const;
  readonly meshes = new Map<string, TerrainMeshData>();

  async requestTileGeometry(z: number, x: number, y: number): Promise<TerrainMeshData | null> {
    return this.meshes.get(`${z}/${x}/${y}`) ?? null;
  }

  getTileDataAvailable(z: number, x: number, y: number): boolean {
    return this.meshes.has(`${z}/${x}/${y}`);
  }

  destroy(): void { /* nothing */ }
}

/**
 * n×n 规则网格 mesh（无裙边）：顶点经纬度均匀覆盖瓦片，高度由
 * heightFn(lonDeg, latDeg) 给出，位置按引擎约定（归一化 ECEF，高度折入模长）。
 * 三角化：每个 quad 沿反对角线 ((iy,ix+1)–(iy+1,ix)) 一分为二。
 */
function gridMesh(
  z: number,
  x: number,
  y: number,
  height: (lon: number, lat: number) => number,
  n: number
): TerrainMeshData {
  const b = geographicTileBounds(z, x, y);
  const positions = new Float32Array(n * n * 3);
  const tmp: [number, number, number] = [0, 0, 0];
  let minH = Infinity;
  let maxH = -Infinity;
  for (let iy = 0; iy < n; iy++) {
    for (let ix = 0; ix < n; ix++) {
      const lon = b.west + (ix / (n - 1)) * (b.east - b.west);
      const lat = b.south + (iy / (n - 1)) * (b.north - b.south);
      const h = height(lon, lat);
      ecefFromLonLatHeight(lon, lat, h, tmp);
      const i = (iy * n + ix) * 3;
      positions[i] = tmp[0];
      positions[i + 1] = tmp[1];
      positions[i + 2] = tmp[2];
      minH = Math.min(minH, h);
      maxH = Math.max(maxH, h);
    }
  }
  const indices = new Uint32Array((n - 1) * (n - 1) * 6);
  let o = 0;
  for (let iy = 0; iy < n - 1; iy++) {
    for (let ix = 0; ix < n - 1; ix++) {
      const r0 = iy * n;
      const r1 = (iy + 1) * n;
      indices[o++] = r0 + ix;
      indices[o++] = r1 + ix;
      indices[o++] = r0 + ix + 1;
      indices[o++] = r0 + ix + 1;
      indices[o++] = r1 + ix;
      indices[o++] = r1 + ix + 1;
    }
  }
  return { positions, indices, minHeight: minH, maxHeight: maxH, center: [0, 0, 0] };
}

/** 从归一化 ECEF 位置反推高度（米）——真椭球约定（ecefFromLonLatHeight，
 * 2026-10-22）：hypot(x, y, z/(1−e²)) = n·scale。 */
function heightOf(p: Cartesian3): number {
  const x = p.x;
  const y = p.y;
  const z = p.z;
  const e2 = 1 - POLAR_RATIO * POLAR_RATIO;
  const lat = Math.atan2(z / (1 - e2), Math.hypot(x, y));
  const n = 1 / Math.sqrt(1 - e2 * Math.sin(lat) * Math.sin(lat));
  return (Math.hypot(x, y, z / (1 - e2)) / n - 1) * SEMI_MAJOR;
}

/** 从交错顶点缓冲的一个顶点反推 (lonDeg, latDeg)（真椭球面约定：
 * tanφ = z/((1−e²)·hypot(x,y))，n 消去闭式）。 */
function lonLatOfVertex(v: Float32Array, base: number, stride: number): [number, number] {
  const x = v[base];
  const y = v[base + 1];
  const z = v[base + 2];
  const e2 = 1 - POLAR_RATIO * POLAR_RATIO;
  return [Math.atan2(y, x) * RAD2DEG, Math.atan2(z / (1 - e2), Math.hypot(x, y)) * RAD2DEG];
}

/** 地表点（h = 0）的归一化 ECEF。 */
function surfacePoint(lon: number, lat: number): Cartesian3 {
  const tmp: [number, number, number] = [0, 0, 0];
  ecefFromLonLatHeight(lon, lat, 0, tmp);
  return new Cartesian3(tmp[0], tmp[1], tmp[2]);
}

/**
 * sampleHeight 的期望值：与实现相同的网格离散化 + 同一三角化反对角线下的
 * 线性（重心）插值——独立推导的参数化公式，测「实现自洽」。
 * 网格内任何索引 off-by-one 会产生 ~一个格距的高度摆动（~30 m）而被捕获。
 */
function expectedSampledHeight(lon: number, lat: number): number {
  const spanLon = bounds.east - bounds.west;
  const spanLat = bounds.north - bounds.south;

  const fx = ((lon - bounds.west) / spanLon) * (N - 1);
  const fy = ((lat - bounds.south) / spanLat) * (N - 1);
  const ix = Math.min(Math.max(Math.floor(fx), 0), N - 2);
  const iy = Math.min(Math.max(Math.floor(fy), 0), N - 2);
  const u = fx - ix; // 东西向格内参数
  const v = fy - iy; // 南北向格内参数

  const h00 = heightFn(lonAt(ix), latAt(iy));         // (iy, ix)
  const h10 = heightFn(lonAt(ix), latAt(iy + 1));     // (iy+1, ix)
  const h01 = heightFn(lonAt(ix + 1), latAt(iy));     // (iy, ix+1)
  const h11 = heightFn(lonAt(ix + 1), latAt(iy + 1)); // (iy+1, ix+1)

  if (u + v <= 1) {
    // 三角形 [(iy,ix), (iy+1,ix), (iy,ix+1)]：NW 权 v，SE 权 u
    return h00 + u * (h01 - h00) + v * (h10 - h00);
  }
  // 三角形 [(iy,ix+1)SE, (iy+1,ix)NW, (iy+1,ix+1)NE]：SE 权 1-v，NW 权 1-u
  return h11 + (1 - v) * (h01 - h11) + (1 - u) * (h10 - h11);
}

/** 可复现的均匀随机数（mulberry32，固定种子）。 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 建层并经公开链路（provider 请求 → update 选中 → LRU 入缓存）装载网格。 */
async function makeLayer(): Promise<TerrainLayer> {
  const provider = new ScriptedProvider();
  provider.meshes.set(`${TILE.z}/${TILE.x}/${TILE.y}`, gridMesh(TILE.z, TILE.x, TILE.y, heightFn, N));

  // maximumLevel 钉在瓦片层级：相机足够近时 SSE 一路细化到 z8 停止
  const layer = new TerrainLayer({ provider, maximumLevel: TILE.z });
  const camera = new Camera();
  camera.setView({
    destination: {
      lon: (bounds.west + bounds.east) / 2,
      lat: (bounds.south + bounds.north) / 2,
      height: 0.015 * SEMI_MAJOR, // ~96 km，与 stitching.spec 的停止高度同量级
    },
  });
  await vi.waitFor(() => {
    layer.update(camera, 1080);
    expect(layer.getMesh(TILE.z, TILE.x, TILE.y)).not.toBeNull();
  });
  return layer;
}

/** TerrainLayer → 采样回调（消费端注入的标准形态）。 */
const sampler = (lon: number, lat: number): number => layer.sampleHeight(lon, lat);

let layer: TerrainLayer;
beforeAll(async () => {
  layer = await makeLayer();
});

// ── 1. sampleHeight 精度 ─────────────────────────────────────────────────────

describe('TerrainLayer.sampleHeight 对解析高度场的自洽精度', () => {
  it('内部网格节点返回解析高度（±0.8 m，float32 位置量化界）', () => {
    // 顶点位置经 float32 存储，高度反解量化误差上限 ~0.66 m（实测 0.38 m）；
    // 索引 off-by-one 会产生 ~30 m 级误差，1 m 内即证明网格索引正确。
    // 瓦片边缘节点当前有已知缺陷（见下方 it.fails），此处只扫内部。
    let maxErr = 0;
    for (let iy = 1; iy < N - 1; iy++) {
      for (let ix = 1; ix < N - 1; ix++) {
        maxErr = Math.max(
          maxErr,
          Math.abs(layer.sampleHeight(lonAt(ix), latAt(iy)) - heightFn(lonAt(ix), latAt(iy))),
        );
      }
    }
    expect(maxErr).toBeLessThan(0.8);
  });

  it('瓦片边缘网格节点应返回解析高度（已修复：SAMPLE_BARYCENTRIC_EPSILON 吸收 f32 顶点抖动）', () => {
    // 历史 bug：闭合瓦片语义下包含检查把 lon == east / lat == north 视为
    // 覆盖内，但顶点经纬度由 float32 位置反解（抖动 ~4.5e-8°，折合格距的
    // ~4e-6），严格重心容差 -1e-6 吸收不了 → 边缘三角形全部拒绝，
    // sampleHeight 返回 0。消费端把 0 当「无地形数据」贴基础面，误差达
    // 数百米。实测（65×65 单瓦片）：east 列 19 / north 行 24 / west 列 1 个
    // 节点返回 0。修复：TerrainLayer._interpolateHeight 传入
    // SAMPLE_BARYCENTRIC_EPSILON(-2e-3)（与缝合通道 STITCH_BARYCENTRIC_
    // EPSILON 同源同解），边缘节点全部命中、插值无副作用。
    const check = (lon: number, lat: number): void => {
      expect(Math.abs(layer.sampleHeight(lon, lat) - heightFn(lon, lat))).toBeLessThan(0.8);
    };
    for (let iy = 0; iy < N; iy++) {
      check(bounds.west, latAt(iy)); // 西列
      check(bounds.east, latAt(iy)); // 东列
      check(lonAt(iy), bounds.south); // 南行
      check(lonAt(iy), bounds.north); // 北行
    }
  });

  it('网格内 200 随机点：与同网格离散化线性插值期望误差 < 1 m', () => {
    const rand = mulberry32(0x5eed1234);
    let maxErr = 0;
    let hMin = Infinity;
    let hMax = -Infinity;
    for (let i = 0; i < 200; i++) {
      const lon = bounds.west + rand() * (bounds.east - bounds.west);
      const lat = bounds.south + rand() * (bounds.north - bounds.south);
      const got = layer.sampleHeight(lon, lat);
      const expected = expectedSampledHeight(lon, lat);
      maxErr = Math.max(maxErr, Math.abs(got - expected));
      hMin = Math.min(hMin, got);
      hMax = Math.max(hMax, got);
    }
    // 高度场在瓦片内摆动数百米，证明采样真的跟随了地形起伏（非退化恒值）
    expect(hMax - hMin).toBeGreaterThan(100);
    expect(maxErr).toBeLessThan(1.0);
  });

  it('瓦片边界外（未加载区）返回 0', () => {
    // 边界上（闭合瓦片语义）的可采样性由上面 it.fails 用例统一覆盖；
    // 此处断言边界外一格 → 无覆盖瓦片 → 0（消费端按「贴基础面」处理）
    const midLon = (bounds.west + bounds.east) / 2;
    const midLat = (bounds.south + bounds.north) / 2;
    expect(layer.sampleHeight(bounds.west - 0.01, midLat)).toBe(0);
    expect(layer.sampleHeight(bounds.east + 0.01, midLat)).toBe(0);
    expect(layer.sampleHeight(midLon, bounds.south - 0.01)).toBe(0);
    expect(layer.sampleHeight(midLon, bounds.north + 0.01)).toBe(0);
  });

  it('level 参数限制搜索层级；destroy 后一律返回 0', () => {
    const midLon = (bounds.west + bounds.east) / 2;
    const midLat = (bounds.south + bounds.north) / 2;

    // 网格在 z8：限制到 z7 以下时视为未加载
    expect(layer.sampleHeight(midLon, midLat, TILE.z - 1)).toBe(0);
    expect(layer.sampleHeight(midLon, midLat, TILE.z)).toBeGreaterThan(0);

    // destroy 语义：独立小层直接注入私有缓存验证（同步、不依赖相机链路）
    const provider = new ScriptedProvider();
    provider.meshes.set(
      `${TILE.z}/${TILE.x}/${TILE.y}`,
      gridMesh(TILE.z, TILE.x, TILE.y, heightFn, N),
    );
    const l2 = new TerrainLayer({ provider, maximumLevel: TILE.z });
    (l2 as unknown as { _putMesh: (z: number, x: number, y: number, m: TerrainMeshData) => void })
      ._putMesh(TILE.z, TILE.x, TILE.y, provider.meshes.get(`${TILE.z}/${TILE.x}/${TILE.y}`)!);
    expect(l2.sampleHeight(midLon, midLat)).toBeGreaterThan(0);
    l2.destroy();
    expect(l2.sampleHeight(midLon, midLat)).toBe(0);
  });
});

// ── 2. Measure.clampToTerrain ↔ TerrainLayer 链 ─────────────────────────────

describe('Measure.clampToTerrain 接真实 TerrainLayer', () => {
  it('多点 clamp 后高度 == sampleHeight(同经纬度)，方向不变', () => {
    const rand = mulberry32(0xc1a3b007);
    for (let i = 0; i < 50; i++) {
      const lon = bounds.west + rand() * (bounds.east - bounds.west);
      const lat = bounds.south + rand() * (bounds.north - bounds.south);
      // 从 20 km 高空下贴，路径必须穿过 sampleHeight 的网格插值
      const tmp: [number, number, number] = [0, 0, 0];
      ecefFromLonLatHeight(lon, lat, 20000, tmp);
      const above = new Cartesian3(tmp[0], tmp[1], tmp[2]);

      const clamped = Measure.clampToTerrain(above, sampler);

      // 高度精确回到该点地形采样值（全程 f64，误差应在毫米级以下）
      expect(Math.abs(heightOf(clamped) - layer.sampleHeight(lon, lat))).toBeLessThan(1e-3);
      // 贴地不改变经纬度方向（< 1e-6°）
      const [lonC, latC] = lonLatOfVertex(
        new Float32Array([clamped.x, clamped.y, clamped.z]), 0, 3,
      );
      expect(Math.abs(lonC - lon)).toBeLessThan(1e-6);
      expect(Math.abs(latC - lat)).toBeLessThan(1e-6);
    }
  });

  it('瓦片外点退化为贴基础面（引擎 h=0 参数化面）；clamp 幂等', () => {
    // 瓦片东侧 0.01° 外：sampleHeight → 0 → 贴「基础面」。
    // 注意引擎基础面 = 参数化球面 y/z 压缩（与地形网格 h=0 顶点完全一致），
    // 幅值 ≠ 1（本纬度带 ~1 - 6.4e-7），故与 ecefFromLonLatHeight(lon,lat,0) 对比。
    const outLon = bounds.east + 0.01;
    const midLat = (bounds.south + bounds.north) / 2;
    const clampedOut = Measure.clampToTerrain(surfacePoint(outLon, midLat), sampler);
    const ref: [number, number, number] = [0, 0, 0];
    ecefFromLonLatHeight(outLon, midLat, 0, ref);
    expect(clampedOut.x).toBeCloseTo(ref[0], 12);
    expect(clampedOut.y).toBeCloseTo(ref[1], 12);
    expect(clampedOut.z).toBeCloseTo(ref[2], 12);

    // 幂等：贴地点再贴一次不变（毫米级）
    const midLon = (bounds.west + bounds.east) / 2;
    const inside = surfacePoint(midLon, midLat);
    const once = Measure.clampToTerrain(inside, sampler);
    const twice = Measure.clampToTerrain(once, sampler);
    expect(Cartesian3.distance(once, twice) * SEMI_MAJOR).toBeLessThan(1e-3);
    expect(Math.abs(heightOf(twice) - layer.sampleHeight(midLon, midLat))).toBeLessThan(1e-3);
  });
});

// ── 3. PolylineGeometry clampToGround 真实地形链 ────────────────────────────

describe('PolylineGeometry clampToGround 接真实 TerrainLayer', () => {
  /**
   * 收窄 ribbon 宽度：角点 = 贴地点 ± 半宽侧偏，宽度越小角点经纬度越贴近
   * 中心线，高度断言才反映「贴地」而非横向偏移（宽度不影响贴地逻辑本身）。
   */
  const NARROW = { width: 1e-7 };

  it('两远距点跨 32 段加密，每个生成顶点高度 == 该点 sampleHeight（±0.5 m）', () => {
    const a = surfacePoint(0.05, 0.35);
    const b = surfacePoint(0.65, 0.35); // ~0.6° 弧段 → steps = max(ceil(0.6/2), 32) = 32
    const buffers = PolylineGeometry.fromCartesianPositions([a, b], {
      heightReference: 'clampToGround',
      sampleHeight: sampler,
      ...NARROW,
    });

    // 32 细分 × 每段 4 顶点（ribbon 左右各一）
    expect(buffers.vertexCount).toBe(32 * 4);

    const stride = buffers.vertices.length / buffers.vertexCount;
    let maxErr = 0;
    let hMin = Infinity;
    let hMax = -Infinity;
    for (let i = 0; i < buffers.vertexCount; i++) {
      const [lon, lat] = lonLatOfVertex(buffers.vertices, i * stride, stride);
      const h = heightOf(new Cartesian3(
        buffers.vertices[i * stride],
        buffers.vertices[i * stride + 1],
        buffers.vertices[i * stride + 2],
      ));
      expect(lon).toBeGreaterThan(bounds.west);
      expect(lon).toBeLessThan(bounds.east);
      maxErr = Math.max(maxErr, Math.abs(h - layer.sampleHeight(lon, lat)));
      hMin = Math.min(hMin, h);
      hMax = Math.max(hMax, h);
    }
    // 折线横穿高度场数个波长 → 高度摆动数百米（证明跟随地形而非贴椭球）
    expect(hMax - hMin).toBeGreaterThan(100);
    // float32 顶点存储的高度量化上限 ~0.66 m，实测应显著低于 0.5 m
    expect(maxErr).toBeLessThan(0.5);
  });

  it('跨出瓦片边界：界内顶点贴地形，界外顶点退化为椭球面（h = 0）', () => {
    const a = surfacePoint(0.3, 0.35);
    const b = surfacePoint(0.9, 0.35); // 东侧 0.2° 落在未加载区
    const buffers = PolylineGeometry.fromCartesianPositions([a, b], {
      heightReference: 'clampToGround',
      sampleHeight: sampler,
      ...NARROW,
    });
    const stride = buffers.vertices.length / buffers.vertexCount;

    let insideChecked = 0;
    let outsideChecked = 0;
    for (let i = 0; i < buffers.vertexCount; i++) {
      const [lon, lat] = lonLatOfVertex(buffers.vertices, i * stride, stride);
      const h = heightOf(new Cartesian3(
        buffers.vertices[i * stride],
        buffers.vertices[i * stride + 1],
        buffers.vertices[i * stride + 2],
      ));
      if (lon <= bounds.east) {
        expect(Math.abs(h - layer.sampleHeight(lon, lat))).toBeLessThan(0.5);
        insideChecked++;
      } else {
        // 未加载区 sampleHeight → 0 → 贴基础面（float32 量化 ±~0.7 m 内）
        expect(Math.abs(h)).toBeLessThan(1.0);
        outsideChecked++;
      }
    }
    expect(insideChecked).toBeGreaterThan(0);
    expect(outsideChecked).toBeGreaterThan(0);
  });
});

// ── 4. PolygonGeometry clampToGround 真实地形链 ─────────────────────────────

describe('PolygonGeometry clampToGround 接真实 TerrainLayer', () => {
  it('环顶点全部贴合 sampleHeight（±0.5 m），各角落在不同高度上', () => {
    const ring = [
      surfacePoint(0.1, 0.1),
      surfacePoint(0.6, 0.1),
      surfacePoint(0.6, 0.6),
      surfacePoint(0.1, 0.6),
    ];
    const buffers = PolygonGeometry.fromCartesianPositions(ring, {
      heightReference: 'clampToGround',
      sampleHeight: sampler,
    });

    expect(buffers.vertexCount).toBe(4); // 仅环顶点贴地（内部三角化不加密）
    const stride = buffers.vertices.length / buffers.vertexCount;
    let maxErr = 0;
    const heights: number[] = [];
    for (let i = 0; i < buffers.vertexCount; i++) {
      const [lon, lat] = lonLatOfVertex(buffers.vertices, i * stride, stride);
      const h = heightOf(new Cartesian3(
        buffers.vertices[i * stride],
        buffers.vertices[i * stride + 1],
        buffers.vertices[i * stride + 2],
      ));
      maxErr = Math.max(maxErr, Math.abs(h - layer.sampleHeight(lon, lat)));
      heights.push(h);
    }
    // 四角横跨高度场 → 高度互不相同（跟随地形的证据）
    expect(Math.max(...heights) - Math.min(...heights)).toBeGreaterThan(50);
    expect(maxErr).toBeLessThan(0.5);
  });
});
