import { Cartesian3 } from '../math/Cartesian3';
import { Matrix4 } from '../math/Matrix4';
import { Transforms } from '../math/Transforms';
import { GeometryPipeline } from './GeometryPipeline';
import type { InterleavedGeometryBuffers } from './GeometryPipeline';
import { earcut2D } from './PolygonGeometry';

/**
 * 经纬度（度）平面点。城市建筑 footprint 的最小表达 —— 相比
 * `PolygonGeometry` 的归一化 ECEF 输入，批量建筑用经纬度环更省一次
 * 调用方换算（内部一次性转 ECEF）。
 */
export interface LonLatPoint {
  /** 经度（度） */
  lon: number;
  /** 纬度（度） */
  lat: number;
}

export interface ExtrudedGeometryOptions {
  /**
   * footprint 外环（经纬度，度）。顶点顺序不限 —— 内部会按 ENU 投影的
   * 有向面积归一化为 CCW 后再生成侧面/顶面，保证法向朝外（ Primitive
   * 默认 frontFace ccw + cullMode back，见 polarFillWinding 历史坑）。
   * 首末点重复（GeoJSON 常见闭合写法）会自动去重。
   */
  ring: LonLatPoint[];
  /** 拉伸高度（米）。<= 0 时退化为纯平顶面（无侧面）。 */
  heightM: number;
  /** 底部海拔（米），默认 0（椭球面）。 */
  baseHeightM?: number;
  /**
   * 语义 ID，透传到输出的 `semanticIds`（默认 0）。
   * 供后续分割 pass（F3）按楼栋切分顶点属性使用。
   */
  semanticId?: number;
  /**
   * 压扁世界放置（默认 false = 未压扁 WGS84 系，fromDegreesNormalized）。
   * true 时底/顶环与 up 走引擎地表同款线性压扁映射（单位球方向 ×(1+h/a)，
   * y/z × ρ）——与 Globe/影像瓦片同一表面。城市要贴影像地面时必须开：
   * 两系表面在中纬度差 ~13km（2026-09-30 实测光谷 30.5°N 差 13.2km），
   * 默认系城市悬在影像上空即「白膜飘在空中」。
   */
  flattened?: boolean;
}

/**
 * 拉伸体几何缓冲 —— 在 `InterleavedGeometryBuffers` 之上追加每顶点
 * semanticId 数组（u32，与 vertices 顶点一一对应）。顶点布局本身仍是
 * 8 float / 32 字节标准交错格式，`Primitive` 可直接消费。
 */
export interface ExtrudedGeometryBuffers extends InterleavedGeometryBuffers {
  /** 每顶点语义 ID（u32）。F3 分割 pass 的顶点属性数据源。 */
  semanticIds: Uint32Array;
}

/**
 * ExtrudedGeometry - 建筑白模拉伸体：经纬度 footprint 环 + 高度（米）
 * 生成侧面 + 顶面（底面省略 —— 贴地不可见且省 2n 顶点），输出归一化
 * ECEF 交错缓冲（Earth 赤道半径 = 1，米经 `fromDegreesNormalized` 换算）。
 *
 * ## Winding 约定（对齐 Primitive frontFace:'ccw' + cullMode:'back'）
 *
 * 在 footprint 质心 ENU 框架（east × north = up，右手系）下：
 * - 外环先归一化为 CCW（有向面积 > 0）；
 * - 侧面四边形 (bᵢ, bⱼ, tⱼ, tᵢ)：边方向 e = bⱼ - bᵢ，则 e × up 恰为
 *   CCW 环的外向水平法向，三角形 (bᵢ, bⱼ, tⱼ) 的几何法向 = h·(e×up)
 *   朝外 ✓；第二个三角形 (bᵢ, tⱼ, tᵢ) 同理；
 * - 顶面用 `earcut2D` 三角化（CCW 输入 → CCW 输出三角形），几何法向
 *   朝上 ✓。
 *
 * 相邻建筑各自独立 mesh、不共边，无裙边需求（垂直侧壁不悬空）。
 */
export class ExtrudedGeometry {
  /**
   * 生成拉伸体几何。
   * @returns 交错缓冲 + semanticIds；ring 少于 3 个有效点时返回空缓冲
   *          （与 `GeometryPipeline.emptyBuffers` 同约定，WebGPU 禁止
   *          零尺寸 buffer，调用方须跳过空几何）。
   */
  static create(options: ExtrudedGeometryOptions): ExtrudedGeometryBuffers {
    const semanticId = options.semanticId ?? 0;

    // 去除 GeoJSON 式闭合首末重复点（只比经纬度，浮点相等判定）
    const ring: LonLatPoint[] = [];
    for (const p of options.ring) {
      const last = ring[ring.length - 1];
      if (last && last.lon === p.lon && last.lat === p.lat) continue;
      ring.push({ lon: p.lon, lat: p.lat });
    }
    if (ring.length >= 3) {
      const first = ring[0];
      const last = ring[ring.length - 1];
      if (first.lon === last.lon && first.lat === last.lat) ring.pop();
    }
    if (ring.length < 3) {
      return { ...GeometryPipeline.emptyBuffers(), semanticIds: new Uint32Array(0) };
    }

    const n = ring.length;
    const baseM = options.baseHeightM ?? 0;
    const topM = baseM + Math.max(0, options.heightM);
    const extrude = options.heightM > 0;
    // 放置映射:默认未压扁 WGS84(fromDegreesNormalized);flattened 时走引擎
    // 地表同款线性压扁(与 Globe/影像瓦片共面)。ENU 投影(求 winding)两系通用。
    const RHO = 6356752.3142 / 6378137.0;
    const pt = (lonDeg: number, latDeg: number, hM: number): Cartesian3 => {
      if (!options.flattened) return Cartesian3.fromDegreesNormalized(lonDeg, latDeg, hM);
      const lon = (lonDeg * Math.PI) / 180, lat = (latDeg * Math.PI) / 180;
      const s = 1.0 + hM / 6378137.0;
      const cosLat = Math.cos(lat);
      return new Cartesian3(
        cosLat * Math.cos(lon) * s,
        cosLat * Math.sin(lon) * RHO * s,
        Math.sin(lat) * RHO * s,
      );
    };

    // ENU 投影（footprint 尺度 << 地球半径，质心经纬度均值即足够）
    let lonSum = 0, latSum = 0;
    for (const p of ring) { lonSum += p.lon; latSum += p.lat; }
    const center = Cartesian3.fromDegrees(lonSum / n, latSum / n, 0);
    const inv = Matrix4.inverse(Transforms.eastNorthUpToFixedFrame(center), new Matrix4()).values;

    // ENU 平面投影 + 有向面积（决定顶点序归一化方向）
    const east: number[] = new Array(n);
    const north: number[] = new Array(n);
    let area2 = 0; // 2×有向面积，> 0 即 CCW（east×north=up 右手系）
    for (let i = 0; i < n; i++) {
      const p = Cartesian3.fromDegreesNormalized(ring[i].lon, ring[i].lat, baseM);
      east[i] = inv[0] * p.x + inv[4] * p.y + inv[8] * p.z + inv[12];
      north[i] = inv[1] * p.x + inv[5] * p.y + inv[9] * p.z + inv[13];
      const j = (i + 1) % n;
      area2 += east[i] * north[j] - east[j] * north[i];
    }
    if (area2 < 0) {
      // CW 输入 → 反转，保证侧面外向推导（e×up = 外向）成立
      ring.reverse();
      east.reverse();
      north.reverse();
    }

    // ECEF 位置：底环 / 顶环（米 → 归一化经 fromDegreesNormalized）
    const bottom: Cartesian3[] = new Array(n);
    const top: Cartesian3[] = new Array(n);
    for (let i = 0; i < n; i++) {
      const p = ring[i];
      bottom[i] = pt(p.lon, p.lat, baseM);
      top[i] = pt(p.lon, p.lat, topM);
    }

    // up：顶面法向 / 侧面外向基准（质心径向）
    const up = Cartesian3.normalize(
      pt(lonSum / n, latSum / n, 0),
      new Cartesian3(),
    );

    // 顶面 UV：ENU 包围盒归一化（与 PolygonGeometry 同约定）
    let eMin = Infinity, eMax = -Infinity, nMin = Infinity, nMax = -Infinity;
    for (let i = 0; i < n; i++) {
      eMin = Math.min(eMin, east[i]); eMax = Math.max(eMax, east[i]);
      nMin = Math.min(nMin, north[i]); nMax = Math.max(nMax, north[i]);
    }
    const eRange = eMax - eMin > 1e-15 ? eMax - eMin : 1.0;
    const nRange = nMax - nMin > 1e-15 ? nMax - nMin : 1.0;

    // 顶点预算：侧面 4n（每边独立 4 顶点，平直着色不共边）+ 顶面 n
    const sideVertCount = extrude ? n * 4 : 0;
    const vertexCount = sideVertCount + n;
    const sideIndexCount = extrude ? n * 6 : 0;
    const indexCount = sideIndexCount + (n - 2) * 3;

    const positions = new Float32Array(vertexCount * 3);
    const normals = new Float32Array(vertexCount * 3);
    const uvs = new Float32Array(vertexCount * 2);
    const indices = new Uint32Array(indexCount);
    const semanticIds = new Uint32Array(vertexCount);
    semanticIds.fill(semanticId);

    const writeVert = (
      v: number, p: Cartesian3,
      nx: number, ny: number, nz: number,
      u: number, w: number,
    ): void => {
      positions[v * 3] = p.x; positions[v * 3 + 1] = p.y; positions[v * 3 + 2] = p.z;
      normals[v * 3] = nx; normals[v * 3 + 1] = ny; normals[v * 3 + 2] = nz;
      uvs[v * 2] = u; uvs[v * 2 + 1] = w;
    };

    // ── 侧面：每条边独立四边形，法向 = normalize(e × up)（CCW 环外向）──
    if (extrude) {
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        const e = Cartesian3.normalize(
          Cartesian3.subtract(top[j], top[i], new Cartesian3()),
          new Cartesian3(),
        );
        const out = Cartesian3.normalize(Cartesian3.cross(e, up, new Cartesian3()), new Cartesian3());
        const b = i * 4; // 本边 4 顶点在侧面段的基址
        writeVert(b,     bottom[i], out.x, out.y, out.z, 0, 0);
        writeVert(b + 1, bottom[j], out.x, out.y, out.z, 1, 0);
        writeVert(b + 2, top[j],    out.x, out.y, out.z, 1, 1);
        writeVert(b + 3, top[i],    out.x, out.y, out.z, 0, 1);
        const t = i * 6;
        indices[t]     = b;
        indices[t + 1] = b + 1;
        indices[t + 2] = b + 2;
        indices[t + 3] = b;
        indices[t + 4] = b + 2;
        indices[t + 5] = b + 3;
      }
    }

    // ── 顶面：earcut 三角化（CCW 输入 → CCW 三角形 → 法向朝上）──────
    const flat: number[] = [];
    for (let i = 0; i < n; i++) flat.push(east[i], north[i]);
    const topTris = earcut2D(flat, undefined, 2);
    const topBase = sideVertCount;
    for (let i = 0; i < n; i++) {
      const v = topBase + i;
      writeVert(v, top[i], up.x, up.y, up.z, (east[i] - eMin) / eRange, (north[i] - nMin) / nRange);
    }
    for (let t = 0; t < topTris.length; t++) {
      indices[sideIndexCount + t] = topBase + topTris[t];
    }

    return {
      vertices: GeometryPipeline.interleave(positions, normals, uvs),
      indices,
      vertexCount,
      indexCount,
      semanticIds,
    };
  }
}

export default ExtrudedGeometry;
