import type { TerrainLayer } from '../terrain/TerrainLayer';
import type { HeightSampler } from '../media/GimbalMath';

/**
 * 经纬度矩形（度）。与 `TerrainLayer.TerrainTileBounds` 结构兼容。
 */
export interface RectangleDeg {
  west: number;
  south: number;
  east: number;
  north: number;
}

/** {@link FloodAnalysis.computeFlooded} 的结果。 */
export interface FloodResult {
  /**
   * 淹没区等值多边形环（经纬度度），`polygons[i]` 为第 i 个环，
   * 环内元素为 `[lonDeg, latDeg]`，首尾点严格相同（闭合）。
   */
  polygons: number[][][];
  /** 淹没面积（km²，球面小格近似 + 纬度 cos 修正）。 */
  areaKm2: number;
  /** 本次水位（米）。 */
  level: number;
  /** 分析区（度）。 */
  bounds: RectangleDeg;
  /** 采样网格边长（gridN × gridN）。 */
  gridN: number;
}

export interface FloodAnalysisOptions {
  /** 采样网格边长（正方形）。Default: `128` */
  gridN?: number;
  /**
   * 显式高程采样器（`(lonDeg, latDeg) => 米`）。缺省时读
   * `terrainLayer.sampleHeight`。供测试 / 合成 DEM / 已有采样回调注入。
   */
  heightSampler?: HeightSampler;
}

/**
 * FloodAnalysis - 水深/淹没分析（纯计算，不渲染，照 MeasureCollection 模式）。
 *
 * 流程：
 * 1. {@link prepareRegion}：`TerrainLayer.ensureRegion` 预载分析区瓦片 +
 *    `pinRegion` 钉住（防 LRU 挤出），幂等；
 * 2. {@link computeFlooded}：gridN × gridN 网格采样高程 → 淹没掩码
 *    （`h <= waterLevel`）→ marching squares 提取水位等值多边形 →
 *    球面小格面积累加（逐行 cos(lat) 修正）。
 *
 * 只负责数据（多边形环 + 面积），渲染由调用方组合
 * {@link import('../effect/FloodWater').FloodWater} / Entity 完成。
 *
 * @example
 * ```typescript
 * const flood = new FloodAnalysis(viewer.globe.terrainLayer, { gridN: 128 });
 * await flood.prepareRegion(bounds, 10);
 * const res = flood.computeFlooded(bounds, 500); // 500 m 水位
 * console.log(res.areaKm2, res.polygons.length);
 * flood.release(); // unpinAll
 * ```
 */
export class FloodAnalysis {
  private readonly _layer: TerrainLayer;
  private readonly _gridN: number;
  private readonly _sampler: HeightSampler | null;
  private _released = false;

  constructor(layer: TerrainLayer, options?: FloodAnalysisOptions) {
    const gridN = options?.gridN ?? DEFAULT_GRID_N;
    if (!Number.isFinite(gridN) || gridN < 2) {
      throw new Error(`FloodAnalysis: gridN 必须 >= 2，收到 ${gridN}`);
    }
    this._layer   = layer;
    this._gridN   = Math.floor(gridN);
    this._sampler = options?.heightSampler ?? null;
  }

  /** 采样网格边长。 */
  get gridN(): number {
    return this._gridN;
  }

  /**
   * 预载并 pin 分析区地形瓦片（幂等：缓存命中不重复请求，Set 登记不重复 pin）。
   *
   * @param bounds 分析区（度）
   * @param level  瓦片层级。Default: `10`（pin 的最高允许层级）
   */
  async prepareRegion(bounds: RectangleDeg, level: number = MAX_PIN_LEVEL): Promise<void> {
    if (this._released) throw new Error('FloodAnalysis: 已释放（release 后不可再用）');
    await this._layer.ensureRegion(bounds, level);
    this._layer.pinRegion(bounds, level);
  }

  /**
   * 采样 `bounds` 的高程网格（行主序 Float64Array，`grid[iy * n + ix]`），
   * lon 西→东、lat 南→北（含边界，网格点落在 bounds 四角上）。
   * 暴露给渲染层（如 FloodWater 的注入数据源）。
   */
  heightGrid(bounds: RectangleDeg, n: number = this._gridN): Float64Array {
    if (!Number.isFinite(n) || n < 2) {
      throw new Error(`FloodAnalysis.heightGrid: n 必须 >= 2，收到 ${n}`);
    }
    const sampler = this._sampler ?? ((lon: number, lat: number) => this._layer.sampleHeight(lon, lat));
    const grid = new Float64Array(n * n);
    const dLon = (bounds.east - bounds.west) / (n - 1);
    const dLat = (bounds.north - bounds.south) / (n - 1);
    for (let iy = 0; iy < n; iy++) {
      const lat = bounds.south + dLat * iy;
      for (let ix = 0; ix < n; ix++) {
        grid[iy * n + ix] = sampler(bounds.west + dLon * ix, lat);
      }
    }
    return grid;
  }

  /**
   * 计算水位 `waterLevel`（米）下的淹没结果：等值多边形 + 面积。
   * 纯内存计算（网格已由 prepareRegion / 渲染调度装进缓存），可高频调用。
   */
  computeFlooded(bounds: RectangleDeg, waterLevel: number): FloodResult {
    if (this._released) throw new Error('FloodAnalysis: 已释放（release 后不可再用）');
    if (!Number.isFinite(waterLevel)) {
      throw new Error(`FloodAnalysis.computeFlooded: 水位非法 ${waterLevel}`);
    }

    const n    = this._gridN;
    const grid = this.heightGrid(bounds, n);

    // ── 面积：淹没格逐行累加，格面积用该行纬度的球面小格近似 ──────────
    // 每格宽 = Δlon·111.32·cos(lat) km，高 = Δlat·110.86 km（经线/纬线
    // 1° 近似弧长；小区域误差远低于 DEM 精度）。
    const dLonDeg = (bounds.east - bounds.west) / (n - 1);
    const dLatDeg = (bounds.north - bounds.south) / (n - 1);
    const dLatKm  = dLatDeg * KM_PER_DEG_LAT;
    let floodedCells = 0;
    let areaKm2 = 0.0;
    for (let iy = 0; iy < n; iy++) {
      const lat    = bounds.south + dLatDeg * iy;
      const dLonKm = dLonDeg * KM_PER_DEG_LON * Math.cos((lat * Math.PI) / 180.0);
      const cellKm2 = dLonKm * dLatKm;
      for (let ix = 0; ix < n; ix++) {
        if (grid[iy * n + ix] <= waterLevel) {
          floodedCells++;
          areaKm2 += cellKm2;
        }
      }
    }

    // ── 等值多边形：marching squares（16 case 线性内插）+ 线段成环 ──────
    const segments = marchingSquaresSegments(grid, n, waterLevel, bounds);
    const polygons = chainSegments(segments);

    return {
      polygons,
      areaKm2,
      level: waterLevel,
      bounds: { ...bounds },
      gridN: n,
    };
  }

  /** 释放分析 pin（unpinAll）。之后本实例不可再用。 */
  release(): void {
    if (this._released) return;
    this._released = true;
    this._layer.unpinAll();
  }
}

// ── Marching squares（等值线提取）────────────────────────────────────────────

/** 单条等值线段（经纬度度端点）。 */
interface Seg {
  ax: number; ay: number;
  bx: number; by: number;
}

/**
 * 网格上的 marching squares：对每个 cell 按 4 角相对阈值的 16 case
 * 生成等值线段，cell 边上的交点线性内插。角编号：
 * `c0=SW, c1=SE, c2=NE, c3=NW`，case 位 `idx = c0 | c1<<1 | c2<<2 | c3<<3`
 * （角值 > 阈值置位）。saddle case（5/10）按固定对角拆分（一致性优先，
 * 二义性在小格尺度不可感知）。
 */
function marchingSquaresSegments(
  grid: Float64Array,
  n: number,
  level: number,
  bounds: RectangleDeg
): Seg[] {
  const segs: Seg[] = [];
  const spanLon = bounds.east - bounds.west;
  const spanLat = bounds.north - bounds.south;
  const step = 1.0 / (n - 1);

  // cell 边上的内插点：(x0,y0)-(x1,y1) 为网格参数坐标（0..1），va/vb 为两端高程。
  const edgePoint = (
    x0: number, y0: number, x1: number, y1: number,
    va: number, vb: number
  ): [number, number] => {
    const t = Math.abs(vb - va) < 1e-12 ? 0.5 : (level - va) / (vb - va);
    const tc = Math.min(1.0, Math.max(0.0, t));
    return [
      bounds.west + (x0 + (x1 - x0) * tc) * spanLon,
      bounds.south + (y0 + (y1 - y0) * tc) * spanLat,
    ];
  };

  for (let iy = 0; iy < n - 1; iy++) {
    for (let ix = 0; ix < n - 1; ix++) {
      const v0 = grid[iy * n + ix];       // SW
      const v1 = grid[iy * n + ix + 1];   // SE
      const v2 = grid[(iy + 1) * n + ix + 1]; // NE
      const v3 = grid[(iy + 1) * n + ix];     // NW

      let idx = 0;
      if (v0 > level) idx |= 1;
      if (v1 > level) idx |= 2;
      if (v2 > level) idx |= 4;
      if (v3 > level) idx |= 8;
      if (idx === 0 || idx === 15) continue; // 全陆 / 全水：无等值线

      const x0 = ix * step;
      const x1 = (ix + 1) * step;
      const y0 = iy * step;
      const y1 = (iy + 1) * step;
      const bottom = (): [number, number] => edgePoint(x0, y0, x1, y0, v0, v1);
      const right  = (): [number, number] => edgePoint(x1, y0, x1, y1, v1, v2);
      const top    = (): [number, number] => edgePoint(x0, y1, x1, y1, v3, v2);
      const left   = (): [number, number] => edgePoint(x0, y0, x0, y1, v0, v3);

      const push = (a: [number, number], b: [number, number]): void => {
        segs.push({ ax: a[0], ay: a[1], bx: b[0], by: b[1] });
      };

      switch (idx) {
        case 1:  push(left(), bottom()); break;
        case 2:  push(bottom(), right()); break;
        case 3:  push(left(), right()); break;
        case 4:  push(right(), top()); break;
        case 5:  push(left(), bottom()); push(right(), top()); break; // saddle
        case 6:  push(bottom(), top()); break;
        case 7:  push(left(), top()); break;
        case 8:  push(top(), left()); break;
        case 9:  push(bottom(), top()); break;
        case 10: push(bottom(), right()); push(top(), left()); break; // saddle
        case 11: push(right(), top()); break;
        case 12: push(left(), right()); break;
        case 13: push(bottom(), right()); break;
        case 14: push(left(), bottom()); break;
      }
    }
  }
  return segs;
}

/**
 * 线段端点量化键（1e-9 度 ≈ 0.1 mm，远低于任何 DEM 分辨率；
 * 相邻 cell 共享边上的内插点由同一对角值算出，量化后必然同键）。
 */
function pointKey(lon: number, lat: number): string {
  return `${Math.round(lon * 1e9)}_${Math.round(lat * 1e9)}`;
}

/**
 * 线段连接成环：端点邻接表 + 沿链行走。闭合链回到起点成环；触及网格
 * 边界的开链强制闭合（append 首点）。输出环首尾点严格相同。
 */
function chainSegments(segs: Seg[]): number[][][] {
  if (segs.length === 0) return [];

  // 端点 → 邻接线段索引
  const adj = new Map<string, number[]>();
  const addAdj = (lon: number, lat: number, si: number): void => {
    const k = pointKey(lon, lat);
    const list = adj.get(k);
    if (list) list.push(si);
    else adj.set(k, [si]);
  };
  for (let i = 0; i < segs.length; i++) {
    addAdj(segs[i].ax, segs[i].ay, i);
    addAdj(segs[i].bx, segs[i].by, i);
  }

  const used = new Array<boolean>(segs.length).fill(false);
  const rings: number[][][] = [];

  for (let i = 0; i < segs.length; i++) {
    if (used[i]) continue;
    used[i] = true;

    const start: [number, number] = [segs[i].ax, segs[i].ay];
    const ring: number[][] = [start];
    let cur: [number, number] = [segs[i].bx, segs[i].by];
    ring.push(cur);

    for (;;) {
      const node = adj.get(pointKey(cur[0], cur[1]));
      let next = -1;
      if (node) {
        for (const si of node) {
          if (!used[si]) { next = si; break; }
        }
      }
      if (next < 0) break; // 链终点（闭合环回到起点 / 开链触网格边界）

      const s = segs[next];
      used[next] = true;
      // s 的一端量化等于 cur，走另一端
      cur = pointKey(s.ax, s.ay) === pointKey(cur[0], cur[1])
        ? [s.bx, s.by]
        : [s.ax, s.ay];
      ring.push(cur);
    }

    // 强制闭合：末点替换为首点值（量化同键的微差一并消除），
    // 保证环严格闭合（首尾点相同）。
    ring[ring.length - 1] = [start[0], start[1]];
    if (ring.length >= 4) rings.push(ring); // <4 点为退化碎屑（两点一线）
  }
  return rings;
}

/** 默认采样网格边长。 */
const DEFAULT_GRID_N = 128;

/** prepareRegion 默认（也是 pin 允许的最高）瓦片层级。 */
const MAX_PIN_LEVEL = 10;

/** 纬线方向 1° 近似弧长（km，WGS84 平均）。 */
const KM_PER_DEG_LAT = 110.86;
/** 经线方向 1° 近似弧长（km，赤道；乘 cos(lat) 修正）。 */
const KM_PER_DEG_LON = 111.32;

export default FloodAnalysis;
