import { Cartesian3 } from '../math/Cartesian3';
import { Cartographic } from '../math/Cartographic';

/**
 * Measure - 空间量测核心计算（语义对齐 EV-Globe 的 EV_Analysis）。
 *
 * ## 归一化坐标系约定（重要）
 * - 本引擎内部使用**归一化 ECEF 坐标**：地球 WGS84 赤道半径 = 1.0，
 *   即 1 个内部单位 = {@link Measure.EARTH_SCALE} 米（6378137 m）。
 * - 所有输入 `Cartesian3` 均为归一化 ECEF（如 `Camera.position`、
 *   场景 pick 得到的世界坐标）。
 * - 距离/高度返回值为**米**，面积返回值为**平方米（m²）**。
 *
 * 所有方法均为纯静态方法，不修改输入，不依赖渲染状态（无 GPU）。
 */
export class Measure {
  /** 比例因子：1 归一化单位 = EARTH_SCALE 米（WGS84 赤道半径） */
  static readonly EARTH_SCALE = 6378137.0;

  /** WGS84 极半径/赤道半径比（归一化 b = 6356752.3142 / 6378137） */
  static readonly POLAR_RATIO = 6356752.3142 / 6378137.0;

  /** WGS84 平均半径（米），用于球面面积计算 */
  static readonly MEAN_RADIUS = 6371008.8;

  /** WGS84 扁率 */
  private static readonly F = 1 / 298.257223563;

  /**
   * 将点贴地：归一化到椭球单位半径（|p| = 1）。
   * 零向量原样返回。
   */
  static clampToEllipsoid(point: Cartesian3, result?: Cartesian3): Cartesian3 {
    const out = result ?? new Cartesian3();
    const mag = Cartesian3.magnitude(point);
    if (mag < 1e-15) {
      out.x = point.x;
      out.y = point.y;
      out.z = point.z;
      return out;
    }
    out.x = point.x / mag;
    out.y = point.y / mag;
    out.z = point.z / mag;
    return out;
  }

  /**
   * 将点贴到**地形表面**：ECEF → 经纬度（度）→ `sampleHeight` → 重建引擎
   * ECEF，径向 `1 + h / 6378137`。
   *
   * 经纬度采用**引擎面约定**（与 `EllipsoidGeometry` / 地形网格一致的
   * y/z 极半径压缩）：方向先按极半径比还原，再取参数化球面角——这正是
   * 地形瓦片网格顶点的 (lon, lat) 键，因此本函数对已贴地点幂等。
   *
   * `sampleHeight` 返回 0 或非有限值（该处无地形数据）时退化为贴基础面
   * （h = 0）。几何模块通过 options 注入本函数（见 `PolylineGeometry` /
   * `PolygonGeometry` 的 `sampleHeight`），analysis 层不依赖 terrain 模块。
   *
   * @param position - 归一化 ECEF 坐标
   * @param sampleHeight - `(lonDeg, latDeg) => 高程（米）`，通常绑定为
   *        `(lon, lat) => globe.terrainLayer.sampleHeight(lon, lat)`
   * @param result - 可选输出对象
   */
  static clampToTerrain(
    position: Cartesian3,
    sampleHeight: (lon: number, lat: number) => number,
    result?: Cartesian3,
  ): Cartesian3 {
    if (typeof sampleHeight !== 'function') {
      throw new Error('Measure.clampToTerrain: sampleHeight callback is required');
    }
    const mag = Cartesian3.magnitude(position);
    if (mag < 1e-15) {
      const out = result ?? new Cartesian3();
      out.x = position.x;
      out.y = position.y;
      out.z = position.z;
      return out;
    }

    // Parametric lon/lat under the TRUE-ellipsoid engine surface convention
    // (matches ecefFromLonLatHeight / EllipsoidGeometry.createRegion, 2026-10-22):
    // tanφ = z/((1−e²)·hypot(x,y)); n cancels in the inverse.
    const inv = 1.0 / mag;
    const ux = position.x * inv;
    const uy = position.y * inv;
    const uz = position.z * inv;
    const e2 = 1.0 - Measure.POLAR_RATIO * Measure.POLAR_RATIO;
    const lon = Math.atan2(uy, ux);
    const lat = Math.atan2(uz / (1.0 - e2), Math.hypot(ux, uy));

    const h = sampleHeight(lon * (180 / Math.PI), lat * (180 / Math.PI));
    const height = Number.isFinite(h) ? h : 0.0;

    // Rebuild the surface point: true-ellipsoid parameterisation radially
    // scaled by 1 + height / EARTH_SCALE (identical to the terrain
    // mesh vertex formula `ecefFromLonLatHeight`).
    const cosLat = Math.cos(lat);
    const sinLat = Math.sin(lat);
    const n = 1.0 / Math.sqrt(1.0 - e2 * sinLat * sinLat);
    const scale = 1.0 + height / Measure.EARTH_SCALE;

    const out = result ?? new Cartesian3();
    out.x = n * cosLat * Math.cos(lon) * scale;
    out.y = n * cosLat * Math.sin(lon) * scale;
    out.z = n * (1.0 - e2) * sinLat * scale;
    return out;
  }

  /**
   * 折线长度量测（米）。
   *
   * 两点以上按相邻点分段累加。每段长度 = 三维直线弦长 × 球面修正系数，
   * 修正系数 k = 地表测地线长 / 地表投影弦长（Andoyer-Lambert 椭球测地线
   * 近似，对地表点精度 < 0.1%）；两点同经纬度（地表投影重合）时退化为纯弦长。
   *
   * @param points - 归一化 ECEF 坐标数组（至少 2 个）
   * @param clamped - true 时每点先贴地（归一化到半径 1）再累加，
   *                  此时每段即为纯地表测地线距离
   * @param options - 可选地形钩子；提供 `clampToTerrain` 时每个点先经其
   *                  贴到地形表面再累加（优先于 `clamped` 的半径 1 贴地）
   * @returns 总长度（米）
   */
  static distance(
    points: Cartesian3[],
    clamped = false,
    options?: MeasureTerrainOptions,
  ): number {
    if (!points || points.length < 2) return 0.0;
    const terrainClamp = options?.clampToTerrain;
    if (terrainClamp) {
      points = points.map((p) => terrainClamp(p));
    }

    let total = 0.0;
    for (let i = 1; i < points.length; i++) {
      let a = points[i - 1];
      let b = points[i];
      if (clamped) {
        a = Measure.clampToEllipsoid(a);
        b = Measure.clampToEllipsoid(b);
      }
      total += Measure._segmentLength(a, b);
    }
    return total;
  }

  /**
   * 单段长度：三维弦长 × 球面修正系数（测地线/地表投影弦长之比）。
   */
  private static _segmentLength(a: Cartesian3, b: Cartesian3): number {
    const chord = Cartesian3.distance(a, b) * Measure.EARTH_SCALE;

    Measure.clampToEllipsoid(a, Measure._v0);
    Measure.clampToEllipsoid(b, Measure._v1);
    const surfaceChord = Cartesian3.distance(Measure._v0, Measure._v1) * Measure.EARTH_SCALE;
    if (surfaceChord < 1e-9) {
      // 同经纬度（只有高度差），球面修正无意义，返回纯弦长
      return chord;
    }

    const geodesic = Measure._andoyerLambert(Measure._v0, Measure._v1);
    return chord * (geodesic / surfaceChord);
  }

  /**
   * 两点间 Andoyer-Lambert 椭球测地线距离（米）。
   * 输入应为单位向量（贴地点）；近似精度约 1e-4 量级（<0.1%）。
   */
  private static _andoyerLambert(u0: Cartesian3, u1: Cartesian3): number {
    const lat1 = Math.asin(clamp(u0.z, -1, 1));
    const lat2 = Math.asin(clamp(u1.z, -1, 1));
    const lon1 = Math.atan2(u0.y, u0.x);
    const lon2 = Math.atan2(u1.y, u1.x);

    const f = (lat1 + lat2) / 2;
    const g = (lat1 - lat2) / 2;
    const l = (lon2 - lon1) / 2;

    const sinF = Math.sin(f);
    const cosF = Math.cos(f);
    const sinG = Math.sin(g);
    const cosG = Math.cos(g);
    const sinL = Math.sin(l);
    const cosL = Math.cos(l);

    const s = sinG * sinG * cosL * cosL + cosF * cosF * sinL * sinL;
    const c = cosG * cosG * cosL * cosL + sinF * sinF * sinL * sinL;
    if (s < 1e-18 || c < 1e-18) return 0.0;

    const omega = Math.atan(Math.sqrt(s / c));
    const r = Math.sqrt(s * c) / omega;
    const d = 2 * omega * Measure.EARTH_SCALE;
    const h1 = (3 * r - 1) / (2 * c);
    const h2 = (3 * r + 1) / (2 * s);

    return (
      d *
      (1 +
        h1 * Measure.F * sinF * sinF * cosG * cosG -
        h2 * Measure.F * cosF * cosF * sinG * sinG)
    );
  }

  /**
   * 球面多边形面积量测（平方米）。
   *
   * Girard 定理：面积 = |内角和 − (n−2)π| × R²。
   * 各顶点先投影为单位向量（贴地），用相邻边大圆法向的叉积求每个
   * 顶点的内角；R = WGS84 平均半径 6371008.8 m。
   * 顶点顺序（顺/逆时针）均可；面积取绝对值。
   *
   * @param points - 归一化 ECEF 坐标数组（至少 3 个，不含重复首尾点）
   * @param options - 可选地形钩子；提供 `clampToTerrain` 时每个顶点先经其
   *                  贴到地形表面再计算
   * @returns 球面多边形面积（m²）
   */
  static area(points: Cartesian3[], options?: MeasureTerrainOptions): number {
    if (!points || points.length < 3) return 0.0;
    const terrainClamp = options?.clampToTerrain;
    if (terrainClamp) {
      points = points.map((p) => terrainClamp(p));
    }

    const n = points.length;
    let angleSum = 0.0;

    for (let i = 0; i < n; i++) {
      const prev = Measure.clampToEllipsoid(points[(i - 1 + n) % n], Measure._v0);
      const cur = Measure.clampToEllipsoid(points[i], Measure._v1);
      const next = Measure.clampToEllipsoid(points[(i + 1) % n], Measure._v2);

      Cartesian3.cross(prev, cur, Measure._v3);
      Cartesian3.cross(cur, next, Measure._v4);
      const m1 = Cartesian3.magnitude(Measure._v3);
      const m2 = Cartesian3.magnitude(Measure._v4);

      if (m1 < 1e-15 || m2 < 1e-15) {
        // 邻边共点/对跖，弧退化：按平角处理
        angleSum += Math.PI;
        continue;
      }

      Cartesian3.multiplyByScalar(Measure._v3, 1 / m1, Measure._v3);
      Cartesian3.multiplyByScalar(Measure._v4, 1 / m2, Measure._v4);

      const cosAngle = clamp(Cartesian3.dot(Measure._v3, Measure._v4), -1.0, 1.0);
      // 大圆面夹角的补角 = 多边形内角（顺/逆时针序下一致成立）
      angleSum += Math.PI - Math.acos(cosAngle);
    }

    const sphericalExcess = angleSum - (n - 2) * Math.PI;
    return Math.abs(sphericalExcess) * Measure.MEAN_RADIUS * Measure.MEAN_RADIUS;
  }

  /**
   * 高度量测：两点椭球（大地）高度差，单位米。
   *
   * 实现选用 **Cartographic.height 精确版**：将归一化 ECEF 缩放回米制后，
   * 经 `Cartographic.fromCartesian`（Bowring 初值 + Newton 迭代）求大地高，
   * 精度为浮点级；而非 |p|−1 的球面近似。
   *
   * @returns `{ deltaHeight }` = h(p1) − h(p0)（米，可为负）
   */
  static height(p0: Cartesian3, p1: Cartesian3): { deltaHeight: number } {
    return { deltaHeight: Measure._geodeticHeight(p1) - Measure._geodeticHeight(p0) };
  }

  /**
   * 单点大地高（米）：归一化 ECEF → 米制 ECEF → Cartographic.height（归一化）→ 米。
   */
  private static _geodeticHeight(point: Cartesian3): number {
    Measure._v0.x = point.x * Measure.EARTH_SCALE;
    Measure._v0.y = point.y * Measure.EARTH_SCALE;
    Measure._v0.z = point.z * Measure.EARTH_SCALE;
    return Cartographic.fromCartesian(Measure._v0).height * Measure.EARTH_SCALE;
  }

  /**
   * 射线与单位 WGS84 椭球（a = 1，b = {@link Measure.POLAR_RATIO}）求交。
   *
   * 求解二次方程 x²/a² + y²/a² + z²/b² = 1（p = origin + t·direction），
   * 返回最近正 t 对应的交点；direction 会先归一化，故 t 即为归一化单位下的距离。
   *
   * @param origin - 射线起点（归一化 ECEF）
   * @param direction - 射线方向（任意模长，内部归一化）
   * @returns 最近正交交点；无交或无正根时返回 null
   */
  static rayEllipsoid(origin: Cartesian3, direction: Cartesian3): Cartesian3 | null {
    const a = 1.0; // 归一化半长轴
    const b = Measure.POLAR_RATIO; // 归一化半短轴
    const invA2 = 1.0 / (a * a);
    const invB2 = 1.0 / (b * b);

    const dir = Measure.clampToEllipsoid(direction, Measure._v0); // 复用作方向归一化缓冲

    const ox = origin.x;
    const oy = origin.y;
    const oz = origin.z;
    const dx = dir.x;
    const dy = dir.y;
    const dz = dir.z;

    const A = dx * dx * invA2 + dy * dy * invA2 + dz * dz * invB2;
    const B = 2 * (ox * dx * invA2 + oy * dy * invA2 + oz * dz * invB2);
    const C = ox * ox * invA2 + oy * oy * invA2 + oz * oz * invB2 - 1.0;

    if (Math.abs(A) < 1e-18) {
      // 方向沿椭球等值面切向：仅当起点在椭球面上时退化为点解
      return Math.abs(C) < 1e-12 ? origin.clone() : null;
    }

    const disc = B * B - 4 * A * C;
    if (disc < 0) return null;

    const sqrtDisc = Math.sqrt(disc);
    const t1 = (-B - sqrtDisc) / (2 * A);
    const t2 = (-B + sqrtDisc) / (2 * A);

    let t: number;
    if (t1 > 1e-12) {
      t = t1;
    } else if (t2 > 1e-12) {
      t = t2;
    } else {
      return null; // 椭球在起点身后
    }

    return new Cartesian3(ox + t * dx, oy + t * dy, oz + t * dz);
  }

  // 模块级复用缓冲（单线程调用安全；不会泄漏到返回值）
  private static readonly _v0 = new Cartesian3();
  private static readonly _v1 = new Cartesian3();
  private static readonly _v2 = new Cartesian3();
  private static readonly _v3 = new Cartesian3();
  private static readonly _v4 = new Cartesian3();
}

/** 局部 clamp（避免引入 CesiumMath 依赖，语义同 CesiumMath.clamp） */
function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

/**
 * 量测的地形钩子（可选参数，见 `Measure.distance` / `Measure.area`）。
 *
 * 调用方显式注入贴地函数（典型实现：`(p) => Measure.clampToTerrain(p,
 * (lon, lat) => globe.terrainLayer.sampleHeight(lon, lat))`），Measure 不
 * 直接依赖 terrain 模块，保持 analysis 层纯净。缺省时不贴地（0 高假设）。
 */
export interface MeasureTerrainOptions {
  /** 将归一化 ECEF 点贴到地形表面，返回贴地后的点。 */
  clampToTerrain?: (position: Cartesian3) => Cartesian3;
}

/** 量测类型 */
export type MeasureType = 'distance' | 'area' | 'height';

/** 单条量测结果（渲染由调用方组合 Entity/label 完成） */
export interface MeasureItem {
  /** 唯一 id（自动生成，形如 "measure-1"） */
  id: string;
  /** 量测类型 */
  type: MeasureType;
  /** 参与计算的归一化 ECEF 顶点 */
  positions: Cartesian3[];
  /** 量测值：distance → 米，area → m²，height → 米 */
  value: number;
  /** 展示文本，如 "1,234.5 m" / "12.3 km" / "1.2 万 m²" */
  label: string;
}

/**
 * MeasureCollection - 量测结果管理 + 文本标签数据生成。
 *
 * 只负责数据（增删查 + label 文本生成），不做任何渲染；
 * 渲染由调用方读取 {@link MeasureItem.label} / {@link MeasureItem.positions} 后组合。
 */
export class MeasureCollection {
  private _items: MeasureItem[] = [];
  private _nextId = 1;

  /** 当前量测条数 */
  get count(): number {
    return this._items.length;
  }

  /**
   * 由类型与顶点计算一条量测（value + label 自动生成）并加入集合。
   * - distance：`Measure.distance(positions, clamped)`
   * - area：`Measure.area(positions)`
   * - height：`Measure.height(positions[0], positions[1])`
   */
  createMeasurement(type: MeasureType, positions: Cartesian3[], clamped = false): MeasureItem {
    let value = 0.0;
    if (type === 'distance') {
      value = Measure.distance(positions, clamped);
    } else if (type === 'area') {
      value = Measure.area(positions);
    } else {
      if (!positions || positions.length < 2) {
        throw new Error('MeasureCollection: height measurement requires 2 positions');
      }
      value = Measure.height(positions[0], positions[1]).deltaHeight;
    }

    return this.add({
      id: `measure-${this._nextId++}`,
      type,
      positions: positions.map((p) => p.clone()),
      value,
      label: MeasureCollection.formatValue(type, value),
    });
  }

  /** 直接加入一条量测结果 */
  add(item: MeasureItem): MeasureItem {
    this._items.push(item);
    return item;
  }

  /** 按 id 移除，返回是否移除成功 */
  remove(id: string): boolean {
    const index = this._items.findIndex((item) => item.id === id);
    if (index < 0) return false;
    this._items.splice(index, 1);
    return true;
  }

  /** 按 id 查询 */
  get(id: string): MeasureItem | undefined {
    return this._items.find((item) => item.id === id);
  }

  /** 全部量测（只读视图） */
  getAll(): readonly MeasureItem[] {
    return this._items;
  }

  /** 清空 */
  clear(): void {
    this._items.length = 0;
  }

  /** 按类型生成展示文本 */
  static formatValue(type: MeasureType, value: number): string {
    if (type === 'area') return MeasureCollection.formatArea(value);
    return MeasureCollection.formatDistance(value);
  }

  /**
   * 距离/高度文本：< 10000 m → "1,234.5 m"；否则 → "12.3 km"
   */
  static formatDistance(meters: number): string {
    if (Math.abs(meters) >= 10000.0) {
      return `${MeasureCollection._formatNumber(meters / 1000.0, 1)} km`;
    }
    return `${MeasureCollection._formatNumber(meters, 1)} m`;
  }

  /**
   * 面积文本：< 10000 m² → "1,234.5 m²"；否则 → "1.2 万 m²"
   */
  static formatArea(areaM2: number): string {
    if (Math.abs(areaM2) >= 10000.0) {
      return `${MeasureCollection._formatNumber(areaM2 / 10000.0, 1)} 万 m²`;
    }
    return `${MeasureCollection._formatNumber(areaM2, 1)} m²`;
  }

  /** 千分位 + 固定小数位（不依赖 Intl，保证跨环境一致） */
  private static _formatNumber(value: number, decimals: number): string {
    const fixed = value.toFixed(decimals);
    const dotIndex = fixed.indexOf('.');
    const intPart = dotIndex < 0 ? fixed : fixed.slice(0, dotIndex);
    const fracPart = dotIndex < 0 ? '' : fixed.slice(dotIndex);
    const withSeparators = intPart.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    return `${withSeparators}${fracPart}`;
  }
}
