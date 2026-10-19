import type { Context } from '../renderer/Context';
import { Primitive } from '../scene/Primitive';
import { Color } from '../math/Color';
import { Cartesian3 } from '../math/Cartesian3';
import {
  WindArrowGeometry, WIND_ARROW_VERTEX_COUNT, WIND_ARROW_INDEX_COUNT,
} from './WindArrowGeometry';
import { GeometryPipeline } from './GeometryPipeline';
import type { InterleavedGeometryBuffers } from './GeometryPipeline';

/** wind-field/v1 的 origin 段（经纬度 + 基准海拔）。 */
export interface WindFieldOrigin {
  lon: number;
  lat: number;
  alt_m?: number;
}

/** wind-field/v1 的单支箭头（格点）记录。 */
export interface WindFieldArrow {
  i?: number; j?: number; k?: number;
  lat?: number; lon?: number; alt_m?: number;
  /** 相对 origin 的 ENU 米制位置 [east, north, up]。 */
  enu: [number, number, number];
  /** ENU 风速矢量 [u, v, w]（m/s，u≈东、v≈北、w≈垂直）。 */
  vec: [number, number, number];
  speed: number;
}

/** wind-field/v1 的 stats 段（全部可选 —— 缺省时由 arrows 现算）。 */
export interface WindFieldStats {
  count?: number;
  speed_min?: number;
  speed_max?: number;
}

/** 平台侧 wind-field/v1 JSON 契约（工具产出，引擎只读消费，勿改）。 */
export interface WindFieldData {
  schema?: string;
  origin: WindFieldOrigin;
  grid?: {
    nx?: number; ny?: number; dx_m?: number; dy_m?: number; heights_m?: number[];
  };
  arrows: WindFieldArrow[];
  stats?: WindFieldStats;
}

/** 一个速度档的纯数据结果（无 GPU 依赖，可直接测试 / 离线检查）。 */
export interface WindFieldBucket {
  /** 速度档序号（0 = 最慢 … bucketCount-1 = 最快）。 */
  bucket: number;
  /** 该档单色 [r,g,b]（0..1 线性），Primitive 基色。 */
  color: [number, number, number];
  /** 该档速度下界（m/s，含）/ 上界（最快档含 speed_max）。 */
  speedLo: number;
  speedHi: number;
  /** 合并后的箭头几何（8 float 交错 + 索引），`Primitive` 直接消费。 */
  geometry: InterleavedGeometryBuffers;
  /** 该档包含的箭头数。 */
  arrowCount: number;
}

export interface WindFieldBuilderOptions {
  /**
   * 风场 origin（经纬度 + 基准海拔米）。缺省取 `field.origin`——
   * 两者都缺时抛错（没有锚点无法放置）。
   */
  origin?: WindFieldOrigin;
  /**
   * 压扁世界放置（默认 false = 未压扁 WGS84 系，fromDegreesNormalized）。
   * true 时走引擎地表同款线性压扁映射（`ExtrudedGeometry.flattened` 同
   * 语义）——页面带 Globe/影像底图时必须开，否则箭头场悬在影像上空
   * ~13km（中纬度）。
   */
  flattened?: boolean;
  /**
   * RTC 几何中心：归一化 ECEF [x,y,z]。缺省 undefined = 自动取 origin 的
   * 世界坐标（顶点相对它存储，Primitive.origin 走 encoded-camera 精确
   * 深度路径，boundingSphere 回世界系加回中心）；显式 null = 顶点直接
   * 烘焙绝对世界坐标（旧行为，页面 `?rtc=0`）。
   */
  rtcOrigin?: [number, number, number] | null;
  /** 速度分档数（默认 8）。 */
  bucketCount?: number;
  /** 箭头最短长度（米，默认 8）。 */
  lengthMinM?: number;
  /** 箭头最长长度（米，默认 45；speed_max 线性映射到此）。 */
  lengthMaxM?: number;
  /** 速度范围覆盖（缺省用 field.stats / arrows 现算）。 */
  speedMin?: number;
  speedMax?: number;
}

/** `buildWindField` 的选项：builder 选项 + 渲染上下文。 */
export interface WindFieldBuildOptions extends WindFieldBuilderOptions {
  /** WebGPU 上下文（`viewer.context`）。 */
  context: Context;
}

/**
 * WindFieldBuilder - 三维风场箭头场批量合批构建器（wind-field/v1）。
 *
 * ## 放置（ENU → 世界系）
 *
 * 每格点的 `enu`（相对 origin 米制）经 WGS84 局部米/度系数换回经纬高
 * （97 大屏 enuToLonLatAlt 同款精确系数），再按 `flattened` 选择压扁/
 * 未压扁归一化 ECEF 映射。箭头姿态取水平风向（vec 的 e/n 分量，垂直
 * 分量忽略），在格点 ENU 框架内构造正交基 (r, f, up)（f = 水平风向）
 * 旋转局部 +Y 轴箭头。
 *
 * ## 速度分桶配色（零 shader 改动）
 *
 * speed 在 [speed_min, speed_max] 均分 8 档，每档一个 Primitive、单色
 * 材质（蓝→青→绿→黄→橙→红手写色带）；同档箭头顶点合并进同一
 * buffer（一次 drawIndexed），最多 8 个 Primitive。
 *
 * ## 箭头长度
 *
 * `speed / speed_max × lengthMaxM` 线性映射后 clamp 到
 * [lengthMinM, lengthMaxM]（默认 8..45 m）。
 *
 * ## 退化箭头
 *
 * 水平风速 < 1e-9 m/s（vec (0,0,0) 或纯垂直）的格点**跳过**（静默，
 * 计入 skippedCount）——零长箭头画不出方向也没有信息量。
 *
 * 注意：返回的 Primitive 必须经 `scene.addPrimitive(p)` 入场（同
 * `CityBlockBuilder` 约定，直接 `scene.primitives.add` 会缺 bind group
 * 被静默跳过）。
 */
export class WindFieldBuilder {
  /** WGS84 赤道半径（米）。 */
  static readonly EARTH_A = 6378137.0;
  /** WGS84 极半径（米）。 */
  static readonly EARTH_B = 6356752.3142;
  /** 压扁比 ρ = b/a（压扁世界 y/z 缩放，`ExtrudedGeometry` 同值）。 */
  static readonly FLATTEN_RHO = WindFieldBuilder.EARTH_B / WindFieldBuilder.EARTH_A;

  /** 默认速度档数。 */
  static readonly DEFAULT_BUCKETS = 8;
  /** 默认箭头长度范围（米）。 */
  static readonly LENGTH_MIN_M = 8;
  static readonly LENGTH_MAX_M = 45;

  /**
   * 速度档色带（8 档，线性 [r,g,b]）：蓝 → 青 → 绿 → 黄 → 橙 → 红。
   * 手写档色阶梯 —— 档色即 Primitive 基色，页面图例同源。
   */
  static readonly BUCKET_COLORS: readonly (readonly [number, number, number])[] = [
    [0.20, 0.40, 1.00],   // 慢：蓝
    [0.10, 0.70, 1.00],   // 蓝青
    [0.10, 0.95, 0.85],   // 青
    [0.30, 0.95, 0.30],   // 绿
    [0.90, 0.95, 0.10],   // 黄绿
    [1.00, 0.80, 0.10],   // 黄橙
    [1.00, 0.45, 0.10],   // 橙
    [1.00, 0.20, 0.10],   // 快：红
  ];

  /** 水平风速低于该值（m/s）的格点跳过。 */
  static readonly SPEED_EPS = 1e-9;

  private readonly _originLon: number;
  private readonly _originLat: number;
  private readonly _originAltM: number;
  private readonly _flattened: boolean;
  private readonly _rtc: [number, number, number] | null;
  private readonly _buckets: number;
  private readonly _lenMin: number;
  private readonly _lenMax: number;
  private readonly _speedMin: number;
  private readonly _speedMax: number;

  private readonly _arrows: WindFieldArrow[];

  private _arrowCount = 0;
  private _skippedCount = 0;
  private _bucketsOut: WindFieldBucket[] | null = null;

  constructor(field: WindFieldData, options: WindFieldBuilderOptions = {}) {
    this._arrows = Array.isArray(field.arrows) ? field.arrows : [];
    const origin = options.origin ?? field.origin;
    if (!origin || !Number.isFinite(origin.lon) || !Number.isFinite(origin.lat)) {
      throw new Error('WindFieldBuilder: 缺少 origin（options.origin / field.origin）');
    }
    this._originLon = origin.lon;
    this._originLat = origin.lat;
    this._originAltM = origin.alt_m ?? 0;
    this._flattened = options.flattened ?? false;
    this._buckets = Math.max(1, Math.floor(options.bucketCount ?? WindFieldBuilder.DEFAULT_BUCKETS));
    this._lenMin = options.lengthMinM ?? WindFieldBuilder.LENGTH_MIN_M;
    this._lenMax = Math.max(this._lenMin, options.lengthMaxM ?? WindFieldBuilder.LENGTH_MAX_M);

    // 速度范围：显式覆盖 > field.stats > arrows 现算（跳过非有限值）。
    let sMin = options.speedMin ?? field.stats?.speed_min;
    let sMax = options.speedMax ?? field.stats?.speed_max;
    if (sMin === undefined || sMax === undefined) {
      let lo = Infinity, hi = -Infinity;
      for (const a of field.arrows) {
        if (!Number.isFinite(a.speed)) continue;
        if (a.speed < lo) lo = a.speed;
        if (a.speed > hi) hi = a.speed;
      }
      if (!Number.isFinite(lo)) { lo = 0; hi = 0; }
      if (sMin === undefined) sMin = lo;
      if (sMax === undefined) sMax = hi;
    }
    if (sMax < sMin) [sMin, sMax] = [sMax, sMin];
    this._speedMin = sMin;
    this._speedMax = sMax;

    // RTC：显式 null = 绝对坐标；缺省自动 = origin 的世界坐标。
    this._rtc = options.rtcOrigin === undefined
      ? this._place(this._originLon, this._originLat, this._originAltM)
      : options.rtcOrigin;
  }

  /** 实际使用的风场 origin。 */
  get origin(): { lon: number; lat: number; altM: number } {
    return { lon: this._originLon, lat: this._originLat, altM: this._originAltM };
  }

  /** RTC 几何中心（null = 顶点绝对世界坐标）。 */
  get rtcOrigin(): [number, number, number] | null { return this._rtc; }

  /** 速度档数。 */
  get bucketCount(): number { return this._buckets; }

  /** 纳入渲染的箭头数（退化跳过不计）。 */
  get arrowCount(): number { return this._arrowCount; }

  /** 跳过的退化箭头数（零水平风速等）。 */
  get skippedCount(): number { return this._skippedCount; }

  /** 速度范围（分桶 + 长度映射的基准）。 */
  get speedMin(): number { return this._speedMin; }
  get speedMax(): number { return this._speedMax; }

  /**
   * speed → 桶号：[sMin, sMax] 均分，floor 落档、两端 clamp。
   * 跨度 < 1e-12（全部同速）时恒为 0 号档。
   */
  static speedToBucket(
    speed: number, sMin: number, sMax: number, buckets = WindFieldBuilder.DEFAULT_BUCKETS,
  ): number {
    if (buckets <= 1) return 0;
    const span = sMax - sMin;
    if (!(span > 1e-12)) return 0;
    const t = ((speed - sMin) / span) * buckets;
    return Math.max(0, Math.min(buckets - 1, Math.floor(t)));
  }

  /** speed → 箭头长度（米）：speed_max 线性映射到 lenMax，两端 clamp。 */
  static speedToLengthM(
    speed: number, sMax: number,
    lenMin = WindFieldBuilder.LENGTH_MIN_M, lenMax = WindFieldBuilder.LENGTH_MAX_M,
  ): number {
    if (!(sMax > WindFieldBuilder.SPEED_EPS)) return lenMin;
    return Math.max(lenMin, Math.min(lenMax, (speed / sMax) * lenMax));
  }

  /** 档 i 的单色 [r,g,b]（越界 clamp）。 */
  static bucketColor(i: number): [number, number, number] {
    const colors = WindFieldBuilder.BUCKET_COLORS;
    const k = Math.max(0, Math.min(colors.length - 1, Math.round(i)));
    return [colors[k]![0]!, colors[k]![1]!, colors[k]![2]!];
  }

  /**
   * 输出速度分桶的纯数据（无 GPU 依赖）。只返回**非空**档，按档序排列；
   * 结果缓存（同一 builder 重复调用返回同一数组）。
   */
  buildBuckets(): WindFieldBucket[] {
    if (this._bucketsOut) return this._bucketsOut;

    const nb = this._buckets;
    const stride = GeometryPipeline.VERTEX_STRIDE;
    // 第一遍：逐箭头定档 + 统计（退化箭头跳过）。
    const perBucket: { arrow: WindFieldArrow; lengthM: number }[][] =
      Array.from({ length: nb }, () => []);
    for (const arrow of this._arrows) {
      const enu = arrow.enu, vec = arrow.vec;
      if (!enu || !vec || enu.length < 3 || vec.length < 2) { this._skippedCount++; continue; }
      if (!(Math.hypot(vec[0]!, vec[1]!) > WindFieldBuilder.SPEED_EPS)) { this._skippedCount++; continue; }
      const b = WindFieldBuilder.speedToBucket(arrow.speed, this._speedMin, this._speedMax, nb);
      perBucket[b]!.push({
        arrow,
        lengthM: WindFieldBuilder.speedToLengthM(arrow.speed, this._speedMax, this._lenMin, this._lenMax),
      });
      this._arrowCount++;
    }

    // 第二遍：每档预分配 + 旋转平移写顶点。
    const out: WindFieldBucket[] = [];
    const span = this._speedMax - this._speedMin;
    const bucketW = span > 1e-12 ? span / nb : 0;
    for (let b = 0; b < nb; b++) {
      const items = perBucket[b]!;
      if (items.length === 0) continue;
      const vertCount = items.length * WIND_ARROW_VERTEX_COUNT;
      const vertices = new Float32Array(vertCount * stride);
      const indices = new Uint32Array(items.length * WIND_ARROW_INDEX_COUNT);

      let vOff = 0;      // 顶点 float 游标
      let iOff = 0;      // 索引游标
      let vertBase = 0;  // 本档内箭头顶点基址
      for (const it of items) {
        const g = WindArrowGeometry.create({ lengthM: it.lengthM });
        this._bakeArrow(g, it.arrow, vertices, vOff, indices, iOff, vertBase);
        vOff += g.vertices.length;
        iOff += g.indices.length;
        vertBase += g.vertexCount;
      }

      out.push({
        bucket: b,
        color: WindFieldBuilder.bucketColor(b),
        speedLo: this._speedMin + bucketW * b,
        speedHi: b === nb - 1 ? this._speedMax : this._speedMin + bucketW * (b + 1),
        geometry: { vertices, indices, vertexCount: vertCount, indexCount: indices.length },
        arrowCount: items.length,
      });
    }
    this._bucketsOut = out;
    return out;
  }

  /**
   * 构建 `Primitive` 列表（每非空速度档一个，单色材质一次 drawIndexed）。
   * @param context WebGPU 上下文（`viewer.context`）
   */
  build(context: Context): Primitive[] {
    const out: Primitive[] = [];
    for (const b of this.buildBuckets()) {
      if (b.geometry.indexCount === 0 || b.geometry.vertices.byteLength === 0) continue;
      const [r, g, bl] = b.color;
      out.push(new Primitive(context, {
        vertices: b.geometry.vertices,
        indices:  b.geometry.indices,
        color:    new Color(r, g, bl, 1),
        ...(this._rtc ? { origin: this._rtc } : {}),
      }));
    }
    return out;
  }

  // ── Private ───────────────────────────────────────────────────────────────

  /** 单箭头局部顶点/索引 → 世界系写进档缓冲（position/normal 同步旋转平移）。 */
  private _bakeArrow(
    g: InterleavedGeometryBuffers, arrow: WindFieldArrow,
    vertices: Float32Array, vOff: number,
    indices: Uint32Array, iOff: number, vertBase: number,
  ): void {
    const vec = arrow.vec, enu = arrow.enu;
    const { latDeg, lonDeg, altM } = this._enuToLonLatAlt(enu[0]!, enu[1]!, enu[2] ?? 0);

    // 格点 ENU 框架（真方向空间，单位向量）
    const lonR = lonDeg * Math.PI / 180;
    const latR = latDeg * Math.PI / 180;
    const cLo = Math.cos(lonR), sLo = Math.sin(lonR);
    const cLa = Math.cos(latR), sLa = Math.sin(latR);
    const east: [number, number, number]  = [-sLo, cLo, 0];
    const north: [number, number, number] = [-sLa * cLo, -sLa * sLo, cLa];
    const up: [number, number, number]    = [cLa * cLo, cLa * sLo, sLa];

    // 水平风向（归一化）→ 箭头 +Y；r = f × up 使 (r, f, up) 右手正交。
    const vh = Math.hypot(vec[0]!, vec[1]!);
    const f: [number, number, number] = [
      (east[0] * vec[0]! + north[0] * vec[1]!) / vh,
      (east[1] * vec[0]! + north[1] * vec[1]!) / vh,
      (east[2] * vec[0]! + north[2] * vec[1]!) / vh,
    ];
    const r = cross3(f, up);

    // 格点世界坐标（米制 ENU 偏移已折算进经纬高）
    const base = this._place(lonDeg, latDeg, altM);

    const R = WindFieldBuilder.EARTH_A;
    const rho = this._flattened ? WindFieldBuilder.FLATTEN_RHO : 1;
    const stride = GeometryPipeline.VERTEX_STRIDE;
    const src = g.vertices;
    for (let v = 0; v < g.vertexCount; v++) {
      const s = v * stride;
      const lx = src[s]!,   ly = src[s + 1]!, lz = src[s + 2]!;
      const nx = src[s + 3]!, ny = src[s + 4]!, nz = src[s + 5]!;
      // 局部 (x,y,z) 米 → 世界方向偏移（米）→ 归一化（压扁系 y/z × ρ）
      const ox = r[0] * lx + f[0] * ly + up[0] * lz;
      const oy = r[1] * lx + f[1] * ly + up[1] * lz;
      const oz = r[2] * lx + f[2] * ly + up[2] * lz;
      // 法线随姿态旋转（正交旋转保模长，残余 f32 误差再归一）
      let wnx = r[0] * nx + f[0] * ny + up[0] * nz;
      let wny = r[1] * nx + f[1] * ny + up[1] * nz;
      let wnz = r[2] * nx + f[2] * ny + up[2] * nz;
      const nl = Math.hypot(wnx, wny, wnz) || 1;
      wnx /= nl; wny /= nl; wnz /= nl;

      const d = vOff + s;
      const wx = base[0] + ox * rho / R;
      const wy = base[1] + oy * rho / R;
      const wz = base[2] + oz * rho / R;
      if (this._rtc) {
        vertices[d] = wx - this._rtc[0]; vertices[d + 1] = wy - this._rtc[1]; vertices[d + 2] = wz - this._rtc[2];
      } else {
        vertices[d] = wx; vertices[d + 1] = wy; vertices[d + 2] = wz;
      }
      vertices[d + 3] = wnx; vertices[d + 4] = wny; vertices[d + 5] = wnz;
      vertices[d + 6] = src[s + 6]!; vertices[d + 7] = src[s + 7]!; // uv 透传
    }
    for (let i = 0; i < g.indices.length; i++) {
      indices[iOff + i] = vertBase + g.indices[i]!;
    }
  }

  /** ENU 米制偏移（相对 origin）→ 经纬高（度/米，WGS84 局部精确系数）。 */
  private _enuToLonLatAlt(e: number, n: number, u: number): { latDeg: number; lonDeg: number; altM: number } {
    const A = WindFieldBuilder.EARTH_A;
    const B = WindFieldBuilder.EARTH_B;
    const e2 = 1 - (B * B) / (A * A);
    const lat0 = this._originLat * Math.PI / 180;
    const s2 = 1 - e2 * Math.sin(lat0) * Math.sin(lat0);
    const mPerDegLat = A * (1 - e2) / (s2 * Math.sqrt(s2)) * (Math.PI / 180);
    const mPerDegLon = A / Math.sqrt(s2) * Math.cos(lat0) * (Math.PI / 180);
    return {
      lonDeg: this._originLon + e / mPerDegLon,
      latDeg: this._originLat + n / mPerDegLat,
      altM: this._originAltM + u,
    };
  }

  /** 经纬高 → 归一化 ECEF 世界坐标 [x,y,z]（按 flattened 选映射）。 */
  private _place(lonDeg: number, latDeg: number, hM: number): [number, number, number] {
    if (!this._flattened) {
      const p = Cartesian3.fromDegreesNormalized(lonDeg, latDeg, hM);
      return [p.x, p.y, p.z];
    }
    const rho = WindFieldBuilder.FLATTEN_RHO;
    const lon = lonDeg * Math.PI / 180, lat = latDeg * Math.PI / 180;
    const s = 1.0 + hM / WindFieldBuilder.EARTH_A;
    const cLat = Math.cos(lat);
    return [cLat * Math.cos(lon) * s, cLat * Math.sin(lon) * rho * s, Math.sin(lat) * rho * s];
  }
}

/** 3 维叉积（局部工具，避免为几处叉积引入 Cartesian3 分配）。 */
function cross3(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/**
 * 一步式入口：wind-field/v1 JSON → `Primitive[]`（每非空速度档一个，
 * 调用方逐个 `scene.addPrimitive` 入场）。等价于
 * `new WindFieldBuilder(field, opts).build(opts.context)`。
 */
export function buildWindField(field: WindFieldData, opts: WindFieldBuildOptions): Primitive[] {
  const { context, ...builderOpts } = opts;
  return new WindFieldBuilder(field, builderOpts).build(context);
}
