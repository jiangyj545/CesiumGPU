import { describe, expect, it } from 'vitest';
import { WindFieldBuilder } from '../../src/geometry/WindFieldBuilder';
import { WIND_ARROW_VERTEX_COUNT, WIND_ARROW_INDEX_COUNT } from '../../src/geometry/WindArrowGeometry';
import { GeometryPipeline } from '../../src/geometry/GeometryPipeline';
import type { WindFieldData, WindFieldArrow } from '../../src/geometry/WindFieldBuilder';

/**
 * WindFieldBuilder 纯数据单测（无 GPU 依赖，cityBlockInstanceIds 同模式）：
 * wind-field/v1 → 速度分桶（8 档单色 Primitive）的纯数据层。
 * 契约要点：分桶边界、长度映射、退化箭头跳过、每档顶点数 = 箭头数×32、
 * RTC 顶点 + origin 加和回世界系（Primitive.boundingSphere 同约定，
 * Primitive.ts:260-264）。
 */

const ORIGIN = { lon: 114.405, lat: 30.501, alt_m: 23 };
const STRIDE = GeometryPipeline.VERTEX_STRIDE;

/** 合成风场：speeds 逐格点（enu 沿东向排开），vec 单位东风。 */
function makeField(speeds: number[], withStats = true): WindFieldData {
  const arrows: WindFieldArrow[] = speeds.map((speed, idx) => ({
    i: idx, j: 0, k: 0,
    enu: [idx * 100, (idx % 3) * 100, 30],
    vec: [1, 0, 0],
    speed,
  }));
  return {
    schema: 'wind-field/v1',
    origin: ORIGIN,
    grid: { nx: speeds.length, ny: 1, dx_m: 100, dy_m: 100, heights_m: [30] },
    arrows,
    stats: withStats
      ? { count: speeds.length, speed_min: Math.min(...speeds), speed_max: Math.max(...speeds) }
      : undefined,
  };
}

describe('WindFieldBuilder 分桶与长度映射', () => {
  it('speedToBucket 边界值：sMin→0、跨档、sMax→末档、越界 clamp', () => {
    const sMin = 0.8, sMax = 5.6, w = (sMax - sMin) / 8;   // 档宽 0.6
    expect(WindFieldBuilder.speedToBucket(sMin, sMin, sMax)).toBe(0);
    // 档内（边界上下探千分之一档宽，避开 f64 临界）
    expect(WindFieldBuilder.speedToBucket(sMin + w * 0.9, sMin, sMax)).toBe(0);
    expect(WindFieldBuilder.speedToBucket(sMin + w * 1.1, sMin, sMax)).toBe(1);
    expect(WindFieldBuilder.speedToBucket(sMin + w * 3.5, sMin, sMax)).toBe(3);
    expect(WindFieldBuilder.speedToBucket(sMax - w * 0.1, sMin, sMax)).toBe(7);
    expect(WindFieldBuilder.speedToBucket(sMax, sMin, sMax)).toBe(7);
    expect(WindFieldBuilder.speedToBucket(sMax + 10, sMin, sMax)).toBe(7);  // 上越界 clamp
    expect(WindFieldBuilder.speedToBucket(sMin - 10, sMin, sMax)).toBe(0);  // 下越界 clamp
  });

  it('speedToBucket 跨度归零（全同速）恒为 0 号档', () => {
    expect(WindFieldBuilder.speedToBucket(3.3, 3.3, 3.3)).toBe(0);
  });

  it('speedToLengthM：speed_max→45 线性、两端 clamp 8..45、sMax 无效回退 lenMin', () => {
    expect(WindFieldBuilder.speedToLengthM(5.6, 5.6)).toBe(45);
    expect(WindFieldBuilder.speedToLengthM(2.8, 5.6)).toBeCloseTo(22.5, 9);
    expect(WindFieldBuilder.speedToLengthM(0.01, 5.6)).toBe(8);    // 下 clamp
    expect(WindFieldBuilder.speedToLengthM(99, 5.6)).toBe(45);     // 上 clamp
    expect(WindFieldBuilder.speedToLengthM(3, 0)).toBe(8);         // sMax 无效
    expect(WindFieldBuilder.speedToLengthM(5.6, 5.6, 10, 30)).toBe(30);
  });

  it('色带：8 档、蓝起红终、越界 clamp', () => {
    const c = WindFieldBuilder.BUCKET_COLORS;
    expect(c).toHaveLength(8);
    expect(c[0]).toEqual([0.2, 0.4, 1.0]);    // 蓝
    expect(c[7]).toEqual([1.0, 0.2, 0.1]);    // 红
    expect(WindFieldBuilder.bucketColor(-1)).toEqual(c[0]);
    expect(WindFieldBuilder.bucketColor(99)).toEqual(c[7]);
    expect(c[7]![0]!).toBeGreaterThan(c[0]![0]!);   // 冷暖方向：红分量递增
  });

  it('stats 缺省 → 从 arrows 现算速度范围', () => {
    const b = new WindFieldBuilder(makeField([1.5, 4.2, 0.9], false));
    expect(b.speedMin).toBeCloseTo(0.9, 12);
    expect(b.speedMax).toBeCloseTo(4.2, 12);
  });
});

describe('WindFieldBuilder buildBuckets 汇总', () => {
  it('每档顶点数 = 箭头数×32，索引 = ×42，索引值在顶点范围内', () => {
    const speeds = [0.8, 1.4, 2.2, 3.0, 3.7, 4.4, 5.0, 5.6, 2.5, 4.9];  // 10 格点
    const b = new WindFieldBuilder(makeField(speeds));
    const buckets = b.buildBuckets();
    expect(b.arrowCount).toBe(10);
    expect(b.skippedCount).toBe(0);

    let vertSum = 0, arrowSum = 0;
    for (const bk of buckets) {
      expect(bk.geometry.vertexCount).toBe(bk.arrowCount * WIND_ARROW_VERTEX_COUNT);
      expect(bk.geometry.vertexCount % WIND_ARROW_VERTEX_COUNT).toBe(0);
      expect(bk.geometry.indexCount).toBe(bk.arrowCount * WIND_ARROW_INDEX_COUNT);
      expect(bk.geometry.vertices.length).toBe(bk.geometry.vertexCount * STRIDE);
      for (const idx of bk.geometry.indices) {
        expect(idx).toBeGreaterThanOrEqual(0);
        expect(idx).toBeLessThan(bk.geometry.vertexCount);
      }
      vertSum += bk.geometry.vertexCount;
      arrowSum += bk.arrowCount;
    }
    expect(vertSum).toBe(10 * WIND_ARROW_VERTEX_COUNT);
    expect(arrowSum).toBe(10);
  });

  it('分桶落位：speed 落在对应档 [speedLo, speedHi] 内，首档含 sMin、末档含 sMax', () => {
    const speeds = [0.8, 2.0, 3.0, 5.6];    // → 档 0 / 2 / 3 / 7（手算见边界用例）
    const b = new WindFieldBuilder(makeField(speeds));
    const buckets = b.buildBuckets();
    expect(buckets.map((x) => x.bucket)).toEqual([0, 2, 3, 7]);
    for (const bk of buckets) {
      expect(bk.speedLo).toBeGreaterThanOrEqual(b.speedMin);
      expect(bk.speedHi).toBeLessThanOrEqual(b.speedMax + 1e-12);
      const speed = speeds[buckets.indexOf(bk)]!;
      expect(speed).toBeGreaterThanOrEqual(bk.speedLo);
      expect(speed).toBeLessThanOrEqual(bk.speedHi);
    }
    expect(buckets[0]!.speedLo).toBeCloseTo(0.8, 12);
    expect(buckets[buckets.length - 1]!.speedHi).toBeCloseTo(5.6, 12);
  });

  it('空 vec(0,0,0) / 纯垂直风 → 跳过（skippedCount，不入任何档）', () => {
    const field = makeField([2.0, 3.0, 4.0]);
    field.arrows[1]!.vec = [0, 0, 0];          // 零矢量
    field.arrows[2]!.vec = [0, 0, 5];          // 纯垂直（水平化后无方向）
    const b = new WindFieldBuilder(field);
    const buckets = b.buildBuckets();
    expect(b.arrowCount).toBe(1);
    expect(b.skippedCount).toBe(2);
    let total = 0;
    for (const bk of buckets) total += bk.arrowCount;
    expect(total).toBe(1);
  });

  it('全同速 → 单档输出（span 0 恒 0 号档）', () => {
    const b = new WindFieldBuilder(makeField([3.3, 3.3, 3.3]));
    const buckets = b.buildBuckets();
    expect(buckets).toHaveLength(1);
    expect(buckets[0]!.bucket).toBe(0);
    expect(buckets[0]!.arrowCount).toBe(3);
  });

  it('空箭头表 → 空档列表', () => {
    const b = new WindFieldBuilder(makeField([]));
    expect(b.buildBuckets()).toEqual([]);
    expect(b.arrowCount).toBe(0);
  });
});

describe('WindFieldBuilder 放置与 RTC', () => {
  const RHO = WindFieldBuilder.FLATTEN_RHO;
  const R = WindFieldBuilder.EARTH_A;

  it('RTC 默认开启：顶点相对中心存储，加回 rtcOrigin 回世界系（boundingSphere 约定）', () => {
    const field = makeField([2.0, 5.6, 3.3]);
    const rtc = new WindFieldBuilder(field, { flattened: true });                  // 默认自动 RTC
    const abs = new WindFieldBuilder(field, { flattened: true, rtcOrigin: null }); // 绝对坐标
    expect(rtc.rtcOrigin).not.toBeNull();
    const o = rtc.rtcOrigin!;

    const rb = rtc.buildBuckets();
    const ab = abs.buildBuckets();
    expect(rb).toHaveLength(ab.length);
    // 逐顶点：rtc 顶点 + origin ≈ 绝对顶点（f32 量化 < 2 ulp(1.0)，留 5e-7）
    for (let i = 0; i < rb.length; i++) {
      const rv = rb[i]!.geometry.vertices, av = ab[i]!.geometry.vertices;
      expect(rv.length).toBe(av.length);
      for (let v = 0; v < rv.length; v += STRIDE) {
        for (let c = 0; c < 3; c++) {
          expect(rv[v + c]! + o[c]!).toBeCloseTo(av[v + c]!, 6);
        }
      }
    }
  });

  it('rtcOrigin=null 显式关闭 → 顶点即绝对世界坐标（?rtc=0 路径）', () => {
    const b = new WindFieldBuilder(makeField([3.0]), { rtcOrigin: null });
    expect(b.rtcOrigin).toBeNull();
    const bk = b.buildBuckets()[0]!;
    // 未压扁系下位移后的顶点幅度 ≈ 1（归一化地球），远大于箭头米级尺度
    let maxR = 0;
    for (let v = 0; v < bk.geometry.vertices.length; v += STRIDE) {
      const r = Math.hypot(
        bk.geometry.vertices[v]!, bk.geometry.vertices[v + 1]!, bk.geometry.vertices[v + 2]!);
      maxR = Math.max(maxR, r);
    }
    expect(maxR).toBeGreaterThan(0.99);
    expect(maxR).toBeLessThan(1.01);
  });

  it('箭头指向 = 水平风向（vec 东风 → 尖端朝东），长度 = speed_max 映射', () => {
    const b = new WindFieldBuilder(makeField([5.6]), { flattened: true });
    const bk = b.buildBuckets()[0]!;
    // 顶点相对中心的最大水平偏移（扣除 up 分量）应 ≈ lengthMax/R（尖端）
    const lat = ORIGIN.lat * Math.PI / 180, lon = ORIGIN.lon * Math.PI / 180;
    const up = [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)] as const;
    // 压扁系下的东向（真 ECEF 东向量过 diag(1,ρ,ρ)）
    const eastFlat = [-Math.sin(lon), Math.cos(lon) * RHO, 0];
    const eastLen = Math.hypot(eastFlat[0], eastFlat[1], eastFlat[2]);
    let bestDot = -Infinity, bestMag = 0;
    for (let v = 0; v < bk.geometry.vertices.length; v += STRIDE) {
      const dx = bk.geometry.vertices[v]!, dy = bk.geometry.vertices[v + 1]!, dz = bk.geometry.vertices[v + 2]!;
      const radial = dx * up[0] + dy * up[1] + dz * up[2];
      const hx = dx - radial * up[0], hy = dy - radial * up[1], hz = dz - radial * up[2];
      const mag = Math.hypot(hx, hy, hz);
      if (mag > bestMag) {
        bestMag = mag;
        bestDot = (hx * eastFlat[0] + hy * eastFlat[1] + hz * eastFlat[2]) / (mag * eastLen);
      }
    }
    expect(bestMag).toBeGreaterThan(0);       // f32 下 45m ≈ 7.06e-6
    // 压扁系 y/z×ρ 使水平模长比 45/R 小 <0.3%（合法缩放），相对容差 1%
    expect(Math.abs(bestMag - 45 / R)).toBeLessThan((45 / R) * 0.01);
    expect(bestDot).toBeGreaterThan(0.99);    // 最大水平偏移沿东 = 风向
  });

  it('缺 origin（field 与 options 皆无）→ 抛错', () => {
    const bad = { schema: 'wind-field/v1', arrows: [] } as unknown as WindFieldData;
    expect(() => new WindFieldBuilder(bad)).toThrow();
  });
});
