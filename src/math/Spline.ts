import { Cartesian3 } from './Cartesian3';

/**
 * Evaluate the cubic Hermite basis at parameter t in [0, 1].
 * p(u) = h00*p0 + h10*m0 + h01*p1 + h11*m1
 */
function hermite(
  p0: Cartesian3,
  m0: Cartesian3,
  p1: Cartesian3,
  m1: Cartesian3,
  u: number,
  result: Cartesian3
): Cartesian3 {
  const u2 = u * u;
  const u3 = u2 * u;
  const h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
  const h10 = u3 - 2.0 * u2 + u;
  const h01 = -2.0 * u3 + 3.0 * u2;
  const h11 = u3 - u2;

  result.x = h00 * p0.x + h10 * m0.x + h01 * p1.x + h11 * m1.x;
  result.y = h00 * p0.y + h10 * m0.y + h01 * p1.y + h11 * m1.y;
  result.z = h00 * p0.z + h10 * m0.z + h01 * p1.z + h11 * m1.z;
  return result;
}

/** Options for constructing a {@link HermiteSpline}. */
export interface HermiteSplineOptions {
  /** Strictly increasing time values. */
  times: number[];
  /** Control points, one per time. */
  points: Cartesian3[];
  /**
   * Tangent at each control point (parameter-space derivative).
   * If omitted, Catmull-Rom style tangents are computed automatically.
   */
  tangents?: Cartesian3[];
}

/**
 * HermiteSpline - cubic Hermite interpolation through a set of
 * {@link Cartesian3} control points.
 */
export class HermiteSpline {
  readonly times: number[];
  readonly points: Cartesian3[];
  readonly tangents: Cartesian3[];

  constructor(options: HermiteSplineOptions) {
    const { times, points } = options;

    if (times.length !== points.length) {
      throw new Error('HermiteSpline: times.length and points.length must match.');
    }
    if (times.length < 2) {
      throw new Error('HermiteSpline: at least two control points are required.');
    }
    for (let i = 1; i < times.length; i++) {
      if (times[i] <= times[i - 1]) {
        throw new Error('HermiteSpline: times must be strictly increasing.');
      }
    }

    this.times = times;
    this.points = points;
    this.tangents = options.tangents ?? HermiteSpline.computeTangents(points);
  }

  /** First time value. */
  get startTime(): number {
    return this.times[0];
  }

  /** Last time value. */
  get stopTime(): number {
    return this.times[this.times.length - 1];
  }

  /**
   * Catmull-Rom tangents: (p[i+1] - p[i-1]) / 2 for interior points,
   * one-sided differences at the endpoints.
   */
  static computeTangents(points: Cartesian3[]): Cartesian3[] {
    const n = points.length;
    const tangents: Cartesian3[] = new Array(n);

    tangents[0] = Cartesian3.subtract(points[1], points[0], new Cartesian3());
    for (let i = 1; i < n - 1; i++) {
      const t = Cartesian3.subtract(points[i + 1], points[i - 1], new Cartesian3());
      tangents[i] = Cartesian3.multiplyByScalar(t, 0.5, t);
    }
    tangents[n - 1] = Cartesian3.subtract(points[n - 1], points[n - 2], new Cartesian3());
    return tangents;
  }

  /**
   * Locate the segment containing `time` via binary search.
   * Time outside [startTime, stopTime] is clamped.
   * @returns the segment start index `i` (so the segment spans
   *   `[times[i], times[i+1]]`) and the local parameter `t` in [0, 1].
   */
  findSegment(time: number): { index: number; t: number } {
    const times = this.times;
    const n = times.length;

    if (time <= times[0]) return { index: 0, t: 0.0 };
    if (time >= times[n - 1]) return { index: n - 2, t: 1.0 };

    // Binary search: last i with times[i] <= time.
    let low = 0;
    let high = n - 1;
    while (high - low > 1) {
      const mid = (low + high) >>> 1;
      if (times[mid] <= time) low = mid;
      else high = mid;
    }
    return { index: low, t: (time - times[low]) / (times[high] - times[low]) };
  }

  /**
   * Evaluate the spline at `time` with cubic Hermite interpolation.
   * @param time - Time value, clamped to [startTime, stopTime].
   * @param result - Optional output value.
   */
  evaluate(time: number, result?: Cartesian3): Cartesian3 {
    if (!result) result = new Cartesian3();
    const { index, t } = this.findSegment(time);
    return hermite(
      this.points[index],
      this.tangents[index],
      this.points[index + 1],
      this.tangents[index + 1],
      t,
      result
    );
  }
}

/**
 * CatmullRomSpline - a {@link HermiteSpline} whose tangents are computed
 * automatically as (p[i+1] - p[i-1]) / 2 (one-sided at the endpoints),
 * producing a C1-continuous curve that passes through every control point.
 */
export class CatmullRomSpline extends HermiteSpline {
  constructor(options: { times: number[]; points: Cartesian3[] }) {
    super({
      times: options.times,
      points: options.points,
      tangents: HermiteSpline.computeTangents(options.points),
    });
  }
}

export default HermiteSpline;
