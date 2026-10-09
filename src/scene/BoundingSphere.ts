import { Cartesian3 } from '../math/Cartesian3';

/**
 * BoundingSphere - a world-space sphere. `center` is in normalised ECEF
 * (engine units, Earth radius = 1); `radius` is in **metres** (matching the
 * `HeadingPitchRange.range` convention used by `zoomTo` /
 * `flyToBoundingSphere`).
 */
export class BoundingSphere {
  constructor(
    public center: Cartesian3,
    public radius: number
  ) {}

  /**
   * Bounding sphere of a point cloud (Ritter's algorithm): seed from the
   * extremes along the dominant axis, then grow to contain every point.
   * Points are in normalised ECEF; the returned radius is in metres.
   */
  static fromPoints(positions: readonly Cartesian3[], result?: BoundingSphere): BoundingSphere {
    if (positions.length === 0) {
      const s = result ?? new BoundingSphere(new Cartesian3(), 0);
      s.center = new Cartesian3();
      s.radius = 0;
      return s;
    }
    const axis = axisName(
      ([0, 1, 2] as const).reduce((worst, a) =>
        axisExtent(positions, a) > axisExtent(positions, worst) ? a : worst
      )
    );
    const extremes = positions.reduce(
      (acc, p) => {
        if (acc.lo === null || p[axis] < acc.lo[axis]) acc.lo = p;
        if (acc.hi === null || p[axis] > acc.hi[axis]) acc.hi = p;
        return acc;
      },
      { lo: null as Cartesian3 | null, hi: null as Cartesian3 | null }
    );
    const lo = extremes.lo ?? positions[0];
    const hi = extremes.hi ?? positions[0];
    let center = new Cartesian3(
      (lo.x + hi.x) / 2,
      (lo.y + hi.y) / 2,
      (lo.z + hi.z) / 2
    );
    let radius = Cartesian3.distance(lo, hi) / 2;
    for (const p of positions) {
      const d = Cartesian3.distance(p, center);
      if (d > radius) {
        const k = (d - radius) / 2;
        center = new Cartesian3(
          center.x + ((p.x - center.x) / d) * k,
          center.y + ((p.y - center.y) / d) * k,
          center.z + ((p.z - center.z) / d) * k
        );
        radius = (radius + d) / 2;
      }
    }
    const s = result ?? new BoundingSphere(center, 0);
    s.center = center;
    s.radius = radius * 6378137.0; // normalised units → metres
    return s;
  }
}

function axisExtent(positions: readonly Cartesian3[], axis: 0 | 1 | 2): number {
  const key = axisName(axis);
  let min = Infinity;
  let max = -Infinity;
  for (const p of positions) {
    if (p[key] < min) min = p[key];
    if (p[key] > max) max = p[key];
  }
  return max - min;
}

function axisName(axis: number): 'x' | 'y' | 'z' {
  return axis === 0 ? 'x' : axis === 1 ? 'y' : 'z';
}

export default BoundingSphere;
