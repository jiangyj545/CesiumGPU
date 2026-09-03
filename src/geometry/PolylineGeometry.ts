import { Cartesian3 } from '../math/Cartesian3';
import { Measure } from '../analysis/Measure';
import { GeometryPipeline } from './GeometryPipeline';
import type { InterleavedGeometryBuffers } from './GeometryPipeline';

export interface PolylineGeometryOptions {
  /**
   * Full ribbon width, in normalized world units (Earth equatorial radius = 1).
   * Note: 1 unit ~ 6378 km, so the default 0.001 ~ 6.4 km on the surface.
   * Default: 0.001
   */
  width?: number;
  /**
   * Vertex height mode.
   * - `'none'` (default): positions are used as-is.
   * - `'clampToGround'`: every vertex is snapped onto the terrain surface
   *   before the ribbon is built, and each segment is densified along the
   *   great circle (spherical step 2 degrees, at least 32 subdivisions) so
   *   the line follows valleys between vertices instead of cutting through
   *   them.
   *
   * Requires {@link sampleHeight}; without it this behaves as `'none'`.
   */
  heightReference?: 'none' | 'clampToGround';
  /**
   * Terrain height sampler `(lonDeg, latDeg) => heightMetres`, typically
   * bound to `globe.terrainLayer.sampleHeight` by the caller. The geometry
   * module never imports the terrain stack — the callback keeps layering
   * clean. Required by `heightReference: 'clampToGround'`.
   */
  sampleHeight?: (lon: number, lat: number) => number;
}

/** Squared length below which a segment is treated as degenerate. */
const DEGENERATE_EPSILON_SQ = 1e-20;
/** Densification step for clamped polylines: at most 2 degrees of arc per sub-vertex. */
const DENSIFY_STEP_DEG = 2.0;
/** Minimum number of subdivisions per segment when clamping (bounds sag between vertices). */
const MIN_SUBDIVISIONS = 32;

/**
 * PolylineGeometry - Expands a polyline into a camera-independent triangle
 * ribbon, in normalized ECEF space (Earth equatorial radius = 1.0).
 *
 * Each segment emits 4 vertices and 2 triangles; left/right offsets are
 * width/2 along a side vector that lies in the surface tangent plane
 * (perpendicular to both the segment tangent and the radial direction),
 * so the ribbon hugs the globe regardless of the view direction.
 *
 * Per-vertex uv: `u` accumulates arc length along the line (normalized world
 * units, usable by dash/flow materials); `v` is 0 on the left edge, 1 on the
 * right edge.
 */
export class PolylineGeometry {
  /**
   * Build the ribbon geometry for a polyline.
   *
   * @param positions - polyline points in normalized ECEF (Earth radius = 1)
   * @param options   - ribbon options
   */
  static fromCartesianPositions(
    positions: Cartesian3[],
    options: PolylineGeometryOptions = {}
  ): InterleavedGeometryBuffers {
    // Terrain clamping (incl. along-line densification); no-op unless
    // heightReference === 'clampToGround' and a sampler was provided.
    positions = PolylineGeometry._clampForTerrain(positions, options);

    const n = positions.length;
    if (n < 2) {
      return GeometryPipeline.emptyBuffers();
    }

    const width = options.width ?? 0.001;
    const halfWidth = width / 2;

    // Per-point tangents (adjacent-segment average for interior points).
    const tangents = PolylineGeometry._computeTangents(positions);
    if (tangents.length !== n) {
      return GeometryPipeline.emptyBuffers();
    }

    // Per-point side vectors and cumulative arc length.
    const sides: Cartesian3[] = new Array(n);
    const cumulative = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      sides[i] = PolylineGeometry._computeSide(positions[i], tangents[i]);
      if (i > 0) {
        cumulative[i] = cumulative[i - 1] + Cartesian3.distance(positions[i - 1], positions[i]);
      }
    }

    const segmentCount = n - 1;
    const vertexCount = segmentCount * 4;
    const indexCount = segmentCount * 6;

    const positionsOut = new Float32Array(vertexCount * 3);
    const normalsOut = new Float32Array(vertexCount * 3);
    const uvsOut = new Float32Array(vertexCount * 2);
    const indices = new Uint32Array(indexCount);

    let pi = 0;
    let ti = 0;
    let ii = 0;

    for (let seg = 0; seg < segmentCount; seg++) {
      const i0 = seg;
      const i1 = seg + 1;
      const base = seg * 4;

      // Corner order: [L0, R0, L1, R1]; side = cross(tangent, radial).
      const left0 = PolylineGeometry._offset(positions[i0], sides[i0], -halfWidth);
      const right0 = PolylineGeometry._offset(positions[i0], sides[i0], halfWidth);
      const left1 = PolylineGeometry._offset(positions[i1], sides[i1], -halfWidth);
      const right1 = PolylineGeometry._offset(positions[i1], sides[i1], halfWidth);
      const corners = [left0, right0, left1, right1];
      const radials = [
        PolylineGeometry._radial(positions[i0]),
        PolylineGeometry._radial(positions[i0]),
        PolylineGeometry._radial(positions[i1]),
        PolylineGeometry._radial(positions[i1]),
      ];
      const vCoords = [0, 1, 0, 1];
      const uCoords = [cumulative[i0], cumulative[i0], cumulative[i1], cumulative[i1]];

      for (let c = 0; c < 4; c++) {
        const p = corners[c];
        const radial = radials[c];

        positionsOut[pi] = p.x;
        positionsOut[pi + 1] = p.y;
        positionsOut[pi + 2] = p.z;
        normalsOut[pi] = radial.x;
        normalsOut[pi + 1] = radial.y;
        normalsOut[pi + 2] = radial.z;
        uvsOut[ti] = uCoords[c];
        uvsOut[ti + 1] = vCoords[c];
        pi += 3;
        ti += 2;
      }

      // Triangles (L0, R0, L1) and (R0, R1, L1): normals face outward.
      indices[ii++] = base;
      indices[ii++] = base + 1;
      indices[ii++] = base + 2;
      indices[ii++] = base + 1;
      indices[ii++] = base + 3;
      indices[ii++] = base + 2;
    }

    return {
      vertices: GeometryPipeline.interleave(positionsOut, normalsOut, uvsOut),
      indices,
      vertexCount,
      indexCount,
    };
  }

  /**
   * Terrain clamping pass for `heightReference: 'clampToGround'`: every
   * input vertex is snapped onto the terrain surface, and each segment is
   * densified along the great circle so the ribbon follows the terrain
   * between vertices (a straight chord would tunnel through valleys).
   *
   * Subdivision count per segment = max(ceil(spherical angle / 2°), 32).
   * No-op when the options leave the default `'none'` mode or omit the
   * sampler; the input array is never modified.
   */
  private static _clampForTerrain(
    positions: Cartesian3[],
    options: PolylineGeometryOptions,
  ): Cartesian3[] {
    if (options.heightReference !== 'clampToGround' || !options.sampleHeight) {
      return positions;
    }
    const sample = options.sampleHeight;

    const out: Cartesian3[] = [];
    out.push(Measure.clampToTerrain(positions[0], sample));
    for (let i = 1; i < positions.length; i++) {
      const a = positions[i - 1];
      const b = positions[i];
      const ua = PolylineGeometry._radial(a);
      const ub = PolylineGeometry._radial(b);
      const dot = Cartesian3.dot(ua, ub);
      const angle = Math.acos(dot < -1 ? -1 : dot > 1 ? 1 : dot);
      const steps = Math.max(
        Math.ceil((angle * 180 / Math.PI) / DENSIFY_STEP_DEG),
        MIN_SUBDIVISIONS,
      );
      const sinAngle = Math.sin(angle);
      if (sinAngle < 1e-12 || steps <= 1) {
        // Degenerate arc: slerp falls back to a plain lerp + normalize.
        for (let s = 1; s < steps; s++) {
          const t = s / steps;
          out.push(Measure.clampToTerrain(
            new Cartesian3(
              ua.x + (ub.x - ua.x) * t,
              ua.y + (ub.y - ua.y) * t,
              ua.z + (ub.z - ua.z) * t,
            ),
            sample,
          ));
        }
      } else {
        // Slerp on the unit sphere: (sin((1-t)ω)a + sin(tω)b) / sin(ω).
        for (let s = 1; s < steps; s++) {
          const t = s / steps;
          const k0 = Math.sin((1 - t) * angle) / sinAngle;
          const k1 = Math.sin(t * angle) / sinAngle;
          out.push(Measure.clampToTerrain(new Cartesian3(
            ua.x * k0 + ub.x * k1,
            ua.y * k0 + ub.y * k1,
            ua.z * k0 + ub.z * k1,
          ), sample));
        }
      }
      out.push(Measure.clampToTerrain(b, sample));
    }
    return out;
  }

  /**
   * Offset a position sideways: p + side * amount.
   */
  private static _offset(position: Cartesian3, side: Cartesian3, amount: number): Cartesian3 {
    return new Cartesian3(
      position.x + side.x * amount,
      position.y + side.y * amount,
      position.z + side.z * amount
    );
  }

  /**
   * Per-point unit tangents: end points use the single adjacent segment,
   * interior points use the normalized average of adjacent segments
   * (miter approximation). Falls back to the outgoing segment when adjacent
   * segments reverse direction (the average collapses to zero).
   */
  private static _computeTangents(positions: Cartesian3[]): Cartesian3[] {
    const n = positions.length;

    // Raw segment directions, reusing neighbours when a segment is degenerate.
    const segmentDirs: Array<Cartesian3 | undefined> = new Array(n - 1);
    for (let i = 0; i < n - 1; i++) {
      const diff = Cartesian3.subtract(positions[i + 1], positions[i], new Cartesian3());
      if (Cartesian3.magnitudeSquared(diff) < DEGENERATE_EPSILON_SQ) {
        segmentDirs[i] = undefined;
      } else {
        segmentDirs[i] = Cartesian3.normalize(diff, new Cartesian3());
      }
    }

    const firstValid = (from: number, step: number): Cartesian3 | undefined => {
      let i = from;
      while (i >= 0 && i < n - 1) {
        if (segmentDirs[i]) return segmentDirs[i];
        i += step;
      }
      return undefined;
    };

    const first = firstValid(0, 1);
    if (!first) return []; // All segments degenerate.

    const tangents: Cartesian3[] = new Array(n);
    for (let i = 0; i < n; i++) {
      const prev = firstValid(i - 1, -1) ?? first;
      const next = firstValid(i, 1) ?? prev;

      const sum = Cartesian3.add(prev, next, new Cartesian3());
      if (Cartesian3.magnitudeSquared(sum) < DEGENERATE_EPSILON_SQ) {
        tangents[i] = next; // Sharp reversal: use the outgoing segment.
      } else {
        tangents[i] = Cartesian3.normalize(sum, new Cartesian3());
      }
    }
    return tangents;
  }

  /**
   * Unit side vector in the surface tangent plane, perpendicular to the
   * segment tangent: side = normalize(cross(tangent, radial)). When the
   * tangent is parallel to the radial (line pointing straight up), fall
   * back to an arbitrary perpendicular in 3D.
   */
  private static _computeSide(position: Cartesian3, tangent: Cartesian3): Cartesian3 {
    const radial = PolylineGeometry._radial(position);
    const cross = Cartesian3.cross(tangent, radial, new Cartesian3());
    const lenSq = Cartesian3.magnitudeSquared(cross);
    if (lenSq < 1e-12) {
      // Degenerate: pick the world axis least aligned with the tangent.
      const tx = Math.abs(tangent.x);
      const ty = Math.abs(tangent.y);
      const tz = Math.abs(tangent.z);
      const axis = tx <= ty && tx <= tz ? Cartesian3.UNIT_X : ty <= tz ? Cartesian3.UNIT_Y : Cartesian3.UNIT_Z;
      return Cartesian3.normalize(Cartesian3.cross(tangent, axis, new Cartesian3()), new Cartesian3());
    }
    return Cartesian3.normalize(cross, new Cartesian3());
  }

  /**
   * Unit radial (outward surface) direction at a position.
   */
  private static _radial(position: Cartesian3): Cartesian3 {
    const lenSq = Cartesian3.magnitudeSquared(position);
    if (lenSq < 1e-20) {
      return Cartesian3.UNIT_Z.clone();
    }
    return Cartesian3.multiplyByScalar(position, 1 / Math.sqrt(lenSq), new Cartesian3());
  }
}

export default PolylineGeometry;
