import { CesiumMath } from '../math/CesiumMath';
import { GeometryPipeline } from './GeometryPipeline';
import type { InterleavedGeometryBuffers } from './GeometryPipeline';

export interface RectangleGeometryOptions {
  /**
   * Subdivisions along each direction (grid of (subdivisions+1)^2 vertices).
   * Default: 32
   */
  subdivisions?: number;
  /**
   * Height above the surface, in normalized units (Earth radius = 1).
   * Default: 0 (on the surface)
   */
  height?: number;
}

/* WGS84 parameters (matching Cartesian3.fromRadians / EllipsoidGeometry). */
const SEMI_MAJOR = 6378137.0;
const SEMI_MINOR = 6356752.3142;
const ECCENTRICITY_SQ = 1.0 - (SEMI_MINOR * SEMI_MINOR) / (SEMI_MAJOR * SEMI_MAJOR);

/**
 * RectangleGeometry - Tessellates a lat/lon rectangle as a spherical patch in
 * normalized ECEF space (Earth equatorial radius = 1.0, polar axis scaled by
 * b/a, matching `EllipsoidGeometry` and the globe primitive).
 *
 * Rows go north -> south, columns west -> east, using the same quad-triangle
 * pattern and CCW-from-outside winding as `EllipsoidGeometry`.
 */
export class RectangleGeometry {
  /**
   * Build the rectangle patch geometry.
   *
   * @param west    - western longitude, degrees
   * @param south   - southern latitude, degrees
   * @param east    - eastern longitude, degrees
   * @param north   - northern latitude, degrees
   * @param options - subdivisions / height
   */
  static fromDegrees(
    west: number,
    south: number,
    east: number,
    north: number,
    options: RectangleGeometryOptions = {}
  ): InterleavedGeometryBuffers {
    const westRad = CesiumMath.toRadians(west);
    const eastRad = CesiumMath.toRadians(east);
    const southRad = CesiumMath.clamp(CesiumMath.toRadians(south), -CesiumMath.PI_OVER_TWO, CesiumMath.PI_OVER_TWO);
    const northRad = CesiumMath.clamp(CesiumMath.toRadians(north), -CesiumMath.PI_OVER_TWO, CesiumMath.PI_OVER_TWO);

    if (westRad === eastRad || southRad === northRad) {
      return GeometryPipeline.emptyBuffers();
    }

    const subdivisions = Math.max(1, Math.floor(options.subdivisions ?? 32));
    const height = options.height ?? 0.0;
    const heightMeters = height * SEMI_MAJOR;

    const cols = subdivisions + 1;
    const rows = subdivisions + 1;
    const vertexCount = rows * cols;
    const indexCount = subdivisions * subdivisions * 6;

    const positionsOut = new Float32Array(vertexCount * 3);
    const normalsOut = new Float32Array(vertexCount * 3);
    const uvsOut = new Float32Array(vertexCount * 2);
    const indices = new Uint32Array(indexCount);

    let vi = 0;
    let ti = 0;

    for (let row = 0; row < rows; row++) {
      // Latitude from north to south (matches EllipsoidGeometry stack order).
      const lat = northRad - (row * (northRad - southRad)) / subdivisions;

      const cosLat = Math.cos(lat);
      const sinLat = Math.sin(lat);
      const n = SEMI_MAJOR / Math.sqrt(1.0 - ECCENTRICITY_SQ * sinLat * sinLat);

      // Geodetic surface position, normalized to equatorial radius 1.
      const px = ((n + heightMeters) * cosLat) / SEMI_MAJOR;
      const pz = ((n * (1.0 - ECCENTRICITY_SQ) + heightMeters) * sinLat) / SEMI_MAJOR;

      // Geodetic surface normal: normalize(x/a^2, y/a^2, z/b^2) evaluated on
      // the (normalized) surface point, i.e. axial components scaled from the
      // normalized position. Unit length, independent of height.
      const nxAxial = px / SEMI_MAJOR;
      const nzAxial = pz * SEMI_MAJOR / (SEMI_MINOR * SEMI_MINOR);
      const invLen = 1.0 / Math.sqrt(nxAxial * nxAxial + nzAxial * nzAxial);
      const nx = nxAxial * invLen;
      const nz = nzAxial * invLen;

      for (let col = 0; col < cols; col++) {
        const lon = westRad + (col * (eastRad - westRad)) / subdivisions;
        const cosLon = Math.cos(lon);
        const sinLon = Math.sin(lon);

        positionsOut[vi] = px * cosLon;
        positionsOut[vi + 1] = px * sinLon;
        positionsOut[vi + 2] = pz;
        normalsOut[vi] = nx * cosLon;
        normalsOut[vi + 1] = nx * sinLon;
        normalsOut[vi + 2] = nz;
        uvsOut[ti] = col / subdivisions;
        uvsOut[ti + 1] = row / subdivisions; // v = 0 at north edge
        vi += 3;
        ti += 2;
      }
    }

    // Two triangles per quad; row stride = cols (same pattern as EllipsoidGeometry).
    let ii = 0;
    const stride = cols;
    for (let row = 0; row < subdivisions; row++) {
      for (let col = 0; col < subdivisions; col++) {
        const a = row * stride + col;
        const b = a + stride;
        const c = b + 1;
        const d = a + 1;

        indices[ii++] = a;
        indices[ii++] = b;
        indices[ii++] = d;

        indices[ii++] = b;
        indices[ii++] = c;
        indices[ii++] = d;
      }
    }

    return {
      vertices: GeometryPipeline.interleave(positionsOut, normalsOut, uvsOut),
      indices,
      vertexCount,
      indexCount,
    };
  }
}

export default RectangleGeometry;
