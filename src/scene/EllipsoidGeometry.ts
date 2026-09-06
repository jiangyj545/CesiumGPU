import { Cartesian3 } from '../math/Cartesian3';
import { CesiumMath } from '../math/CesiumMath';

export interface EllipsoidGeometryOptions {
  /** Semi-axes in meters. Default: WGS84 */
  radii?: Cartesian3;
  /** Number of subdivisions in the longitude direction. Default: 64 */
  stackPartitions?: number;
  /** Number of subdivisions in the latitude direction. Default: 32 */
  slicePartitions?: number;
}

export interface GeometryBuffers {
  positions: Float32Array;    // vec3 per vertex
  normals: Float32Array;      // vec3 per vertex
  uvs: Float32Array;          // vec2 per vertex
  indices: Uint32Array;       // triangles
  vertexCount: number;
  indexCount: number;
}

export interface CreateRegionOptions {
  /**
   * UV sub-rect mapped over the region (u0/v0 at the northwest corner).
   * Default: the full [0, 1] range. Used to sample an ancestor tile texture
   * when a tile's own texture has not loaded yet.
   */
  uvRect?: { u0: number; v0: number; u1: number; v1: number };
  /**
   * Radial scale applied to the surface positions. Default: 1. Tile
   * primitives use a value slightly above 1 to sit just above the base
   * globe and avoid z-fighting.
   */
  radiusScale?: number;
  /**
   * Map v through the Web-Mercator function instead of linearly. Mercator
   * tile images distribute their pixel rows uniformly in Mercator-y space,
   * which is non-linear in latitude — a linear v makes every row's content
   * sit too far north (≈7.7° at mid latitudes, worse toward the poles),
   * squeezing the whole map toward the Arctic. Default: false (linear).
   */
  mercatorV?: boolean;
}

/**
 * EllipsoidGeometry - Generates tessellated ellipsoid geometry (UV sphere).
 * Matches Cesium's EllipsoidGeometry API.
 */
export class EllipsoidGeometry {
  private _options: Required<EllipsoidGeometryOptions>;

  constructor(options: EllipsoidGeometryOptions = {}) {
    this._options = {
      radii: options.radii ?? new Cartesian3(6378137.0, 6378137.0, 6356752.3142),
      stackPartitions: options.stackPartitions ?? 64,
      slicePartitions: options.slicePartitions ?? 32,
    };
  }

  /**
   * Generate vertex and index buffers for the ellipsoid.
   * Returns interleaved positions, normals, uvs and triangle indices.
   */
  createGeometry(): GeometryBuffers {
    const { radii, stackPartitions, slicePartitions } = this._options;

    // Normalize radii to unit scale (renderer works in unit-sphere space).
    // The model matrix will scale to actual size.
    const rx = 1.0;
    const ry = radii.y / radii.x;
    const rz = radii.z / radii.x;

    const stacks = stackPartitions;
    const slices = slicePartitions;

    const vertexCount = (stacks + 1) * (slices + 1);
    const positions = new Float32Array(vertexCount * 3);
    const normals   = new Float32Array(vertexCount * 3);
    const uvs       = new Float32Array(vertexCount * 2);

    let vi = 0;
    let ni = 0;
    let ti = 0;

    for (let stack = 0; stack <= stacks; stack++) {
      // phi: [π/2, -π/2] (latitude from north to south)
      const phi = CesiumMath.PI_OVER_TWO - stack * Math.PI / stacks;
      const cosPhi = Math.cos(phi);
      const sinPhi = Math.sin(phi);

      for (let slice = 0; slice <= slices; slice++) {
        // theta: [0, 2π] (longitude)
        const theta = slice * CesiumMath.TWO_PI / slices;
        const cosTheta = Math.cos(theta);
        const sinTheta = Math.sin(theta);

        // Unit-sphere position (with ellipsoid scaling)
        const nx = cosPhi * cosTheta;
        const ny = cosPhi * sinTheta;
        const nz = sinPhi;

        // Ellipsoid surface normal direction (normalised for a sphere = same as position on unit sphere)
        // For a general ellipsoid, the geodetic normal differs from the surface position.
        const nx2 = nx / (rx * rx);
        const ny2 = ny / (ry * ry);
        const nz2 = nz / (rz * rz);
        const nLen = Math.sqrt(nx2 * nx2 + ny2 * ny2 + nz2 * nz2);

        positions[vi]     = rx * nx;
        positions[vi + 1] = ry * ny;
        positions[vi + 2] = rz * nz;
        vi += 3;

        normals[ni]     = nx2 / nLen;
        normals[ni + 1] = ny2 / nLen;
        normals[ni + 2] = nz2 / nLen;
        ni += 3;

        uvs[ti]     = slice / slices;
        uvs[ti + 1] = stack / stacks;
        ti += 2;
      }
    }

    // Generate indices
    const indexCount = stacks * slices * 6;
    const indices = new Uint32Array(indexCount);
    let idx = 0;
    const stride = slices + 1;

    for (let stack = 0; stack < stacks; stack++) {
      for (let slice = 0; slice < slices; slice++) {
        const a = stack * stride + slice;
        const b = a + stride;
        const c = b + 1;
        const d = a + 1;

        // Two triangles per quad
        indices[idx++] = a;
        indices[idx++] = b;
        indices[idx++] = d;

        indices[idx++] = b;
        indices[idx++] = c;
        indices[idx++] = d;
      }
    }

    return { positions, normals, uvs, indices, vertexCount, indexCount };
  }

  /**
   * Returns interleaved vertex buffer: [position(3), normal(3), uv(2)] per vertex.
   */
  createInterleavedBuffer(): { vertices: Float32Array; indices: Uint32Array; vertexCount: number; indexCount: number } {
    const { positions, normals, uvs, indices, vertexCount, indexCount } = this.createGeometry();

    const stride = 8; // 3 + 3 + 2 floats per vertex
    const vertices = new Float32Array(vertexCount * stride);

    for (let i = 0; i < vertexCount; i++) {
      const vi = i * stride;
      vertices[vi]     = positions[i * 3];
      vertices[vi + 1] = positions[i * 3 + 1];
      vertices[vi + 2] = positions[i * 3 + 2];
      vertices[vi + 3] = normals[i * 3];
      vertices[vi + 4] = normals[i * 3 + 1];
      vertices[vi + 5] = normals[i * 3 + 2];
      vertices[vi + 6] = uvs[i * 2];
      vertices[vi + 7] = uvs[i * 2 + 1];
    }

    return { vertices, indices, vertexCount, indexCount };
  }

  /**
   * Generate interleaved geometry for a latitude/longitude sub-region of the
   * ellipsoid (used by the imagery tile system: one patch per Web Mercator
   * tile). UVs run from northwest (u0, v0) to southeast (u1, v1) of the
   * supplied `uvRect`, defaulting to the normalised [0, 1] range.
   *
   * Positions use the same normalised convention as {@link createGeometry}
   * (radii scaled so the equatorial radius is 1, y/z squashed by the WGS84
   * polar ratio), optionally scaled radially by `radiusScale`.
   *
   * @param west       Western edge longitude (radians)
   * @param south      Southern edge latitude (radians)
   * @param east       Eastern edge longitude (radians)
   * @param north      Northern edge latitude (radians)
   * @param partitions Subdivisions per axis (8×8 is ample for a tile)
   */
  static createRegion(
    west: number,
    south: number,
    east: number,
    north: number,
    partitions: number,
    options: CreateRegionOptions = {}
  ): { vertices: Float32Array; indices: Uint32Array; vertexCount: number; indexCount: number } {
    const uvRect  = options.uvRect ?? { u0: 0, v0: 0, u1: 1, v1: 1 };
    const scale   = options.radiusScale ?? 1.0;
    const ratio   = WGS84_POLAR_RATIO;
    const ratioSq = ratio * ratio;

    // Mercator rows: v must follow the Mercator-y of the latitude, not the
    // latitude itself (see CreateRegionOptions.mercatorV). Precompute the
    // edge ys so each stack maps lat → normalized y → v.
    const mercatorV  = options.mercatorV ?? false;
    const mercNorthY = Math.asinh(Math.tan(north));
    const mercSouthY = Math.asinh(Math.tan(south));

    const stacks = partitions; // rows: north → south
    const slices = partitions; // columns: west → east

    const vertexCount = (stacks + 1) * (slices + 1);
    const vertices    = new Float32Array(vertexCount * 8); // [pos(3), normal(3), uv(2)]

    let vi = 0;
    for (let stack = 0; stack <= stacks; stack++) {
      const t   = stack / stacks;
      const lat = north + (south - north) * t; // row 0 = north edge
      let v: number;
      if (mercatorV) {
        // Normalise this row's Mercator-y into [0, 1] across the region.
        // tan() diverges at ±π/2 — polar-cap callers must stay in the
        // Mercator domain (cap patches sample the top/bottom row linearly).
        const y = Math.asinh(Math.tan(lat));
        const tn = (y - mercNorthY) / (mercSouthY - mercNorthY);
        v = uvRect.v0 + (uvRect.v1 - uvRect.v0) * tn;
      } else {
        v = uvRect.v0 + (uvRect.v1 - uvRect.v0) * t;
      }
      const cosLat = Math.cos(lat);
      const sinLat = Math.sin(lat);

      for (let slice = 0; slice <= slices; slice++) {
        const s   = slice / slices;
        const lon = west + (east - west) * s;
        const u   = uvRect.u0 + (uvRect.u1 - uvRect.u0) * s;

        const nx = cosLat * Math.cos(lon);
        const ny = cosLat * Math.sin(lon);
        const nz = sinLat;

        // Position (EllipsoidGeometry normalisation: x on unit sphere,
        // y/z squashed by the polar ratio).
        vertices[vi++] = scale * nx;
        vertices[vi++] = scale * ny * ratio;
        vertices[vi++] = scale * nz * ratio;

        // Ellipsoid surface normal (same math as createGeometry).
        const ny2  = ny / ratioSq;
        const nz2  = nz / ratioSq;
        const nLen = Math.sqrt(nx * nx + ny2 * ny2 + nz2 * nz2);
        vertices[vi++] = nx / nLen;
        vertices[vi++] = ny2 / nLen;
        vertices[vi++] = nz2 / nLen;

        vertices[vi++] = u;
        vertices[vi++] = v;
      }
    }

    const indexCount = stacks * slices * 6;
    const indices    = new Uint32Array(indexCount);
    let   idx        = 0;
    const stride     = slices + 1;

    for (let stack = 0; stack < stacks; stack++) {
      for (let slice = 0; slice < slices; slice++) {
        const a = stack * stride + slice;
        const b = a + stride;
        const c = b + 1;
        const d = a + 1;

        indices[idx++] = a;
        indices[idx++] = b;
        indices[idx++] = d;

        indices[idx++] = b;
        indices[idx++] = c;
        indices[idx++] = d;
      }
    }

    return { vertices, indices, vertexCount, indexCount };
  }
}

/** WGS84 polar / equatorial semi-axis ratio (6356752.3142 / 6378137). */
const WGS84_POLAR_RATIO = 6356752.3142 / 6378137.0;

export default EllipsoidGeometry;
