import { Matrix4 } from '../math/Matrix4';

/**
 * Interleaved geometry buffers consumed by `Primitive`.
 *
 * Vertex layout (identical to `EllipsoidGeometry.createInterleavedBuffer`):
 * [position(3), normal(3), uv(2)] = 8 floats / 32 bytes per vertex.
 * Indices are `Uint32Array` triangle lists.
 */
export interface InterleavedGeometryBuffers {
  vertices: Float32Array;
  indices: Uint32Array;
  vertexCount: number;
  indexCount: number;
}

/**
 * GeometryPipeline - Utility functions for converting raw geometry arrays
 * into the interleaved layout consumed by the renderer.
 *
 * Winding convention: triangles are counter-clockwise when viewed from the
 * outside (front face), matching `EllipsoidGeometry`.
 */
export class GeometryPipeline {
  /** Number of floats per interleaved vertex (3 position + 3 normal + 2 uv). */
  static readonly VERTEX_STRIDE = 8;

  /**
   * Interleave separate position/normal/uv arrays into the single vertex
   * buffer layout consumed by `Primitive`:
   * [px, py, pz, nx, ny, nz, u, v] per vertex.
   *
   * @param positions - 3 floats per vertex
   * @param normals   - 3 floats per vertex
   * @param uvs       - 2 floats per vertex
   */
  static interleave(positions: Float32Array, normals: Float32Array, uvs: Float32Array): Float32Array {
    const vertexCount = positions.length / 3;
    if (positions.length % 3 !== 0) {
      throw new Error('GeometryPipeline.interleave: positions length must be a multiple of 3');
    }
    if (normals.length !== positions.length) {
      throw new Error('GeometryPipeline.interleave: normals length must match positions length');
    }
    if (uvs.length !== vertexCount * 2) {
      throw new Error('GeometryPipeline.interleave: uvs length must be 2 floats per vertex');
    }

    const stride = GeometryPipeline.VERTEX_STRIDE;
    const vertices = new Float32Array(vertexCount * stride);

    for (let i = 0; i < vertexCount; i++) {
      const vi = i * stride;
      const pi = i * 3;
      const ti = i * 2;
      vertices[vi]     = positions[pi];
      vertices[vi + 1] = positions[pi + 1];
      vertices[vi + 2] = positions[pi + 2];
      vertices[vi + 3] = normals[pi];
      vertices[vi + 4] = normals[pi + 1];
      vertices[vi + 5] = normals[pi + 2];
      vertices[vi + 6] = uvs[ti];
      vertices[vi + 7] = uvs[ti + 1];
    }

    return vertices;
  }

  /**
   * Compute smooth per-vertex normals by accumulating face normals
   * (area-weighted via unnormalized cross products) and normalizing.
   *
   * @param positions - 3 floats per vertex
   * @param indices   - triangle list (Uint16 or Uint32)
   * @returns 3 floats per vertex; degenerate faces contribute nothing and
   *          orphan vertices keep a zero normal.
   */
  static computeNormals(positions: Float32Array, indices: Uint16Array | Uint32Array): Float32Array {
    const vertexCount = positions.length / 3;
    const normals = new Float32Array(vertexCount * 3);

    for (let t = 0; t < indices.length; t += 3) {
      const ia = indices[t] * 3;
      const ib = indices[t + 1] * 3;
      const ic = indices[t + 2] * 3;

      const ax = positions[ia], ay = positions[ia + 1], az = positions[ia + 2];
      const bx = positions[ib], by = positions[ib + 1], bz = positions[ib + 2];
      const cx = positions[ic], cy = positions[ic + 1], cz = positions[ic + 2];

      // Cross product of (b - a) x (c - a): CCW winding -> outward normal.
      const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
      const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
      const nx = e1y * e2z - e1z * e2y;
      const ny = e1z * e2x - e1x * e2z;
      const nz = e1x * e2y - e1y * e2x;

      // Unnormalized accumulation acts as area weighting.
      normals[ia] += nx; normals[ia + 1] += ny; normals[ia + 2] += nz;
      normals[ib] += nx; normals[ib + 1] += ny; normals[ib + 2] += nz;
      normals[ic] += nx; normals[ic + 1] += ny; normals[ic + 2] += nz;
    }

    for (let i = 0; i < vertexCount; i++) {
      const vi = i * 3;
      const len = Math.sqrt(
        normals[vi] * normals[vi] +
        normals[vi + 1] * normals[vi + 1] +
        normals[vi + 2] * normals[vi + 2]
      );
      if (len > 1e-12) {
        normals[vi] /= len;
        normals[vi + 1] /= len;
        normals[vi + 2] /= len;
      }
    }

    return normals;
  }

  /**
   * Transform a position array in place by a 4x4 matrix
   * (positions are treated as points, i.e. w = 1).
   *
   * @param positions - 3 floats per vertex, transformed in place
   * @param matrix    - column-major 4x4 transform
   * @returns the same `positions` array, for chaining
   */
  static transformPositions(positions: Float32Array, matrix: Matrix4): Float32Array {
    const m = matrix.values;
    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i];
      const y = positions[i + 1];
      const z = positions[i + 2];
      positions[i]     = m[0] * x + m[4] * y + m[8]  * z + m[12];
      positions[i + 1] = m[1] * x + m[5] * y + m[9]  * z + m[13];
      positions[i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
    }
    return positions;
  }

  /**
   * Returns empty buffers (used by geometry factories for degenerate input).
   */
  static emptyBuffers(): InterleavedGeometryBuffers {
    return {
      vertices: new Float32Array(0),
      indices: new Uint32Array(0),
      vertexCount: 0,
      indexCount: 0,
    };
  }
}

export default GeometryPipeline;
