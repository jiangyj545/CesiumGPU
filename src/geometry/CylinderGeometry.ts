import { GeometryPipeline } from './GeometryPipeline';
import type { InterleavedGeometryBuffers } from './GeometryPipeline';
import { Camera } from '../scene/Camera';

export interface CylinderGeometryOptions {
  /** Cylinder length in metres (axis = local +Z of the entity frame). */
  length: number;
  /** Top cap radius in metres (0 = cone tip). */
  topRadius: number;
  /** Bottom cap radius in metres. */
  bottomRadius: number;
  /** Radial segments. Default 16. */
  slices?: number;
}

/**
 * CylinderGeometry - smooth-shaded cylinder / cone in the entity local frame
 * (+Z axis, centred on the entity position), mirroring Cesium's
 * `CylinderGeometry` subset used by `Entity.cylinder` (length / topRadius /
 * bottomRadius).
 *
 * Dimensions are given in **metres** and emitted in normalised engine units
 * (÷ EARTH_SCALE) so the primitive's model matrix needs no scale — the
 * visualizer composes it via `composeEntityFrameMatrix(position,
 * orientation)`. Side normals follow the slanted surface for cones
 * (topRadius ≠ bottomRadius); caps are emitted when the radius is > 0.
 */
export class CylinderGeometry {
  static createGeometry(options: CylinderGeometryOptions): InterleavedGeometryBuffers {
    const slices = Math.max(3, options.slices ?? 16);
    const h = options.length / Camera.EARTH_SCALE;
    const rTop = options.topRadius / Camera.EARTH_SCALE;
    const rBottom = options.bottomRadius / Camera.EARTH_SCALE;
    const zTop = h / 2;
    const zBottom = -h / 2;
    // Side normal slope: dz per radial unit (constant along the slant).
    const dz = (rBottom - rTop) / Math.max(h, 1e-12);

    const positions: number[] = [];
    const normals: number[] = [];
    const uvs: number[] = [];
    const indices: number[] = [];
    let v = 0;

    const push = (x: number, y: number, z: number, nx: number, ny: number, nz: number, u: number, w: number): void => {
      positions.push(x, y, z);
      normals.push(nx, ny, nz);
      uvs.push(u, w);
      v++;
    };

    // ── Side wall: duplicate rim vertices per ring for correct slant shading.
    const bottomStart = v;
    for (let i = 0; i <= slices; i++) {
      const a = (i / slices) * Math.PI * 2;
      const c = Math.cos(a), s = Math.sin(a);
      const nl = Math.hypot(1, dz);
      push(rBottom * c, rBottom * s, zBottom, c / nl, s / nl, dz / nl, i / slices, 0);
    }
    const topStart = v;
    for (let i = 0; i <= slices; i++) {
      const a = (i / slices) * Math.PI * 2;
      const c = Math.cos(a), s = Math.sin(a);
      const nl = Math.hypot(1, dz);
      push(rTop * c, rTop * s, zTop, c / nl, s / nl, dz / nl, i / slices, 1);
    }
    for (let i = 0; i < slices; i++) {
      const b = bottomStart + i, t = topStart + i;
      indices.push(b, b + 1, t, t, b + 1, t + 1);
    }

    // ── Caps (only when the radius is non-zero; a cone tip needs no cap).
    const cap = (radius: number, z: number, nz: number): void => {
      if (radius <= 0) return;
      const start = v;
      for (let i = 0; i <= slices; i++) {
        const a = (i / slices) * Math.PI * 2;
        push(radius * Math.cos(a), radius * Math.sin(a), z, 0, 0, nz, i / slices, 1);
      }
      const centre = v;
      push(0, 0, z, 0, 0, nz, 0.5, 0.5);
      for (let i = 0; i < slices; i++) {
        if (nz > 0) indices.push(start + i, start + i + 1, centre);
        else indices.push(start + i + 1, start + i, centre); // 反向绕序，法线朝下
      }
    };
    cap(rTop, zTop, 1);
    cap(rBottom, zBottom, -1);

    return {
      vertices: GeometryPipeline.interleave(
        new Float32Array(positions),
        new Float32Array(normals),
        new Float32Array(uvs)
      ),
      indices: new Uint32Array(indices),
      vertexCount: v,
      indexCount: indices.length,
    };
  }

  /**
   * Change-detection key for per-frame rebuild decisions: length / radii /
   * slices rounded to float-printable precision (the visualizer rebuilds the
   * geometry only when this string changes).
   */
  static signature(options: CylinderGeometryOptions): string {
    return [
      options.length.toFixed(4),
      options.topRadius.toFixed(4),
      options.bottomRadius.toFixed(4),
      options.slices ?? 16,
    ].join('|');
  }
}

export default CylinderGeometry;
