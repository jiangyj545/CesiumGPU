import { Matrix4 } from '../math/Matrix4';
import { Cartesian3 } from '../math/Cartesian3';

/**
 * Frustum - A view-frustum culling volume extracted from a view-projection
 * matrix (Gribb-Hartmann plane extraction).
 *
 * The six clip planes (left, right, bottom, top, near, far) are derived by
 * adding / subtracting the rows of the view-projection matrix (row 3 — the
 * w row — acts as the "inside" reference). Each plane is stored normalised as
 * `a·x + b·y + c·z + d = 0` with the normal pointing INTO the frustum, so the
 * interior is the positive half-space of every plane.
 */
export class Frustum {
  /** 6 planes × 4 components (a, b, c, d), flattened. */
  private readonly _planes = new Float32Array(6 * 4);

  /**
   * Extract the six planes from a view-projection matrix.
   * The matrix is read at call time — rebuild per frame (the camera's
   * `viewProjectionMatrix` is cached per update, so extraction is cheap).
   */
  static fromViewProjection(viewProjection: Matrix4): Frustum {
    const frustum = new Frustum();
    frustum._extract(viewProjection.values);
    return frustum;
  }

  /**
   * Conservative sphere test: returns true when the sphere is not entirely
   * outside any single plane. A sphere straddling a plane still counts as
   * intersecting (no partial-rejection), which keeps culling conservative.
   */
  intersectsSphere(center: Cartesian3, radius: number): boolean {
    const planes = this._planes;
    for (let i = 0; i < 6; i++) {
      const o = i * 4;
      const distance =
        planes[o]     * center.x +
        planes[o + 1] * center.y +
        planes[o + 2] * center.z +
        planes[o + 3];
      if (distance < -radius) return false;
    }
    return true;
  }

  // ── Private ────────────────────────────────────────────────────────────────

  /**
   * Gribb-Hartmann extraction. Storage is column-major (element (row, col)
   * = m[col * 4 + row]), so row i is [m[i], m[4+i], m[8+i], m[12+i]] and
   * plane = row3 ± rowK, K ∈ {x, y, z} for {left/right, bottom/top, near/far}.
   */
  private _extract(m: Float32Array): void {
    // Draw-order pairs: left, right, bottom, top, near, far.
    const rows  = [0, 0, 1, 1, 2, 2];
    const signs = [1, -1, 1, -1, 1, -1];

    for (let p = 0; p < 6; p++) {
      const k    = rows[p];
      const sign = signs[p];
      const o    = p * 4;

      const a = m[3]  + sign * m[k];
      const b = m[7]  + sign * m[4 + k];
      const c = m[11] + sign * m[8 + k];
      const d = m[15] + sign * m[12 + k];

      // Normalise; guard against a degenerate matrix (keep raw coefficients
      // so the test stays deterministic instead of producing NaN).
      const len = Math.sqrt(a * a + b * b + c * c);
      if (len > 1e-12) {
        this._planes[o]     = a / len;
        this._planes[o + 1] = b / len;
        this._planes[o + 2] = c / len;
        this._planes[o + 3] = d / len;
      } else {
        this._planes[o]     = a;
        this._planes[o + 1] = b;
        this._planes[o + 2] = c;
        this._planes[o + 3] = d;
      }
    }
  }
}

export default Frustum;
