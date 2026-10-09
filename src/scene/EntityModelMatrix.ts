import { Cartesian3 } from '../math/Cartesian3';
import { Matrix4 } from '../math/Matrix4';
import { Quaternion } from '../math/Quaternion';
import { Transforms } from '../math/Transforms';
import { Camera } from './Camera';

/**
 * Compose the world (normalised ECEF) model matrix for an Entity model:
 *
 * ```
 * modelMatrix = ENU(position) × rotation(orientation) × Y-up fix × scale
 * ```
 *
 * - `position` is in **normalised ECEF** (engine units, Earth radius = 1);
 *   it is converted to metres internally because
 *   `Transforms.eastNorthUpToFixedFrame` expects metre-ECEF origins.
 * - `orientation` is the ENU-frame quaternion from
 *   `Transforms.headingPitchRollQuaternion` (identity when omitted → model
 *   axes align with the local ENU frame).
 * - The GLTF Y-up fix (`Matrix4.fromRotationX(π/2)`) follows the convention
 *   documented on `Model.modelMatrix`.
 */
export function composeEntityModelMatrix(
  position: Cartesian3,
  orientation?: Quaternion,
  scale = 1.0,
  result?: Matrix4
): Matrix4 {
  // GLTF is Y-up: the +90° X rotation aligns the model's +Y with local up.
  const frame = composeEntityFrameMatrix(position, orientation, scale, result);
  return Matrix4.multiply(frame, Matrix4.fromRotationX(Math.PI / 2, new Matrix4()), frame);
}

/**
 * Entity local frame → world (normalised ECEF), without the glTF Y-up fix:
 * `frame = ENU(position) × rotation(orientation) × scale`. The local +Z axis
 * maps to the surface normal (identity orientation). Used by native engine
 * graphics whose geometry already assumes a Z-up local frame
 * (`Entity.cylinder` → `CylinderGeometry`).
 */
export function composeEntityFrameMatrix(
  position: Cartesian3,
  orientation?: Quaternion,
  scale = 1.0,
  result?: Matrix4
): Matrix4 {
  const positionMetres = Cartesian3.multiplyByScalar(
    position,
    Camera.EARTH_SCALE,
    new Cartesian3()
  );
  let m = Transforms.eastNorthUpToFixedFrame(positionMetres, new Matrix4());
  if (
    orientation &&
    !(orientation.x === 0 && orientation.y === 0 && orientation.z === 0 && orientation.w === 1)
  ) {
    m = Matrix4.multiply(m, Matrix4.fromQuaternion(orientation, new Matrix4()), new Matrix4());
  }
  if (scale !== 1.0) {
    m = Matrix4.multiply(m, Matrix4.fromUniformScale(scale, new Matrix4()), new Matrix4());
  }
  if (result && result !== m) {
    result.values.set(m.values);
    return result;
  }
  return m;
}

/**
 * Resolve the effective uniform scale for a model with `minimumPixelSize`
 * (CSS px): the model's nominal extent is `scale` world-units, so the
 * on-screen footprint must be at least `minimumPixelSize` pixels.
 *
 * `metresPerPixel = 2 · distance · tan(fovY/2) / viewportHeight` at the
 * entity; the required world size is `minimumPixelSize × metresPerPixel`
 * (converted to normalised units). Returns `scale` when the model already
 * covers the minimum.
 */
export function resolveMinimumPixelScale(
  scale: number,
  minimumPixelSize: number | undefined,
  distanceMetres: number,
  fovY: number,
  viewportHeightPx: number
): number {
  if (!minimumPixelSize || minimumPixelSize <= 0 || viewportHeightPx <= 0) return scale;
  const metresPerPixel = (2 * distanceMetres * Math.tan(fovY * 0.5)) / viewportHeightPx;
  const required = (minimumPixelSize * metresPerPixel) / Camera.EARTH_SCALE;
  return Math.max(scale, required);
}
