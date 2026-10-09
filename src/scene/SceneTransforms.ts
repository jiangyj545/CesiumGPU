import { Cartesian2 } from '../math/Cartesian2';
import { Cartesian3 } from '../math/Cartesian3';
import { Camera } from './Camera';

/**
 * Structural scene contract consumed by SceneTransforms: a camera with a
 * view-projection matrix plus the canvas-backed context. `Scene` satisfies
 * this structurally; tests can pass a plain object.
 */
export interface SceneTransformsScene {
  readonly camera: Pick<Camera, 'viewProjectionMatrix'>;
  readonly context: { readonly canvas: HTMLCanvasElement };
}

/**
 * SceneTransforms - world / WGS84 → window (CSS pixel) projections,
 * mirroring the Cesium namespace subset used by hexapod_gcs
 * (`worldToWindowCoordinates`, `wgs84ToWindowCoordinates`).
 *
 * Results are in **CSS pixels** relative to the canvas' top-left corner,
 * regardless of the device pixel ratio: the device buffer is divided by
 * `canvas.width / canvas.clientWidth` before NDC → pixel conversion.
 */
export class SceneTransforms {
  /**
   * Project a world-space position (normalised ECEF, Earth radius = 1) to
   * window coordinates.
   * @returns `Cartesian2` in CSS pixels (y down), or `undefined` when the
   *   point is behind the camera (w ≤ 0) or on the near/far clip planes.
   */
  static worldToWindowCoordinates(
    scene: SceneTransformsScene,
    position: Cartesian3
  ): Cartesian2 | undefined {
    const vp = scene.camera.viewProjectionMatrix.values;
    const { x, y, z } = position;
    const w = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
    if (w <= 1e-9) return undefined; // behind the camera
    const ndcX = (vp[0] * x + vp[4] * y + vp[8] * z + vp[12]) / w;
    const ndcY = (vp[1] * x + vp[5] * y + vp[9] * z + vp[13]) / w;
    const ndcZ = (vp[2] * x + vp[6] * y + vp[10] * z + vp[14]) / w;
    if (ndcZ <= 0 || ndcZ >= 1) return undefined; // clipped

    const canvas = scene.context.canvas;
    const cssWidth = canvas.clientWidth || canvas.width;
    const cssHeight = canvas.clientHeight || canvas.height;
    return new Cartesian2(
      (ndcX * 0.5 + 0.5) * cssWidth,
      (0.5 - ndcY * 0.5) * cssHeight
    );
  }

  /**
   * Project a WGS84 position to window coordinates. Accepts either a
   * metre-ECEF `Cartesian3` (e.g. `Cartesian3.fromDegrees(...)`) or a
   * `{ lon, lat, height }` literal (degrees / metres); both are converted
   * to the engine's normalised ECEF world and delegated to
   * {@link SceneTransforms.worldToWindowCoordinates}.
   */
  static wgs84ToWindowCoordinates(
    scene: SceneTransformsScene,
    position: Cartesian3 | { lon: number; lat: number; height?: number }
  ): Cartesian2 | undefined {
    const scale = 1.0 / 6378137.0;
    // Metre-ECEF points scale directly; { lon, lat, height } literals use
    // the engine's flattened-world convention (Camera.resolveDestination).
    const world =
      'x' in position && 'y' in position && 'z' in position
        ? Cartesian3.multiplyByScalar(position, scale, new Cartesian3())
        : Camera.resolveDestination(position);
    return SceneTransforms.worldToWindowCoordinates(scene, world);
  }
}

export default SceneTransforms;
