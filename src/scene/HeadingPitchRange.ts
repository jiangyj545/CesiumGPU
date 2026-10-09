/**
 * HeadingPitchRange - camera offset relative to a target point, mirroring
 * Cesium's `HeadingPitchRange`. Used by `Camera.lookAt`,
 * `Camera.flyToBoundingSphere` and `Viewer.zoomTo`.
 *
 * All angles in radians; `range` is the distance from the target in metres.
 */
export class HeadingPitchRange {
  constructor(
    public heading = 0,
    public pitch = 0,
    public range = 0
  ) {}
}

export default HeadingPitchRange;
