/**
 * HeadingPitchRoll - orientation value type (all angles in radians),
 * mirroring Cesium's `HeadingPitchRoll`. Consumed by
 * `Transforms.headingPitchRollQuaternion` and drone-style entity orientation
 * bookkeeping (hexapod droneController call shape).
 *
 * heading: yaw around local up, 0 = north, positive clockwise (toward east).
 * pitch:   0 = level, negative = nose down.
 * roll:    around the forward axis.
 */
export class HeadingPitchRoll {
  constructor(
    public heading = 0,
    public pitch = 0,
    public roll = 0
  ) {}

  static fromRadians(heading: number, pitch: number, roll: number): HeadingPitchRoll {
    return new HeadingPitchRoll(heading, pitch, roll);
  }

  clone(result?: HeadingPitchRoll): HeadingPitchRoll {
    if (result) {
      result.heading = this.heading;
      result.pitch = this.pitch;
      result.roll = this.roll;
      return result;
    }
    return new HeadingPitchRoll(this.heading, this.pitch, this.roll);
  }

  equals(other: HeadingPitchRoll, epsilon = 1e-10): boolean {
    return (
      Math.abs(this.heading - other.heading) <= epsilon &&
      Math.abs(this.pitch - other.pitch) <= epsilon &&
      Math.abs(this.roll - other.roll) <= epsilon
    );
  }
}

export default HeadingPitchRoll;
