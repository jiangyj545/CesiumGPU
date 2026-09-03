import { Cartesian3 } from '../math/Cartesian3';
import { CesiumMath } from '../math/CesiumMath';
import { Cartographic } from '../math/Cartographic';
import { EasingFunction } from '../math/EasingFunction';
import { Camera } from './Camera';
import type { FlyToOptions } from './Camera';

/**
 * Minimum camera altitude enforced mid-flight, as a fraction of the Earth's
 * radius (0.5 % ≈ 32 km).  Blended out at both endpoints so the flight still
 * lands exactly on the requested destination height.
 */
const MIN_ALTITUDE = 0.005;

/**
 * CameraFlightPath - animated camera flight between two positions/orientations.
 *
 * The horizontal path follows the great circle between the start and end
 * geographic positions using a cubic Hermite blend of the two unit position
 * vectors (tangents taken along the great-circle direction, scaled by the arc
 * angle) renormalised onto the sphere each evaluation.
 *
 * Altitude follows a parabolic arch over the straight lerp:
 *
 * ```
 * h(t) = lerp(h0, h1, t) + peak * sin(π·t)
 * peak = max(0, arcAngle * R * 0.25 - (h0 + h1) / 2)
 * ```
 *
 * with a floor of {@link MIN_ALTITUDE} blended by sin(π·t) so interior points
 * never dip below the surface while t = 0 / t = 1 remain exact.
 *
 * Orientation (heading/pitch) is interpolated numerically along the shortest
 * angular path for heading (wrap at 2π); the start values are derived from the
 * camera's current direction/up via {@link Camera._deriveHeadingPitch}.
 */
export class CameraFlightPath {
  /** Flight duration in seconds (>= 1 frame). */
  readonly duration: number;
  /** Easing applied to raw progress before evaluation. */
  readonly easingFunction: EasingFunction.Callback;
  /** Invoked when the flight reaches t = 1. */
  readonly completeCallback: (() => void) | null;
  /** Invoked when the flight is cancelled (user input or a new flyTo). */
  readonly cancelCallback: (() => void) | null;

  // ── Position path ─────────────────────────────────────────────────────────
  private readonly _u0: Cartesian3;          // unit position at start
  private readonly _u1: Cartesian3;          // unit position at end
  private readonly _tangent0: Cartesian3;    // great-circle tangent at start (× arc angle)
  private readonly _tangent1: Cartesian3;    // great-circle tangent at end   (× arc angle)
  private readonly _h0: number;              // start height, normalised units
  private readonly _h1: number;              // end height,   normalised units
  private readonly _peak: number;            // parabolic arch height, normalised units

  // ── Orientation path ──────────────────────────────────────────────────────
  private readonly _startHeading: number;
  private readonly _startPitch: number;
  private readonly _headingDelta: number;    // shortest-path heading delta
  private readonly _endPitch: number;
  private readonly _endRoll: number;

  private _elapsed = 0;

  constructor(camera: Camera, options: FlyToOptions) {
    if (!options.destination) {
      throw new Error('CameraFlightPath: options.destination is required.');
    }

    this.duration = Math.max(options.duration ?? 3.0, 1.0 / 60.0);
    this.easingFunction = options.easingFunction ?? EasingFunction.QUINTIC_IN_OUT;
    this.completeCallback = options.complete ?? null;
    this.cancelCallback = options.cancel ?? null;

    // ── Start / end cartographics ───────────────────────────────────────────
    const startPos = Cartesian3.clone(camera.position);
    const startCarto = Cartographic.fromCartesian(
      Cartesian3.multiplyByScalar(startPos, Camera.EARTH_SCALE, new Cartesian3()),
    );

    const endPos = Camera.resolveDestination(options.destination);       // normalised ECEF
    const endCarto = Cartographic.fromCartesian(
      Cartesian3.multiplyByScalar(endPos, Camera.EARTH_SCALE, new Cartesian3()),
    );

    this._h0 = startCarto.height;
    this._h1 = endCarto.height;

    // ── Great-circle geometry ───────────────────────────────────────────────
    this._u0 = Cartesian3.normalize(startPos, new Cartesian3());
    this._u1 = Cartesian3.normalize(endPos, new Cartesian3());
    const cosArc = CesiumMath.clamp(Cartesian3.dot(this._u0, this._u1), -1.0, 1.0);
    const arcAngle = Math.acos(cosArc);

    // Rotation axis of the great circle (degenerate → pick any perpendicular).
    const axis = new Cartesian3();
    Cartesian3.cross(this._u0, this._u1, axis);
    if (Cartesian3.magnitude(axis) < 1e-12) {
      const ref = Math.abs(this._u0.z) < 0.9 ? Cartesian3.UNIT_Z : Cartesian3.UNIT_X;
      Cartesian3.cross(this._u0, ref, axis);
    }
    Cartesian3.normalize(axis, axis);

    // Hermite tangents: great-circle travel direction scaled by the arc angle
    // (matches the derivative magnitude of a constant-angular-speed slerp).
    const travelAt0 = new Cartesian3();
    const travelAt1 = new Cartesian3();
    Cartesian3.cross(axis, this._u0, travelAt0);
    Cartesian3.cross(axis, this._u1, travelAt1);
    this._tangent0 = Cartesian3.multiplyByScalar(travelAt0, arcAngle, new Cartesian3());
    this._tangent1 = Cartesian3.multiplyByScalar(travelAt1, arcAngle, new Cartesian3());

    // Parabolic altitude arch: keeps long arcs from grazing the surface.
    // R = 1 in normalised units (Earth radius).
    const meanHeight = (this._h0 + this._h1) * 0.5;
    this._peak = Math.max(0.0, arcAngle * 1.0 * 0.25 - meanHeight);

    // ── Orientation path ────────────────────────────────────────────────────
    const derived = camera._deriveHeadingPitch();
    this._startHeading = derived.heading;
    this._startPitch = derived.pitch;

    const orientation = options.orientation;
    const endHeading = orientation?.heading ?? this._startHeading;
    this._endPitch = CesiumMath.clamp(
      orientation?.pitch ?? this._startPitch, -Math.PI / 2, Math.PI / 2,
    );
    this._endRoll = orientation?.roll ?? 0.0;

    // Shortest angular path for heading (wrap at 2π).
    this._headingDelta = CesiumMath.negativePiToPi(endHeading - this._startHeading);
  }

  /** True once the flight has reached (or passed) its destination. */
  get finished(): boolean {
    return this._elapsed >= this.duration;
  }

  /**
   * Advance the flight by `dtSeconds` and write the interpolated state into
   * `camera`.
   */
  update(camera: Camera, dtSeconds: number): void {
    this._elapsed += dtSeconds;
    const raw = CesiumMath.clamp(this._elapsed / this.duration, 0.0, 1.0);
    this.evaluate(camera, this.easingFunction(raw));
  }

  /**
   * Write the flight state at normalised progress `t` (already eased, [0, 1])
   * into the camera's position / direction / up / right.
   */
  evaluate(camera: Camera, t: number): void {
    // ── Position: Hermite-blended unit direction on the sphere ─────────────
    const t2 = t * t;
    const t3 = t2 * t;
    const h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    const h10 = t3 - 2.0 * t2 + t;
    const h01 = -2.0 * t3 + 3.0 * t2;
    const h11 = t3 - t2;

    const u0 = this._u0;
    const u1 = this._u1;
    const m0 = this._tangent0;
    const m1 = this._tangent1;
    const dir = new Cartesian3(
      h00 * u0.x + h10 * m0.x + h01 * u1.x + h11 * m1.x,
      h00 * u0.y + h10 * m0.y + h01 * u1.y + h11 * m1.y,
      h00 * u0.z + h10 * m0.z + h01 * u1.z + h11 * m1.z,
    );
    Cartesian3.normalize(dir, dir);

    // Altitude: linear lerp + parabolic arch, floored above the surface.
    // The floor is blended by sin(π·t) so both endpoints stay exact.
    const dome = Math.sin(Math.PI * t);
    const height = Math.max(
      CesiumMath.lerp(this._h0, this._h1, t) + this._peak * dome,
      MIN_ALTITUDE * dome,
    );

    const radius = 1.0 + height;
    camera.position.x = dir.x * radius;
    camera.position.y = dir.y * radius;
    camera.position.z = dir.z * radius;

    // ── Orientation ─────────────────────────────────────────────────────────
    const heading = this._startHeading + this._headingDelta * t;
    const pitch = CesiumMath.lerp(this._startPitch, this._endPitch, t);
    camera._applyOrientation(heading, pitch, this._endRoll);
  }
}

export default CameraFlightPath;
