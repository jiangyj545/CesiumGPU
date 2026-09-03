import { Matrix4 } from '../math/Matrix4';
import { Cartesian3 } from '../math/Cartesian3';
import { CesiumMath } from '../math/CesiumMath';
import { Cartographic } from '../math/Cartographic';
import { EasingFunction } from '../math/EasingFunction';
import { CameraFlightPath } from '../core/CameraFlightPath';

/** Camera orientation expressed as heading / pitch / roll (radians). */
export interface Orientation {
  /** Yaw around the local up axis. 0 = north, positive = clockwise (toward east). */
  heading?: number;
  /** Pitch. 0 = look at the horizon, negative = look down, -π/2 = nadir. */
  pitch?: number;
  /** Roll around the look direction. */
  roll?: number;
}

/**
 * Destination accepted by `setView` / `flyTo`:
 * - `Cartesian3`: ECEF position in **metres** (e.g. `Cartesian3.fromDegrees(...)`).
 * - `{ lon, lat, height }`: longitude/latitude in **degrees**, height in **metres**.
 */
export type Destination = Cartesian3 | { lon: number; lat: number; height?: number };

export interface SetViewOptions {
  destination?: Destination;
  orientation?: Orientation;
}

export interface FlyToOptions extends SetViewOptions {
  /** Flight duration in seconds. Default 3. */
  duration?: number;
  /** Easing curve. Default `EasingFunction.QUINTIC_IN_OUT`. */
  easingFunction?: EasingFunction.Callback;
  /** Invoked when the flight completes normally. */
  complete?: () => void;
  /** Invoked when the flight is cancelled (user input or a new flyTo). */
  cancel?: () => void;
}

/**
 * Camera - Perspective camera with orbit controls, matching Cesium's Camera API.
 *
 * Uses a right-handed coordinate system.  The Earth's centre is at the origin.
 * The camera's position is expressed in the same normalised ECEF space where the
 * Earth's equatorial radius ≈ 1.0 (scaled by EARTH_SCALE).
 */
export class Camera {
  /** Vertical field of view in radians. */
  frustum = {
    fov: CesiumMath.toRadians(60),
    aspectRatio: 1.0,
    near: 0.005,   // 0.5% of normalised Earth radius (≈32 km); safely below min camera-to-surface distance of 0.05
    far: 100.0,    // covers max orbit radius of 50 normalised units with generous margin
  };

  /** World-space position in normalised ECEF (Earth radius = 1). */
  position: Cartesian3;
  /** World-space direction the camera is looking. */
  direction: Cartesian3;
  /** World-space "up" vector. */
  up: Cartesian3;
  /** World-space "right" vector. */
  right: Cartesian3;

  private _viewMatrix: Matrix4;
  private _projMatrix: Matrix4;
  private _viewProjMatrix: Matrix4;
  private _viewDirty = true;
  private _projDirty = true;

  /** Active camera flight, or null when the camera is idle. */
  private _flight: CameraFlightPath | null = null;

  /** Minimum dot product of camera direction with nadir before pitch is rejected (~85° max tilt). */
  private static readonly _MAX_TILT_NADIR_DOT = Math.cos(CesiumMath.toRadians(85));

  /** Scale factor: 1 internal unit = EARTH_SCALE metres */
  static readonly EARTH_SCALE = 6378137.0;

  constructor() {
    // Start above the equator at ~3× Earth radius
    this.position  = new Cartesian3(0, 0, 3.0);
    this.direction = new Cartesian3(0, 0, -1);
    this.up        = new Cartesian3(0, 1, 0);
    this.right     = new Cartesian3(1, 0, 0);

    this._viewMatrix     = new Matrix4();
    this._projMatrix     = new Matrix4();
    this._viewProjMatrix = new Matrix4();
  }

  // ── Matrix accessors ───────────────────────────────────────────────────────

  get viewMatrix(): Matrix4 {
    if (this._viewDirty) this._updateViewMatrix();
    return this._viewMatrix;
  }

  get projectionMatrix(): Matrix4 {
    if (this._projDirty) this._updateProjMatrix();
    return this._projMatrix;
  }

  get viewProjectionMatrix(): Matrix4 {
    if (this._viewDirty || this._projDirty) {
      if (this._viewDirty) this._updateViewMatrix();
      if (this._projDirty) this._updateProjMatrix();
      Matrix4.multiply(this._projMatrix, this._viewMatrix, this._viewProjMatrix);
    }
    return this._viewProjMatrix;
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Position the camera using geographic coordinates.
   * @param options.destination - ECEF position in metres (`Cartesian3.fromDegrees`)
   *   or a `{ lon, lat, height }` shortcut (degrees / metres).
   * @param options.orientation - heading / pitch / roll in radians.  When omitted
   *   the camera looks straight down at the Earth centre (nadir).
   */
  setView(options: SetViewOptions): void {
    if (options.destination) {
      const dest = Camera.resolveDestination(options.destination);
      this.position.x = dest.x;
      this.position.y = dest.y;
      this.position.z = dest.z;

      if (options.orientation) {
        this._applyOrientation(
          options.orientation.heading ?? 0.0,
          CesiumMath.clamp(options.orientation.pitch ?? -Math.PI / 2, -Math.PI / 2, Math.PI / 2),
          options.orientation.roll ?? 0.0,
        );
      } else {
        this._rebuildFrame();
      }
    }
    this._viewDirty = true;
  }

  /**
   * Set aspect ratio (called when canvas is resized).
   */
  setAspectRatio(aspect: number): void {
    this.frustum.aspectRatio = aspect;
    this._projDirty = true;
  }

  /**
   * Orbit the camera around the Earth centre by delta angles (radians).
   */
  rotate(deltaLon: number, deltaLat: number): void {
    const radius = Cartesian3.magnitude(this.position);

    // Unit vector of the current camera position (before the orbit step)
    const oldUnit = new Cartesian3();
    Cartesian3.normalize(this.position, oldUnit);

    // Current lon/lat
    const lon = Math.atan2(this.position.y, this.position.x);
    const lat = Math.asin(CesiumMath.clamp(this.position.z / radius, -1, 1));

    const newLon = lon + deltaLon;
    const newLat = CesiumMath.clamp(lat + deltaLat, -Math.PI / 2 + 0.01, Math.PI / 2 - 0.01);

    this.position.x = radius * Math.cos(newLat) * Math.cos(newLon);
    this.position.y = radius * Math.cos(newLat) * Math.sin(newLon);
    this.position.z = radius * Math.sin(newLat);

    // Rotate the camera frame by the same rotation that maps oldUnit → newUnit
    // so that any pitch / heading applied via middle-mouse is preserved.
    const newUnit = new Cartesian3();
    Cartesian3.normalize(this.position, newUnit);
    const rotAxis = new Cartesian3();
    Cartesian3.cross(oldUnit, newUnit, rotAxis);
    const axisLen = Cartesian3.magnitude(rotAxis);
    if (axisLen > 1e-9) {
      Cartesian3.normalize(rotAxis, rotAxis);
      const angle = Math.acos(CesiumMath.clamp(Cartesian3.dot(oldUnit, newUnit), -1, 1));
      Camera._rotateByAxis(this.direction, rotAxis, angle, this.direction);
      Camera._rotateByAxis(this.up,        rotAxis, angle, this.up);
      Camera._rotateByAxis(this.right,     rotAxis, angle, this.right);
    }

    this._viewDirty = true;
  }

  /**
   * Zoom the camera toward/away from Earth centre.
   * @param delta - negative = zoom in, positive = zoom out
   */
  zoom(delta: number): void {
    const minRadius = 1.05; // just above surface
    const maxRadius = 50.0;
    const radius = Cartesian3.magnitude(this.position);
    const newRadius = CesiumMath.clamp(radius + delta, minRadius, maxRadius);
    const scale = newRadius / radius;
    this.position.x *= scale;
    this.position.y *= scale;
    this.position.z *= scale;
    this._viewDirty = true;
  }

  /**
   * Fly the camera to a destination along a great-circle path with a
   * parabolic altitude arch, animating over `duration` seconds (default 3).
   * Any user input (drag / wheel / pinch) or a new `flyTo` cancels the
   * current flight and invokes its `cancel` callback.
   */
  flyTo(options: FlyToOptions): void {
    this.cancelFlight();
    this._flight = new CameraFlightPath(this, options);
  }

  /** True while a camera flight is in progress. */
  get inFlight(): boolean {
    return this._flight !== null && !this._flight.finished;
  }

  /** Abort the active flight (if any) and fire its `cancel` callback. */
  cancelFlight(): void {
    const flight = this._flight;
    if (!flight) return;
    this._flight = null;
    flight.cancelCallback?.();
  }

  /**
   * Advance the active flight by `dtSeconds`.  Called once per frame by the
   * Viewer render loop; evaluates the flight and writes the result into this
   * camera.
   */
  updateFlight(dtSeconds: number): void {
    const flight = this._flight;
    if (!flight) return;

    flight.update(this, dtSeconds);

    if (flight.finished) {
      this._flight = null;
      flight.completeCallback?.();
    }
  }

  /**
   * Cast a ray from the camera through the given normalised device coordinates
   * and return the first intersection with the unit sphere (the Earth).
   * @param ndcX - normalised device X in [-1, 1] (right is positive)
   * @param ndcY - normalised device Y in [-1, 1] (up is positive)
   * @returns World-space intersection point, or null if the ray misses the globe.
   */
  pickGlobe(ndcX: number, ndcY: number): Cartesian3 | null {
    const tanHalfFovY = Math.tan(this.frustum.fov * 0.5);
    const tanHalfFovX = tanHalfFovY * this.frustum.aspectRatio;

    // World-space ray direction through the clicked pixel
    const rayDir = new Cartesian3(
      this.direction.x + ndcX * tanHalfFovX * this.right.x + ndcY * tanHalfFovY * this.up.x,
      this.direction.y + ndcX * tanHalfFovX * this.right.y + ndcY * tanHalfFovY * this.up.y,
      this.direction.z + ndcX * tanHalfFovX * this.right.z + ndcY * tanHalfFovY * this.up.z,
    );
    Cartesian3.normalize(rayDir, rayDir);

    // Ray-sphere intersection: |origin + t*dir|² = 1  (Earth radius = 1)
    // → t² + 2(origin·dir)t + (|origin|² − 1) = 0
    const b = 2.0 * Cartesian3.dot(this.position, rayDir);
    const c = Cartesian3.magnitudeSquared(this.position) - 1.0;
    const discriminant = b * b - 4.0 * c;
    if (discriminant < 0) return null;

    const t = (-b - Math.sqrt(discriminant)) * 0.5;
    if (t < 0) return null;

    return new Cartesian3(
      this.position.x + t * rayDir.x,
      this.position.y + t * rayDir.y,
      this.position.z + t * rayDir.z,
    );
  }

  /**
   * Orbit the camera around a pivot point on the Earth's surface.
   * Horizontal delta rotates around the outward surface normal at the pivot
   * (heading); vertical delta tilts the camera toward/away from the horizon
   * (pitch).  Both axes pass through the pivot point so it stays fixed on
   * screen.
   * @param pivot        - 3D pivot in normalised ECEF (Earth radius = 1)
   * @param deltaHeading - heading rotation in radians (positive = rightward)
   * @param deltaPitch   - pitch rotation in radians (positive = tilt down)
   */
  orbitAroundPivot(pivot: Cartesian3, deltaHeading: number, deltaPitch: number): void {
    // Arm vector: from pivot to camera position
    const arm = new Cartesian3(
      this.position.x - pivot.x,
      this.position.y - pivot.y,
      this.position.z - pivot.z,
    );

    // ── Step 1: Heading — rotate arm and the full camera frame around the
    //   pivot's outward surface normal.  This spins the camera around the
    //   globe point while keeping the camera's orientation relative to the
    //   surface intact.
    const pivotNormal = new Cartesian3();
    Cartesian3.normalize(pivot, pivotNormal);
    Camera._rotateByAxis(arm,            pivotNormal, deltaHeading, arm);
    Camera._rotateByAxis(this.direction, pivotNormal, deltaHeading, this.direction);
    Camera._rotateByAxis(this.up,        pivotNormal, deltaHeading, this.up);
    Camera._rotateByAxis(this.right,     pivotNormal, deltaHeading, this.right);

    // ── Step 2: Pitch — rotate arm, direction and up around the camera's
    //   (already-updated) right axis.  right is the rotation axis so it does
    //   not change.  Save state first so we can cancel an over-rotation.
    const armX = arm.x, armY = arm.y, armZ = arm.z;
    const dirX = this.direction.x, dirY = this.direction.y, dirZ = this.direction.z;
    const upX  = this.up.x,        upY  = this.up.y,        upZ  = this.up.z;

    Camera._rotateByAxis(arm,            this.right, deltaPitch, arm);
    Camera._rotateByAxis(this.direction, this.right, deltaPitch, this.direction);
    Camera._rotateByAxis(this.up,        this.right, deltaPitch, this.up);

    // Reject the pitch if it would tip the camera past ~85° from nadir
    // (dot product with nadir direction < cos(85°) ≈ 0.087).
    const newPos = new Cartesian3(pivot.x + arm.x, pivot.y + arm.y, pivot.z + arm.z);
    const newNadir = new Cartesian3(-newPos.x, -newPos.y, -newPos.z);
    Cartesian3.normalize(newNadir, newNadir);
    if (Cartesian3.dot(this.direction, newNadir) < Camera._MAX_TILT_NADIR_DOT) {
      arm.x = armX; arm.y = armY; arm.z = armZ;
      this.direction.x = dirX; this.direction.y = dirY; this.direction.z = dirZ;
      this.up.x = upX;         this.up.y = upY;         this.up.z = upZ;
    }

    // ── Apply new position ────────────────────────────────────────────────────
    this.position.x = pivot.x + arm.x;
    this.position.y = pivot.y + arm.y;
    this.position.z = pivot.z + arm.z;

    // Clamp to minimum altitude (1.05 = Earth radius + 5 % safety margin above surface)
    const r = Cartesian3.magnitude(this.position);
    if (r < 1.05) {
      const scale = 1.05 / r;
      this.position.x *= scale;
      this.position.y *= scale;
      this.position.z *= scale;
    }

    this._viewDirty = true;
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  /**
   * Normalise a destination to internal ECEF units (Earth radius = 1).
   * Accepts metres-ECEF `Cartesian3` or a `{ lon, lat, height }` shortcut
   * (lon/lat in degrees, height in metres).
   */
  static resolveDestination(destination: Destination): Cartesian3 {
    const scale = 1.0 / Camera.EARTH_SCALE;
    if ('x' in destination && 'y' in destination && 'z' in destination) {
      return Cartesian3.multiplyByScalar(destination, scale, new Cartesian3());
    }
    // { lon, lat, height } — route through the WGS84 cartographic conversion
    // so it matches `Cartesian3.fromDegrees` exactly.
    const metres = Cartographic.fromDegrees(
      destination.lon,
      destination.lat,
      (destination.height ?? 0.0) / Camera.EARTH_SCALE,
    ).toCartesian();
    return Cartesian3.multiplyByScalar(metres, scale, new Cartesian3());
  }

  /**
   * Derive the heading / pitch (radians) implied by the camera's current
   * direction vector relative to the local east-north-up frame at its position.
   * heading: 0 = north, positive clockwise (toward east).
   * pitch:   0 = horizon, negative = down, ±π/2 = zenith / nadir.
   */
  _deriveHeadingPitch(): { heading: number; pitch: number } {
    const up = Cartesian3.normalize(this.position, new Cartesian3());
    const east = new Cartesian3();
    const north = new Cartesian3();
    Camera._eastNorthUp(up, east, north);

    const pitch = Math.asin(CesiumMath.clamp(Cartesian3.dot(this.direction, up), -1, 1));
    const heading = Math.atan2(
      Cartesian3.dot(this.direction, east),
      Cartesian3.dot(this.direction, north),
    );
    return { heading, pitch };
  }

  /**
   * Rebuild direction / up / right from a heading / pitch / roll triple at the
   * camera's current position.  heading = 0 / pitch = 0 looks north along the
   * horizon; negative pitch tilts the look direction downward.
   */
  _applyOrientation(heading: number, pitch: number, roll: number): void {
    const up = Cartesian3.normalize(this.position, new Cartesian3());
    const east = new Cartesian3();
    const north = new Cartesian3();
    Camera._eastNorthUp(up, east, north);

    // Heading: rotate north / east around the up axis (clockwise seen from
    // above, so positive heading turns the look direction toward east).
    const lookH = Camera._rotateByAxis(north, up, -heading, new Cartesian3());
    const rightH = Camera._rotateByAxis(east, up, -heading, new Cartesian3());

    // Pitch around the horizontal right axis: negative = look down.
    const dir = Camera._rotateByAxis(lookH, rightH, pitch, new Cartesian3());
    const camUp = Camera._rotateByAxis(up, rightH, pitch, new Cartesian3());

    // Roll around the final look direction.
    const right = new Cartesian3();
    Cartesian3.cross(dir, camUp, right);
    Cartesian3.normalize(right, right);
    const finalUp = new Cartesian3();
    if (roll !== 0.0) {
      Camera._rotateByAxis(right, dir, roll, right);
      Camera._rotateByAxis(camUp, dir, roll, finalUp);
    } else {
      finalUp.x = camUp.x; finalUp.y = camUp.y; finalUp.z = camUp.z;
    }
    Cartesian3.normalize(dir, this.direction);
    Cartesian3.normalize(finalUp, this.up);
    Cartesian3.normalize(right, this.right);

    this._viewDirty = true;
  }

  /**
   * Compute the local east / north unit vectors at a surface up vector
   * (geocentric normal).  Falls back to a fixed east at the poles.
   */
  private static _eastNorthUp(up: Cartesian3, east: Cartesian3, north: Cartesian3): void {
    // east = normalize(cross(UNIT_Z, up)) = normalize(-up.y, up.x, 0)
    const ex = -up.y;
    const ey = up.x;
    const eMag = Math.hypot(ex, ey);
    if (eMag < 1e-6) {
      // Degenerate at the poles — fixed fallback (same as _rebuildFrame).
      east.x = 1; east.y = 0; east.z = 0;
    } else {
      east.x = ex / eMag; east.y = ey / eMag; east.z = 0;
    }
    Cartesian3.cross(up, east, north);
    Cartesian3.normalize(north, north);
  }

  /**
   * Rotate vector v around unit axis by angle using Rodrigues' formula.
   * result may alias v.
   */
  private static _rotateByAxis(
    v: Cartesian3, axis: Cartesian3, angle: number, result: Cartesian3,
  ): Cartesian3 {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const dot = Cartesian3.dot(axis, v);
    const cross = new Cartesian3();
    Cartesian3.cross(axis, v, cross);
    // Compute into temporaries so result can safely alias v
    const rx = v.x * c + cross.x * s + axis.x * dot * (1 - c);
    const ry = v.y * c + cross.y * s + axis.y * dot * (1 - c);
    const rz = v.z * c + cross.z * s + axis.z * dot * (1 - c);
    result.x = rx; result.y = ry; result.z = rz;
    return result;
  }

  /**
   * Rebuild the camera's direction/right/up vectors so the camera looks
   * toward the Earth centre (nadir) from its current position.
   */
  private _rebuildFrame(): void {
    // Nadir direction: from camera position toward Earth centre
    const nadir = new Cartesian3(
      -this.position.x, -this.position.y, -this.position.z,
    );
    Cartesian3.normalize(nadir, nadir);

    // Right vector: perpendicular to nadir and world-up (north pole)
    const worldUp = new Cartesian3(0, 0, 1);
    Cartesian3.cross(nadir, worldUp, this.right);
    if (Cartesian3.magnitude(this.right) < 1e-6) {
      // Degenerate at poles — fall back to a fixed right vector
      this.right.x = 1; this.right.y = 0; this.right.z = 0;
    } else {
      Cartesian3.normalize(this.right, this.right);
    }

    // Camera looks toward Earth centre
    this.direction.x = nadir.x;
    this.direction.y = nadir.y;
    this.direction.z = nadir.z;

    // Camera up stays perpendicular to direction and right
    Cartesian3.cross(this.right, this.direction, this.up);
    Cartesian3.normalize(this.up, this.up);

    this._viewDirty = true;
  }

  private _updateViewMatrix(): void {
    const target = new Cartesian3(
      this.position.x + this.direction.x,
      this.position.y + this.direction.y,
      this.position.z + this.direction.z
    );
    Matrix4.lookAt(this.position, target, this.up, this._viewMatrix);
    this._viewDirty = false;
    // Invalidate combined matrix
    Matrix4.multiply(this._projMatrix, this._viewMatrix, this._viewProjMatrix);
  }

  private _updateProjMatrix(): void {
    Matrix4.perspective(
      this.frustum.fov,
      this.frustum.aspectRatio,
      this.frustum.near,
      this.frustum.far,
      this._projMatrix
    );
    this._projDirty = false;
    Matrix4.multiply(this._projMatrix, this._viewMatrix, this._viewProjMatrix);
  }
}

export default Camera;
