import { Matrix4 } from '../math/Matrix4';
import { Cartesian3 } from '../math/Cartesian3';
import { CesiumMath } from '../math/CesiumMath';
import { Cartographic } from '../math/Cartographic';
import { EasingFunction } from '../math/EasingFunction';
import { Measure } from '../analysis/Measure';
import { CameraFlightPath } from './CameraFlightPath';

/**
 * Minimal structural contract the camera needs from the globe for
 * terrain-aware picking. `Globe` satisfies it through its `terrainLayer`
 * (attached by the terrain stack); kept structural so `Camera` does not
 * depend on the terrain implementation.
 */
export interface CameraGlobeSurface {
  /**
   * Terrain height sampler, present while terrain tiles are streaming.
   * `sampleHeight(lonDeg, latDeg)` returns the terrain elevation in metres,
   * or `0` when no tile covers the location.
   */
  terrainLayer?: { sampleHeight(lon: number, lat: number): number } | null;
}

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

  /**
   * Globe surface reference used by `pickGlobe` to snap hits onto the terrain.
   * Assigned by `Scene` at construction; `null` (no terrain sampling) keeps
   * the legacy ellipsoid-only picking behaviour.
   */
  globe: CameraGlobeSurface | null = null;

  /** Minimum dot product of camera direction with nadir before pitch is rejected (~85° max tilt). */
  private static readonly _MAX_TILT_NADIR_DOT = Math.cos(CesiumMath.toRadians(85));

  /** Scratch for the geodetic-altitude conversion in `_updateProjMatrix` (metres ECEF). */
  private static readonly _scratchEcefM = new Cartesian3();
  /** Scratch cartographic result for `_updateProjMatrix` (height in normalised units). */
  private static readonly _scratchCarto = new Cartographic();

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
   * Zoom the camera, anchored at the surface point under the view centre.
   *
   * The camera moves along the anchor→camera line by an exponential factor,
   * so the screen composition is preserved while zooming — targets stay in
   * frame and simply grow. A purely radial (toward-Earth-centre) zoom keeps
   * pitch fixed and slides the target off the top of the screen as the
   * camera descends (94 示例「一缩放就看不到」的根因).
   *
   * @param delta - natural-log zoom factor: exp(delta) is the distance ratio
   *                (e.g. -0.22 ≈ move to 80 % = zoom in one notch)
   */
  zoom(delta: number): void {
    const pos = this.position;
    const s = Math.exp(delta);
    // 锚 = 视线中心与「当前纬度地表球」的交点；看天空未命中时退回径向缩放
    const anchor = this._pickSurfaceCentre();
    let next: Cartesian3;
    if (anchor) {
      next = new Cartesian3(
        anchor.x + (pos.x - anchor.x) * s,
        anchor.y + (pos.y - anchor.y) * s,
        anchor.z + (pos.z - anchor.z) * s,
      );
    } else {
      next = Cartesian3.multiplyByScalar(pos, s, new Cartesian3());
    }
    // 贴地/上限钳制：沿径向抬回安全高度（触底时锚已近在脚下，跳变可忽略）
    const r2 = Cartesian3.magnitude(next);
    const minR = Camera.surfaceRadius(next) + 0.00002;
    const clamped = CesiumMath.clamp(r2, minR, 50.0);
    if (clamped !== r2) {
      const k = clamped / r2;
      next.x *= k;
      next.y *= k;
      next.z *= k;
    }
    this.position.x = next.x;
    this.position.y = next.y;
    this.position.z = next.z;
    this._viewDirty = true;
  }

  /**
   * Intersection of the view-centre ray with the "surface-radius sphere" at
   * the camera's latitude (WGS84 fast approximation). Used as the zoom
   * anchor; null when the view centre misses the ground.
   */
  private _pickSurfaceCentre(): Cartesian3 | null {
    const dir = Cartesian3.normalize(this.direction, new Cartesian3());
    const surface = Camera.surfaceRadius(this.position);
    // |o + t·d|² = surface² → t² + 2(o·d)t + (|o|² − surface²) = 0
    const b = 2.0 * Cartesian3.dot(this.position, dir);
    const c = Cartesian3.magnitudeSquared(this.position) - surface * surface;
    const disc = b * b - 4.0 * c;
    if (disc < 0.0) return null;
    const t = (-b - Math.sqrt(disc)) * 0.5;
    if (t <= 0.0) return null;
    return new Cartesian3(
      this.position.x + dir.x * t,
      this.position.y + dir.y * t,
      this.position.z + dir.z * t,
    );
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

    const hit = new Cartesian3(
      this.position.x + t * rayDir.x,
      this.position.y + t * rayDir.y,
      this.position.z + t * rayDir.z,
    );
    return Camera._clampHitToTerrain(hit, this.globe?.terrainLayer);
  }

  /**
   * Snap an ellipsoid hit onto the terrain surface with a fixed-point
   * iteration (max 3 steps): ECEF → geodetic lon/lat → `sampleHeight` →
   * surface ECEF at `1 + h / 6378137`, repeated until the sampled height
   * stops changing (< 1 cm). Terrain elevations shift the surface point,
   * which slightly moves its geodetic latitude, so one sample is not always
   * enough for full convergence.
   *
   * When no terrain layer is attached, or the sampler reports no data (0)
   * at the hit location, the original ellipsoid hit is returned unchanged.
   */
  private static _clampHitToTerrain(
    hit: Cartesian3,
    sampler: { sampleHeight(lon: number, lat: number): number } | null | undefined,
  ): Cartesian3 {
    if (!sampler) return hit;

    let point = hit;
    let prevHeight = Number.NaN;
    for (let i = 0; i < 3; i++) {
      const height = Camera._sampleTerrainHeight(point, sampler);
      if (!Number.isFinite(height) || height === 0.0) {
        // No terrain coverage here: keep the ellipsoid hit.
        return i === 0 ? hit : point;
      }
      if (Math.abs(height - prevHeight) < 0.01) break; // converged
      prevHeight = height;
      point = Measure.clampToTerrain(point, (lon, lat) => sampler.sampleHeight(lon, lat));
    }
    return point;
  }

  /**
   * Terrain elevation (metres) at a normalised ECEF point. Uses the engine
   * surface convention (y/z squashed by the WGS84 polar ratio — same as
   * `Measure.clampToTerrain` / the terrain tile grid): unsquash the
   * direction, then read the parametric lon/lat in degrees.
   */
  private static _sampleTerrainHeight(
    point: Cartesian3,
    sampler: { sampleHeight(lon: number, lat: number): number },
  ): number {
    const ratio = Measure.POLAR_RATIO;
    const inv = 1.0 / Cartesian3.magnitude(point);
    const ux = point.x * inv;
    const uy = (point.y * inv) / ratio;
    const uz = (point.z * inv) / ratio;
    const lon = Math.atan2(uy, ux);
    const lat = Math.atan2(uz, Math.hypot(ux, uy));
    return sampler.sampleHeight(
      CesiumMath.toDegrees(lon),
      CesiumMath.toDegrees(lat),
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

    // Clamp to minimum altitude：贴当前纬度椭球地表 + ~127m 余量
    // （绝对 1.05 是单位球假设，椭球世界一拖拽就会把相机推到球面上方 319km）
    const minR = Camera.surfaceRadius(this.position) + 0.00002;
    const r = Cartesian3.magnitude(this.position);
    if (r < minR) {
      const scale = minR / r;
      this.position.x *= scale;
      this.position.y *= scale;
      this.position.z *= scale;
    }

    this._viewDirty = true;
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  /**
   * WGS84 椭球地表矢长（归一化单位）在给定方向上的快速近似：
   * 1 − (e²/2)·sin²φ，误差 < 20 m。引擎的「椭球 ECEF 线性归一化」世界中
   * 纬度 30.5° 地表 ≈ 0.99914（单位球面下方 ~5.5 km）——任何「地表高度」
   * 判断（滚轮下限、拖拽钳制）都必须用它，严禁假设地表 = 1。
   */
  static surfaceRadius(position: Cartesian3): number {
    const r = Cartesian3.magnitude(position);
    if (r < 1e-9) return 1.0;
    const sinLat = CesiumMath.clamp(position.z / r, -1.0, 1.0);
    return 1.0 - 0.003349 * sinLat * sinLat;
  }

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
    // Dynamic near/far: at high orbit keep the 0.005 near baseline (depth
    // precision first); below ~32 km altitude shrink near proportionally so
    // low-altitude roams (e.g. 4-8 km cruises) don't near-clip the ground in
    // front of the camera. far is tightened alongside to keep the near/far
    // ratio ≲ 2000 for depth-buffer precision. frustum.near/far are kept in
    // sync so effect/pick passes reading them stay consistent with the matrix.
    //
    // Altitude is the GEODETIC height above the WGS84 ellipsoid (normalised),
    // not `|position| - 1.0`: that shortcut equals the height only at the
    // equator. At mid-latitude the ellipsoid surface itself has |p| < 1
    // (0.99914 at lat 30.5°), so every low-altitude camera clamped to
    // altitude 0 and got stuck at the near floor (1e-5 = 63.8 m) — the near
    // plane then sat ABOVE a drone flying at 4-40 m AGL and near-clipped the
    // entire ground in sensor views; the clipped ground exposed the
    // 31.9 km-deep tile-skirt curtains, whose interior fragments surfaced as
    // the constant ~13 km "ground" depth band (SensorCam W4 defect).
    Cartesian3.multiplyByScalar(this.position, Camera.EARTH_SCALE, Camera._scratchEcefM);
    const altitude = Math.max(
      Cartographic.fromCartesian(Camera._scratchEcefM, Camera._scratchCarto).height,
      0.0,
    );
    // Near floor 1e-6 (≈6.4 m): keeps the ground unclipped for nadir/oblique
    // sensors at 6-40 m AGL. The historical 1e-5 floor (63.8 m) sat ABOVE such
    // flights — every ground pixel nearer than 63.8 m view-z was near-clipped,
    // the clipped rays continued underground and pierced the 31.9 km-deep tile
    // skirt curtains, which surfaced as the constant ~13 km "ground" depth band
    // (depth32float keeps <1 m linearised error at 300 m+ with n=1e-6, f=0.02).
    this.frustum.near = Math.min(0.005, Math.max(altitude * 0.1, 1e-6));
    this.frustum.far = Math.max(this.frustum.near * 2000, Math.min(100.0, altitude * 150 + 0.02));
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
