import { Cartesian3 } from '../math/Cartesian3';
import { CesiumMath } from '../math/CesiumMath';
import { Cartographic } from '../math/Cartographic';
import { EasingFunction } from '../math/EasingFunction';
import { Camera } from './Camera';
import type { FlyToOptions } from './Camera';

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
 * A mid-flight safety floor keeps interior points above the LOCAL surface
 * (see the class-doc note in `evaluate`); when either endpoint is below the
 * surface (dive / resurface flights) the floor is disabled — the water column
 * is a legitimate flight space, and the old absolute "unit sphere + 0.005"
 * (≈ +32 km) floor lifted every short hop ~30 km into the sky (96 页潜水
 * 切换中帧实锤 2026-09-30).
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
  private readonly _r0: number;              // start position magnitude（含椭球深度的真实矢长）
  private readonly _r1: number;              // end   position magnitude
  private readonly _peak: number;            // parabolic arch height, normalised units
  private readonly _floorOn: boolean;         // 两端点都在地表之上时启用中途安全地板

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

    // 双约定海拔（判别同 zoom 的 surfaceRadius/wgs84Radius）：压扁系机位被
    // Cartographic 逆解成 ~-11 km 虚假大地高 → meanHeight 大负值 → parabolic
    // arch 对任何小跳都 +11 km（96 页潜水/上浮中帧弧顶出水、初始入水飞行
    // 长时间悬在高空的根因 2026-09-30）。取「绝对值小」者：真海拔总在
    // 米级，错系候选总在公里级（与 _updateProjMatrix 的双候选同思路）。
    this._h0 = CameraFlightPath._altitude(startPos, startCarto.height);
    this._h1 = CameraFlightPath._altitude(endPos, endCarto.height);
    // 起终点真实矢长（resolveDestination 的椭球 ECEF 线性归一化在中纬度
    // 地表 ≈ 0.999x，即球面下数 km）。飞行半径必须插值真实矢长而非
    // 「1 + 大地高」，否则 flyTo 落点被抬到单位球面、与 setView /
    // tileset 等椭球世界错位数 km（94 示例 root 被视锥剔除的根因）。
    this._r0 = Cartesian3.magnitude(startPos);
    this._r1 = Cartesian3.magnitude(endPos);

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
    // 中途安全地板开关：两端点都在（各自方向的）压扁地表之上才启用；
    // 起或终点在水下（潜水/上浮）→ 关闭，路径不抬升（96 页）。
    this._floorOn = this._r0 >= Camera.surfaceRadius(startPos)
      && this._r1 >= Camera.surfaceRadius(endPos);

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

    // Altitude: interpolate the endpoints' true position magnitudes (which
    // carry the ellipsoid depth of the linear-ECEF/R convention) plus the
    // parabolic arch. 安全地板相对「路径方向的当地地表矢长 + 2 m」且仅在
    // 两端点都在地表之上时生效，sin(π·t) 端点归零 → t = 0/1 仍精确落在
    // 起/终点（与 setView 同坐标约定）。旧地板 min(base,1)+0.005 对任何
    // 飞行无条件抬 +32 km（含潜水小跳），见类头注释。
    const dome = Math.sin(Math.PI * t);
    const base = CesiumMath.lerp(this._r0, this._r1, t);
    const radius = this._floorOn
      ? Math.max(
        base + this._peak * dome,
        Camera.surfaceRadius(dir) + Camera.SURFACE_CLEARANCE * dome,
      )
      : base + this._peak * dome;
    camera.position.x = dir.x * radius;
    camera.position.y = dir.y * radius;
    camera.position.z = dir.z * radius;

    // ── Orientation ─────────────────────────────────────────────────────────
    const heading = this._startHeading + this._headingDelta * t;
    const pitch = CesiumMath.lerp(this._startPitch, this._endPitch, t);
    camera._applyOrientation(heading, pitch, this._endRoll);
  }

  /**
   * 双约定海拔取「绝对值小」者（归一化单位）：`geodetic` 是
   * Cartographic.fromCartesian 的大地高（未压扁 WGS84 逆解），`flat` 是
   * 压扁世界矢径差 |pos| − surfaceRadius。压扁系机位的 geodetic 是
   * ~-11 km 的虚假值（错系逆解），未压扁系机位的 flat 是 ~-17 km 的
   * 虚假值——真海拔总在米级，两个候选里绝对值小的那个总是对的。
   */
  private static _altitude(pos: Cartesian3, geodetic: number): number {
    const flat = Cartesian3.magnitude(pos) - Camera.surfaceRadius(pos);
    return Math.abs(flat) <= Math.abs(geodetic) ? flat : geodetic;
  }
}

export default CameraFlightPath;
