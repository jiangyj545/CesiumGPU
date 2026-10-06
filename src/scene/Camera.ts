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
  /**
   * 上次 _updateProjMatrix 时的相机位置快照。near/far 依赖海拔（随 position
   * 变化），但 position 写点众多（setView/rotate/zoom/飞行路径直写坐标），
   * 逐一补 _projDirty 必有漏网——快照比对把「位置变了→投影失效」收敛到
   * 矩阵 getter 单点（2026-09-30 实锤：96 页 7m 相机 near 冻结在启动 450m
   * 算出的 45m，近场海面整片近平面裁掉、透出卫星底图灰墙）。
   */
  private _projPos = new Cartesian3(NaN, NaN, NaN);

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

  /** WGS84 极半径比 b/a：引擎线性归一化世界中 y/z 的压扁系数（同 Globe/EllipsoidGeometry/TerrainProvider）。 */
  static readonly WGS84_POLAR_RATIO = 6356752.3142 / 6378137.0;
  /**
   * 防穿地钳制的贴地净空（归一化米）：2 m。旧值 0.00002 = 127.6 m，把贴地
   * /水下相机抬到百米高空（96 页水下滚轮一格 +11.5 km、飞行中段地板 32 km
   * 同族根因）。zoom / orbitAroundPivot / CameraFlightPath 共用。
   */
  static readonly SURFACE_CLEARANCE = 2.0 / 6378137.0;
  /**
   * 水下（球内）相机滚轮缩放的步长基准上限（归一化米）：100 m。视线前向
   * 出射距离（头顶水面）本身只有米级；朝海床看时穿球出射 ~6371 km，若
   * 不设上限，一格滚轮按出射距离比例走几百公里。见 zoom() 球内分支。
   */
  static readonly ZOOM_DOLLY_MAX = 100.0 / 6378137.0;

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
    if (this._projDirty || !this._projPositionMatches()) this._updateProjMatrix();
    return this._projMatrix;
  }

  get viewProjectionMatrix(): Matrix4 {
    if (this._viewDirty || this._projDirty || !this._projPositionMatches()) {
      if (this._viewDirty) this._updateViewMatrix();
      // 位置快照失配也算 proj 脏（near/far 随海拔变）——无条件重跑
      this._updateProjMatrix();
      Matrix4.multiply(this._projMatrix, this._viewMatrix, this._viewProjMatrix);
    }
    return this._viewProjMatrix;
  }

  /** position 是否与上次投影更新时一致（NaN 快照首轮恒失配 → 必重算）。 */
  private _projPositionMatches(): boolean {
    const p = this._projPos;
    return p.x === this.position.x && p.y === this.position.y && p.z === this.position.z;
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
    // 锚 = 视线中心与「当前纬度地表球」的交点；看天空未命中时：球外相机
    // 退回径向缩放，球内（水下/贴地）相机改沿视线推拉（见 else 分支注释）。
    const anchor = this._pickSurfaceCentre();
    let next: Cartesian3;
    if (anchor) {
      next = new Cartesian3(
        anchor.x + (pos.x - anchor.x) * s,
        anchor.y + (pos.y - anchor.y) * s,
        anchor.z + (pos.z - anchor.z) * s,
      );
    } else {
      // 球内相机：视线与两球面均无前向交点（c<0 被 _pickSurfaceCentre 跳过）。
      // 径向缩放在近地表是灾难——|p|≈1，×0.8 一格砍掉 0.2 个地球半径
      //（水下滚轮一格 -1273 km 实测）；而旧钳制又把相机抬出水面。改沿视线
      // 推拉：步长 = 缩放比 × 前向近出射距离（抬头 = 头顶水面米级；朝海床 =
      // 穿球 ~6371 km，钳到 ZOOM_DOLLY_MAX=100 m 保米级步长）。拉近朝目标
      // 前进、拉远后退出水，均为米级增量；球外相机保持径向语义（太空缩放）。
      const dir = Cartesian3.normalize(this.direction, new Cartesian3());
      const pd = Cartesian3.dot(pos, dir);
      const rSq = Cartesian3.magnitudeSquared(pos);
      let exitT = Infinity;
      for (const surface of [
        Camera.surfaceRadius(pos),
        Camera.wgs84Radius(pos),
      ]) {
        const c = rSq - surface * surface;
        if (c >= 0.0) continue;             // 该球外：径向语义
        const t = -pd + Math.sqrt(pd * pd - c); // 前向近出射根（球内恒 >0）
        if (t > 0.0 && t < exitT) exitT = t;
      }
      if (Number.isFinite(exitT)) {
        const step = (1.0 - s) * Math.min(exitT, Camera.ZOOM_DOLLY_MAX);
        next = new Cartesian3(pos.x + dir.x * step, pos.y + dir.y * step, pos.z + dir.z * step);
      } else {
        next = Cartesian3.multiplyByScalar(pos, s, new Cartesian3());
      }
    }
    // 贴地/上限钳制：沿径向抬回安全高度（触底时锚已近在脚下，跳变可忽略）。
    // 2026-09-30 世界判别取代双面取大：双约定并存（Globe 压扁系 / 未压扁 WGS84 系），
    // 判别依据是「正海拔更小」——错系球面必然给出穿地负海拔或虚高正海拔，真世界
    // 的高度总在米级~公里级。旧 max() 把压扁系相机（Globe 世界页 + 97 页白模贴地
    // 后）下限抬到未压扁面 +13km：5km 高度一拉近即跳 13km，压扁世界低空缩放全废。
    // 未压扁系行为不变（其 wgs84 海拔恒小于压扁系虚高 → 仍钳 wgs84Radius）。
    const r2 = Cartesian3.magnitude(next);
    const altFlat = r2 - Camera.surfaceRadius(next);
    const altWgs = r2 - Camera.wgs84Radius(next);
    const minSurface = altFlat >= 0.0 && (altWgs < 0.0 || altFlat <= altWgs)
      ? Camera.surfaceRadius(next)
      : Camera.wgs84Radius(next);
    // 防穿地下限：仅当相机当前在所选地表「之上」时生效。椭球面是海平面
    // 而非地面——水层是合法机位空间，旧下限对水下相机照常生效，把 -3 m
    // 潜水机位沿径向一把抬到错系球面上方 +11.5 km（96 页水下滚轮一格
    // 实锤 2026-09-30）。余量 2 m（SURFACE_CLEARANCE）：旧 0.00002 =
    // 127.6 m，贴地相机每被钳一次即跳百米高空。防穿地意图不变：水上
    // 相机仍被钳回地表 +2 m。
    const minR = Cartesian3.magnitude(pos) < minSurface
      ? 0.0
      : minSurface + Camera.SURFACE_CLEARANCE;
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
   * Intersection of the view-centre ray with the surface sphere at the
   * camera's latitude (WGS84 fast approximation). Used as the zoom anchor;
   * null when the view centre misses the ground.
   *
   * 双约定球面取最近正交点（2026-09-30 根因修复）：position 可能是压扁世界
   * 系（Globe/resolveDestination {lon,lat} 分支）或未压扁 WGS84 系（fromDegrees
   * 米制 ÷ EARTH_SCALE）。surfaceRadius 对未压扁坐标比真地表低 ~13km——浅视线
   * 与它的交点在几十公里外，zoom 锚过去每档把相机水平拖走数公里（97 页滚轮
   * 拉近白膜出画根因：300m/-45° 实测 tFlat=19439m vs tWgs=424m）。两球面都解、
   * 取最近正 t：真地表总在错系球面之前；相机位于某球面内部（c<0，正根是
   * 穿出点）时跳过该球面——压扁系相机在 wgs84 球面内部 ~13km，行为不变。
   */
  private _pickSurfaceCentre(): Cartesian3 | null {
    const dir = Cartesian3.normalize(this.direction, new Cartesian3());
    const b = 2.0 * Cartesian3.dot(this.position, dir);
    const r2 = Cartesian3.magnitudeSquared(this.position);
    let tMin = Infinity;
    for (const surface of [
      Camera.surfaceRadius(this.position),
      Camera.wgs84Radius(this.position),
    ]) {
      const c = r2 - surface * surface;
      if (c < 0.0) continue; // 相机在该球面内部：正根是穿出点，非可见地表
      const disc = b * b - 4.0 * c;
      if (disc < 0.0) continue; // 视线朝天，未命中该球面
      const t = (-b - Math.sqrt(disc)) * 0.5;
      if (t > 0.0 && t < tMin) tMin = t;
    }
    if (!Number.isFinite(tMin)) return null;
    return new Cartesian3(
      this.position.x + dir.x * tMin,
      this.position.y + dir.y * tMin,
      this.position.z + dir.z * tMin,
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
   * and return the first intersection with the local ellipsoid surface
   * (radius from `Camera.surfaceRadius`; underwater cameras take the
   * near-range exit root through the overhead surface).
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

    // 双约定球面取最近命中（与 _pickSurfaceCentre 同思路，2026-09-30）：
    // position 可能是压扁系（Globe/{lon,lat} 落点）或未压扁 WGS84 系
    //（fromDegrees 米制 ÷ EARTH_SCALE）。旧代码硬编码 |p|=1 单位球——压扁
    // 世界中纬度地表模长 ≈0.997（31°N 实测），相机恒在拾取球内部 → 近根
    // t<0 全拒 → pickGlobe 永远 null → 中键拖动无法启动（赤道外全纬度失效）。
    // 相机在某球面内部（水下）：出射根仅在近距（头顶水面，几米级）有效，
    // 对跖出射（~2 万 km）以 64 km 上限剔除。
    const b = 2.0 * Cartesian3.dot(this.position, rayDir);
    const r2 = Cartesian3.magnitudeSquared(this.position);
    let tMin = Infinity;
    for (const surface of [
      Camera.surfaceRadius(this.position),
      Camera.wgs84Radius(this.position),
    ]) {
      const c = r2 - surface * surface;
      const disc = b * b - 4.0 * c;
      if (disc < 0.0) continue; // 视线未命中该球面
      const sq = Math.sqrt(disc);
      let t = (-b - sq) * 0.5;          // 球外相机：可见侧近根
      if (t < 0) {
        const exitT = (-b + sq) * 0.5;  // 球内（水下）相机：近距出射根
        if (exitT < 0 || exitT > 0.01) continue;
        t = exitT;
      }
      if (t < tMin) tMin = t;
    }
    if (!Number.isFinite(tMin)) return null;
    const t = tMin;

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
    const wasAbove = Cartesian3.magnitude(this.position) >= Camera.surfaceRadius(this.position);
    this.position.x = pivot.x + arm.x;
    this.position.y = pivot.y + arm.y;
    this.position.z = pivot.z + arm.z;

    // Clamp to minimum altitude：贴当前纬度椭球地表 + 2 m 余量
    // （绝对 1.05 是单位球假设，椭球世界一拖拽就会把相机推到球面上方 319km）。
    // 仅当轨道步之前相机在地表之上时生效：水层是合法机位空间，旧 +127.6 m
    //（0.00002）余量把中键拖拽的水下相机一把抬出水面（96 页潜水视角）；
    // 水上相机被轨道步压进椭球时仍抬回地表 +2 m（防穿地意图不变）。
    const surface = Camera.surfaceRadius(this.position);
    const minR = surface + Camera.SURFACE_CLEARANCE;
    const r = Cartesian3.magnitude(this.position);
    if (wasAbove && r < minR) {
      const scale = minR / r;
      this.position.x *= scale;
      this.position.y *= scale;
      this.position.z *= scale;
    }

    this._viewDirty = true;
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  /**
   * 引擎世界（单位球方向向量，y/z × WGS84_POLAR_RATIO 线性归一化）中，
   * 位置所在经纬度的地表矢长。先把压扁位置还原成单位方向 u，再取
   * √(ux² + ρ²(1−ux²))（地表点 = (ux, uy·ρ, uz·ρ)）。
   *
   * 此前的近似 1 − (e²/2)·sin²φ 是「旋转椭球」公式，而本引擎世界不是旋转
   * 椭球——y 压扁使地表矢长随经度变化（如 121.65°E/31.14°N 实际 0.99732，
   * 旧公式给 0.99910，偏高 ~11.3 km），会让 zoom/拖拽下限与缩放锚点整体
   * 悬空。严禁假设地表 = 1，也严禁用旋转椭球近似。
   */
  static surfaceRadius(position: Cartesian3): number {
    const r = Cartesian3.magnitude(position);
    if (r < 1e-9) return 1.0;
    const rho = Camera.WGS84_POLAR_RATIO;
    const inv = 1.0 / r;
    const ux = position.x * inv;
    const uy = (position.y * inv) / rho;
    const uz = (position.z * inv) / rho;
    const n = Math.sqrt(ux * ux + uy * uy + uz * uz); // ≈1，消浮点漂移
    const cx2 = ((ux / n) * (ux / n));                // cosφcosλ 分量的平方
    return Math.sqrt(cx2 + rho * rho * (1.0 - cx2));
  }

  /**
   * 未压扁 WGS84 椭球表面矢长（归一化）——数值法：沿单位方向解椭球
   * x²/a² + y²/a² + z²/b² = 1 的 t 倍（a=1, b=rho）。与 surfaceRadius（压扁世界
   * 公式）配对用于 zoom 的双面取大钳制，见 zoom() 内注释（2026-09-29）。
   */
  static wgs84Radius(position: Cartesian3): number {
    const r = Cartesian3.magnitude(position);
    if (r < 1e-9) return 1.0;
    const rho = Camera.WGS84_POLAR_RATIO;
    const ux = position.x / r, uy = position.y / r, uz = position.z / r;
    return 1.0 / Math.sqrt(ux * ux + uy * uy + (uz * uz) / (rho * rho));
  }

  /**
   * 左键拖动轨道灵敏度（弧度/像素）。贴指公式：一像素 = 视野高度的 1/H 份额
   * 对应的地表弧长，dθ/px = 2·h·tan(fovY/2)/H —— 低空地图贴合指尖、高空自动加速。
   *
   * 2026-09-29 根因修复：旧公式 r·0.0003 用地心距离（归一化 ≈1 恒定）与离地高度
   * 无关，恒 ≈1913 m/px —— 300m 低空拖 12 步白膜（500m 城市）完全出画剩农田，
   * 97 页「一拖动就看不到白膜」即此。
   *
   * 高度取 Cartographic 大地高（与 _updateProjMatrix 的 near/far 同源，W4 路径）：
   * 矢长差 |p|−surface 在两系都不准——米制 fromDegrees 系高估 ~3 倍（矢径 vs 法向），
   * resolveDestination 压扁系直接 ≤0（首版实锤：300m/2000m 灵敏度同为贴地下限）。
   * 下限 12.7 米防贴地时零速卡死。
   */
  static orbitSensitivity(
    position: Cartesian3,
    fovY: number,
    viewportHeightPx: number,
  ): number {
    Cartesian3.multiplyByScalar(position, Camera.EARTH_SCALE, Camera._scratchEcefM);
    // fromCartesian().height 已是归一化高度（内部米制 ÷ EARTH_SCALE）——勿再除。
    // 2026-09-30 压扁世界分支：压扁系相机（Globe 世界页）的大地高恒 ~-13km
    // （被错系逆解），旧逻辑全部落到 12.7m 下限 → 拖拽灵敏度冻结。压扁系
    // 矢径差 |p|−surfaceRadius 才是真离地；「正海拔更小」判别（与 zoom
    // 钳制同款），未压扁系（大地高更小）行为不变。
    let h = Cartographic.fromCartesian(Camera._scratchEcefM, Camera._scratchCarto).height;
    const altFlatRad = Cartesian3.magnitude(position) - Camera.surfaceRadius(position);
    if (altFlatRad >= 0.0 && (h < 0.0 || altFlatRad < h)) h = altFlatRad;
    h = Math.max(h, 12.7 / Camera.EARTH_SCALE);
    const H = Math.max(viewportHeightPx, 1);
    return (2 * h * Math.tan(fovY * 0.5)) / H;
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
    // { lon, lat, height } — 与引擎世界地表同约定：Globe/EllipsoidGeometry/
    // ecefFromLonLatHeight/OceanFFT/FloodWater 全是「单位球方向向量 ×
    // (1 + h/a)，y/z 再乘 WGS84 极半径比」的线性归一化世界。此前走
    // Cartographic.toCartesian（真 WGS84 大地转换），其地表悬在压扁世界
    // 地表上方 ~11 km（随经度变化）：setView/flyTo 的 height 全被抬到
    // 等效 ~12 km——96/92 示例低空海面出画的根因。
    const lonRad = CesiumMath.toRadians(destination.lon);
    const latRad = CesiumMath.toRadians(destination.lat);
    const hScale = 1.0 + (destination.height ?? 0.0) / Camera.EARTH_SCALE;
    const cosLat = Math.cos(latRad);
    const rho = Camera.WGS84_POLAR_RATIO;
    return new Cartesian3(
      cosLat * Math.cos(lonRad) * hScale,
      cosLat * Math.sin(lonRad) * rho * hScale,
      Math.sin(latRad) * rho * hScale,
    );
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
    // ── 海拔（归一化）：双约定并存（Globe/ecefFromLonLatHeight 压扁系 vs
    // 传感器/Cartesian3.fromDegrees 米制系），两组候选取「绝对值小」者——
    // 错系的候选必然离谱（压扁坐标喂 WGS84 逆解在本纬度 ~-11 km；米制
    // 坐标套压扁面公式 ~+17 km），真海拔总是更近 0 的那个（zoom 的
    // surfaceRadius/wgs84Radius 双面取大同思路）。旧实现只走 Cartographic：
    // 压扁系相机（96 页 demo 全部机位）海拔恒 -11 km 量级 → clamp 0。
    const rNorm = Cartesian3.magnitude(this.position);
    const altFlattened = rNorm - Camera.surfaceRadius(this.position);
    Cartesian3.multiplyByScalar(this.position, Camera.EARTH_SCALE, Camera._scratchEcefM);
    const altWgs84 = Cartographic.fromCartesian(Camera._scratchEcefM, Camera._scratchCarto).height;
    const altitude = Math.max(
      Math.abs(altFlattened) <= Math.abs(altWgs84) ? altFlattened : altWgs84,
      0.0,
    );
    // Near floor 3e-7（≈1.9 m）：贴水机位（~7 m）与潜水（~-3 m）要求近场
    // 水面/仰视水面进画——1e-6（6.4 m）地板会把 -3 m 相机正上方 3 m 处的
    // 水面近平面裁掉；无人机 6-40 m AGL 场景 altitude*0.1 ≥ 6e-7 量级仍受
    // 地板保护（历史 1e-5 地板穿地实锤见上）。far 同步 ≥2000:1。
    this.frustum.near = Math.min(0.005, Math.max(altitude * 0.1, 3e-7));
    this.frustum.far = Math.max(this.frustum.near * 2000, Math.min(100.0, altitude * 150 + 0.02));
    // 位置快照：getter 据此判「海拔变了 → 投影失效」（见 _projPos 注释）
    this._projPos.x = this.position.x;
    this._projPos.y = this.position.y;
    this._projPos.z = this.position.z;
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
