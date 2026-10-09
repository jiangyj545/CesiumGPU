import { Cartesian3 } from '../math/Cartesian3';
import { CesiumMath } from '../math/CesiumMath';
import type { Entity } from '../dataSource/Entity';
import { resolveProperty } from '../dataSource/CallbackProperty';
import type { Camera } from './Camera';
import { HeadingPitchRange } from './HeadingPitchRange';

/**
 * EntityTracker - the follow logic behind `Viewer.trackedEntity`.
 *
 * When an entity is tracked, the camera translates in lockstep with it:
 * each `update()` applies the entity's frame-to-frame position delta to the
 * camera position, keeping the view orientation (and therefore any user
 * orbit / zoom adjustments made while tracking) intact. The relative offset
 * is implicitly "recomputed" every frame — user input moves the camera
 * relative to the entity, the next update resumes following from the new
 * offset.
 *
 * Pure math on top of `Camera` + `Entity` — no GPU/DOM, unit-testable in
 * node (mirrors `tests/scene/cameraOrbit.spec.ts`).
 */
export class EntityTracker {
  private _camera: Camera;
  private _entity: Entity | undefined;
  private _lastEntityPos: Cartesian3 | undefined;

  constructor(camera: Camera) {
    this._camera = camera;
  }

  /** Currently tracked entity (undefined = free camera). */
  get entity(): Entity | undefined {
    return this._entity;
  }

  /** Offset captured when tracking started (position + heading/pitch/range). */
  get initialOffset(): HeadingPitchRange | undefined {
    return this._initialOffset;
  }
  private _initialOffset: HeadingPitchRange | undefined;

  /**
   * Start tracking `entity`: capture the current camera→entity offset
   * (distance in metres plus heading / pitch of the camera relative to the
   * entity's local ENU frame).
   */
  track(entity: Entity): void {
    const pos = resolveProperty(entity.position);
    this._entity = entity;
    this._lastEntityPos = pos ? new Cartesian3(pos.x, pos.y, pos.z) : undefined;
    this._initialOffset = pos ? headingPitchRangeFromCamera(this._camera, pos) : undefined;
  }

  /** Stop tracking (free camera). */
  release(): void {
    this._entity = undefined;
    this._lastEntityPos = undefined;
    this._initialOffset = undefined;
  }

  /**
   * Per-frame follow step: translate the camera by the entity's position
   * delta since the last update. Call once per rendered frame (Viewer render
   * loop does this automatically).
   */
  update(time?: unknown): void {
    const entity = this._entity;
    if (!entity) return;
    const pos = resolveProperty(entity.position, time);
    if (!pos) return;
    const last = this._lastEntityPos;
    if (last) {
      this._camera.translate(
        new Cartesian3(pos.x - last.x, pos.y - last.y, pos.z - last.z)
      );
    }
    this._lastEntityPos = new Cartesian3(pos.x, pos.y, pos.z);
  }
}

/**
 * Derive the `HeadingPitchRange` of a camera relative to a target position
 * (normalised ECEF): range = distance in metres, heading / pitch of the
 * target→camera direction in the target's local ENU frame.
 */
export function headingPitchRangeFromCamera(
  camera: Camera,
  target: Cartesian3
): HeadingPitchRange {
  const offset = new Cartesian3(
    camera.position.x - target.x,
    camera.position.y - target.y,
    camera.position.z - target.z
  );
  const range = Cartesian3.magnitude(offset) * 6378137.0;

  const up = Cartesian3.normalize(target, new Cartesian3());
  // east = normalize(cross(UNIT_Z, up)); degenerate at the poles.
  const ex = -up.y, ey = up.x;
  const eMag = Math.hypot(ex, ey);
  const east =
    eMag < 1e-6
      ? new Cartesian3(1, 0, 0)
      : new Cartesian3(ex / eMag, ey / eMag, 0);
  const north = Cartesian3.cross(up, east, new Cartesian3());
  Cartesian3.normalize(north, north);

  const dir = Cartesian3.normalize(offset, new Cartesian3());
  // Negated: HeadingPitchRange pitch is the *look-down* convention (−π/2 =
  // camera overhead), while asin(dot(dir, up)) is the elevation above the
  // horizon (+π/2 = overhead).
  const pitch = -Math.asin(CesiumMath.clamp(Cartesian3.dot(dir, up), -1, 1));
  const heading = Math.atan2(Cartesian3.dot(dir, east), Cartesian3.dot(dir, north));
  return new HeadingPitchRange(heading, pitch, range);
}

export default EntityTracker;
