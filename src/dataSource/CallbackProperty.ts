import { Event } from './Event';

/**
 * Minimal structural contract of a Cesium-style `Property`: anything with a
 * `getValue(time?)` accessor can be assigned to Entity graphics fields
 * (`position`, `orientation`, `polyline.positions`, `label.text`, ...).
 */
export interface PropertyLike<T> {
  /** Sample the property value at the given time (optional for engine clocks that tick in plain seconds). */
  getValue(time?: unknown): T;
}

/** A value assigned to an Entity field: either a raw value or a Property. */
export type PropertyOrValue<T> = T | PropertyLike<T>;

/** Return true when the value is a Property (has a callable getValue). */
export function isProperty<T>(value: PropertyOrValue<T> | undefined): value is PropertyLike<T> {
  return (
    value !== null &&
    value !== undefined &&
    typeof (value as PropertyLike<T>).getValue === 'function'
  );
}

/**
 * Resolve an Entity field value: Properties are sampled via `getValue(time)`,
 * raw values pass through unchanged. `undefined` stays `undefined`.
 */
export function resolveProperty<T>(
  value: PropertyOrValue<T> | undefined,
  time?: unknown
): T | undefined {
  if (!isProperty(value)) return value as T | undefined;
  return value.getValue(time);
}

/**
 * CallbackProperty - a Property whose value is produced by a user callback,
 * mirroring Cesium's `CallbackProperty`.
 *
 * The callback runs every time the value is sampled (once per frame per
 * consumer when attached to an Entity). Mark the property `isConstant` when
 * the value never changes so renderers can skip re-sampling it.
 *
 * ```typescript
 * const t = { v: 0 };
 * const position = new CallbackProperty(() => nextPosition(t.v));
 * viewer.scene.entities.add(new Entity({ position, point: { pixelSize: 8 } }));
 * ```
 */
export class CallbackProperty<T> implements PropertyLike<T> {
  private _callback: (time?: unknown) => T;
  private _isConstant: boolean;

  /**
   * Raised when the property definition changes (callback / constancy
   * replaced). Per-frame value changes are NOT raised — Cesium semantics:
   * consumers poll `getValue` each frame instead.
   */
  readonly definitionChanged: Event<[CallbackProperty<T>]> = new Event();

  constructor(callback: (time?: unknown) => T, isConstant = false) {
    this._callback = callback;
    this._isConstant = isConstant;
  }

  /** True when the value never changes (renderers may sample once). */
  get isConstant(): boolean {
    return this._isConstant;
  }

  /** Replace the sampling callback. */
  setCallback(callback: (time?: unknown) => T): void {
    this._callback = callback;
    this.definitionChanged.raiseEvent(this);
  }

  /** Sample the value at the given time (time is passed through to the callback). */
  getValue(time?: unknown): T {
    return this._callback(time);
  }
}

export default CallbackProperty;
