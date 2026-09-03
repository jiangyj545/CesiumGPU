/**
 * EasingFunction - easing curves for animations (e.g. camera flyTo),
 * matching Cesium's EasingFunction naming.
 */
export namespace EasingFunction {
  /** Easing callback: maps input t in [0, 1] to output in [0, 1]. */
  export type Callback = (t: number) => number;

  /** Linear interpolation. */
  export const LINEAR_NONE: Callback = (t) => t;

  /** Quadratic ease-in-out. */
  export const QUADRATIC_IN_OUT: Callback = (t) =>
    t < 0.5 ? 2.0 * t * t : 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0;

  /** Cubic ease-in-out. */
  export const CUBIC_IN_OUT: Callback = (t) =>
    t < 0.5 ? 4.0 * t ** 3 : 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0;

  /** Quintic ease-in-out. */
  export const QUINTIC_IN_OUT: Callback = (t) =>
    t < 0.5 ? 16.0 * t ** 5 : 1.0 - (-2.0 * t + 2.0) ** 5 / 2.0;

  /** Sinusoidal ease-in-out. */
  export const SINUSOIDAL_IN_OUT: Callback = (t) => -(Math.cos(Math.PI * t) - 1.0) / 2.0;
}

export default EasingFunction;
