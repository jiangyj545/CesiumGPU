import { Cartesian3 } from '../math/Cartesian3';
import { isProperty, type PropertyLike } from '../dataSource/CallbackProperty';

/** One sampled track point. */
export interface PathSample {
  /** Sample time in seconds (engine clock). */
  time: number;
  /** Position in normalised ECEF. */
  position: Cartesian3;
}

/**
 * PathHistory - time-tagged ring buffer of past entity positions, powering
 * `Entity.path` (trail rendering). Pure data structure, no GPU / DOM
 * dependencies, fully unit-testable.
 *
 * Samples are appended with {@link push}; {@link trail} returns the samples
 * younger than `trailTime` and prunes everything older (amortised O(1)
 * cleanup per query).
 */
export class PathHistory {
  private _samples: PathSample[] = [];

  /** Append a sample (call once per frame with the resolved entity position). */
  push(time: number, position: Cartesian3): void {
    this._samples.push({ time, position: new Cartesian3(position.x, position.y, position.z) });
  }

  /** Total stored samples (including pruned-eligible ones until the next query). */
  get length(): number {
    return this._samples.length;
  }

  /**
   * Samples with `time >= now - trailTime`, oldest first. Prunes older
   * samples from the buffer. `now` defaults to the newest sample time.
   */
  trail(trailTime: number, now?: number): PathSample[] {
    if (this._samples.length === 0) return [];
    const cutoff = (now ?? this._samples[this._samples.length - 1].time) - trailTime;
    let first = 0;
    while (first < this._samples.length - 1 && this._samples[first].time < cutoff) first++;
    if (first > 0) this._samples = this._samples.slice(first);
    return this._samples;
  }

  /** Drop everything (entity hidden / path removed). */
  clear(): void {
    this._samples.length = 0;
  }
}

/**
 * Sample `leadTime` seconds of future positions from a dynamic position
 * Property by evaluating it at `now + k·resolution` (k = 0..steps). Static
 * values (non-Property) yield a single sample — the current value. The last
 * sample is clamped to exactly `now + leadTime` so the lead segment always
 * ends on the horizon regardless of rounding.
 */
export function sampleLeadPositions(
  position: PropertyLike<Cartesian3> | Cartesian3,
  now: number,
  leadTime: number,
  resolution = 1.0
): Cartesian3[] {
  if (!isProperty(position)) return [position];
  const step = resolution > 0 ? resolution : 1.0;
  const out: Cartesian3[] = [];
  for (let t = 0; t < leadTime - 1e-9; t += step) {
    const v = position.getValue(now + t);
    if (v) out.push(v);
  }
  const horizon = position.getValue(now + leadTime);
  if (horizon) out.push(horizon);
  return out;
}
