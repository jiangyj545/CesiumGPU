import { describe, it, expect } from 'vitest';
import { PathHistory, sampleLeadPositions } from '../../src/scene/PathHistory';
import {
  composeEntityModelMatrix,
  resolveMinimumPixelScale,
} from '../../src/scene/EntityModelMatrix';
import { Cartesian3 } from '../../src/math/Cartesian3';
import { Matrix4 } from '../../src/math/Matrix4';
import { Quaternion } from '../../src/math/Quaternion';
import { Transforms } from '../../src/math/Transforms';
import { CallbackProperty } from '../../src/dataSource/CallbackProperty';

const pos = (x: number, y = 0, z = 0) => new Cartesian3(x, y, z);

describe('PathHistory', () => {
  it('appends samples in order and prunes older-than-trailTime entries', () => {
    const h = new PathHistory();
    for (let t = 0; t <= 10; t++) h.push(t, pos(t));
    expect(h.length).toBe(11);

    const trail = h.trail(3, 10);
    expect(trail.map((s) => s.time)).toEqual([7, 8, 9, 10]);
    // Pruned: the buffer only keeps the visible window now.
    expect(h.length).toBe(4);
    // Repeated queries are stable.
    expect(h.trail(3, 10).map((s) => s.time)).toEqual([7, 8, 9, 10]);
  });

  it('keeps at least the newest sample even when it ages out', () => {
    const h = new PathHistory();
    h.push(0, pos(0));
    h.push(1, pos(1));
    const trail = h.trail(0.5, 100); // everything is older than the window
    expect(trail.map((s) => s.time)).toEqual([1]);
  });

  it('snapshots positions (later mutation of the source does not leak in)', () => {
    const h = new PathHistory();
    const p = pos(1);
    h.push(0, p);
    p.x = 99;
    expect(h.trail(1, 0)[0].position.x).toBe(1);
  });

  it('clear empties the buffer', () => {
    const h = new PathHistory();
    h.push(0, pos(0));
    h.clear();
    expect(h.trail(10, 5)).toEqual([]);
  });
});

describe('sampleLeadPositions', () => {
  it('evaluates a dynamic Property at future times up to the horizon', () => {
    const p = new CallbackProperty((t) => pos(Number(t)));
    const lead = sampleLeadPositions(p, 10, 3, 1);
    expect(lead.map((v) => v.x)).toEqual([10, 11, 12, 13]); // 10..12 + exact horizon 13
  });

  it('returns the bare value for non-Property positions', () => {
    const p = pos(5, 6, 7);
    expect(sampleLeadPositions(p, 0, 10)).toEqual([p]);
  });
});

describe('composeEntityModelMatrix', () => {
  // On the equator at lon 0: ENU basis is east=(0,1,0), north=(0,0,1), up=(1,0,0).
  const surface = pos(1, 0, 0);

  it('maps model local +Y (glTF up) onto the surface normal (Y-up fix)', () => {
    const m = composeEntityModelMatrix(surface);
    // Column 1 of the matrix = image of local (0,1,0).
    expect(m.values[4]).toBeCloseTo(1, 9); // up.x
    expect(m.values[5]).toBeCloseTo(0, 9);
    expect(m.values[6]).toBeCloseTo(0, 9);
    // Translation column places the model at the entity position.
    expect(m.values[12]).toBeCloseTo(1, 9);
    expect(m.values[13]).toBeCloseTo(0, 9);
    expect(m.values[14]).toBeCloseTo(0, 9);
  });

  it('with identity orientation equals ENU × Y-up fix directly', () => {
    const viaQuaternion = composeEntityModelMatrix(surface, Quaternion.IDENTITY.clone());
    const enu = Transforms.eastNorthUpToFixedFrame(
      Cartesian3.multiplyByScalar(surface, 6378137.0, new Cartesian3())
    );
    const direct = Matrix4.multiply(enu, Matrix4.fromRotationX(Math.PI / 2), new Matrix4());
    for (let i = 0; i < 16; i++) {
      expect(viaQuaternion.values[i]).toBeCloseTo(direct.values[i], 9);
    }
  });

  it('heading rotation keeps the model up axis invariant and stays orthonormal', () => {
    const hpr = Transforms.headingPitchRollQuaternion(
      Cartesian3.multiplyByScalar(surface, 6378137.0, new Cartesian3()),
      Math.PI / 2,
      0,
      0
    );
    const m = composeEntityModelMatrix(surface, hpr);
    // Up (column 1) unchanged by a heading rotation about up.
    expect(m.values[4]).toBeCloseTo(1, 9);
    expect(m.values[5]).toBeCloseTo(0, 9);
    expect(m.values[6]).toBeCloseTo(0, 9);
    // Column 0 stays a unit vector perpendicular to east (0,1,0).
    const c0 = [m.values[0], m.values[1], m.values[2]];
    const len = Math.hypot(c0[0], c0[1], c0[2]);
    expect(len).toBeCloseTo(1, 9);
    expect(c0[1]).toBeCloseTo(0, 9); // dot with east
  });

  it('applies uniform scale after rotation', () => {
    const m = composeEntityModelMatrix(surface, undefined, 0.5);
    const unscaled = composeEntityModelMatrix(surface);
    for (let i = 0; i < 12; i++) {
      expect(m.values[i]).toBeCloseTo(unscaled.values[i] * 0.5, 9);
    }
    // Translation is not scaled.
    expect(m.values[12]).toBeCloseTo(unscaled.values[12], 9);
  });
});

describe('resolveMinimumPixelScale', () => {
  const EARTH = 6378137.0;

  it('returns the plain scale when no minimum pixel size is set', () => {
    expect(resolveMinimumPixelScale(2, undefined, 1e6, 1.0, 800)).toBe(2);
    expect(resolveMinimumPixelScale(2, 0, 1e6, 1.0, 800)).toBe(2);
  });

  it('scales up so the model covers the minimum pixel footprint when far away', () => {
    const fovY = Math.PI / 3;
    const distance = 100_000; // metres
    const heightPx = 1000;
    const metresPerPixel = (2 * distance * Math.tan(fovY / 2)) / heightPx; // 230.9 m/px
    const minPx = 10;
    const expected = (minPx * metresPerPixel) / EARTH;
    const scale = resolveMinimumPixelScale(expected / 2, minPx, distance, fovY, heightPx);
    expect(scale).toBeCloseTo(expected, 12);
    // Already large enough → untouched.
    expect(resolveMinimumPixelScale(expected * 3, minPx, distance, fovY, heightPx)).toBeCloseTo(
      expected * 3,
      12
    );
  });
});
