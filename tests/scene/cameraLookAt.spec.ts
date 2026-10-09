import { describe, it, expect } from 'vitest';
import { Camera } from '../../src/scene/Camera';
import { HeadingPitchRange } from '../../src/scene/HeadingPitchRange';
import { Matrix4 } from '../../src/math/Matrix4';
import { Cartesian3 } from '../../src/math/Cartesian3';
import { BoundingSphere } from '../../src/scene/BoundingSphere';
import { Entity } from '../../src/dataSource/Entity';
import { CallbackProperty } from '../../src/dataSource/CallbackProperty';
import { EntityTracker, headingPitchRangeFromCamera } from '../../src/scene/EntityTracker';

const EARTH = 6378137.0;

describe('Camera.lookAt', () => {
  const target = new Cartesian3(EARTH, 0, 0); // equator, lon 0, metres ECEF

  it('pitch −90° places the camera straight above the target looking down', () => {
    const camera = new Camera();
    camera.lookAt(target, new HeadingPitchRange(0, -Math.PI / 2, 1000));
    const range = 1000 / EARTH;
    expect(camera.position.x).toBeCloseTo(1 + range, 9);
    expect(camera.position.y).toBeCloseTo(0, 9);
    expect(camera.position.z).toBeCloseTo(0, 9);
    // Looking straight back at the target → direction ≈ −up = (−1,0,0).
    expect(camera.direction.x).toBeCloseTo(-1, 6);
    expect(camera.direction.y).toBeCloseTo(0, 6);
    expect(camera.direction.z).toBeCloseTo(0, 6);
  });

  it('pitch 0° / heading 0° places the camera due north of the target on the horizon', () => {
    const camera = new Camera();
    camera.lookAt(target, new HeadingPitchRange(0, 0, 2000));
    const range = 2000 / EARTH;
    // north at (1,0,0) is (0,0,1)
    expect(camera.position.x).toBeCloseTo(1, 9);
    expect(camera.position.z).toBeCloseTo(range, 9);
    // The view pitch equals the offset pitch (horizon → level flight).
    const { pitch } = camera._deriveHeadingPitch();
    expect(pitch).toBeCloseTo(0, 6);
  });

  it('keeps the offset distance from the target', () => {
    const camera = new Camera();
    camera.lookAt(target, new HeadingPitchRange(0.7, -0.9, 5000));
    const dist = Cartesian3.distance(camera.position, new Cartesian3(1, 0, 0));
    expect(dist).toBeCloseTo(5000 / EARTH, 9);
  });

  it('lookAt(undefined) is a no-op (unlock call sites)', () => {
    const camera = new Camera();
    const before = camera.position.x;
    camera.lookAt(undefined, new HeadingPitchRange(0, 0, 1));
    expect(camera.position.x).toBe(before);
  });
});

describe('Camera.lookAtTransform', () => {
  it('Matrix4.IDENTITY is an accepted no-op unlock', () => {
    const camera = new Camera();
    expect(() => camera.lookAtTransform(Matrix4.IDENTITY)).not.toThrow();
  });

  it('rejects non-identity transforms with a descriptive error', () => {
    const camera = new Camera();
    const m = Matrix4.identity();
    m.values[12] = 1;
    expect(() => camera.lookAtTransform(m)).toThrow(/IDENTITY/);
  });
});

describe('Camera.flyToBoundingSphere / zoomTo envelope', () => {
  it('BoundingSphere.fromPoints covers all points (Ritter)', () => {
    const a = new Cartesian3(0, 0, 0);
    const b = new Cartesian3(1, 0, 0);
    const c = new Cartesian3(0.5, 0.2, 0);
    const s = BoundingSphere.fromPoints([a, b, c]);
    expect(s.center.x).toBeCloseTo(0.5, 9);
    // radius in metres: at least half the 0..1 spread along x.
    expect(s.radius).toBeGreaterThanOrEqual(0.5 * EARTH - 1e-6);
    // Every point is inside the sphere.
    for (const p of [a, b, c]) {
      expect(Cartesian3.distance(p, s.center)).toBeLessThanOrEqual(s.radius / EARTH + 1e-9);
    }
  });

  it('flyToBoundingSphere starts a flight toward the sphere framing position', () => {
    const camera = new Camera();
    camera.setView({ destination: { lon: 10, lat: 10, height: 20_000_000 } });
    const center = new Cartesian3(1, 0, 0); // normalised ECEF
    camera.flyToBoundingSphere(new BoundingSphere(center, 1000), {
      offset: new HeadingPitchRange(0, -Math.PI / 2, 4000),
      duration: 2,
    });
    expect(camera.inFlight).toBe(true);
    camera.cancelFlight();
    expect(camera.inFlight).toBe(false);
  });
});

describe('EntityTracker (Viewer.trackedEntity follow logic)', () => {
  it('translates the camera in lockstep with the entity each update', () => {
    const camera = new Camera();
    camera.setView({ destination: { lon: 0, lat: 0, height: 3000 } });
    const startCam = new Cartesian3(camera.position.x, camera.position.y, camera.position.z);

    const entity = new Entity({ position: new Cartesian3(1, 0, 0) });
    const tracker = new EntityTracker(camera);
    tracker.track(entity);
    expect(tracker.entity).toBe(entity);
    expect(tracker.initialOffset).toBeDefined();

    // Entity moves by (+0.001, 0, 0): camera must move by exactly the delta.
    entity.position = new Cartesian3(1.001, 0, 0);
    tracker.update();
    expect(camera.position.x - startCam.x).toBeCloseTo(0.001, 9);
    expect(camera.position.y - startCam.y).toBeCloseTo(0, 9);

    // And again by a z delta — the offset from the (moved) camera is kept.
    const secondCam = new Cartesian3(camera.position.x, camera.position.y, camera.position.z);
    entity.position = new Cartesian3(1.001, 0, 0.002);
    tracker.update();
    expect(camera.position.z - secondCam.z).toBeCloseTo(0.002, 9);
    expect(camera.position.x - secondCam.x).toBeCloseTo(0, 9);
  });

  it('follows a CallbackProperty position (dynamic entity)', () => {
    const camera = new Camera();
    camera.setView({ destination: { lon: 0, lat: 0, height: 5000 } });
    let x = 0;
    const entity = new Entity({
      position: new CallbackProperty(() => new Cartesian3(1 + x, 0, 0)),
    });
    const tracker = new EntityTracker(camera);
    tracker.track(entity);
    tracker.update(); // establish baseline
    const before = camera.position.clone
      ? camera.position.clone()
      : new Cartesian3(camera.position.x, camera.position.y, camera.position.z);
    x = 0.01;
    tracker.update();
    expect(camera.position.x - before.x).toBeCloseTo(0.01, 9);
  });

  it('release stops following', () => {
    const camera = new Camera();
    const entity = new Entity({ position: new Cartesian3(1, 0, 0) });
    const tracker = new EntityTracker(camera);
    tracker.track(entity);
    tracker.release();
    expect(tracker.entity).toBeUndefined();
    entity.position = new Cartesian3(1.5, 0, 0);
    tracker.update();
    expect(camera.position.x).toBeCloseTo(0, 12); // untouched default camera
  });

  it('user zoom while tracking shifts the relative offset, following continues', () => {
    const camera = new Camera();
    camera.setView({ destination: { lon: 0, lat: 0, height: 3000 } });
    const entity = new Entity({ position: new Cartesian3(1, 0, 0) });
    const tracker = new EntityTracker(camera);
    tracker.track(entity);

    camera.zoom(-0.5); // user input moves the camera relative to the entity
    const afterZoom = new Cartesian3(camera.position.x, camera.position.y, camera.position.z);

    entity.position = new Cartesian3(1.0005, 0, 0);
    tracker.update();
    // Camera translated by the entity delta from its user-adjusted position.
    expect(camera.position.x - afterZoom.x).toBeCloseTo(0.0005, 9);
  });
});

describe('headingPitchRangeFromCamera', () => {
  it('derives nadir offset as pitch −90° and the distance in metres', () => {
    const camera = new Camera();
    const targetMetres = new Cartesian3(EARTH, 0, 0);
    camera.lookAt(targetMetres, new HeadingPitchRange(0.3, -Math.PI / 2, 2500));
    const hpr = headingPitchRangeFromCamera(camera, new Cartesian3(1, 0, 0));
    expect(hpr.pitch).toBeCloseTo(-Math.PI / 2, 6);
    expect(hpr.range).toBeCloseTo(2500, -3); // metres (≈, 0.1% tolerance)
  });
});
