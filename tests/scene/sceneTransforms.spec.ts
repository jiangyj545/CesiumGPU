import { describe, it, expect } from 'vitest';
import { SceneTransforms } from '../../src/scene/SceneTransforms';
import { Camera } from '../../src/scene/Camera';
import { Cartesian3 } from '../../src/math/Cartesian3';

interface FakeCanvas {
  width: number;
  height: number;
  clientWidth: number;
  clientHeight: number;
}

const EARTH = 6378137.0;
/** Camera straight above (1,0,0) looking straight down (nadir, heading 0). */
function makeNadirScene(canvas: FakeCanvas) {
  const camera = new Camera();
  camera.setAspectRatio(canvas.clientWidth / canvas.clientHeight);
  camera.lookAt(new Cartesian3(EARTH, 0, 0), {
    heading: 0,
    pitch: -Math.PI / 2,
    range: 1_000_000,
  });
  return {
    camera,
    scene: { camera, context: { canvas: canvas as unknown as HTMLCanvasElement } },
  };
}

describe('SceneTransforms.worldToWindowCoordinates', () => {
  const canvas: FakeCanvas = { width: 200, height: 200, clientWidth: 100, clientHeight: 100 };

  it('projects the view-centre point to the canvas centre (CSS pixels over a DPR-2 buffer)', () => {
    const { scene } = makeNadirScene(canvas);
    const p = SceneTransforms.worldToWindowCoordinates(scene, new Cartesian3(1, 0, 0));
    expect(p).toBeDefined();
    expect(p!.x).toBeCloseTo(50, 3); // centre of the 100×100 CSS viewport
    expect(p!.y).toBeCloseTo(50, 3);
  });

  it('scales with the CSS viewport, not the device buffer (DPR handling)', () => {
    // Same camera geometry, doubled CSS viewport: coordinates double.
    const small = makeNadirScene({ width: 200, height: 200, clientWidth: 100, clientHeight: 100 });
    const large = makeNadirScene({ width: 400, height: 400, clientWidth: 200, clientHeight: 200 });
    // A point offset east of the target moves off-centre.
    const east = new Cartesian3(0, 0.001, 1);
    const a = SceneTransforms.worldToWindowCoordinates(small.scene, east)!;
    const b = SceneTransforms.worldToWindowCoordinates(large.scene, east)!;
    expect(a.x).not.toBeCloseTo(50, 3);
    expect(b.x).toBeCloseTo(a.x * 2, 3);
    expect(b.y).toBeCloseTo(a.y * 2, 3);
  });

  it('returns undefined for points behind the camera (w ≤ 0)', () => {
    const { scene } = makeNadirScene(canvas);
    // The camera sits at ≈(1.157, 0, 0) looking toward the origin: a point
    // further out along +x is behind it.
    expect(SceneTransforms.worldToWindowCoordinates(scene, new Cartesian3(2, 0, 0))).toBeUndefined();
  });
});

describe('SceneTransforms.wgs84ToWindowCoordinates', () => {
  const canvas: FakeCanvas = { width: 200, height: 200, clientWidth: 100, clientHeight: 100 };

  it('accepts metre-ECEF Cartesian3 and delegates to the world projection', () => {
    const { scene } = makeNadirScene(canvas);
    const viaWorld = SceneTransforms.worldToWindowCoordinates(scene, new Cartesian3(1, 0, 0));
    const viaWgs = SceneTransforms.wgs84ToWindowCoordinates(scene, new Cartesian3(EARTH, 0, 0));
    expect(viaWgs).toBeDefined();
    expect(viaWgs!.x).toBeCloseTo(viaWorld!.x, 6);
    expect(viaWgs!.y).toBeCloseTo(viaWorld!.y, 6);
  });

  it('accepts { lon, lat, height } literals', () => {
    const { scene } = makeNadirScene(canvas);
    const p = SceneTransforms.wgs84ToWindowCoordinates(scene, { lon: 0, lat: 0, height: 0 });
    expect(p).toBeDefined();
    expect(p!.x).toBeCloseTo(50, 3);
    expect(p!.y).toBeCloseTo(50, 3);
  });
});
