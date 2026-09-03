import { Viewer } from '../core/Viewer';
import { Cartesian3 } from '../math/Cartesian3';
import { Matrix4 } from '../math/Matrix4';
import { Transforms } from '../math/Transforms';
import { Color } from '../math/Color';
import { Model } from '../scene/Model';
import { Camera } from '../scene/Camera';
import { Primitive } from '../scene/Primitive';
import { Context } from '../renderer/Context';
import type { PickResult } from '../picking/PickPass';

/**
 * T2.3 验证页: 正式版 GPU 拾取 (PickPass)。
 *
 * 场景 = 地球 + cone4-red.glb 模型 + 两个多边形面。
 * 点击任一对象 → 显示拾取结果 (id / 类型 / 世界坐标转经纬度) 与延迟;
 * 页面加载后自动做 50 次连续拾取并统计延迟。
 * 结果镜像到 window.__pickPassDemo。
 */

interface PickLogEntry {
  source: 'click' | 'auto';
  pixelCss: [number, number] | null;
  pixelDevice: [number, number];
  id: number;
  type: string;
  lon: number | null;
  lat: number | null;
  heightM: number | null;
  wallMs: number;
  readbackMs: number;
}

interface Auto50Report {
  totalMs: number;
  avgWallMs: number;
  minWallMs: number;
  maxWallMs: number;
  avgReadbackMs: number;
  maxReadbackMs: number;
  hits: Record<string, number>;
  pass: boolean;
}

interface DemoReport {
  initialized: boolean;
  targets: Record<string, [number, number] | null>;
  clicks: PickLogEntry[];
  auto50: Auto50Report | null;
  lastPick: PickLogEntry | null;
  consoleErrors: string[];
}

const report: DemoReport = {
  initialized: false,
  targets:     { model: null, polygonA: null, polygonB: null, globe: null },
  clicks:      [],
  auto50:      null,
  lastPick:    null,
  consoleErrors: [],
};

// Capture console errors (GPU validation failures surface here).
{
  const origError = console.error.bind(console);
  console.error = (...args: unknown[]) => {
    report.consoleErrors.push(args.map((a) => (a instanceof Error ? a.message : String(a))).join(' '));
    origError(...args);
  };
}

const statsEl = document.getElementById('stats');

function renderHud(): void {
  if (!statsEl) return;
  const lines: string[] = [];
  lines.push('── 最近拾取 ──');
  const lp = report.lastPick;
  if (lp) {
    lines.push(`类型: ${lp.type}   id: ${lp.id}`);
    if (lp.lon !== null) {
      lines.push(`经纬: ${lp.lon.toFixed(4)}°E, ${lp.lat!.toFixed(4)}°N  高程: ${lp.heightM!.toFixed(0)} m`);
    }
    lines.push(`延迟: wall=${lp.wallMs.toFixed(2)}ms  readback=${lp.readbackMs.toFixed(2)}ms`);
  } else {
    lines.push('(尚未拾取)');
  }

  const recent = report.clicks.slice(-50);
  if (recent.length > 0) {
    const walls = recent.map((c) => c.wallMs);
    const reads = recent.map((c) => c.readbackMs);
    const avg   = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
    lines.push('');
    lines.push(`── 最近 ${recent.length} 次拾取统计 ──`);
    lines.push(`wall   平均 ${avg(walls).toFixed(2)}ms  最大 ${Math.max(...walls).toFixed(2)}ms`);
    lines.push(`readback 平均 ${avg(reads).toFixed(2)}ms  最大 ${Math.max(...reads).toFixed(2)}ms`);
  }

  const a = report.auto50;
  if (a) {
    lines.push('');
    lines.push('── 自动 50 次连续拾取 ──');
    lines.push(`总耗时 ${a.totalMs.toFixed(1)}ms  平均 ${a.avgWallMs.toFixed(2)}ms/次`);
    lines.push(`wall min ${a.minWallMs.toFixed(2)} / max ${a.maxWallMs.toFixed(2)}ms`);
    lines.push(`readback 平均 ${a.avgReadbackMs.toFixed(2)} / max ${a.maxReadbackMs.toFixed(2)}ms`);
    lines.push(`命中分布: ${JSON.stringify(a.hits)}`);
    lines.push(`结论: ${a.pass ? 'PASS (全部命中且延迟 < 8ms)' : 'FAIL'}`);
  }
  statsEl.textContent = lines.join('\n');
}

function waitFrames(n: number): Promise<void> {
  return new Promise((resolve) => {
    let count = 0;
    const step = (): void => {
      count += 1;
      if (count >= n) resolve();
      else requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}

/** Project a world point to canvas device pixels (top-left origin). */
function projectToPixel(viewer: Viewer, world: Cartesian3): [number, number] {
  const vp    = viewer.scene.camera.viewProjectionMatrix.values; // column-major
  const x     = world.x, y = world.y, z = world.z;
  const clipX = vp[0] * x + vp[4] * y + vp[8]  * z + vp[12];
  const clipY = vp[1] * x + vp[5] * y + vp[9]  * z + vp[13];
  const clipW = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
  const ndcX  = clipX / clipW;
  const ndcY  = clipY / clipW;
  return [
    (ndcX * 0.5 + 0.5) * viewer.canvas.width,
    (0.5 - ndcY * 0.5) * viewer.canvas.height, // NDC bottom-left → DOM top-left flip
  ];
}

/**
 * Build a filled polygon primitive lying on the ellipsoid surface.
 * Triangle fan around the centre; corners ordered CCW seen from outside.
 */
function makeSurfacePolygon(
  context: Context,
  centerLon: number,
  centerLat: number,
  halfDeg: number,
  color: Color
): Primitive {
  const S = Camera.EARTH_SCALE;
  const toUnit = (lon: number, lat: number): Cartesian3 => {
    // Lift 100 km: a flat triangle fan spanning several degrees chords ~35 km
    // below the ellipsoid at its edges, which would sink under the surface.
    const p = Cartesian3.fromDegrees(lon, lat, 100_000);
    return new Cartesian3(p.x / S, p.y / S, p.z / S);
  };

  const center = toUnit(centerLon, centerLat);
  const corners = [
    toUnit(centerLon - halfDeg, centerLat - halfDeg),
    toUnit(centerLon + halfDeg, centerLat - halfDeg),
    toUnit(centerLon + halfDeg, centerLat + halfDeg),
    toUnit(centerLon - halfDeg, centerLat + halfDeg),
  ];

  const verts: number[] = [];
  const push = (v: Cartesian3): void => {
    const n = Cartesian3.normalize(v, new Cartesian3());
    verts.push(v.x, v.y, v.z, n.x, n.y, n.z, 0, 0);
  };
  push(center);
  for (const c of corners) push(c);

  const indices: number[] = [];
  for (let i = 1; i <= 4; i++) indices.push(0, i, (i % 4) + 1);

  return new Primitive(context, {
    vertices:    new Float32Array(verts),
    indices:     new Uint32Array(indices),
    color,
    translucent: false,
  });
}

/** Normalised ECEF → lon/lat/height(m) for display. */
function toLonLatHeight(p: Cartesian3): { lon: number; lat: number; heightM: number } {
  const mag = Cartesian3.magnitude(p);
  return {
    lon:     (Math.atan2(p.y, p.x) * 180) / Math.PI,
    lat:     (Math.atan2(p.z, Math.hypot(p.x, p.y)) * 180) / Math.PI,
    heightM: (mag - 1.0) * Camera.EARTH_SCALE,
  };
}

async function run(): Promise<void> {
  const container = document.getElementById('cesiumContainer');
  if (!container) throw new Error('pick-pass-spike: #cesiumContainer not found');

  const viewer = new Viewer(container, { powerPreference: 'high-performance' });
  await viewer.initialize();
  (window as unknown as { __viewer: Viewer }).__viewer = viewer; // QA probe handle

  viewer.camera.setView({
    destination: Cartesian3.fromDegrees(116.4, 39.9, 6_000_000),
  });

  viewer.globe.loadImageryFromUrl('/images/world.jpg').catch(() => {
    viewer.globe.generateProceduralTexture();
  });

  // ── GLB model near Beijing ─────────────────────────────────────────────────
  const ecef      = Cartesian3.fromDegrees(116.4, 39.9, 0);
  const enu       = Transforms.eastNorthUpToFixedFrame(ecef);
  const yUpToZUp  = Matrix4.fromRotationX(Math.PI / 2);
  const placement = Matrix4.multiply(enu, yUpToZUp, new Matrix4());

  await Model.fromGltfAsync({
    url:         '/models/cone4-red.glb',
    scene:       viewer.scene,
    modelMatrix: placement,
    scale:       500_000 / Camera.EARTH_SCALE,
  });

  // ── Two surface polygons ───────────────────────────────────────────────────
  const polygonA = makeSurfacePolygon(viewer.context, 113.0, 42.8, 2.5, new Color(1.0, 0.55, 0.10, 1.0));
  const polygonB = makeSurfacePolygon(viewer.context, 119.8, 37.0, 2.5, new Color(0.62, 0.30, 0.92, 1.0));
  viewer.scene.addPrimitive(polygonA);
  viewer.scene.addPrimitive(polygonB);

  // ── Type labels (all scene primitives are pickable automatically) ──────────
  const labels = new Map<Primitive, string>();
  const globePrim = viewer.scene.globe.primitive;
  if (globePrim) labels.set(globePrim, '地球');
  for (const prim of viewer.scene.primitives) {
    if (prim === polygonA)      labels.set(prim, '多边形 A');
    else if (prim === polygonB) labels.set(prim, '多边形 B');
    else                        labels.set(prim, 'GLB 模型');
  }

  function typeOf(result: PickResult): string {
    if (result.id === 0) return '背景';
    const p = result.primitive;
    return (p && labels.get(p)) ?? `未知(id=${result.id})`;
  }

  function record(
    source: 'click' | 'auto',
    pixelDevice: [number, number],
    pixelCss: [number, number] | null,
    result: PickResult,
    wallMs: number
  ): PickLogEntry {
    const geo = result.worldPosition ? toLonLatHeight(result.worldPosition) : null;
    const entry: PickLogEntry = {
      source,
      pixelCss,
      pixelDevice,
      id: result.id,
      type: typeOf(result),
      lon: geo ? geo.lon : null,
      lat: geo ? geo.lat : null,
      heightM: geo ? geo.heightM : null,
      wallMs,
      readbackMs: result.timing.readbackMs,
    };
    report.clicks.push(entry);
    report.lastPick = entry;
    return entry;
  }

  // ── Interactive clicks through the public viewer API ───────────────────────
  viewer.onPick = (result, pos) => {
    if (!result) return;
    const px = Math.round(pos.x * (viewer.canvas.width / Math.max(1, viewer.canvas.clientWidth)));
    const py = Math.round(pos.y * (viewer.canvas.height / Math.max(1, viewer.canvas.clientHeight)));
    const wallMs = result.timing.encodeMs + result.timing.readbackMs;
    record('click', [px, py], [pos.x, pos.y], result, wallMs);
    renderHud();
  };

  await waitFrames(10);

  // ── Locate target pixels by scanning around projected centres ──────────────
  const cw = viewer.canvas.width;
  const ch = viewer.canvas.height;

  async function findTarget(
    world: Cartesian3,
    wantType: string
  ): Promise<[number, number] | null> {
    const [px, py] = projectToPixel(viewer, world);
    const offsets: Array<[number, number]> = [[0, 0]];
    for (let ring = 1; ring <= 8; ring++) {
      for (let a = 0; a < 16; a++) {
        const ang = (a / 16) * Math.PI * 2;
        offsets.push([Math.cos(ang) * ring * 12, Math.sin(ang) * ring * 12]);
      }
    }
    for (const [dx, dy] of offsets) {
      const tx = Math.round(px + dx);
      const ty = Math.round(py + dy);
      if (tx < 0 || ty < 0 || tx >= cw || ty >= ch) continue;
      const result = await viewer.scene.pick(tx, ty);
      if (result && typeOf(result) === wantType) return [tx, ty];
    }
    return null;
  }

  report.targets.model    = await findTarget(ecef, 'GLB 模型');
  report.targets.polygonA = await findTarget(Cartesian3.fromDegrees(113.0, 42.8, 100_000), '多边形 A');
  report.targets.polygonB = await findTarget(Cartesian3.fromDegrees(119.8, 37.0, 100_000), '多边形 B');
  report.targets.globe    = await findTarget(Cartesian3.fromDegrees(100.0, 30.0, 0), '地球');

  report.initialized = true;
  renderHud();

  // ── Auto: 50 consecutive picks over the four targets (with ±1px jitter so
  //    every pick is a fresh GPU readback, not a cache hit) ───────────────────
  const targetList: Array<[string, [number, number]]> = [];
  if (report.targets.model)    targetList.push(['GLB 模型', report.targets.model]);
  if (report.targets.polygonA) targetList.push(['多边形 A', report.targets.polygonA]);
  if (report.targets.polygonB) targetList.push(['多边形 B', report.targets.polygonB]);
  if (report.targets.globe)    targetList.push(['地球',     report.targets.globe]);

  const hits: Record<string, number> = {};
  const walls: number[] = [];
  const reads: number[] = [];
  const t0 = performance.now();

  for (let i = 0; i < 50 && targetList.length > 0; i++) {
    const [name, pixel] = targetList[i % targetList.length];
    const jx = ((i % 3) - 1) + pixel[0];
    const jy = (i % 2) + pixel[1];
    const s = performance.now();
    const result = await viewer.scene.pick(jx, jy);
    const wall = performance.now() - s;
    if (!result) continue;
    walls.push(wall);
    reads.push(result.timing.readbackMs);
    hits[name] = (hits[name] ?? 0) + 1;
    record('auto', [jx, jy], null, result, wall);
  }

  const totalMs = performance.now() - t0;
  const avg = (v: number[]): number => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0);
  const allHit = targetList.length === 4 && targetList.every(([name]) => (hits[name] ?? 0) > 0);
  report.auto50 = {
    totalMs,
    avgWallMs:     avg(walls),
    minWallMs:     walls.length ? Math.min(...walls) : 0,
    maxWallMs:     walls.length ? Math.max(...walls) : 0,
    avgReadbackMs: avg(reads),
    maxReadbackMs: reads.length ? Math.max(...reads) : 0,
    hits,
    pass: allHit && avg(walls) < 8 && report.consoleErrors.length === 0,
  };

  renderHud();

  (window as unknown as { __pickPassDemo: DemoReport }).__pickPassDemo = report;
  console.log('PICK_PASS_DEMO_REPORT', JSON.stringify(report));
}

run().catch((err: unknown) => {
  const message = err instanceof Error ? `${err.message}\n${err.stack ?? ''}` : String(err);
  console.error('pick-pass-spike failed:', err);
  if (statsEl) statsEl.textContent = `验证页失败:\n${message}`;
});
