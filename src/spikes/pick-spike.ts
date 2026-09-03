import { Viewer } from '../core/Viewer';
import { Cartesian3 } from '../math/Cartesian3';
import { Matrix4 } from '../math/Matrix4';
import { Transforms } from '../math/Transforms';
import { Model } from '../scene/Model';
import { Camera } from '../scene/Camera';
import { Primitive } from '../scene/Primitive';
import { PickPass, PickResult } from '../picking/PickPass';

/**
 * Spike: GPU picking via id-encoded render-to-texture + 1×1 readback.
 *
 * Measures:
 *  - rgba8unorm id precision (write 0x123456 → read back)
 *  - single pick end-to-end latency (submit → mapAsync resolve)
 *  - 100 consecutive picks (wall time + per-pick readback stats)
 *  - in-flight throttling (concurrent pick() reuses one promise)
 *  - GPU worldPos readback vs CPU raycast cross-check
 *
 * Note: depth-buffer readback is NOT attempted — WebGPU forbids partial
 * (sub-resource) copies of depth/stencil formats, so a 1×1 depth read would
 * invalidate the whole frame's command buffer (verified in this spike).
 *
 * Results are rendered into #stats and mirrored on window.__pickSpike.
 */

interface PickSpikeReport {
  precision: {
    testId: number;
    decodedId: number;
    lossless: boolean;
  };
  modelPick: {
    pixel: [number, number];
    id: number;
    expectedIds: number[];
    hit: boolean;
    worldPosition: [number, number, number] | null;
    worldPositionMeters: [number, number, number] | null;
    timing: PickResult['timing'];
    wallMs: number;
  } | null;
  backgroundPick: {
    pixel: [number, number];
    id: number;
    expectedId: number;
    pass: boolean;
    wallMs: number;
  };
  globeCrossCheck: {
    gpuWorldPosMeters: [number, number, number] | null;
    cpuRaycastMeters: [number, number, number] | null;
    deltaMeters: number | null;
  };
  throttling: {
    concurrentCalls: number;
    distinctPromises: number;
    pass: boolean;
  };
  batch100: {
    totalMs: number;
    avgPerPickMs: number;
    avgReadbackMs: number;
    minReadbackMs: number;
    maxReadbackMs: number;
  };
  debug: {
    hookCalls: number;
    encodes: number;
    flushes: number;
    gpuErrors: number;
    consoleErrors: string[];
  };
}

const statsEl = document.getElementById('stats');

/** Capture console errors (GPU validation failures surface here). */
const consoleErrors: string[] = [];
{
  const origError = console.error.bind(console);
  console.error = (...args: unknown[]) => {
    consoleErrors.push(args.map((a) => (a instanceof Error ? a.message : String(a))).join(' '));
    origError(...args);
  };
}

function log(line: string): void {
  if (statsEl) statsEl.textContent += `\n${line}`;
}

function setLog(text: string): void {
  if (statsEl) statsEl.textContent = text;
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

/** Project a world point to canvas pixels (top-left origin, same as DOM). */
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

function length(v: [number, number, number]): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

async function run(): Promise<void> {
  const container = document.getElementById('cesiumContainer');
  if (!container) throw new Error('pick-spike: #cesiumContainer not found');

  setLog('初始化 Viewer…');

  // ── Viewer + globe + model ─────────────────────────────────────────────────
  const viewer = new Viewer(container, { powerPreference: 'high-performance' });
  await viewer.initialize();

  viewer.camera.setView({
    destination: Cartesian3.fromDegrees(116.4, 39.9, 20_000_000),
  });

  viewer.globe.loadImageryFromUrl('/images/world.jpg').catch(() => {
    viewer.globe.generateProceduralTexture();
  });

  const ecef       = Cartesian3.fromDegrees(116.4, 39.9, 0);
  const enu        = Transforms.eastNorthUpToFixedFrame(ecef);
  const yUpToZUp   = Matrix4.fromRotationX(Math.PI / 2);
  const placement  = Matrix4.multiply(enu, yUpToZUp, new Matrix4());

  const model = await Model.fromGltfAsync({
    url:         '/models/cone4-red.glb',
    scene:       viewer.scene,
    modelMatrix: placement,
    scale:       500_000 / Camera.EARTH_SCALE,
  });
  log(`模型已加载: cone4-red.glb (${model.primitiveCount} 个 mesh primitive)`);

  // ── PickPass: scene-graph primitives (globe + model) pick automatically ────
  const picker = new PickPass(viewer.scene);

  const idLabels = new Map<number, string>();
  const globePrim: Primitive | null = viewer.scene.globe.primitive;
  if (globePrim) {
    idLabels.set(globePrim.pickId, 'Globe');
  }
  for (const prim of viewer.scene.primitives) {
    idLabels.set(prim.pickId, 'Model-part');
  }

  await waitFrames(10);
  setLog('运行拾取测试…');

  const report: PickSpikeReport = {
    precision:       { testId: 0, decodedId: 0, lossless: false },
    modelPick:       null,
    backgroundPick:  { pixel: [0, 0], id: -1, expectedId: 0, pass: false, wallMs: 0 },
    globeCrossCheck: { gpuWorldPosMeters: null, cpuRaycastMeters: null, deltaMeters: null },
    throttling:      { concurrentCalls: 0, distinctPromises: 0, pass: false },
    batch100: {
      totalMs: 0, avgPerPickMs: 0, avgReadbackMs: 0, minReadbackMs: 0, maxReadbackMs: 0,
    },
    debug: { hookCalls: 0, encodes: 0, flushes: 0, gpuErrors: 0, consoleErrors: [] },
  };

  // ── 1. id precision (0x123456 round-trip through rgba8unorm) ──────────────
  const testId = 0x123456;
  const decodedId = await picker.verifyIdPrecision(testId);
  report.precision = { testId, decodedId, lossless: decodedId === testId };
  log(`[1] id 精度: 0x${testId.toString(16)} → 0x${decodedId.toString(16)} ${decodedId === testId ? '(无损 PASS)' : '(有损 FAIL)'}`);

  // ── 2. pick the model (scan around the projected base point on miss) ──────
  const [px, py] = projectToPixel(viewer, ecef);
  const modelIds = [...idLabels.entries()]
    .filter(([, label]) => label === 'Model-part')
    .map(([id]) => id);

  const offsets: Array<[number, number]> = [[0, 0]];
  for (let ring = 1; ring <= 5 && offsets.length < 26; ring++) {
    for (let a = 0; a < 8; a++) {
      const ang = (a / 8) * Math.PI * 2;
      offsets.push([Math.cos(ang) * ring * 25, Math.sin(ang) * ring * 25]);
    }
  }

  let modelResult: PickResult | null = null;
  let modelPixel: [number, number] = [px, py];
  let modelWallMs = 0;
  for (const [dx, dy] of offsets) {
    const tx = px + dx;
    const ty = py + dy;
    if (tx < 0 || ty < 0 || tx >= viewer.canvas.width || ty >= viewer.canvas.height) continue;
    const t0 = performance.now();
    const result = await picker.pick(tx, ty);
    if (!result) continue;
    modelWallMs = performance.now() - t0;
    if (result.id !== 0 && modelIds.includes(result.id)) {
      modelResult = result;
      modelPixel = [tx, ty];
      break;
    }
    if (result.id !== 0 && !modelResult) {
      // first non-background hit (likely globe) — keep as fallback target
      modelResult = result;
      modelPixel = [tx, ty];
    }
  }

  if (modelResult) {
    const toMeters = (v: [number, number, number]): [number, number, number] =>
      [v[0] * Camera.EARTH_SCALE, v[1] * Camera.EARTH_SCALE, v[2] * Camera.EARTH_SCALE];
    const gpuPos = modelResult.worldPosition;
    const worldPosTuple: [number, number, number] | null = gpuPos ? [gpuPos.x, gpuPos.y, gpuPos.z] : null;
    report.modelPick = {
      pixel: modelPixel,
      id: modelResult.id,
      expectedIds: modelIds,
      hit: modelIds.includes(modelResult.id),
      worldPosition: worldPosTuple,
      worldPositionMeters: worldPosTuple ? toMeters(worldPosTuple) : null,
      timing: modelResult.timing,
      wallMs: modelWallMs,
    };
    const label = idLabels.get(modelResult.id) ?? `id=${modelResult.id}`;
    log(`[2] 模型拾取 @(${modelPixel[0].toFixed(0)},${modelPixel[1].toFixed(0)}) → ${label} id=${modelResult.id}`);
    log(`    worldPos(米) = [${(report.modelPick.worldPositionMeters ?? [0, 0, 0]).map((v) => v.toFixed(1)).join(', ')}]`);
    log(`    wall=${modelWallMs.toFixed(1)}ms  readback=${modelResult.timing.readbackMs.toFixed(2)}ms  total=${modelResult.timing.totalMs.toFixed(2)}ms`);
  } else {
    log('[2] 模型拾取: 未命中任何对象');
  }

  // ── 3. background pick (screen corner) ────────────────────────────────────
  {
    const t0 = performance.now();
    const bg = await picker.pick(10, 10);
    if (!bg) throw new Error('pick-spike: scene not ready for pick');
    const wallMs = performance.now() - t0;
    report.backgroundPick = { pixel: [10, 10], id: bg.id, expectedId: 0, pass: bg.id === 0, wallMs };
    log(`[3] 背景拾取 @(10,10) → id=${bg.id} ${bg.id === 0 ? '(PASS)' : '(FAIL: 应为 0)'} wall=${wallMs.toFixed(1)}ms`);
  }

  // ── 4. GPU worldPos vs CPU raycast (globe point) ──────────────────────────
  {
    const gx = px + 120;
    const gy = py + 120;
    const result = await picker.pick(gx, gy);
    if (result?.worldPosition) {
      const cw = viewer.canvas.width;
      const ch = viewer.canvas.height;
      const ndcX = (gx / (cw - 1)) * 2 - 1;
      const ndcY = 1 - (gy / (ch - 1)) * 2;
      const cpu = viewer.scene.camera.pickGlobe(ndcX, ndcY);
      const gpuMeters: [number, number, number] = [
        result.worldPosition.x * Camera.EARTH_SCALE,
        result.worldPosition.y * Camera.EARTH_SCALE,
        result.worldPosition.z * Camera.EARTH_SCALE,
      ];
      const cpuMeters: [number, number, number] | null =
        cpu ? [cpu.x * Camera.EARTH_SCALE, cpu.y * Camera.EARTH_SCALE, cpu.z * Camera.EARTH_SCALE] : null;
      const deltaMeters = cpuMeters
        ? length([gpuMeters[0] - cpuMeters[0], gpuMeters[1] - cpuMeters[1], gpuMeters[2] - cpuMeters[2]])
        : null;
      report.globeCrossCheck = { gpuWorldPosMeters: gpuMeters, cpuRaycastMeters: cpuMeters, deltaMeters };
      log(`[4] 地球交叉验证: GPU worldPos vs CPU raycast δ=${deltaMeters !== null ? deltaMeters.toFixed(3) + 'm' : 'n/a'}`);
    }
  }

  // ── 5. in-flight throttling: concurrent pick() reuses one promise ─────────
  {
    const promises = [
      picker.pick(px, py), picker.pick(10, 10), picker.pick(20, 20),
      picker.pick(30, 30), picker.pick(40, 40),
    ];
    const distinct = new Set(promises).size;
    await promises[0];
    report.throttling = { concurrentCalls: promises.length, distinctPromises: distinct, pass: distinct === 1 };
    log(`[5] 并发节流: ${promises.length} 次调用 → ${distinct} 个不同 promise ${distinct === 1 ? '(PASS)' : '(FAIL)'}`);
  }

  // ── 6. 100 consecutive picks ──────────────────────────────────────────────
  {
    const readbacks: number[] = [];
    const t0 = performance.now();
    for (let i = 0; i < 100; i++) {
      const tx = i % 2 === 0 ? px : 10 + (i % 7) * 13;
      const ty = i % 2 === 0 ? py : 10 + (i % 11) * 17;
      const result = await picker.pick(tx, ty);
      if (!result) continue;
      readbacks.push(result.timing.readbackMs);
    }
    const totalMs = performance.now() - t0;
    const sum = readbacks.reduce((a, b) => a + b, 0);
    report.batch100 = {
      totalMs,
      avgPerPickMs: totalMs / 100,
      avgReadbackMs: sum / readbacks.length,
      minReadbackMs: Math.min(...readbacks),
      maxReadbackMs: Math.max(...readbacks),
    };
    log(`[6] 连续 100 次 pick: 总耗时 ${totalMs.toFixed(1)}ms (${(totalMs / 100).toFixed(2)}ms/次)`);
    log(`    readback 平均 ${report.batch100.avgReadbackMs.toFixed(2)}ms  最小 ${report.batch100.minReadbackMs.toFixed(2)}ms  最大 ${report.batch100.maxReadbackMs.toFixed(2)}ms`);
  }

  // ── Summary ───────────────────────────────────────────────────────────────
  log('');
  log('结论: 见 docs/decisions/0002-gpu-picking-readback.md');

  report.debug = {
    ...picker.debug,
    consoleErrors: consoleErrors.slice(0, 10),
  };

  (window as unknown as { __pickSpike: PickSpikeReport }).__pickSpike = report;
  console.log('PICK_SPIKE_REPORT', JSON.stringify(report));
}

run().catch((err: unknown) => {
  const message = err instanceof Error ? `${err.message}\n${err.stack ?? ''}` : String(err);
  console.error('pick-spike failed:', err);
  setLog(`Spike 失败:\n${message}`);
});
