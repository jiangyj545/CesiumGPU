import { Viewer } from '../core/Viewer';
import { Cartesian3 } from '../math/Cartesian3';
import { CircleScan } from '../effect/CircleScan';

/**
 * Spike: CircleScan 地表圆形扫描特效（T3.3，EV_CircleScan 对齐）。
 *
 * 验证点：
 *  - 北京（青色，500 km，3 环 / 2.5 s）与上海（橙红，250 km，2 环 / 1.6 s）
 *    两个不同半径 / 颜色 / 周期的扫描圈同时运行
 *  - 扫描波从中心周期性向外扩散（fract 波前）
 *  - 圆盘逐顶点投影到椭球面 —— 贴地无悬浮
 *  - additive（one, one）叠加在已合成的场景之上
 *  - FPS HUD（viewer.preRender 统计）
 *
 * 无 WebGPU 环境下仅报告 JS 模块加载无报错，不运行渲染。
 * 状态镜像到 window.__scanSpike。
 */

interface ScanSpikeReport {
  jsLoaded: boolean;
  webgpu: boolean;
  fps: number;
  avgMs: number;
  samples: number;
}

const BEIJING  = { lon: 116.4,  lat: 39.9 };
const SHANGHAI = { lon: 121.47, lat: 31.23 };

async function main(): Promise<void> {
  const container = document.getElementById('cesiumContainer');
  const statsEl = document.getElementById('stats');
  if (!container) throw new Error('scan-spike: #cesiumContainer not found');
  if (!statsEl) throw new Error('scan-spike: #stats not found');

  // ── 无 GPU 环境：仅确认模块加载无报错 ─────────────────────────────────────
  const gpu = (navigator as { gpu?: unknown }).gpu;
  if (!gpu) {
    statsEl.textContent = 'WebGPU 不可用 — scan-spike JS 模块加载无报错（未运行渲染）';
    (window as unknown as { __scanSpike: ScanSpikeReport }).__scanSpike = {
      jsLoaded: true, webgpu: false, fps: 0, avgMs: 0, samples: 0,
    };
    return;
  }

  const viewer = new Viewer(container, { powerPreference: 'high-performance' });
  await viewer.initialize();

  // 初始视角：覆盖北京—上海两点（华北—华东上空）。
  viewer.camera.setView({
    destination: Cartesian3.fromDegrees(119.0, 35.5, 5_500_000),
  });

  viewer.globe.loadImageryFromUrl('/images/world.jpg').catch((err: unknown) => {
    console.warn('scan-spike: world.jpg not found, procedural fallback', err);
    viewer.globe.generateProceduralTexture();
  });

  // ── 两个扫描特效：不同半径 / 颜色 / 周期 / 环数 ───────────────────────────
  const beijingScan = new CircleScan(viewer.context, {
    center: { lon: BEIJING.lon, lat: BEIJING.lat },
    radius: 500_000,                       // 500 km
    color: [0.2, 0.9, 1.0, 0.85],          // 青色
    speed: 2.5,                            // 2.5 s / 轮
    rings: 3,
  });
  beijingScan.setCamera(viewer.camera);
  viewer.scene.effects.add(beijingScan);

  const shanghaiScan = new CircleScan(viewer.context, {
    center: { lon: SHANGHAI.lon, lat: SHANGHAI.lat },
    radius: 250_000,                       // 250 km
    color: [1.0, 0.35, 0.15, 0.9],         // 橙红
    speed: 1.6,
    rings: 2,
  });
  shanghaiScan.setCamera(viewer.camera);
  viewer.scene.effects.add(shanghaiScan);

  // ── FPS HUD ──────────────────────────────────────────────────────────────
  let last = performance.now();
  let acc = 0;
  let n = 0;
  let lastStatsAt = last;
  const report: ScanSpikeReport = { jsLoaded: true, webgpu: true, fps: 0, avgMs: 0, samples: 0 };

  const renderStats = (): void => {
    statsEl.textContent = [
      'circle scan  : 北京 500km 青 (3环/2.5s) + 上海 250km 橙红 (2环/1.6s)',
      `fps          : ${report.fps}`,
      `frame avg    : ${report.avgMs.toFixed(2)} ms`,
      `samples      : ${report.samples}`,
      'pipeline     : effects.execute() → additive (one, one) onto swap-chain',
      'keys         : B = 北京开关, S = 上海开关',
    ].join('\n');
  };

  viewer.preRender = () => {
    const now = performance.now();
    const delta = now - last;
    last = now;
    if (delta > 0.5 && delta < 1000) {
      acc += delta;
      n += 1;
    }
    if (now - lastStatsAt >= 500) {
      report.avgMs = n > 0 ? acc / n : 0;
      report.fps = report.avgMs > 0 ? Math.round(1000 / report.avgMs) : 0;
      report.samples = n;
      renderStats();
      (window as unknown as { __scanSpike: ScanSpikeReport }).__scanSpike = report;
      acc = 0;
      n = 0;
      lastStatsAt = now;
    }
  };

  // ── show 开关演示 ────────────────────────────────────────────────────────
  window.addEventListener('keydown', (e: KeyboardEvent) => {
    if (e.key === 'b' || e.key === 'B') beijingScan.show = !beijingScan.show;
    if (e.key === 's' || e.key === 'S') shanghaiScan.show = !shanghaiScan.show;
  });

  console.info('scan-spike: CircleScan 北京/上海已挂载（scene.effects），B/S 键切换可见性');
}

main().catch((err: unknown) => {
  console.error('scan-spike failed:', err);
  const errorEl = document.getElementById('error');
  if (errorEl) {
    errorEl.style.display = 'block';
    errorEl.textContent = `scan-spike error: ${err instanceof Error ? err.message : String(err)}`;
  }
});
