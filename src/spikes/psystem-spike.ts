import { Viewer } from '../core/Viewer';
import { Cartesian3 } from '../math/Cartesian3';
import { ParticleSystem, type ParticleMode } from '../effect/ParticleSystem';

/**
 * 粒子系统正式版验证页（T3.4）：四种模式切换 + G-Buffer 深度遮挡验证。
 *
 * URL 参数：
 *   ?count=10000           粒子数
 *   ?mode=fire             初始模式 'fire' | 'fountain' | 'fireworks' | 'smoke'
 *   ?cam=4000000           相机高度（米，默认 4000 km，侧视 30° 偏移）
 *
 * 统计渲染到 #stats 并镜像到 window.__psystemSpike。
 */

interface SpikeStats {
  count: number;
  mode: string;
  fps: number;
  avgMs: number;
  minMs: number;
  maxMs: number;
  overBudgetFrames: number; // 采样窗口内超过 16.67 ms 预算的帧数
  sampledFrames: number;
  destroyed: number;        // 累计销毁的 ParticleSystem 实例数
}

/** 发射器：北京（116.391E, 39.907N），地表。 */
const EMITTER_LON = 116.391;
const EMITTER_LAT = 39.907;

const MODES: ParticleMode[] = ['fire', 'fountain', 'fireworks', 'smoke'];

async function main(): Promise<void> {
  const container = document.getElementById('cesiumContainer');
  const statsEl = document.getElementById('stats');
  if (!container) throw new Error('psystem-spike: #cesiumContainer not found');

  const query = new URLSearchParams(window.location.search);
  const count = Math.max(64, Number(query.get('count') ?? 10_000));
  const modeRaw = query.get('mode');
  const initialMode = MODES.includes(modeRaw as ParticleMode)
    ? (modeRaw as ParticleMode) : 'fire';
  const camAlt = Number(query.get('cam') ?? 4_000_000);

  // ── Viewer：地球为背景 ───────────────────────────────────────────────────
  const viewer = new Viewer(container, { powerPreference: 'high-performance' });
  await viewer.initialize();

  // 相机从发射点以西 30° 侧视，避免正俯视时粒子柱被压扁。
  viewer.camera.setView({
    destination: Cartesian3.fromDegrees(EMITTER_LON - 30, EMITTER_LAT, camAlt),
  });

  viewer.globe.loadImageryFromUrl('/images/world.jpg').catch((err: unknown) => {
    console.warn('psystem-spike: world.jpg not found, procedural fallback', err);
    viewer.globe.generateProceduralTexture();
  });

  // ── 粒子系统：切换即销毁旧实例（buffer destroy），创建新实例 ────────────
  let current: ParticleSystem | null = null;
  let destroyedCount = 0;

  const stats: SpikeStats = {
    count, mode: initialMode,
    fps: 0, avgMs: 0, minMs: Infinity, maxMs: 0,
    overBudgetFrames: 0, sampledFrames: 0, destroyed: 0,
  };

  const setMode = (mode: ParticleMode): void => {
    if (current) {
      viewer.scene.effects.remove(current);
      current.destroy(); // 释放 particle/params/quad 全部 GPU buffer
      destroyedCount += 1;
    }
    current = new ParticleSystem(viewer.context, {
      scene: viewer.scene,
      count,
      origin: { lon: EMITTER_LON, lat: EMITTER_LAT, height: 0 },
      mode,
    });
    viewer.scene.effects.add(current); // EffectManager：composite 之后 execute

    for (const btn of document.querySelectorAll<HTMLButtonElement>('#controls button')) {
      btn.classList.toggle('active', btn.dataset.mode === mode);
    }
    stats.mode = mode;
  };

  for (const mode of MODES) {
    const btn = document.querySelector<HTMLButtonElement>(
      `#controls button[data-mode="${mode}"]`,
    );
    if (!btn) throw new Error(`psystem-spike: button for "${mode}" not found`);
    btn.addEventListener('click', () => setMode(mode));
  }
  setMode(initialMode);

  // ── 帧计时（rAF delta 覆盖 compute + render + present）─────────────────
  let last = performance.now();
  let acc = 0;
  let n = 0;
  let lastStatsAt = performance.now();

  const renderStats = (): void => {
    if (!statsEl) return;
    statsEl.textContent = [
      `particles   : ${stats.count.toLocaleString()} (${stats.mode})`,
      `fps         : ${stats.fps}`,
      `frame avg   : ${stats.avgMs.toFixed(2)} ms`,
      `frame min   : ${stats.minMs === Infinity ? '-' : stats.minMs.toFixed(2)} ms`,
      `frame max   : ${stats.maxMs.toFixed(2)} ms`,
      `over budget : ${stats.overBudgetFrames} / ${stats.sampledFrames} (>16.67 ms)`,
      `switches    : ${stats.destroyed} destroyed`,
      `depth test  : G-Buffer depthView (depthReadOnly + less)`,
    ].join('\n');
  };

  viewer.preRender = () => {
    const now = performance.now();
    const delta = now - last;
    last = now;

    // 跳过预热样本与切后台的异常大 delta。
    if (delta > 0.5 && delta < 1000) {
      acc += delta;
      n += 1;
      stats.sampledFrames += 1;
      if (delta < stats.minMs) stats.minMs = delta;
      if (delta > stats.maxMs) stats.maxMs = delta;
      if (delta > 16.67) stats.overBudgetFrames += 1;
    }

    if (now - lastStatsAt >= 500) {
      stats.avgMs = n > 0 ? acc / n : 0;
      stats.fps = stats.avgMs > 0 ? Math.round(1000 / stats.avgMs) : 0;
      stats.destroyed = destroyedCount;
      renderStats();
      (window as unknown as { __psystemSpike: SpikeStats }).__psystemSpike = stats;
      acc = 0;
      n = 0;
      lastStatsAt = now;
    }
  };

  console.info(
    `psystem-spike: ${count} particles @ Beijing, mode=${initialMode}, ` +
    `EffectManager-mounted with G-Buffer depth testing`,
  );

  // QA automation hook: occlusion checks drive the camera programmatically.
  (window as unknown as { __psystemViewer: Viewer }).__psystemViewer = viewer;
}

main().catch((err: unknown) => {
  console.error('psystem-spike failed:', err);
  const errorEl = document.getElementById('error');
  if (errorEl) {
    errorEl.style.display = 'block';
    errorEl.textContent = `psystem-spike error: ${err instanceof Error ? err.message : String(err)}`;
  }
});
