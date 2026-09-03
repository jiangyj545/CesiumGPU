import { Viewer } from '../core/Viewer';
import { Cartesian3 } from '../math/Cartesian3';
import { CesiumMath } from '../math/CesiumMath';
import type { FlyToOptions } from '../scene/Camera';

/**
 * Spike: camera flyTo animation (T2.6, CameraFlightPath).
 *
 * Verifies:
 *  - Beijing → Shanghai (3 s, medium arc)
 *  - Beijing → New York (5 s, long arc) — parabolic altitude arch, no ground hit
 *  - Low-altitude 1 km look-down at pitch = -60° — exact endpoint landing
 *  - complete callback fires on arrival
 *  - drag / wheel during flight cancels it (cancel callback + camera control)
 *  - min altitude per flight is logged to console (no ground penetration)
 *
 * Without WebGPU the page reports "JS loaded, no errors" and exits quietly.
 * Report object is mirrored on window.__flyToSpike.
 */

interface FlyToSpikeReport {
  jsLoaded: boolean;
  webgpu: boolean;
  flightsCompleted: number;
  flightsCancelled: number;
  /** Lowest camera altitude (km) sampled during the current/last flight. */
  minAltitudeKm: number | null;
  noGroundHit: boolean;
  lastEvent: string;
}

const EARTH_RADIUS_KM = 6378.137;

const BEIJING  = { lon: 116.4,   lat: 39.9 };
const SHANGHAI = { lon: 121.47,  lat: 31.23 };
const NEWYORK  = { lon: -74.006, lat: 40.713 };

async function main(): Promise<void> {
  const container = document.getElementById('cesiumContainer');
  const statsEl = document.getElementById('stats');
  if (!container) throw new Error('flyto-spike: #cesiumContainer not found');
  if (!statsEl) throw new Error('flyto-spike: #stats not found');

  // ── No-GPU environment: confirm the module loaded without errors ──────────
  const gpu = (navigator as { gpu?: unknown }).gpu;
  if (!gpu) {
    statsEl.textContent = 'WebGPU 不可用 — flyto-spike JS 模块加载无报错（未运行动画）';
    const w = window as { __flyToSpike?: FlyToSpikeReport };
    w.__flyToSpike = { jsLoaded: true, webgpu: false, flightsCompleted: 0, flightsCancelled: 0, minAltitudeKm: null, noGroundHit: true, lastEvent: 'no-webgpu' };
    return;
  }

  const viewer = new Viewer(container);
  try {
    await viewer.initialize();
  } catch (err: unknown) {
    statsEl.textContent = `WebGPU 初始化失败: ${String(err)}\n(JS 模块本身无报错)`;
    return;
  }

  // Initial view: 20 000 km above Beijing
  viewer.camera.setView({
    destination: { lon: BEIJING.lon, lat: BEIJING.lat, height: 20_000_000 },
  });

  viewer.globe.loadImageryFromUrl('/images/world.jpg').catch((err: unknown) => {
    console.warn('flyto-spike: imagery load failed, procedural fallback in effect.', err);
  });

  const report: FlyToSpikeReport = {
    jsLoaded: true,
    webgpu: true,
    flightsCompleted: 0,
    flightsCancelled: 0,
    minAltitudeKm: null,
    noGroundHit: true,
    lastEvent: 'ready',
  };
  (window as { __flyToSpike?: FlyToSpikeReport }).__flyToSpike = report;

  let flightMinAltKm = Infinity;

  function startFlight(name: string, options: FlyToOptions): void {
    flightMinAltKm = Infinity;
    report.minAltitudeKm = null;
    report.lastEvent = `${name} 飞行中…`;
    viewer.camera.flyTo({
      ...options,
      complete: () => {
        report.flightsCompleted++;
        report.minAltitudeKm = Number.isFinite(flightMinAltKm) ? flightMinAltKm : null;
        report.noGroundHit = report.minAltitudeKm === null || report.minAltitudeKm > 0;
        report.lastEvent = `${name} 完成`;
        console.log(
          `[flyTo] ${name} 完成 — 本次最低高度 ${report.minAltitudeKm?.toFixed(1) ?? 'n/a'} km` +
          (report.noGroundHit ? '（未穿地）' : '（穿地！）'),
        );
      },
      cancel: () => {
        report.flightsCancelled++;
        report.lastEvent = `${name} 被打断`;
        console.log(`[flyTo] ${name} 被用户输入打断`);
      },
    });
  }

  function bind(id: string, fn: () => void): void {
    document.getElementById(id)?.addEventListener('click', fn);
  }

  bind('btn-shanghai', () => startFlight('上海', {
    destination: { lon: SHANGHAI.lon, lat: SHANGHAI.lat, height: 2_000_000 },
    duration: 3,
  }));

  bind('btn-newyork', () => startFlight('纽约', {
    destination: { lon: NEWYORK.lon, lat: NEWYORK.lat, height: 15_000_000 },
    duration: 5,
  }));

  bind('btn-lowalt', () => startFlight('北京低空 1km', {
    destination: { lon: BEIJING.lon, lat: BEIJING.lat, height: 1_000 },
    duration: 4,
    orientation: { heading: 0, pitch: CesiumMath.toRadians(-60) },
  }));

  bind('btn-reset', () => startFlight('北京上空', {
    destination: { lon: BEIJING.lon, lat: BEIJING.lat, height: 20_000_000 },
    duration: 3,
  }));

  viewer.preRender = () => {
    const cam = viewer.camera;
    const r = Cartesian3.magnitude(cam.position);
    const altKm = (r - 1) * EARTH_RADIUS_KM;
    if (cam.inFlight && altKm < flightMinAltKm) flightMinAltKm = altKm;

    const lonDeg = CesiumMath.toDegrees(Math.atan2(cam.position.y, cam.position.x));
    const latDeg = CesiumMath.toDegrees(Math.asin(CesiumMath.clamp(cam.position.z / r, -1, 1)));
    const { heading, pitch } = cam._deriveHeadingPitch();
    const status = cam.inFlight
      ? `飞行中（最低 ${Number.isFinite(flightMinAltKm) ? flightMinAltKm.toFixed(1) : '…'} km）`
      : '空闲（拖拽/滚轮可打断飞行）';

    statsEl.textContent = [
      'flyTo Spike (T2.6)',
      `位置: lon ${lonDeg.toFixed(2)}°  lat ${latDeg.toFixed(2)}°`,
      `高度: ${altKm.toFixed(1)} km`,
      `朝向: heading ${CesiumMath.toDegrees(CesiumMath.negativePiToPi(heading)).toFixed(1)}°  pitch ${CesiumMath.toDegrees(pitch).toFixed(1)}°`,
      `状态: ${status}`,
      `完成 ${report.flightsCompleted} 次 / 打断 ${report.flightsCancelled} 次 / 最低高度 ${report.minAltitudeKm?.toFixed(1) ?? '-'} km`,
      `最近: ${report.lastEvent}`,
    ].join('\n');
  };
}

main().catch((err: unknown) => {
  console.error('flyto-spike: fatal error', err);
});

export {};
