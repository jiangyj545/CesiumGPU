import { Viewer } from '../core/Viewer';
import { Cartesian3 } from '../math/Cartesian3';
import { CesiumMath } from '../math/CesiumMath';
import { Camera } from '../scene/Camera';
import { Weather } from '../effect/Weather';
import type { WeatherType } from '../effect/Weather';

/**
 * Spike: weather effects (T3.5) - camera-anchored rain / snow over the globe.
 *
 * UI: rain / snow / stop buttons + density slider + FPS HUD.
 *
 * URL parameters:
 *   ?type=rain|snow   initial weather (default rain)
 *   ?density=0.6      initial intensity 0..1
 *   ?scale=8          visual size scale (streak length / flake size)
 *   ?alt=3000         camera altitude in metres
 *
 * Stats are mirrored on window.__weatherSpike.
 */

interface SpikeStats {
  type: string;
  intensity: number;
  instances: number;
  fps: number;
  avgMs: number;
  overBudgetFrames: number;
  sampledFrames: number;
}

/** Demo location: Beijing (116.4E, 39.9N). */
const LON = 116.4;
const LAT = 39.9;

async function main(): Promise<void> {
  const container = document.getElementById('cesiumContainer');
  const statsEl = document.getElementById('stats');
  if (!container) throw new Error('weather-spike: #cesiumContainer not found');

  const query = new URLSearchParams(window.location.search);
  const initialType: WeatherType = query.get('type') === 'snow' ? 'snow' : 'rain';
  const initialDensity = clamp01(Number(query.get('density') ?? 0.6));
  const sizeScale = Math.max(Number(query.get('scale') ?? 8), 0.5);
  const altitude = Math.max(Number(query.get('alt') ?? 3000), 200);
  // Camera tilt from nadir in degrees (default -65); &pitch=-90 = nadir view.
  const pitchDeg = Number(query.get('pitch') ?? -65);
  // Near plane in metres; &near=0 keeps the engine default (~32 km).
  const nearM = Number(query.get('near') ?? 20);

  // ── Viewer with the globe as background ─────────────────────────────────
  const viewer = new Viewer(container, { powerPreference: 'high-performance' });
  await viewer.initialize();

  // Surface async WebGPU validation errors (they do not throw in JS).
  const gpuErrors: string[] = [];
  (window as unknown as { __gpuErrors: string[] }).__gpuErrors = gpuErrors;
  viewer.context.device.addEventListener('uncapturederror', ((ev: Event) => {
    const msg = (ev as unknown as { error: { message: string } }).error.message;
    if (gpuErrors.length < 8) gpuErrors.push(msg.slice(0, 600));
    console.error('weather-spike uncaptured GPU error:', msg);
    const errorEl = document.getElementById('error');
    if (errorEl && errorEl.style.display === 'none') {
      errorEl.style.display = 'block';
      errorEl.textContent = `WebGPU error: ${msg}`;
    }
  }) as EventListener);

  // The engine default near plane is ~32 km (space-scale); pull it in so the
  // few-hundred-metre weather volume around the camera is not clipped.
  if (nearM > 0) viewer.camera.frustum.near = nearM / Camera.EARTH_SCALE;

  viewer.camera.setView({
    destination: Cartesian3.fromDegrees(LON, LAT, altitude),
    orientation: { pitch: CesiumMath.toRadians(pitchDeg) },
  });

  viewer.globe.loadImageryFromUrl('/images/world.jpg').catch((err: unknown) => {
    console.warn('weather-spike: world.jpg not found, procedural fallback', err);
    viewer.globe.generateProceduralTexture();
  });

  // ── Weather effect lifecycle ─────────────────────────────────────────────
  let weather: Weather | null = null;
  let currentType: WeatherType | null = null;
  let intensity = initialDensity;

  const spawn = (type: WeatherType): void => {
    if (weather) {
      viewer.scene.effects.remove(weather);
      weather.destroy();
    }
    weather = new Weather(viewer.context, { type, intensity, sizeScale });
    weather.setCamera(viewer.camera);
    viewer.scene.effects.add(weather);
    weather.show = true;
    currentType = type;
    updateButtons();
  };

  const stop = (): void => {
    if (weather) weather.show = false;
    currentType = null;
    updateButtons();
  };

  const updateButtons = (): void => {
    for (const btn of document.querySelectorAll<HTMLButtonElement>('[data-weather]')) {
      btn.classList.toggle('active', btn.dataset.weather === currentType);
    }
  };

  for (const btn of document.querySelectorAll<HTMLButtonElement>('[data-weather]')) {
    btn.addEventListener('click', () => {
      const type = btn.dataset.weather;
      if (type === 'rain' || type === 'snow') spawn(type);
      else stop();
    });
  }

  const slider = document.querySelector<HTMLInputElement>('#density');
  const densityLabel = document.querySelector<HTMLSpanElement>('#densityValue');
  if (slider) {
    slider.value = String(initialDensity);
    if (densityLabel) densityLabel.textContent = `${Math.round(initialDensity * 100)}%`;
    slider.addEventListener('input', () => {
      intensity = clamp01(Number(slider.value));
      weather?.setIntensity(intensity);
      if (densityLabel) densityLabel.textContent = `${Math.round(intensity * 100)}%`;
    });
  }

  spawn(initialType);

  // Debug/verification handle (mirrors particle-spike's ?debug pattern).
  (window as unknown as { __weatherDebug: unknown }).__weatherDebug = {
    get type() { return currentType; },
    get intensity() { return intensity; },
    get show() { return weather?.show ?? false; },
    get liveIntensity() { return weather?.intensity ?? 0; },
    get count() { return weather ? Math.round((currentType === 'rain' ? 20000 : 8000) * intensity) : 0; },
  };

  // ── Frame timing + depth-view refresh (G-Buffer recreates on resize) ─────
  let last = performance.now();
  let acc = 0;
  let n = 0;
  let lastStatsAt = performance.now();
  const stats: SpikeStats = {
    type: initialType, intensity: initialDensity, instances: 0,
    fps: 0, avgMs: 0, overBudgetFrames: 0, sampledFrames: 0,
  };

  const renderStats = (): void => {
    if (!statsEl) return;
    statsEl.textContent = [
      `weather     : ${stats.type} (density ${Math.round(stats.intensity * 100)}%)`,
      `instances   : ${stats.instances.toLocaleString()}`,
      `fps         : ${stats.fps}`,
      `frame avg   : ${stats.avgMs.toFixed(2)} ms`,
      `over budget : ${stats.overBudgetFrames} / ${stats.sampledFrames} (>16.67 ms)`,
      `camera alt  : ${((Cartesian3.magnitude(viewer.camera.position) - 1) * Camera.EARTH_SCALE / 1000).toFixed(1)} km`,
      `ops         : drag = orbit, wheel = zoom`,
    ].join('\n');
  };

  viewer.preRender = () => {
    const now = performance.now();
    const delta = now - last;
    last = now;

    // Keep the occlusion depth binding current (recreated on canvas resize).
    weather?.setDepthView(viewer.scene.gbufferPass?.getDepthView() ?? null);

    if (delta > 0.5 && delta < 1000) {
      acc += delta;
      n += 1;
      stats.sampledFrames += 1;
      if (delta > 16.67) stats.overBudgetFrames += 1;
    }

    if (now - lastStatsAt >= 500) {
      stats.avgMs = n > 0 ? acc / n : 0;
      stats.fps = stats.avgMs > 0 ? Math.round(1000 / stats.avgMs) : 0;
      stats.type = currentType ?? 'off';
      stats.intensity = intensity;
      stats.instances = weather && weather.show && currentType
        ? Math.round((currentType === 'rain' ? 20000 : 8000) * intensity)
        : 0;
      renderStats();
      (window as unknown as { __weatherSpike: SpikeStats }).__weatherSpike = stats;
      acc = 0;
      n = 0;
      lastStatsAt = now;
    }
  };

  console.info(`weather-spike: ${initialType} density=${intensity} sizeScale=${sizeScale}`);
}

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

main().catch((err: unknown) => {
  console.error('weather-spike failed:', err);
  const errorEl = document.getElementById('error');
  if (errorEl) {
    errorEl.style.display = 'block';
    errorEl.textContent = `weather-spike error: ${err instanceof Error ? err.message : String(err)}`;
  }
});
