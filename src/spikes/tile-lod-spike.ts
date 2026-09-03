import { Viewer } from '../core/Viewer';
import { Cartesian3 } from '../math/Cartesian3';
import { XyzImageryProvider } from '../imagery/XyzImageryProvider';
import type { ImageryLayer } from '../imagery/ImageryLayer';

/**
 * Spike: imagery tile system (T1.1).
 *
 * Loads an OSM XYZ imagery layer onto the globe and streams tiles through
 * the quadtree LOD pipeline (SSE refinement, frustum + horizon culling,
 * 10-uploads-per-frame budget, LRU texture pool, parent-tile upsampling).
 *
 * The HUD shows selected tile count, render primitive count, LRU cache
 * hit rate and FPS; toolbar buttons fly the camera between altitudes so
 * the level transitions can be exercised.
 *
 * Open at: /spikes/tile-lod-spike.html
 */

/** Camera target for the altitude presets: Beijing. */
const TARGET_LON = 116.4;
const TARGET_LAT = 39.9;

const EARTH_RADIUS_M = 6378137.0;

/** WGS84 polar ratio — the rendered globe's ellipsoid squash. */
const POLAR_RATIO = 6356752.3142 / 6378137.0;

/**
 * Camera height above the rendered (squashed) ellipsoid, in metres.
 * Solves where the camera's radial ray intersects the engine ellipsoid
 * `x² + (y² + z²) / ratio² = 1` and converts the gap to metres.
 */
function cameraAltitudeMetres(position: Cartesian3): number {
  const mag      = Cartesian3.magnitude(position);
  const ux       = position.x / mag;
  const uy       = position.y / mag;
  const uz       = position.z / mag;
  const r2       = POLAR_RATIO * POLAR_RATIO;
  const surfaceT = 1.0 / Math.sqrt(ux * ux + ((uy * uy) + (uz * uz)) / r2);
  return Math.max(0, (mag - surfaceT) * EARTH_RADIUS_M);
}

async function main(): Promise<void> {
  const container = document.getElementById('cesiumContainer');
  const statsEl   = document.getElementById('stats');
  if (!container || !statsEl) {
    throw new Error('tile-lod-spike: #cesiumContainer / #stats not found');
  }

  // ── Viewer + globe ──────────────────────────────────────────────────────
  const viewer = new Viewer(container, { powerPreference: 'high-performance' });
  await viewer.initialize();

  // Start over Beijing at continental range.
  viewer.camera.setView({
    destination: Cartesian3.fromDegrees(TARGET_LON, TARGET_LAT, 4_000_000),
  });

  // Legacy fallback texture underneath the tiles (also proves the fallback
  // path still renders where tiles have not streamed in yet).
  viewer.globe.loadImageryFromUrl('/images/world.jpg').catch((err: unknown) => {
    console.warn('tile-lod-spike: world.jpg not found, procedural fallback', err);
    viewer.globe.generateProceduralTexture();
  });

  // ── The system under test: imagery layer ────────────────────────────────
  // Default: the OSM XYZ preset. URL params allow pointing at any XYZ
  // source (useful on networks where OSM is unreachable):
  //   ?tiles=<url-template>&s=<comma,separated,subdomains>&maxlevel=18
  const query      = new URLSearchParams(window.location.search);
  const tilesParam = query.get('tiles');
  const subdomains = (query.get('s') ?? '').split(',').filter(s => s.length > 0);
  const maxLevel   = Number(query.get('maxlevel') ?? 18);
  const layer = viewer.globe.layers.addImageryProvider(
    tilesParam !== null
      ? new XyzImageryProvider({
          url:        tilesParam,
          subdomains,
          maximumLevel: Number.isFinite(maxLevel) ? maxLevel : 18,
          tileSize:   256,
        })
      : XyzImageryProvider.osm()
  );

  // Console/debug handle (same convention as particle-spike's window handle).
  (window as unknown as {
    __tileLodSpike: { viewer: Viewer; layer: ImageryLayer };
  }).__tileLodSpike = { viewer, layer };

  // Altitude preset buttons → fly to different LOD regimes.
  const toolbar = document.getElementById('toolbar');
  toolbar?.querySelectorAll<HTMLButtonElement>('button[data-alt]').forEach(button => {
    button.addEventListener('click', () => {
      const altitude = Number(button.dataset.alt);
      if (Number.isFinite(altitude) && altitude > 0) {
        viewer.camera.flyTo({
          destination: Cartesian3.fromDegrees(TARGET_LON, TARGET_LAT, altitude),
        });
      }
    });
  });

  // ── HUD: FPS + tile pipeline stats ──────────────────────────────────────
  let frames      = 0;
  let windowStart = performance.now();
  let fps         = 0;
  let lastHud     = 0;

  viewer.preRender = () => {
    frames++;
    const now = performance.now();
    if (now - windowStart >= 500) {
      fps = Math.round((frames * 1000) / (now - windowStart));
      frames      = 0;
      windowStart = now;
    }
    if (now - lastHud < 200) return;
    lastHud = now;

    const s        = layer.stats;
    const altitude = cameraAltitudeMetres(viewer.camera.position);
    statsEl.textContent =
      `FPS:          ${fps}\n` +
      `相机高度:     ${formatKm(altitude)}\n` +
      `选中瓦片:     ${s.selectedTiles}\n` +
      `渲染瓦片:     ${viewer.globe.tilePrimitiveCount}\n` +
      `纹理缓存:     ${s.cache.size} / ${s.cache.capacity}  (命中率 ${(s.cache.hitRate * 100).toFixed(1)}%)\n` +
      `下载中:       ${s.inFlightRequests}   待上传: ${s.pendingUploads}   失败: ${s.failedTiles}\n` +
      `累计:         命中 ${s.cache.hits}  未命中 ${s.cache.misses}  淘汰 ${s.cache.evictions}`;
  };
}

/** Human-readable metric altitude (km above 1 km, else metres). */
function formatKm(metres: number): string {
  return metres >= 1000 ? `${(metres / 1000).toFixed(1)} km` : `${metres.toFixed(0)} m`;
}

main().catch((error: unknown) => {
  console.error(error);
  const el = document.getElementById('error');
  if (el) {
    el.style.display = 'block';
    el.textContent = `tile-lod-spike failed: ${error instanceof Error ? error.message : String(error)}`;
  }
});
