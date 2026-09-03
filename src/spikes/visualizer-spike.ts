import { Viewer } from '../core/Viewer';
import { Cartesian3 } from '../math/Cartesian3';
import { Color } from '../math/Color';
import { XyzImageryProvider } from '../imagery/XyzImageryProvider';
import { Entity } from '../dataSource/Entity';

/**
 * Spike: entity rendering bridge (T2.2).
 *
 * Verifies:
 * - 10 points  → OverlayPass (screen space, constant pixel size)
 * - 3 labels   → OverlayPass (Canvas2D-baked text, depth-tested)
 * - 2 polylines→ WBOIT translucent queue
 * - 2 polygons → one opaque (G-Buffer, lit) + one translucent (WBOIT)
 * - globe occlusion: points behind the Earth must disappear
 *
 * Open: /spikes/visualizer-spike.html  (requires WebGPU)
 */

// ── Helpers ────────────────────────────────────────────────────────────────────

/**
 * lon/lat (degrees) + height (metres) → engine normalised ECEF.
 * Same convention as `GeoJsonDataSource`: unit-length radial direction scaled
 * by (1 + height / EARTH_RADIUS).
 */
function lonLat(lon: number, lat: number, heightMetres = 0): Cartesian3 {
  const dir = Cartesian3.normalize(Cartesian3.fromDegrees(lon, lat, 0), new Cartesian3());
  const scale = 1 + heightMetres / Cartesian3.EARTH_RADIUS;
  return new Cartesian3(dir.x * scale, dir.y * scale, dir.z * scale);
}

function colorOf(css: string, alpha = 1): Color {
  const c = Color.fromCssColorString(css);
  c.alpha = alpha;
  return c;
}

// ── Entity data ────────────────────────────────────────────────────────────────

const CITIES: Array<{
  name: string; lon: number; lat: number; css: string; size: number; label?: string;
}> = [
  { name: '北京',   lon: 116.39,  lat: 39.90,  css: '#ff3030', size: 14, label: '北京' },
  { name: '上海',   lon: 121.47,  lat: 31.23,  css: '#ffd400', size: 18, label: '上海' },
  { name: '广州',   lon: 113.26,  lat: 23.13,  css: '#30d0ff', size: 12, label: '广州' },
  { name: '成都',   lon: 104.07,  lat: 30.57,  css: '#ff40ff', size: 10 },
  { name: '乌鲁木齐', lon: 87.62,  lat: 43.83,  css: '#ff8020', size: 12 },
  { name: '拉萨',   lon: 91.11,   lat: 29.65,  css: '#a0ff40', size: 10 },
  { name: '哈尔滨', lon: 126.63,  lat: 45.75,  css: '#ffffff', size: 10 },
  { name: '香港',   lon: 114.17,  lat: 22.32,  css: '#ff60a0', size: 12 },
  // Back-face occlusion tests: visible only when the globe is turned.
  { name: '纽约',   lon: -74.01,  lat: 40.71,  css: '#ff2020', size: 12 },
  { name: '悉尼',   lon: 151.21,  lat: -33.87, css: '#20ff60', size: 12 },
];

const JINGGUANG = [
  [116.39, 39.90], [113.63, 34.75], [114.31, 30.59], [112.94, 28.23], [113.26, 23.13],
];

const MARITIME = [
  [121.47, 31.23], [121.74, 25.13], [120.98, 14.60], [103.85, 1.29],
];

// Lifted slightly off the sphere: the WBOIT accumulation pass depth-tests
// with 'less', so a ribbon exactly coplanar with the globe would z-fight.
const LINE_LIFT_METRES = 2000;

// ── Main ───────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const container = document.getElementById('cesiumContainer');
  const statsEl   = document.getElementById('stats');
  if (!container || !statsEl) {
    throw new Error('visualizer-spike: #cesiumContainer / #stats not found');
  }

  const viewer = new Viewer(container);
  await viewer.initialize();

  // ── Base imagery: AMap (高德) satellite XYZ tiles ──────────────────────────
  viewer.globe.layers.addImageryProvider(new XyzImageryProvider({
    url:          'https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
    subdomains:   ['1', '2', '3', '4'],
    maximumLevel: 18,
    tileSize:     256,
  }));

  viewer.camera.setView({
    destination: Cartesian3.fromDegrees(105, 35, 6_000_000),
  });

  // ── The system under test: scene.entities ──────────────────────────────────
  const entities = viewer.scene.entities;

  for (const city of CITIES) {
    entities.add(new Entity({
      name:     city.name,
      position: lonLat(city.lon, city.lat),
      point:    { color: colorOf(city.css), pixelSize: city.size },
      label: city.label
        ? { text: city.label, fillColor: colorOf('#ffffff') }
        : undefined,
    }));
  }

  entities.add(new Entity({
    name:     '京广线',
    polyline: {
      positions: JINGGUANG.map(([lon, lat]) => lonLat(lon, lat, LINE_LIFT_METRES)),
      width:     3,
      color:     colorOf('#ff4040'),
    },
  }));

  entities.add(new Entity({
    name:     '海上丝路',
    polyline: {
      positions: MARITIME.map(([lon, lat]) => lonLat(lon, lat, LINE_LIFT_METRES)),
      width:     4,
      color:     colorOf('#4080ff'),
    },
  }));

  // Opaque polygon → G-Buffer (lit).
  entities.add(new Entity({
    name: '渤海(不透明)',
    polygon: {
      hierarchy: {
        positions: [
          lonLat(117.8, 39.0), lonLat(121.0, 40.5), lonLat(122.2, 39.0),
          lonLat(120.5, 37.6), lonLat(118.0, 38.1),
        ],
      },
      height: 2000 / Cartesian3.EARTH_RADIUS,
      color:  colorOf('#3070ff', 1),
    },
  }));

  // Translucent polygon → WBOIT composite.
  entities.add(new Entity({
    name: '塔克拉玛干(半透明)',
    polygon: {
      hierarchy: {
        positions: [
          lonLat(77.0, 41.5), lonLat(88.0, 42.5),
          lonLat(90.0, 39.0), lonLat(80.0, 36.5),
        ],
      },
      height: 3000 / Cartesian3.EARTH_RADIUS,
      color:  colorOf('#ffcc00', 0.35),
    },
  }));

  // ── Debug handle + toolbar ─────────────────────────────────────────────────
  (window as unknown as {
    __visualizerSpike: { viewer: Viewer; entityCount: number };
  }).__visualizerSpike = { viewer, entityCount: entities.length };

  const views: Record<string, { lon: number; lat: number; alt: number }> = {
    china:  { lon: 105,    lat: 35,   alt: 6_000_000 },
    north:  { lon: 116.39, lat: 39.9, alt: 2_000_000 },
    street: { lon: 116.39, lat: 39.9, alt: 100_000 },
    back:   { lon: 20,     lat: 0,    alt: 25_000_000 },
  };
  document.querySelectorAll<HTMLButtonElement>('#toolbar button[data-view]').forEach(button => {
    button.addEventListener('click', () => {
      const v = views[button.dataset.view ?? ''];
      if (!v) return;
      viewer.camera.flyTo({
        destination: Cartesian3.fromDegrees(v.lon, v.lat, v.alt),
      });
    });
  });

  // ── Stats ──────────────────────────────────────────────────────────────────
  viewer.preRender = () => {
    statsEl.textContent =
      `WebGPU 实体桥接 (T2.2)\n` +
      `实体:     ${entities.length}\n` +
      `几何图元: ${viewer.scene.primitives.length}\n` +
      `检查: 点恒定像素大小 / 背面遮挡 / 半透明多边形`;
  };
}

main().catch((error: unknown) => {
  console.error(error);
  const el = document.getElementById('error');
  if (el) {
    el.style.display = 'block';
    el.textContent = `visualizer-spike failed: ${error instanceof Error ? error.message : String(error)}`;
  }
});
