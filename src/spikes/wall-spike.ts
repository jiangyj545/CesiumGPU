import { Viewer } from '../core/Viewer';
import { Camera } from '../scene/Camera';
import { Cartesian3 } from '../math/Cartesian3';
import { XyzImageryProvider } from '../imagery/XyzImageryProvider';
import { DynamicWall } from '../effect/DynamicWall';

/**
 * Spike: 动态墙特效 DynamicWall（T3.2，对齐 EV_DynamicWall）。
 *
 * 验证：
 * - 墙体贴地竖立（底排贴地、顶排沿地表法线抬高）
 * - 横向光带从底往上涌（speed 控制流速）
 * - 顶部渐隐（竖直包络 fade）
 * - 地球遮挡：旋转到地球远侧的墙段不可见
 * - show 开关 + EffectManager 挂载
 *
 * 打开方式：dev server 下访问 /spikes/wall-spike.html（需 WebGPU）。
 */

/** 米制 ECEF（Cartesian3.fromDegrees）→ 归一化 ECEF（Earth radius = 1）。 */
function lonLat(lon: number, lat: number, heightMetres = 0): Cartesian3 {
  const p = Cartesian3.fromDegrees(lon, lat, heightMetres);
  const s = Camera.EARTH_SCALE;
  return new Cartesian3(p.x / s, p.y / s, p.z / s);
}

/** 北京周边一圈不规则闭合路径（经纬度，首尾相接成环）。 */
const BEIJING_RING: Array<[number, number]> = [
  [116.0, 40.25],
  [116.95, 40.5],
  [117.9, 40.15],
  [118.35, 39.4],
  [117.6, 38.75],
  [116.5, 38.55],
  [115.45, 38.9],
  [115.05, 39.75],
  [115.7, 40.3],
  [116.0, 40.25], // 闭合
];

/** 折线形墙：模拟一条边界线（不闭合）。 */
const BORDER_LINE: Array<[number, number]> = [
  [110.0, 41.4],
  [111.4, 40.9],
  [112.8, 41.6],
  [114.3, 41.1],
  [115.8, 41.8],
  [117.4, 41.2],
  [119.0, 41.9],
];

async function main(): Promise<void> {
  const container = document.getElementById('cesiumContainer');
  const statsEl = document.getElementById('stats');
  const btnRing = document.getElementById('btn-ring');
  const btnBorder = document.getElementById('btn-border');
  if (!container || !statsEl || !btnRing || !btnBorder) {
    throw new Error('wall-spike: #cesiumContainer / #stats / buttons not found');
  }

  const viewer = new Viewer(container);
  await viewer.initialize();

  // ── Base imagery: AMap (高德) satellite XYZ tiles（同 visualizer-spike）──
  viewer.globe.layers.addImageryProvider(new XyzImageryProvider({
    url:          'https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
    subdomains:   ['1', '2', '3', '4'],
    maximumLevel: 18,
    tileSize:     256,
  }));

  // ── The system under test: DynamicWall ────────────────────────────────────
  const ringWall = viewer.scene.effects.add(new DynamicWall(viewer.context, {
    positions: BEIJING_RING.map(([lon, lat]) => lonLat(lon, lat)),
    height: 0.02,                       // ≈ 127 km
    color: [0.25, 0.75, 1.0, 0.9],      // 青蓝
    speed: 0.5,
    repeat: 3,
    camera: viewer.scene.camera,
  }));

  const borderWall = viewer.scene.effects.add(new DynamicWall(viewer.context, {
    positions: BORDER_LINE.map(([lon, lat]) => lonLat(lon, lat)),
    height: 0.012,                      // ≈ 76 km
    color: [1.0, 0.45, 0.15, 0.85],     // 橙红
    speed: 0.8,
    repeat: 2,
    camera: viewer.scene.camera,
  }));

  // ── Camera: 俯视北京周边 ──────────────────────────────────────────────────
  viewer.camera.setView({
    destination: Cartesian3.fromDegrees(117.0, 37.5, 2_000_000),
  });

  // ── HUD: show 开关 + 信息 ─────────────────────────────────────────────────
  let ringOn = true;
  let borderOn = true;
  const refresh = (): void => {
    statsEl.textContent =
      'DynamicWall spike (T3.2)\n' +
      `环形墙: ${ringOn ? '开' : '关'}  (10 段, h=0.02≈127km, speed=0.5, repeat=3)\n` +
      `折线墙: ${borderOn ? '开' : '关'}  (6 段, h=0.012≈76km, speed=0.8, repeat=2)\n` +
      '拖动旋转 / 滚轮缩放 — 转到地球背面侧墙段应被遮挡';
  };
  refresh();

  btnRing.addEventListener('click', () => {
    ringOn = !ringOn;
    ringWall.show = ringOn;
    btnRing.textContent = `环形墙: ${ringOn ? '开' : '关'}`;
    refresh();
  });
  btnBorder.addEventListener('click', () => {
    borderOn = !borderOn;
    borderWall.show = borderOn;
    btnBorder.textContent = `折线墙: ${borderOn ? '开' : '关'}`;
    refresh();
  });

  // 调试句柄（同 tile-lod-spike 的 __tileLodSpike 模式），供 QA 脚本驱动相机。
  (window as unknown as Record<string, unknown>).__wallSpike = {
    viewer,
    ringWall,
    borderWall,
  };
}

main().catch((err) => {
  console.error(err);
  const statsEl = document.getElementById('stats');
  if (statsEl) statsEl.textContent = `Spike 失败: ${String(err)}`;
});
