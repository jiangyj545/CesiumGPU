import { describe, expect, it } from 'vitest';
import { Camera } from '../../src/scene/Camera';
import { Cartesian3 } from '../../src/math/Cartesian3';

/**
 * 左键拖动轨道灵敏度（2026-09-29 根因修复的回归闸门）：
 * 旧公式 r·0.0003 rad/px 恒 ≈1913 m/px 与离地高度无关——300m 低空拖 12 步
 * 白膜（500m 城市）完全出画（97 页「一拖动就看不到白膜」）。新贴指公式
 * dθ/px = 2·h·tan(fovY/2)/H 必须随高度缩放：低地图贴合指尖、高空自动加速。
 */

const FOV = (60 * Math.PI) / 180; // 引擎缺省 frustum.fov
const H = 900;                    // 视口像素高

/** 经纬高（度/米）→ 标准 WGS84 ECEF（米）。97 页真实路径：
 *  Cartesian3.fromDegrees 米制 → resolveDestination x/y/z 分支 ×1/EARTH_SCALE 入引擎。 */
const posAtM = (lonDeg: number, latDeg: number, hMeters: number): Cartesian3 => {
  const lon = (lonDeg * Math.PI) / 180, lat = (latDeg * Math.PI) / 180;
  const a = 6378137, e2 = 0.00669437999014;
  const sinLat = Math.sin(lat), cosLat = Math.cos(lat);
  const N = a / Math.sqrt(1 - e2 * sinLat * sinLat);
  return new Cartesian3(
    (N + hMeters) * cosLat * Math.cos(lon),
    (N + hMeters) * cosLat * Math.sin(lon),
    (N * (1 - e2) + hMeters) * sinLat,
  );
};
/** 引擎归一化坐标（orbitSensitivity 的入参约定） */
const posAt = (lonDeg: number, latDeg: number, hMeters: number): Cartesian3 =>
  new Cartesian3(
    posAtM(lonDeg, latDeg, hMeters).x / 6378137,
    posAtM(lonDeg, latDeg, hMeters).y / 6378137,
    posAtM(lonDeg, latDeg, hMeters).z / 6378137,
  );

describe('Camera.orbitSensitivity', () => {
  it('低空灵敏度贴指：300m 高 ≈ 视野高度的 1/H 每像素（≈0.39 m/px），比旧公式小 4 个数量级', () => {
    // dθ = 2·h·tan30°/H；m/px = dθ × 6378137
    const sens = Camera.orbitSensitivity(posAt(114.393, 30.5, 300), FOV, H);
    const mPerPx = sens * 6378137;
    expect(mPerPx).toBeGreaterThan(0.3);
    expect(mPerPx).toBeLessThan(0.5);
    // 旧公式 0.0003 rad/px ≈ 1913 m/px：新值必须小 ≥3 个数量级
    expect(sens).toBeLessThan(1e-6); // 实际 ~6e-8
  });

  it('灵敏度随高度线性缩放：2000m 约为 300m 的 6~7 倍', () => {
    const s300 = Camera.orbitSensitivity(posAt(114.393, 30.5, 300), FOV, H);
    const s2000 = Camera.orbitSensitivity(posAt(114.393, 30.5, 2000), FOV, H);
    const ratio = s2000 / s300;
    expect(ratio).toBeGreaterThan(6.0);
    expect(ratio).toBeLessThan(7.0);
  });

  it('贴地（h→0）不卡死：下限 2e-6 保证正灵敏度', () => {
    const sens = Camera.orbitSensitivity(posAt(114.393, 30.5, 0), FOV, H);
    expect(sens).toBeGreaterThan(0);
    // 下限值本身：2×2e-6×tan30°/900 ≈ 2.56e-9 rad/px（≈16 mm/px）
    expect(sens).toBeLessThan(1e-8);
  });

  it('高空上限（zoom maxR=50）一屏仍可拖过地球视直径', () => {
    // h ≈ 49 归一化（zoom 钳制上限 50），dθ/px = 2·49·tan30°/900 ≈ 6.28e-5
    const sens = Camera.orbitSensitivity(posAt(114.393, 0, 49 * 6378137), FOV, H);
    const fullScreen = sens * H;                    // 拖满一屏的轨道角
    const earthApparent = 2 * Math.asin(1 / 49);    // 地球视直径（弧度）
    expect(fullScreen).toBeGreaterThan(earthApparent);
  });

  it('大地高各纬度一致：500m 高在 -60°~60° 纬度都给出同量级灵敏度（贴指）', () => {
    const sens = [];
    for (const lat of [-60, -30, 0, 30, 60]) {
      const s = Camera.orbitSensitivity(posAt(0, lat, 500), FOV, H);
      expect(s).toBeGreaterThan(0);
      sens.push(s);
    }
    const mPerPx = sens.map((s) => s * 6378137);
    // 500m 高、fov60°、900px：0.64 m/px 附近（±10% 容差盖住椭球局部差异）
    for (const m of mPerPx) {
      expect(m).toBeGreaterThan(0.58);
      expect(m).toBeLessThan(0.71);
    }
  });
});

describe('Camera.rotate 拖动行为（新灵敏度下）', () => {
  it('300m 低空拖 500px：相机水平位移 ~190m——500m 城市白膜仍在画面内', () => {
    const cam = new Camera();
    cam.setView({
      destination: posAtM(114.393, 30.5, 300), // setView 对 x/y/z 自行 ×1/EARTH_SCALE
      orientation: { heading: 0, pitch: -Math.PI / 3, roll: 0 },
    });
    const lon0 = Math.atan2(cam.position.y, cam.position.x);
    const lat0 = Math.asin(cam.position.z / Cartesian3.magnitude(cam.position));
    const r0 = Cartesian3.magnitude(cam.position);

    const sens = Camera.orbitSensitivity(cam.position, FOV, H);
    // 左拖 500px（屏幕位移 → 轨道角，Viewer 同款符号）
    cam.rotate(-500 * sens, 0);

    const lon1 = Math.atan2(cam.position.y, cam.position.x);
    const lat1 = Math.asin(cam.position.z / Cartesian3.magnitude(cam.position));
    const arcM = Math.hypot(lon1 - lon0, lat1 - lat0) * 6378137;
    expect(arcM).toBeGreaterThan(120);
    expect(arcM).toBeLessThan(250); // 旧公式此值 ≈ 956 km
    // 高度不变（轨道保持半径；椭球 |p|<1 属正常——勿用 1+h/a 球面假设）
    expect(Math.abs(Cartesian3.magnitude(cam.position) - r0)).toBeLessThan(1e-12);
    // 纬度未受水平拖动影响
    expect(Math.abs(lat1 - lat0)).toBeLessThan(1e-12);
  });
});
