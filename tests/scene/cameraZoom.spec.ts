import { describe, expect, it } from 'vitest';
import { Camera } from '../../src/scene/Camera';
import { Cartesian3 } from '../../src/math/Cartesian3';
import { Cartographic } from '../../src/math/Cartographic';
import { ExtrudedGeometry } from '../../src/geometry/ExtrudedGeometry';

/**
 * zoom 锚点双约定回归闸门（2026-09-30 根因修复）：
 * _pickSurfaceCentre 旧实现只用压扁系 surfaceRadius 解视线交点。97 页相机走
 * fromDegrees 米制 ÷ EARTH_SCALE（未压扁 WGS84 系），压扁球面比真地表低
 * ~13km——300m/-45° 浅视线的交点在 19.4km 外（真地表 424m），zoom 每档把
 * 相机水平拖走 ~3.9km；叠加页面监控「地心纬度回读→大地纬度 setView」每轮
 * 再丢 0.138°（≈15km），滚轮 8 档相机南移 123km，白膜出画。
 */

const A = Camera.EARTH_SCALE;
const LON = 114.393, LAT = 30.5;   // 97 页城市锚点
const NOTCH = -120 * 0.0022;       // Viewer WHEEL 一格滚轮的 zoom 入参（×0.77）

/** 未压扁 WGS84 ECEF（米）。97 页真实相机路径：fromDegrees 米制直接作
 *  destination，resolveDestination 的 x/y/z 分支 ÷EARTH_SCALE 入引擎（预除
 *  会二次缩放把相机压到原点附近）。 */
const wgsPos = (lonDeg: number, latDeg: number, hMeters: number): Cartesian3 =>
  Cartesian3.fromDegrees(lonDeg, latDeg, hMeters);

/** 相机归一化位置 → 米制 ECEF（groundDistM / Cartographic 入参） */
const toMeters = (p: Cartesian3): Cartesian3 =>
  new Cartesian3(p.x * A, p.y * A, p.z * A);

/** 未压扁 WGS84 离地高度（米），入参归一化位置 */
const wgsAltM = (p: Cartesian3): number =>
  (Cartesian3.magnitude(p) - Camera.wgs84Radius(p)) * A;

/** 两位置地表弧长（米，小角度近似），入参米制 ECEF */
const groundDistM = (m1: Cartesian3, m2: Cartesian3): number => {
  const c1 = Cartographic.fromCartesian(m1);
  const c2 = Cartographic.fromCartesian(m2);
  const dLat = (c2.latitude - c1.latitude) * A;
  const dLon = (c2.longitude - c1.longitude) * A * Math.cos(c1.latitude);
  return Math.hypot(dLat, dLon);
};

const unflattenedCam = (pitchDeg: number, hMeters = 300): Camera => {
  const cam = new Camera();
  cam.setView({
    destination: wgsPos(LON, LAT, hMeters),
    orientation: { heading: 0, pitch: (pitchDeg * Math.PI) / 180, roll: 0 },
  });
  return cam;
};

describe('Camera.zoom 锚点（未压扁 WGS84 系）', () => {
  it('浅视线（-45°）锚在真地表：单档滚轮水平位移 < 300m（旧实现 ~3.9km）', () => {
    const cam = unflattenedCam(-45);
    const before = toMeters(cam.position);
    cam.zoom(NOTCH);
    expect(groundDistM(before, toMeters(cam.position))).toBeLessThan(300);
  });

  it('-60° 俯视同样贴真地表锚点：单档水平位移 < 300m', () => {
    const cam = unflattenedCam(-60);
    const before = toMeters(cam.position);
    cam.zoom(NOTCH);
    expect(groundDistM(before, toMeters(cam.position))).toBeLessThan(300);
  });

  it('连续 8 档 + 页面监控等价回环（大地纬度回读）总漂移 < 1km（旧实现 123km 南移）', () => {
    const cam = unflattenedCam(-60);
    const origin = wgsPos(LON, LAT, 0);
    for (let i = 0; i < 8; i++) {
      cam.zoom(NOTCH);
      // 97 页 200ms 监控等价：低于 300m 抬回当前经纬度 300m/-45°
      // （Cartographic 大地纬度回读——fromCartesian∘fromDegrees 精确互逆）
      if (wgsAltM(cam.position) < 300) {
        const c = Cartographic.fromCartesian(toMeters(cam.position));
        cam.setView({
          destination: Cartesian3.fromDegrees(
            (c.longitude * 180) / Math.PI, (c.latitude * 180) / Math.PI, 300),
          orientation: { heading: 0, pitch: -45 * Math.PI / 180, roll: 0 },
        });
      }
    }
    expect(groundDistM(origin, toMeters(cam.position))).toBeLessThan(1000);
  });

  it('拉近钳制在 SURFACE_CLEARANCE(2m) 地板，任意档数不穿地', () => {
    // 旧断言「10 档后 > 100 m」钉的是已废弃的 127.6 m 地板（0.00002）：
    // Camera.SURFACE_CLEARANCE 2026-09-30 根因修复把它缩到 2 m——旧值把
    // 贴地/水下相机每钳一次抬到百米高空（96 页水下 +11.5 km / 贴水 7 m
    // 机位弹百米实锤，见 Camera.ts 常量注释）。数学上：锚定 zoom 每档
    // ×e^|NOTCH|≈0.768 指数逼近锚点，300 m/-45° 十档后 ≈22 m（实测
    // 21.6 m），要满足 >100 m 只能恢复被废弃的常量——故按现行语义改写，
    // 并把「钳制真实生效」补成硬断言（地板恰为 2 m、极地不穿地）。
    const cam = unflattenedCam(-45);
    for (let i = 0; i < 10; i++) cam.zoom(NOTCH);
    expect(wgsAltM(cam.position)).toBeGreaterThan(0); // 逼近中，未到地板
    // 拉到极限：地板 = wgs84Radius + SURFACE_CLEARANCE。
    for (let i = 0; i < 60; i++) cam.zoom(NOTCH);
    expect(wgsAltM(cam.position)).toBeCloseTo(Camera.SURFACE_CLEARANCE * A, 1);
    expect(wgsAltM(cam.position)).toBeGreaterThanOrEqual(0);

    // 极地（89.5°N）同样换算正确、不穿地。
    const polar = new Camera();
    polar.setView({
      destination: wgsPos(20, 89.5, 300),
      orientation: { heading: 0, pitch: (-60 * Math.PI) / 180, roll: 0 },
    });
    for (let i = 0; i < 60; i++) polar.zoom(NOTCH);
    expect(wgsAltM(polar.position)).toBeCloseTo(Camera.SURFACE_CLEARANCE * A, 1);
  });

  it('朝天（pitch +30°）无锚：退回径向缩放，经纬度不变', () => {
    const cam = unflattenedCam(30);
    const before = toMeters(cam.position);
    cam.zoom(-0.264);
    expect(groundDistM(before, toMeters(cam.position))).toBeLessThan(1);
    expect(Cartesian3.magnitude(cam.position)).toBeLessThan(Cartesian3.magnitude(before));
  });
});

describe('Camera.zoom 锚点（压扁世界系，行为不变闸门）', () => {
  it('压扁系相机锚压扁球面：单档水平位移 < 300m 且不穿压扁地表', () => {
    const cam = new Camera();
    cam.setView({
      destination: { lon: LON, lat: LAT, height: 300 },   // resolveDestination 压扁分支
      orientation: { heading: 0, pitch: -45 * Math.PI / 180, roll: 0 },
    });
    const before = toMeters(cam.position);
    cam.zoom(NOTCH);
    // 水平位移小：若误吃 wgs84 球面（相机在其内部 ~13km），位移会达公里级
    const dLat = (Math.atan2(cam.position.z, Math.hypot(cam.position.x, cam.position.y))
      - Math.atan2(before.z, Math.hypot(before.x, before.y))) * A;
    expect(Math.abs(dLat)).toBeLessThan(300);
    // 不穿压扁地表（+127m 余量钳制）
    expect(Cartesian3.magnitude(cam.position))
      .toBeGreaterThan(Camera.surfaceRadius(cam.position) + 0.00001);
  });
});

describe('Camera 压扁世界相机（flattenedWorld 页面,2026-09-30 白膜贴地迁移）', () => {
  it('zoom 连续拉近钳在压扁地表 +127m,不跳未压扁面 +13km(旧 max() 钳制实锤)', () => {
    const cam = new Camera();
    cam.setView({
      destination: { lon: LON, lat: LAT, height: 5000 },   // 压扁系 5km
      orientation: { heading: 0, pitch: -45 * Math.PI / 180, roll: 0 },
    });
    for (let i = 0; i < 12; i++) cam.zoom(NOTCH);
    const altFlatM = (Cartesian3.magnitude(cam.position) - Camera.surfaceRadius(cam.position)) * A;
    expect(altFlatM).toBeGreaterThan(100);        // 不穿压扁地
    expect(altFlatM).toBeLessThan(1200);          // 旧 max() 会钳到 ~13.2km
  });

  it('orbitSensitivity 压扁系相机按真离地缩放(旧逻辑恒贴 12.7m 下限 → 冻结)', () => {
    const flatCam = { lon: LON, lat: LAT, height: 300 };
    const pFlat = Camera.resolveDestination(flatCam);
    const pWgsM = wgsPos(LON, LAT, 300);
    const pWgs = new Cartesian3(pWgsM.x / A, pWgsM.y / A, pWgsM.z / A);
    const sFlat = Camera.orbitSensitivity(pFlat, Math.PI / 3, 900);
    const sWgs = Camera.orbitSensitivity(pWgs, Math.PI / 3, 900);
    expect(sFlat / sWgs).toBeGreaterThan(0.5);
    expect(sFlat / sWgs).toBeLessThan(2.0);
  });
});

describe('ExtrudedGeometry flattened 放置(白膜贴影像地表)', () => {
  it('flattened 城市底面与 Globe 压扁地表矢长一致(<2m),默认系保持未压扁', async () => {
    const ring = [
      { lon: LON - 0.001, lat: LAT - 0.001 },
      { lon: LON + 0.001, lat: LAT - 0.001 },
      { lon: LON + 0.001, lat: LAT + 0.001 },
      { lon: LON - 0.001, lat: LAT + 0.001 },
    ];
    for (const flattened of [false, true]) {
      const g = ExtrudedGeometry.create({ ring, heightM: 80, flattened });
      // 底环顶点 = 每边四边形的第 1/2 顶点(布局:侧面 4n〔b,b+1=底; b+2,b+3=顶〕+ 顶面 n)
      const n = ring.length;
      let maxErrM = 0;
      for (let i = 0; i < n; i++) {
        const p = new Cartesian3(
          g.vertices[i * 4 * 8], g.vertices[i * 4 * 8 + 1], g.vertices[i * 4 * 8 + 2]);
        // flattened:底面在压扁地表上(|p|−surfaceRadius≈0);
        // 默认:底面在真 WGS84 椭球面上(Cartographic 大地高≈0——wgs84Radius
        // 是地心方向矢长,对大地系放好的点差公里级,勿用)
        const err = flattened
          ? Math.abs((Cartesian3.magnitude(p) - Camera.surfaceRadius(p)) * A)
          : Math.abs(Cartographic.fromCartesian(
              new Cartesian3(p.x * A, p.y * A, p.z * A)).height * A);
        maxErrM = Math.max(maxErrM, err);
      }
      expect(maxErrM).toBeLessThan(2);
    }
  });
});
