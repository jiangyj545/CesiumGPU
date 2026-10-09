# 相机与飞行（Camera）

`Camera` 的姿态语义、`setView` / `flyTo`、`CameraFlightPath` 防穿地机制，以及 `Spline` 路径漫游。

## 目录

1. [30 秒上手](#1-30-秒上手)
2. [姿态语义：heading / pitch / roll](#2-姿态语义heading--pitch--roll)
3. [destination：两种写法与单位换算](#3-destination两种写法与单位换算)
4. [setView：立即跳转](#4-setview立即跳转)
5. [flyTo：带动画的飞行](#5-flyto带动画的飞行)
6. [CameraFlightPath：飞行轨迹与防穿地](#6-cameraflightpath飞行轨迹与防穿地)
7. [Spline 路径漫游](#7-spline-路径漫游)
8. [常见问题](#8-常见问题)
9. [lookAt / trackedEntity / zoomTo（P0 跟机闭环）](#9-lookat--trackedentity--zoomto-p0-跟机闭环)

## 相关文档

- [快速开始](./quick-start.md) — 相机最小示例
- [特效系统](./effects.md) — 与相机联动的特效（天气体积跟随相机）
- [引擎架构](./architecture.md) — 归一化坐标系与渲染管线
- [API 参考](../api/index.html) — `Camera` / `CameraFlightPath` / `EasingFunction` 完整签名

---

## 1. 30 秒上手

```typescript
import { Viewer, Cartesian3, EasingFunction } from './src/index';

const viewer = new Viewer('cesiumContainer');
await viewer.initialize();

// 立即跳转（无动画）
viewer.camera.setView({
  destination: { lon: 116.4, lat: 39.9, height: 2_000_000 },
});

// 带动画飞行到上海（默认 3 秒）
viewer.camera.flyTo({
  destination: Cartesian3.fromDegrees(121.47, 31.23, 1_500_000),
  duration: 4,
  complete: () => console.log('到达'),
  cancel:   () => console.log('被用户输入打断'),
});
```

相机默认位置在赤道上空 3 倍地球半径处（`initialize()` 内设置 `(0, 0, 3.0)`），朝向地心。

## 2. 姿态语义：heading / pitch / roll

`Orientation`（`src/scene/Camera.ts`）三个字段均为**弧度**：

```typescript
export interface Orientation {
  /** Yaw around the local up axis. 0 = north, positive = clockwise (toward east). */
  heading?: number;
  /** Pitch. 0 = look at the horizon, negative = look down, -π/2 = nadir. */
  pitch?: number;
  /** Roll around the look direction. */
  roll?: number;
}
```

| 字段 | 语义 | 默认 |
|---|---|---|
| `heading` | 绕本地 up 轴偏航；0 = 正北，正值顺时针（向东转） | 0 |
| `pitch` | 俯仰；0 = 看地平线，负值向下看，`-π/2` = 垂直向下看地心（nadir） | `-π/2` |
| `roll` | 绕视线方向滚转 | 0 |

约束与行为：

- `pitch` 会被 clamp 到 `[-π/2, π/2]`；
- 拖拽产生的俯仰有**约 85° 的最大倾角限制**（相机视线与 nadir 方向的点积低于阈值时拒绝继续压低），防止视角翻越地心；
- 省略 `orientation` 时相机垂直向下看向地心。

相机坐标系为右手系，地心在原点，Z 轴指向北极。

## 3. destination：两种写法与单位换算

`Destination` 类型（`src/scene/Camera.ts`）：

```typescript
export type Destination = Cartesian3 | { lon: number; lat: number; height?: number };
```

| 写法 | 单位 | 示例 |
|---|---|---|
| `Cartesian3` | **米制** ECEF（`Cartesian3.fromDegrees` 的输出，模长 ≈ 6.37e6） | `Cartesian3.fromDegrees(116.4, 39.9, 2_000_000)` |
| `{ lon, lat, height }` | 经纬度**度** + 高度**米** | `{ lon: 116.4, lat: 39.9, height: 2_000_000 }` |

两种写法内部都会按 `Camera.EARTH_SCALE`（6 378 137 米）换算成归一化 ECEF（地球半径 = 1）。**注意与量测 API（`Measure`）相反**：量测输入要求归一化坐标，相机 destination 接受米制/字面量，无需手动换算。

## 4. setView：立即跳转

```typescript
export interface SetViewOptions {
  destination?: Destination;
  orientation?: Orientation;
}

viewer.camera.setView({
  destination: { lon: 116.4, lat: 39.9, height: 2_000_000 },
  orientation: { heading: 0, pitch: -Math.PI / 4, roll: 0 },  // 45° 俯视
});
```

`setView` 同步修改 `camera.position` / `direction` / `up`，本帧生效，无动画、无回调。若需强制打断进行中的飞行，先调用 `cancelFlight()`。

## 5. flyTo：带动画的飞行

`FlyToOptions` 完整定义（`src/scene/Camera.ts`）：

```typescript
export interface FlyToOptions extends SetViewOptions {
  /** 飞行时长（秒）。默认 3。 */
  duration?: number;
  /** 缓动曲线。默认 `EasingFunction.QUINTIC_IN_OUT`。 */
  easingFunction?: EasingFunction.Callback;
  /** 正常到达时回调。 */
  complete?: () => void;
  /** 被打断（用户拖拽/滚轮或新的 flyTo）时回调。 */
  cancel?: () => void;
}
```

```typescript
// 自定义缓动：匀速直线感
viewer.camera.flyTo({
  destination: { lon: 121.47, lat: 31.23, height: 1_500_000 },
  duration: 5,
  easingFunction: EasingFunction.LINEAR_NONE,
  complete: () => viewer.camera.flyTo({ destination: { lon: 113.3, lat: 23.1, height: 800_000 } }),
});

// 状态查询与主动中止
console.log(viewer.camera.inFlight);  // 飞行中为 true
viewer.camera.cancelFlight();         // 立即中止并触发 cancel 回调
```

取消规则：

- **任何用户输入**（左键拖拽、滚轮、双指捏合）都会取消进行中的飞行并触发 `cancel`；
- 发起新的 `flyTo` 会先 `cancelFlight()` 再开始新飞行；
- `flyTo` 的路径由 `CameraFlightPath` 生成（见下节），飞行推进由 Viewer 渲染循环每帧驱动。

`EasingFunction`（`src/math/EasingFunction.ts`）可用曲线：

| 常量 | 曲线 |
|---|---|
| `LINEAR_NONE` | 线性 |
| `QUADRATIC_IN_OUT` | 二次缓入缓出 |
| `CUBIC_IN_OUT` | 三次缓入缓出 |
| `QUINTIC_IN_OUT` | 五次缓入缓出（`flyTo` 默认） |
| `SINUSOIDAL_IN_OUT` | 正弦缓入缓出 |

均为 `EasingFunction.Callback = (t: number) => number`（输入输出均在 [0, 1]）。

### 相机状态与矩阵

`Camera` 的公开状态字段与矩阵 getter（`src/scene/Camera.ts`）：

```typescript
class Camera {
  static readonly EARTH_SCALE: number;      // 6378137.0（1 内部单位的米数）
  position: Cartesian3;                     // 世界空间位置（归一化 ECEF）
  direction: Cartesian3;                    // 视线方向（世界空间）
  right: Cartesian3;                        // 右方向
  up: Cartesian3;                           // 上方向

  get viewMatrix(): Matrix4;                // 视图矩阵（惰性重建）
  get projectionMatrix(): Matrix4;          // 投影矩阵
  get viewProjectionMatrix(): Matrix4;      // view × projection

  setAspectRatio(aspect: number): void;     // 画布缩放时由 Viewer 自动调用
}
```

直接改 `position` 后方向基不会自动更新（保持原朝向）。想让相机在当前位置重新垂直看向地心，用 `setView({ destination: camera.position })` 且**省略 orientation**——此时走 nadir 重建分支（内部 `_rebuildFrame()`）。只读矩阵 getter 惰性求值，可每帧安全访问。

## 6. CameraFlightPath：飞行轨迹与防穿地

`camera.flyTo` 内部就是构造一个 `CameraFlightPath`（`src/core/CameraFlightPath.ts`）。它也可以单独实例化：

```typescript
const flight = new CameraFlightPath(viewer.camera, {
  destination: { lon: 121.47, lat: 31.23, height: 1_500_000 },
  duration: 4,
});
// flight.finished — 是否到达
// flight.update(camera, dtSeconds) — 推进一帧并把插值结果写入 camera
```

`flyTo` 与取消联动（`Camera` 内部实现摘录）：

```typescript
flyTo(options: FlyToOptions): void {
  this.cancelFlight();                    // 新飞行打断旧飞行，触发旧 cancel
  this._flight = new CameraFlightPath(this, options);
}

cancelFlight(): void {
  const flight = this._flight;
  if (!flight) return;
  this._flight = null;
  flight.cancelCallback?.();              // 用户输入 → cancel 回调
}
```

### 轨迹生成算法

**水平路径**（大圆弧）：起终点单位向量之间做三次 Hermite 插值——切向量取大圆行进方向、长度按弧角缩放（等价于等角速度 slerp 的导数），每帧求值后重新归一化投影回球面。

**高度路径**（抛物线拱起）：

```
h(t) = lerp(h0, h1, t) + peak · sin(π·t)
peak = max(0, arcAngle · R · 0.25 − (h0 + h1) / 2)
```

`peak` 随起终点弧角增大而抬高——**飞得越远，中途拱得越高**，模拟真实飞行器的绕行轨迹。

**防穿地下限**：

```
h(t) = max(h(t), MIN_ALTITUDE · sin(π·t))
MIN_ALTITUDE = 0.005    // 地球半径的 0.5% ≈ 32 km
```

下限乘 `sin(π·t)` 在两端衰减为 0，因此**中段永不低于地表 ~32 km，而起点/终点高度仍然精确落在请求值上**。

**姿态插值**：

- `heading` 沿最短角路径插值（按 2π 取 wrap，避免绕远路）；
- `pitch` 线性插值，clamp 到 `[-π/2, π/2]`；起点姿态由相机当前视线/上方向反推，终点取 `options.orientation`（省略时保持起点 pitch、roll 归零）。

## 7. Spline 路径漫游

需要自定义巡游路线（多点、折返、盘旋）时，用 `HermiteSpline` / `CatmullRomSpline`（`src/math/Spline.ts`）+ `viewer.preRender` 钩子驱动相机：

```typescript
import { CatmullRomSpline, Cartesian3 } from './src/index';

// 关键帧：归一化 ECEF 顶点 + 时间（秒）
const spline = new CatmullRomSpline({
  times: [0, 4, 8, 12],
  points: [
    new Cartesian3(-0.728, 0.227, 0.648),   // 北京上空（地表半径 1.02 倍）
    new Cartesian3(-0.700, 0.180, 0.692),
    new Cartesian3(-0.660, 0.150, 0.735),
    new Cartesian3(-0.627, 0.211, 0.750),   // 上海上空
  ],
});

let elapsed = 0;
let last = performance.now();
viewer.preRender = () => {
  if (viewer.camera.inFlight) return;      // flyTo 期间让位
  const now = performance.now();
  elapsed = (elapsed + (now - last) / 1000) % 12;   // 循环巡游
  last = now;
  const p = spline.evaluate(elapsed);
  viewer.camera.setView({
    destination: Cartesian3.multiplyByScalar(p, Cartesian3.EARTH_RADIUS, new Cartesian3()),
    orientation: { pitch: -Math.PI / 4 },
  });
};
```

Spline API 摘要：

```typescript
export class HermiteSpline {
  constructor(options: {
    times: number[];        // 每个控制点的时刻（严格递增）
    points: Cartesian3[];   // 控制点（与 times 等长）
    tangents: Cartesian3[]; // 每个控制点的切向量
  });

  /** 三次 Hermite 插值；time 被 clamp 到 [startTime, stopTime] */
  evaluate(time: number, result?: Cartesian3): Cartesian3;

  /** 由相邻点差分自动计算切向量：(p[i+1] − p[i−1]) / 2（端点单侧） */
  static computeTangents(points: Cartesian3[]): Cartesian3[];
}

export class CatmullRomSpline extends HermiteSpline {
  // 切向量自动 = computeTangents(points)，曲线 C1 连续且过所有控制点
  constructor(options: { times: number[]; points: Cartesian3[] });
}
```

选型建议：

- **平滑巡游、不想手动调切向**：`CatmullRomSpline`（自动切向量、C1 连续）；
- **精确控制每个关节的速度方向**（如贴地飞行、急转弯）：`HermiteSpline` 手动给 `tangents`；
- `evaluate` 超出 `times` 范围会被 clamp 到端点，不会外插。

## 8. 常见问题

**Q: `flyTo` 传了坐标但相机不动？**
确认 `destination` 的写法。`Cartesian3.fromDegrees` 的米制结果和 `{ lon, lat, height }` 字面量都可以（内部换算）；但如果你传入**手动归一化后的** `Cartesian3`（模长 ≈ 1），它会被再除以一次 `EARTH_SCALE` 导致目标点几乎不动。要么用米制/字面量，要么直接操作 `camera.position.set(x, y, z)`（归一化单位）。

**Q: 飞行中途被莫名打断？**
鼠标拖拽 / 滚轮 / 触摸都会取消飞行并触发 `cancel`。程序化飞行期间若要屏蔽用户输入，可临时解绑 handler 输入（见 [interaction.md](./interaction.md)）。

**Q: 飞行会穿过地球吗？**
不会。`CameraFlightPath` 有抛物线拱起 + `MIN_ALTITUDE`（地球半径 0.5% ≈ 32 km）双保险，且下限在两端按 `sin(π·t)` 衰减，起终点高度保持精确。

**Q: 如何实现「飞行到然后旋转环绕」？**
`flyTo` 结束后（`complete` 回调）用 `preRender` 逐帧修改 `heading`。保持 `position` 不变、绕目标点旋转需自行维护轨道参数（半径 + 方位角），每帧 `setView` 更新；`Camera` 的拖拽控制已是环绕模式，程序化环绕可参考其 `_rotateByAxis` 实现思路。

**Q: `evaluate` 的时间戳要求？**
`times` 必须严格递增；`time` 超界被 clamp（不会抛错、不会外插）。

## 9. lookAt / trackedEntity / zoomTo（P0 跟机闭环）

### HeadingPitchRange

```typescript
import { HeadingPitchRange, CesiumMath } from './src/index';
const offset = new HeadingPitchRange(0, CesiumMath.toRadians(-90), 500); // 正下看，500 m
```

`heading` / `pitch` 为弧度（pitch −90° = 相机在目标正上方俯视），`range` 为米。

### Camera.lookAt(target, offset)

`target` 为**米制 ECEF**（`Cartesian3.fromDegrees(...)`），相机被放到目标局部 ENU 框内由 offset 描述的位置并回望目标。pitch −90° = 正上方俯视；pitch 0° = 地平线上水平看向目标。与 Cesium 不同，本引擎 lookAt **不锁定**相机变换——拖拽/缩放照常可用。`camera.lookAt(undefined)` 为 no-op（兼容解锁调用点）。

`Camera.lookAtTransform(Matrix4.IDENTITY)` 是 Cesium 语义的「解锁」兼容点（no-op）；非单位阵会抛错（用 `lookAt` 替代）。

### Viewer.trackedEntity（三模式映射）

hexapod_gcs 的相机三模式（自由 / 定位 / 跟随）映射：

| hexapod 模式 | CesiumGPU 做法 |
|---|---|
| 自由 | `viewer.trackedEntity = undefined`（+ `camera.lookAtTransform(Matrix4.IDENTITY)` 兼容调用，no-op） |
| 定位 | `viewer.zoomTo(entity, new HeadingPitchRange(0, -Math.PI/2, range))` |
| 跟随 | `viewer.trackedEntity = entity` |

跟随实现（`EntityTracker`，纯数学可在 node 测试）：设置时记录当前相机→实体偏移（`headingPitchRangeFromCamera`）；每帧渲染循环把实体位置帧间增量平移到相机（`camera.translate`），视角朝向不动——用户旋转/缩放即调整相对偏移，下一帧从新偏移继续跟随；置 `undefined` 释放。

### Viewer.zoomTo(target, headingPitchRange?) / Camera.flyToBoundingSphere

`zoomTo` 接受 `Entity`（解析其 position，含动态 Property）、米制 `Cartesian3`、归一化 ECEF `Cartesian3[]`（`BoundingSphere.fromPoints` Ritter 包络球）或 `Tileset3D`（根瓦片包围球），默认 heading 0 / pitch −90° / range = max(2.5·radius, 50 m)，经 `Camera.flyToBoundingSphere` 飞行到位（默认 1.5 s，可传 `offset` 覆盖）。

**P1 补充**：`HeadingPitchRoll` 值类（heading/pitch/roll 弧度，`fromRadians` / `clone` / `equals`）已导出，可直接喂给 `Transforms.headingPitchRollQuaternion(origin, hpr)`（对象重载），对齐 hexapod `droneController` 的调用形态。
