# 快速开始（Quick Start）

本指南带你从零搭建一个基于 **CesiumGPU** 的三维地球应用：安装依赖、配置 Vite、初始化 `Viewer`、加载影像、添加实体并接上交互。

> 前置要求：浏览器需支持 **WebGPU**（Chrome / Edge 113+）。CesiumGPU 是纯 WebGPU 引擎，不依赖 WebGL。

---

## 目录

1. [安装](#1-安装)
2. [Vite 配置](#2-vite-配置)
3. [最小可运行示例](#3-最小可运行示例)
4. [Viewer 生命周期](#4-viewer-生命周期)
5. [相机：setView 与 flyTo](#5-相机setview-与-flyto)
6. [加载影像图层](#6-加载影像图层)
7. [添加实体](#7-添加实体)
8. [交互：拾取一个点](#8-交互拾取一个点)
9. [窗口缩放与销毁](#9-窗口缩放与销毁)
10. [常见问题](#10-常见问题)

---

## 1. 安装

CesiumGPU 目前以源码形式（TypeScript）随项目构建，通过 Vite 直接引用 `src/index.ts`：

```bash
git clone <your-repo-url> my-globe-app
cd my-globe-app
npm install
npm run dev
```

生产构建：

```bash
npm run build   # tsc && vite build
```

依赖要求（package.json devDependencies）：

| 依赖 | 用途 |
|---|---|
| `typescript` ^5.4 | 类型系统 |
| `vite` ^5.2 | 构建与开发服务器 |
| `@webgpu/types` ^0.1.69 | WebGPU API 类型定义 |

---

## 2. Vite 配置

引擎的 WGSL 着色器以 `.wgsl` 文件存放于 `src/shaders/`，Vite 需要一个内联插件把它们转成字符串模块（`vite.config.ts` 已内置）：

```typescript
import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  publicDir: 'public',
  build: {
    outDir: 'dist',
    sourcemap: true,
    target: 'esnext',
  },
  server: {
    port: 3000,
    open: true,
  },
  // Inline WGSL shader files as raw strings
  assetsInclude: ['**/*.wgsl'],
  plugins: [
    {
      name: 'wgsl-loader',
      transform(src, id) {
        if (id.endsWith('.wgsl')) {
          return {
            code: `export default ${JSON.stringify(src)};`,
            map: null,
          };
        }
      },
    },
  ],
});
```

HTML 宿主只需要一个容器元素：

```html
<body>
  <div id="cesiumContainer"></div>
  <script type="module" src="./src/main.ts"></script>
</body>
```

---

## 3. 最小可运行示例

所有公共 API 从 `src/index.ts` 统一导出（仿 Cesium 的顶层命名空间）：

```typescript
import { Viewer, Cartesian3 } from './src/index';

// 1. 创建 Viewer（传入容器 id 或 HTMLElement）
const viewer = new Viewer('cesiumContainer');

// 2. 初始化 WebGPU —— 必须等待完成，之后 viewer 才可用
await viewer.initialize();

// 3. 相机飞到北京上空 2 万公里
viewer.camera.flyTo({
  destination: Cartesian3.fromDegrees(116.4, 39.9, 20_000_000), // 经度, 纬度, 高度(米)
});

// 4. 加载一张整球底图（本地图片，Vite 静态目录）
viewer.globe.loadImageryFromUrl('/images/world.jpg').catch(() => {
  // 图片缺失时退化为程序化纹理
  viewer.globe.generateProceduralTexture();
});
```

运行 `npm run dev`，浏览器打开 `http://localhost:3000`，应看到可拖拽旋转、滚轮缩放的地球。

---

## 4. Viewer 生命周期

`Viewer` 是引擎的入口对象，负责容器、Canvas、WebGPU Context 与渲染循环：

```typescript
class Viewer {
  constructor(containerOrId: string | HTMLElement, options?: ViewerOptions);

  /** 初始化 WebGPU 并准备全部渲染资源。必须 await。 */
  async initialize(): Promise<void>;
  /** 停止渲染并释放全部 GPU 资源。 */
  destroy(): void;

  get scene(): Scene;                 // 场景（渲染管线编排）
  get camera(): Camera;               // 等价 scene.camera
  get globe(): Globe;                 // 等价 scene.globe
  get canvas(): HTMLCanvasElement;    // 渲染画布
  get context(): Context;             // WebGPU 封装（创建 buffer/纹理/管线）
  get screenSpaceEventHandler(): ScreenSpaceEventHandler | null;
}

interface ViewerOptions {
  /** GPU 电源偏好。默认 'high-performance'。 */
  powerPreference?: GPUPowerPreference;
  /** 初始背景色（太空）。 */
  backgroundColor?: Color;
}
```

约定：

- `initialize()` **必须** `await`；未初始化前访问场景资源会得到空引用。
- 一个页面可以创建多个 `Viewer`，但每个 Viewer 独占一个 `GPUDevice`，注意显存开销。
- 页面卸载前调用 `viewer.destroy()` 释放 GPU 资源。

---

## 5. 相机：setView 与 flyTo

相机工作在**归一化 ECEF 坐标**（地球赤道半径 = 1.0），`Cartesian3.fromDegrees` 返回**米制** ECEF。相机 `flyTo` / `setView` 的 `destination` 内部做单位换算，也接受 `{ lon, lat, height }` 字面量：

```typescript
// 立即跳转（无动画）
viewer.camera.setView({
  destination: { lon: 116.4, lat: 39.9, height: 2_000_000 },
  // orientation 可选；省略时垂直向下看向地心（nadir）
  orientation: { heading: 0, pitch: -Math.PI / 4, roll: 0 }, // 弧度
});

// 带动画飞行（默认 3 秒，QUINTIC_IN_OUT 缓动）
viewer.camera.flyTo({
  destination: Cartesian3.fromDegrees(121.47, 31.23, 1_500_000), // 上海
  duration: 4,                     // 秒
  complete: () => console.log('到达'),
  cancel:   () => console.log('被用户输入打断'),
});

// 中止当前飞行
viewer.camera.cancelFlight();
```

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

> 鼠标左键拖拽 = 环绕地球旋转，滚轮 = 缩放，任何用户输入都会取消进行中的 `flyTo` 并触发其 `cancel` 回调。

---

## 6. 加载影像图层

引擎支持两类影像来源：

**单图模式**（简单场景/离线演示）：

```typescript
viewer.globe.loadImageryFromUrl('/images/world.jpg'); // 整球贴图
viewer.globe.generateProceduralTexture();             // 程序化纹理兜底
```

**XYZ 瓦片流式模式**（真实业务，按 LOD 按需拉取）：

```typescript
import { XyzImageryProvider } from './src/index';

// 内置 OSM 预设
const osmLayer = viewer.globe.layers.addImageryProvider(XyzImageryProvider.osm());

// 自定义 XYZ 服务（URL 模板支持 {z}/{x}/{y}/{s}）
const amapLayer = viewer.globe.layers.addImageryProvider(
  new XyzImageryProvider({
    url: 'https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
    subdomains: ['1', '2', '3', '4'],
    maximumLevel: 18,
    tileSize: 256,
  })
);

// 图层栈管理（Cesium 风格的 ImageryLayerCollection）
viewer.globe.layers.raiseToTop(amapLayer);  // 置顶（当前渲染取最顶层可见图层）
viewer.globe.layers.remove(osmLayer);       // 移除并释放 GPU 纹理
```

瓦片 LOD、纹理池、并发控制等调优参数（`maximumScreenSpaceError` / `uploadsPerFrame` / `textureCacheCapacity`）详见 [imagery.md](./imagery.md)。

---

## 7. 添加实体

实体（`Entity`）通过 `viewer.scene.entities`（一个 `EntityCollection`）管理。**注意坐标系约定**：实体的 `position` 与图形 `positions` 使用**归一化 ECEF**（地球赤道半径 = 1.0），而 `Cartesian3.fromDegrees` 返回米制 ECEF，需要按半径归一：

```typescript
import { Entity, Cartesian3, Color } from './src/index';

// lon/lat(度) + 高度(米) → 归一化 ECEF（与 GeoJsonDataSource 同一约定）
function lonLat(lon: number, lat: number, heightMetres = 0): Cartesian3 {
  const dir = Cartesian3.normalize(Cartesian3.fromDegrees(lon, lat, 0), new Cartesian3());
  const scale = 1 + heightMetres / Cartesian3.EARTH_RADIUS;
  return new Cartesian3(dir.x * scale, dir.y * scale, dir.z * scale);
}

const entities = viewer.scene.entities;

// 折线（京广线示意）
entities.add(new Entity({
  name: '京广线',
  polyline: {
    positions: [lonLat(116.4, 39.9), lonLat(113.3, 23.1)],
    width: 3,                        // 像素宽
    color: Color.RED,
  },
}));

// 面（带抬升高度，半透明）
entities.add(new Entity({
  polygon: {
    hierarchy: {
      positions: [lonLat(77, 41.5), lonLat(88, 42.5), lonLat(90, 39), lonLat(80, 36.5)],
    },
    height: 3000 / Cartesian3.EARTH_RADIUS,  // 归一化单位（1 = 地球半径）
    color: new Color(1, 0.8, 0, 0.35),
  },
}));
```

多边形/折线走 G-Buffer（不透明）或 WBOIT 透明路径渲染，点/标注走 Overlay 屏幕空间路径——详见 [entities.md](./entities.md)。

---

## 8. 交互：拾取一个点

引擎默认已创建鼠标/触摸处理器（`viewer.screenSpaceEventHandler`），相机拖拽由 Viewer 内部处理。业务交互推荐三件套：

```typescript
import { ScreenSpaceEventHandler } from './src/index';

// 方式 A：复用 Viewer 内置 handler（canvas 已绑定）
const handler = viewer.screenSpaceEventHandler!;

// 方式 B：独立创建
const own = new ScreenSpaceEventHandler(viewer.canvas);

handler.setInputAction((pos) => {
  // pos: ScreenSpacePosition { x, y } — 画布 CSS 像素，左上原点
  console.log('左键点击', pos.x, pos.y);
}, 'LEFT_CLICK');
```

GPU 拾取（返回命中的图元与地表坐标）：

```typescript
const result = await viewer.scene.pick(pos.x, pos.y);
if (result && result.id !== 0) {
  console.log('命中图元 id:', result.id);
  console.log('地表坐标(归一化 ECEF):', result.worldPosition);
  console.log('耗时:', result.timing.totalMs, 'ms');
}
```

绘制（点/线/面/矩形）、`ScreenSpaceEventHandler` 事件类型全表与 `PickPass` 原理，详见 [interaction.md](./interaction.md)。

---

## 9. 窗口缩放与销毁

窗口尺寸变化由 Viewer 内部监听处理（Canvas 自适应容器并重建帧缓冲），业务层无需干预。若要每帧注入逻辑（如更新时钟/太阳方向），使用 `preRender` 钩子：

```typescript
viewer.preRender = () => {
  // 每帧渲染前调用
};
```

页面卸载时释放资源：

```typescript
window.addEventListener('beforeunload', () => viewer.destroy());
```

---

## 10. 常见问题

**Q: 页面白屏，控制台报 `navigator.gpu is undefined`？**
浏览器不支持 WebGPU。使用 Chrome / Edge 113+，或在 `chrome://flags` 开启 WebGPU 实验开关。

**Q: `flyTo` 传了坐标但相机不动？**
确认坐标在引擎世界空间内（归一化 ECEF，模长约 1.0）。`Cartesian3.fromDegrees` 的米制结果（模长 ≈ 6.37e6）用于**度量** API（`Measure`、`distance` 等）；相机/实体位置请归一化（见第 7 节 `lonLat` 帮助函数）。

**Q: 影像瓦片加载很慢？**
检查 `uploadsPerFrame` / `maxConcurrentRequests`（默认各 10），并确认服务端 CORS 允许跨域；`XyzImageryProvider` 请求失败会静默回退到父级瓦片，不会抛错。

**Q: 能否在 Vue / React 中使用？**
可以。`Viewer` 只需要一个 `HTMLElement`：组件 `mounted` 中 `new Viewer(el)` + `await initialize()`，`unmounted` 中 `destroy()` 即可。

---

## 下一步

- [影像瓦片体系](./imagery.md) — LOD 调优与自定义 Provider
- [Entity 数据模型](./entities.md) — GeoJSON 加载与渲染路径
- [交互三件套](./interaction.md) — 事件 / 绘制 / 拾取
- [API 参考](../api/index.html) — typedoc 生成的完整 API 文档
