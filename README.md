# CesiumGPU

一款仿 Cesium API 的 **WebGPU** 三维地球渲染引擎。

## 功能矩阵

| 能力 | 状态 | 说明 / 入口 |
|---|---|---|
| 影像瓦片 | ✅ | XYZ Provider + 图层栈 + LRU 纹理池，见 [Imagery 指南](docs/guides/imagery.md) |
| 矢量（Entity） | ✅ | 折线 / 多边形 / 点 / 标注，见 [Entities 指南](docs/guides/entities.md) |
| GeoJSON | ✅ | `GeoJsonDataSource`，见 [Entities 指南](docs/guides/entities.md) |
| 绘制 | ✅ | 点 / 线 / 面 / 矩形球面绘制，见 [Interaction 指南](docs/guides/interaction.md) |
| 拾取 | ✅ | GPU 1×1 回读（24-bit id + 地表坐标），见 [Interaction 指南](docs/guides/interaction.md) |
| 量测 | ✅ | 距离 / 面积 / 高度（椭球测地线 / Girard 定理），见 [Analysis 指南](docs/guides/analysis.md) |
| flyTo | ✅ | 大圆弧 + 抛物线拱起 + 防穿地，见 [Camera 指南](docs/guides/camera.md) |
| lookAt / zoomTo / trackedEntity | ✅ | HeadingPitchRange 定点视角、包络球飞行、跟机三模式（自由/定位/跟随），见 [Camera 指南](docs/guides/camera.md) |
| 动态 Entity | ✅ | CallbackProperty 逐帧取值 + model（glTF 每帧重定位）+ path 轨迹线，见 [Entities 指南](docs/guides/entities.md) |
| SceneTransforms | ✅ | worldToWindowCoordinates / wgs84ToWindowCoordinates（CSS 像素，DPR 感知），见 [Interaction 指南](docs/guides/interaction.md) |
| 3DTiles feature 挑拣 | ✅ | b3dm `_BATCHID` → `PickResult.feature`（`Cesium3DTileFeature`·getProperty/batchTable），见 [Interaction 指南](docs/guides/interaction.md) |
| 认证 headers | ✅ | 影像 / 地形 / 3DTiles provider 与 tileset.json 全链路 `headers` 透传，见 [Terrain 指南](docs/guides/terrain.md) |
| 地形采样 | ✅ | `sampleTerrainMostDetailed` / `sampleTerrain`（包络预载 + 高度回填），见 [Terrain 指南](docs/guides/terrain.md) |
| Lib 构建 | ✅ | `npm run build:lib` → dist-lib/（ESM + UMD `CesiumGPU` global + d.ts） |
| Entity cylinder | ✅ | 圆柱/圆锥 graphics（动态 length/radius，HPR 朝向），见 [Entities 指南](docs/guides/entities.md) |
| 标注细节 | ✅ | LabelStyle 描边 / H·V 锚点 / DistanceDisplayCondition / disableDepthTestDistance / HeightReference，见 [Entities 指南](docs/guides/entities.md) |
| 按需渲染 | ✅ | `requestRenderMode` + `scene.requestRender()`（RenderScheduler 脏标记），见 [Interaction 指南](docs/guides/interaction.md) |
| FlowArrow 流动箭头线 | ✅ | polyline.material type:'flow'（弧长相位滚动 + 动态 positions 重建），见 [Entities 指南](docs/guides/entities.md) |
| 雾 / 闪电 | ✅ | FogEffect（距离+高度雾+染色）/ LightningEffect（strike/包络/随机序列），见 [Effects 指南](docs/guides/effects.md) |
| LERC 地形 | ✅ | ArcGIS Terrain3D LERC1 解码 + provider（headers/maximumLevel），见 [Terrain 指南](docs/guides/terrain.md) |
| pnts 点云瓦片 | ✅ | 3DTiles pnts（非压缩/量化/Draco 点压缩 + RTC_CENTER），magic 嗅探分派（.e3m 后缀放行），见 [3DTiles 指南](docs/guides/3dtiles.md) |
| 动态墙 | ✅ | `DynamicWall` 光带上涌墙体，见 [Effects 指南](docs/guides/effects.md) |
| 扫描 | ✅ | `CircleScan` 贴地圆形扩散波前，见 [Effects 指南](docs/guides/effects.md) |
| 粒子 | ✅ | `ParticleSystem` compute shader（火焰/喷泉/烟花/烟雾），见 [Effects 指南](docs/guides/effects.md) |
| 天气 | ✅ | `Weather` 相机锚定雨/雪体积，见 [Effects 指南](docs/guides/effects.md) |

另有：延迟光照、WBOIT 透明渲染、glTF/GLB 模型加载、屏幕空间标注（Overlay）。

## 快速开始

```bash
npm install
npm run dev        # http://localhost:3000
```

```typescript
import { Viewer, Cartesian3, Color, Entity } from './src/index';

// 1. 创建 Viewer 并初始化 WebGPU（必须 await）
const viewer = new Viewer('cesiumContainer');
await viewer.initialize();

// 2. 相机飞到北京上空（destination 接受米制 ECEF 或 { lon, lat, height }）
viewer.camera.flyTo({
  destination: Cartesian3.fromDegrees(116.4, 39.9, 20_000_000), // 经度, 纬度, 高度(米)
  duration: 3,
});

// 3. 加载整球底图（失败时退化为程序化纹理）
viewer.globe.loadImageryFromUrl('/images/world.jpg').catch(() => {
  viewer.globe.generateProceduralTexture();
});

// 4. 添加一条折线实体（positions 为归一化 ECEF，地球半径 = 1）
function lonLat(lon: number, lat: number): Cartesian3 {
  const d = Cartesian3.normalize(Cartesian3.fromDegrees(lon, lat, 0), new Cartesian3());
  return new Cartesian3(d.x, d.y, d.z);
}
viewer.scene.entities.add(new Entity({
  name: '京广线',
  polyline: { positions: [lonLat(116.4, 39.9), lonLat(113.3, 23.1)], width: 3, color: Color.RED },
}));
```

更多片段见 [docs/guides/quick-start.md](docs/guides/quick-start.md)。

## 示例

`npm run dev` 后访问 **<http://localhost:3001/examples/>**：影像、相机飞行、GeoJSON、几何、绘制量测等可运行页面。

## 文档

| 文档 | 说明 |
|---|---|
| [指南（guides）](docs/guides/README.md) | 8 篇主题指南：快速开始 / 影像 / 实体 / 交互 / 量测 / 相机 / 特效 / 架构 |
| [API 参考（api）](docs/api/index.html) | TypeDoc 生成，`npm run docs:api` |
| [设计决策（decisions）](docs/decisions/) | ADR：瓦片纹理池、GPU 拾取回读、compute 粒子 |

## 渲染架构

```
每帧渲染流程（Scene.render()）:

1. G-Buffer Pass（不透明几何体）
   ├── Target 0: Albedo   (rgba8unorm)
   ├── Target 1: Normal   (rgba16float)
   └── Target 2: WorldPos (rgba32float, COPY_SRC 供拾取回读)

2. Lighting Pass（延迟光照 — 全屏三角形）
   ├── 读取 G-Buffer
   ├── Blinn-Phong 漫反射 + 高光
   ├── 大气散射边缘效果
   └── Tone Mapping + Gamma 矫正

2.5 Overlay Pass（点/标注，屏幕空间，深度测试）

3. Transparent Accumulation Pass（WBOIT 累积）
   ├── 加权颜色积累 (rgba16float)
   └── 透明度乘积 (r8unorm)

4. Transparent Composite Pass（透明混合）

5. Effect Passes（动态墙/扫描/粒子/天气，叠加发光，最后画）
```

世界坐标为**归一化 ECEF**（WGS84 赤道半径 = 1.0，1 单位 = 6 378 137 米），详见[架构指南 · 坐标系约定](docs/guides/architecture.md)。

## 目录结构

```
src/
├── index.ts          # 公共 API 统一出口
├── core/             # Viewer / CameraFlightPath / Clock / ScreenSpaceEventHandler
├── scene/            # Scene / Camera / Globe / Primitive / passes(GBuffer/Lighting/
│                     #   Transparent/Overlay/IPass) / EntityVisualizer / Model
├── effect/           # EffectManager / DynamicWall / CircleScan / ParticleSystem / Weather
├── shaders/          # WGSL 着色器（GBuffer/Lighting/Transparent/Overlay/特效/拾取）
├── imagery/          # ImageryProvider / XyzImageryProvider / 图层栈 / 瓦片纹理缓存
├── renderer/         # Context / MipmapGenerator
├── picking/          # PickPass（GPU 拾取）
├── geometry/         # Polygon / Polyline / Rectangle / GeometryPipeline
├── dataSource/       # Entity / EntityCollection / Event / GeoJsonDataSource
├── handler/          # DrawHandler（球面绘制）
├── analysis/         # Measure / MeasureCollection（量测）
├── loader/           # GltfLoader
├── math/             # Cartesian / Matrix4 / Quaternion / Ellipsoid / Spline / …
└── spikes/           # 技术验证页（不进公共 API）
```

## 数学库

| 类 | 描述 |
|---|---|
| `Cartesian2` | 二维向量 |
| `Cartesian3` | 三维向量 / ECEF 坐标 |
| `Cartesian4` | 四维向量 |
| `Matrix4` | 4×4 矩阵（列主序） |
| `Quaternion` | 四元数 |
| `Ellipsoid` | 旋转椭球体（WGS84） |
| `Color` | RGBA 颜色 |
| `CesiumMath` | 数学工具函数 |
| `EasingFunction` | 动画缓动曲线 |
| `HermiteSpline` / `CatmullRomSpline` | 路径插值曲线 |

## 浏览器支持

需要支持 **WebGPU** 的浏览器：

- Chrome 113+
- Edge 113+
- Firefox Nightly（需开启 `dom.webgpu.enabled` flag）

## 技术栈

- **WebGPU** + **WGSL** 着色器
- **TypeScript**
- **Vite** 构建工具
