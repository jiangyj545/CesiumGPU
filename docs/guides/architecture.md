# 引擎架构（Architecture）

4-pass 延迟渲染管线全貌、源码目录结构、归一化 ECEF 坐标系约定，以及关键设计决策（ADR）索引。

## 目录

1. [一图看懂：每帧渲染流程](#1-一图看懂每帧渲染流程)
2. [各 Pass 职责](#2-各-pass-职责)
3. [拾取路径（PickPass）](#3-拾取路径pickpass)
4. [归一化 ECEF 坐标系约定](#4-归一化-ecef-坐标系约定)
5. [目录结构](#5-目录结构)
6. [设计决策（ADR）索引](#6-设计决策adr索引)
7. [扩展点](#7-扩展点)
8. [混合与深度状态速查](#8-混合与深度状态速查)
9. [常见问题](#9-常见问题)

## 相关文档

- [快速开始](./quick-start.md) — 引擎最小示例
- [特效系统](./effects.md) — `IPass` / `EffectManager` 与自定义 pass 指南
- [影像瓦片体系](./imagery.md) — 图层栈与瓦片流送（管线第 0 步的供给方）
- [API 参考](../api/index.html) — 完整类型签名

---

## 1. 一图看懂：每帧渲染流程

`Scene.render()` 的实际编排顺序（与 `src/scene/Scene.ts` 中步骤注释一一对应）：

```
                              ┌────────────────────────────────────┐
                              │  Viewer.render loop (每帧)          │
                              │  preRender 钩子 → Scene.render()    │
                              └────────────────┬───────────────────┘
                                               │
   ┌───────────────────────────────────────────┴──────────────────────────────┐
   │  0. 前置更新（每帧）                                                       │
   │     camera.update(dt) 飞行推进 · globe.update(camera) 瓦片流送/上传         │
   │     lightingPass.updateLights · transparentPass.updateCamera              │
   └───────────────────────────────────────────┬──────────────────────────────┘
                                               │  共享 1 个 GPUCommandEncoder
   ┌───────────────────────────────────────────┴──────────────────────────────┐
   │  1. G-Buffer Pass（不透明几何）                    GBufferPass             │
   │     ├─ MRT 0: Albedo    rgba8unorm                                        │
   │     ├─ MRT 1: Normal    rgba16float（世界空间法线）                          │
   │     ├─ MRT 2: WorldPos  rgba32float（世界空间位置，带 COPY_SRC 供拾取回读）    │
   │     └─ Depth:           depth32float（供后续所有 pass 只读测试）              │
   │     内容: Globe(球体+瓦片影像) → opaque 管线 / unlit 管线切换 → 其他图元       │
   └───────────────────────────────────────────┬──────────────────────────────┘
                                               │
   ┌───────────────────────────────────────────┴──────────────────────────────┐
   │  2. Lighting Pass（延迟光照，全屏三角形）           LightingPass            │
   │     读 G-Buffer → Blinn-Phong 漫反射+高光 → 大气散射边缘 →                  │
   │     Tone Mapping + Gamma → 输出到 swap-chain                              │
   └───────────────────────────────────────────┬──────────────────────────────┘
                                               │
   ┌───────────────────────────────────────────┴──────────────────────────────┐
   │  2.5 Overlay Pass（点/标注，屏幕空间）             OverlayPass             │
   │     depth-tested（读 G-Buffer 深度）· loadOp:'load' 画到 swap-chain        │
   │     ※ 刻意排在透明合成之前：让 WBOIT 结果盖过 billboard                      │
   └───────────────────────────────────────────┬──────────────────────────────┘
                                               │
   ┌───────────────────────────────────────────┴──────────────────────────────┐
   │  3. Transparent Accumulation Pass（WBOIT 累积）   TransparentPass          │
   │     ├─ 加权颜色积累  rgba16float                                           │
   │     └─ 透明度乘积    r8unorm          （读 G-Buffer 深度做深度测试）          │
   ├───────────────────────────────────────────────────────────────────────────┤
   │  4. Transparent Composite Pass（透明混合）        TransparentPass          │
   │     与不透明场景混合 → swap-chain                                           │
   └───────────────────────────────────────────┬──────────────────────────────┘
                                               │
   ┌───────────────────────────────────────────┴──────────────────────────────┐
   │  5. Effect Passes（叠加发光，最后画）              EffectManager            │
   │     DynamicWall / CircleScan / ParticleSystem / Weather / 自定义 IPass     │
   │     execute(encoder, swapChainView, time) · show 开关 · additive 混合      │
   └───────────────────────────────────────────┬──────────────────────────────┘
                                               │
                                    canvas.present() → 屏幕
```

补充：拾取（`Scene.pick`）走**独立的 PickPass**（按需懒创建，见第 3 节），不在每帧主管线内。

## 2. 各 Pass 职责

| Pass | 文件 | 职责 | 关键资源 |
|---|---|---|---|
| G-Buffer | `src/scene/passes/GBufferPass.ts` | 收集不透明几何的 albedo / normal / worldPos + 深度 | 3 张 MRT 颜色纹理 + depth32float；`beginRenderPass(encoder)` 由 Scene 驱动 |
| Lighting | `src/scene/passes/LightingPass.ts` | 延迟光照（Blinn-Phong、大气边缘、tonemap/gamma），输出到 swap-chain | 读 G-Buffer 三张纹理；`updateLights(lights)` 每帧同步灯光 |
| Overlay | `src/scene/passes/OverlayPass.ts` | Entity 的点/标注 billboard（屏幕空间），深度测试 | 读 G-Buffer 深度 + 文字图集纹理；已实现 `IPass` |
| Transparent | `src/scene/passes/TransparentPass.ts` | WBOIT 两段式：accumulation + composite | rgba16float 加权色 + r8unorm 透明度乘积；`depthReadOnly` 只读深度 |
| Effects | `src/effect/EffectManager.ts` | 特效统一挂载点，composite 之后逐个 execute | `Effect = IPass & { show: boolean }` |
| Pick | `src/picking/PickPass.ts` | id-encode 渲染 + 1×1 回读，GPU 拾取 | rgba8unorm 拾取纹理 + worldPos 回读 |

管线演进：主管线 4 pass 尚未迁移到 `IPass`（`GBufferPass` / `LightingPass` / `TransparentPass` 标记为 future，深度耦合 `Scene.render()` 编排）；**特效层已全面 `IPass` 化**——新增渲染能力优先实现 `IPass` 挂 `scene.effects`（见 [effects.md](./effects.md) 第 9 节）。

## 3. 拾取路径（PickPass）

拾取不占每帧主管线，`Scene.pick(x, y)` 懒创建 `PickPass` 并挂在 frameHooks 上：

```
scene.pick(x, y)
      │
      ▼
 ┌─────────────────────────────────────────────────┐
 │ PickPass（按需）                                  │
 │  ① id-encode shader：24-bit 图元 id 渲入          │
 │     rgba8unorm 拾取纹理（r/g/b 各 8 位）            │
 │  ② copyTextureToBuffer 回读 1×1 像素               │
 │  ③ worldPos（rgba32float）同像素回读               │
 └─────────────────────────────────────────────────┘
      │
      ▼
 PickResult { id, primitive, worldPosition, timing }
```

- id 编码 `r=id&255, g=(id>>8)&255, b=(id>>16)&255`，id=0 保留给背景；rgba8unorm 量化在整数字节值上**无损**（验证见 ADR-0002）；
- `worldPosition` 用 G-Buffer 的 worldPos（rgba32float，构造时带 `COPY_SRC`）回读，而非深度反投影（depth24plus 无法 1×1 回读）；
- 每帧最多 1 次回读的事件队列模式，避免 `mapAsync` 排队阻塞。

## 4. 归一化 ECEF 坐标系约定

**全引擎只用一套世界坐标**：归一化 ECEF，右手系，地心为原点，Z 轴指向北极：

> **1 个内部单位 = WGS84 赤道半径 = `Camera.EARTH_SCALE` = `Measure.EARTH_SCALE` = 6 378 137 米**
>
> 即地表点模长 ≈ 1.0（极区略扁，b/a = `Measure.POLAR_RATIO` ≈ 0.996647）。

由此衍生的一致性规则：

| API | 接受 / 返回 | 说明 |
|---|---|---|
| `Camera.position`、实体 `position`、图形 `positions` | 归一化 ECEF | 引擎世界空间 |
| `Cartesian3.fromDegrees(lon, lat, h)` | 返回**米制** ECEF | 需归一化后才可作为实体/量测输入 |
| `Camera.setView / flyTo` 的 `destination` | 米制 `Cartesian3` 或 `{ lon, lat, height }` | **内部自动换算**到归一化空间 |
| `Measure.distance / area / height` 输入 | 归一化 ECEF | 返回**米 / m²** |
| `scene.pick().worldPosition`、`DrawHandler` 的 `positions` | 归一化 ECEF | 可直接进量测 |
| 特效 `DynamicWallOptions.positions` | 归一化 ECEF | `height` 也是归一化单位 |
| `ParticleSystem` 的 `origin` | 米制 `Cartesian3` 或 `{ lon, lat, height }` | 内部归一化（normalizeOrigin） |
| `CircleScanOptions.center / radius` | 度 / 米 | 内部投影到 WGS84 椭球 |

归一化帮助函数（与 `GeoJsonDataSource` 同一约定）：

```typescript
function lonLat(lon: number, lat: number, heightMetres = 0): Cartesian3 {
  const dir = Cartesian3.normalize(Cartesian3.fromDegrees(lon, lat, 0), new Cartesian3());
  const scale = 1 + heightMetres / Cartesian3.EARTH_RADIUS;
  return new Cartesian3(dir.x * scale, dir.y * scale, dir.z * scale);
}
```

为什么归一化：WebGPU `float32` 在 6.37e6 量级下只有 ~0.5 m 精度（顶点抖动肉眼可见），归一化到 1.0 量级后世界空间计算全程保持毫米级精度；「米」只出现在 API 边界（量测结果、尺寸参数）。

## 5. 目录结构

```
src/
├── index.ts               # 公共 API 统一出口（仿 Cesium 顶层命名空间）
├── main.ts                # Vite 入口 demo
│
├── core/                  # 应用框架层
│   ├── Viewer.ts          #   容器/canvas/初始化/渲染循环/输入装配
│   ├── CameraFlightPath.ts#   flyTo 轨迹（大圆 + 抛物线 + 防穿地）
│   ├── Clock.ts           #   模拟时钟（JulianDate）
│   └── ScreenSpaceEventHandler.ts  # 鼠标/触摸事件抽象
│
├── scene/                 # 场景与渲染编排
│   ├── Scene.ts           #   ★ 渲染编排：5 步 pass 顺序
│   ├── Camera.ts          #   相机（orbit 控制 / setView / flyTo / 射线）
│   ├── Globe.ts           #   地球球体 + 瓦片影像绘制
│   ├── Primitive.ts       #   图元 + opaque/unlit 双管线构建
│   ├── EntityVisualizer.ts#   Entity → 图元的响应式桥
│   ├── DirectionalLight.ts / SunPosition.ts / EllipsoidGeometry.ts / Model.ts
│   └── passes/            #   GBufferPass / LightingPass / TransparentPass
│                          #   / OverlayPass / IPass（特效抽象）
│
├── effect/                # 特效（全部实现 IPass）
│   ├── EffectManager.ts   #   挂载点（add/remove/execute/destroy）
│   ├── DynamicWall.ts     #   动态墙
│   ├── CircleScan.ts      #   地表圆形扫描
│   ├── ParticleSystem.ts  #   GPU 计算粒子
│   └── Weather.ts         #   雨雪（相机锚定体积）
│
├── shaders/               # WGSL 着色器（字符串常量导出）
│   ├── GBufferShader.ts / LightingShader.ts / TransparentShader.ts
│   ├── OverlayShader.ts / PickShader.ts / SkyboxShader.ts
│   └── DynamicWallShader.ts / CircleScanShader.ts
│       / ParticleCompute.ts / WeatherShader.ts
│
├── imagery/               # 影像瓦片流送
│   ├── ImageryProvider.ts / XyzImageryProvider.ts   # 数据源抽象 + XYZ 实现
│   ├── ImageryLayer.ts / ImageryLayerCollection.ts  # 图层栈
│   ├── TileKey.ts         #   z/x/y 瓦片键（Web Mercator）
│   └── TileTextureCache.ts  # texture-per-tile + LRU 池（ADR-0001）
│
├── renderer/              # WebGPU 基础设施
│   ├── Context.ts         #   device/格式/缓冲上传封装
│   └── MipmapGenerator.ts #   render-based mip 链生成
│
├── picking/
│   └── PickPass.ts        # GPU 拾取（id 回读 + worldPos 回读，ADR-0002）
│
├── geometry/              # 参数几何
│   ├── GeometryPipeline.ts / PolygonGeometry.ts
│   ├── PolylineGeometry.ts / RectangleGeometry.ts
│
├── dataSource/            # 数据模型层
│   ├── Entity.ts / EntityCollection.ts / Event.ts
│   └── GeoJsonDataSource.ts
│
├── handler/
│   └── DrawHandler.ts     # 球面绘制（点/线/面/矩形）
│
├── analysis/
│   └── Measure.ts         # 量测纯计算 + MeasureCollection
│
├── loader/
│   └── GltfLoader.ts      # glTF/GLB 解析
│
├── math/                  # 数学库（仿 Cesium 命名）
│   ├── Cartesian2/3/4.ts / Matrix4.ts / Quaternion.ts
│   ├── Cartographic.ts / Ellipsoid.ts / Color.ts / JulianDate.ts
│   ├── CesiumMath.ts / EasingFunction.ts / Spline.ts / Transforms.ts
│
└── spikes/                # 技术验证页（不进公共 API）
```

依赖方向（自上而下单向）：`core → scene → {renderer, geometry, imagery, effect, picking, dataSource, handler, analysis, loader, math}`；`shaders` 被 scene/effect/picking 引用；`math` 位于最底层被所有层引用。

## 6. 设计决策（ADR）索引

| ADR | 主题 | 一句话结论 |
|---|---|---|
| [ADR-0001](../decisions/0001-tile-texture-pool.md) | 瓦片纹理方案：texture-per-tile + LRU 纹理池 | 每瓦片独立 256×256 纹理（完整 mip 链，render-based 生成）+ LRU 淘汰可行；`uploadsPerFrame` 限流控制帧尖峰 |
| [ADR-0002](../decisions/0002-gpu-picking-readback.md) | GPU 拾取：rgba8unorm 24-bit id + 1×1 回读 | id 编码经 rgba8unorm 往返**无损**；worldPos 用 G-Buffer rgba32float 回读（深度格式不可回读）；每帧最多 1 次回读 |
| [ADR-0003](../decisions/0003-compute-particles.md) | 粒子系统：compute shader GPU 模拟 | 单 storage buffer（32 B/粒子）in-place 更新，无需原子/双缓冲；正式版接入 G-Buffer 深度解决背面透出 |

写新 ADR：涉及「有多个可行方案、且后果不可轻易回退」的技术选型（纹理方案、拾取路径、模拟方式、混合策略等）时，在 `docs/decisions/` 新增 `NNNN-短标题.md`，沿用现有格式（状态 / 日期 / 关联 spike / 关联实现 / 背景 / 决策 / 验证数据）。

## 7. 扩展点

| 想扩展 | 入口 | 文档 |
|---|---|---|
| 新影像数据源 | 实现 `ImageryProvider` 接口（`requestTile` 失败 resolve null，绝不 reject） | [imagery.md](./imagery.md) |
| 新特效 | 实现 `IPass` + 挂 `scene.effects` | [effects.md](./effects.md) 第 9 节 |
| 自发光几何 | `Primitive` + `unlit: true`（G-Buffer 内切 unlit 管线） | [effects.md](./effects.md) 第 8 节 |
| 新绘制类型 | 参考 `DrawHandler`（LEFT_CLICK 加点 / RIGHT_CLICK、双击结束） | [interaction.md](./interaction.md) |
| 新量测类型 | `Measure` 加静态方法 + `MeasureCollection.createMeasurement` 分发 | [analysis.md](./analysis.md) |
| 飞行/漫游定制 | `CameraFlightPath` 或 `HermiteSpline`/`CatmullRomSpline` + `preRender` | [camera.md](./camera.md) 第 7 节 |
| 新 tileset 内容格式 | `TilesetContentLoader` 测试缝替换 `loadTileContent`（magic 嗅探 `sniffTileContentKind` → 解码 → Primitive[]/Effect[]） | [3dtiles.md](./3dtiles.md) |

## 8. 混合与深度状态速查

各 pass / 特效在 swap-chain 与 G-Buffer 深度上的状态一览（新增特效时对照选型）：

| Pass / 特效 | 混合 | G-Buffer 深度 | 时机 |
|---|---|---|---|
| G-Buffer（opaque 管线） | 不透明，写深度 | **写入** depth32float | 步骤 1 |
| G-Buffer（unlit 管线） | 不透明自发光，写深度 | **写入**（同一 MRT） | 步骤 1（管线切换） |
| Lighting | 不透明（全屏覆盖） | 不参与 | 步骤 2 |
| Overlay（点/标注） | 标准 alpha，`loadOp: 'load'` | 只读测试（点/标注被地形遮挡时剔除） | 步骤 2.5（**先于**透明合成，让 WBOIT 盖过 billboard） |
| WBOIT accumulation / composite | 加权累积 / 混合回写 | `depthReadOnly: true`（不可设 load/storeOp） | 步骤 3–4 |
| DynamicWall | **src-alpha / one** 叠加发光 | 无深度附件——片元内射线-单位球求交剔除远侧 | 步骤 5 |
| CircleScan | **(one, one)** 加法，无深度测试 | 无 | 步骤 5 |
| ParticleSystem | 加法发光 | `depthReadOnly` + `depthCompare: 'less'` 不写深度；billboard 取粒子中心点投影 z | 步骤 5 |
| Weather | rain: additive（alpha 0.35）；snow: 标准 alpha（0.8） | `setDepthView` 提供 G-Buffer 深度；未提供则跳过绘制（告警一次） | 步骤 5 |

两条硬约束（WebGPU 规范，均已在真机验证）：

1. `depthReadOnly: true` 的 depth attachment **不可**同时声明 `depthLoadOp` / `depthStoreOp`；
2. 同一 depth texture 在一个 encoder 内「先写后只读」合法（pass 边界有隐式 barrier），`TransparentPass` 与 `ParticleSystem` 都依赖这一点。

## 9. 常见问题

**Q: 为什么世界坐标要归一化？直接用米制 ECEF 不行吗？**
WebGPU `float32` 在 6.37e6 量级下仅有 ~0.5 m 精度，顶点会肉眼可见地抖动；归一化到 1.0 量级后全程毫米级精度。「米」只保留在 API 边界（量测结果、特效尺寸参数）。

**Q: 为什么 `Cartesian3.fromDegrees` 的输出不能直接给实体/量测？**
它返回米制 ECEF（模长 ≈ 6.37e6）。实体/量测/特效 positions 需要归一化（除以 `Cartesian3.EARTH_RADIUS`）；只有 `Camera.setView / flyTo` 的 `destination` 例外——内部自动换算。

**Q: 为什么特效画在最后（步骤 5）而 Overlay 反而在透明之前？**
特效是叠加发光（additive），画最后不被任何 pass 覆盖；Overlay 的点/标注若排在透明之后，会被 WBOIT 的半透明结果错误地盖住/穿透，所以刻意放在合成之前。

**Q: 为什么主管线 pass 还没迁移到 `IPass`？**
`GBufferPass` 是资源持有者（`beginRenderPass` 由 Scene 驱动），`TransparentPass` 是双子 pass 且与 `Scene.render()` 的条件分支耦合，迁移需要渲染编排重构——见 `IPass` 接口文档中的迁移状态表。特效层已全部 `IPass` 化，新能力请走特效路线。

**Q: 新增主管线 pass 的步骤？**
目前需改 `src/scene/Scene.ts` 的 `render()` 编排（插入步骤、申请纹理、处理 resize/destroy）；纯叠加型效果优先实现 `IPass` 挂 `scene.effects`，见 [effects.md](./effects.md) 第 9 节。

**Q: 拾取为什么不直接用深度反投影算世界坐标？**
深度附件是 depth24plus，WebGPU 不支持对它做 1×1 `copyTextureToBuffer` 回读；G-Buffer 的 worldPos 是 rgba32float 且构造时已带 `COPY_SRC`，直接同像素回读即可——完整推导见 [ADR-0002](../decisions/0002-gpu-picking-readback.md)。

## 附：名词表

| 术语 | 含义 |
|---|---|
| G-Buffer | Geometry Buffer，延迟渲染第一步输出的几何信息纹理集（albedo/normal/worldPos/depth） |
| WBOIT | Weighted Blended Order-Independent Transparency，加权混合式半透明排序无关渲染 |
| MRT | Multiple Render Targets，一次几何绘制同时写多张颜色纹理 |
| ECEF | Earth-Centered Earth-Fixed，地心地固坐标系（本引擎归一化，赤道半径 = 1） |
| IPass | 引擎统一的渲染 pass 抽象接口（特效层地基），见 [effects.md](./effects.md) |
| ADR | Architecture Decision Record，架构决策记录，见 `docs/decisions/` |
| swap-chain | WebGPU 画布交换链，`context.getCurrentTextureView()` 即其当前帧颜色目标 |
| readback | GPU→CPU 数据回传（`copyTextureToBuffer` + `mapAsync`），拾取的核心机制 |
