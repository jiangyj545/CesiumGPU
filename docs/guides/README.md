# CesiumGPU 指南（Guides）

按推荐阅读顺序排列。全部示例代码与 `src/` 源码签名一致；坐标约定统一为「归一化 ECEF（地球半径 = 1）」，详见[引擎架构](./architecture.md)第 4 节。

## 目录

| # | 指南 | 一句话简介 |
|---|---|---|
| 1 | [快速开始（Quick Start）](./quick-start.md) | 从零到可拖拽地球：Viewer 生命周期、相机、底图、实体、拾取的最小闭环 |
| 2 | [影像瓦片体系（Imagery）](./imagery.md) | `ImageryProvider` 抽象、XYZ 数据源、图层栈与 Web Mercator 瓦片坐标 |
| 3 | [Entity 数据模型（Entities）](./entities.md) | 声明式实体（折线/多边形/点/标注）、`EntityCollection` 与 GeoJSON 加载 |
| 4 | [交互三件套（Interaction）](./interaction.md) | `ScreenSpaceEventHandler` 事件全表、`DrawHandler` 球面绘制、GPU 拾取 |
| 5 | [空间量测（Analysis）](./analysis.md) | `Measure` 距离/面积/高度纯计算、`MeasureCollection` 结果管理、绘制+量测组合 |
| 6 | [相机与飞行（Camera）](./camera.md) | heading/pitch 语义、`setView`/`flyTo`、防穿地飞行轨迹、Spline 路径漫游 |
| 7 | [特效系统（Effects）](./effects.md) | `IPass`/`EffectManager`、动态墙/扫描/粒子/天气、自定义特效 pass 与 unlit 管线 |
| 8 | [引擎架构（Architecture）](./architecture.md) | 4-pass 延迟渲染管线、目录结构、归一化 ECEF 坐标系约定、ADR 索引 |
| 9 | [地形与贴地（Terrain）](./terrain.md) | `TerrainProvider` 双格式（quantized-mesh/Terrarium）、`sampleHeight` 采样、拾取/量测/几何贴地 |

## 其他文档

- **API 参考** — [docs/api/index.html](../api/index.html)（由 `npm run docs:api` 从 `src/index.ts` 生成）
- **设计决策（ADR）** — [docs/decisions/](../decisions/)：瓦片纹理池（0001）、GPU 拾取回读（0002）、compute 粒子（0003）、地形 Provider 双格式（0004）、影像破洞系统性修复（0005）

## 示例

`public/examples/` 提供可运行页面（`npm run dev` 后访问 `http://localhost:3001/examples/`）：

| 页面 | 对应指南 |
|---|---|
| [welcome.html](http://localhost:3001/examples/welcome.html) | 引擎能力总览 |
| [01_imagery.html](http://localhost:3001/examples/01_imagery.html) | 影像瓦片体系 |
| [02_camera.html](http://localhost:3001/examples/02_camera.html) | 相机与飞行 |
| [03_geojson.html](http://localhost:3001/examples/03_geojson.html) | Entity / GeoJSON |
| [04_geometry.html](http://localhost:3001/examples/04_geometry.html) | 几何与图元 |
| [05_draw.html](http://localhost:3001/examples/05_draw.html) | 绘制与量测 |
| [14_terrain.html](http://localhost:3001/examples/14_terrain.html) | 地形与贴地 |
