# EV-Globe WebGL → CesiumGPU 全量示例迁移报告

**日期**：2026-09-07
**执行**：agent team（8 个迁移 worker + team-lead 协调验证）
**源**：`E:\BaiduNetdiskDownload\webgl压缩包\webgl\EV-Globe_WebGL_1.1\04.Implement\EV-Globe WebGL 1.1\trunk\Examples`（397 个 HTML，去重后约 170 主题）
**目标**：`D:\work\CesiumGPU\public\examples\`（74 个新示例，编号 15-88）

## 最终验证结论

| 验证项 | 结果 |
|---|---|
| 浏览器全量验证（74 页） | **74/74 PASS**（`__DEMO_OK` + 无错误面板 + CDP 截图像素非空，存证 `.omc/tmp/shots-w16b/`） |
| typecheck | 0 错误 |
| 引擎回归测试 | 243/243 全绿 |
| config.js 完整性 | 88 条登记 ↔ 88 文件双向一致，0 死链 0 漏登，14 分类内编号升序 |

## 迁移产出

### 示例分类（74 新 + 14 原有 = 88 页）

| 分类 | 编号 | 覆盖 trunk 系列 |
|---|---|---|
| 影像图层 | 15-17, 60-62 | 01_*（多源/图层组/单图/对比工具） |
| 地形 | 18-19, 63-64 | 02_*（DEM 加载/剖面/高程设色） |
| 矢量数据 | 20-24 | 03_*（KML/CZML/SHP/点聚合/大数据聚合） |
| 几何与标绘 | 25-34 | 04_*（billboard/椭圆/柱/墙/光晕线/虚线/球管/自定义几何/探测/顶点编辑） |
| 三维模型 | 35-40 | 06_*（glTF/动画/小场景/点云近似/剖切/E3M 占位） |
| 相机与视角 | 41-46 | 07_*（环绕/路径漫游/地面漫游/定点旋转/地下/历史书签） |
| 特效 | 47-59, 67-70 | 08_*（水面/海洋/流线/航迹/航班/热力/昼夜/星空/聚光/光柱/bloom/云/体积/建筑） |
| 球体视图 | 65-66 | 08_*（半透明/模式切换） |
| 空间分析 | 76-80, 86 | 10_*（缓冲区/天际线/可视域/淹没/路径剖面/路径分析） |
| 工具 | 71-75 | 09_*（坐标拾取/罗盘/截图/快捷键/场景时间） |
| 物理仿真 | 81-82 | 12_*（落体/弹跳，欧拉积分替代 three.js+ammo） |
| 标绘 | 83-85 | 13_*（3D 雷达/雷达融合/简版军标） |
| 数据可视化 | 87-88 | 14_*/15_*（mapV 迁徙/echarts 飞线，原生 canvas 替代） |

### 引擎增强（src/，迁移过程沉淀）

- `DynamicWater`（296 行 + shader）——水面/海洋特效，对齐 EV_DynamicWaters/EV_Ocean
- `HeatmapLayer`（483 行 + shader，IPass）——贴地热力图，对齐 EV_Heatmap
- `writeCameraUniforms/CAMERA_UBO_SIZE` 公开导出——自定义 IPass 特效的公共依赖
- `DynamicWall.speed` getter——实时流速调节
- **fix `GltfLoader`**：相对 URL 作 baseUrl 时 `new URL()` 抛 Invalid base URL（外部 .bin 引用场景），load() 先规范化绝对 URL

### 数据资产（public/data/，22 项）

gltf（CVN-76/DF26/city/groundvehicle）、model（CesiumAir/Man/MilkTruck/plane/750kV 变压器）、kml、shp、json（buildings/china/flights/gis_point 等）、wind、images、custom_skybox

### 离线服务接入

- **XYZ 影像**：`http://localhost:8890/tiles/Mapnik/{z}/{x}/{y}.png`（stand-in 脚本 `.omc/tmp/maptiles_server.py`，数据 E:/maptiles，z0-16）
- **quantized-mesh 地形**：`http://localhost:8888/api/v1/terrain-3d/serve/terrain_hubei/{z}/{x}/{y}.terrain`（stand-in `.omc/tmp/terrain_server.py`，EPSG:4326 TMS 南起，L0-14，大巴山/江汉平原海拔实测自洽）

## 明确跳过项（及理由）

| trunk 系列 | 理由 |
|---|---|
| 11_* 电力行业（20 个） | 强依赖 BIMDB 电网模型 + EM 私有服务端，无替代数据 |
| 06_loadBIM | 依赖 BIMService 服务端接口 |
| 10_modelAnalysis/spatialAnalysis | 服务端空间计算（前端近似的 76-80 已覆盖核心语义） |
| 13_MilStd* 军标私库系列 | EMMilitaryPloting 私有库不可移植，降级为 85 简版标绘 |
| 12_* 物理大部分 | three.js+ammo 私有集成，降级为 81/82 轻量欧拉积分 |
| 08_videoFusionEditor/putInVideoFix | 已有 13_videoOverlay 覆盖 |
| 05/06/16 的 _copy/_test 重复文件 | 纯重复 |

## 迁移中的关键修复（验证阶段发现）

1. **81-84**：`JulianDate.secondsDifference` 静态调用误用（实为实例方法）→ 修正
2. **37**：J10C/MisYJ88 模型为 KTX2 压缩纹理（引擎不支持 basisu）→ 最终阵容 CVN-76 + DF26×2 + groundvehicle×2 + z9；750kV 变压器模型（中文路径目录）触发 GltfLoader 相对 baseUrl 缺陷 → 换 groundvehicle 并反哺引擎修复（见"引擎增强"）；CVN-76 纹理扩展名与内容不符（jpg 实为 PNG）→ 数据级修正；DF26 引用大写 .JPG vite 不识别（SPA fallback 返回 text/html）→ 改小写引用
3. **47-58**：12 页缺 `__DEMO_OK` 探针（验证误报）→ 补齐
4. **35-40**：容器 id `container` vs demo-utils 预期 `cesiumContainer` → 批量修正（spec 模板缺陷）
5. **19**：sampleHeight 非有限值崩溃 + 剖面图未加载区误画海平面 → 兜底 + 缺口分段

## 已知引擎级限制（示例页面已诚实注明）

- KTX2/basisu 压缩纹理不支持（KHR_texture_basisu）
- TerrainLayer 缓存 64 瓦片上限且无粗层级 pin——sampleHeight 只对已加载 patch 有效（大范围剖面受限，建议后续参考 imagery PIN_LEVEL 策略）
- 无 3D Tiles/E3M/BIM/点云原生加载（以 glTF/点阵近似）
- 相机近平面 ≈32km，贴地视角近处地表被裁剪
- 引擎无 postRender 钩子（只有 preRender）；`Matrix4.values`（无 `.elements`）
