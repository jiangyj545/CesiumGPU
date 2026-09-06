# 0004 - 地形 Provider 双格式方案（quantized-mesh + Terrarium）

- 状态：已接受（Accepted）
- 日期：2026-09-03
- 关联实现：`src/terrain/TerrainProvider.ts`（接口 + HTTP 基类）、
  `src/terrain/LocalQuantizedMeshProvider.ts`、`src/terrain/TerrariumProvider.ts`、
  `src/terrain/QuantizedMeshDecoder.ts`（小端变体解码器）
- 关联测试：`tests/terrain/`（decoder / provider / terrarium decode 共 3 个 spec）
- 关联验证：`public/examples/14_terrain.html`（湖北地形 + 贴地折线/量测）
- 关联消费端：`Camera.pickGlobe` 贴合、`Measure.clampToTerrain`、
  `PolylineGeometry` / `PolygonGeometry` 的 `heightReference: 'clampToGround'`

## 背景

Wave2 需要为引擎接入真实高程（DEM）瓦片：渲染地形网格、并支撑拾取/量测/
几何的贴地（clamp to ground）。市场上有两类主流地形瓦片格式，需要调研对比后
确定支持的格式与默认数据源：

1. **本地后端服务**：任务环境提供湖北区域的本地地形服务（localhost:8888），
   返回 `.terrain` 二进制瓦片（Cesium quantized-mesh 家族）；
2. **公网免费源**：AWS open data 上的 Mapzen/Tilezen Terrarium PNG 高程瓦片，
   可作离线开发与备选数据源。

## 格式调研结论对比

| 维度 | quantized-mesh（本地服务） | Terrarium PNG |
|---|---|---|
| 数据形态 | gzip 压缩二进制，顶点 u/v/h zigzag-δ 编码 + 索引 | PNG 灰度图，每像素 `(R*256 + G + B/256) − 32768` 米 |
| 解码成本 | 需专用二进制解码器（变体头布局差异大，见下） | `createImageBitmap` + 像素读取（Node 测试环境不可用） |
| 网格质量 | 自带三角化（TIN 连接）+ 边缘裙边（防裂缝） | CPU 按像素三角化，无裙边（接缝可见） |
| 高程精度 | 顶点级（原始 DEM 重采样），含 min/max 高程头 | 像素级（1 px ≈ 1 高程样本） |
| 瓦片方案 | geographic EPSG:4326 TMS（south-up，z0 = 2×1） | Web Mercator slippy XYZ（north-up） |
| 精度坑 | 头部为**小端变体**（见下节）；v 方向 south-up | 无 NoData 值，海洋为负值 |
| 典型来源 | 本地 3D 服务 / Cesium ion | AWS `elevation-tiles-prod`（免费公开） |

结论：两类格式覆盖「内网部署」与「公网兜底」两种场景，均值得支持；
二者的差异（瓦片方案、解码路径、裙边）全部收敛在 Provider 内部，
对上层只暴露统一的 `TerrainProvider` 接口（`requestTileGeometry` →
`TerrainMeshData`，positions 为归一化 ECEF，`1 + h/6378137` 径向缩放）。

## 决策

**双 Provider 并行，统一接口：**

1. **`LocalQuantizedMeshProvider`（首选，本地服务）**
   - geographic TMS 瓦片方案；gzip → 小端变体解码 → `TerrainMeshData`；
   - 解码器按真实瓦片逐字节校验（湖北数据集 z12/x6553/y2707：中心
     lon ≈ 108°、lat ≈ +28.99°，证实 south-up）。
2. **`TerrariumProvider`（备选，公网免费源）**
   - Web Mercator 瓦片方案；PNG → 高度网格 → CPU 三角化；
   - 纯解码函数（`decodeTerrariumPixel`）可在 Node 单测直测。
3. 两者共享 `HttpTerrainProviderBase`：URL 模板、会话内失败缓存
   （失败瓦片不重试）、并发上限 4、协作式 destroy（abort + 排空队列）。

**贴地（Wave2-B 消费端）不感知格式**：`sampleHeight(lonDeg, latDeg) → 米`
由 TerrainLayer 汇聚 Provider 网格后提供；拾取（`Camera.pickGlobe`）、
量测（`Measure.clampToTerrain` / options 钩子）、几何
（`heightReference: 'clampToGround'` + `sampleHeight` 回调）全部只依赖该
回调签名，geometry/analysis 模块不 import terrain，保持分层。

## 小端变体解码器背景

本地服务的 `.terrain` 并非 Cesium 官方大端布局，而是**全小端变体**，
逐字节对照真实瓦片验证出的布局（详见 `QuantizedMeshDecoder.ts` 头注释）：

```
+0   centerX/Y/Z   3 × float64（瓦片中心 ECEF 米）
+24  horizonOcclusion 2 × float32（变体：f32，跳过）
+32  centerX/Y/Z   3 × float64（变体重复头，跳过）
+56  minHeight/maxHeight 2 × float32
     - 兜底：当 f32 值不合理时按 float64 对读取（真实瓦片此处为
       ~3373.004 / -0.27）
+64  misc          3 × float64（包围球前缀，跳过）
+88  vertexCount   uint32
+92  顶点数据      u / v / h 三通道 zigzag-δ uint16
     索引区       indexCount uint32 + watermark uint32 + 索引
     （watermark ≠ 65536 时每个索引值需 + watermark；边缘索引与
      octvertexnormals 等扩展跳过）
```

关键点：

- **v 方向为 south-up TMS**：`v = 0` 是南边缘；若按 north-up 解读会落到
  南半球（实测湖北瓦片中心纬度 +28.99° vs −28.99° 之别）。
- **裙边（skirt）**：边缘顶点按 `max(50, maxHeight − minHeight)` 米向内
  复制，消除相邻 LOD 瓦片间的裂缝。

## 数据源 URL

| Provider | 默认 URL | 说明 |
|---|---|---|
| `LocalQuantizedMeshProvider` | `http://localhost:8888/api/v1/terrain-3d/serve/terrain_hubei/{z}/{x}/{y}.terrain` | 本地 terrain_hubei 服务（湖北，maximumLevel 14）；离线时引擎自动降级为 0 高椭球 |
| `TerrariumProvider` | `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png` | AWS open data（Mapzen/Tilezen），maximumLevel 15 |

## 后果

- 上层（拾取/量测/几何）与数据格式完全解耦，新增格式只需再实现一个
  `TerrainProvider`。
- Terrarium 路径依赖浏览器图像解码，Node 单测只能覆盖纯解码函数
  （已在 `tests/terrain/terrarium.spec.ts` 落实）。
- quantized-mesh 变体头是经验性布局，若后端升级为官方大端布局，
  需按 `QuantizedMeshDecoder` 的兜底分支（f64 minHeight/maxHeight）
  重新校验。
