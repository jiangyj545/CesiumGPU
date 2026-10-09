# 地形与贴地（Terrain）

地形系统让地球从「光滑椭球」变成真实起伏的地表：`TerrainProvider` 负责把
DEM 瓦片解码成三角网，`TerrainLayer`（挂在 `globe.terrainLayer`）汇聚瓦片
并对外提供 `sampleHeight(lon, lat)` 高程采样；拾取、量测、几何三类消费端
据此实现**贴地（clamp to ground）**。

> 坐标约定与全引擎一致：内部使用归一化 ECEF（地球赤道半径 = 1），
> 高程以米为单位按 `1 + h / 6378137` 折入半径。格式选型见
> [ADR 0004 - 地形 Provider 双格式方案](../decisions/0004-terrain-providers.md)。

## 1. Provider 配置

引擎内置两个 `TerrainProvider` 实现（可自选其一传入 TerrainLayer）：

### LocalQuantizedMeshProvider（本地 quantized-mesh 服务）

```ts
import { LocalQuantizedMeshProvider } from 'cesiumgpu';

const provider = new LocalQuantizedMeshProvider({
  // 默认值即本地 terrain_hubei 服务，可省略；此处显式写出便于替换
  url: 'http://localhost:8888/api/v1/terrain-3d/serve/terrain_hubei/{z}/{x}/{y}.terrain',
  maximumLevel: 14,   // 默认 14
});
```

- 瓦片方案：geographic EPSG:4326 **TMS（south-up，z0 = 2×1）**；
- 瓦片格式：gzip 压缩的 quantized-mesh **小端变体**（解码器已按湖北真实
  瓦片逐字节校验，含裙边防裂缝）；
- 失败瓦片会进入会话内失败缓存，不再重复请求。

### TerrariumProvider（公网 Terrarium PNG，备选）

```ts
import { TerrariumProvider } from 'cesiumgpu';

const provider = new TerrariumProvider({
  // 默认：AWS elevation-tiles-prod（Mapzen/Tilezen）
  url: 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png',
  maximumLevel: 15,
});
```

- 瓦片方案：Web Mercator slippy XYZ（north-up）；
- 像素编码：`(R*256 + G + B/256) − 32768` 米（海洋为负，无 NoData）；
- 依赖浏览器图像解码 API，Node 单测环境不可用（仅纯解码函数可测）。

### 挂到 Globe

```ts
viewer.globe.terrainLayer = new TerrainLayer({ providers: [provider] });
```

挂载后地形瓦片随相机 LOD 流式加载渲染，同时 `sampleHeight` 开始可用；
卸载 / 切换时调用 `destroy()` 释放（会 abort 在途请求）。

## 2. 双格式说明（何时用哪个）

| | LocalQuantizedMeshProvider | TerrariumProvider |
|---|---|---|
| 适用场景 | 内网 / 本地后端（如湖北 terrain_hubei） | 无后端、公网可用时快速验证 |
| 网格质量 | 顶点级 TIN + 裙边，接缝干净 | 像素三角化，无裙边，接缝可见 |
| 解码开销 | 纯二进制解码（快） | PNG 解码 + CPU 三角化（较慢） |
| 瓦片方案 | geographic TMS（south-up） | Web Mercator XYZ（north-up） |

两者对上层完全同构：`requestTileGeometry(z, x, y) → TerrainMeshData`，
自定义数据源只需再实现该接口。

## 3. 贴地用法

所有贴地消费端都只依赖一个采样回调：
`sampleHeight(lonDeg, latDeg) → 高程（米）`，无数据时返回 `0`。
先把回调绑好（geometry / analysis 模块不 import terrain，保持分层）：

```ts
const sampleHeight = (lon, lat) => viewer.globe.terrainLayer.sampleHeight(lon, lat);
```

### 拾取贴合（自动生效）

`camera.pickGlobe(ndcX, ndcY)` 命中地表后自动经 `globe.terrainLayer` 把
命中点沿法向吸附到地形面（定点迭代 ≤ 3 次，采样高程变化 < 1 cm 即收敛；
无数据处保持椭球命中）。Viewer 的中键 orbit pivot 复用该结果，拖拽旋转
自然贴合山地。`DrawHandler` 的绘制拾取同样自动贴地（GPU pick 本就命中
地形网格；`Measure.rayEllipsoid` 回退路径也接入了同一贴合）。

### 量测贴合（显式传钩子）

`Measure` 默认保持 0 高行为；需要贴地量测时注入钩子，分析层不感知地形模块：

```ts
import { Measure } from 'cesiumgpu';

const clamp = (p) => Measure.clampToTerrain(p, sampleHeight);
const meters = Measure.distance(positions, false, { clampToTerrain: clamp });
const areaM2 = Measure.area(positions, { clampToTerrain: clamp });
```

`Measure.clampToTerrain(position, sampleHeight)` 本身也是公开工具：
ECEF → 经纬度（度）→ `sampleHeight` → 返回贴地 ECEF。

### 几何贴合（heightReference + sampleHeight）

折线在段内自动加密（每段按球面角 2° 步长、至少 32 细分），避免长边直接
切穿山谷；多边形对每个环顶点贴地（内部为粗壳）：

```ts
// Entity 或 Primitive 的几何参数
polyline: {
  positions,
  width: 3,
  heightReference: 'clampToGround',  // 默认 'none'（行为不变）
  sampleHeight,                      // 缺省时退回 'none' 行为
}
polygon: {
  hierarchy: { positions: ring },
  heightReference: 'clampToGround',
  sampleHeight,
}
```

## 4. 离线 / 服务未启动

`LocalQuantizedMeshProvider` 需要**本地后端 8888 在线**
（`http://localhost:8888/api/v1/terrain-3d/serve/terrain_hubei/...`）。
服务离线时：

- 瓦片请求失败进入失败缓存，不报错、不阻塞渲染；
- `sampleHeight` 一律返回 `0`，引擎自动降级为 **0 高椭球**：
  拾取/量测/几何保持可用，只是没有地形起伏；
- 可运行示例页 `public/examples/14_terrain.html`（flyTo 湖北 → 拉近，
  含贴地折线、贴地多边形与贴地量测）自检服务连通性。

## 5. 示例

- [14_terrain.html](../../public/examples/14_terrain.html)：湖北地形 + 贴地折线/多边形 + 贴地量测完整示例。

## 相关文档

- [ADR 0004 - 地形 Provider 双格式方案](../decisions/0004-terrain-providers.md)：格式对比、小端变体解码器背景、数据源 URL
- [空间量测（Analysis）](./analysis.md)：`Measure` 距离/面积/高度计算
- [交互三件套（Interaction）](./interaction.md)：`DrawHandler` 球面绘制

## P1 新增：认证 headers 与 sampleTerrainMostDetailed

### provider headers（P1-F）

`TerrariumProvider` / `LocalQuantizedMeshProvider` / `HttpTerrainProviderBase` 子类的 options 增加 `headers?: Record<string, string>`，随每个瓦片请求发送——对齐 Cesium `new Resource({ url, headers: { Authorization: 'Bearer …' } })` 的用法（构造 provider 时传即可，无需 Resource 类）。

### sampleTerrainMostDetailed / sampleTerrain（P1-H）

```typescript
import { sampleTerrainMostDetailed, Cartographic } from './src/index';

const positions = [Cartographic.fromDegrees(116.4, 39.9)];
const sampled = await sampleTerrainMostDetailed(terrainLayer, positions);
console.log(sampled[0].height); // 米；无覆盖瓦片 → 0（椭球短路）
```

语义对齐 Cesium：对 positions 的包络矩形按 provider 最高层 `ensureRegion` 预载，逐点 `sampleHeight` 回填；返回**新数组**（入参不改写）；`sampleTerrain(layer, positions, level)` 可钉定层级。

`viewer.terrainLayer`（P2）：Viewer 只读访问器直连当前地形层，移植时无需适配：

```typescript
const sampled = await sampleTerrainMostDetailed(viewer.terrainLayer!, positions);
```

## P2-8：LERC 地形（ArcGIS World Elevation Terrain3D）

```typescript
const provider = new LercTerrainProvider({
  // url 默认 https://elevation3d.arcgis.com/arcgis/rest/services/
  //        WorldElevation3D/Terrain3D/ImageServer/tile/{z}/{y}/{x}（y/x 顺序）
  maximumLevel: 15,
  // headers: { Authorization: 'Bearer …' }, // 私有服务（P1-F）
});
viewer.globe.setTerrainProvider(provider);
```

- **解码** `decodeLerc1`：LERC1（"CntZImage"）——头 + 掩码 RLE 位图 + 块编码（constant-0 / constant-offset / bitstuffed），位填充 MSB-first 按小端 u32 字消费、尾字无效字节左移对齐（移植自 Esri lerc.js Apache-2.0，见 LercDecoder.ts 头注）。无效像素（掩码洞）取 0。
- **适配**：与 TerrariumProvider 同一模式（webmercator 瓦片 → `triangulateHeightGrid` 规则网格三角化 → 归一化 ECEF）。
- **测试**：测试专用最小编码器 `encodeLerc1TestTile`（常值/梯度/掩码洞三档）闭环解码断言（`tests/terrain/lercDecoder.spec.ts`），provider 层断言 URL 模板（y/x 顺序）/headers 透传/网格语义（`lercTerrainProvider.spec.ts`）——不 mock 解码器。
- 在线源失败（离线/防火墙）时走既有失败缓存路径：单瓦片失败不炸树，整层静默降级为椭球（示例 100 有降级说明）。
- LERC 移植来源与许可：Esri lerc.js（npm `lerc` 2.0.0，Apache-2.0，Copyright 2015-2018 Esri）；CesiumJS 1.135 同源打包。
