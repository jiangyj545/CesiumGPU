# 3DTiles 瓦片与 pnts 点云（3DTiles & Point Cloud）

3DTiles 调度与集成层指南：tileset.json 树调度（`Tileset3D`）、b3dm/glb/pnts
内容分派、**pnts 点云瓦片**（含 `3DTILES_draco_point_compression`）与
`.e3m` 等私有后缀的 magic 嗅探、RTC_CENTER 包围体合成。坐标约定统一为
「归一化 ECEF（地球半径 = 1）」，详见[引擎架构](./architecture.md)第 4 节。

## 1. 调度骨架（Tileset3D）

`Tileset3D` 每帧 `update(camera, viewportHeight)` 三段：SSE 树遍历细分 →
并发 8 的内容请求队列（去重 / 近者优先 / 祖先优先）→ REPLACE/ADD 渲染集
状态机 + LRU 两段式延迟销毁。单瓦片失败不炸整树（`stats.failed` 计数 +
30s 重试窗口）。用法见示例
[94_tileset3d](http://localhost:3001/examples/94_tileset3d.html)。

## 2. 内容分派：magic 嗅探（不是扩展名）

content fetch 回字节后读**前 4 字节**判定格式（`sniffTileContentKind`）：

| magic | kind | 处理 |
|---|---|---|
| `b3dm` | b3dm | 剥壳 → GLB → `Primitive[]` |
| `glTF` | glb | 整包 GLB / glTF（3DTiles 1.1 直 content） |
| `pnts` | pnts | `decodePnts` → `PointCloudPrimitive`（特效 pass）；magic 为文件字节 `70 6e 74 73`（小端 uint32 `0x73746e70`） |

- **扩展名仅作嗅探失败时的回退提示**（`.b3dm`/`.glb`/`.pnts` 预判，
  再不识别按 GLB 试解析）。
- **`.e3m` 等私有后缀不阻断**：地面站数据常把标准 pnts 换个后缀，
  magic 嗅探才是真相——tileset.json 里 `content.uri: "original.e3m"`
  直接放行入队。

## 3. pnts 点云瓦片

`PntsDecoder.decodePnts(buffer)` 解码 FeatureTable：

- **非压缩**：`POSITION`（float32×3）或 `POSITION_QUANTIZED`
  （uint16×3 + `QUANTIZED_VOLUME_SCALE`/`OFFSET` 反量化
  `q·scale/(2^16−1)+offset`）；
- **颜色**：`RGB`（u8×3）/ `RGBA`（u8×4，丢 alpha）/ `CONSTANT_RGBA`
  （全局平铺）/ 无色填白；
- **RTC_CENTER**：JSON 数组与 binary（`{byteOffset}` float32×3）两种
  形态都支持，解码层逐点合入（输出米制 ECEF）；
- **Draco 扩展**：见下节。

渲染走 `PointCloudPrimitive`（`src/scene/PointCloudPrimitive.ts`，
骨架同 FlowArrowPolyline）：每点 6 顶点 quad、恒定像素圆点（CSS px ×dpr，
透视正确）、src-alpha 混合、**GBuffer 深度遮挡**（被地形/建筑遮住的点
discard）。它实现 `IPass`（Effect 形态），由 `Tileset3D` 经
`scene.effects` 挂载——`BuiltTileContent.effects` 与 `primitives`
同语义地参与渲染集显隐（`show`）、LRU 与两段式销毁。

示例：[101_pnts](http://localhost:3001/examples/101_pnts.html)
（数据由 `node scripts/buildPntsDemo.mjs` 生成：5 万点山丘 + 高度色 +
RTC_CENTER 北京西山）。

## 4. Draco 点压缩（3DTILES_draco_point_compression）

`{"extensions":{"3DTILES_draco_point_compression":{"byteOffset":0,
"byteLength":453632,"properties":{"POSITION":0,"RGB":1,"BATCH_ID":2}}}}`

- `byteOffset/byteLength` 在 **featureTable 二进制体内**切出 Draco 流；
  `properties` 的数字 = Draco attribute **uniqueId**；
- `DracoDecoder`（`src/loader/DracoDecoder.ts`）单例服务：fetch wrapper
  文本 → `new Function(...)` 取 `DracoDecoderModule` 工厂 →
  `{wasmBinary}` 实例化 → `DecodeBufferToPointCloud` +
  `GetAttributeByUniqueId` 按 id 取 POSITION（DracoFloat32Array）/
  RGB·RGBA（DracoUInt8Array），临时对象 `module.destroy` 全释放；
- 失败语义：`ready()` 失败返 null（不抛）；`decodePointCloud` 异常
  console.warn 一次后返 null → 该 tile 计入 `stats.failed`（30s 重试）。

### wasm 部署路径

wrapper 与 wasm 约定**同目录**（`draco_wasm_wrapper.js` +
`draco_decoder.wasm`，来自 `draco3d` 官方包）。缺省候选序列逐一探测
（fetch 200 即用）：

1. `/wasm/draco/draco_wasm_wrapper.js` —— 本仓库自带（`public/wasm/draco/`，
   vite 静态服务，开箱即用）；
2. `/3d/public/Cesium/ThirdParty/draco_wasm_wrapper.js` —— 外部 Cesium
   部署（不存在自动跳过）。

异构部署在加载前显式指定（wasm 同目录推导）：

```ts
engine.DracoDecoder.setWasmUrl('/assets/draco/draco_wasm_wrapper.js');
```

## 5. RTC_CENTER 包围体合成

部分数据源的 tileset.json 根 `boundingVolume` 是**原点附近的局部小球**
（半径百米级），而内容的真实位置由 `RTC_CENTER`（米制 ECEF，湖北某处）
决定——直接拿局部球做视锥剔除/SSE 会把点云判到地心附近。

对齐 CesiumJS 语义：pnts 内容加载完成后，用解码点位（RTC 已合入、瓦片
世界矩阵已烘焙）的**实测世界包围球**替换 `node.sphere`
（`BuiltTileContent.worldSphere` → `Tileset3D._startRequest`）。
`bound = rtcCenter + localSphere` 的位置由点位实测收紧，剔除/距离更准；
仅 pnts 路径生效，b3dm/glb 保持 tileset.json 声明值不变。

### 调度期回退整球（局部根球）

内容加载前若直接按局部球换算（中心经归一化矩阵映射后落在原点附近），
地心视锥剔除会把根 tile 整棵错杀。`TilesetContentLoader` 在解析期判定：
tile **无 transform 链**且 boundingVolume 为近地心（距地心 < 1000 km）
小球/盒（半径 < 100 km，典型局部 authoring 米制坐标，如湖北点云
`sphere [84.126,40.125,0,r=100]`）→ 调度剔除/SSE 用**回退整球**
（`needsFallbackWorldSphere`：center = 原点、radius = 1.6）。内容加载后
由上文 worldSphere 回写机制自然修正为真实位置；有 transform 的数据集
不受影响。
