# 影像瓦片体系（Imagery）

CesiumGPU 的影像系统负责把 XYZ 瓦片服务"流式"贴到椭球面上：按相机视域选择合适 LOD 层级的瓦片、并发受限地拉取、GPU 纹理池化复用、缺失瓦片向上回退到祖先瓦片。

涉及的核心类型：

| 类型 | 职责 |
|---|---|
| `ImageryProvider` | 瓦片源抽象接口（URL 模板 + 拉取） |
| `XyzImageryProvider` | 标准 XYZ 瓦片服务实现（含 OSM 预设） |
| `TileKey` | 瓦片坐标（level / x / y）与几何换算 |
| `ImageryLayer` | 一层影像：LOD 选片、请求调度、纹理池 |
| `ImageryLayerCollection` | 图层栈（挂在 `viewer.globe.layers`） |

---

## 目录

1. [30 秒上手](#1-30-秒上手)
2. [坐标系：TileKey 与 Web Mercator](#2-坐标系tilekey-与-web-mercator)
3. [ImageryProvider 接口](#3-imageryprovider-接口)
4. [XyzImageryProvider](#4-xyzimageryprovider)
5. [ImageryLayerCollection 图层栈](#5-imagerylayercollection-图层栈)
6. [LOD 参数调优](#6-lod-参数调优)
7. [自定义 Provider](#7-自定义-provider)
8. [渲染管线中的位置](#8-渲染管线中的位置)
9. [故障排查](#9-故障排查)

---

## 1. 30 秒上手

```typescript
import { Viewer, XyzImageryProvider } from '../src/index';

const viewer = new Viewer('cesiumContainer');
await viewer.initialize();

// 内置 OSM 预设（最短路径）
viewer.globe.layers.addImageryProvider(XyzImageryProvider.osm());

// 或自定义 XYZ 服务
viewer.globe.layers.addImageryProvider(new XyzImageryProvider({
  url: 'https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
  subdomains: ['1', '2', '3', '4'],
  maximumLevel: 18,
  tileSize: 256,
}));
```

`addImageryProvider` 返回 `ImageryLayer`，之后可调整图层属性或从图层栈移除。

---

## 2. 坐标系：TileKey 与 Web Mercator

`TileKey` 是标准 XYZ 切片坐标：`level`（0 = 整球）、`x`（列，[0, 2^level-1]）、`y`（行，**0 = 最北**，图像空间向下增长）。

```typescript
const key = new TileKey(5, 26, 12);

key.rectangle();   // TileRectangle：瓦片经纬度范围（弧度），Web Mercator 边界
key.parent();      // 父瓦片；level 0 返回 null
key.children();    // 四个子瓦片 [西北, 东北, 西南, 东南]
key.toString();    // "5/26/12" — 与大多数 XYZ 服务的路径一致

interface TileRectangle {
  /** 西边经度（弧度）。 */
  west: number;
  /** 东边经度（弧度）。 */
  east: number;
  /** 南边纬度（弧度）。 */
  south: number;
  /** 北边纬度（弧度）。 */
  north: number;
}
```

引擎内部用 `rectangle()` 在球面上生成瓦片网格几何，并把纹理 UV 映射到该范围——因此任何符合 Web Mercator（OSM / 天地图 / 高德）的 XYZ 服务都可直接接入。

---

## 3. ImageryProvider 接口

`ImageryProvider` 是瓦片源的抽象，自定义数据源只需实现它（`src/imagery/ImageryProvider.ts`）：

```typescript
export interface ImageryProvider {
  /**
   * URL 模板，支持占位符：
   * - `{z}` 瓦片层级
   * - `{x}` 瓦片列
   * - `{y}` 瓦片行
   * - `{s}` 子域（从 subdomains 轮转）
   */
  readonly url: string;

  /** 本 Provider 能提供的最高层级。 */
  readonly maximumLevel: number;

  /** 瓦片原生尺寸（像素），标准 XYZ 服务为 256。 */
  readonly tileSize: number;

  /** `{s}` 可用的子域列表（模板无 {s} 时为空）。 */
  readonly subdomains: readonly string[];

  /** Provider 是否就绪（元数据已加载）。 */
  readonly ready: boolean;

  /** 解析某瓦片的最终请求 URL（替换 {z}/{x}/{y}，确定性地轮转 {s}）。 */
  resolveUrl(key: TileKey): string;

  /**
   * 拉取一个瓦片。任何失败（网络错误 / 非 2xx / 解码失败）resolve 为
   * `null`，**绝不 reject** —— 调用方会跳过该瓦片并继续用父瓦片渲染。
   */
  requestTile(key: TileKey): Promise<ImageBitmap | null>;
}
```

设计要点：

- **失败即 null**：`requestTile` 从不抛异常。一次网络抖动不应打断整条渲染循环。
- **确定性子域轮转**：`{s}` 由瓦片坐标哈希决定（模块导出的 `rotateSubdomain` 函数），连续瓦片自然分散到不同子域，避免浏览器对单域的 6 连接限制。

---

## 4. XyzImageryProvider

标准实现，`fetch → blob → createImageBitmap` 三步取图：

```typescript
export interface XyzImageryProviderOptions {
  /** URL 模板，含 {z}/{x}/{y} 与可选 {s} 占位符。 */
  url: string;
  /** {s} 子域轮转列表。默认 []（不轮转）。 */
  subdomains?: string[];
  /** 最高可用层级。默认 18。 */
  maximumLevel?: number;
  /** 瓦片原生尺寸（像素）。默认 256。 */
  tileSize?: number;
}

export class XyzImageryProvider implements ImageryProvider {
  constructor(options: XyzImageryProviderOptions);

  /** 内置 OpenStreetMap 栅格瓦片预设。 */
  static osm(): XyzImageryProvider;

  resolveUrl(key: TileKey): string;
  async requestTile(key: TileKey): Promise<ImageBitmap | null>;
}
```

用法示例：

```typescript
// OSM
const osm = XyzImageryProvider.osm();
osm.resolveUrl(new TileKey(3, 4, 2));

// 高德卫星（多子域）
const amap = new XyzImageryProvider({
  url: 'https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
  subdomains: ['1', '2', '3', '4'],
  maximumLevel: 18,
});
```

---

## 5. ImageryLayerCollection 图层栈

图层栈挂在 `viewer.globe.layers`，接口对齐 Cesium 的 `ImageryLayerCollection`（简化版）：

```typescript
class ImageryLayerCollection {
  /** 追加一层（支持直接传 Provider，内部自动包一层 ImageryLayer）。 */
  addImageryProvider(
    provider: ImageryProvider | ImageryLayer,
    options?: Omit<ImageryLayerOptions, 'provider'>
  ): ImageryLayer;

  /** 移除一层（并释放其 GPU 纹理）。不存在时返回 false。 */
  remove(layer: ImageryLayer): boolean;
  /** 移除并销毁全部图层。 */
  removeAll(): void;
  /** 按下标取层（0 = 栈底）。 */
  get(index: number): ImageryLayer;
  /** 层的下标，不存在返回 -1。 */
  indexOf(layer: ImageryLayer): number;
  /** 把一层移到栈顶（渲染时盖住其他层）。 */
  raiseToTop(layer: ImageryLayer): void;
  /** 释放所有图层的 GPU 纹理。 */
  destroy(): void;

  /** 当前实际渲染的层：最顶部的可见层；全部隐藏时为 null。 */
  get activeLayer(): ImageryLayer | null;
}
```

> 当前不透明渲染路径只合成**最顶层可见图层**；`ImageryLayer.alpha` 的逐层透明混合为后续合成 pass 预留。

图层级属性在 `addImageryProvider` 的第二个参数或 `ImageryLayer` 构造时设置：

```typescript
viewer.globe.layers.addImageryProvider(
  new XyzImageryProvider({ url: '...' }),
  {
    show: true,
    alpha: 1,                    // 预留
    maximumScreenSpaceError: 2,  // LOD 细化阈值（见下）
    maximumLevel: 17,            // 覆盖 provider 的 maximumLevel，提前截断
    uploadsPerFrame: 10,         // 每帧纹理上传上限
    maxConcurrentRequests: 10,   // 并发拉取上限
    textureCacheCapacity: 200,   // LRU 纹理池容量
  }
);
```

---

## 6. LOD 参数调优

每帧 `ImageryLayer.update(device, camera, viewportHeight)` 执行一轮工作：**重选瓦片 → 受限并发请求 → 按 `uploadsPerFrame` 预算上传纹理**。

### maximumScreenSpaceError（SSE，默认 2）

屏幕空间误差（Screen Space Error）衡量"当前瓦片在屏幕上有多粗"。瓦片投影尺寸超出阈值就细分到子级：

- **调小**（如 1）：更高分辨率、更多瓦片、更多请求与显存。
- **调大**（如 4）：更快、省流量，海岸线略糊。
- 经验值：桌面端 2，移动端 3~4。

### maximumLevel（默认 provider.maximumLevel）

细化天花板。就算相机贴地，也不会请求超过该级的瓦片。若服务只到 17 级，显式设 17 可避免 404。

### uploadsPerFrame（默认 10）

每帧纹理上传条数上限，防止一次上传几十张 256×256 导致掉帧尖峰。注意这与 ADR-0001 的纹理池共同作用：上传的是**池中槽位**，不是每瓦片新建纹理。低端机可降到 4~6。

### maxConcurrentRequests（默认 10）

在飞请求上限。XYZ 服务通常限制单域 6 连接；配了多子域时可适当调大。

### textureCacheCapacity（默认 200）

LRU 纹理池容量（张数）。一张 256×256 RGBA8 ≈ 256 KB，200 张 ≈ 50 MB 显存。估算公式：

```
容量 ≈ 预算(MB) × 4    （即 预算字节 / (tileSize² × 4 字节)）
     例：128 MB → 512 张；50 MB → 200 张（默认）
```

池满时最久未用的瓦片纹理被回收，需要时重新拉取。

### 回退机制（ancestor fallback）

某瓦片纹理未就绪时，`resolveTexture(tile)` 沿父链回退（上限 `MAX_FALLBACK_LEVELS` 级），返回祖先纹理 + UV 子矩形，渲染端按子矩形采样——肉眼上是"先模糊后清晰"，不会出现黑洞：

```typescript
const resolved = layer.resolveTexture(tile);
if (resolved) {
  // resolved.texture: GPUTextureView（祖先纹理）
  // resolved.uvRect:  该瓦片在祖先纹理中的 UV 子矩形
  // u = x / 2^ancestorLevels … (x+1) / 2^ancestorLevels
}
```

---

## 7. 自定义 Provider

示例：把一组**本地预切瓦片**（`public/tiles/{z}/{x}/{y}.png`）包装成 Provider：

```typescript
import type { ImageryProvider } from '../src/index';
import { TileKey } from '../src/index';

class LocalTileProvider implements ImageryProvider {
  readonly url = '/tiles/{z}/{x}/{y}.png';
  readonly maximumLevel = 14;
  readonly tileSize = 256;
  readonly subdomains: readonly string[] = [];
  readonly ready = true;

  resolveUrl(key: TileKey): string {
    return this.url
      .replace('{z}', String(key.level))
      .replace('{x}', String(key.x))
      .replace('{y}', String(key.y));
  }

  async requestTile(key: TileKey): Promise<ImageBitmap | null> {
    try {
      const res = await fetch(this.resolveUrl(key));
      if (!res.ok) return null;          // 非 2xx → null，走父瓦片回退
      return await createImageBitmap(await res.blob());
    } catch {
      return null;                        // 网络/解码失败 → null，绝不 reject
    }
  }
}

viewer.globe.layers.addImageryProvider(new LocalTileProvider(), {
  maximumLevel: 14,
  maximumScreenSpaceError: 3,   // 本地瓦片量大，放宽一些
});
```

进阶场景：

- **WMTS / TMS**：在 `resolveUrl` 里换算 REST 风格 URL；TMS 的 y 轴翻转：`y = 2^level - 1 - y`。
- **需要鉴权**：在 `requestTile` 里给 `fetch` 加 headers；注意 CORS 预检，或改为同源代理。
- **矢量底图栅格化**：在 `requestTile` 里用 OffscreenCanvas 绘制后 `createImageBitmap`。

---

## 8. 渲染管线中的位置

影像层不经过独立的 pass，而是寄生在 **G-Buffer pass** 中：

```
每帧（Scene.render）
 ├─ ImageryLayerCollection.activeLayer.update(device, camera, viewportHeight)
 │    ├─ 选片（SSE 驱动的四叉树细分）
 │    ├─ 调度请求（并发上限 + 失败记录）
 │    └─ 上传纹理（帧预算内，写入 LRU 池）
 ├─ GBufferPass
 │    ├─ 球体本体（单张影像纹理或程序化纹理）
 │    └─ Globe.drawTiles(...)  ← 逐瓦片图元，UV 子矩形回退采样
 └─ ...
```

`Globe` 把每块可见瓦片包装为独立 `Primitive`（`viewer.globe.tilePrimitives` / `tilePrimitiveCount` 可观测），因此瓦片同样参与 GPU 拾取。

---

## 9. 故障排查

**Q: 瓦片一直显示模糊（祖先回退不消失）？**
① 请求实际 404（层级超过服务上限）→ 设置 `maximumLevel`；② CORS 拦截 → 控制台看网络面板，配代理或响应头；③ `subdomains` 写错导致域名不存在。

**Q: 控制台大量 "tile fetch failed" 警告？**
`XyzImageryProvider` 对每块失败瓦片只警告一次且本会话不再重试，持续放大一般是服务限流——调低 `maxConcurrentRequests`。

**Q: 相机快速移动时掉帧？**
上调 `maximumScreenSpaceError`、下调 `uploadsPerFrame` 与 `textureCacheCapacity`；必要时限制 `maximumLevel`。

**Q: 想两层影像叠加（如路网叠影像）？**
当前版本只渲染 `activeLayer`（最顶层可见层），逐层 alpha 合成尚未实现。替代方案：选择带路网注记的瓦片源，或自行实现第二套瓦片图元。

---

## 相关文档

- [ADR-0001 瓦片纹理方案：texture-per-tile + LRU 纹理池](../decisions/0001-tile-texture-pool.md) — 纹理池设计与上传预算的完整推导
- [快速开始](./quick-start.md) — 影像接入的最小示例
- [API 参考](../api/index.html) — `ImageryProvider` / `XyzImageryProvider` / `ImageryLayer` 完整签名

## P1 新增：XYZ provider 认证 headers

`XyzImageryProvider({ url, headers })` —— headers 透传到每个瓦片请求（如私有影像服务的 `Authorization: Bearer …`）；不传时行为不变。
