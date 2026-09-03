# Entity 数据模型（Entities）

Entity 是 CesiumGPU 的地理数据抽象层，语义对齐 Cesium 的 `Entity` 子集：一个 Entity 描述"一个地理位置上的一组图形"（点 / 折线 / 面 / 标注），由 `EntityCollection` 集中管理、`EntityVisualizer` 负责翻译成 GPU 图元。

数据流总览：

```
GeoJSON 文件 ──load──▶ GeoJsonDataSource ──▶ EntityCollection（viewer.scene.entities）
                                                 │
                                                 ▼
                                          EntityVisualizer（Scene 内部创建）
                             ┌───────────────────┴───────────────────┐
                    polygon / polyline                          point / label
                    （世界空间图元）                            （屏幕空间公告板）
                             │                                       │
                  不透明 → G-Buffer Pass                     Overlay Pass
                  半透明 → WBOIT Transparent Pass            （billboard 绘制）
```

---

## 目录

1. [Entity 模型](#1-entity-模型)
2. [坐标约定：归一化 ECEF](#2-坐标约定归一化-ecef)
3. [四种图形（Graphics）](#3-四种图形graphics)
4. [EntityCollection 与响应式](#4-entitycollection-与响应式)
5. [GeoJSON 加载](#5-geojson-加载)
6. [EntityVisualizer 渲染路径](#6-entityvisualizer-渲染路径)
7. [性能与最佳实践](#7-性能与最佳实践)

---

## 1. Entity 模型

```typescript
import { Entity } from '../src/index';

export interface EntityOptions {
  id?: string;              // 缺省自动生成 RFC4122 v4 uuid
  name?: string;
  show?: boolean;
  showPointOnTop?: boolean; // 同时有 point 与其他图形时，点是否盖在最上
  position?: Cartesian3;    // 归一化 ECEF（点/标注锚点）
  point?: PointGraphics;
  polyline?: PolylineGraphics;
  polygon?: PolygonGraphics;
  label?: LabelGraphics;
}

const entity = new Entity({ name: '北京站' });
entity.id;    // 读取（构造时确定）
```

一个 Entity 的图形属性是**可写属性**——重新赋值会触发 `definitionChanged` 事件，EntityVisualizer 收到后自动重建该实体的 GPU 图元：

```typescript
entity.polygon = { hierarchy: { positions: [/* ... */] }, height: 0.001, color: Color.RED };
```

---

## 2. 坐标约定：归一化 ECEF

**引擎世界空间 = 归一化 ECEF**：地球赤道半径 = 1.0（1 单位 = `EARTH_SCALE` = 6 378 137 米）。

`Cartesian3.fromDegrees(lon, lat, h)` 返回的是**米制** ECEF（模长 ≈ 6.37e6），直接赋给实体会被画到"无限远处"。必须归一化：

```typescript
import { Cartesian3 } from '../src/index';

// lon/lat(度) + 高度(米) → 归一化 ECEF
// 约定与 GeoJsonDataSource 一致：单位径向方向 × (1 + height / EARTH_RADIUS)
function lonLat(lon: number, lat: number, heightMetres = 0): Cartesian3 {
  const dir = Cartesian3.normalize(Cartesian3.fromDegrees(lon, lat, 0), new Cartesian3());
  const scale = 1 + heightMetres / Cartesian3.EARTH_RADIUS;
  return new Cartesian3(dir.x * scale, dir.y * scale, dir.z * scale);
}

lonLat(116.4, 39.9);            // 地表点（模长 = 1）
lonLat(116.4, 39.9, 1000);      // 海拔 1 km
```

> `GeoJsonDataSource.load` 会**自动做这个换算**；手写 Entity 时才需要上述帮助函数。

面的 `height` 同样是归一化单位（不是米）：

```typescript
height: 3000 / Cartesian3.EARTH_RADIUS   // 抬升 3 km
```

---

## 3. 四种图形（Graphics）

### PointGraphics — 屏幕空间点

```typescript
export interface PointGraphics {
  color?: Color;
  pixelSize?: number;   // 屏幕像素直径；渲染层换算为世界单位
}
```

### PolylineGraphics — 折线

```typescript
export interface PolylineGraphics {
  positions?: Cartesian3[];  // 归一化 ECEF，≥ 2 个点
  width?: number;            // 像素宽（见下方换算说明）
  color?: Color;
}
```

> 折线底层是 `PolylineGeometry` 的**世界空间条带（ribbon）**，宽度换算系数为 1 px ≈ 64 m 世界宽度（`PIXELS_TO_WORLD_WIDTH = 1e-5`），因此屏幕上的实际线宽随缩放变化。这是当前版本的已知折衷。

### PolygonGraphics — 多边形

```typescript
export interface PolygonGraphics {
  hierarchy?: PolygonHierarchy;
  height?: number;    // 归一化抬升（1 = 地球半径）；用于挤出/贴地判断
  color?: Color;
}

export interface PolygonHierarchy {
  positions: Cartesian3[];   // 外环，≥ 3 个点，不需要重复首尾点
  holes?: Cartesian3[][];    // 内环（洞）
}
```

### LabelGraphics — 标注

```typescript
export interface LabelGraphics {
  text?: string;
  font?: string;
  fillColor?: Color;
  show?: boolean;
}
```

### 组合示例

```typescript
const entities = viewer.scene.entities;

entities.add(new Entity({
  id: 'terminal-pek',
  name: '首都机场',
  position: lonLat(116.6, 40.08),
  point:   { color: Color.LIME, pixelSize: 10 },
  label:   { text: 'PEK', font: '16px sans-serif', fillColor: Color.WHITE },
}));
```

---

## 4. EntityCollection 与响应式

```typescript
class EntityCollection {
  /** 添加实体；id 冲突时抛错。返回加入的实体。 */
  add(entityOrOptions: Entity | EntityOptions): Entity;
  /** 移除实体，返回是否找到并移除。 */
  remove(entity: Entity): boolean;
  /** 清空。以单次 collectionChanged 通知全部移除。 */
  removeAll(): void;
  /** 按 id 查询，不存在返回 undefined。 */
  getById(id: string): Entity | undefined;

  /** 实体数量。 */
  get length(): number;
  /** 增删时触发，监听器收到 (added, removed, moved) 三个数组。 */
  readonly collectionChanged: Event;
}

class Event<Args extends unknown[] = unknown[]> {
  get numberOfListeners(): number;
  addListener(listener: (...args: Args) => void): void;
  removeListener(listener: (...args: Args) => void): void;
  raiseEvent(...args: Args): void;
}
```

监听集合变化（数据绑定的基础）：

```typescript
const { collectionChanged } = viewer.scene.entities;

collectionChanged.addListener((added, removed, moved) => {
  console.log(`+${added.length} -${removed.length} ~${moved.length}`);
});
```

监听单个实体变化：

```typescript
entity.definitionChanged.addListener((e, propertyName) => {
  console.log(`${e.id} 的 ${propertyName} 被重新赋值`);
});
```

---

## 5. GeoJSON 加载

`GeoJsonDataSource` 一次加载整个 FeatureCollection，自动换算坐标并生成实体：

```typescript
class GeoJsonDataSource {
  constructor(name: string);
  get name(): string;
  get entities(): EntityCollection;

  /**
   * 从 URL 或内存对象加载。
   * 支持：Point / MultiPoint / LineString / MultiLineString /
   *       Polygon / MultiPolygon（Feature 与 FeatureCollection）
   * 所有坐标自动转为归一化 ECEF。
   */
  static load(input: string | object, options?: GeoJsonLoadOptions):
    Promise<GeoJsonDataSource>;
}

export interface GeoJsonLoadOptions {
  /** 数据源名。URL 加载时默认取文件名，否则 'GeoJSON'。 */
  name?: string;
}
```

### URL 加载

```typescript
import { GeoJsonDataSource } from '../src/index';

const ds = await GeoJsonDataSource.load('/data/china-provinces.geojson');
// 载入后把实体并入场景集合
ds.entities;
// GeoJsonDataSource 自带 entities 集合，可直接遍历其内容并入 scene
```

### 内存对象加载

```typescript
const ds = await GeoJsonDataSource.load({
  type: 'FeatureCollection',
  features: [
    {
      type: 'Feature',
      properties: { name: '样例区' },
      geometry: {
        type: 'Polygon',
        coordinates: [[[116, 39], [117, 39], [117, 40], [116, 40], [116, 39]]],
      },
    },
  ],
});
```

### 每要素样式（properties.style）

Feature 的 `properties.style` 成员会被读作**逐要素样式覆盖**，字段映射到对应 Graphics 属性：

```json
{
  "type": "Feature",
  "properties": {
    "name": "京广线",
    "style": { "color": "#ff4040", "width": 3 }
  },
  "geometry": {
    "type": "LineString",
    "coordinates": [[116.4, 39.9], [113.3, 23.1]]
  }
}
```

> GeoJSON 坐标是 `[lon, lat, height?]`（度/米）。加载器内部执行与第 2 节相同的归一化，面的 `height` 取环上首个位置的高度。

---

## 6. EntityVisualizer 渲染路径

`EntityVisualizer` 由 `Scene` 在初始化时**内部创建**（业务层一般不直接实例化）：

```typescript
// Scene.initialize() 内部
this._entityVisualizer = new EntityVisualizer(this._context, this.entities, this);
```

其宿主接口（`EntityVisualizerHost`）由 `Scene` 结构化满足：

```typescript
export interface EntityVisualizerHost {
  readonly primitives: PrimitiveCollection;
  /** 不透明图元走 G-Buffer 路径。 */
  addPrimitive(primitive: Primitive): void;
  /** 半透明图元走 WBOIT 透明路径。 */
  addTransparentPrimitive(primitive: Primitive): void;
}
```

### G-Buffer vs Overlay：谁走哪条路

| 图形 | 渲染路径 | 说明 |
|---|---|---|
| polygon / polyline（不透明） | **G-Buffer Pass** | 写入 Albedo/Normal/WorldPos，参与延迟光照，**可被 GPU 拾取** |
| polygon / polyline（半透明，alpha < 1） | **WBOIT Transparent Pass** | 加权混合无序透明 |
| point / label | **Overlay Pass** | 屏幕空间 billboard，按像素尺寸绘制，不参与 G-Buffer |

判定规则：构造图元时检查颜色 alpha（或面 `height` 决定挤出/贴地），`translucent` 为真走 `addTransparentPrimitive`，否则 `addPrimitive`。

### 响应式重建

- Entity `definitionChanged` → 只重建该实体的图元；
- Entity 从集合移除 → 其图元经 `PrimitiveCollection.remove` 回收（销毁 GPU buffer）；
- `destroy()` → 释放全部图元与事件监听。

几何构建入口（内部 `_buildVisuals`）：

- `polygon.hierarchy.positions.length >= 3` → `PolygonGeometry` 挤出/贴地面片；
- `polyline.positions.length >= 2` → `PolylineGeometry` 条带。

---

## 7. 性能与最佳实践

1. **复用 id**：需要跨帧更新（如车辆轨迹）时固定 `id`；重复 `add` 同 id 会抛错——先 `getById` 判断再 `remove` + `add`，或直接改属性触发 `definitionChanged`。
2. **面优先、线其次、点最后**：面和线是世界空间图元（一次绘制、可拾取），点是屏幕空间公告板（每实例开销更高）。
3. **半透明面数量控制**：WBOIT pass 每个半透明面都是一组累加绘制，大面积半透明面（如行政区划填色）尽量合并。
4. **批量静态数据用 GeoJSON**：一次 `load` 生成实体集合，比逐帧手工 `add` 更快也更易维护。
5. **折线宽度是米级世界宽度**：全球视角下 1 px ≈ 64 m；需要"恒定屏幕宽"的引导线可考虑 Overlay 层自绘。

---

## 相关文档

- [快速开始](./quick-start.md) — Entity 最小示例
- [交互三件套](./interaction.md) — 拾取 Entity 生成的图元
- [引擎架构](./architecture.md) — G-Buffer / WBOIT / Overlay 在整条管线中的位置
- [API 参考](../api/index.html) — `Entity` / `EntityCollection` / `GeoJsonDataSource` 完整签名
