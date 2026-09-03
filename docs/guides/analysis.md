# 空间量测（Analysis）

距离 / 面积 / 高度量测，以及量测结果管理与「绘制 → 量测」组合流程。

## 目录

1. [30 秒上手](#1-30-秒上手)
2. [坐标系约定：归一化 ECEF](#2-坐标系约定归一化-ecef)
3. [Measure：纯计算函数集](#3-measure纯计算函数集)
4. [MeasureCollection：量测结果管理](#4-measurecollection量测结果管理)
5. [与 DrawHandler 组合：交互式量测](#5-与-drawhandler-组合交互式量测)
6. [在场景中渲染量测结果](#6-在场景中渲染量测结果)
7. [精度与算法说明](#7-精度与算法说明)
8. [常见问题](#8-常见问题)

## 相关文档

- [快速开始](./quick-start.md) — 引擎最小示例
- [交互三件套](./interaction.md) — `DrawHandler` 手势与事件全表
- [Entity 数据模型](./entities.md) — 量测结果落库为实体
- [引擎架构](./architecture.md) — 归一化坐标系在管线中的约定
- [API 参考](../api/index.html) — `Measure` / `MeasureCollection` 完整签名

---

## 1. 30 秒上手

量测 API 全部从 `src/index.ts` 导出：

```typescript
import {
  Measure,               // 纯静态计算：距离 / 面积 / 高度 / 射线求交
  MeasureCollection,     // 量测结果管理 + 展示文本生成
} from './src/index';

const collection = new MeasureCollection();

// 由顶点直接生成一条带文本的量测记录
const item = collection.createMeasurement('distance', [beijing, shanghai]);
console.log(item.label);   // "1,068.3 km"（自动格式化的展示文本）
console.log(item.value);   // 1068300.x（米，数值）
```

`Measure` 只做计算（纯静态、无 GPU、不改输入）；渲染（画线、标注文字）由调用方组合 `Entity` 完成，见第 6 节。

## 2. 坐标系约定：归一化 ECEF

**这是量测模块最重要的前提。** 引擎内部世界坐标为**归一化 ECEF**：WGS84 赤道半径 = 1.0，即

> **1 个内部单位 = `Measure.EARTH_SCALE` = 6 378 137 米**

而 `Cartesian3.fromDegrees(lon, lat, h)` 返回**米制** ECEF（模长 ≈ 6.37e6）。量测函数接受的输入是**归一化**坐标（如 `Camera.position`、`scene.pick()` 返回的 `worldPosition`、`DrawHandler` 的 `positions`），返回值是**米 / 平方米**：

| 输入 / 输出 | 单位 |
|---|---|
| 顶点 `positions` | 归一化 ECEF（地球半径 = 1） |
| `distance()` 返回 | 米 |
| `area()` 返回 | 平方米（m²） |
| `height()` 返回 | 米（`deltaHeight` 字段） |

从经纬度（度 + 米）构造归一化 ECEF 的常用帮助函数：

```typescript
import { Cartesian3 } from './src/index';

function lonLat(lon: number, lat: number, heightMetres = 0): Cartesian3 {
  const dir = Cartesian3.normalize(Cartesian3.fromDegrees(lon, lat, 0), new Cartesian3());
  const scale = 1 + heightMetres / Cartesian3.EARTH_RADIUS;
  return new Cartesian3(dir.x * scale, dir.y * scale, dir.z * scale);
}
```

## 3. Measure：纯计算函数集

`Measure`（`src/analysis/Measure.ts`）全部为静态方法。签名摘要：

```typescript
export class Measure {
  /** 比例因子：1 归一化单位 = EARTH_SCALE 米（WGS84 赤道半径） */
  static readonly EARTH_SCALE: number;      // 6378137.0
  /** WGS84 极半径 / 赤道半径比（归一化 b ≈ 0.996647） */
  static readonly POLAR_RATIO: number;
  /** WGS84 平均半径（米），用于球面面积 */
  static readonly MEAN_RADIUS: number;      // 6371008.8

  /** 将点贴地：归一化到椭球单位半径（|p| = 1），零向量原样返回 */
  static clampToEllipsoid(point: Cartesian3, result?: Cartesian3): Cartesian3;

  /** 折线长度（米）。相邻点分段累加，椭球测地线近似 */
  static distance(points: Cartesian3[], clamped = false): number;

  /** 球面多边形面积（m²）。顶点先贴地，顶点顺序不限，取绝对值 */
  static area(points: Cartesian3[]): number;

  /** 两点大地高差（米）。返回 { deltaHeight } */
  static height(p0: Cartesian3, p1: Cartesian3): { deltaHeight: number };

  /** 射线与 WGS84 椭球（归一化）最近交点；脱靶返回 null */
  static rayEllipsoid(origin: Cartesian3, direction: Cartesian3): Cartesian3 | null;
}
```

三个量测的最小示例：

```typescript
// ① 距离：至少 2 个点，按折线分段累加
const length = Measure.distance([a, b, c]);            // 米

// ② 面积：至少 3 个点，不含重复的闭合首尾点
const area = Measure.area([p1, p2, p3, p4]);           // m²

// ③ 高度差：恰好 2 个点
const { deltaHeight } = Measure.height(ground, peak);  // 米
```

`distance(points, clamped)` 的 `clamped` 参数：传 `true` 时每段先调用 `clampToEllipsoid` 把顶点投影到地表再计算，用于「顶点带高度、只想要水平投影长度」的场景；默认 `false` 按原坐标计算。

`rayEllipsoid` 常用于自定义拾取 / 视线分析：

```typescript
const hit = Measure.rayEllipsoid(camera.position, viewDirection);
if (hit) {
  console.log('视线落点（归一化 ECEF）:', hit);
} else {
  console.log('射线脱靶（射向太空）');
}
```

## 4. MeasureCollection：量测结果管理

`MeasureCollection` 只负责**数据**（增删查 + label 文本生成），不做任何渲染：

```typescript
export interface MeasureItem {
  /** 唯一 id（自动生成，形如 "measure-1"） */
  id: string;
  /** 量测类型 */
  type: MeasureType;                    // 'distance' | 'area' | 'height'
  /** 参与计算的归一化 ECEF 顶点 */
  positions: Cartesian3[];
  /** 量测值：distance → 米，area → m²，height → 米 */
  value: number;
  /** 展示文本，如 "1,234.5 m" / "12.3 km" / "1.2 万 m²" */
  label: string;
}

export class MeasureCollection {
  /** 当前量测条数 */
  get count(): number;

  /** 由类型与顶点计算一条量测（value + label 自动生成）并加入集合 */
  createMeasurement(type: MeasureType, positions: Cartesian3[], clamped = false): MeasureItem;

  /** 直接加入一条量测结果 */
  add(item: MeasureItem): MeasureItem;
  /** 按 id 移除，返回是否移除成功 */
  remove(id: string): boolean;
  /** 按 id 查询 */
  get(id: string): MeasureItem | undefined;
  /** 全部量测（只读视图） */
  getAll(): readonly MeasureItem[];
  /** 清空 */
  clear(): void;

  /** 按类型生成展示文本（area → formatArea，其余 → formatDistance） */
  static formatValue(type: MeasureType, value: number): string;
  /** "1,234.5 m"（< 10000 m）；"12.3 km"（≥ 10000 m） */
  static formatDistance(meters: number): string;
  /** "1.2 万 m²"；≥ 1 亿 m² 时为 "123.4 万 km²" */
  static formatArea(areaM2: number): string;
}
```

典型用法——由顶点一次性生成带文本的量测记录：

```typescript
const collection = new MeasureCollection();

const item = collection.createMeasurement('area', ringPositions);
console.log(item.id);     // "measure-1"
console.log(item.value);  // 12345678.9       （m²，数值）
console.log(item.label);  // "1,234.5 万 m²"  （已格式化的展示文本）

// 文本格式化也可以单独使用（无需集合实例）
MeasureCollection.formatDistance(1234.5);   // "1,234.5 m"
MeasureCollection.formatDistance(12345);    // "12.3 km"
MeasureCollection.formatArea(12345.6);      // "1.2 万 m²"
```

`createMeasurement` 内部按类型分发：

| type | 计算入口 | 顶点要求 |
|---|---|---|
| `'distance'` | `Measure.distance(positions, clamped)` | ≥ 2 |
| `'area'` | `Measure.area(positions)` | ≥ 3 |
| `'height'` | `Measure.height(positions[0], positions[1]).deltaHeight` | 恰好 2（否则抛错） |

## 5. 与 DrawHandler 组合：交互式量测

`DrawHandler`（`src/handler/DrawHandler.ts`）负责把鼠标手势变成归一化 ECEF 顶点，`Measure` 负责算数——两者组合即得交互式量测：

```typescript
import {
  DrawHandler,
  MeasureCollection,
  Entity, Color,
} from './src/index';
import type { DrawEndEvent } from './src/index';

const collection = new MeasureCollection();
const draw = new DrawHandler(viewer);

draw.drawEnd.addListener(({ positions, entity }: DrawEndEvent) => {
  if (entity.polygon) {
    // 面量测：多边形顶点（不含重复闭合点）
    const item = collection.createMeasurement('area', positions);
    entity.label = { text: item.label, fillColor: Color.WHITE };
  } else if (entity.polyline) {
    // 距离量测：折线顶点
    const item = collection.createMeasurement('distance', positions);
    entity.label = { text: item.label, fillColor: Color.WHITE };
  }
});

// 工具栏驱动
document.querySelector('#measure-line')!.addEventListener('click', () => draw.start('polyline'));
document.querySelector('#measure-area')!.addEventListener('click', () => draw.start('polygon'));
document.querySelector('#measure-stop')!.addEventListener('click', () => draw.stop());
```

`DrawHandler` 公开 API 速查（完整手势表见 [interaction.md](./interaction.md)）：

```typescript
export class DrawHandler {
  constructor(viewer: Viewer);

  /** 进入绘制模式时触发，携带新的 DrawType */
  readonly drawStart: Event<[DrawType]>;
  /** 形状完成并加入 scene.entities 时触发 */
  readonly drawEnd: Event<[DrawEndEvent]>;

  /** 当前绘制模式；空闲为 null */
  get activeType(): DrawType | null;

  start(type: DrawType): void;   // 进入（或切换）绘制模式
  stop(): void;                  // 退出绘制模式
  activate(): void;              // 重新进入最近一次使用的模式
  deactivate(): void;            // stop 的别名（EV_DrawHandler 命名）
  destroy(): void;               // 退出并释放 DOM 监听
}

export interface DrawEndEvent {
  /** 完成形状的顶点（归一化 ECEF，不含重复的闭合点） */
  positions: Cartesian3[];
  /** 已加入 scene.entities 的最终实体 */
  entity: Entity;
}
```

两点注意：

- `DrawEndEvent.positions` 与实体顶点是**同一批归一化 ECEF 坐标**，直接喂给 `Measure` / `createMeasurement` 即可，无需换算；
- 高度量测（两点高差）没有对应的绘制类型，推荐流程：先 `draw.start('polyline')` 取两点，在 `drawEnd` 中检查 `positions.length === 2` 后调用 `createMeasurement('height', positions)`。

## 6. 在场景中渲染量测结果

量测的「值」和「文本」是数据，落到画面上有两条路：

**方式 A：挂在 DrawHandler 产出的实体上（推荐）**——如第 5 节示例，`entity.label.text = item.label`，点/标注由 Overlay 屏幕空间路径渲染，随相机会自动深度测试。

**方式 B：为每条量测补独立实体**——适合自建顶点（非绘制流程）的场景：

```typescript
const entities = viewer.scene.entities;

// 量测线
entities.add(new Entity({
  name: 'distance',
  polyline: { positions: item.positions, width: 2, color: Color.YELLOW },
}));

// 中点标注
const mid = Cartesian3.add(item.positions[0], item.positions[1], new Cartesian3());
entities.add(new Entity({
  position: Cartesian3.multiplyByScalar(mid, 0.5, new Cartesian3()),
  label: { text: item.label, fillColor: Color.WHITE },
}));
```

删除一条量测时，记得同时移除集合记录与关联实体：

```typescript
collection.remove(item.id);
entities.remove(labelEntity);
```

## 7. 精度与算法说明

| 量测 | 算法 | 精度特性 |
|---|---|---|
| 距离 | 弦长 × Andoyer-Lambert 椭球测地线近似修正 | 对地表点误差 < 0.1%；同经纬度（地表投影重合）时退化为纯弦长 |
| 面积 | Girard 定理：`面积 = |内角和 − (n−2)π| × R²`，R = 6 371 008.8 m | 顶点先贴地（单位向量）；顺/逆时针均可，结果取绝对值 |
| 高度 | 两点 WGS84 大地高之差 | 精确差值，不做插值 |

实现细节（供交叉验证）：

- 距离修正系数 `k = 地表测地线长 / 地表投影弦长`，按段独立计算后累加；
- 面积顶点角由相邻边大圆法向的叉积求得，邻边共点 / 对跖等退化弧按平角处理；
- 所有方法纯静态、无渲染状态依赖，可在 Node 环境（无 WebGPU）下单测。

## 8. 常见问题

**Q: `Measure.distance` 返回值小得离谱（0.00x 量级）？**
输入是米制 ECEF（`Cartesian3.fromDegrees` 的直接输出）。量测接口期望归一化 ECEF；用第 2 节的 `lonLat` 帮助函数，或先 `Cartesian3.multiplyByScalar(p, 1 / Cartesian3.EARTH_RADIUS, out)` 归一。

**Q: 为什么量测 API 不直接返回实体？**
`Measure` / `MeasureCollection` 被刻意设计为纯数据层（可单测、可接自定义 UI）；渲染样式（线宽、标注位置、颜色）由业务层用 `Entity` 组合，见第 6 节。

**Q: 面积量测能把带高度的顶点算进去吗？**
`Measure.area` 会把顶点贴地（投影到单位椭球面）后按球面多边形计算，顶点高度不参与。如需「表面积」请用其他方式自行建模。

**Q: 高度量测传了 3 个点会怎样？**
`MeasureCollection.createMeasurement('height', positions)` 只取前两点；少于 2 点时抛 `Error`。`Measure.height` 本身固定接受 `p0`、`p1` 两个参数。
