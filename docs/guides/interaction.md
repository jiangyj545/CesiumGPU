# 交互三件套：事件 / 绘制 / 拾取（Interaction）

CesiumGPU 的交互体系由三个模块组成，各司其职：

| 模块 | 一句话职责 | 源码 |
|---|---|---|
| `ScreenSpaceEventHandler` | 把 DOM 鼠标/触摸事件翻译成屏幕空间语义事件 | `src/core/ScreenSpaceEventHandler.ts` |
| `DrawHandler` | 在球面上交互式画点/线/面/矩形，产出 `Entity` | `src/handler/DrawHandler.ts` |
| `PickPass` | GPU 拾取：屏幕坐标 → 命中的图元与地表坐标 | `src/picking/PickPass.ts` |

典型组合：`ScreenSpaceEventHandler` 感知点击 → `scene.pick` 拿到位置 → `DrawHandler` 或业务实体消费结果。

---

## 目录

1. [ScreenSpaceEventHandler](#1-screenspaceeventhandler)
2. [DrawHandler：球面绘制](#2-drawhandler球面绘制)
3. [PickPass：GPU 拾取](#3-pickpassgpu-拾取)
4. [三件套组合实战](#4-三件套组合实战)
5. [与相机输入的共存](#5-与相机输入的共存)
6. [常见问题](#6-常见问题)

---

## 1. ScreenSpaceEventHandler

### 创建

```typescript
import { ScreenSpaceEventHandler } from '../src/index';

// 绑定到任意 canvas
const handler = new ScreenSpaceEventHandler(viewer.canvas);

// Viewer 已内置一个（相机控制共用），可直接复用：
const builtin = viewer.screenSpaceEventHandler; // ScreenSpaceEventHandler | null
```

### 事件类型全表

```typescript
export type ScreenSpaceEventType =
  | 'LEFT_CLICK'
  | 'LEFT_DOUBLE_CLICK'   // 两次 CLICK 间隔 < 500 ms 时额外发出
  | 'RIGHT_CLICK'
  | 'MOUSE_MOVE'
  | 'LEFT_DOWN'
  | 'LEFT_UP'
  | 'RIGHT_DOWN'
  | 'RIGHT_UP'
  | 'MIDDLE_DOWN'
  | 'MIDDLE_UP'
  | 'WHEEL'
  | 'PINCH';
```

### 注册 / 注销

`setInputAction` 按事件类型重载了回调签名：

```typescript
class ScreenSpaceEventHandler {
  constructor(canvas: HTMLCanvasElement);

  // 位置类事件：ScreenSpacePosition
  setInputAction(action: (position: ScreenSpacePosition) => void,
                 type: ScreenSpaceEventType): void;
  // 滚轮：WheelPosition
  setInputAction(action: (position: WheelPosition) => void, type: 'WHEEL'): void;
  // 双指：PinchPosition
  setInputAction(action: (position: PinchPosition) => void, type: 'PINCH'): void;

  removeInputAction(type: ScreenSpaceEventType): void;
  destroy(): void;   // 解绑全部 DOM 监听并清空注册表
}
```

三种载荷（坐标均为**画布 CSS 像素，左上角原点**）：

```typescript
interface ScreenSpacePosition { x: number; y: number; }
interface WheelPosition extends ScreenSpacePosition { deltaY: number; }
interface PinchPosition  extends ScreenSpacePosition { /* 双指中心与间距 */ }
```

### 示例：点击 + 悬停 + 缩放

```typescript
handler.setInputAction((pos) => {
  console.log('点击于', pos.x, pos.y);
}, 'LEFT_CLICK');

handler.setInputAction((pos) => {
  console.log('移动到', pos.x, pos.y);
}, 'MOUSE_MOVE');

handler.setInputAction((pos) => {
  console.log('滚轮 delta', pos.deltaY);
}, 'WHEEL');

// 注销
handler.removeInputAction('MOUSE_MOVE');
```

> 双击检测内建：500 ms 内的两次按下/抬起会被追加为 `LEFT_DOUBLE_CLICK`，无需自己计数。

---

## 2. DrawHandler：球面绘制

`DrawHandler` 在**专用输入通道**上工作（占用期间相机的指针拖拽被抑制，滚轮缩放保留），把点击落到地表（先 `scene.pick`，失败回退相机射线-椭球求交），绘制完成后生成 `Entity` 自动加入 `viewer.scene.entities`。

### 支持的类型与交互手势

```typescript
export type DrawType = 'point' | 'polyline' | 'polygon' | 'rectangle';
```

| 类型 | 手势 |
|---|---|
| `point` | 单击放置一个点，立即完成 |
| `polyline` | 左键加点；**右键 / 双击**结束（少于 2 点则取消） |
| `polygon` | 左键加点；**右键 / 双击**结束（少于 3 点则取消），自动闭合 |
| `rectangle` | 两次左键定对角；拖动过程实时预览 |

绘制过程中始终有**预览实体**（草稿线/面）跟随鼠标。

### API

```typescript
export class DrawHandler {
  constructor(viewer: Viewer);

  /** 进入绘制模式时触发，携带新的 DrawType。 */
  readonly drawStart: Event<[DrawType]>;
  /** 形状完成并加入 scene.entities 时触发。 */
  readonly drawEnd: Event<[DrawEndEvent]>;

  get activeType(): DrawType | null;  // 当前绘制模式；空闲为 null

  start(type: DrawType): void;   // 进入（或切换）绘制模式
  stop(): void;                  // 退出绘制模式
  activate(): void;              // 重新进入最近一次使用的模式
  deactivate(): void;            // stop 的别名（EV_DrawHandler 命名）
  destroy(): void;               // 退出并释放 DOM 监听
}

export interface DrawEndEvent {
  /** 完成形状的顶点（归一化 ECEF，不含重复的闭合点）。 */
  positions: Cartesian3[];
  /** 已加入 scene.entities 的最终实体。 */
  entity: Entity;
}
```

### 完整示例

```typescript
import { DrawHandler, Measure, MeasureCollection } from '../src/index';
import type { DrawEndEvent } from '../src/index';

const draw = new DrawHandler(viewer);

draw.drawEnd.addListener(({ positions, entity }: DrawEndEvent) => {
  // positions: 归一化 ECEF 顶点；entity: 已入库的实体
  if (entity.polygon) {
    const area = Measure.area(positions);           // m²
    console.log(`${entity.name} 面积 ${MeasureCollection.formatArea(area)}`);
  } else if (entity.polyline) {
    const len = Measure.distance(positions);        // m
    console.log(`${entity.name} 长度 ${MeasureCollection.formatDistance(len)}`);
  }
});

// 按钮驱动
document.querySelector('#draw-line')!.addEventListener('click', () => draw.start('polyline'));
document.querySelector('#draw-stop')!.addEventListener('click', () => draw.stop());
```

> `draw.start()` 内部会重置上一轮草稿；切换类型直接再调 `start(newType)` 即可。

---

## 3. PickPass：GPU 拾取

### 原理（详见 ADR-0002）

1. 每个可拾取图元分配一个 **24-bit pick id**，写入顶点属性；
2. 拾取时把相机切到"拾取视图"，把场景以 id 编码渲染进 **1×1 rgba8unorm** 目标；
3. `copyTextureToBuffer` + `mapAsync` 读回 4 字节，解码为 id；
4. id → `Primitive` 映射（每次 encode 时重建），背景/未命中 id = 0。

实测性能（50 次连续拾取）：平均 **6.0 ms/次**（wall），readback 平均 2.14 ms——帧帧出结果，无异步批量延迟。

### 业务层 API：一行 `scene.pick`

`PickPass` 由 `Scene` 在**首次 pick 时懒创建**，业务层推荐只调：

```typescript
// x/y 为画布设备像素（左上原点，与 DOM clientX/Y 同向，无需 y 翻转）
const result = await viewer.scene.pick(x, y);
// 场景未初始化 / 已销毁时返回 null
```

`PickResult` 结构：

```typescript
interface PickResult {
  id: number;                        // 24-bit；0 = 背景/未命中
  primitive: Primitive | null;       // 命中的图元（含瓦片、GLB 子图元）
  worldPosition: Cartesian3 | null;  // 命中点世界坐标（归一化 ECEF）
  timing: {
    encodeMs: number;    // 拾取视图编码耗时
    readbackMs: number;  // submit → mapAsync 的 GPU 回读延迟
    totalMs: number;
  };
}
```

### 典型用法：点击显示经纬度

```typescript
handler.setInputAction(async (pos) => {
  const hit = await viewer.scene.pick(pos.x, pos.y);
  if (!hit || hit.id === 0) return;              // 背景或未初始化

  if (hit.worldPosition) {
    const carto = Cartographic.fromCartesian(hit.worldPosition);
    console.log(
      `命中 ${hit.primitive?.constructor.name}`,
      `经度 ${CesiumMath.toDegrees(carto.longitude).toFixed(4)}°`,
      `纬度 ${CesiumMath.toDegrees(carto.latitude).toFixed(4)}°`,
    );
  }
}, 'LEFT_CLICK');
```

### 高级：场景图之外的图元

不经过 `scene.addPrimitive` 的图元（自定义管线绘制）需要手动注册：

```typescript
// 通过 scene 内部 PickPass（惰性创建；首次 pick 后可用）
const pickPass = viewer.scene['pickPass'];
const pickId = pickPass.register(myPrimitive);   // 返回 24-bit id
pickPass.unregister(myPrimitive);
pickPass.invalidate();                            // 手动改注册表后清缓存
```

诊断信息（只读）：

```typescript
pickPass.debug;  // { hookCalls, encodes, flushes, gpuErrors }
```

---

## 4. 三件套组合实战

需求：**框选统计**——用 `rectangle` 绘制框选区域，完成后输出角点经纬度：

```typescript
import { DrawHandler, Cartographic, CesiumMath } from '../src/index';

const draw = new DrawHandler(viewer);

draw.drawStart.addListener((type) => {
  document.body.style.cursor = type ? 'crosshair' : 'default';
});

draw.drawEnd.addListener(({ positions, entity }) => {
  document.body.style.cursor = 'default';
  if (entity.polygon && positions.length === 4 + 1) {   // rectangle 自动闭合
    const [nw, ne, se, sw] = positions;
    for (const p of [nw, ne, se, sw]) {
      const c = Cartographic.fromCartesian(p);
      console.log(
        `${CesiumMath.toDegrees(c.longitude).toFixed(4)}, ` +
        `${CesiumMath.toDegrees(c.latitude).toFixed(4)}`
      );
    }
  }
});

// 点击工具栏启动框选
startRectButton.addEventListener('click', () => draw.start('rectangle'));
```

注意 `DrawHandler` 内部同样调用 `scene.pick` 获取地表点——三件套在引擎内部就是这么组合的。

---

## 5. 与相机输入的共存

- **空闲态**：左键拖拽 = 环绕旋转，滚轮 = 缩放。
- **绘制态**（`draw.start` 后）：指针事件进入绘制专用通道，**相机拖拽被抑制**，滚轮/双指缩放保留——用户可以在绘制中继续拉近观察。
- **飞行中**：任何用户输入（拖拽/滚轮/双指）或新的 `flyTo` 都会取消当前飞行并触发其 `cancel` 回调。

---

## 6. 常见问题

**Q: `pick` 返回 `null` 而不是 `PickResult`？**
场景尚未 `initialize()` 完成或已 `destroy()`。null 与"id=0 的未命中"是两种语义。

**Q: 拾取命中了瓦片，怎么区分业务图元和地球瓦片？**
`result.primitive` 引用比对：业务图元保存你 `new Primitive(...)` 时的引用；瓦片图元在 `viewer.globe.tilePrimitives` 里。

**Q: 为什么绘制时相机不能拖拽？**
设计行为——绘制模式独占指针事件以避免误操作。需要临时拖拽时先 `draw.stop()`，或用 `deactivate()`/`activate()` 挂起/恢复。

**Q: `LEFT_DOUBLE_CLICK` 不触发？**
确认两次点击都在 500 ms 内且位置几乎不动；快速连点画线时会被识别为双击——这是预期行为，画线结束请用右键。

**Q: 拾取坐标是归一化的，怎么转经纬度？**
`Cartographic.fromCartesian(p)`（内部按米制换算），再用 `CesiumMath.toDegrees`。

---

## 相关文档

- [ADR-0002 GPU 拾取方案](../decisions/0002-gpu-picking-readback.md) — 1×1 readback、24-bit id 编码与精度验证的完整推导
- [量测分析](./analysis.md) — DrawHandler + Measure 的面积/距离量测
- [Entity 数据模型](./entities.md) — DrawHandler 产出的 Entity 去了哪里
- [API 参考](../api/index.html)
