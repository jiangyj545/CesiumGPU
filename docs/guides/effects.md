# 特效系统（Effects）

`IPass` 统一抽象、`EffectManager` 挂载点，以及动态墙 / 圆形扫描 / 粒子系统 / 天气四类内置特效，最后给出自定义特效 pass 的完整指南。

## 目录

1. [30 秒上手](#1-30-秒上手)
2. [IPass：统一渲染 Pass 抽象](#2-ipass统一渲染-pass-抽象)
3. [EffectManager：特效挂载点](#3-effectmanager特效挂载点)
4. [DynamicWall：动态墙](#4-dynamicwall动态墙)
5. [CircleScan：地表圆形扫描](#5-circlescan地表圆形扫描)
6. [ParticleSystem：GPU 计算粒子](#6-particlesystemgpu-计算粒子)
7. [Weather：雨雪天气](#7-weather雨雪天气)
8. [unlit 管线：自发光几何体](#8-unlit-管线自发光几何体)
9. [自定义特效 Pass 指南](#9-自定义特效-pass-指南)
10. [常见问题](#10-常见问题)

## 相关文档

- [引擎架构](./architecture.md) — 特效在 4-pass 管线中的位置
- [交互三件套](./interaction.md) — 用拾取/绘制驱动特效位置
- [ADR-0003 粒子系统方案](../decisions/0003-compute-particles.md) — compute shader 选型依据
- [API 参考](../api/index.html) — `EffectManager` / 四类特效完整签名

---

## 1. 30 秒上手

所有特效实现 `IPass` 接口，统一挂在 `viewer.scene.effects`（一个 `EffectManager`）上：

```typescript
import { CircleScan } from './src/index';

const scan = new CircleScan(viewer.context, {
  center: { lon: 116.4, lat: 39.9 },
  radius: 300_000,
  color: [0.2, 0.9, 1.0, 0.85],
  speed: 2.0,
  rings: 3,
});
scan.setCamera(viewer.camera);       // 特效需要相机做投影
viewer.scene.effects.add(scan);      // 挂载后每帧自动执行

scan.show = false;                   // 隐藏（不销毁）
viewer.scene.effects.remove(scan);   // 从挂载点移除（不销毁）
scan.destroy();                      // 释放 GPU 资源
```

执行时机：`Scene.render()` 在 **Transparent composite 之后**逐个执行特效——特效多为叠加发光（emissive / additive），画在最后不被后续 pass 覆盖。

## 2. IPass：统一渲染 Pass 抽象

`IPass`（`src/scene/passes/IPass.ts`）是所有新增特效 pass 必须实现的接口：

```typescript
export interface IPass {
  /** 调试标签（也用于 RenderPass descriptor 的 label）。 */
  readonly label: string;

  /** 幂等初始化（管线 / bindGroup）。可安全重复调用。 */
  prepare(): void;

  /**
   * 执行单个 pass。
   * @param encoder - 当前帧共享的 command encoder
   * @param target  - 本 pass 的颜色输出（通常为 swap-chain view）
   * @param time    - 统一效果时间（秒，performance.now() / 1000），供动画特效使用
   */
  execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void;

  /** 画布尺寸变化时调用（可选）。 */
  resize?(width: number, height: number): void;

  /** 释放 GPU 资源。调用后本 pass 不可再用。 */
  destroy(): void;
}
```

既有管线 pass 的迁移状态（接口文档中的原表）：

| Pass | 状态 |
|---|---|
| `OverlayPass` | 已实现 `IPass` |
| `GBufferPass` | future — 资源持有者，`beginRenderPass` 由 `Scene.render()` 驱动，无独立 execute |
| `LightingPass` | future — 签名接近但缺 label/prepare/resize，深度耦合 `Scene.render()` |
| `TransparentPass` | future — 双子 pass（accumulation + composite）拆分执行 |

特效层的 `Effect` 类型在 `IPass` 之上加了可见性开关：

```typescript
/** 特效的完整形状：一个 IPass 加上可见性开关。 */
export type Effect = IPass & { show: boolean };
```

四类内置特效都有 `show = true` 公开字段，`EffectManager` 依据它跳过 `execute`。

## 3. EffectManager：特效挂载点

`EffectManager`（`src/effect/EffectManager.ts`）：

```typescript
export class EffectManager {
  /** 挂载一个特效。不检查重复；同一实例重复 add 会执行两次。 */
  add(effect: Effect): Effect;

  /** 移除一个特效（不 destroy，生命周期归调用方）。 */
  remove(effect: Effect): boolean;

  /** 当前挂载的特效列表（只读视图）。 */
  get list(): readonly Effect[];

  /** 在 Transparent composite 之后执行所有可见特效。 */
  execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void;

  /** 转发画布 resize 给所有支持 resize 的特效。 */
  resize(width: number, height: number): void;

  /** 销毁所有特效并清空列表。 */
  destroy(): void;
}
```

生命周期约定：

- `add` / `remove` 只管**挂载**；`destroy()` 由**调用方**负责（`remove` 不会释放 GPU 资源）；
- `viewer.destroy()` → `scene.destroy()` → `effects.destroy()` 会统一销毁所有特效；
- 时间统一由 Scene 传入（秒），特效自身不做时钟管理。

## 4. DynamicWall：动态墙

将地表路径挤出为垂直墙体，片元级渲染「底部亮、向上渐隐、横向光带从底往上涌」（T3.2，对齐 EV-Globe 的 `EV_DynamicWall`）。

```typescript
export interface DynamicWallOptions {
  /** 底边路径点，归一化 ECEF（地表，Earth radius = 1）。至少 2 个点。 */
  positions: Cartesian3[];
  /** 墙高，归一化单位（1 = 地球半径 ≈ 6378 km）。Default: 0.02（≈ 127 km） */
  height?: number;
  /** 墙体颜色 RGBA。Default: `[0.25, 0.75, 1.0, 0.9]`（青蓝色） */
  color?: [number, number, number, number];
  /** 流速（光带上涌速度倍率）。Default: 1.0 */
  speed?: number;
  /** u 沿路径累积长度的重复倍数。Default: 3 */
  repeat?: number;
  /**
   * 相机引用（必填）：墙体每帧自行从该相机读取 viewProj / camPos 上传 UBO。
   * 通常传 `viewer.scene.camera`。
   */
  camera: Camera;
}
```

用法：

```typescript
import { DynamicWall, Cartesian3 } from './src/index';

const wall = viewer.scene.effects.add(new DynamicWall(viewer.scene.context, {
  positions: [                          // 归一化 ECEF（地球半径 = 1）
    new Cartesian3(-0.728, 0.227, 0.648),
    new Cartesian3(-0.700, 0.180, 0.692),
    new Cartesian3(-0.660, 0.150, 0.735),
  ],
  height: 0.02,                         // ≈ 127 km
  color: [0.25, 0.75, 1.0, 0.9],
  speed: 1.0,
  repeat: 3,
  camera: viewer.scene.camera,          // 必填
}));

wall.show = false;  // 关闭
wall.destroy();     // 用完释放
```

实现要点：

- 几何为「底排贴地 + 顶排沿地表法线抬高 `height`」的三角带，每段 4 顶点 2 三角；uv 的 u = 累积线长 × repeat、v 底 0 / 顶 1；
- 绘制在 swap-chain 上，混合为 **src-alpha / one 的叠加发光**；
- 因 `IPass.execute` 无深度附件，片元内用**射线-单位球求交**剔除被地球遮挡的远侧墙体；
- 相机不经过 `EffectManager` 下发——构造时必须传 `camera`，每帧自行读取 `viewProj / camPos` 上传 UBO。

## 5. CircleScan：地表圆形扫描

在以 `center` 为圆心、`radius`（米）为半径的地表圆盘上绘制从中心周期性向外扩散的扫描波前（同心 `rings` 圈，每 `speed` 秒扫完一轮），附加中心恒亮光斑（T3.3，对齐 `EV_CircleScan`）。

```typescript
export interface CircleScanOptions {
  /** 圆心经纬度（度）。 */
  center: { lon: number; lat: number };
  /** 扫描半径（米）。默认 100_000。 */
  radius?: number;
  /** 扫描颜色（线性 RGBA，a 为基准不透明度）。默认青蓝色。 */
  color?: [number, number, number, number];
  /** 扫描周期（秒/轮）。默认 2.0。 */
  speed?: number;
  /** 同心环数（同时可见的波前圈数）。默认 3。 */
  rings?: number;
}
```

用法：

```typescript
const scan = new CircleScan(viewer.context, {
  center: { lon: 116.4, lat: 39.9 },
  radius: 300_000,
  color: [0.2, 0.9, 1.0, 0.85],
  speed: 2.0,
  rings: 3,
});
scan.setCamera(viewer.camera);       // 挂载后调用一次即可（viewProj 每帧重传）
viewer.scene.effects.add(scan);
```

实现要点：

- 圆盘为极坐标网格（径向 32 × 周向 64），逐顶点投影到 WGS84 椭球面——大半径下依然贴合地表、无悬浮；
- **additive（one, one）混合**直接画到 swap-chain，无深度测试；
- 相机通过 `setCamera(camera)` 提供（与 DynamicWall 的构造参数方式不同）。

## 6. ParticleSystem：GPU 计算粒子

compute pass 就地读写单 storage buffer（32 B/粒子），渲染为 instanced screen-aligned billboard，**加法混合 + G-Buffer 深度测试**（地球背面的粒子被遮挡不再透出）（T3.4，对齐 `EV_ParticleSystem`，选型见 [ADR-0003](../decisions/0003-compute-particles.md)）。

```typescript
export type ParticleMode = 'fire' | 'fountain' | 'fireworks' | 'smoke';

/** 发射器位置：米制 ECEF `Cartesian3`，或 `{ lon, lat, height }`（度 / 米）。 */
export type ParticleOrigin = Cartesian3 | { lon: number; lat: number; height?: number };

export interface ParticleOptions {
  /** 模式预设。默认 `'fire'`。 */
  mode?: ParticleMode;
  /** 粒子池容量（向上取整到 64 的倍数）。默认 10000。 */
  count?: number;
  /** 发射器位置。 */
  origin: ParticleOrigin;
  /** 发射速率（粒子/秒）。默认不限（count 全量激活）。 */
  emissionRate?: number;
  /** 所属场景：提供相机（billboard 投影）与 G-Buffer 深度（遮挡剔除）。 */
  scene: Scene;
}
```

用法：

```typescript
import { ParticleSystem } from './src/index';

const ps = new ParticleSystem(viewer.context, {
  scene: viewer.scene,             // 必须已 initialize（依赖 gbufferPass 深度）
  mode: 'fire',                    // 'fire' | 'fountain' | 'fireworks' | 'smoke'
  count: 10_000,                   // 粒子池容量
  origin: { lon: 116.391, lat: 39.907, height: 0 },
  emissionRate: 2000,              // 粒子/秒（省略 = count 全量激活）
});
viewer.scene.effects.add(ps);

ps.setOrigin(newOrigin);           // 动态移动发射器，已发射粒子保持原轨迹
console.log(ps.mode, ps.count);    // 反查当前模式 / 池容量
ps.destroy();                      // 释放全部 GPU buffer
```

四类模式预设（`ParticleSystem.presets`，静态可查）：

| mode | 重力 | 寿命 (s) | 形状 | 色带 | 视觉 |
|---|---|---|---|---|---|
| `fire` | 0.08 | 1.2–2.0 | cone | fire | 向上锥形喷发，白黄→橙红，强重力回落 |
| `fountain` | 0.06 | 1.8–2.5 | cone | water | 宽锥高抛物线，水蓝白 |
| `fireworks` | 0.035 | 1.5–2.6 | sphere（锥角 π） | multi | 爆点球形四散，多彩拖尾 |
| `smoke` | 0 | 3.5–6.0 | cone | smoke | 近零初速慢速上升，大尺寸灰白低 alpha |

实现细节：

- `emissionRate` 遵循 Little 定律预算：`rate × 平均寿命 ≈ 稳态活跃数`，封顶 `count`；
- 模拟与场景共用同一个 command encoder；`execute(…, time)` 内部由相邻帧差分得 `dt`，**clamp 到 0.1 s** 防止切后台返回时爆模拟；
- 深度接入：RenderPass 传 `depthView` + `depthReadOnly: true`（此时不可设 depthLoadOp/StoreOp），pipeline 层 `depthCompare: 'less'` 且不写深度；billboard 整体取粒子中心点投影 z 参与测试。

## 7. Weather：雨雪天气

相机锚定的雨/雪体积（T3.5，对齐 `EV_Rain / EV_Snow`）：静态种子点云完全在顶点着色器中动画（`p = boxOrigin + mod(seed*boxSize + velocity*t, boxSize)`），体积盒每帧平移到 `camPos + forward · depth/2`，雨雪因此**跟随相机而无需任何 CPU 逐粒子工作**。遮挡靠对 G-Buffer 深度做 GPU 深度测试。

```typescript
export type WeatherType = 'rain' | 'snow';

/** 天气体积尺寸（米），居中于相机前方。 */
export interface WeatherArea {
  width: number;    // 相机 right 轴向 extent
  height: number;   // 相机 up 轴向 extent
  depth: number;    // 相机 forward 轴向 extent
}

// 构造（options 含 type / intensity? / area? / sizeScale? / wind?）
new Weather(viewer.context, {
  type: 'rain',            // 'rain' | 'snow'
  intensity: 0.6,          // 密度 0..1，默认 1.0（clamp01）
  area: { width: 800, height: 600, depth: 800 },   // 米，默认 800/600/800
  sizeScale: 1.0,          // 粒子尺寸倍率，默认 1.0
  wind: 2.0,               // 风速 m/s；默认 rain 2.0 / snow 0.8
});
```

关键方法：

```typescript
weather.setCamera(viewer.camera);                                   // 必调（每帧在 execute 中重传投影）
weather.setDepthView(viewer.scene.gbufferPass!.textures.depthView); // 地形遮挡；resize 后需重设
weather.setIntensity(0.3);                                          // 运行时调密度
weather.show = false;                                               // 停止
weather.destroy();
```

雨/雪的差异：

| | rain | snow |
|---|---|---|
| 1 实例 | 1 条沿下落速度方向拉长的屏幕朝向四边形（长 = speed × 0.06 s） | 1 个带正弦摆动的软边圆形 billboard |
| 混合 | additive，alpha 0.35 | 标准 alpha 混合，alpha 0.8 |
| 默认风速 | 2.0 m/s | 0.8 m/s |

`setDepthView` 注意：canvas resize 后 G-Buffer 会重建纹理，需重新设置（官方注释建议在每帧 `preRender` 钩子中读取 `scene.gbufferPass!.textures.depthView`）。**未提供 depth view 时绘制被跳过**（仅告警一次）。

## 8. unlit 管线：自发光几何体

特效之外，`Primitive` 还有一条 **unlit 管线**用于「不受光照」的自发光几何体（如霓虹描边、光柱），在 G-Buffer pass 内切换：

```typescript
const prim = new Primitive(context, {
  vertices,
  indices,
  color: new Color(1.0, 0.6, 0.1, 1.0),
  unlit: true,        // ← 关键：走 unlit 管线，跳过延迟光照
});
scene.addOpaquePrimitive(prim);
```

渲染编排（`Scene.render()` G-Buffer 段摘录）：

```typescript
// 其他不透明图元：emissive 几何切换到 unlit 管线
// （两种管线的 bind group 布局兼容，无需重建资源）
const opaquePipeline = this._opaquePipeline;
const unlitPipeline  = this._unlitPipeline;
let currentPipeline  = opaquePipeline;
for (const prim of this.primitives.opaque) {
  const wanted = prim.unlit && unlitPipeline ? unlitPipeline : opaquePipeline;
  if (wanted !== currentPipeline) {
    pass.setPipeline(wanted);
    currentPipeline = wanted;
  }
  prim.drawOpaque(pass, this._cameraBindGroup);
}
```

选型对照：

| 需求 | 用法 |
|---|---|
| 场景中的自发光几何（描边、光柱、标记牌面片） | `Primitive` + `unlit: true`（进 G-Buffer，受深度排序） |
| 叠加发光的屏幕特效（墙、扫描、粒子、天气） | 实现 `IPass` 挂 `scene.effects`（composite 后画，叠加混合） |

## 9. 自定义特效 Pass 指南

实现一个最小可用的 `IPass` 特效（全屏叠加渐晕示例）：

```typescript
import type { IPass } from './src/index';
import { Context } from './src/index';

const WGSL = /* wgsl */ `
@group(0) @binding(0) var<uniform> params: vec4<f32>;   // (color.rgb, strength)
@group(0) @binding(1) var tSampler: sampler;
@group(0) @binding(2) var tInput: texture_2d<f32>;

struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex fn vs(@builtin(vertex_index) i: u32) -> VsOut {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
  var out: VsOut;
  out.pos = vec4(p[i], 0.0, 1.0);
  out.uv  = p[i] * 0.5 + 0.5;
  return out;
}

@fragment fn fs(in: VsOut) -> @location(0) vec4<f32> {
  let scene = textureSample(tInput, tSampler, in.uv);
  let d = distance(in.uv, vec2(0.5));
  let vignette = 1.0 - params.w * smoothstep(0.35, 0.85, d);
  return vec4(scene.rgb * vignette, 1.0);
}
`;

export class VignetteEffect implements IPass {
  readonly label = 'VignetteEffect';
  show = true;                       // EffectManager 依据它跳过 execute
  private _context: Context;
  private _pipeline: GPURenderPipeline | null = null;
  private _bg: GPUBindGroup | null = null;
  private _ubo: GPUBuffer | null = null;

  constructor(context: Context, color: [number, number, number], strength: number) {
    this._context = context;
    this._ubo = context.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // ... 创建 shader module / pipeline / bindGroup（见下文约定）
  }

  /** 幂等初始化；可安全重复调用 */
  prepare(): void { /* 管线/bindGroup 创建放这里或构造函数 */ }

  execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void {
    if (!this._pipeline) return;
    // ① 共享 encoder：不要 submit！只 record render pass
    const pass = encoder.beginRenderPass({
      label: this.label,
      colorAttachments: [{
        view: target,                                  // ② 直接画到 swap-chain view
        loadOp: 'load',                                // ③ load：叠加在已有画面上
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, this._bg!);
    pass.draw(3);
    pass.end();
  }

  resize(width: number, height: number): void { /* 重建尺寸相关资源 */ }

  destroy(): void {
    this._ubo?.destroy();
    this._pipeline = null;
  }
}

// 挂载
viewer.scene.effects.add(new VignetteEffect(viewer.context, [1, 0.5, 0], 0.4));
```

约定与坑（全部来自内置特效的实现经验）：

1. **共享 command encoder**——`execute` 收到的 `encoder` 与场景各 pass 共用，绝不在此 `submit()`，只 record render pass；
2. **`loadOp: 'load'`**——特效画在 swap-chain 已有画面（光照/透明合成结果）之上，clear 会清空整帧；
3. **深度不可用**——`IPass.execute` 没有深度附件。需要地球遮挡剔除时，参考 `ParticleSystem` 把 `scene.gbufferPass.textures.depthView` 作为 `depthReadOnly: true` 的 depth attachment（此时不可设 depthLoadOp/depthStoreOp），或像 `DynamicWall` 在片元内做射线-单位球求交；
4. **相机自行获取**——`EffectManager` 不向特效推送相机；构造时收 `Camera` 引用（DynamicWall / Weather / ParticleSystem 方式）或提供 `setCamera()`（CircleScan 方式），每帧在 `execute` 内读取 `viewProj` 上传；
5. **时间只用 `time` 参数**——秒；需要 dt 就像 `ParticleSystem` 一样做帧间差分并 clamp（≤ 0.1 s，防切后台）；
6. **`show` 字段必填**——`Effect = IPass & { show: boolean }`，类型上即要求此字段；
7. **resize 与 destroy 对称**——canvas 缩放后 `EffectManager` 会转发 `resize`（可选实现）；`destroy` 后实例不可再用；
8. **`prepare()` 幂等**——管线/bindGroup 创建放这里可安全重复调用；`Weather` / `ParticleSystem` 选择在构造函数里建好（要求 `Context.initialize()` 已完成，构造时可校验 `context.initialized`，`ParticleSystem` 还会校验 `scene.gbufferPass` 存在）。

## 10. 常见问题

**Q: 特效画上去整个画面变黑/被清空？**
RenderPass 的 `loadOp` 写成了 `'clear'`。必须用 `'load'` 保留前一 pass 的合成结果。

**Q: 特效不动（静止画面）？**
`time` 参数来自 `Scene.render()` 的 `performance.now() / 1000`，确认没有绕过 `execute(…, time)` 自己缓存了时间；另检查 `show` 是否为 `true`。

**Q: 同一个特效 add 了两次？**
`EffectManager.add` 不查重——重复 add 会每帧执行两次（叠加发光加倍）。用 `list` 检查后再挂载。

**Q: 天气/粒子 resize 后消失或深度错乱？**
G-Buffer 纹理在 resize 时重建，`Weather.setDepthView` 需要重设（建议每帧 `preRender` 里重读 `scene.gbufferPass!.textures.depthView`）；`ParticleSystem` 每帧从 `options.scene` 现取深度，无需处理。

**Q: 为什么 DynamicWall 要在构造时传 camera，CircleScan 却用 setCamera？**
历史接口差异，二者语义等价（每帧重传 viewProj）。自定义特效任选其一并保持一致即可，参考第 9 节约定第 4 条。
