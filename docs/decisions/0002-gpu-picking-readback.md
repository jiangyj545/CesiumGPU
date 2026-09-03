# 0002 — GPU 拾取回读技术验证（spike）结论

- **状态**: Accepted (spike 已验证，正式版已落地，见文末「正式版」小节)
- **日期**: 2026-09-02
- **关联代码**: `src/picking/PickPass.ts`（spike 版 `PickRenderer.ts` 已删除，由 PickPass 接替）、`src/shaders/PickShader.ts`、`src/scene/Scene.ts`（`frameHooks` / `gbufferPass` / `opaquePipeline` 访问器）、`src/scene/passes/GBufferPass.ts`（worldPos 增加 `COPY_SRC`）
- **验证页**: `public/spikes/pick-spike.html`（逻辑在 `src/spikes/pick-spike.ts`，结果镜像到 `window.__pickSpike`）

## 背景

为拾取（点击选中对象）做技术选型验证：用 id-encode shader 将 24-bit 对象 id 渲入
rgba8unorm 拾取纹理，再经 `copyTextureToBuffer` + `mapAsync` 回读 1×1 像素解码；
worldPosition 从 G-Buffer 的 worldPos（rgba32float）同像素回读。

## 实测数据

环境：Windows 11 + Chrome (Dawn)，视口 1280×800（canvas 2558×998，DPR 1.998），
场景 = 地球（球面网格）+ cone4-red.glb（1 mesh primitive），相机高度 20 000 km。

| 指标 | 数值 |
| --- | --- |
| 单次 pick 端到端延迟（submit → mapAsync resolve） | 首次 6.5 ms；稳态平均 **2.28 ms**（min 0.8 / max 5.1 ms） |
| 单次 pick 调用到拿到结果（含等下一帧 hook） | ≈ 8.7–12 ms（≈ 1 帧） |
| 连续 100 次 pick 总耗时 | **606 ms**（6.06 ms/次，即每帧 1 次、帧帧有结果） |
| id 精度（0x123456 写入 → 回读） | **无损**：decodedId === 0x123456 |
| 命中验证 | 模型像素 id=2 ✓；背景像素 id=0 ✓ |
| GPU worldPos vs CPU raycast（同像素） | δ ≈ 10.1 km（该视角 ≈8 km/px，即 ≈1.3 像素，属像素量化误差） |
| GPU uncaptured error | 0（修复深度拷贝问题后） |

## 结论 1：mapAsync 延迟是否 > 1 帧

**否。** 稳态回读延迟 0.8–5.1 ms（均值 2.3 ms），小于一个 60 Hz 帧（16.7 ms）。
延迟构成 = CPU→GPU 提交 + 拾取 pass 执行 + 信号量回传，1×1 × 256 B 的拷贝本身
开销可忽略；大头是拾取 pass 重绘场景（本 spike 全场景仅 2 个 draw call）。
用户可感知的延迟主要是"下一帧才编码拾取请求"（≤ 1 帧），总体 call→result ≤ 2 帧。

## 结论 2：rgba8unorm 24-bit id 精度

**无损，方案可用。** 编码 `r=id&255, g=(id>>8)&255, b=(id>>16)&255, a=255`，
经 fragment 输出 → rgba8unorm 渲染目标 → `copyTextureToBuffer` → Uint8Array 解码，
0x123456 往返完全一致。unorm 量化 `round(v/255*255)` 在整数字节值上不引入误差。
注意 id=0 必须保留给背景（clearValue 即 0），id 分配从 1 开始。

## 结论 3：事件队列模式（每帧最多 1 次回读）是否够用

**够用，且是必要约束。** 实现方式：PickRenderer（spike 版，现已删除；该实现模式由正式版 `src/picking/PickPass.ts` 接替沿用）向 `scene.frameHooks` 注册钩子，
在帧 encoder submit 前追加拾取 pass + 拷贝命令；`queueMicrotask` 延迟到 submit 之后
再 `mapAsync`（WebGPU 禁止提交引用已映射 buffer 的命令缓冲）。in-flight 期间
`pick()` 复用同一个 pending promise（实测 5 次并发调用 → 1 个 promise）。每帧最多
一次回读使 GPU 端拾取 pass 频率封顶为帧率，连续 100 次拾取仍能帧帧出结果。

## 结论 4：worldPos 纹理回读 vs 深度反投影

**选 worldPos（rgba32float）1×1 回读。关键发现：深度反投影方案在本引擎不可行——
WebGPU/Dawn 禁止对 depth/stencil 格式做部分（子区域）拷贝**，报错
`The entire subresource must be copied when the format (Depth32Float) is a
depth/stencil format`；即 1×1 深度回读必须拷贝整张 2558×998×4B ≈ 10 MB，每帧一次
完全不可接受（本 spike 初版即因此令整条命令缓冲失效、所有回读读到旧数据）。

worldPos 方案对比优势：
- 1×1 rgba32float = 16 B 回读，CPU 端零反投影计算（无需 CPU 侧逆矩阵）；
- 与像素严格一致（拾取到哪个三角形就是哪个世界坐标），实测与 CPU raycast 误差
  ≈1.3 像素（像素量化级别）；
- 代价仅是 G-Buffer worldPos 纹理多一个 `COPY_SRC` usage 位。

## 实现要点 / 坑

1. `copyTextureToBuffer` 的 `bytesPerRow` 必须 256 对齐 → 回读 buffer 取 256 B。
2. 坐标系：`pick(x, y)` 采用与 DOM/WebGPU 纹理一致的左上原点像素坐标，**无需 y 翻转**；
   仅当从 NDC（左下原点）换算时需要 `py = (1 - ndcY) * 0.5 * h`。
3. 拾取管线直接复用 opaque pipeline 的 `getBindGroupLayout(0/1)`，现有 Primitive 的
   camera/model bind group 天然兼容；对象 id 走独立的 group 2 uniform，每 primitive
   一个 256 B UBO，注册时写一次。
4. 拾取 pass 的顶点布局、cullMode、depthCompare 必须与 G-Buffer pass 完全一致，
   否则拾取结果与画面不一致。
5. `drawOpaque(pass, cameraGroup)` 同时绑定 group 0/1，group 2 在其之前设置即可
   （bind group 按槽位覆盖，drawOpaque 不会触碰槽位 2）。

## 后续（生产化建议，未在本 spike 范围）

- 拾取纹理可降到 canvas 的 1/2 或 1/4 分辨率（拾取天然允许 ±2px 容差），pass 成本再降 4–16 倍；
- id 分配改为 LRU 复用（primitive 销毁后回收 id）；
- 触发策略：pointerdown 才置 pending，`pointercancel`/拖拽取消未执行的请求；
- 多点采样（点击容差半径内 3×3 采样）可在同一帧的拾取纹理上多次 1×1 拷贝，几乎零增量成本。

## 正式版（T2.3，2026-09-02 追加）

spike 转正为引擎 API，实现于 `src/picking/PickPass.ts`（删除 spike 版 `PickRenderer.ts`，
`src/spikes/pick-spike.ts` 已迁移到新 API；新验证页 `public/spikes/pick-pass-spike.html`
+ `src/spikes/pick-pass-spike.ts`）。

### API 签名

```ts
class PickPass {
  constructor(scene: Scene);
  pick(x: number, y: number): Promise<PickResult | null>; // x/y = canvas 设备像素，左上原点
  register(primitive: Primitive): number;   // 仅场景图之外的图元需要；返回其 pickId
  unregister(primitive: Primitive): void;
  invalidate(): void;                        // 手动清缓存
  verifyIdPrecision(testId: number): Promise<number>;
  destroy(): void;
  readonly debug: { hookCalls; encodes; flushes; gpuErrors };
}

interface PickResult {
  id: number;                    // 24-bit；0 = 背景/未命中
  primitive: Primitive | null;   // 命中的图元（Entity 集成待 dataSource 任务）
  worldPosition: Cartesian3 | null; // 归一化 ECEF（地球半径 = 1）
  timing: { encodeMs; readbackMs; totalMs };
}

// 集成
class Scene {
  pick(x: number, y: number): Promise<PickResult | null>; // 懒创建 PickPass，挂 frameHooks
  get context(): Context;
}
class Viewer {
  onPick: ((result: PickResult | null, pos: ScreenSpacePosition) => void) | null; // LEFT_CLICK 触发
  get screenSpaceEventHandler(): ScreenSpaceEventHandler | null;
}
class Primitive {
  get pickId(): number; // 构造时分配，24-bit 递增（静态计数器），渲染路径零改动
}
class Globe {
  get tilePrimitives(): Primitive[]; // 供 PickPass 枚举瓦片（仅拾取帧调用）
}
```

### 自动拾取集合

与 G-Buffer pass 完全同序同状态：`globe.primitive`（基础球）→ `globe.tilePrimitives`
（瓦片）→ `scene.primitives.opaque` → `PickPass.register()` 手动注册项。拾取管线复用
opaque pipeline 的 BGL 0/1，对象 id 走独立 group 2 UBO（按 pickId 惰性创建并缓存）。

### 行为保证

- 每帧最多 1 次回读：in-flight 期间 `pick()` 复用同一 pending promise；
- 结果缓存：同像素 + 相机视图指纹 + 场景指纹（图元数/瓦片数/注册数）不变即命中缓存
  （缓存命中时 timing 置 0，避免误当新测量）；
- `pick()` 在场景未初始化时返回 `null`；命中失败时返回 `id=0` 的 `PickResult`。

### 正式版实测（真机：Windows 11 + Chrome Dawn，同一 GPU 环境）

场景 = 地球（legacy 影像）+ cone4-red.glb（1 mesh）+ 两个椭圆面多边形
（`pick-pass-spike.html`，自动 50 次连续拾取在 4 类目标间轮转，±1px 抖动规避缓存）：

| 指标 | 数值 |
| --- | --- |
| 50 次连续拾取总耗时 | **300.3 ms（6.0 ms/次，帧帧出结果）** |
| 单次 wall 延迟（call → result） | 平均 **5.97 ms**，min 1.1 / max 14.3 ms（< 8 ms 达标） |
| readback 延迟（submit → mapAsync） | 平均 **2.14 ms**，max 4.8 ms |
| 命中分布（50 次） | GLB 模型 13 / 多边形 A 13 / 多边形 B 12 / 地球 12（全命中） |
| 真实 LEFT_CLICK 链路（5 次） | 模型/多边形 A/多边形 B/地球/背景 全部类型正确，id 与经纬度正确，背景 id=0 |
| GPU uncaptured error / console 错误 | 0 / 0 |

### 正式版验证中的坑

1. **平面多边形弦下沉**：多边形用平面三角扇近似球面，跨越 2θ° 的弦在边缘下沉
   `(1−cos θ)·R`（6° 时 ≈ 35 km）。抬升量必须大于弦下沉，否则整个多边形被
   深度测试剔除（表现为"不可见且不可拾取"）。验证页取抬升 100 km。
2. **后台标签页 rAF 停摆**：拾取帧钩子挂在渲染循环上，Chrome 后台标签不执行
   rAF → pick promise 永不 resolve。QA 用 Worker 定时器泵 `scene.render()` 绕过；
   真实前台使用不受影响。
3. **视角尺寸 vs 扫描步长**：自动找目标像素的环形扫描步长必须小于目标的屏幕尺寸，
   否则会跨过小目标造成"未命中"假象。
