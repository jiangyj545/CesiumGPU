# 0003 - 粒子系统方案：compute shader GPU 模拟

- 状态：已接受（Accepted）
- 日期：2026-09-02
- 关联 spike：`public/spikes/particle-spike.html`（已删除，由正式版验证页接替）
- 关联验证页：`public/spikes/psystem-spike.html`（`src/spikes/psystem-spike.ts`）
- 关联实现：`src/effect/ParticleSystem.ts`（正式版 T3.4）、`src/shaders/ParticleCompute.ts`
  （spike 版 `ParticleSim.ts` 已删除，逻辑被正式版吸收）

## 背景

要为引擎加入地表喷发 / 火焰类粒子效果，需要验证：

1. compute pass 模拟 + instanced billboard 渲染能否达到 10 万粒子 60 fps（<16.6 ms）；
2. compute pass 能否与现有 4-pass 延迟渲染（G-Buffer → Lighting → WBOIT accumulate →
   composite）在**同一个 command encoder** 内共存；
3. storage buffer 容量与 dispatch 数量级上限；
4. 若 GPU 路径不可用，CPU 模拟降级的触发阈值。

## 决策

**采用 GPU compute 模拟 + 同 encoder 渲染的方案。**

- 粒子状态存单个 storage buffer，**32 字节/粒子**：
  `{ pos: vec3<f32>, age: f32, vel: vec3<f32>, life: f32 }`（WGSL 对齐后正好 32B，无浪费）。
- **in-place 单 buffer 读写**即可：每个粒子只被拥有它的那个 invocation 读写，粒子间完全
  独立，无需原子操作、无需 double-buffer ping-pong，也无需 workgroup 同步。compute pass
  与后续 render pass（vertex 阶段 `var<storage, read>`）之间的依赖由 WebGPU 在 pass 边界
  隐式插入 barrier 保证，实测 readback 结果正确（0 NaN，位置/速度符合弹道预期）。
- 每 workgroup 64 粒子，`dispatchWorkgroups(ceil(N/64))`。reset 时随机方向用 PCG hash
  （`pcgHash`/`rand01`，seed 含 frame counter）替代 random。
- 渲染为 4 顶点 triangle-strip quad + `@builtin(instance_index)` 从 storage buffer 读位置
  的 screen-aligned billboard；clip 空间偏移 `corner * size/tanHalfFov`（除以 `clip.w` 后
  自然得到透视正确的 NDC 偏移），加法混合合成到 Lighting pass 输出之上。
- **共存机制**：`Scene` 新增 `frameHooks: Array<(encoder, swapChainView) => void>`，
  在 `Scene.render()` 的 composite 之后、`submit` 之前调用。粒子的 compute pass 和
  draw pass 由此追加进与 4 个场景 pass **同一个** `'Frame Encoder'`。实测无任何
  validation error、无渲染冲突（swap-chain view 在同一 encoder 内被多个顺序 render pass
  复用是合法的，Lighting 与 composite 本来就这么做）。
- UBO 更新沿用引擎现有模式：每帧 `queue.writeBuffer`（sim params 48B + draw params 80B），
  在 encode 期间调用是安全的，操作在 queue 时间线上先于本次 submit。

### 已知限制（正式版待解决）

- 粒子 pass 未做深度测试，地球背面的粒子会加法透出（需把 `GBufferPass.textures.depthView`
  作为 depth attachment 接入，`depthCompare: 'less'` + 不写深度）。
- `dt` 上限 clamp 到 0.1 s，切后台返回不会爆模拟。

## 验证数据

测试环境：Chrome WebGPU / NVIDIA Blackwell（D3D12 后端），165 Hz 显示器，Windows 11。
指标为 spike 页 rAF 帧间隔（包含：场景 4 pass + 粒子 compute pass + 粒子 draw pass +
present），滚动窗口统计，URL 参数 `?count=N&mode=fire&size=15000`。

| 粒子数 | 平均帧耗时 | fps | min / max | 超 16.67 ms 帧数 |
|--------|-----------|-----|-----------|------------------|
| 100 000 | 6.06 ms | 165 | 0.6 / ~18 ms | ~2 / 1500 |
| 500 000 | 6.06 ms | 165 | 0.7 / 55 ms（离群值） | 2 / 303 |
| 1 000 000 | 6.06 ms | 165 | 2.8 / 16.7 ms | 1 / 380 |

结论：**10 万粒子 compute+draw 每帧总耗时远低于 16.6 ms 预算**。所有规模下平均帧耗时
均钉在 6.06 ms ≈ 1000/165，即帧耗时由垂直同步而非负载决定；从 10 万到 100 万粒子
帧耗时没有可观测增长，非受限帧（min）0.6~2.9 ms，说明粒子路径本身的 CPU+GPU 开销在
低个位数毫秒级。注意：rAF 帧间隔被 vsync 钳制，无法分离纯 GPU 段耗时；正式版
`ParticleSystem` 应启用 `timestamp-query` feature 拿精确 GPU 时间。

### 容量与 dispatch 上限

- 实测本机 `adapter.limits`：`maxBufferSize = 2 GiB`、`maxStorageBufferBindingSize ≈ 2 GiB`、
  `maxComputeWorkgroupsPerDimension = 65535`。注意 WGSL 规范默认值只有
  256 MiB / 128 MiB，Chrome 在 D3D12 上会暴露更高值；引擎代码应 `requestDevice` 时显式
  请求所需 limits，不要假设 2 GiB 一定可用。
- 32 B/粒子下：
  - 10 万粒子 = 3.2 MiB（规范默认 128 MiB storage 上限的 2.5%，2 GiB 的 0.16%）；
  - 规范默认 128 MiB binding 上限 → 约 **419 万粒子**；
  - 2 GiB（本机）→ 约 **6710 万粒子**（此时早被 fillrate 限制）。
- dispatch 上限：X 维 65535 个 workgroup × 64 粒子 ≈ **419 万粒子/次 dispatch**，
  与规范默认 storage 上限恰好同量级；更大规模需多次 dispatch 或 2D global_invocation 映射。

### CPU 降级触发阈值建议

CPU 侧实测参考：纯 JS 积分（重力+位移，不含上传与渲染）约 **1.4 ms / 10 万粒子 / 帧**
（本机桌面 CPU，≈14 ns/粒子/帧）；按 3 ms 模拟预算约 20 万粒子/帧为桌面 CPU 上限，
且每帧 32B×N 的整块 `writeBuffer` 回传很快成为瓶颈。

建议触发条件（按优先级）：

1. **无 WebGPU 或 fallback adapter（软件渲染）** → 直接走 CPU 路径，且粒子数上限
   **2 万**（≈0.3 ms/帧模拟 + 渲染端改用 CPU 每帧重建的动态 buffer）。
2. **移动端/集成 GPU**（adapter info 判断）→ GPU 路径但粒子数上限 **20~30 万**，
   billboard 改小、避免全屏加法 overdraw。
3. **桌面 GPU**：目标粒子数下若 spike 同款统计窗口（5 s）内 `avgMs > 16.67` 且
   超预算帧占比 > 2% → 先降粒子数 / 加 LOD（距离剔除+尺寸衰减），**仅当粒子数
   ≤ 5 万仍超预算**时才降级 CPU；10 万以上规模在实测中不可能触达该阈值。

## 文件清单（spike 期，历史）

- ~~`src/effect/ParticleSim.ts`~~ — spike 版，已删除，逻辑被正式版吸收。
- `src/shaders/ParticleCompute.ts` — `PARTICLE_COMPUTE_SHADER`（积分+PCG 重生）、
  `PARTICLE_RENDER_SHADER`（instanced billboard + 色带渐变）。
- `src/scene/Scene.ts` — `frameHooks`（spike 期加入；正式版改走 `EffectManager`，
  frameHooks 保留供其它用途）。
- `src/core/Viewer.ts` — `get context()` 访问器。
- ~~`src/spikes/particle-spike.ts` + `public/spikes/particle-spike.html`~~ —
  spike 验证页，已删除，由 `psystem-spike` 接替。

## 正式版（T3.4，2026-09-02）

### 深度接入结论

粒子 render pass 已接 `GBufferPass.textures.depthView`（depth32float）做**深度测试**，
spike 期"地球背面粒子加法透出"的限制已解决：

- **RenderPass 层**：`depthStencilAttachment` 传
  `{ view: gbuffer.textures.depthView, depthReadOnly: true }`。
  注意 WebGPU 规范要求 `depthReadOnly: true` 时**不可**设置 `depthLoadOp` /
  `depthStoreOp`（与 `TransparentPass` 同款模式）。深度 view 每帧从
  `scene.gbufferPass` 现取，G-Buffer resize 重建后自动跟随，无需重建管线。
- **Pipeline 层**：`depthStencil: { format: 'depth32float', depthWriteEnabled: false,
  depthCompare: 'less' }`——只读不写，粒子是加法发光叠加，不得污染场景深度。
- **Billboard 深度**：quad 四个顶点的 `clipPos.z` 统一取粒子**中心点**的投影 z
  （`base.z / base.w`），整个 billboard 用一个深度参与测试；粒子中心被地球遮挡时
  整个 sprite 丢弃。这是刻意取舍：逐像素软边深度会产生粒子和地面交界的硬切边，
  中心点深度在视觉上更干净。
- **兼容性**：同一 depth texture 在一个 encoder 内先被 G-Buffer pass 写入、之后被
  粒子 pass 只读使用是合法的（不同的 usage scope，pass 边界有隐式 barrier），
  真机验证无 validation error。

### API（正式版）

```ts
const ps = new ParticleSystem(viewer.context, {
  scene: viewer.scene,           // 相机（billboard 投影）+ G-Buffer 深度（遮挡剔除）
  mode: 'fire',                  // 'fire' | 'fountain' | 'fireworks' | 'smoke'
  count: 10_000,                 // 粒子池容量
  origin: { lon: 116.391, lat: 39.907, height: 0 },  // 或米制 ECEF Cartesian3
  emissionRate: 2000,            // 粒子/秒（省略 = count 全量激活）
});
viewer.scene.effects.add(ps);    // EffectManager：composite 之后 execute，show 开关
ps.setOrigin(newOrigin);         // 动态移动，已发射粒子保持原轨迹
ps.destroy();                    // 释放全部 GPU buffer
```

实现 `IPass`（label / prepare / execute / destroy），兼容 `EffectManager` 的
`show` 开关。时间由 `execute(…, time)` 传入（秒），内部差分 `dt` 并 clamp 到 0.1 s。

### 各模式粒子数建议

预设参数见 `ParticleSystem.presets`。实测 10 万粒子仍在 vsync 预算内（见上文
验证数据），但加法混合的 billboard overdraw 是主要瓶颈，**建议按屏幕占比而非
绝对数选粒子数**：

| 模式 | 建议粒子数 | 说明 |
|------|-----------|------|
| fire | 5 000 ~ 15 000（默认 10 000） | 短寿命高替换率，锥形密集；billboard 4~8 km |
| fountain | 8 000 ~ 20 000 | 抛物线稀疏，寿命稍长；billboard 3~6 km |
| fireworks | 3 000 ~ 8 000 | 球形爆点 + 尺寸随寿命衰减（拖尾感），多彩；billboard 1.5~3.5 km |
| smoke | 2 000 ~ 5 000 | 大 billboard（9~16 km）+ 低 alpha，overdraw 最高，数量务必压低 |

`emissionRate` 用 Little 定律换算活跃数（`rate × 平均寿命`），未激活粒子在
compute 侧驻留为死亡状态，池容量不变、无重分配。

### 验证页

`public/spikes/psystem-spike.html`（`src/spikes/psystem-spike.ts`）：地球 + 北京
（116.391E, 39.907N）发射器 + 四模式切换按钮（切换即 destroy 旧实例）+ FPS HUD
（`window.__psystemSpike`）。`?count=&mode=&cam=` 可调。真机验证四模式视觉正确、
旋转地球到背面时粒子被深度遮挡、10 000 粒子默认帧率 ≥ 60。
