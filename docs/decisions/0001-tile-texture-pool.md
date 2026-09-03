# 0001 - 瓦片纹理方案：texture-per-tile + LRU 纹理池

- 状态：已接受（Accepted）
- 日期：2026-09-02
- 关联 spike：`spikes/tile-spike.html`（`spikes/ts/tile-spike.ts`）
- 关联实现：`src/renderer/MipmapGenerator.ts`、`src/imagery/TileTextureCache.ts`

## 背景

Globe 当前用单张 1024×512 程序化纹理（`Globe.loadImageryTexture`，`mipLevelCount: 1`，见
`src/scene/Globe.ts:84-90`）。要接入真实影像瓦片流送（XYZ），需要验证：

1. 每瓦片独立 GPU 纹理（256×256，带完整 mip 链）+ 每瓦片独立 bindGroup 是否可行；
2. WebGPU 无原生 `generateMipmaps`，render-based mip 链生成的开销是否可接受；
3. 批量上传是否会造成帧尖峰，如何控制；
4. 纹理池的 LRU 淘汰与 `texture.destroy()` 是否可靠。

## 决策

**采用 texture-per-tile + bind group 缓存方案。**

- 每瓦片一张独立 `rgba8unorm` 纹理，`mipLevelCount = floor(log2(max(w,h))) + 1`（256×256 → 9 级），
  usage 必须含 `RENDER_ATTACHMENT | TEXTURE_BINDING`（+ `COPY_DST` 用于 level 0 上传）。
- mip 链由 `MipmapGenerator.generateMips()` 生成：level 1..N-1 每层一次全屏三角形 blit
  （采样上一层 → 渲染到下一层），pipeline/sampler 按 device+format 缓存，重复调用只付
  命令编码 + GPU 开销。
- 每瓦片独立 bindGroup，沿用 `Primitive` layout1 三槽位模式：
  `{ 0: modelUniformBuffer, 1: textureView, 2: sampler }`（`src/scene/Primitive.ts:333-340`）。
  bindGroup 与纹理同生命周期，随瓦片纹理一起创建/丢弃。
- 纹理驻留由 `TileTextureCache`（LRU，默认容量 200）管理：`get` 命中刷新时效，
  `put` 超容量 evict 最旧并 `destroy()`，暴露 hits/misses/evictions 统计。

## Spike 实测数据（Windows 11，真实 Chrome WebGPU，硬件 GPU）

| 指标 | 数值 | 判定 |
| --- | --- | --- |
| 真实 OSM 瓦片（z=10/x=843/y=388）加载 + 上传 | 成功（CORS 允许） | PASS |
| mip 链层数（256×256） | 9 级，canvas 逐级可视化无异常 | PASS |
| 单张 create+copy+genMips（冷，含 blit pipeline 编译） | 0.20 ms | — |
| 单张 create+copy+genMips（热均，20 次） | 0.135 ms | PASS |
| 200 张单帧连发：同步总耗时 | 48.4 ms | — |
| 200 张单帧连发：rAF 帧间隔 | **54.6 ms**（60Hz 下丢约 3 帧） | 不可单帧全量 |
| 200 张按 10 张/帧流送：最大 rAF 帧间隔 | **6.2 ms**（16.7 ms 预算内） | PASS |
| 流送期间单张均摊 | 0.060 ms | — |
| LRU（容量 8，放入 20 张）evictions | 12/12，evicted 纹理全部 `destroy()`，常驻纹理未误毁，get 命中/未命中正确 | PASS |

## 对比：texture_2d_array 方案（否决）

将瓦片放入一张大 `texture_2d_array`，每层一瓦片：

- 层数上限受 `maxTextureArrayLayers`（常见默认 256）与整块预分配显存约束，不能像独立纹理
  那样按需增长/收缩；
- LRU 逐层管理复杂：淘汰某层需要在数组内做 `copyTextureToTexture` 搬移压缩或维护 free-list，
  碎片化后层分配逻辑显著复杂；per-tile 只需 `destroy()` 即可归还显存；
- 淘汰/复用时的层拷贝本身制造额外上传与同步点，收益（减少 bindGroup 数量）对瓦片渲染场景
  不明显——每瓦片本来就需要独立的模型 uniform，bindGroup 无论如何逐瓦片提交；
- 同尺寸限制（所有层必须 256×256）虽然对标准瓦片成立，但失去了对非标准影像（概略图、
  全局底图、不同分辨率层）的兼容性。

结论：**倾向 texture-per-tile**。驱动侧对"多纹理 + 逐 draw 切 bindGroup"的处理已足够快
（200 张纹理驻留、每帧可见瓦片通常 <100 张），复杂度留在 CPU 侧一个简单的 LRU Map 上。

## 执行约束（正式版必须遵守）

1. **上传必须分帧**：预算 16.7 ms/帧。10 张/帧实测峰值 6.2 ms，可按视口瓦片优先级排队列逐帧消化，
   严禁一次性 burst（54.6 ms 尖峰）。
2. 每瓦片纹理 usage 固定为 `TEXTURE_BINDING | COPY_DST | RENDER_ATTACHMENT`，交给
   `MipmapGenerator` 生成 mip 链后再建 bindGroup。
3. `TileTextureCache` evict 时同步丢弃该瓦片的 bindGroup 引用（bindGroup 随瓦片对象一起缓存/销毁），
   避免引用已 destroy 纹理的 validation error。
4. 采样器可全局复用（linear/linear/linear, clamp-to-edge, 各向异性 4），不必每瓦片一份。

## 后果

- 新增 `src/renderer/MipmapGenerator.ts`（render-based mip 生成，pipeline 缓存 per device+format）。
- 新增 `src/imagery/TileTextureCache.ts`（LRU 池 spike 版，正式版在此之上加瓦片队列/优先级）。
- `Globe.loadImageryTexture` 的 `mipLevelCount: 1` 与 `// TODO: generate mipmaps if needed`
  （`src/scene/Globe.ts:98`）在正式版中替换为上述管线。
