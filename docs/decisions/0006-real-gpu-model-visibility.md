# 0006 - 真机 GPU「模型图元不光栅化」定案：Model.show 传播修复 + 亚像素坍缩建档

- 状态：已接受（Accepted）
- 日期：2026-10-10
- 关联实现：`src/scene/Model.ts`（show 传播）、`src/scene/EntityModelMatrix.ts`（实体原生 minPx）、`src/loader/GltfLoader.ts`（`bounds`，同日并行提交）
- 关联测试：`tests/scene/modelShow.spec.ts`、`tests/loader/modelBounds.spec.ts`

## 背景

真机（NVIDIA Blackwell，Electron 42 / Chromium 148 / Dawn D3D12）上无人机 GLB 模型
（644 子图元）在 G-Buffer 不透明循环里逐帧提交 drawIndexed 却「看不见」；无头
SwiftShader 88 示例全过。调查初期怀疑光栅化管线本身，最终用 **G-Buffer worldPos
纹理直接 GPU 回读**（绕过合成器与 canvas）定案：

1. **光栅化从未坏**。隔离场景（仅模型可见）下按 scale 扫描，回读非零纹素数随尺寸
   平方增长：radius 41 m→182 纹素、78.5 m→734、157 m→3394、314 m→20766；截图
   中央在投影位置确有灰色机身像素。管线、bind group、矩阵、索引全部正常。
2. **真尺度（≈1.5 m）在 900 m 视距下投影仅 ~2.4 px**，coaxial50 的 ~620 万三角形
   平均边长亚毫米 → 单三角形投影 ≪ 1/8 px。D3D12 光栅器按规范把顶点吸附到
   8 位定点子像素网格，整模型全部三角形吸附后零面积 → **零像素**。SwiftShader
   用浮点光栅无吸附 → 无头测试全绿（假阴性来源）。
3. **兜底机制 minimumPixelSize 当时不生效**：直连 `Model` 的宿主（如 hexapod 门面）
   在 preRender 钩子里按 `Model.bounds` 距离缩放；调查时 `bounds` 尚不存在（同日
   由 GltfLoader POSITION accessor min/max 累积补齐）→ 模型恒为真尺度 → 命中 2。
4. **`Model.show` 是普通数据字段**，而 G-Buffer/透明/pick 循环只看子图元
   `prim.show` —— `model.show=false` 后模型照常渲染，一切模型级开关像素差恒 0，
   污染了排查证据（`EntityVisualizer` 的 `entry.model.show = entity.show` 同样
   静默失效）。
5. 取证环境陷阱（记录在案）：Electron 窗口被遮挡时 rAF 冻结 → 引擎按需渲染循环
   整体停摆 → canvas 内容陈旧；`drawImage(webgpuCanvas)` 读到旧帧，模型级像素
   diff 恒 0。真机取证应以 `copyTextureToBuffer` G-Buffer 回读或强制
   `scene.render()` + `Page.captureScreenshot`（强制重新合成）为准。

## 决策

- **`Model.show` 改为传播属性**：setter 同步全部子图元（opaque 与 translucent 两条
  注册路径），异步加载期建出的图元继承当前值，构造参数 `show:false` 从第一帧
  生效 —— 与 CesiumJS `Model.show` 语义对齐。这是唯一的引擎代码修复；光栅化
  路径零改动。
- **亚像素坍缩不引擎侧修**：这是 D3D12/Vulkan 规范行为，正确解法是
  `minimumPixelSize`。实体模型走引擎原生 `resolveMinimumPixelScale`
  （`Scene.render` 内每帧、画前生效，无钩子依赖）；直连 `Model` 的宿主走
  `Model.bounds` + 宿主侧每帧缩放（门面在 preRender mux 中实现，要求渲染循环
  存活）。GLB 三角形密度越高（Draco 重模型），需要越大的 minPx 才能保证「最
  大三角形」过 1/8 px 阈值 —— bounds 半对角径对稀疏大跨度部件（旋翼）偏保守，
  实测 minPx=64 目标下机身可见尺寸 ~15 px。
- **无头 SwiftShader 通过 ≠ 真机可见**：凡依赖小目标光栅化的用例必须配
  minimumPixelSize 或在真机验证。

## 验证（真机，Blackwell/D3D12）

- G-Buffer 回读：minPx 链路（preRender → scale 4.19e-6 = 26.8×）下模型 182 非零
  纹素，`model.show=false`（修复后）全部消失 —— 开关像素差成立。
- 截图（Page.captureScreenshot）：投影位置 (642,403) 处灰色机身像素串
  （R≈G≈B≈150-185），地形为 (76,117,85)。
- `tests/scene/modelShow.spec.ts`：3 用例锁传播语义（默认可见 / 开关全量同步 /
  构造即隐藏）。
