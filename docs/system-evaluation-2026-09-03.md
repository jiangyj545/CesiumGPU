# CesiumGPU 系统评估报告

- **日期**：2026-09-03
- **方法**：ultrawork 并行评估（构建基线 + 4 个独立审计 agent：架构 / 代码质量 / GPU 资源 / 功能完成度 + 测试文档）
- **范围**：~15,700 行 TypeScript（80 个 src 文件），18 个已修改未提交文件 + 40+ 未跟踪生产文件

---

## 一、总体结论

| 维度 | 评分/状态 | 一句话结论 |
|---|---|---|
| 构建健康 | typecheck ✅ / build ✅ / **lint ❌** | 无 ESLint 配置，lint script 形同虚设 |
| 功能完成度 | **88-90%** | README 声明的 11 项功能全部真实实现，无虚标 |
| 代码质量 | **7 / 10** | 类型安全好（零 any），扣分在异步竞态、资源释放、加载器健壮性 |
| 架构 | 「可工作原型 → 可维护引擎」临界点 | 分层意图清晰但约束靠约定；3 处循环依赖 |
| 渲染/GPU | 1 严重 + 3 高危 | WBOIT/延迟着色/G-Buffer 核心实现正确；存在 uniform 对齐错位与拾取 DPR 缺陷 |
| 测试 | **0 个可运行测试** | 数学库 1,695 行纯函数零覆盖 |
| 文档 | 高质量且与代码一致 | 8 指南 + 3 ADR 抽查全部吻合，诚实标注已知缺陷 |
| 仓库卫生 | 差 | ~5,000 行生产代码未提交；根目录杂物；无 CI |

**综合判断**：这是一个功能真实、文档诚实、数学实现严谨的 WebGPU 地球引擎原型，距「可维护的引擎」还差：提交代码、修高危缺陷、建立测试与 CI 三步。

---

## 二、关键风险清单（合并去重，按优先级）

### P0 — 立即处理

1. **~5,000 行生产代码 + 核心渲染管线改动全部未提交**（丢失风险最高）。
   18 个已跟踪修改（Scene/Camera/Globe/Primitive/passes）+ 未跟踪的 imagery/effect/geometry/dataSource/handler/analysis/picking 整目录。`src/spikes/`、`public/spikes/`、`agent-workspace/`、根目录 PNG、`seed.spec.ts`、`.omc/` 不应入库。
2. **[严重] OverlayCamera uniform 布局错位** — `src/shaders/OverlayShader.ts:17-24`
   注释声称 `depthBias@72`，但 WGSL uniform 16 字节对齐使其实际在 `@80`，minBindingSize=112，而 UBO 仅 80 字节（`OverlayPass.ts:190-194`）→ validation error 或 depthBias 恒为 0。
3. **[高] GltfLoader 索引模型法线计算错误** — `src/loader/GltfLoader.ts:625-652`
   `_generateFlatNormals` 按顶点序遍历，但索引模型应按 indices 三元组；无 NORMAL 的索引式 glTF（极常见）会静默渲染错误。

### P1 — 本周内

4. **[高] 拾取坐标缺 DPR 换算** — `ScreenSpaceEventHandler.ts:166-169` 传 CSS 像素，`PickPass.ts:543-546` 按 device 像素使用；DPR>1 屏幕拾取系统性偏移。
5. **[高] ResizeObserver 泄漏** — `Viewer.ts:209` 的 `ro` 在 `destroy()` 中未 `disconnect()`。
6. **[高] `Primitive.ts:86,94` 的 `as ArrayBuffer` 断言** — `subarray()` 视图会写整个底层 buffer，产生脏数据。
7. **[中] 初始化竞态** — `Viewer.initialize`/`Scene.initialize` 的守卫标志在多次 await 后才置位，并发调用会重复创建管线/渲染循环。修复：缓存进行中 Promise。
8. **[高] Globe.ts:113-148 Promise 未 catch** — `createImageBitmap(...).then(...)` 无 `.catch`，且有 `void bitmap;` slop 残迹。
9. **ESLint 配置缺失** — `npm run lint` 直接崩溃，质量门禁为零。
10. **零测试** — 引入 Vitest，先覆盖 `Matrix4.ts`（305 行，所有 pass 的底座）。

### P2 — 规划内改进

11. **相机 UBO 256B 布局双份维护**（`PickPass.ts:461-478` 复制 `Scene.ts:349-367`）→ 抽出统一 `CameraUniforms.write()`；同时修 `nearFar` 静默错位（实际 @224，代码写 216/220，当前无 shader 引用，属潜伏地雷）。
12. **device.lost 无处理**（`Context.ts:62-64` 仅 console）→ 至少停帧 + uncapturederror 全局监听。
13. **无视锥剔除** — G-Buffer 无条件遍历全部 primitives；拾取帧全画布重绘 + mapAsync 主线程 stall（自测 ~6ms）→ 1×1 viewport/scissor 可大幅缓解。
14. **架构循环依赖**：core↔scene（Camera↔CameraFlightPath）、scene↔picking（PickPass 渗透 Scene 内部）。
15. **公共 API 泄漏内部物**：`index.ts` 导出 Context、4 个具体 Pass、shader 源码常量；`main.ts` 尾部重复 re-export。
16. **IPass 未被一致遵守**：GBuffer/Lighting/Transparent 未实现接口，Scene 直捅 pass 内部（`Scene.ts:186-196,305-319`）。
17. **无 CI**：建最小 GitHub Actions（typecheck + lint + vitest）。
18. **其他**：极区拾取按正球求交（WGS84 偏差 ~21km）、glTF 无 skinning/animation/KTX2/Draco、瓦片失败无重试、热路径每帧 new Float32Array、`_testPick` 每次重建 pipeline。

### 已确认正确（无需动）

WBOIT 实现（累乘/composite/depth 状态）、TileTextureCache 2 帧延迟销毁、G-Buffer worldPos COPY_SRC、rgba32float unfilterable-float 采样规避、量测数学（Andoyer-Lambert/Girard）、影像降级路径（父级回退 + 程序化纹理）、WebGPU 不可用提示。

---

## 三、修复路线图

**阶段 1（保护资产，半天）**
- [ ] 清理 .gitignore（`.omc/`、`agent-workspace/`、根目录 `*.png`、`docs/api/` 产物）
- [ ] 删除杂物（w-afterdrag.png、wall-spike-check.png、tile-lod-region-800km、seed.spec.ts）
- [ ] 分批提交：生产代码（src + docs + public/examples + typedoc.json）；spikes 单独决定去留
- [ ] 补 `.eslintrc` 使 lint 可运行

**阶段 2（高危缺陷，1-2 天）**
- [ ] 修 OverlayCamera 对齐（UBO ≥112B，JS 写入偏移同步）
- [ ] 修 GltfLoader 索引法线 + 5 处 `!` 显式校验
- [ ] 拾取链路乘 DPR；PickPass 改 1×1 viewport
- [ ] Viewer.destroy 补 ro.disconnect()；Globe 补 .catch
- [ ] initialize 缓存 Promise 防竞态

**阶段 3（质量地基，2-3 天）**
- [ ] Vitest + Matrix4/Transforms/Ellipsoid 单测
- [ ] 最小 CI（typecheck + lint + test）
- [ ] CameraUniforms 统一写入模块
- [ ] device.lost 停帧处理

**阶段 4（架构演进，迭代）**
- [ ] GBuffer/Lighting/Transparent 实现 IPass
- [ ] 解循环依赖；收敛 index.ts API 面
- [ ] 视锥剔除（包围球 vs 6 平面）
- [ ] 地形（DEM）支持——当前量测/绘制均为 0 高椭球假设

---

## 四、证据索引

| 发现 | 位置 |
|---|---|
| lint 崩溃 | 运行 `npm run lint`：ESLint couldn't find a configuration file |
| Overlay 对齐 | OverlayShader.ts:17-24 / OverlayPass.ts:190-194,246 |
| 法线错误 | GltfLoader.ts:625-652（索引在 :441 使用） |
| DPR 缺陷 | ScreenSpaceEventHandler.ts:166-169 → PickPass.ts:183,543-546 |
| UBO 双份维护 | PickPass.ts:461-478 vs Scene.ts:349-367 |
| 循环依赖 | Camera.ts:6↔CameraFlightPath.ts:5；Scene.ts:14↔PickPass.ts:1 |
| API 泄漏 | index.ts:41,93-96,127-129；main.ts 尾部 |
| 功能矩阵核对 | 逐项证据见各模块文件（ImageryLayer.ts:234-241、Measure.ts:52-91、ParticleSystem.ts:227 等） |
