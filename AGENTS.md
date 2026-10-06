# 项目工作规则（用户指定，长期有效）

## 双 Agent 改验循环（用户 2026-10-03 指定）

对视觉/效果类任务（以及用户未另行说明的修改任务），采用「一个 agent 改、一个 agent 检查」的循环，**直到负责检查的 agent 回复达到用户要求（PASS）才能收尾**：

1. **实现 agent**（task 子代理）：按检查 agent 上一轮的差距清单修改代码；只改不验。
2. **检查 agent**（task 子代理）：独立截图验证，输出严格 JSON 裁决 `{"verdict":"PASS"|"FAIL","underwater":{...},"aboveWater":{...},"gaps":[...]}`；检查结论权威，PASS 需同时满足用户的验收标准（如两视角评分 ≥8/10、噪声指标达标）。
3. FAIL → 把 gaps 清单发给实现 agent 下一轮；PASS → 主会话统一跑 vitest/tsc 回归后收尾。
4. 两个 agent 的任务书必须包含完整方法（不得假设子代理知道上下文）。

### 检查 agent 的验证方法（视觉任务，已在 96_oceanFFT 上验证有效）
- 浏览器截图：`browser.open` 等待加载/动画稳定（本页 14-15s）→ `page.screenshot`。
- 视觉模型分析：PNG 包进 SVG（`<image xlink:href="data:image/png;base64,...">`）写盘，用 `read` 工具读 `<file>.svg:img?q=问题` 获得画面文字描述——直接看 PNG 的 `:img` 不支持。
- 定量指标配合视觉描述（如天空噪声 = 天空区相邻像素 |ΔR|>20 占比 <5%；相机 30s 高度采样曲线）。
- 时序敏感状态要等稳定（瓦片流送 ~15s），上浮/入水过渡帧不作数。

### 历史经验（96 页水下/天空效果迭代实证）
- 子代理内核有 `browser` 全局对象，可直接截图。
- 实现 agent 不跑 vitest/tsc/build（主会话统一跑）；钉旧实现文本的测试断言随行为更新。
- 仓库存在并发会话改动（Camera/Viewer/GBufferPass 等），二分定位时先用 `git stash` 对照基线。
