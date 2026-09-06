# 0005 - 影像"破洞"系统性修复：瓦片调度、极冠、光照与深度抬升

- 状态：已接受（Accepted）
- 日期：2026-09-06
- 关联实现：`src/imagery/ImageryLayer.ts`、`src/imagery/TileTextureCache.ts`、
  `src/scene/Globe.ts`、`src/scene/EllipsoidGeometry.ts`、`src/scene/passes/GBufferPass.ts`
- 关联测试：`tests/scene/globeLighting.spec.ts`（新增）
- 用户验收：2026-09-06 确认"现在正常了"

## 背景

影像瓦片流送上线后，用户持续多轮报告球面"有破洞 / 没覆盖到"。经像素级 + 浏览器内探针
逐层排查，确认**六个独立缺陷叠加**，且互相掩盖——每修一层，下一层才显形。本文记录全部
六项根因、修复方案与判读方法论，避免回归。

## 六个根因与修复

### 1. SSE 阈值语义错误（全球马赛克）

- **症状**：任何高度都不触发细分，全球冻结在 L2/L3。
- **根因**：`DEFAULT_MAXIMUM_SSE = 2.0`，而引擎 SSE 语义是"每纹素覆盖的屏幕像素数"
  （1.0 = 像素密度匹配点），阈值 2.0 意味着"糊 2 倍才细分"。
- **修复**：`ImageryLayer` / `TerrainLayer` 的 `DEFAULT_MAXIMUM_SSE = 1.0`。

### 2. LRU 淘汰杀祖先链（淘汰风暴破洞）

- **症状**：容量压力下瓦片成片消失，露底色。
- **根因**：粗层级瓦片（L≤4）是所有后代的回退纹理，被 LRU 逐出后整棵子树
  `resolveTexture → null`。
- **修复**：`TileTextureCache` 增加 `PIN_LEVEL = 4`，pin 住粗层级条目不参与淘汰，
  且 pinned 数量豁免容量（effectiveCapacity = capacity + pinned）。

### 3. 祖先请求饿死（新区域永远空洞）

- **症状**：淘汰风暴下 10 个并发槽被重取的叶子占满，新区域祖先排不上队
  （诊断特征：pinRes 恒定不增长）。
- **修复**：祖先预取请求排叶子**之前**（父优先，CesiumJS 同款）——队列距离
  减去 `ANCESTOR_REQUEST_PENALTY`。

### 4. 隐藏图层冷启动（切换瞬间全屏洞）

- **症状**：点开隐藏图层瞬间全屏空洞数秒。
- **根因**：`activeLayer` 只更新栈顶可见层，隐藏层从加载起零瓦片。
- **修复**：`ImageryLayer.update` 隐藏分支走 `_warmRootTiles`（16 个 L2 根，pin 常驻）；
  `Globe.update` 循环更新隐藏层。失败记 TTL 重试，不静默吞错。

### 5. 默认太阳光压黑背光半球（用户看到的"破洞"主体）

- **症状**：球面一半几乎全黑，与太空背景肉眼不可分。
- **根因**：`LightingShader` 无条件 Lambert（`diffuse = albedo × NdotL`），Scene 默认
  太阳光 `ambientIntensity = 0.08` → 背光半球只剩 `ambient × albedo`（海洋 ≈ 25/255）。
  CesiumJS 默认 `globe.enableLighting = false`，本引擎此前无此开关，属行为回退。
- **修复**：`Globe.enableLighting` 默认 `false`（CesiumJS 语义）。为 false 时
  `GBufferPass` 把 base globe + 瓦片 + 极冠整体切 unlit 管线（`fs_unlit` 零法线标记，
  原始 albedo 直出，无着色/雾效/色调映射）；画完恢复 opaque 管线。bind group 布局
  两管线恒等复用（`buildUnlitPipeline` 取 opaque 的 BGL），切换零重建开销。
- **测试**：`tests/scene/globeLighting.spec.ts` — mock pass encoder 记录 `setPipeline`
  序列，覆盖默认值 / unlit 切换 / lit 保持 / null 回退 / show=false 五种路径。

### 6. 瓦片深度抬升不足（"没覆盖到"的真凶，被 5 掩盖）

- **症状**：关掉光照后球面 83.8% 裸露底色（赤道视角 98.3%），只剩瓦片边缘可见；
  极冠贴片反而正常。
- **根因**：`TILE_RADIUS_SCALE = 1.001`（0.1% 抬升）< 瓦片三角形的弦切下沉量。
  瓦片网格 `TILE_PARTITIONS = 8`，L2 根每格 ≈11.25°，弦 vs 弧 sagitta ≈
  `1 − cos(5.6°) ≈ 0.0048`；而 base globe 128 分区（≈2.8° 格）仅 ≈0.0003。
  瓦片中部沉到 base globe 之下 → 深度测试失败 → 整片中段被吞。
- **为何长期未暴露**：太阳光照下"底色 × NdotL 渐变"与"影像 × NdotL 渐变"视觉不可分，
  完美掩盖缺失；修掉光照后立即现形。
- **修复**：`TILE_RADIUS_SCALE = 1.008`（> 最坏 0.0048 × 1.5 裕量；≈50 km @ 真实
  地球尺度，全球视角视觉无感）。极冠共用该常量同步受益。
- **备选方案（未采用）**：per-primitive 深度状态需新增 pipeline 变体，改动面大；
  抬升常数是零成本解。若未来出现"瓦片浮起"视觉问题，再考虑瓦片专用的
  `depthWriteEnabled: false` pipeline。

### 附带修复

- 极冠贴片（±85.05°~90° Web Mercator 结构性无影像区）：`Globe._syncPolarCaps`
  用 L2 顶/底行纹理拉伸覆盖极点，`createRegion` 无纬度 clamp 生成越界区。
- Mercator v 映射：`EllipsoidGeometry.createRegion` 增 `mercatorV` 选项，v 按
  Mercator-y（`asinh(tan(lat))`）而非纬度线性分布，消除低层级下中纬 ≈7.7° 北偏
  （"影像挤向北极"）。
- 示例页 `html,body{height:100%}` 统一注入（容器塌陷致 canvas 半屏、SSE 视口因子砍半）；
  `ResizeObserver loop` 良性噪声过滤（不再弹红色错误面板）。
- 天地图接入：`XyzImageryProvider.tianditu(tk, layer)`（DataServer 端点），
  示例 `01_imagery.html` 四图层（高德卫星/矢量/OSM/天地图）。

## 修复效果（像素级验收，1744×634 宽窗 CDP 截图）

| 场景 | 修复前裸露 | 修复后 |
| --- | --- | --- |
| 默认全球视角 | 83.8% | 0.1%（影像 94.4%） |
| 赤道视角 | 98.3% | 0.0%（影像 92.6%） |
| 南半球视角 | 77.6% | 0.5%（影像 92.0%） |
| 北京 300km 城市级 | — | 影像 97.9%，L5-L8 正常细分 |
| 连续扫动压力后 | — | 0.0% 裸露，瓦片数稳定 |
| 淘汰压力（容量注入 30） | 53/58 帧洞 | 0/58 帧洞 |
| 图层切换（隐藏层预热） | 瞬间 57 洞 | ≤1.5s 内 nulls=0 |

类型检查 0 错误，219/219 测试通过。

## 判读方法论（防回归的验证手段）

1. **底色铁证**：恒定 `(26,89,153)` = `baseColor(0.1,0.35,0.6)×255` 线性直出 =
   裸底色。天地图海洋真色是 `(7,6,40)` 近黑（浏览器内拉真瓦片采样确认）。
   二者区分是"覆盖 vs 未覆盖"的判据，别把底色误读成"海洋原色"。
2. **量化色数**：unlit 帧全球量化色 527 > lit 439——无光照反而保留更多原始色彩信息。
3. **真瓦片对照**：在页面上下文内 `fetch` 与引擎同端点的瓦片 → canvas 采样，
   得到"真值"色，Node 直连会 403（缺 Referer/UA）。
4. **A/B/A/B 交替截图**：同页面连续切换开关各拍一帧，排除加载时序干扰
   （并同步读 `tilePrimitiveCount` 证明 primitive 集合未变）。
5. **几何反算**：页面内读 `_geometryCache` 顶点（`[pos3,normal3,uv2]×8`）反算
   经纬度/半径，直接验证几何落点，比像素推断可靠。
6. **无头验证环境**：有头 Chrome + `--remote-debugging-port=9444` +
   `Emulation.setFocusEmulationEnabled`（无头拿不到 WebGPU adapter；后台标签 rAF 停）。
   WebGPU canvas 用 `Page.captureScreenshot`（2D drawImage 读回全黑）。
