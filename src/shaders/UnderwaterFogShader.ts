import { FULLSCREEN_VS } from './postprocess/Fullscreen';

/**
 * UnderwaterFogShader - 水下观感全屏叠加（tidewater "split underwater/above-water
 * view" 的水下侧简化版）。
 *
 * 相机处于海面之下时画一张全屏三角（draw(3)），以 premultiplied-alpha 混合
 * （one, one-minus-src-alpha）叠加到已合成场景之上。无场景深度纹理——深度感
 * 用两个近似表达：
 *   1. 视线俯仰渐变：相机 forward（viewMatrix 第三行取反）与"上"方向
 *      （归一化 ECEF 世界中 = normalize(cameraPosition)）点积，低头看海床
 *      时雾更浓、抬头看水面时透光更强；
 *   2. 屏幕径向暗角：边缘视线穿水路径更长 → 雾更厚、色调更暗。
 *
 * 四个视觉成分：
 *   - 蓝绿深度雾：水体染色，透明度随俯仰/暗角渐变；
 *   - 浮动颗粒：两层屏幕空间 hash 点阵，慢速漂移 + 闪烁，密度由
 *     particleDensity 调制；
 *   - 焦散光斑：两层正弦波纹相乘调制亮度（呼应 OceanCaustics 的观感，
 *     但独立实现），抬头方向权重更大；
 *   - 体积焦散光柱（Track G 联动，tidewater src/post/Underwater.js
 *     _shaftCode 同法）：沿视线 20 步 ray-march 采样 OceanCaustics 真焦散
 *     纹理（R=浅焦面 / G=深焦面），光柱随焦散网/波浪摆动；HG(g=0.85)
 *     前向散射相函数朝太阳方位会聚。sunShaftIntensity 默认 0 = 完全
 *     关闭（零 march、零贡献，零回归）；未绑定焦散源（bindCaustics 未
 *     调用 / null）时同样不 march。
 *
 * Uniform 布局：
 *   binding 0 : CameraUniforms（256 B，布局单一来源 src/renderer/CameraUniforms.ts）
 *   binding 1 : UWParams（144 B = 9 x vec4）
 *   binding 2 : G-Buffer worldPos（距离衰减雾的水路长，可选）
 *   binding 4 : 焦散纹理（rgba16float，filterable；1x1 fallback 占位）
 *   binding 5 : 焦散采样器（repeat / linear / mipLinear）
 *
 * 输出为 premultiplied 颜色（rgb 已乘覆盖权重），配合
 * { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } 混合。
 * intensity = 0 时 rgb/alpha 全 0 → dst 逐像素原样保留（零贡献）。
 */
export const UNDERWATER_FOG_SHADER = FULLSCREEN_VS + /* wgsl */ `
struct CameraUniforms {
  viewMatrix     : mat4x4<f32>,   // @0   (64 B)
  projMatrix     : mat4x4<f32>,   // @64  (64 B)
  viewProjMatrix : mat4x4<f32>,   // @128 (64 B)
  cameraPosition : vec4<f32>,     // @192 (16 B) w 未用
  viewportSize   : vec2<f32>,     // @208
  nearFar        : vec2<f32>,     // @224
}

struct UWParams {
  params0    : vec4<f32>,   // x = time (s), y = intensity, z = particleDensity, w = hasScenePos(0/1)
  viewport   : vec4<f32>,   // xy = 画布尺寸 (px), z = 海面球半径（归一化 ECEF）, w = 参考海床深（米，0=关）
  waterColor : vec4<f32>,   // rgb = 水体染色, w = unused
  sunShaft   : vec4<f32>,   // xyz = 太阳方向（归一化 ECEF，指向太阳）, w = sunShaftIntensity（默认 0 = 关）
  // ── 体积焦散光柱（tidewater _shaftCode）专用 ──────────────────────
  cCenter    : vec4<f32>,   // xyz = 沙洲中心（归一化 ECEF）, w = coverMeters（0 = 未绑定焦散源）
  cEast      : vec4<f32>,   // xyz = ENU east（归一化 ECEF）, w = sunLocal.x（太阳方向的 ENU 分量）
  cUp        : vec4<f32>,   // xyz = ENU up,   w = sunLocal.y
  cNorth     : vec4<f32>,   // xyz = ENU north, w = sunLocal.z
  cFocal     : vec4<f32>,   // x = 浅焦面深 D0（米）, y = 深焦面深 D1（米）, zw 未用
}

@group(0) @binding(0) var<uniform> cam : CameraUniforms;
@group(0) @binding(1) var<uniform> uw  : UWParams;
// G-Buffer worldPos（rgba32float，归一化 ECEF；unfilterable → non-filtering
// 采样 + textureSampleLevel）。距离衰减雾的水下路径长来源。
@group(0) @binding(2) var scenePosTex : texture_2d<f32>;
@group(0) @binding(3) var posSmp      : sampler;
// 焦散纹理（OceanCaustics 生成，rgba16float，R=浅焦面 / G=深焦面，
// repeat 语义 + 双线性 + mip 三线性——march 用固定模糊级 1.4 取宽级联
// 大尺度软束（tidewater Underwater.js L247 同 level 1.5；Round 3 检查
// 发现 LOD 0.6 细网被 20 步 march 沿光路拉成细条纹）。分工：水面柱
// = 宽束（此处），沙面网 = 细网（CausticsFloor 采细级）。
@group(0) @binding(4) var causticsTex : texture_2d<f32>;
@group(0) @binding(5) var causticsSmp : sampler;
// 静态 IGN jitter（无时间项——本引擎无 TAA，含时间项会闪烁；
// tidewater 用逐帧移动 jitter + TAA 收敛，此处改为静态版防闪）。
fn ign(px: vec2<f32>) -> f32 {
  return fract(fract(dot(px, vec2<f32>(0.06711056, 0.00583715))) * 52.9829189);
}

// VSOut 由 FULLSCREEN_VS 前置声明（重复声明会使模块解析失败——同 BrightShader）

fn hash21(p: vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

// 单层屏幕空间浮游颗粒：cells x cells 网格，每格一枚慢速漂移的亮点，
// 带独立相位的呼吸式闪烁。seed 让两层错开（同一 hash 域不重叠）。
fn particleLayer(p: vec2<f32>, cells: f32, t: f32, seed: f32) -> f32 {
  let g    = p * cells;
  let id   = floor(g);
  let h1   = hash21(id + seed);
  let h2   = hash21(id + seed + 41.7);
  let drift = vec2<f32>(
    sin(t * 0.13 + h1 * 6.2832) * 0.35,
    cos(t * 0.11 + h2 * 6.2832) * 0.35
  );
  let local = fract(g) - vec2<f32>(0.5) - drift;
  // 2026-09-30 颗粒实体收细：0.16 半径在 9 格层上 ≈14 px 白斑（视觉模型
  // 判"bokeh 雪片"非悬浮物）；0.05 → ~4 px 细屑，漂移同款。
  let spark = 1.0 - smoothstep(0.0, 0.05, length(local));
  let tw    = 0.55 + 0.45 * sin(t * (1.1 + h1 * 1.7) + h1 * 6.2832);
  return spark * tw;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let t         = uw.params0.x;
  let intensity = clamp(uw.params0.y, 0.0, 4.0);
  let dens      = clamp(uw.params0.z, 0.0, 4.0);
  let vp        = max(uw.viewport.xy, vec2<f32>(1.0));
  let aspect    = vp.x / vp.y;

  // ── 相机视线俯仰（归一化 ECEF：up = 径向向外）──────────────────────
  // 安全归一化：相机 UBO 兜底全零时 normalize(0) 会产 NaN，整屏坏帧
  let upRaw  = cam.cameraPosition.xyz;
  let up     = upRaw / max(length(upRaw), 1e-6);
  // viewMatrix 第三行 = 世界空间相机 forward 的取反（列主序取 [col][row=2]）
  let fwdRaw = -vec3<f32>(cam.viewMatrix[0][2], cam.viewMatrix[1][2], cam.viewMatrix[2][2]);
  let fwd    = fwdRaw / max(length(fwdRaw), 1e-6);
  let lookDown = clamp(-dot(fwd, up), 0.0, 1.0); // 1 = 低头看海床
  let lookUp   = clamp( dot(fwd, up), 0.0, 1.0); // 1 = 抬头看水面

  // ── 屏幕径向（暗角；uv 翻转对称，径向距离不受影响）────────────────
  let ndc  = in.uv * 2.0 - vec2<f32>(1.0);
  let r    = length(ndc);
  let vig  = smoothstep(0.55, 1.45, r);

  // ── 1、蓝绿深度雾 ─────────────────────────────────────────────────
  // 深度感知版（2026-09-29 用户"水下应该是透明的"）：G-Buffer worldPos 重建
  // 像素水下路径长（米）→ Beer 衰减 fogDist = 1-exp(-dist·σ)。2026-09-30
  // 三修历程：① 背景像素（GBuffer worldPos 清零 = 水下视角的水面本体 +
  // 掠过水面的天空）旧按 |wp-camPos|≈地球半径处理 → 全屏 ~0.83 青纱连
  // Snell 窗一起糊死，改射线-海面球出射距离（抬头 3m 即透、水平 6km 全
  // 雾、朝下被 GBuffer 几何优先接管）；② σ 0.06→0.035 矫枉过正（用户
  // "看不到水下效果"）——tint 随 alpha 同淡、潜水后无水下观感 → σ 回
  // 0.05、基底 0.30、cap 0.87，染色 ×1.25 与浓度解耦：Snell 窗（≤6m
  // 路径）仍透光，远景/环境保有明确青绿水色。未绑定场景位置
  // （params0.w=0）回退俯仰近似（零回归）。
  let fogBase = mix(0.22, 0.45, lookDown);
  var fog     = 0.0;
  var distPx  = 0.0;   // 水下路径长（米）——体积光柱 march 复用（下方 4）
  if ( uw.params0.w > 0.5 ) {
    // 世界射线（与 AtmosphereSky.fs_main 同式：view 行 = 相机基，proj 对角）
    let camR = vec3<f32>(cam.viewMatrix[0][0], cam.viewMatrix[1][0], cam.viewMatrix[2][0]);
    let camU = vec3<f32>(cam.viewMatrix[0][1], cam.viewMatrix[1][1], cam.viewMatrix[2][1]);
    let camF = vec3<f32>(-cam.viewMatrix[0][2], -cam.viewMatrix[1][2], -cam.viewMatrix[2][2]);
    let rd = normalize(camR * (ndc.x / cam.projMatrix[0][0])
                     + camU * (ndc.y / cam.projMatrix[1][1]) + camF);
    let wp = textureSampleLevel( scenePosTex, posSmp, in.uv, 0.0 ).xyz;
    let ro = cam.cameraPosition.xyz;
    let dn = dot( rd, up );
    let camDepthM = ( uw.viewport.z - length( ro ) ) * 6378137.0;
    let isBg = dot(wp, wp) < 0.25;
    if ( dn < -1e-3 && uw.viewport.w > 0.5 ) {
      // 俯视（2026-09-30 用户"想看水底效果"）：水下下半球射线穿不出海面球
      //（出射在地球另一端）——背景像素与远景像素都拿不到有效水路长：前者
      // 无几何，后者是四叉树瓦片弦高下垂到海面下的远景几何（公里级 wp）。
      // 两者一律按参考海床深取 (海床深-相机深)/|俯视分量|，近景真实几何
      //（船底 <100m）用 min 保真——浅水海床/焦散透出，掠射向水路自然拉长。
      // 2026-09-30 沙底可见性：CausticsFloor 是 gbuffer 之外的叠加 pass，
      // 沙面像素在此恒 isBg——掠射向 (|dn|<0.08) 平面参考水路按 1/|dn|
      // 发散到数百米，雾顶满 cap，前方 50-150 m 沙洲斜坡全灭。按浅水参考
      // 平面最大展幅 90 m 封顶：近/中距沙面雾随真实距离增长，开阔水与
      // 网格外远景仍沉入深青。
      // 2026-10-06 检查差距3（焦散网对比被压）：封顶 90→60 m——真焦散网
      //（CausticsFloor）在 30-60 m 沙面被 Beer 项压成均匀青罩（最终画面
      // p90/p10 仅 1.5:1 实锤），收顶让网纹带透出、开阔水与远景仍沉青。
      let dSeabed = min( max( uw.viewport.w - camDepthM, 0.5 ) / ( - dn ), 60.0 );
      let dGeom = select( 1e9, length( wp - ro ) * 6378137.0, !isBg );
      distPx = min( dGeom, dSeabed );
    } else if ( isBg ) {
      // 仰视背景：相机在海面球内，出射根 t = -b+√(b²-c)，有理化避免相消；
      // c<0（水下）保证分母恒正。tExit 命名避开外层时间 t。
      let b  = dot(ro, rd);
      let cc = dot(ro, ro) - uw.viewport.z * uw.viewport.z;
      let tExit = -cc / ( b + sqrt(max(b * b - cc, 0.0)) );
      distPx = max(tExit, 0.0) * 6378137.0;
    } else {
      distPx = length( wp - ro ) * 6378137.0;
    }
    // σ 0.016→0.011（检查差距3 三调）：0.016 时 60 m 沙面 Beer 项 0.62、
    // 叠 fogBase 后 fog≈0.57——焦散网脊/网眼被均匀青罩摊平（最终画面
    // p90/p10 仅 1.5:1 的主因之一）。0.011 下 30 m 清澈 0.28、60 m 封顶
    // 0.48，网纹对比透出，远景仍渐沉青。
    let fogDist = 1.0 - exp( - distPx * 0.011 );
    fog = clamp( ( fogBase * 0.22 + fogDist * 1.0 ) * intensity, 0.0, 0.92 );
  } else {
    fog = clamp( fogBase * ( 0.6 + 0.4 * vig ) * intensity, 0.0, 0.85 );
  }

  // 轻微色偏青（绿/蓝通道略抬、红压暗）+ 边缘暗化。染色浓度 ×1.25 与
  // alpha 解耦（2026-09-30 用户"看不到水下效果"：α 降透明时水色也跟着
  // 变淡水下感消失——浓度独立保底）。
  var tint = uw.waterColor.rgb * vec3<f32>(0.85, 1.06, 1.04) * 1.25;
  tint = min(tint, vec3<f32>(1.0));
  tint = mix(tint, tint * 0.55, vig * 0.6);

  // ── 2、浮动颗粒：两层 hash 点阵，密度受 particleDensity 调制 ───────
  let sp = vec2<f32>(in.uv.x * aspect, in.uv.y);
  let p1 = particleLayer(sp,  9.0, t, 0.0);
  let p2 = particleLayer(sp, 17.0, t * 1.35 + 3.7, 19.19);
  let particles = clamp(p1 + p2 * 0.6, 0.0, 1.0) * dens * intensity;

  // ── 3、焦散光斑：两层正弦波纹相乘缓动（抬头方向权重更大）──────────
  // 2026-09-30 假波纹收敛：真焦散网（CausticsFloor）已是水下主体后，此
  // 屏幕空间正弦"焦散"与真网叠成双重软斑（视觉模型判 bokeh 弥散）——
  // 俯视基底 0.16→0.04（只留水面漏光残迹），仰视权重保留（抬头看水面
  // 的光斑语义不变），pow 2.6→3.2 收窄。
  let w1 = sin(sp.x * 9.0 + sp.y * 4.0 + t * 1.70);
  let w2 = sin(sp.x * 3.5 - sp.y * 7.5 - t * 1.15);
  let shimmer = pow(clamp(w1 * w2, 0.0, 1.0), 3.2);
  let shafts  = shimmer * (0.04 + 0.66 * lookUp) * intensity;

  // ── 4、体积焦散光柱（tidewater src/post/Underwater.js _shaftCode 同法）──
  // 沿视线 20 步 ray-march：每步在采样点做平折射回溯（entry = p.xz +
  // LsUp.xz·(z/mu)，与 CausticsFloor 同式）到水面焦散域，采真焦散纹理
  //（R=浅焦面 / G=深焦面，按深度 wD 插值，固定模糊级 1.4——宽级联
  //  主导 → 大尺度软束；细网分工给 CausticsFloor 在沙面呈现，避免
  //  Round 3 俯视"细条纹放射感"与仰视同细波纹尺度不可分）；
  // (cNet-0.72)×2.6 去均值并放大束对比（LOD 1.4 下 R/G 以宽层为主、
  // 均值下移——0.72 为宽层均值近似，区别于 Round 2 细网域的 0.78）。
  // 透射 Tl=exp(-sigT·(s+z/mu))（tidewater 公式），sigA/sigS/sigT 取 tidewater
  // Frame.js L50-52 的水常数。相位 = 0.75·HG(g=0.85) + 0.25·各向同性，
  // cosθ=dot(rdLoc, LsUp)（LsUp = 朝太阳的水下平折射方向）。
  // 无 TAA → 静态 IGN jitter（见文件头）防闪烁。门控全部 uniform：
  // sInt>0 且有 scenePos 且绑了焦散源（cover>0）且太阳在水上，任一不
  // 满足则零 march 零贡献（零回归）。
  var sunBeam = vec3<f32>(0.0);
  let sInt = clamp(uw.sunShaft.w, 0.0, 4.0);
  if (sInt > 0.0 && uw.params0.w > 0.5 && uw.cCenter.w > 0.5) {
    let sunLocal = vec3<f32>(uw.cEast.w, uw.cUp.w, uw.cNorth.w);
    if (sunLocal.y > 0.0) {
      const sigA = vec3<f32>(0.42, 0.075, 0.035);   // tidewater 吸收
      const sigS = vec3<f32>(0.012, 0.018, 0.024);  // tidewater 散射
      const sigT = sigA + sigS;
      // 局部 ENU 系（米，原点 = 沙洲中心；rd 重算自相机基，同 1 中公式）
      let camR2 = vec3<f32>(cam.viewMatrix[0][0], cam.viewMatrix[1][0], cam.viewMatrix[2][0]);
      let camU2 = vec3<f32>(cam.viewMatrix[0][1], cam.viewMatrix[1][1], cam.viewMatrix[2][1]);
      let camF2 = vec3<f32>(-cam.viewMatrix[0][2], -cam.viewMatrix[1][2], -cam.viewMatrix[2][2]);
      let rdW = normalize(camR2 * (ndc.x / cam.projMatrix[0][0])
                       + camU2 * (ndc.y / cam.projMatrix[1][1]) + camF2);
      let roC   = (cam.cameraPosition.xyz - uw.cCenter.xyz) * 6378137.0;
      let roLoc = vec3<f32>(dot(roC, uw.cEast.xyz), dot(roC, uw.cUp.xyz), dot(roC, uw.cNorth.xyz));
      let rdLoc = normalize(vec3<f32>(dot(rdW, uw.cEast.xyz), dot(rdW, uw.cUp.xyz), dot(rdW, uw.cNorth.xyz)));
      // 朝太阳的水下折射方向（平折射；mu=max(LsUp.y,0.15) 防掠射发散）
      let LsUp = -refract(-sunLocal, vec3<f32>(0.0, 1.0, 0.0), 1.0 / 1.333);
      let mu   = max(LsUp.y, 0.15);
      let cover = uw.cCenter.w;
      let d0 = uw.cFocal.x;
      let d1 = max(uw.cFocal.y - d0, 1e-3);
      let maxD = min(distPx, 22.0);   // tidewater：22 m 截断
      let ds   = maxD / 20.0;
      let j0   = ign(in.pos.xy);      // 静态 IGN jitter（步起点错栅格）
      var shaftAcc = vec3<f32>(0.0);
      for (var i: i32 = 0; i < 20; i = i + 1) {
        let s  = (f32(i) + j0) * ds;
        let p  = roLoc + rdLoc * s;
        let z  = max(-p.y, 0.01);     // 采样点到水面的竖直深
        let entry = p.xz + LsUp.xz * (z / mu);
        let c = textureSampleLevel(causticsTex, causticsSmp, entry / cover + vec2<f32>(0.5), 1.4);
        let wD   = clamp((z - d0) / d1, 0.0, 1.0);
        let cNet = mix(c.r, c.g, wD);
        // 束对比锐化：(cNet-0.72)×2.6 + clamp(-1.2, 2.2)——宽层均值下移
        // 故增益 2.4→2.6；正峰上限 2.6→2.2 压俯视近沙面辉光漂白
        //（Round 3 差距 2：细网高频正峰叠成辉光）。
        var x = clamp((cNet - 0.72) * 2.6, -1.2, 2.2);
        shaftAcc = shaftAcc + vec3<f32>(x) * exp(-sigT * (s + z / mu)) * ds;
      }
      // 汇总：shaftAcc·sunTint·phase·sInt·K。K=12.0 为 LDR 显示域标定
      //（Round 2 差距 1）：tidewater 的 sunE·sigS·2.5 是 HDR 线性域
      //（tonemap 前合成）物理小系数，本 pass 画在 tonemap 后的 LDR
      // swap-chain 上，同系数小 ~20-50×——等效放大 K 替代 sigS 汇总
      //（Tl 内保留 sigT 谱消光 → 红衰/蓝绿束仍在）。Round 3 由 14.0 收
      // 至 12.0：配合下方前向峰加强 + 俯视减益，防太阳侧过曝压暗 Snell
      // 窗天空色。sunTint 暖阳近似。
      let ct = dot(rdLoc, LsUp);
      let g  = 0.85;
      let g2 = g * g;
      let hg = (1.0 - g2) / (12.566371 * pow(1.0 + g2 - 2.0 * g * ct, 1.5));
      let phase = 0.75 * hg + 0.25 * 0.0795775;   // 0.25/(4π)
      // 仰视朝太阳（ct>0.55，连续过渡无开关跳变）前向峰增益峰值 ×2.1、
      // 锥更窄（0.55→0.80）——Round 3 差距 1：仰视时在 Snell 窗内沿太阳
      // 方向出独立束状漏光，而非与波纹亮斑连续的拉伸条带；窄锥避开
      // 大面积波纹亮斑摊平束对比。
      let fwdBoost = 1.0 + 1.1 * smoothstep(0.55, 0.80, ct);
      let sunTint = vec3<f32>(1.0, 0.97, 0.90);
      // 俯视减益（Round 3 差距 2，Round 4 加深至 ×0.45）：低头看沙面
      // 时上方散射贡献进一步压——沙面已有真焦散网（CausticsFloor），
      // 辉光只做衬底不摊平网对比；仰视（lookDown→0）不受影响，连续
      // mix 无跳变。
      let downDamp = mix(1.0, 0.45, lookDown);
      // 仰视加权（Round 4 差距 3，Round 5 收 1.45→1.2）：抬头方向
      // 水柱短、贡献本小；但 ×1.45 会把 Snell 窗上 1/3 与束心推近纯白
      // （过曝、波纹结构被冲掉），Round 5 收敛至 ×1.2 并配合下方软
      // 限幅膝保层次。
      let upBoost = mix(1.0, 1.2, lookUp);
      sunBeam = shaftAcc * sunTint * phase * sInt * 12.0 * fwdBoost * downDamp * upBoost;
      // 束软限幅（Round 5 仰视过曝）：0.55 以下线性不动（暗束/中层
      // 保层次），之上 Reinhard 膝压缩（峰值 2 → ≈0.76、1 → ≈0.74），
      // 束不得把 Snell 窗冲成纯白。
      let beamPeak = max(sunBeam.r, max(sunBeam.g, sunBeam.b));
      sunBeam = sunBeam / (1.0 + 0.55 * max(beamPeak - 0.55, 0.0));
    }
  }

  // ── 输出：premultiplied（rgb 已乘覆盖权重），高光在薄雾区
  //    自然呈加性提亮；intensity = 0 且 sInt = 0 → rgb/alpha 全 0 →
  //    dst 原样保留（零贡献/零回归）──
  let lift   = (particles * 0.35 + shafts * 0.30) * vec3<f32>(0.55, 0.85, 0.90)
             + sunBeam;   // 体积焦散光柱：加性光（rgb 直加，premultiplied 管线）
  let rgb    = tint * fog + lift;
  // 保对比合成（Round 5 俯视漂白）：premultiplied 语义下 dst 实际
  // 贡献为 dst×(1−alpha)，故把束亮度以 0.75 计入 alpha——束亮处
  // dst 被同步压暗，束“替换”而非“白加”底色，俯视下 1/3 沙面焦散
  // 网在束下保持可读（用限幅后的 sunBeam，防 alpha 过冲）。
  let alpha  = clamp(fog + (particles + shafts) * 0.25 + dot(sunBeam, vec3<f32>(0.3333)) * 0.75, 0.0, 1.0);
  return vec4<f32>(rgb, alpha);
}
`;
