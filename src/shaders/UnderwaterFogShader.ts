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
 * 四个视觉成分（全部程序生成，零纹理采样）：
 *   - 蓝绿深度雾：水体染色，透明度随俯仰/暗角渐变；
 *   - 浮动颗粒：两层屏幕空间 hash 点阵，慢速漂移 + 闪烁，密度由
 *     particleDensity 调制；
 *   - 焦散光斑：两层正弦波纹相乘调制亮度（呼应 OceanCaustics 的观感，
 *     但独立实现），抬头方向权重更大；
 *   - 水下太阳光柱（Track G 联动）：太阳方向投影到屏幕位，视线→太阳夹角
 *     用「像素到太阳屏幕位的距离」作代理，Henyey-Greenstein（g≈0.7）前向
 *     峰值 × 径向角条纹 × 已有 shimmer（波浪间隙漏光），sunShaftIntensity
 *     默认 0 = 完全关闭（零回归）。
 *
 * Uniform 布局：
 *   binding 0 : CameraUniforms（256 B，布局单一来源 src/renderer/CameraUniforms.ts）
 *   binding 1 : UWParams（64 B = 4 x vec4）
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
}

@group(0) @binding(0) var<uniform> cam : CameraUniforms;
@group(0) @binding(1) var<uniform> uw  : UWParams;
// G-Buffer worldPos（rgba32float，归一化 ECEF；unfilterable → non-filtering
// 采样 + textureSampleLevel）。距离衰减雾的水下路径长来源。
@group(0) @binding(2) var scenePosTex : texture_2d<f32>;
@group(0) @binding(3) var posSmp      : sampler;

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
  if ( uw.params0.w > 0.5 ) {
    // 世界射线（与 AtmosphereSky.fs_main 同式：view 行 = 相机基，proj 对角）
    let camR = vec3<f32>(cam.viewMatrix[0][0], cam.viewMatrix[1][0], cam.viewMatrix[2][0]);
    let camU = vec3<f32>(cam.viewMatrix[0][1], cam.viewMatrix[1][1], cam.viewMatrix[2][1]);
    let camF = vec3<f32>(-cam.viewMatrix[0][2], -cam.viewMatrix[1][2], -cam.viewMatrix[2][2]);
    let rd = normalize(camR * (ndc.x / cam.projMatrix[0][0])
                     + camU * (ndc.y / cam.projMatrix[1][1]) + camF);
    let wp = textureSampleLevel( scenePosTex, posSmp, in.uv, 0.0 ).xyz;
    var distPx : f32;
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
      // 远景仍沉入深青。
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

  // ── 4、水下太阳光柱（Track G 联动，god rays 的水下侧近似）───────────
  // 太阳方向投到屏幕（相机 + 无穷远点过 viewProj），视线→太阳夹角用
  // 「像素到太阳屏幕位的距离」作屏幕空间代理；HG(g=0.7) 前向峰值近似
  // 体积光各向异性，条纹沿径向角展开并乘已有 shimmer（波浪间隙漏光）。
  var sunBeam = vec3<f32>(0.0);
  let sInt = clamp(uw.sunShaft.w, 0.0, 4.0);
  if (sInt > 0.0) {
    // 太阳须在水面上方才进光（sunDir·up > 0；安全归一化防零向量）
    let sunDir = uw.sunShaft.xyz / max(length(uw.sunShaft.xyz), 1e-6);
    let above  = clamp(dot(sunDir, up) * 3.0, 0.0, 1.0);
    if (above > 0.0) {
      // sunPt 而非 target：target 是 WGSL 保留关键字（守卫捕获，同 SwashComputeShader）
      let sunPt  = cam.cameraPosition.xyz + sunDir * 10000.0;
      let clip   = cam.viewProjMatrix * vec4<f32>(sunPt, 1.0);
      if (clip.w > 0.0) {
        let sunUv = vec2<f32>(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
        let toSun = (sunUv - in.uv) * vec2<f32>(aspect, 1.0);
        let d     = length(toSun);
        // HG(g≈0.7)：cosθ 用 1-d² 近似（小屏幕角 → 前向峰值）
        let g   = 0.7;
        let g2  = g * g;
        let ct  = max(1.0 - d * d, 0.0);
        let hg  = (1.0 - g2) / (12.566371 * pow(max(1.0 + g2 - 2.0 * g * ct, 1e-3), 1.5));
        // 条纹：径向角 + 距离的正弦条纹，随时间缓动；抬头方向权重更大
        let ang    = atan2(toSun.y, toSun.x);
        let streak = 0.55 + 0.45 * sin(ang * 9.0 + d * 14.0 - t * 0.7);
        sunBeam = vec3<f32>(hg * streak * shimmer) * above * sInt * (0.35 + 0.65 * lookUp);
      }
    }
  }

  // ── 输出：premultiplied（rgb 已乘覆盖权重），高光在薄雾区
  //    自然呈加性提亮；intensity = 0 且 sInt = 0 → rgb/alpha 全 0 →
  //    dst 原样保留（零贡献/零回归）──
  let lift   = (particles * 0.35 + shafts * 0.30) * vec3<f32>(0.55, 0.85, 0.90)
             + sunBeam * 0.5 * vec3<f32>(0.55, 0.85, 0.90);
  let rgb    = tint * fog + lift;
  let alpha  = clamp(fog + (particles + shafts) * 0.25 + dot(sunBeam, vec3<f32>(0.3333)) * 0.4, 0.0, 1.0);
  return vec4<f32>(rgb, alpha);
}
`;
