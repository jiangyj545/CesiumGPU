/**
 * 体积云层 shader（tidewater 观感追赶 Track C）。
 *
 * 两个入口：
 *  - {@link CLOUD_RENDER_WGSL}    全屏三角形渲染：相机射线 × 云高球面求交，
 *    主云（Beer-Powder 散射 + 银边 + 半球环境光 + 3 方向能量传播）与
 *    卷云第二层（纤维状 FBM，同一 pass 内先后两次球面求交 over 叠加），
 *    预乘 alpha 叠加在已合成场景之上（depthCompare equal 只写背景像素，
 *    同 AtmosphereSky）。
 *  - {@link CLOUD_SHADOW_BAKE_WGSL}  compute 正交烘焙：同款云密度场 + 卷云
 *    密度×0.3 叠加写入 rgba8unorm storage 纹理 R 通道（r8unorm 不可作
 *    storage，WebGPU 限制），供地形（LightingPass 组 3）与海面（主会话
 *    T6 接线）作云影采样。
 *
 * UBO 布局单一来源（四份，与 CloudLayer.ts 的 CPU 写入严格对应）：
 *
 *  CloudCamera（96B）——渲染专用：
 *    @0   invViewProj : mat4x4<f32>   逆 viewProj（CPU 逐帧算好上传）
 *    @64  cameraPos   : vec4<f32>     相机归一化 ECEF
 *    @80  viewport    : vec4<f32>     (w, h, 0, 0)
 *
 *  CloudParams（128B）——渲染专用：
 *    @0   sunDirection : vec4<f32>    指向太阳（归一化）
 *    @16  planeCenter  : vec4<f32>    云高平面中心（归一化 ECEF），w = 云球半径（归一化）
 *    @32  eastDir      : vec4<f32>    ENU 东基（单位方向，×EARTH_SCALE = 米）
 *    @48  northDir     : vec4<f32>    ENU 北基（同上）
 *    @64  params0      : vec4<f32>    x=coverage y=density z=horizonFade w=保留
 *    @80  params1      : vec4<f32>    xy=风漂移(公里) z=保留 w=保留
 *    @96  tintLow      : vec4<f32>    云底色（rgb, a 未用）
 *    @112 tintHigh     : vec4<f32>    云顶色（rgb, a 未用）
 *
 *  CirrusParams（48B）——渲染 FS binding 2 与烘焙 compute binding 2 共用
 *  同一 UBO（eastDir/northDir/风漂移与主云共用，无需重复传）：
 *    @0   planeCenter : vec4<f32>    xyz 卷云球心（归一化 ECEF），w = 卷云球半径（归一化）
 *    @16  params0     : vec4<f32>    x=enable(0/1) y=coverage z=density w=保留
 *    @32  params1     : vec4<f32>    x=stretchRad(弧度) y=streak(横向扰动) z=保留 w=保留
 *
 *  CloudShadowParams（96B）——烘焙 compute 与 LightingPass 组 3 消费共用：
 *    @0   planeCenter : vec4<f32>    xyz 云高平面中心（归一化 ECEF），w = 云球半径（归一化）
 *    @16  eastDir     : vec4<f32>
 *    @32  northDir    : vec4<f32>
 *    @48  params      : vec4<f32>    x=extentScale(=EARTH_SCALE/extentM) y=intensity z=coverage w=保留
 *    @64  sunDir      : vec4<f32>    指向太阳（归一化）
 *    @80  windOffset  : vec4<f32>    xy=风漂移(公里)
 *
 * 世界点 → 影贴图 uv 的仿射契约（消费方唯一公式）：
 *    d  = 云高球面交点 - planeCenter.xyz
 *    uv = vec2(dot(d, eastDir), dot(d, northDir)) * params.x + vec2(0.5)
 */

/** 归一化 ECEF：1 内部单位 = 6378137 米（Camera.EARTH_SCALE 同值）。 */
const EARTH_SCALE = 6378137.0;

export const CLOUD_CAMERA_UBO_BYTES = 96;
export const CLOUD_PARAMS_UBO_BYTES = 128;
export const CIRRUS_PARAMS_UBO_BYTES = 48;
export const CLOUD_SHADOW_PARAMS_UBO_BYTES = 96;

// ── 云密度场公共段（渲染 FS 与烘焙 compute 同一份文本）──────────────────────
const CLOUD_NOISE_WGSL = /* wgsl */`
// 2D hash / value noise / 5 octave FBM（旋转防轴向伪影）。
fn cloudHash(p: vec2<f32>) -> f32 {
  // var 而非 let：q 参与二次累加（let 不可变——真机 createShaderModule 解析期
  // 同步抛 "cannot assign to 'let q'"，整帧级联作废黑屏，2026-09-28 实测）。
  var q = fract(p * vec2<f32>(123.34, 456.21));
  q += dot(q, q + 45.32);
  return fract(q.x * q.y);
}

fn cloudVnoise(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (vec2<f32>(3.0) - 2.0 * f);
  let a = cloudHash(i);
  let b = cloudHash(i + vec2<f32>(1.0, 0.0));
  let c = cloudHash(i + vec2<f32>(0.0, 1.0));
  let d = cloudHash(i + vec2<f32>(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn cloudFbm(pIn: vec2<f32>) -> f32 {
  var p = pIn;
  var amp = 0.5;
  var sum = 0.0;
  var norm = 0.0;
  let rot = mat2x2<f32>(0.8, -0.6, 0.6, 0.8);
  for (var i: i32 = 0; i < 5; i++) {
    sum += amp * cloudVnoise(p);
    norm += amp;
    amp *= 0.5;
    p = rot * p * 2.03 + vec2<f32>(11.7, 5.3);
  }
  return sum / norm;
}

/** 云密度场（planeM：云高切平面东西/北 米坐标；windKm：风漂移 公里；
 *  aa：屏幕导数驱动的频率降档（1 = 原频率；>1 掠射/远距抗混叠）。 */
fn cloudDensity(planeM: vec2<f32>, coverage: f32, windKm: vec2<f32>, aa: f32) -> f32 {
  let q = planeM * ( 0.00042 / aa ) + windKm;
  let warp = vec2<f32>(
    cloudFbm(q * 0.5),
    cloudFbm(q * 0.5 + vec2<f32>(3.7, 9.1))
  );
  let n = cloudFbm(q + warp * 0.6);
  let thr = mix(0.74, 0.28, clamp(coverage, 0.0, 1.0));
  return smoothstep(thr, thr + 0.18, n);
}

/**
 * 卷云密度场：纤维状马尾/条纹。把 FBM 的 uv 旋转到拉伸方向坐标系后沿
 * 该方向压缩 ×0.15（噪声沿拉伸向被拉长 6.7×），叠加高频横向扰动让纤维
 * 丝摆动。与主云同坐标系（planeM 米 + windKm 漂移）；stretchRad 为切平面
 * 内拉伸方向（弧度），streak 为横向扰动强度。
 */
fn cirrusDensity(planeM: vec2<f32>, coverage: f32, windKm: vec2<f32>, stretchRad: f32, streak: f32, aa: f32) -> f32 {
  let cs = cos(stretchRad);
  let sn = sin(stretchRad);
  let p = planeM * ( 0.00042 / aa ) + windKm;
  let q = vec2<f32>(p.x * cs + p.y * sn, -p.x * sn + p.y * cs);
  // 纤维状：拉伸方向压缩 ×0.15，横向 ×1.35 提高频带密度。
  let squashed = vec2<f32>(q.x * 0.15, q.y * 1.35);
  // 高频横向扰动：纤维丝沿垂直条纹方向摆动。
  let wobble = (cloudFbm(vec2<f32>(q.x * 0.4, q.y * 2.6)) - 0.5) * streak;
  let n = cloudFbm(squashed + vec2<f32>(0.0, wobble));
  let thr = mix(0.68, 0.3, clamp(coverage, 0.0, 1.0));
  return smoothstep(thr, thr + 0.22, n);
}
`;

// ── 渲染入口 ─────────────────────────────────────────────────────────────────
export const CLOUD_RENDER_WGSL = /* wgsl */`
struct CloudCamera {
  invViewProj : mat4x4<f32>,
  cameraPos   : vec4<f32>,
  viewport    : vec4<f32>,
}

struct CloudParams {
  sunDirection : vec4<f32>,
  planeCenter  : vec4<f32>,
  eastDir      : vec4<f32>,
  northDir     : vec4<f32>,
  params0      : vec4<f32>,
  params1      : vec4<f32>,
  tintLow      : vec4<f32>,
  tintHigh     : vec4<f32>,
}

// 卷云第二层参数（与烘焙 compute binding 2 共用同一 UBO，布局见头注释）。
struct CirrusParams {
  planeCenter : vec4<f32>,
  params0     : vec4<f32>,
  params1     : vec4<f32>,
}

@group(0) @binding(0) var<uniform> cloudCam : CloudCamera;
@group(0) @binding(1) var<uniform> cloud    : CloudParams;
@group(0) @binding(2) var<uniform> cirrus   : CirrusParams;

const CLOUD_EARTH_SCALE: f32 = ${EARTH_SCALE.toFixed(1)};

// Beer 消光系数（Beer-Powder 散射能量 exp(-dens*σ) 的光学深度标定）。
const CLOUD_EXTINCTION: f32 = 3.2;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0)       ndc : vec2<f32>,
}

// 全屏三角形；NDC z = 1 恰与 GBuffer 背景深度相等（depthCompare equal 透传）。
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  var out: VSOut;
  out.pos = vec4<f32>(positions[vid], 1.0, 1.0);
  out.ndc = positions[vid];
  return out;
}

${CLOUD_NOISE_WGSL}

// 最近正交向的云高球面交点距离；无交返回 -1。
fn cloudRaySphereNearest(ro: vec3<f32>, rd: vec3<f32>, radius: f32) -> f32 {
  let b = dot(ro, rd);
  let c = dot(ro, ro) - radius * radius;
  let disc = b * b - c;
  if (disc < 0.0) { return -1.0; }
  let sq = sqrt(disc);
  let t0 = -b - sq;
  if (t0 > 0.0) { return t0; }
  let t1 = -b + sq;
  if (t1 > 0.0) { return t1; }
  return -1.0;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  // 逆 viewProj 重建世界射线（far 平面点 → 方向）。
  let farH = cloudCam.invViewProj * vec4<f32>(in.ndc, 1.0, 1.0);
  let far = farH.xyz / farH.w;
  let ro = cloudCam.cameraPos.xyz;
  let rd = normalize(far - ro);
  let up = normalize(ro);

  // 地平线渐隐：贴地平的掠射交点拉得极远，alpha 压零防摩尔纹（两层共用）。
  let el = dot(rd, up);
  let horizon = smoothstep(0.015, 0.015 + 0.105 * max(cloud.params0.z, 0.001), el);

  // 掠射抗混叠（两层共用预备）：交点米坐标的屏幕导数模长 → 噪声频率降档
  //（点阵颗粒根因是各向同性 FBM 在掠射拉伸下欠采样；AA 除子 1 = 原频率）。
  // dpdx/dpdy 只能在 uniform control flow 调用（真机 "must only be called
  // from uniform control flow"，2026-09-29）——求交/坐标/导数全部排在
  // discard 之前的无条件流；hit<0 时坐标无效但后续分支不使用。
  let hitC = cloudRaySphereNearest(ro, rd, cirrus.planeCenter.w);
  let hit  = cloudRaySphereNearest(ro, rd, cloud.planeCenter.w);
  let dC = ro + rd * select(0.0, hitC, hitC > 0.0) - cirrus.planeCenter.xyz;
  let d  = ro + rd * select(0.0, hit,  hit  > 0.0) - cloud.planeCenter.xyz;
  let planeMC = vec2<f32>(
    dot(dC, cloud.eastDir.xyz),
    dot(dC, cloud.northDir.xyz)
  ) * CLOUD_EARTH_SCALE;
  let planeM = vec2<f32>(
    dot(d, cloud.eastDir.xyz),
    dot(d, cloud.northDir.xyz)
  ) * CLOUD_EARTH_SCALE;
  let fwC = max( length( dpdx( planeMC ) ), length( dpdy( planeMC ) ) );
  let fw  = max( length( dpdx( planeM  ) ), length( dpdy( planeM  ) ) );
  let aaC = clamp( fwC / 900.0 + 1.0, 1.0, 24.0 );
  let aa  = clamp( fw  / 900.0 + 1.0, 1.0, 24.0 );

  if (horizon <= 0.002) { discard; }

  // 预乘域远→近 over 累积：先卷云（高、远），再主云（低、近）。
  var premul = vec3<f32>(0.0);
  var alpha = 0.0;

  // ── 卷云第二层：更高云高球面求交 + 纤维状噪声，alpha 压 ≤ 0.5 ──
  if (cirrus.params0.x > 0.5 && hitC > 0.0) {
    {
      let cDens = cirrusDensity(
        planeMC, cirrus.params0.y, cloud.params1.xy,
        cirrus.params1.x, cirrus.params1.y, aaC
      );
      if (cDens > 0.004) {
        // 卷云冰晶薄层：正午近白，黄昏染金橙（卷云比主云更受低角度阳光）。
        let duskC = 1.0 - clamp(dot(cloud.sunDirection.xyz, up) * 1.6, 0.0, 1.0);
        var cCol = mix(vec3<f32>(1.04, 1.02, 1.0), vec3<f32>(1.25, 0.6, 0.38), duskC * 0.9);
        // 卷云银边：视线朝太阳时纤维丝前向散射增亮。
        let cSilver = pow(clamp(dot(rd, cloud.sunDirection.xyz), 0.0, 1.0), 4.0) * (1.0 - cDens);
        cCol += vec3<f32>(cSilver * 0.6);
        let cA = min(clamp(cDens * cirrus.params0.z, 0.0, 1.0), 0.5) * horizon;
        premul = cCol * cA + premul * (1.0 - cA);
        alpha = cA + alpha * (1.0 - cA);
      }
    }
  }

  // ── 主云层 ──
  if (hit > 0.0) {
    let coverage = cloud.params0.x;
    let dens = cloudDensity(planeM, coverage, cloud.params1.xy, aa);
    if (dens > 0.004) {
      // 朝太阳切平面方向（两层共用 ENU 基）。
      let sun2 = vec2<f32>(
        dot(cloud.sunDirection.xyz, cloud.eastDir.xyz),
        dot(cloud.sunDirection.xyz, cloud.northDir.xyz)
      );
      let sunLen = length(sun2);
      let sunL2 = select(-sun2 / max(sunLen, 1e-4), vec2<f32>(0.0), sunLen < 1e-4);

      // 多方向能量传播近似：太阳向 + 平面内 ±60° 正交抖动，加权差分
      //（+2 次 FBM 采样）——比单梯度差分更平滑的方向性光照。
      let perp2 = vec2<f32>(-sunL2.y, sunL2.x);
      let sideA = (sunL2 + perp2 * 1.732) * 0.5;
      let sideB = (sunL2 - perp2 * 1.732) * 0.5;
      let dSun = cloudDensity(planeM + sunL2 * 140.0, coverage, cloud.params1.xy, aa);
      let dSideA = cloudDensity(planeM + sideA * 160.0, coverage, cloud.params1.xy, aa);
      let dSideB = cloudDensity(planeM + sideB * 160.0, coverage, cloud.params1.xy, aa);
      let grad3 = (dSun - dens) * 0.5 + (dSideA - dens) * 0.25 + (dSideB - dens) * 0.25;
      let directional = clamp(1.0 + grad3 * 2.2, 0.35, 1.65);

      // Beer-Powder 散射能量：穿透项 exp(-dens*σ) × 粉末项
      // (1 - exp(-dens*σ*2))——峰形曲线让云心更亮、边缘透光。
      let beer = exp(-dens * CLOUD_EXTINCTION);
      let powder = 1.0 - exp(-dens * CLOUD_EXTINCTION * 2.0);
      let beerPowder = beer * powder;

      // 云底环境光：半球近似（天顶向云受天光多、地平向偏暗蓝）×0.35。
      let hemi = 0.5 + 0.5 * dot(rd, up);
      let ambient = mix(cloud.tintLow.rgb, cloud.tintHigh.rgb, hemi) * 0.35;

      // 银边 silver lining：视线朝太阳穿薄云区的向阳轮廓发丝光
      //（薄处 1-dens 增强，厚处无）。
      let silver = pow(clamp(dot(rd, cloud.sunDirection.xyz), 0.0, 1.0), 4.0) * (1.0 - dens) * 0.85;

      // 太阳高度染色：正午近云顶白，黄昏整体染橙红（transmittance 近似）。
      let sunUp = dot(cloud.sunDirection.xyz, up);
      let dusk = 1.0 - clamp(sunUp * 1.6, 0.0, 1.0);
      var mCol = mix(cloud.tintHigh.rgb, cloud.tintLow.rgb, clamp(dens, 0.0, 1.0));
      // 太阳直接散射：Beer-Powder 能量（峰值 ≈ 0.38，×2.6 归一）× 方向传播。
      mCol = mCol * (ambient + vec3<f32>(beerPowder * 2.6 * directional));
      // 银边加色（黄昏偏金），随整体 dusk 透射染色乘法耦合。
      mCol += vec3<f32>(silver) * mix(vec3<f32>(1.0, 0.98, 0.94), vec3<f32>(1.2, 0.72, 0.42), dusk);
      mCol = mix(mCol, mCol * vec3<f32>(1.3, 0.55, 0.32), dusk * 0.85);

      let a = clamp(dens * cloud.params0.y, 0.0, 1.0) * horizon;
      premul = mCol * a + premul * (1.0 - a);
      alpha = a + alpha * (1.0 - a);
    }
  }

  if (alpha <= 0.0015) { discard; }
  return vec4<f32>(premul, alpha); // premul 已含 alpha（预乘输出）
}
`;

// ── 云影烘焙 compute ─────────────────────────────────────────────────────────
export const CLOUD_SHADOW_BAKE_WGSL = /* wgsl */`
struct CloudShadowParams {
  planeCenter : vec4<f32>,
  eastDir     : vec4<f32>,
  northDir    : vec4<f32>,
  params      : vec4<f32>,
  sunDir      : vec4<f32>,
  windOffset  : vec4<f32>,
}

// 卷云参数（与渲染 FS binding 2 共用同一 UBO，布局见头注释）。
struct CirrusParams {
  planeCenter : vec4<f32>,
  params0     : vec4<f32>,
  params1     : vec4<f32>,
}

@group(0) @binding(0) var<uniform> shadow : CloudShadowParams;
@group(0) @binding(2) var<uniform> cirrus : CirrusParams;
// r8unorm 不可作 storage（WebGPU 限制）：rgba8unorm 写 R 通道。
@group(0) @binding(1) var shadowTex : texture_storage_2d<rgba8unorm, write>;

${CLOUD_NOISE_WGSL}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(shadowTex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let uv01 = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
    / vec2<f32>(f32(dims.x), f32(dims.y));
  // uv → 覆盖区平面米坐标（params.x = EARTH_SCALE / extentM）。
  let planeM = (uv01 - vec2<f32>(0.5)) / max(shadow.params.x, 1e-9);
  // 烘焙域固定网格采样（无屏幕导数），aa = 1 保持原频率（逐字节不变）。
  let dens = cloudDensity(planeM, shadow.params.z, shadow.windOffset.xy, 1.0);
  // 卷云投影叠加（低权重 ×0.3）：地面上淡淡纤维条纹影。
  var total = dens;
  if (cirrus.params0.x > 0.5) {
    let cD = cirrusDensity(
      planeM, cirrus.params0.y, shadow.windOffset.xy,
      cirrus.params1.x, cirrus.params1.y, 1.0
    );
    total = clamp(dens + cD * 0.3, 0.0, 1.0);
  }
  textureStore(shadowTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(total, total, total, 1.0));
}
`;

// ── 消费模块生成器（与 CLOUD_SHADOW_BAKE_WGSL 的 CloudShadowParams 布局
//    逐字段一致，96B）────────────────────────────────────────────────────────
// 消费方按各自 bind group 组号/绑定号实例化：
//  - LightingPass 地形（LightingShader 锚点注入）：组 3，UBO@0 / tex@1 / samp@2
//  - OceanFFTWater 海面（OCEAN_FFT_RENDER_SHADER 拼接）：组 0，
//    UBO@21 / tex@20 / samp@22
export function cloudShadowReceiverModule(
  group: number,
  uboBinding: number,
  texBinding: number,
  sampBinding: number,
): string {
  return /* wgsl */`
struct CloudShadowParams {
  planeCenter : vec4<f32>,
  eastDir     : vec4<f32>,
  northDir    : vec4<f32>,
  params      : vec4<f32>,
  sunDir      : vec4<f32>,
  windOffset  : vec4<f32>,
}

@group(${group}) @binding(${uboBinding}) var<uniform> cloudShadowParams : CloudShadowParams;
@group(${group}) @binding(${texBinding}) var cloudShadowTex  : texture_2d<f32>;
@group(${group}) @binding(${sampBinding}) var cloudShadowSamp : sampler;

/**
 * 世界点云影因子（1 = 无影）。地面点沿 UBO 太阳方向上投云高球面，
 * 平面仿射取 uv 采样烘焙密度：光照 *= 1 - density * intensity。
 * intensity <= 0 短路恒 1（未绑定/关闭路径零影响）。
 */
fn cloudShadowFactor(worldPos: vec3<f32>) -> f32 {
  let intensity = cloudShadowParams.params.y;
  if (intensity <= 0.0) { return 1.0; }
  let rc = cloudShadowParams.planeCenter.w;
  let b = dot(worldPos, cloudShadowParams.sunDir.xyz);
  let c = dot(worldPos, worldPos) - rc * rc;
  let disc = b * b - c;
  if (disc <= 0.0) { return 1.0; }
  let sq = sqrt(disc);
  var t = -b - sq;
  if (t <= 0.0) { t = -b + sq; }
  if (t <= 0.0) { return 1.0; }
  let d = worldPos + cloudShadowParams.sunDir.xyz * t - cloudShadowParams.planeCenter.xyz;
  let uv = vec2<f32>(
    dot(d, cloudShadowParams.eastDir.xyz),
    dot(d, cloudShadowParams.northDir.xyz)
  ) * cloudShadowParams.params.x + vec2<f32>(0.5);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { return 1.0; }
  let dens = textureSampleLevel(cloudShadowTex, cloudShadowSamp, uv, 0.0).r;
  return clamp(1.0 - dens * intensity, 0.0, 1.0);
}
`;
}

/** LightingPass 地形消费实例（组 3；原导出逐字节不变）。 */
export const CLOUD_SHADOW_RECEIVER_MODULE = cloudShadowReceiverModule(3, 0, 1, 2);
