/**
 * DynamicWaterShader - 动态水面 WGSL（对齐 EV-Globe 的 EV_DynamicWater / EV_Ocean）。
 *
 * 视觉语义（2026-10-22 热带白天档，对齐 threejswaterpro 默认观感）：
 *   - 六层解析行波（620/380/210/120/65/34 m，方向两两交叉）+ 低频 warp /
 *     振幅调制：不规则涌浪 + 碎浪，而不是等距平行条纹；
 *   - 解析天空渐变（天顶蓝 → 地平线灰白）+ 太阳前向 Mie 光晕作为反射色，
 *     Schlick pow5 菲涅尔（用宏观视角主导，避免小坡度把整片海拉成镜面条带）；
 *   - 太阳 glitter：高指数锐瓣 + 距离加宽（抗闪），近景细碎闪点；
 *   - 白沫：波峰高度 × 坡度（Jacobian 近似）× 高频破碎斑块，覆盖率 1-2%；
 *   - 距离雾 8-30 km（掠射视线按气柱长度加权）：淡入该方位地平线天空色，
 *     海天同色、消矩形硬边。
 *
 * 遮挡：特效画在 swapchain 上（IPass 无深度附件），片元级两级剔除：
 *   1) 射线-单位球求交（与 DynamicWallShader 同款）：剔除地球远侧水面；
 *   2) 切平面地平线判定：相机落在片元当地水平面之下 → 该片元在地平线之外
 *      （低空机位的关键；射线-球只对穿地球的远侧生效）。
 *
 * 坐标：worldPos / cameraPosition 为归一化 ECEF（赤道半径 = 1），
 * ×6378137 得米；颜色为 display-referred（直写非 sRGB swapchain）。
 */
export const DYNAMIC_WATER_SHADER = /* wgsl */`
// ── Uniforms ──────────────────────────────────────────────────────────────────

// 256B 相机 UBO — 布局单一来源：src/renderer/CameraUniforms.ts
struct CameraUniforms {
  viewMatrix      : mat4x4<f32>,   // @0   (64B)
  projMatrix      : mat4x4<f32>,   // @64  (64B)
  viewProjMatrix  : mat4x4<f32>,   // @128 (64B)
  cameraPosition  : vec4<f32>,     // @192 (16B)
  viewportSize    : vec2<f32>,     // @208 (8B)
  nearFar         : vec2<f32>,     // @224 (8B) x = near, y = far
}

// 64B：color(16) + sunDir(16) + time/speed/waveScale/glint(16)。
struct WaterUniforms {
  color      : vec4<f32>,   // 水体基色 RGBA（a = 近景不透明度）
  sunDir     : vec4<f32>,   // 太阳方向（指向太阳，归一化 ECEF）
  time       : f32,         // 特效时间（秒）
  speed      : f32,         // 波速倍率
  waveScale  : f32,         // 波高（法线扰动强度）
  glint      : f32,         // 波光高光强度
}

@group(0) @binding(0) var<uniform> camera : CameraUniforms;
@group(0) @binding(1) var<uniform> water  : WaterUniforms;

// 遮挡球半径²：0.998² = 0.996004 大于压扁世界地表模长下界
// polarRatio² ≈ 0.99331——中高纬贴地片元被划进球内，相机升到球外
// 俯视时整面 discard。取 0.9945² < 下界，贴地片元恒在球外
//（对齐 OceanFFTShader / DynamicWallShader 常量约定）。
const GLOBE_OCCLUDE_R2 : f32 = 0.9945 * 0.9945;

// 归一化 ECEF → 米。
const EARTH_R_M : f32 = 6378137.0;

// 天空色（display-referred sRGB 量值）：**实测** 48 页 AtmosphereSky（520 m，
// UI 隐藏，sunIntensity 9.6 / rayleighScale 1.08）：
// 仰角 0° ≈ 114/126/133、20-30° ≈ 107/141/175。
// 地平线常量必须与物理天空同源，否则水面距离雾会在几何地平线处补出
// 比天空亮/暗的色墙或暗带（第 2/3 轮检查阻断项；k2 掠海视角尤其敏感）。
const SKY_ZENITH  : vec3<f32> = vec3<f32>(0.420, 0.553, 0.686);
const SKY_HORIZON : vec3<f32> = vec3<f32>(0.447, 0.494, 0.522);
const SUN_TINT    : vec3<f32> = vec3<f32>(1.000, 0.945, 0.840);
const FOAM_TINT   : vec3<f32> = vec3<f32>(0.900, 0.930, 0.940);

// ── 程序化噪声（warp / 白沫斑块）────────────────────────────────────────────

fn hash2(p : vec2<f32>) -> f32 {
  let q = fract(p * vec2<f32>(0.3183099, 0.3678794) + vec2<f32>(0.71, 0.113));
  return fract(43758.5453 * dot(q, vec2<f32>(23.14, 71.37)));
}

fn vnoise(p : vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash2(i), hash2(i + vec2<f32>(1.0, 0.0)), u.x),
    mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x),
    u.y
  );
}

// ── 解析天空：视线方向 → 天空色（天顶→地平线渐变 + 太阳前向 Mie）─────────

fn skyColor(dir : vec3<f32>, upDir : vec3<f32>) -> vec3<f32> {
  // el = 视线相对当地天顶的高度角分量（1 天顶 / 0 地平线 / <0 地平线下）
  let el = dot(dir, upDir);
  // 物理天空的蓝集中在 5-30° 仰角，地平线附近迅速转灰白（指数 4.5 拟合）
  let h  = pow(clamp(1.0 - el, 0.0, 1.0), 4.5);
  var c  = mix(SKY_ZENITH, SKY_HORIZON, h);
  // 太阳前向散射：宽晕（大范围提亮）+ 锐晕（贴盘光核）
  let cs = max(dot(dir, water.sunDir.xyz), 0.0);
  c += SUN_TINT * (0.018 * pow(cs, 6.0) + 0.18 * pow(cs, 120.0));
  return c;
}

// 2D 旋转（波向按噪声缓慢偏转，打散等距平行波峰）
fn rot2(v : vec2<f32>, a : f32) -> vec2<f32> {
  let c = cos(a);
  let s = sin(a);
  return vec2<f32>(v.x * c - v.y * s, v.x * s + v.y * c);
}

// ── 波场：返回 (切平面坡度 xy, 波高 z, 破碎度 w) ──────────────────────────
// 六层解析行波：波长跨 3 个量级（620 → 34 m）、方向两两交叉、振幅按波长
// 递减；每层按相机距离淡出（>8 km 只留长涌浪 → 远场趋平坦镜面，无摩尔纹）。
// 波数 k = 2π·EARTH_R_M / λ（rad / 归一化世界单位）；最短 34 m 的相位量级
// ~1.1e6，f32 相位误差 < 0.07 rad（< 0.4 m 空间误差，亚像素）。

fn waveField(p : vec2<f32>, t : f32, distM : f32) -> vec4<f32> {
  var g = vec2<f32>(0.0, 0.0);
  var h = 0.0;

  // 低频 warp：打散等距平行波峰（真实海面的破碎感）
  let warp = vec2<f32>(
    vnoise(p * 1900.0 + vec2<f32>(t * 0.010, 0.0)),
    vnoise(p * 1900.0 + vec2<f32>(37.7, t * 0.010))
  ) - 0.5;
  let pw = p + warp * 0.00022;
  // 高频 warp（~0.5 km 尺度）：短波波峰不被拉成连续长条
  let warp2 = vec2<f32>(
    vnoise(p * 14000.0 + vec2<f32>(t * 0.030, 0.0)),
    vnoise(p * 14000.0 + vec2<f32>(71.3, t * 0.030))
  ) - 0.5;
  let pw2 = p + warp2 * 0.00005;
  // 振幅调制：2-3 km 级的涌浪群/平静带
  let am = 0.72 + 0.56 * vnoise(p * 2400.0 + vec2<f32>(11.3, 5.1));
  let am2 = 0.72 + 0.56 * vnoise(p * 9000.0 + vec2<f32>(3.7, 19.1));
  let am3 = 0.50 + 0.90 * vnoise(p * 26000.0 + vec2<f32>(23.9, 7.7));
  // 距离淡出边界的噪声抖动：各层淡出若只按距离，会在地平线方向形成同心环
  let fadeMod = 0.72 + 0.56 * vnoise(p * 5000.0 + vec2<f32>(5.5, 1.2));

  // L0 λ≈620 m 长涌浪（全域）
  var d  = vec2<f32>(0.92, 0.39);
  var ph = dot(p, d) * 64640.0 + t * 0.45;
  g += d * (0.040 * cos(ph) * am);
  h += 0.50 * sin(ph) * am;

  // L1 λ≈380 m 交叉涌浪（26 km 外淡出；波向/波长按 2-3 km 噪声缓慢偏转）
  let f1 = 1.0 - smoothstep(12000.0, 26000.0, distM);
  let a1 = (vnoise(p * 4800.0 + vec2<f32>(9.1, 2.7)) - 0.5) * 1.5;
  let k1 = 105460.0 * (0.80 + 0.40 * vnoise(p * 2200.0 + vec2<f32>(4.4, 8.8)));
  d  = rot2(vec2<f32>(-0.30, 0.95), a1);
  ph = dot(pw, d) * k1 + t * 0.75;
  g += d * (0.050 * cos(ph) * am * f1);
  h += 0.34 * sin(ph) * am * f1;

  // L2 λ≈210 m（18 km 外淡出；波峰近似顺视线 → 竖向退行纹理，破横条）
  let f2 = 1.0 - smoothstep(8000.0, 18000.0, distM);
  let a2 = (vnoise(p * 7000.0 + vec2<f32>(1.3, 15.9)) - 0.5) * 1.8;
  let k2 = 190860.0 * (0.78 + 0.44 * vnoise(p * 3100.0 + vec2<f32>(8.2, 3.1)));
  d  = rot2(vec2<f32>(0.65, 0.76), a2);
  ph = dot(pw, d) * k2 + t * 1.05;
  g += d * (0.060 * cos(ph) * am * f2);
  h += 0.24 * sin(ph) * am * f2;

  // L3 λ≈120 m（7 km 外淡出）
  let f3 = 1.0 - smoothstep(2500.0 * fadeMod, 8000.0 * fadeMod, distM);
  let a3 = (vnoise(p * 12000.0 + vec2<f32>(6.6, 21.4)) - 0.5) * 1.6;
  d  = rot2(vec2<f32>(-0.75, 0.66), a3);
  ph = dot(pw2, d) * 334000.0 + t * 1.55;
  g += d * (0.075 * cos(ph) * am2 * am3 * f3);
  h += 0.16 * sin(ph) * am2 * f3;

  // L4 λ≈65 m（3.5 km 外淡出）
  let f4 = 1.0 - smoothstep(1200.0 * fadeMod, 4200.0 * fadeMod, distM);
  d  = vec2<f32>(0.40, 0.92);
  ph = dot(pw2, d) * 616600.0 + t * 2.20;
  g += d * (0.085 * cos(ph) * am2 * am3 * f4);
  h += 0.10 * sin(ph) * am2 * f4;

  // L5 λ≈34 m 碎浪（1.8 km 外淡出，近景闪点来源）
  let f5 = 1.0 - smoothstep(600.0 * fadeMod, 2200.0 * fadeMod, distM);
  d  = vec2<f32>(-0.93, 0.37);
  ph = dot(pw2, d) * 1178900.0 + t * 3.00;
  g += d * (0.095 * cos(ph) * am2 * am3 * f5);
  h += 0.06 * sin(ph) * am2 * f5;

  // L6 λ≈27 m 细浪（1.5 km 内，交叉向 → 打散 L5 的规则脊线）
  let f6 = 1.0 - smoothstep(400.0 * fadeMod, 1500.0 * fadeMod, distM);
  d  = vec2<f32>(-0.35, 0.94);
  ph = dot(pw2, d) * 1480000.0 + t * 3.40;
  g += d * (0.060 * cos(ph) * am3 * f6);
  h += 0.04 * sin(ph) * am3 * f6;

  // L7 λ≈46 m 细浪（2.6 km 内）
  let f7 = 1.0 - smoothstep(800.0 * fadeMod, 2600.0 * fadeMod, distM);
  d  = vec2<f32>(0.97, -0.24);
  ph = dot(pw2, d) * 870000.0 + t * 2.80;
  g += d * (0.070 * cos(ph) * am2 * am3 * f7);
  h += 0.05 * sin(ph) * am3 * f7;

  // L8 非周期碎浪：value-noise 直接给坡度（无周期 → 消自相关周期峰）
  let chopF = 1.0 - smoothstep(2500.0, 7000.0, distM);
  let chop = vec2<f32>(
    vnoise(pw2 * 90000.0 + vec2<f32>(t * 0.22, 0.0)),
    vnoise(pw2 * 90000.0 + vec2<f32>(19.7, t * 0.22))
  ) - 0.5;
  g += chop * (0.11 * am3 * chopF);
  h += (chop.x + chop.y) * 0.04;

  // waveScale 0-1 → 坡度增益 0.5-1.34（平静海况法线仍保留可读纹理）
  let s = 0.5 + water.waveScale * 1.4;
  let n = normalize(vec3<f32>(-g.x * s, -g.y * s, 1.0));
  // 破碎度：波峰 + 陡坡（Jacobian 近似），供白沫使用
  let brk = h * 0.62 + length(g) * s * 1.15;
  return vec4<f32>(n.xy, h, brk);
}

// ── Vertex ────────────────────────────────────────────────────────────────────

struct VSIn {
  @location(0) position : vec3<f32>,
  @location(1) uv       : vec2<f32>,
}

struct VSOut {
  @builtin(position) clipPos  : vec4<f32>,
  @location(0)       worldPos : vec3<f32>,
}

@vertex
fn vs_main(input : VSIn) -> VSOut {
  var out : VSOut;
  var clip = camera.viewProjMatrix * vec4<f32>(input.position, 1.0);
  // 远场整体上移最多 3 px：盖住 GBuffer 地形裙边/粗 LOD 抬边在地平线上
  // 留下的 1-2 px 深色缝（水面切平面无法越过几何地平线）；近景不移动，
  // 避免画面底边露出底图。
  let dist = length(input.position - camera.cameraPosition.xyz) * EARTH_R_M;
  let liftPx = 6.0 * smoothstep(2000.0, 10000.0, dist);
  clip.y += (liftPx / max(camera.viewportSize.y, 1.0)) * clip.w;
  out.clipPos  = clip;
  out.worldPos = input.position;
  return out;
}

// ── Fragment ──────────────────────────────────────────────────────────────────

@fragment
fn fs_main(input : VSOut) -> @location(0) vec4<f32> {
  // ── 遮挡 1：射线-单位球求交（保留旧不变量）────────────────────────────
  let camPos = camera.cameraPosition.xyz;
  let oc   = input.worldPos - camPos;
  let a    = dot(oc, oc);
  let b    = 2.0 * dot(camPos, oc);
  let c    = dot(camPos, camPos) - GLOBE_OCCLUDE_R2;
  let disc = b * b - 4.0 * a * c;
  if (disc > 0.0) {
    let t = (-b - sqrt(disc)) / (2.0 * a);
    if (t > 1e-3 && t < 1.0 - 1e-3) {
      discard;
    }
  }

  // ── 遮挡 2：切平面地平线（相机在片元当地水平面之下 → 该片元在
  //    地平线之外；低空机位靠它把球面背侧水面挡在天空之外）────────────
  let fragR = length(input.worldPos);
  if (dot(camPos, input.worldPos) < fragR * fragR) {
    discard;
  }

  // ── 基础量 ────────────────────────────────────────────────────────────
  let distM   = length(oc) * EARTH_R_M;
  let viewDir = normalize(camPos - input.worldPos);
  let upDir   = normalize(camPos);
  // 当地 ENU 基（世界 ECEF 里 +Z 是北极，不是"上"——法线必须经基变换，
  // 旧版直接以 +Z 当水面法线会让整片海 fresnel 判定为背光）。
  let surfUp = normalize(input.worldPos);
  // 波场基必须**整屏恒定**（取相机当地 ENU）：若用片元自身的 east/north，
  // 基随位置旋转，波峰会被拧成同心圆环（180 m 机位实测）。
  let upB   = upDir;
  let eRaw  = vec3<f32>(-upB.y, upB.x, 0.0);
  let eLen  = length(eRaw);
  let east  = select(vec3<f32>(1.0, 0.0, 0.0), eRaw / max(eLen, 1e-6), eLen > 1e-4);
  let north = cross(upB, east);
  // 用世界坐标点积（不减相机位置）：基恒定 → 相位世界锁定，镜头平移不游移
  let waveP = vec2<f32>(dot(input.worldPos, east), dot(input.worldPos, north));
  let wf    = waveField(waveP, water.time * max(water.speed, 0.001), distM);
  // 坡面法线（切平面 x/y）→ 世界法线
  let n     = normalize(surfUp + east * wf.x + north * wf.y);
  let h     = wf.z;
  let brk   = wf.w;

  // ── 白沫：白冠（硬边）+ 高频碎裂纹理；不铺软抹痕 ───────────────────────
  let foamPatch = 0.45 + 0.55 * vnoise(waveP * 2600.0 + vec2<f32>(water.time * 0.03, 0.0));
  let foamTexFine = vnoise(waveP * 1500000.0 + vec2<f32>(water.time * 0.25, 0.0));  // ~4 m
  let foamTexMid  = vnoise(waveP * 220000.0 + vec2<f32>(0.0, water.time * 0.18));   // ~29 m
  let foamTex = mix(foamTexMid, foamTexFine, 1.0 - smoothstep(1200.0, 4200.0, distM));
  // ~2 m 侵蚀噪声：把白冠边缘咬碎（不是纸片剪影）
  let foamErode = vnoise(waveP * 2800000.0 + vec2<f32>(water.time * 0.35, 7.1));
  let foam = smoothstep(0.70, 0.80, brk + (foamPatch - 0.5) * 0.22)
    * smoothstep(0.34, 0.52, foamTex)
    * smoothstep(0.18, 0.42, foamErode)
    * (1.0 - smoothstep(5000.0, 11000.0, distM))
    * smoothstep(60.0, 240.0, distM);   // 贴脸处不铺满（近平面附近无浪形）

  // ── 水体色：近景深青绿，波峰受光提亮，掠射水柱变暗（深水吸收）───────
  let ndvFlat = clamp(dot(surfUp, viewDir), 0.0, 1.0);
  let ndvWave = clamp(dot(n, viewDir), 0.0, 1.0);
  // 反射占比以宏观视角为主（70%）：小坡度只做微调，避免整片海被小坡度
  // 拉到掠射镜面（首版"灯芯绒"亮条的直接原因）。
  let ndv  = mix(ndvWave, ndvFlat, mix(0.85, 0.96, smoothstep(2000.0, 9000.0, distM)));
  let fres = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
  let sunUp = max(dot(n, water.sunDir.xyz), 0.0);
  let upShade = clamp(h * 0.5 + 0.5, 0.0, 1.0);
  // 迎光坡亮 / 背光坡暗：让波面在低菲涅尔区也有可读的明暗纹理
  let sunE = dot(water.sunDir.xyz, east);
  let sunN = dot(water.sunDir.xyz, north);
  let tilt = (wf.x * sunE + wf.y * sunN) / max(length(vec2<f32>(sunE, sunN)), 1e-4);
  // 近景细碎波痕（~20 m 尺度的明暗颗粒，4 km 外淡出防闪）
  // fadeMod 复用地表斑块噪声：淡出边界随空间起伏，避免同心环。
  let fadeMod = 0.72 + 0.56 * foamPatch;
  let grainFade = 1.0 - smoothstep(1500.0 * fadeMod, 5000.0 * fadeMod, distM);
  let grain = mix(1.0, 0.90 + 0.18 * vnoise(waveP * 300000.0 + vec2<f32>(water.time * 0.4, 0.0)), grainFade);
  // 中远场收敛明暗对比（掠射下短波压成规则横条 → 交给反射/雾）
  let shadeFade = 1.0 - smoothstep(3000.0 * fadeMod, 10000.0 * fadeMod, distM);
  var body = water.color.rgb * (0.84 + (0.30 * upShade + 0.08 * sunUp + 0.70 * tilt) * shadeFade) * grain;
  body *= mix(1.0, 0.66, smoothstep(0.45, 0.97, 1.0 - ndvFlat));
  body *= mix(vec3<f32>(1.0), vec3<f32>(0.70, 0.77, 0.81),
              smoothstep(2500.0, 26000.0, distM));
  // 近景青绿上浮（参考 near 28/123/129：比中景更青更亮），2.2 km 外归零
  let nearK = 1.0 - smoothstep(700.0, 2200.0, distM);
  body += vec3<f32>(-0.030, 0.075, 0.065) * nearK;

  // ── 天空反射：解析渐变 + 太阳 Mie 光晕，pow5 菲涅尔过渡 ──────────────
  // 反射方向抬高 ≥8°：物理天顶（AtmosphereSky）地平线附近 ~166 且梯度很陡，
  // 波面小角度摆动就会在整片海扫出高对比"灯芯绒"亮条；抬高后反射取
  // 8-25° 的蓝灰天空，横条对比降到可接受，地平线仍由距离雾负责。
  let reflRaw = reflect(-viewDir, n);
  let reflEl  = dot(reflRaw, upDir);
  let refl    = normalize(reflRaw + upDir * max(0.0, 0.14 - reflEl));
  let skyRef  = skyColor(refl, upDir);
  var col = mix(body, skyRef, fres);

  // ── 太阳 glitter：锐瓣（高指数）+ 宽瓣，距离加宽抗闪 ─────────────────
  // 近场（<4 km）叠一层 ~2.5 m 法线抖动：把高光打碎成细密闪点（不糊成白带）。
  let jitFade = 1.0 - smoothstep(1200.0, 4200.0, distM);
  let jit = vec2<f32>(
    vnoise(waveP * 2600000.0 + vec2<f32>(water.time * 0.7, 0.0)),
    vnoise(waveP * 2600000.0 + vec2<f32>(13.7, water.time * 0.7))
  ) - 0.5;
  let nSpec = normalize(n + (east * jit.x + north * jit.y) * (0.22 * jitFade));
  let hv      = normalize(viewDir + water.sunDir.xyz);
  let ndh     = max(dot(nSpec, hv), 0.0);
  let farT    = smoothstep(2000.0, 26000.0, distM);
  let specPow = mix(700.0, 260.0, farT);
  // 锐闪只留近中景（2-11 km 内），远场交给宽瓣 → 不再满屏 confetti
  let sharp   = pow(ndh, specPow) * mix(1.05, 0.30, farT)
    * (1.0 - smoothstep(5000.0, 11000.0, distM));
  let broad   = pow(ndh, 32.0) * 0.028;
  // 随机熄灭部分闪点：打散 wave-jitter 网格形成的规则点阵
  let sparkMask = smoothstep(0.30, 0.72, vnoise(waveP * 700000.0 + vec2<f32>(water.time * 0.5, 3.3)));
  let spec    = (sharp * (0.30 + 0.70 * sparkMask) + broad) * water.glint * 0.30
    * (1.0 - foam * 0.75);
  col += SUN_TINT * spec;

  // ── 白沫合成（偏冷白，远处随雾淡）────────────────────────────────────
  col = mix(col, FOAM_TINT, clamp(foam, 0.0, 1.0) * 0.82);

  // ── 距离雾：与「该方位地平线天空」同源；远场向天顶色收敛 + 水平 haze
  //    纹理，避免在地平线堆出 18+ 行数值完全恒定的平带（第 2 轮阻断项）──
  var toFrag = normalize(input.worldPos - camPos);
  toFrag     = toFrag - upDir * dot(toFrag, upDir);
  let fogDir = select(viewDir, normalize(toFrag), length(toFrag) > 1e-4);
  let fogBase = skyColor(fogDir, upDir);
  let farHaze = smoothstep(20000.0, 80000.0, distM);
  let hazeN   = vnoise(waveP * 1600.0 + vec2<f32>(water.time * 0.01, 0.0));
  let fogCol  = mix(fogBase, SKY_ZENITH * 1.06, 0.10 * farHaze)
    * (0.97 + 0.07 * hazeN);
  // 掠射视线穿过的大气柱更长 → 同样的直线距离更早入雾（空气透视）
  let airK  = mix(1.8, 1.0, smoothstep(0.02, 0.30, ndvFlat));
  // 高空机位（如 47 页 900 km 看太湖）整片水都在 30 km 外：雾必须随
  // 相机高度放大衰减距离，否则高空俯视的湖面会被整片涂成天空灰。
  let camAltM  = max(length(camPos) - fragR, 0.0) * EARTH_R_M;
  let fogScale = 1.0 + camAltM / 3500.0;
  let fog   = smoothstep(7200.0, 27000.0, distM * airK / fogScale) * 0.97;
  col = mix(col, fogCol, fog);

  // ── 不透明度：近景保留水体色（底图透出 <4%），中远场完全不透明 ──────
  let alpha = mix(water.color.a, 1.0, smoothstep(1200.0, 12000.0, distM));

  return vec4<f32>(col, clamp(alpha, 0.0, 1.0));
}
`;
