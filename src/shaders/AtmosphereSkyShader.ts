/**
 * AtmosphereSkyShader — Hillaire 2020 物理大气天空 WGSL（忠实移植 tidewater
 * Atmosphere.js / Sky.js，局部平地世界 → 归一化 ECEF 球的适配见 AtmosphereSky.ts
 * 类注释）。
 *
 * 三级 compute LUT 链（全部 rgba16float write-only storage）：
 *   1. transmittance  256×64   —— 只依赖介质（静态参数），参数不变不重算
 *   2. multiScat      32×32    —— 依赖 transmittance + 静态参数
 *   3. skyView        192×108  —— 依赖以上 + viewHeight + 太阳方向（本地系）
 *
 * LUT 内部全部在「局部平地系」工作：y = 相机天顶方向，太阳方位沿 +x ——
 * CPU 侧（AtmosphereSky）负责把 ECEF 世界太阳方向投影成本地系再写入
 * AtmosphereParams.sunDir，因此 kernel 与 tidewater 逐行一致。
 *
 * AtmosphereParams（64 B，uniform）：
 *   @0  rayleighScale  @4  mieScale   @8  mieG      @12 ozoneScale
 *   @16 groundAlbedo (vec3f)      @28 viewHeight (km, r = RG + h)
 *   @32 sunIlluminance (vec3f)    @48 sunDir (vec3f，本地系：y = 太阳仰角)
 *
 * SkyParams（64 B，uniform）：
 *   @0  moonDir (vec3f, ECEF 世界系)  @12 starIntensity
 *   @16 sunDir  (vec3f, ECEF 世界系)  @28 moonOn (0/1)
 *   @32 sunDiskIntensity              @36 time (秒，星光闪烁)
 *   @40 nightK (CPU 预算，夜系数)     @44 tonemapK (1 = Reinhard+gamma)
 *   @48 moonUpK (月亮本地仰角，CPU 预算)
 *
 * 外部绑定符号约定（skyRadianceWGSL 供海面反射拼接时遵守）：
 *   var<uniform> atmosphereParams : AtmosphereParams   —— 64 B，布局同上
 *   var atmSkyViewLUT : texture_2d<f32>                —— skyView LUT
 *   var atmLinearClamp : sampler                       —— linear/clamp
 *
 * Track A 大气透视（aerial perspective）：ATMO_AERIAL_WGSL 供地形雾拼接
 * （LightingShader AP 变体），atmAerialColor 同源采样 skyView LUT，
 * 绑定契约见该块上方注释；AerialParams 32B 布局单一来源 AERIAL_PARAMS_STRUCT_WGSL。
 */

/** 大气底（行星地表）半径，km（tidewater RG）。 */
export const ATMO_PLANET_RADIUS_KM = 6360.0;
/** 大气顶半径，km（tidewater RT）。 */
export const ATMO_TOP_RADIUS_KM = 6460.0;
/** 太阳角半径，rad（tidewater SUN_ANGULAR_RADIUS）。 */
export const ATMO_SUN_ANGULAR_RADIUS = 0.004675 * 1.15;
/** 大气外太阳照度（场景单位，tidewater SUN_ILLUMINANCE）。 */
export const ATMO_SUN_ILLUMINANCE = 11.0;

/** transmittance LUT 尺寸。 */
export const ATMO_TRANSMITTANCE_SIZE = { width: 256, height: 64 } as const;
/** multiScat LUT 尺寸（正方形）。 */
export const ATMO_MULTISCAT_SIZE = { width: 32, height: 32 } as const;
/** skyView LUT 尺寸。 */
export const ATMO_SKYVIEW_SIZE = { width: 192, height: 108 } as const;

/** AtmosphereParams uniform 字节数。 */
export const ATMO_PARAMS_UBO_BYTES = 64;
/** SkyParams uniform 字节数。 */
export const SKY_PARAMS_UBO_BYTES = 64;

/** 数字 → 合法 WGSL f32 字面量（整数字面量在 WGSL 是 i32，必须带小数点）。 */
function fx(x: number): string {
  if (!Number.isFinite(x)) throw new Error(`AtmosphereSkyShader: non-finite constant ${x}`);
  const s = String(x);
  return s.includes('.') || s.includes('e') || s.includes('E') ? s : `${s}.0`;
}

// ── 公共段：AtmosphereParams struct + 介质/参数化/射线求交（tidewater coreCode）──

// AtmosphereParams struct 文本（导出：海面反射拼接方须先声明同布局 struct +
// 绑定，再拼 ATMO_SKY_REFLECTION_WGSL——该块内含引用 atmosphereParams 的
// 函数，WGSL 声明先于使用，struct 不能由块自带）。
export const ATMO_PARAMS_STRUCT_WGSL = /* wgsl */`
struct AtmosphereParams {
  rayleighScale  : f32,    // @0
  mieScale       : f32,    // @4
  mieG           : f32,    // @8
  ozoneScale     : f32,    // @12
  groundAlbedo   : vec3f,  // @16
  viewHeight     : f32,    // @28 (km, r = ATMO_RG + h)
  sunIlluminance : vec3f,  // @32
  sunDir         : vec3f,  // @48 (本地系: y = 太阳仰角)
}
`;

const ATMO_STRUCT_WGSL = ATMO_PARAMS_STRUCT_WGSL;

// SkyParams struct 文本（导出：同上，海面反射拼接方使用；布局见文件头注释）。
export const SKY_PARAMS_STRUCT_WGSL = /* wgsl */`
struct SkyParams {
  moonDir          : vec3f,  // @0  ECEF 世界系月亮方向
  starIntensity    : f32,    // @12
  sunDir           : vec3f,  // @16 ECEF 世界系太阳方向
  moonOn           : f32,    // @28
  sunDiskIntensity : f32,    // @32
  time             : f32,    // @36
  nightK           : f32,    // @40
  tonemapK         : f32,    // @44
  moonUpK          : f32,    // @48
}
`;

const ATMO_CORE_WGSL = /* wgsl */`
const PI: f32 = 3.14159265358979;
const ATMO_RG: f32 = ${fx(ATMO_PLANET_RADIUS_KM)};
const ATMO_RT: f32 = ${fx(ATMO_TOP_RADIUS_KM)};

fn atmSat(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

struct AtmosphereMedium {
  rayScat    : vec3f,
  mieScat    : f32,
  extinction : vec3f,
  scattering : vec3f,
};

fn atmosphereMedium(hKm: f32) -> AtmosphereMedium {
  let rayDensity = exp(-hKm / 8.0);
  let mieDensity = exp(-hKm / 1.2);
  let ozoneDensity = max(0.0, 1.0 - abs(hKm - 25.0) / 15.0);
  var m: AtmosphereMedium;
  m.rayScat = vec3f(5.802e-3, 13.558e-3, 33.1e-3) * rayDensity * atmosphereParams.rayleighScale;
  m.mieScat = 3.996e-3 * mieDensity * atmosphereParams.mieScale;
  let mieExt = 4.440e-3 * mieDensity * atmosphereParams.mieScale;
  let ozoneAbs = vec3f(0.650e-3, 1.881e-3, 0.085e-3) * ozoneDensity * atmosphereParams.ozoneScale;
  m.extinction = m.rayScat + mieExt + ozoneAbs;
  m.scattering = m.rayScat + m.mieScat;
  return m;
}

// (r, mu) -> transmittance LUT uv
fn atmosphereTransmittanceUV(r: f32, mu: f32) -> vec2f {
  let H = sqrt(ATMO_RT * ATMO_RT - ATMO_RG * ATMO_RG);
  let rho = sqrt(max(r * r - ATMO_RG * ATMO_RG, 0.0));
  let disc = r * r * (mu * mu - 1.0) + ATMO_RT * ATMO_RT;
  let d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
  let dMin = ATMO_RT - r;
  let dMax = rho + H;
  let xMu = (d - dMin) / (dMax - dMin);
  let xR = rho / H;
  // unit -> sub-uv
  return vec2f(
    (xMu + ${fx(0.5 / ATMO_TRANSMITTANCE_SIZE.width)}) * ${fx(ATMO_TRANSMITTANCE_SIZE.width / (ATMO_TRANSMITTANCE_SIZE.width + 1))},
    (xR + ${fx(0.5 / ATMO_TRANSMITTANCE_SIZE.height)}) * ${fx(ATMO_TRANSMITTANCE_SIZE.height / (ATMO_TRANSMITTANCE_SIZE.height + 1))});
}

// nearest positive ray-sphere intersection from ro (km, planet centered), -1 if none
fn atmosphereRaySphereNearest(ro: vec3f, rd: vec3f, radius: f32) -> f32 {
  let b = dot(ro, rd);
  let c = dot(ro, ro) - radius * radius;
  let disc = b * b - c;
  let sq = sqrt(max(disc, 0.0));
  let t0 = -b - sq;
  let t1 = -b + sq;
  return select(select(select(-1.0, t1, t1 > 0.0), t0, t0 > 0.0), -1.0, disc < 0.0);
}
`;

// ── 采样函数段（依赖 transmittance/multiScat LUT + sampler 的 kernel/渲染共用文本）──

const ATMO_TRANS_SAMPLE_WGSL = /* wgsl */`
fn atmosphereSampleTransmittance(r: f32, mu: f32) -> vec3f {
  return textureSampleLevel(atmTransmittanceLUT, atmLinearClamp, atmosphereTransmittanceUV(r, mu), 0.0).rgb;
}
`;
export { ATMO_TRANS_SAMPLE_WGSL };

const ATMO_MS_SAMPLE_WGSL = /* wgsl */`
fn atmosphereSampleMultiScat(r: f32, cosSun: f32) -> vec3f {
  let uv = vec2f(cosSun * 0.5 + 0.5, (r - ATMO_RG) / (ATMO_RT - ATMO_RG));
  let suv = uv * ${fx((ATMO_MULTISCAT_SIZE.width - 1) / ATMO_MULTISCAT_SIZE.width)} + ${fx(0.5 / ATMO_MULTISCAT_SIZE.width)};
  return textureSampleLevel(atmMultiScatLUT, atmLinearClamp, suv, 0.0).rgb;
}
`;

// dirL/sunL 均为本地系（y = up）。参数化太阳方向使渲染端与反射端复用同一文本。
const ATMO_SKY_LUMINANCE_WGSL = /* wgsl */`
// Sky luminance (no sun disk) for local direction dirL (normalized, y = up).
fn atmSkyLuminance(dirL: vec3f, sunL: vec3f) -> vec3f {
  let viewH = atmosphereParams.viewHeight;
  let vHorizon = sqrt(max(viewH * viewH - ATMO_RG * ATMO_RG, 0.0));
  let beta = acos(vHorizon / viewH);
  let zenithHorizonAngle = PI - beta;
  let viewZenithAngle = acos(clamp(dirL.y, -1.0, 1.0));

  let vCoordA = (1.0 - sqrt(max(1.0 - viewZenithAngle / zenithHorizonAngle, 0.0))) * 0.5;
  let vCoordB = sqrt(max((viewZenithAngle - zenithHorizonAngle) / beta, 0.0)) * 0.5 + 0.5;
  let v = select(vCoordB, vCoordA, viewZenithAngle < zenithHorizonAngle);

  // azimuth relative to sun
  let sunH = normalize(vec2f(sunL.x, sunL.z) + vec2f(1e-5, 0.0));
  let dirH = normalize(vec2f(dirL.x, dirL.z) + vec2f(1e-5, 0.0));
  let lightViewCos = dot(sunH, dirH);
  let u = sqrt(clamp(lightViewCos * -0.5 + 0.5, 0.0, 1.0));

  let suv = vec2f(
    u * ${fx((ATMO_SKYVIEW_SIZE.width - 1) / ATMO_SKYVIEW_SIZE.width)} + ${fx(0.5 / ATMO_SKYVIEW_SIZE.width)},
    v * ${fx((ATMO_SKYVIEW_SIZE.height - 1) / ATMO_SKYVIEW_SIZE.height)} + ${fx(0.5 / ATMO_SKYVIEW_SIZE.height)});
  return textureSampleLevel(atmSkyViewLUT, atmLinearClamp, suv, 0.0).rgb;
}
`;

// ── kernel 1：transmittance LUT（tidewater Atmosphere Transmittance）────────

export const ATMO_TRANSMITTANCE_KERNEL_WGSL = /* wgsl */`${ATMO_STRUCT_WGSL}
@group(0) @binding(0) var<uniform> atmosphereParams : AtmosphereParams;
@group(0) @binding(1) var atmTransmittanceLUT : texture_storage_2d<rgba16float, write>;
${ATMO_CORE_WGSL}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.xy;
  let uv = (vec2f(px) + 0.5) / vec2f(${fx(ATMO_TRANSMITTANCE_SIZE.width)}, ${fx(ATMO_TRANSMITTANCE_SIZE.height)});
  let xMu = (uv.x - ${fx(0.5 / ATMO_TRANSMITTANCE_SIZE.width)}) * ${fx(ATMO_TRANSMITTANCE_SIZE.width / (ATMO_TRANSMITTANCE_SIZE.width - 1))};
  let xR = (uv.y - ${fx(0.5 / ATMO_TRANSMITTANCE_SIZE.height)}) * ${fx(ATMO_TRANSMITTANCE_SIZE.height / (ATMO_TRANSMITTANCE_SIZE.height - 1))};
  let H = sqrt(ATMO_RT * ATMO_RT - ATMO_RG * ATMO_RG);
  let rho = xR * H;
  let r = sqrt(rho * rho + ATMO_RG * ATMO_RG);
  let dMin = ATMO_RT - r;
  let dMax = rho + H;
  let d = dMin + xMu * (dMax - dMin);
  let mu = clamp(select((H * H - rho * rho - d * d) / (2.0 * r * d), 1.0, d == 0.0), -1.0, 1.0);

  let ro = vec3f(0.0, r, 0.0);
  let rd = vec3f(sqrt(max(1.0 - mu * mu, 0.0)), mu, 0.0);
  let tMax = atmosphereRaySphereNearest(ro, rd, ATMO_RT);
  let steps = 40;
  let dt = tMax / f32(steps);
  var od = vec3f(0.0);
  for (var i = 0; i < steps; i++) {
    let t = (f32(i) + 0.5) * dt;
    let p = ro + rd * t;
    let h = length(p) - ATMO_RG;
    od += atmosphereMedium(h).extinction * dt;
  }
  textureStore(atmTransmittanceLUT, px, vec4f(exp(-od), 1.0));
}
`;

// ── kernel 2：multiple scattering LUT（tidewater Atmosphere MultiScat）──────

export const ATMO_MULTISCAT_KERNEL_WGSL = /* wgsl */`${ATMO_STRUCT_WGSL}
@group(0) @binding(0) var<uniform> atmosphereParams : AtmosphereParams;
@group(0) @binding(1) var atmTransmittanceLUT : texture_2d<f32>;
@group(0) @binding(2) var atmLinearClamp : sampler;
@group(0) @binding(3) var atmMultiScatOut : texture_storage_2d<rgba16float, write>;
${ATMO_CORE_WGSL}
${ATMO_TRANS_SAMPLE_WGSL}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.xy;
  let uv = ((vec2f(px) + 0.5) / ${fx(ATMO_MULTISCAT_SIZE.width)} - ${fx(0.5 / ATMO_MULTISCAT_SIZE.width)}) * ${fx(ATMO_MULTISCAT_SIZE.width / (ATMO_MULTISCAT_SIZE.width - 1))};
  let cosSun = uv.x * 2.0 - 1.0;
  let r = ATMO_RG + clamp(uv.y, 0.001, 0.999) * (ATMO_RT - ATMO_RG);
  let sunDir = normalize(vec3f(0.0, cosSun, -sqrt(max(1.0 - cosSun * cosSun, 0.0))));
  let ro = vec3f(0.0, r, 0.0);

  var Lsum = vec3f(0.0);
  var fmsSum = vec3f(0.0);
  const SQ = 8;
  let isoPhase = 1.0 / (4.0 * PI);

  for (var i = 0; i < SQ * SQ; i++) {
    let ii = (f32(i % SQ) + 0.5) / f32(SQ);
    let jj = (f32(i / SQ) + 0.5) / f32(SQ);
    let theta = ii * 2.0 * PI;
    let phi = acos(1.0 - jj * 2.0);
    let rd = vec3f(cos(theta) * sin(phi), cos(phi), sin(theta) * sin(phi));

    let tBottom = atmosphereRaySphereNearest(ro, rd, ATMO_RG);
    let tTop = atmosphereRaySphereNearest(ro, rd, ATMO_RT);
    let hitGround = tBottom > 0.0;
    let tMax = select(tTop, tBottom, hitGround);
    let steps = 20;
    let dt = tMax / f32(steps);
    var throughput = vec3f(1.0);
    var L = vec3f(0.0);
    var fms = vec3f(0.0);

    for (var s = 0; s < steps; s++) {
      let t = (f32(s) + 0.3) * dt;
      let p = ro + rd * t;
      let pr = length(p);
      let m = atmosphereMedium(pr - ATMO_RG);
      let up = p / pr;
      let cosSunP = dot(up, sunDir);
      let Tsun = atmosphereSampleTransmittance(pr, cosSunP);
      let shadowT = atmosphereRaySphereNearest(p, sunDir, ATMO_RG);
      let earthShadow = select(1.0, 0.0, shadowT > 0.0);
      let S = Tsun * earthShadow * m.scattering * isoPhase;
      let Tstep = exp(-m.extinction * dt);
      let ext = max(m.extinction, vec3f(1e-6));
      L += throughput * (S - S * Tstep) / ext;
      fms += throughput * (m.scattering - m.scattering * Tstep) / ext;
      throughput *= Tstep;
    }

    if (hitGround) {
      let p = ro + rd * tMax;
      let up = normalize(p);
      let cosS = dot(up, sunDir);
      let Tsun = atmosphereSampleTransmittance(ATMO_RG, cosS);
      L += Tsun * throughput * max(cosS, 0.0) * atmosphereParams.groundAlbedo / PI;
    }

    Lsum += L * (4.0 * PI / f32(SQ * SQ));
    fmsSum += fms * (4.0 * PI / f32(SQ * SQ));
  }

  let Lin = Lsum * isoPhase;
  let fmsAvg = fmsSum * isoPhase;
  let Lms = Lin / (vec3f(1.0) - fmsAvg);
  textureStore(atmMultiScatOut, px, vec4f(Lms, 1.0));
}
`;

// ── kernel 3：sky view LUT（tidewater Atmosphere SkyView，每参数变化重算）────

export const ATMO_SKYVIEW_KERNEL_WGSL = /* wgsl */`${ATMO_STRUCT_WGSL}
@group(0) @binding(0) var<uniform> atmosphereParams : AtmosphereParams;
@group(0) @binding(1) var atmTransmittanceLUT : texture_2d<f32>;
@group(0) @binding(2) var atmLinearClamp : sampler;
@group(0) @binding(3) var atmMultiScatLUT : texture_2d<f32>;
@group(0) @binding(4) var atmSkyViewOut : texture_storage_2d<rgba16float, write>;
${ATMO_CORE_WGSL}
${ATMO_TRANS_SAMPLE_WGSL}
${ATMO_MS_SAMPLE_WGSL}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.xy;
  if (px.x >= ${ATMO_SKYVIEW_SIZE.width}u || px.y >= ${ATMO_SKYVIEW_SIZE.height}u) { return; }
  let uv = (vec2f(px) + 0.5) / vec2f(${fx(ATMO_SKYVIEW_SIZE.width)}, ${fx(ATMO_SKYVIEW_SIZE.height)});
  let u = (uv.x - ${fx(0.5 / ATMO_SKYVIEW_SIZE.width)}) * ${fx(ATMO_SKYVIEW_SIZE.width / (ATMO_SKYVIEW_SIZE.width - 1))};
  let v = (uv.y - ${fx(0.5 / ATMO_SKYVIEW_SIZE.height)}) * ${fx(ATMO_SKYVIEW_SIZE.height / (ATMO_SKYVIEW_SIZE.height - 1))};

  let viewH = atmosphereParams.viewHeight;
  let vHorizon = sqrt(max(viewH * viewH - ATMO_RG * ATMO_RG, 0.0));
  let beta = acos(vHorizon / viewH);
  let zenithHorizonAngle = PI - beta;

  var vzA = 0.0;
  if (v < 0.5) {
    var c = v * 2.0;
    c = 1.0 - c;
    c = c * c;
    c = 1.0 - c;
    vzA = zenithHorizonAngle * c;
  } else {
    var c = v * 2.0 - 1.0;
    c = c * c;
    vzA = zenithHorizonAngle + beta * c;
  }

  let cosViewZenith = cos(vzA);
  let sinViewZenith = sin(vzA);
  let cu = u * u;
  let lightViewCos = -(cu * 2.0 - 1.0);
  let lightViewSin = sqrt(max(1.0 - lightViewCos * lightViewCos, 0.0));

  // local frame: up = +y, sun azimuth along +x
  let sunCosZ = atmosphereParams.sunDir.y;
  let sunSinZ = sqrt(max(1.0 - sunCosZ * sunCosZ, 0.0));
  let sunDir = vec3f(sunSinZ, sunCosZ, 0.0);
  let rd = vec3f(sinViewZenith * lightViewCos, cosViewZenith, sinViewZenith * lightViewSin);
  let ro = vec3f(0.0, viewH, 0.0);

  let tBottom = atmosphereRaySphereNearest(ro, rd, ATMO_RG);
  let tTop = atmosphereRaySphereNearest(ro, rd, ATMO_RT);
  let tMax = select(tTop, tBottom, tBottom > 0.0);
  let steps = 32;
  let cosTheta = dot(rd, sunDir);
  let rayPhase = ${fx(3 / (16 * Math.PI))} * (cosTheta * cosTheta + 1.0);
  let g = atmosphereParams.mieG;
  let g2 = g * g;
  // Cornette-Shanks
  let miePhase = ${fx(3 / (8 * Math.PI))} * ((1.0 - g2) * (cosTheta * cosTheta + 1.0))
    / ((g2 + 2.0) * pow(max(g2 + 1.0 - g * cosTheta * 2.0, 1e-4), 1.5));

  var throughput = vec3f(1.0);
  var L = vec3f(0.0);

  for (var i = 0; i < steps; i++) {
    // quadratic step distribution
    let t0 = f32(i) / f32(steps);
    let t1 = (f32(i) + 1.0) / f32(steps);
    let ta = t0 * t0 * tMax;
    let tb = t1 * t1 * tMax;
    let t = mix(ta, tb, 0.3);
    let dt = tb - ta;
    let p = ro + rd * t;
    let pr = length(p);
    let m = atmosphereMedium(pr - ATMO_RG);
    let up = p / pr;
    let cosSunP = dot(up, sunDir);
    let Tsun = atmosphereSampleTransmittance(pr, cosSunP);
    let shadowT = atmosphereRaySphereNearest(p, sunDir, ATMO_RG);
    let earthShadow = select(1.0, 0.0, shadowT > 0.0);
    let ms = atmosphereSampleMultiScat(pr, cosSunP);
    let phaseScat = m.rayScat * rayPhase + vec3f(m.mieScat * miePhase);
    let S = Tsun * earthShadow * phaseScat + ms * m.scattering;
    let Tstep = exp(-m.extinction * dt);
    let ext = max(m.extinction, vec3f(1e-6));
    L += throughput * (S - S * Tstep) / ext;
    throughput *= Tstep;
  }

  textureStore(atmSkyViewOut, px, vec4f(L * atmosphereParams.sunIlluminance, 1.0));
}
`;

// ── 合成段：日盘 / 星 / 月 / 月天光（tidewater Sky.js，惯性系适配）───────────
// tidewater 的平地世界中 dir.y 即仰角；我们为球面 ECEF 把「仰角类」因子参数化
// （upY = dot(dirWorld, up)），星空 cube 映射保持 ECEF 惯性系（天球不随相机转）。

const SKY_COMPOSITE_WGSL = /* wgsl */`
// hash without sine (D. Hoskins): keeps full precision for large integer coordinates
fn skyHash13(p: vec3f) -> f32 {
  var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

// Sun disk radiance along dirW (world, normalized; includes atmospheric transmittance).
// upY = dot(dirW, up) 是该方向的地平仰角分量（球面世界替代 tidewater 的 dir.y）。
fn skySunDisk(dirW: vec3f, sunDirW: vec3f, upY: f32) -> vec3f {
  let cosA = dot(dirW, sunDirW);
  let ang = acos(clamp(cosA, -1.0, 1.0));
  let r = ang / ${fx(ATMO_SUN_ANGULAR_RADIUS)};
  let mask = smoothstep(1.0, 0.9, r);
  let mu = sqrt(max(1.0 - r * r, 0.0));
  let limb = 1.0 - 0.6 * (1.0 - mu);
  let T = atmosphereSampleTransmittance(atmosphereParams.viewHeight, upY);
  // physically the disk radiance is E / solid angle (~1.6e5); clamp for fp16 targets
  return T * mask * limb * 2500.0 * skyParams.sunDiskIntensity * smoothstep(-0.02, 0.0, upY);
}
`;

const SKY_STARS_WGSL = /* wgsl */`
const SKY_STAR_CELLS: f32 = 160.0; // cells per half cube face (one star candidate per cell)
const SKY_STAR_SIGMA: f32 = 0.1;   // star PSF (cells, ~1 px)

// Stars: one candidate per cell of a cube-face grid (ECEF 惯性系天球), jittered inside it, with a
// power law magnitude distribution (few bright stars, many faint ones), denser along the Milky
// Way. They come out after civil twilight: the brightest first, the faintest once the sky is dark.
// sunUp = 太阳本地仰角（与 atmosphereParams.sunDir.y 同源，由调用方传入）。
fn skyStars(dirW: vec3f, upY: f32, sunUp: f32, time: f32) -> vec3f {
  // cube face coordinates
  let a = abs(dirW);
  let onX = a.x > a.y && a.x > a.z;
  let onY = a.y > a.z;
  let face = select(select(sign(dirW.z) + 8.0, sign(dirW.y) + 5.0, onY), sign(dirW.x) + 2.0, onX);
  let uv = select(select(dirW.xy / a.z, dirW.xz / a.y, onY), dirW.yz / a.x, onX) * SKY_STAR_CELLS;
  let cell = vec3f(floor(uv), face);
  let h = skyHash13(cell);
  let bx = dot(dirW, vec3f(${fx(0.3 / Math.sqrt(1.13))}, ${fx(0.2 / Math.sqrt(1.13))}, ${fx(1 / Math.sqrt(1.13))})) * 4.0;
  let band = exp(-bx * bx);
  // the cell holds a star with probability P (higher along the Milky Way); an independent
  // uniform u ranks its brightness: N(<m) ~ 10^(m/2), brightest about magnitude -1
  let has = h < band * 0.035 + 0.025;
  let uc = max(skyHash13(cell + 7.7), 2e-4);
  let m = log2(uc) * 0.602 + 6.5;
  // limiting magnitude: -1 when the sun is 6 deg below the horizon, 6.5 below 16 deg
  let dark = 1.0 - smoothstep(-0.28, -0.1, sunUp);
  let vis = smoothstep(m - 0.6, m + 0.6, dark * 7.5 - 1.0) * select(0.0, 1.0, has);
  // angular distance to the star (isotropic whatever the cube face distortion), in cells
  let sp = (floor(uv) + vec2f(skyHash13(cell + 3.1), skyHash13(cell + 5.7)) * 0.4 + 0.3) / SKY_STAR_CELLS;
  let sdir = normalize(select(select(vec3f(sp, sign(dirW.z)), vec3f(sp.x, sign(dirW.y), sp.y), onY), vec3f(sign(dirW.x), sp), onX));
  let d = length(dirW - sdir) * SKY_STAR_CELLS;
  // flux relative to a magnitude 6.5 star; bright stars look bigger (glare), not just clipped
  let flux = pow(uc, -0.8);
  let size = log2(flux) * 0.08 + 1.0;
  let psf = exp(d * d / (size * size) * ${fx(-0.5 / (0.1 * 0.1))}) / (size * size);
  // subtle scintillation, stronger low in the sky
  let tw = sin(time * (skyHash13(cell + 13.3) * 9.0 + 5.0) + h * 60.0) * mix(0.18, 0.06, atmSat(upY * 2.0)) + 1.0;
  let col = mix(vec3f(1.0, 0.8, 0.6), vec3f(0.75, 0.85, 1.0), skyHash13(cell + 17.0)) * 0.5 + 0.5;
  let star = col * (psf * flux * vis * tw * 0.0075);
  // diffuse glow of the Milky Way
  let glow = vec3f(0.55, 0.6, 0.75) * (band * dark * 0.0035);
  // atmospheric extinction toward the horizon
  return (star + glow) * skyParams.starIntensity * smoothstep(0.0, 0.2, upY);
}
`;

const SKY_MOON_WGSL = /* wgsl */`
fn skyMoon(dirW: vec3f, moonDirW: vec3f, upY: f32) -> vec3f {
  let cosA = dot(dirW, moonDirW);
  let ang = acos(clamp(cosA, -1.0, 1.0));
  let r = ang / 0.0048;
  let mask = smoothstep(1.0, 0.92, r);
  return vec3f(0.9, 0.92, 1.0) * mask * 3.0 * skyParams.starIntensity * smoothstep(-0.02, 0.02, upY);
}

// Faint blue-grey moonlit sky (a little brighter toward the horizon) and the moon's aureole.
// moonUp = 月亮本地仰角，nightK = 夜系数（替代 tidewater 的 frame.night）。
fn skyMoonSky(dirL: vec3f, moonUp: f32, nightK: f32) -> vec3f {
  let grad = mix(1.7, 1.0, atmSat(dirL.y * 3.0));
  let up = smoothstep(-0.05, 0.15, moonUp);
  return vec3f(0.005, 0.0068, 0.0105) * (grad * nightK) * up;
}
`;

const SKY_TONEMAP_WGSL = /* wgsl */`
// Reinhard + gamma：适配 LDR swap-chain 主路径（与 LightingPass 旧路径一致）；
// tonemapK = 0 时线性直出（接入 HDR 后处理链时由 AtmosphereSky.toneMapped 控制）。
fn atmTonemap(c: vec3f) -> vec3f {
  let m = max(c, vec3f(0.0));
  return pow(m / (m + vec3f(1.0)), vec3f(1.0 / 2.2));
}
`;

// ── 背景渲染：全屏三角形 + 相机基射线重建 + 本地基转换 + 合成 ────────────────

export const ATMO_SKY_RENDER_WGSL = /* wgsl */`
// 256B 相机 UBO — 布局单一来源：src/renderer/CameraUniforms.ts
struct CameraUniforms {
  viewMatrix      : mat4x4<f32>,   // @0   (64B)
  projMatrix      : mat4x4<f32>,   // @64  (64B)
  viewProjMatrix  : mat4x4<f32>,   // @128 (64B)
  cameraPosition  : vec4<f32>,     // @192 (16B) 归一化 ECEF，w 未用
  viewportSize    : vec2<f32>,     // @208 (8B)
  nearFar         : vec2<f32>,     // @224 (8B) x = near, y = far
}
${ATMO_STRUCT_WGSL}
${SKY_PARAMS_STRUCT_WGSL}

@group(0) @binding(0) var<uniform> camera : CameraUniforms;
@group(0) @binding(1) var<uniform> atmosphereParams : AtmosphereParams;
@group(0) @binding(2) var<uniform> skyParams : SkyParams;
@group(0) @binding(3) var atmSkyViewLUT : texture_2d<f32>;
@group(0) @binding(4) var atmTransmittanceLUT : texture_2d<f32>;
@group(0) @binding(5) var atmLinearClamp : sampler;

${ATMO_CORE_WGSL}
${ATMO_TRANS_SAMPLE_WGSL}
${ATMO_SKY_LUMINANCE_WGSL}
${SKY_COMPOSITE_WGSL}
${SKY_STARS_WGSL}
${SKY_MOON_WGSL}
${SKY_TONEMAP_WGSL}

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0)       ndc  : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  var out: VSOut;
  out.pos = vec4<f32>(positions[vid], 1.0, 1.0); // NDC z = 1 == 背景深度，供 depthCompare equal
  out.ndc = positions[vid];
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  // 由 view/proj 矩阵重建世界射线（lookAt：view 行 0/1/2 = 相机右/上/前基）
  let camRight = vec3f(camera.viewMatrix[0][0], camera.viewMatrix[1][0], camera.viewMatrix[2][0]);
  let camUp    = vec3f(camera.viewMatrix[0][1], camera.viewMatrix[1][1], camera.viewMatrix[2][1]);
  let camFwd   = vec3f(-camera.viewMatrix[0][2], -camera.viewMatrix[1][2], -camera.viewMatrix[2][2]);
  let tanX = 1.0 / camera.projMatrix[0][0];
  let tanY = 1.0 / camera.projMatrix[1][1];
  let dir = normalize(camRight * (in.ndc.x * tanX) + camUp * (in.ndc.y * tanY) + camFwd);

  // 本地基：up = 相机归一化 ECEF；与 CPU 侧投影算法逐行一致（必须同步修改）
  let up = normalize(camera.cameraPosition.xyz);
  let upY = dot(dir, up);
  // refAxis：ref 是 WGSL 保留字，用 ref 会使模块解析失败（真机 CDP 验证发现）
  let refAxis = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(up.y) > 0.9);
  let east = normalize(cross(refAxis, up));
  let north = cross(up, east);
  let dirL = vec3f(dot(dir, east), upY, dot(dir, north));
  let sunW = normalize(skyParams.sunDir);
  let sunL = vec3f(dot(sunW, east), dot(sunW, up), dot(sunW, north));

  var L = atmSkyLuminance(dirL, sunL);
  if (skyParams.starIntensity > 0.001) {
    L = L + skyMoonSky(dirL, skyParams.moonUpK, skyParams.nightK)
          + skyStars(dir, upY, sunL.y, skyParams.time);
  }
  if (skyParams.moonOn > 0.5) {
    L = L + skyMoon(dir, normalize(skyParams.moonDir), upY);
  }
  L = L + skySunDisk(dir, sunW, upY);

  var color = L;
  if (skyParams.tonemapK > 0.5) { color = atmTonemap(color); }
  return vec4<f32>(color, 1.0);
}
`;

// ── B1 反射集成文本（skyRadianceWGSL）：水面反射用天空辐射，无日/月盘 ────────
// 依赖调用方声明的符号（名字必须一致，且须先于本块声明——WGSL 声明先于使用）：
//   struct AtmosphereParams（ATMO_PARAMS_STRUCT_WGSL，64B）
//   struct SkyParams（SKY_PARAMS_STRUCT_WGSL，64B）
//   @group(X) @binding(A) var<uniform> atmosphereParams : AtmosphereParams;
//   @group(X) @binding(B) var atmSkyViewLUT : texture_2d<f32>;
//   @group(X) @binding(C) var atmLinearClamp : sampler;
//   @group(X) @binding(D) var<uniform> skyParams : SkyParams;
// AtmosphereSky.skyViewTexture 即 atmSkyViewLUT 应绑定的纹理视图。

export const ATMO_SKY_REFLECTION_WGSL = /* wgsl */`
${ATMO_CORE_WGSL}
${ATMO_SKY_LUMINANCE_WGSL}
${SKY_STARS_WGSL}
${SKY_MOON_WGSL}

// 局地基（与 AtmosphereSky 背景渲染、CPU 投影逐行一致）
fn atmLocalBasis(up: vec3f) -> mat3x3<f32> {
  // refAxis：ref 是 WGSL 保留字，用 ref 会使模块解析失败（真机 CDP 验证发现）
  let refAxis = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(up.y) > 0.9);
  let east = normalize(cross(refAxis, up));
  let north = cross(up, east);
  return mat3x3<f32>(east, up, north); // 列: east / up / north
}

// 水面反射用天空辐射（tidewater skyReflectionRadiance）：无日/月盘，
// 星光仅余 8% 痕迹（粗糙水面把点源抹成闪烁碎光）。
// dirWorld: 归一化反射方向（ECEF）；up: 反射点归一化 ECEF 天顶方向；
// sunDirWorld / moonDirWorld: ECEF 归一化；time: 秒。
// 注意：需调用方同时声明 skyParams（SkyParams，starIntensity 字段生效）。
fn atmSkyReflectionRadiance(
  dirWorld: vec3f,
  up: vec3f,
  sunDirWorld: vec3f,
  moonDirWorld: vec3f,
  time: f32,
) -> vec3f {
  let B = atmLocalBasis(up);
  let dirL = vec3f(dot(dirWorld, B[0]), dot(dirWorld, B[1]), dot(dirWorld, B[2]));
  let sunL = vec3f(dot(sunDirWorld, B[0]), dot(sunDirWorld, B[1]), dot(sunDirWorld, B[2]));
  let upY = dirL.y;
  var L = atmSkyLuminance(dirL, sunL);
  if (skyParams.starIntensity > 0.001) {
    L = L + skyMoonSky(dirL, dot(moonDirWorld, up), 1.0 - smoothstep(-0.16, -0.04, sunL.y))
          + skyStars(dirWorld, upY, sunL.y, time) * 0.08;
  }
  return L;
}
`;

// ── Track A 大气透视（aerial perspective）模块：地形雾随太阳/视角物理染色 ─────
//
// 原理（务实 AP，不建第四张 LUT）：远物的 inscatter 颜色 ≈ 该视线方向的天空
// 辐射 —— skyView LUT 已按 (视线天顶角, 相对太阳方位) 参数化，直接采样即可让
// 远山「融进那个方向的天空」：黄昏地平线方向天空本就橙红 → 远处地形自然染
// 橙红；朝太阳看时前向散射更亮 → 雾更浓。距离混合系数由消费方用
// fogAmount = 1 - exp(-dist / distM) 计算后 mix（见 LightingShader.HAZE_APPLIED）。
//
// **AP 绑定契约**（消费方——目前仅 LightingShader AP 变体——须先于本块声明，
// WGSL 声明先于使用；名字必须一致）：
//   struct AtmosphereParams（ATMO_PARAMS_STRUCT_WGSL，64 B）
//   @group(X) @binding(A) var<uniform> atmosphereParams : AtmosphereParams;
//   @group(X) @binding(B) var atmSkyViewLUT : texture_2d<f32>;   —— 绑 skyViewTexture
//   @group(X) @binding(C) var atmLinearClamp : sampler;
// 入口函数 `atmAerialColor(dirWorld, up, sunDirWorld) -> vec3f`：
//   dirWorld    归一化视线方向（相机 → 地物，ECEF）；
//   up          地物点归一化 ECEF 天顶方向；
//   sunDirWorld 归一化 ECEF 太阳方向（与 AtmosphereSky.setSunDirection 同源）。
// 内部 = atmSkyLuminance 同源查询，**无日/月盘、无星空**（雾不该发光点）。
//
// 注意：本块内含 atmSkyLuminance 定义，**不可与 ATMO_SKY_REFLECTION_WGSL 拼
// 进同一 shader 模块**（重复定义）。局部基函数独立命名 atmAerialLocalBasis
// （算法与 atmLocalBasis 逐行一致），保证两块即使同仓库并存也无符号冲突。

/** AerialParams uniform 字节数（strength/distM/sunDir 布局见下方 struct）。 */
export const AERIAL_PARAMS_UBO_BYTES = 32;

/** AerialParams struct 文本（导出：AP 消费方拼接声明；布局单一来源）。 */
export const AERIAL_PARAMS_STRUCT_WGSL = /* wgsl */`
struct AerialParams {
  strength : f32,      // @0  总开关/强度（0 = 恒走旧路径，零回归）
  distM    : f32,      // @4  参考雾距（米，默认 25000 ≈ 25 km）
  _pad0    : f32,      // @8
  _pad1    : f32,      // @12
  sunDir   : vec3f,    // @16 归一化 ECEF 太阳方向（vec3f 16B 对齐 → @16）
}
`;

const ATMO_AERIAL_BODY_WGSL = /* wgsl */`
// 局地基（与 ATMO_SKY_REFLECTION_WGSL.atmLocalBasis 同算法，独立命名防拼冲突）
fn atmAerialLocalBasis(up: vec3f) -> mat3x3<f32> {
  // refAxis：ref 是 WGSL 保留字，用 ref 会使模块解析失败（真机 CDP 验证发现）
  let refAxis = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(up.y) > 0.9);
  let east = normalize(cross(refAxis, up));
  let north = cross(up, east);
  return mat3x3<f32>(east, up, north); // 列: east / up / north
}

${ATMO_SKY_LUMINANCE_WGSL}

// 大气透视雾色（无日/月盘/星空的天空辐射），契约见文件级注释。
fn atmAerialColor(dirWorld: vec3f, up: vec3f, sunDirWorld: vec3f) -> vec3f {
  let B = atmAerialLocalBasis(up);
  let dirL = vec3f(dot(dirWorld, B[0]), dot(dirWorld, B[1]), dot(dirWorld, B[2]));
  let sunL = vec3f(dot(sunDirWorld, B[0]), dot(sunDirWorld, B[1]), dot(sunDirWorld, B[2]));
  return atmSkyLuminance(dirL, sunL);
}
`;

/** AP 模块（地形雾用，LightingShader AP 变体拼接；绑定契约见上方注释）。 */
export const ATMO_AERIAL_WGSL = ATMO_AERIAL_BODY_WGSL;

