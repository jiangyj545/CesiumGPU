import { FULLSCREEN_VS } from './postprocess/Fullscreen';

/**
 * WeatherFxShader – FogEffect / LightningEffect 共用模块（P2-7）。
 *
 * 两个全屏 pass（各含 fs_fog / fs_lightning 入口），uniform：
 *   binding 0 : viewProj(16) + cameraPos(4) + params(4) + tint(4)  = 28 floats
 *   fog     params = [density, heightFalloff, skyTint, unused]
 *   lightning params = [intensity(当前包络), unused, unused, unused] + tint=闪色
 *   binding 1 : G-Buffer worldPos（rgba32float，归一化 ECEF；w=0 → 天空）
 *
 * 雾：factor = 1 − exp(−density·dist·exp(−falloff·h))（WeatherMath 同式），
 * 与 WeatherMath.computeFogFactor 逐常数对齐；天空像素用 skyTint 占比。
 * 闪电：全屏加性闪白（premultiplied）。
 */
export const WEATHER_FX_SHADER = FULLSCREEN_VS + /* wgsl */ `
struct Uniforms {
  cameraPos : vec4<f32>,  // @0   xyz + unused
  params    : vec4<f32>,  // @16  fog: density/heightFalloff/skyTint | lightning: envelope
  tint      : vec4<f32>,  // @32  fog 染色 / 闪电闪色（rgb）
  viewport  : vec4<f32>,  // @48  xy = 画布尺寸
}

@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var scenePosTex : texture_2d<f32>;
@group(0) @binding(2) var posSmp      : sampler;

@fragment
fn fs_fog(in : VSOut) -> @location(0) vec4<f32> {
  let wp = textureSampleLevel(scenePosTex, posSmp, in.uv, 0.0);
  let density = u.params.x;
  let falloff = u.params.y;
  let skyTint = u.params.z;

  var factor = skyTint;
  var heightMix = 0.5;
  if (wp.w > 0.5) {
    let dist = distance(wp.xyz, u.cameraPos.xyz) * 6378137.0;
    let h = (length(wp.xyz) - 1.0) * 6378137.0;
    factor = 1.0 - exp(-density * 3e-5 * dist * exp(-falloff * h));
    heightMix = clamp(h / 8000.0, 0.0, 1.0);
  }
  // 高处雾色向天空染色淡出（视觉：山体出雾）。
  let a = factor * (1.0 - 0.6 * heightMix);
  return vec4<f32>(u.tint.rgb * a, a);
}

@fragment
fn fs_lightning(in : VSOut) -> @location(0) vec4<f32> {
  // 全屏加性闪白：rgb 已乘包络（premultiplied），a 供 one-blend。
  let e = u.params.x;
  return vec4<f32>(u.tint.rgb * e, e);
}
`;

/** UBO 布局（float 数）：cameraPos(4)+params(4)+tint(4)+viewport(4) = 16。 */
export const WEATHER_FX_UNIFORM_FLOATS = 16;
