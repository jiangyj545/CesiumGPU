import { FULLSCREEN_VS } from './Fullscreen';

/**
 * ToneMappingPass 着色器：none / reinhard / aces 三模式 + sRGB gamma。
 *
 * ACES 拟合曲线（与 src/postprocessing/ToneMapping.ts 的 CPU 参考实现
 * 同一组系数 —— 两处必须同步修改）：
 *   f(x) = x·(1.4x + 1) / (0.7x² + 1.274x + 1)，x ≥ 0.0301
 *   f(x) = x，x < 0.0301（线性段，斜率 1 → knee 处 C0 连续）
 *   性质：f(0)=0、f(1)≈0.807、f(0.18)≈0.180（中灰）、全域单调。
 *
 * reinhard 与旧 LightingShader:136 内嵌公式一致 x/(1+x)。
 * 'none' 为调试直通（不做 gamma）。gamma 1/2.2 在 reinhard/aces 后叠加。
 */
export const TONEMAPPING_SHADER = FULLSCREEN_VS + /* wgsl */`
struct ToneMapParams {
  mode  : u32,   // 0 = none, 1 = reinhard, 2 = aces
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
}

@group(0) @binding(0) var srcTex  : texture_2d<f32>;
@group(0) @binding(1) var srcSamp : sampler;
@group(0) @binding(2) var<uniform> params : ToneMapParams;

// VSOut 由 FULLSCREEN_VS 前置声明（重复声明会使模块解析失败——真机 CDP 验证发现）

// ACES filmic 拟合（有理多项式 + 线性 knee 段）
// 参数须为 vec3<f32>（体内全按 vec3 运算；原 f32 签名致模块解析失败——真机 CDP 验证发现）
fn acesFilmic(x: vec3<f32>) -> vec3<f32> {
  let l = max(x, vec3<f32>(0.0));
  var out = l * (l * 1.4 + vec3<f32>(1.0)) / (l * l * 0.7 + l * 1.274 + vec3<f32>(1.0));
  // 线性段（x < 0.0301）：斜率 1 直通
  out = mix(l, out, step(vec3<f32>(0.0301), l));
  return clamp(out, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let c = textureSample(srcTex, srcSamp, in.uv).rgb;

  var mapped = c;
  if (params.mode == 1u) {
    // Reinhard + gamma（与旧内嵌 LDR 路径公式一致）
    mapped = pow(c / (c + vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
  } else if (params.mode == 2u) {
    // ACES + gamma
    mapped = pow(acesFilmic(c), vec3<f32>(1.0 / 2.2));
  }
  // mode 0：线性直通（调试用）

  return vec4<f32>(mapped, 1.0);
}
`;
