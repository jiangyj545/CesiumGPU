import { FULLSCREEN_VS } from './Fullscreen';

/**
 * DownsamplePass 着色器：13-tap 半分辨率下采样（CoD:AW 风格）。
 *
 * tap 布局（o = 源纹理 texel 尺寸）：
 *   中心        (0,0)          权重 4
 *   近对角      (±1,±1)·o      权重 1 ×4
 *   边缘十字    (±2,0),(0,±2)  权重 2 ×4
 *   远对角      (±1,±1)·2o     权重 0.5 ×4
 *   权重和 18，归一化 /18 —— 近似高斯，抑制降采样走样。
 */
export const BLUR_SHADER = FULLSCREEN_VS + /* wgsl */`
struct BlurParams {
  texel : vec2<f32>,  // 源纹理 1/width, 1/height
  _pad0 : f32,
  _pad1 : f32,
  _pad2 : f32,
}

@group(0) @binding(0) var srcTex  : texture_2d<f32>;
@group(0) @binding(1) var srcSamp : sampler;
@group(0) @binding(2) var<uniform> params : BlurParams;

// VSOut 由 FULLSCREEN_VS 前置声明（重复声明会使模块解析失败——真机 CDP 验证发现）

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let uv = in.uv;
  let o  = params.texel;

  let center = textureSample(srcTex, srcSamp, uv).rgb * 4.0;

  var nearDiag = textureSample(srcTex, srcSamp, uv + o * vec2<f32>(-1.0, -1.0)).rgb;
  nearDiag += textureSample(srcTex, srcSamp, uv + o * vec2<f32>( 1.0, -1.0)).rgb;
  nearDiag += textureSample(srcTex, srcSamp, uv + o * vec2<f32>(-1.0,  1.0)).rgb;
  nearDiag += textureSample(srcTex, srcSamp, uv + o * vec2<f32>( 1.0,  1.0)).rgb;

  var cross4 = textureSample(srcTex, srcSamp, uv + o * vec2<f32>(-2.0, 0.0)).rgb;
  cross4 += textureSample(srcTex, srcSamp, uv + o * vec2<f32>( 2.0, 0.0)).rgb;
  cross4 += textureSample(srcTex, srcSamp, uv + o * vec2<f32>(0.0, -2.0)).rgb;
  cross4 += textureSample(srcTex, srcSamp, uv + o * vec2<f32>(0.0,  2.0)).rgb;

  var farDiag = textureSample(srcTex, srcSamp, uv + o * 2.0 * vec2<f32>(-1.0, -1.0)).rgb * 0.5;
  farDiag += textureSample(srcTex, srcSamp, uv + o * 2.0 * vec2<f32>( 1.0, -1.0)).rgb * 0.5;
  farDiag += textureSample(srcTex, srcSamp, uv + o * 2.0 * vec2<f32>(-1.0,  1.0)).rgb * 0.5;
  farDiag += textureSample(srcTex, srcSamp, uv + o * 2.0 * vec2<f32>( 1.0,  1.0)).rgb * 0.5;

  let sum = center + nearDiag + cross4 + farDiag;   // 权重和 = 4+4+8+2 = 18
  return vec4<f32>(sum / 18.0, 1.0);
}
`;
