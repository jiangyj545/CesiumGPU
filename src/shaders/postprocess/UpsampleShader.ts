import { FULLSCREEN_VS } from './Fullscreen';

/**
 * UpsamplePass 着色器：3×3 tent 上采样 + 与同尺寸低层金字塔线性叠加。
 *
 * tent 核权重（和 = 16）：
 *   [1 2 1]
 *   [2 4 2]
 *   [1 2 1]
 * result = tent(src) * blend + addTex * (1 - blend)。
 * 渐进上采样时 blend 控制新一级贡献（Manager 传 0.5 得到均等混合）。
 */
export const UPSAMPLE_SHADER = FULLSCREEN_VS + /* wgsl */`
struct UpsampleParams {
  texel : vec2<f32>,  // 源纹理（较小那级）texel 尺寸
  blend : f32,        // tent 结果权重 [0,1]
  _pad0 : f32,
  _pad1 : f32,
}

@group(0) @binding(0) var srcTex   : texture_2d<f32>;
@group(0) @binding(1) var addTex   : texture_2d<f32>;
@group(0) @binding(2) var srcSamp  : sampler;
@group(0) @binding(3) var<uniform> params : UpsampleParams;

// VSOut 由 FULLSCREEN_VS 前置声明（重复声明会使模块解析失败——真机 CDP 验证发现）

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let o = params.texel;

  // 3×3 tent（9 tap，权重和 16）
  var tent = textureSample(srcTex, srcSamp, in.uv).rgb * 4.0;
  tent += textureSample(srcTex, srcSamp, in.uv + o * vec2<f32>(-1.0, 0.0)).rgb * 2.0;
  tent += textureSample(srcTex, srcSamp, in.uv + o * vec2<f32>( 1.0, 0.0)).rgb * 2.0;
  tent += textureSample(srcTex, srcSamp, in.uv + o * vec2<f32>(0.0, -1.0)).rgb * 2.0;
  tent += textureSample(srcTex, srcSamp, in.uv + o * vec2<f32>(0.0,  1.0)).rgb * 2.0;
  tent += textureSample(srcTex, srcSamp, in.uv + o * vec2<f32>(-1.0, -1.0)).rgb;
  tent += textureSample(srcTex, srcSamp, in.uv + o * vec2<f32>( 1.0, -1.0)).rgb;
  tent += textureSample(srcTex, srcSamp, in.uv + o * vec2<f32>(-1.0,  1.0)).rgb;
  tent += textureSample(srcTex, srcSamp, in.uv + o * vec2<f32>( 1.0,  1.0)).rgb;
  tent = tent / 16.0;

  let lower = textureSample(addTex, srcSamp, in.uv).rgb;
  let result = tent * params.blend + lower * (1.0 - params.blend);

  return vec4<f32>(result, 1.0);
}
`;
