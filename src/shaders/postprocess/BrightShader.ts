import { FULLSCREEN_VS } from './Fullscreen';

/**
 * BrightPass 着色器：soft-knee 亮度提取 + 半分辨率预滤波。
 *
 * 阈值曲线（soft-knee 近似）：
 *   knee = 0 → 硬阈值 max(luma - threshold, 0)
 *   knee > 0 → 过渡带宽 knee*3 的平方衰减（Unity 式 soft-knee 同族近似）
 */
export const BRIGHT_SHADER = FULLSCREEN_VS + /* wgsl */`
struct BrightParams {
  threshold : f32,   // 亮度阈值（线性域）
  knee      : f32,   // soft-knee 宽度 [0,1]
  _pad0     : f32,
  _pad1     : f32,
}

@group(0) @binding(0) var srcTex   : texture_2d<f32>;
@group(0) @binding(1) var srcSamp  : sampler;
@group(0) @binding(2) var<uniform> params : BrightParams;

// VSOut 由 FULLSCREEN_VS 前置声明（重复声明会使模块解析失败——真机 CDP 验证发现）

fn luma(c: vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  // 5 tap 预滤波（十字 + 中心，抑制半分辨率下的高亮像素闪烁）
  let px = 1.0 / vec2<f32>(textureDimensions(srcTex, 0));
  let c0 = textureSample(srcTex, srcSamp, in.uv).rgb;
  let c1 = textureSample(srcTex, srcSamp, in.uv + px * vec2<f32>( 1.0, 0.0)).rgb;
  let c2 = textureSample(srcTex, srcSamp, in.uv + px * vec2<f32>(-1.0, 0.0)).rgb;
  let c3 = textureSample(srcTex, srcSamp, in.uv + px * vec2<f32>(0.0,  1.0)).rgb;
  let c4 = textureSample(srcTex, srcSamp, in.uv + px * vec2<f32>(0.0, -1.0)).rgb;
  let color = (c0 + c1 + c2 + c3 + c4) * 0.2;

  // soft-knee 提取：knee*3 为过渡带半宽；平方衰减使亮部权重平滑爬升
  let l    = luma(color);
  let knee = params.knee * 3.0;
  let d    = max(l - params.threshold, 0.0);
  let soft = clamp(d / max(knee, 1e-4), 0.0, 1.0);       // 过渡带内 0→1
  let hard = step(0.0, d) * step(knee, d);               // 越过过渡带 → 1
  let w    = max(soft * soft, hard);                     // knee=0 时 soft→0，靠 hard 兜底
  let bright = color * w * step(1e-6, d);

  return vec4<f32>(bright, 1.0);
}
`;
