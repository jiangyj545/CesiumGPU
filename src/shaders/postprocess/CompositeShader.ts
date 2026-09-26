import { FULLSCREEN_VS } from './Fullscreen';

/**
 * CompositePass 着色器：sceneColor + bloom + lensFlare（ghost 序列 + DirtMask）。
 *
 * LensFlare ghost 布局：
 *  - 第 0 个 ghost 即太阳本体（halo），沿 sun↔屏幕中心镜像轴等距散布
 *    t_i = (i+1)/ghosts，位置 = mix(sun, 1-sun, t_i)。
 *  - 每 ghost 对 sceneColor 做缩放回采样（缩放系数随 t 递增），并施加
 *    chromatic offset（RGB 三通道沿径向微偏移，边缘色散）。
 *  - flare 与 DirtMask 纹理逐像素相乘，再乘 intensity 叠加。
 *  - 太阳在屏外时 Manager 将 flareIntensity 置 0（本 shader 无需判屏外）。
 */
export const COMPOSITE_SHADER = FULLSCREEN_VS + /* wgsl */`
struct CompositeParams {
  bloomIntensity : f32,       // bloom 叠加强度
  flareIntensity : f32,       // lensFlare 总强度（屏外 = 0）
  sunUv          : vec2<f32>, // 太阳屏幕位置（uv 域，y 翻转后）
  ghosts         : f32,       // ghost 数量 [4,8]
  time           : f32,       // 秒（halo 微呼吸动画）
  dirtStrength   : f32,       // DirtMask 强度
  aspect         : f32,       // 宽高比（ghost 圆形化修正）
}

@group(0) @binding(0) var sceneTex  : texture_2d<f32>;
@group(0) @binding(1) var bloomTex  : texture_2d<f32>;
@group(0) @binding(2) var dirtTex   : texture_2d<f32>;
@group(0) @binding(3) var samp      : sampler;
@group(0) @binding(4) var<uniform> params : CompositeParams;

// VSOut 由 FULLSCREEN_VS 前置声明（重复声明会使模块解析失败——真机 CDP 验证发现）

// 单个 ghost：径向缩放回采样 + RGB 径向色散
fn ghostSample(sceneUv: vec2<f32>, center: vec2<f32>, scale: f32, aspect: f32) -> vec3<f32> {
  // 距 ghost 中心的径向向量（修正宽高比使色散为圆形）
  let rel    = (sceneUv - center) * vec2<f32>(aspect, 1.0);
  let offset = normalize(rel + vec2<f32>(1e-5)) * length(rel) * (scale - 1.0);

  let chroma = vec2<f32>(0.004, 0.004) * scale;   // 色散幅度随 ghost 缩放
  let base   = sceneUv + offset;
  let r = textureSample(sceneTex, samp, base + chroma).r;
  let g = textureSample(sceneTex, samp, base).g;
  let b = textureSample(sceneTex, samp, base - chroma).b;
  return vec3<f32>(r, g, b);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let scene = textureSample(sceneTex, samp, in.uv).rgb;
  let bloom = textureSample(bloomTex, samp, in.uv).rgb;

  // ── LensFlare ghost 序列 ────────────────────────────────────────────────
  var flare = vec3<f32>(0.0);
  let sun = params.sunUv;
  let center = vec2<f32>(0.5, 0.5);
  let aspect = params.aspect;
  let ghosts = max(params.ghosts, 1.0);

  for (var i: i32 = 0; i < i32(ghosts); i++) {
    let t  = (f32(i) + 1.0) / ghosts;                      // 0..1 沿镜像轴
    let gp = mix(sun, vec2<f32>(1.0) - sun, t);            // 镜像轴等距散布
    let scale = 1.0 + t * 1.5;                             // 越远 ghost 采样越放大
    let g = ghostSample(in.uv, gp, scale, aspect);

    // ghost 权重：靠近太阳强、远离衰减；第 0 个 = 太阳本体 halo 加呼吸
    var w = (1.0 - t) * 0.6 + 0.15;
    if (i == 0) {
      w *= 1.0 + 0.15 * sin(params.time * 2.0);
    }
    // 距离 ghost 中心的径向衰减（环状光斑）
    let d = length((in.uv - gp) * vec2<f32>(aspect, 1.0));
    w *= exp(-d * d * 18.0);

    flare += g * w;
  }

  // DirtMask 相乘（污渍纹理在强光处显影）
  let dirt = textureSample(dirtTex, samp, in.uv).rgb;
  let flareMasked = flare * dirt * params.dirtStrength * params.flareIntensity;

  let color = scene + bloom * params.bloomIntensity + flareMasked;

  return vec4<f32>(color, 1.0);
}
`;
