import { FULLSCREEN_VS } from './Fullscreen';

/**
 * GodRaysShader - 经典屏幕空间 crepuscular rays（径向 light-shaft march）。
 *
 * 两步合单 pass（全分辨率、单三角，与 bloom 链 pass 同范式）：
 *   1. 亮度 mask：采样场景色（rgba16float 线性 HDR），lum 过 threshold
 *      后 soft-knee 归一 —— 高亮像素（太阳/天空亮部）是"光源"，
 *      暗像素（云/建筑剪影）自然成为遮挡物；
 *   2. 径向模糊：从当前像素向太阳屏幕位 march STEPS 步，逐步
 *      illum *= decay 累加 `mask 样本 * illum * weight`（经典
 *      crepuscular rays 参数化，density 控制路径长度）。
 *
 * 输出 rgba16float 光束纹理（已乘 tint），Composite 以 godRayIntensity
 * 叠加。太阳在屏外时 Manager 把 intensity 乘以余晖衰减（|ndc|≤1 全强、
 * 1→2.0 线性到 0），本 shader 无需判屏外（clamp-to-edge 采样天然安全）。
 *
 * Uniform 布局（48 B = 12 x f32）：
 *   [0..1] sunUv   [2] intensity  [3] decay    [4] density
 *   [5] weight     [6] threshold  [7] aspect   [8..10] tint  [11] pad
 */
export const GOD_RAYS_SHADER = FULLSCREEN_VS + /* wgsl */`
struct GodRaysParams {
  sunUv     : vec2<f32>, // 太阳屏幕位置（uv 域，y 已翻转）
  intensity : f32,       // 总强度（Manager 已乘屏外余晖衰减）
  decay     : f32,       // 每步衰减（≈0.96）
  density   : f32,       // 路径长度倍率（≈0.9）
  weight    : f32,       // 单样本权重（≈0.45）
  threshold : f32,       // HDR 亮度阈值（≥1 只取超亮光源）
  aspect    : f32,       // 宽高比（march 路径各向同性修正）
  tint      : vec3<f32>, // 光束染色
  pad       : f32,
}

const STEPS    = 20;  // march 步数（16-24 经验域中点）
const SOFTNESS = 0.6; // 阈值过渡带宽：(lum - threshold) / SOFTNESS

@group(0) @binding(0) var sceneTex : texture_2d<f32>;
@group(0) @binding(1) var samp     : sampler;
@group(0) @binding(2) var<uniform> params : GodRaysParams;

// VSOut 由 FULLSCREEN_VS 前置声明（重复声明会使模块解析失败——真机 CDP 验证发现）

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  // march 向量：当前像素 → 太阳。先在 aspect 修正的各向同性屏幕域取步长，
  // 再回投 uv 域 —— 宽屏上径向步进不会横向拉长。
  let toSunA = (params.sunUv - in.uv) * vec2<f32>(params.aspect, 1.0);
  let delta  = (toSunA / f32(STEPS)) / vec2<f32>(params.aspect, 1.0);

  var uv    = in.uv;
  var illum = 1.0;            // 经典 crepuscular 累积衰减
  var acc   = vec3<f32>(0.0);

  for (var i: i32 = 0; i < STEPS; i++) {
    uv += delta;
    let c   = textureSample(sceneTex, samp, uv).rgb;   // clamp-to-edge：屏外采样安全
    let lum = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    // 亮度 mask：高亮 = 光源，暗像素 = 遮挡物
    let mask = clamp((lum - params.threshold) / SOFTNESS, 0.0, 1.0);
    acc  += c * mask * illum * params.weight;
    illum *= params.decay;
  }

  // tint 在本 pass 末尾施加，Composite 只做强度叠加
  return vec4<f32>(acc * params.intensity * params.tint, 1.0);
}
`;
