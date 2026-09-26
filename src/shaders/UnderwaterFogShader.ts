import { FULLSCREEN_VS } from './postprocess/Fullscreen';

/**
 * UnderwaterFogShader - 水下观感全屏叠加（tidewater "split underwater/above-water
 * view" 的水下侧简化版）。
 *
 * 相机处于海面之下时画一张全屏三角（draw(3)），以 premultiplied-alpha 混合
 * （one, one-minus-src-alpha）叠加到已合成场景之上。无场景深度纹理——深度感
 * 用两个近似表达：
 *   1. 视线俯仰渐变：相机 forward（viewMatrix 第三行取反）与"上"方向
 *      （归一化 ECEF 世界中 = normalize(cameraPosition)）点积，低头看海床
 *      时雾更浓、抬头看水面时透光更强；
 *   2. 屏幕径向暗角：边缘视线穿水路径更长 → 雾更厚、色调更暗。
 *
 * 三个视觉成分（全部程序生成，零纹理采样）：
 *   - 蓝绿深度雾：水体染色，透明度随俯仰/暗角渐变；
 *   - 浮动颗粒：两层屏幕空间 hash 点阵，慢速漂移 + 闪烁，密度由
 *     particleDensity 调制；
 *   - 焦散光斑：两层正弦波纹相乘调制亮度（呼应 OceanCaustics 的观感，
 *     但独立实现），抬头方向权重更大。
 *
 * Uniform 布局：
 *   binding 0 : CameraUniforms（256 B，布局单一来源 src/renderer/CameraUniforms.ts）
 *   binding 1 : UWParams（48 B = 3 x vec4）
 *
 * 输出为 premultiplied 颜色（rgb 已乘覆盖权重），配合
 * { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } 混合。
 * intensity = 0 时 rgb/alpha 全 0 → dst 逐像素原样保留（零贡献）。
 */
export const UNDERWATER_FOG_SHADER = FULLSCREEN_VS + /* wgsl */ `
struct CameraUniforms {
  viewMatrix     : mat4x4<f32>,   // @0   (64 B)
  projMatrix     : mat4x4<f32>,   // @64  (64 B)
  viewProjMatrix : mat4x4<f32>,   // @128 (64 B)
  cameraPosition : vec4<f32>,     // @192 (16 B) w 未用
  viewportSize   : vec2<f32>,     // @208
  nearFar        : vec2<f32>,     // @224
}

struct UWParams {
  params0    : vec4<f32>,   // x = time (s), y = intensity, z = particleDensity, w = unused
  viewport   : vec4<f32>,   // xy = 画布尺寸 (px), zw = unused
  waterColor : vec4<f32>,   // rgb = 水体染色, w = unused
}

@group(0) @binding(0) var<uniform> cam : CameraUniforms;
@group(0) @binding(1) var<uniform> uw  : UWParams;

// VSOut 由 FULLSCREEN_VS 前置声明（重复声明会使模块解析失败——同 BrightShader）

fn hash21(p: vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

// 单层屏幕空间浮游颗粒：cells x cells 网格，每格一枚慢速漂移的亮点，
// 带独立相位的呼吸式闪烁。seed 让两层错开（同一 hash 域不重叠）。
fn particleLayer(p: vec2<f32>, cells: f32, t: f32, seed: f32) -> f32 {
  let g    = p * cells;
  let id   = floor(g);
  let h1   = hash21(id + seed);
  let h2   = hash21(id + seed + 41.7);
  let drift = vec2<f32>(
    sin(t * 0.13 + h1 * 6.2832) * 0.35,
    cos(t * 0.11 + h2 * 6.2832) * 0.35
  );
  let local = fract(g) - vec2<f32>(0.5) - drift;
  // 反转写法：WGSL smoothstep 要求 low < high（CircleScanShader 同款规避）
  let spark = 1.0 - smoothstep(0.0, 0.16, length(local));
  let tw    = 0.55 + 0.45 * sin(t * (1.1 + h1 * 1.7) + h1 * 6.2832);
  return spark * tw;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let t         = uw.params0.x;
  let intensity = clamp(uw.params0.y, 0.0, 4.0);
  let dens      = clamp(uw.params0.z, 0.0, 4.0);
  let vp        = max(uw.viewport.xy, vec2<f32>(1.0));
  let aspect    = vp.x / vp.y;

  // ── 相机视线俯仰（归一化 ECEF：up = 径向向外）──────────────────────
  // 安全归一化：相机 UBO 兜底全零时 normalize(0) 会产 NaN，整屏坏帧
  let upRaw  = cam.cameraPosition.xyz;
  let up     = upRaw / max(length(upRaw), 1e-6);
  // viewMatrix 第三行 = 世界空间相机 forward 的取反（列主序取 [col][row=2]）
  let fwdRaw = -vec3<f32>(cam.viewMatrix[0][2], cam.viewMatrix[1][2], cam.viewMatrix[2][2]);
  let fwd    = fwdRaw / max(length(fwdRaw), 1e-6);
  let lookDown = clamp(-dot(fwd, up), 0.0, 1.0); // 1 = 低头看海床
  let lookUp   = clamp( dot(fwd, up), 0.0, 1.0); // 1 = 抬头看水面

  // ── 屏幕径向（暗角；uv 翻转对称，径向距离不受影响）────────────────
  let ndc  = in.uv * 2.0 - vec2<f32>(1.0);
  let r    = length(ndc);
  let vig  = smoothstep(0.55, 1.45, r);

  // ── 1、蓝绿深度雾：低头更浓，边缘更厚 ─────────────────────────────
  let fogBase = mix(0.28, 0.78, lookDown);
  let fog     = clamp(fogBase * (0.6 + 0.4 * vig) * intensity, 0.0, 0.95);

  // 轻微色偏青（绿/蓝通道略抬、红压暗）+ 边缘暗化
  var tint = uw.waterColor.rgb * vec3<f32>(0.85, 1.06, 1.04);
  tint = mix(tint, tint * 0.55, vig * 0.6);

  // ── 2、浮动颗粒：两层 hash 点阵，密度受 particleDensity 调制 ───────
  let sp = vec2<f32>(in.uv.x * aspect, in.uv.y);
  let p1 = particleLayer(sp,  9.0, t, 0.0);
  let p2 = particleLayer(sp, 17.0, t * 1.35 + 3.7, 19.19);
  let particles = clamp(p1 + p2 * 0.6, 0.0, 1.0) * dens * intensity;

  // ── 3、焦散光斑：两层正弦波纹相乘缓动（抬头方向权重更大）──────────
  let w1 = sin(sp.x * 6.0 + sp.y * 3.0 + t * 0.90);
  let w2 = sin(sp.x * 2.5 - sp.y * 5.0 - t * 0.62);
  let shimmer = pow(clamp(w1 * w2, 0.0, 1.0), 2.0);
  let shafts  = shimmer * (0.12 + 0.55 * lookUp) * intensity;

  // ── 输出：premultiplied（rgb 已乘覆盖权重），高光在薄雾区
  //    自然呈加性提亮；intensity = 0 → rgb/alpha 全 0 → dst 原样保留 ──
  let lift   = (particles * 0.35 + shafts * 0.30) * vec3<f32>(0.55, 0.85, 0.90);
  let rgb    = tint * fog + lift;
  let alpha  = clamp(fog + (particles + shafts) * 0.25, 0.0, 1.0);
  return vec4<f32>(rgb, alpha);
}
`;
