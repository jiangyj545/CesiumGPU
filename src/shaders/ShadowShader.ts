/**
 * ShadowShader — A1 CSSM 级联阴影着色器。
 *
 * 两个独立 WGSL 片段：
 *  1. {@link SHADOW_DEPTH_SHADER} — caster 侧 depth-only 顶点着色器。
 *     管线无颜色目标、无 fragment 阶段（WebGPU 允许：fragment 仅在存在
 *     color targets 时必需）。组 0 为级联光源 view-proj UBO（每级联一个
 *     bind group，同一 UBO 按 256B 对齐偏移），组 1 **复用 G-Buffer 管线的
 *     BGL1**（model UBO + baseColor 纹理 + 采样器）——Primitive 的
 *     `_opaqueModelBindGroup` 按对象身份绑定布局，因此阴影管线必须以
 *     `opaquePipeline.getBindGroupLayout(1)` 建管线才能共用该 bind group
 *     （同 buildUnlitPipeline 的身份复用先例）。
 *
 *  2. {@link SHADOW_RECEIVER_MODULE} — receiver 侧采样片段，由
 *     LightingShader 的 `injectShadowCore()` 以锚点替换注入 SHADER_CORE
 *     （组 2 uniform + texture_depth_2d_array + sampler_comparison，
 *     级联选择 + normal-offset + PCF 3×3）。
 */

/**
 * Caster depth-only 顶点着色器。
 * 顶点布局与 G-Buffer 一致（arrayStride 32，location 0 = position），
 * 但只声明 position —— 阴影深度不需要 normal/uv。
 */
export const SHADOW_DEPTH_SHADER = /* wgsl */`
// ── Uniforms ──────────────────────────────────────────────────────────────────

// 级联光源 view-proj（正交）。每级联一个 bind group（同一 UBO 的 256B 对齐偏移）。
struct ShadowView {
  lightViewProj : mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> shadowView : ShadowView;

// Model uniforms — normalMatrix 深度绘制用不到，但保留同一结构使 BGL1
// 与 G-Buffer 管线逐字段一致，从而能按身份复用 opaquePipeline 的组 1 布局。
struct ModelUniforms {
  modelMatrix  : mat4x4<f32>,
  normalMatrix : mat4x4<f32>,
}

@group(1) @binding(0) var<uniform> model : ModelUniforms;

// ── Depth-only vertex ─────────────────────────────────────────────────────────

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
  return shadowView.lightViewProj * (model.modelMatrix * vec4<f32>(position, 1.0));
}
`;

/**
 * Receiver 采样模块（注入 Lighting SHADER_CORE，位于 fs_main 之前）。
 *
 * ShadowUniforms 布局（256B，与 ShadowMap._writeReceiverUbo 一一对应）：
 *   cascadeMatrix : array<mat4x4<f32>, 3>  // @0    每级联光源 view-proj
 *   splitEnds     : vec4<f32>              // @192  级联远边界（归一化单位）
 *   texelSizes    : vec4<f32>              // @208  每级联纹素世界尺寸（归一化）
 *   params        : vec4<f32>              // @224  x=enabled y=depthBias
 *                                          //       z=normalOffsetScale w=mapSize
 */
export const SHADOW_RECEIVER_MODULE = /* wgsl */`
// ── A1 CSSM 级联阴影（组 2）─────────────────────────────────────────────────

struct ShadowUniforms {
  cascadeMatrix : array<mat4x4<f32>, 3>,
  splitEnds     : vec4<f32>,
  texelSizes    : vec4<f32>,
  params        : vec4<f32>,
}

@group(2) @binding(0) var<uniform> shadow     : ShadowUniforms;
@group(2) @binding(1) var           shadowTex : texture_depth_2d_array;
@group(2) @binding(2) var           shadowCmp : sampler_comparison;

// 按视距（归一化单位）选级联；超出覆盖范围返回 -1（无阴影）。
fn cssmSelectCascade(viewDist: f32) -> i32 {
  if (viewDist < shadow.splitEnds.x) { return 0; }
  if (viewDist < shadow.splitEnds.y) { return 1; }
  if (viewDist < shadow.splitEnds.z) { return 2; }
  return -1;
}

// PCF 3×3 comparison 采样。返回 1 = 全光照，0 = 全影。
fn cssmShadowFactor(worldPos: vec3<f32>, normal: vec3<f32>, viewDist: f32) -> f32 {
  if (shadow.params.x < 0.5) { return 1.0; }

  let ci = cssmSelectCascade(viewDist);
  if (ci < 0) { return 1.0; }

  // Normal-offset bias：沿地表法线把采样点抬离表面一个纹素量级，
  // 消除球面/瓦片斜面上的自阴影痤疮（斜率越大抬得越多）。
  let texelW = select(
    select(shadow.texelSizes.x, shadow.texelSizes.y, ci == 1),
    shadow.texelSizes.z,
    ci == 2);
  let offsetPos = worldPos + normal * (texelW * shadow.params.z);

  let lp   = shadow.cascadeMatrix[ci] * vec4<f32>(offsetPos, 1.0);
  let proj = lp.xyz / lp.w;

  // 出界判定用 NDC：xy ∈ [-1,1]，z ∈ [0,1]（z>=1 远平面外 / z<0 太阳侧外）。
  if (proj.x < -1.0 || proj.x > 1.0 || proj.y < -1.0 || proj.y > 1.0 || proj.z < 0.0 || proj.z >= 1.0) {
    return 1.0;
  }

  // NDC → 纹理 UV：x 半程平移；y 翻转（NDC +1 = 帧缓冲顶行 = 纹理 v 0）。
  let uv = vec2<f32>(proj.x * 0.5 + 0.5, 0.5 - 0.5 * proj.y);

  // 'ref' 是 WGSL 保留字 —— 比较参考深度命名 refDepth。
  let refDepth = proj.z - shadow.params.y;
  let texelUV  = 1.0 / shadow.params.w;
  var sum = 0.0;
  for (var dy: i32 = -1; dy <= 1; dy++) {
    for (var dx: i32 = -1; dx <= 1; dx++) {
      let tap = uv + vec2<f32>(f32(dx), f32(dy)) * texelUV;
      sum += textureSampleCompareLevel(shadowTex, shadowCmp, tap, ci, refDepth);
    }
  }
  return sum / 9.0;
}
`;

export default SHADOW_DEPTH_SHADER;
