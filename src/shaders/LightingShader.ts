/**
 * Deferred lighting pass shader.
 * Reads G-Buffer textures and computes PBR-style directional lighting.
 * Full-screen triangle approach.
 *
 * P2 后处理双路径（见 LightingPass.setHdrOutput）：
 *  - LIGHTING_SHADER     LDR：内嵌 Reinhard + gamma（旧路径，WGSL 逐字节不变）。
 *  - LIGHTING_SHADER_HDR HDR：线性直通（Reinhard/gamma 交给后处理
 *    ToneMappingPass），供 postProcess.enabled 时使用。
 * 两变体共享 SHADER_CORE —— 主体只有一份，尾段分叉。
 */
import { SHADOW_RECEIVER_MODULE } from './ShadowShader';
import { CLOUD_SHADOW_RECEIVER_MODULE } from './CloudShader';

const SHADER_CORE = /* wgsl */`
// ── Uniforms ──────────────────────────────────────────────────────────────────

// 256B camera UBO — layout single source of truth: src/renderer/CameraUniforms.ts
// (uniform vec2 aligns to 16 → viewportSize @208, nearFar @224 — NOT @216)
struct CameraUniforms {
  viewMatrix      : mat4x4<f32>,   // @0   (64B)
  projMatrix      : mat4x4<f32>,   // @64  (64B)
  viewProjMatrix  : mat4x4<f32>,   // @128 (64B)
  cameraPosition  : vec4<f32>,     // @192 (16B)
  viewportSize    : vec2<f32>,     // @208 (8B)
  nearFar         : vec2<f32>,     // @224 (8B) x = near, y = far
}

struct DirectionalLight {
  direction        : vec4<f32>,  // world-space, normalised (w = unused)
  color            : vec4<f32>,  // RGB intensity (w = unused)
  ambientIntensity : f32,
  _pad0            : f32,
  _pad1            : f32,
  _pad2            : f32,
}

struct LightingUniforms {
  lights     : array<DirectionalLight, 8>,
  lightCount : u32,
  _pad0      : u32,
  _pad1      : u32,
  _pad2      : u32,
}

@group(0) @binding(0) var<uniform> camera   : CameraUniforms;
@group(0) @binding(1) var<uniform> lighting : LightingUniforms;

@group(1) @binding(0) var gAlbedo   : texture_2d<f32>;
@group(1) @binding(1) var gNormal   : texture_2d<f32>;
@group(1) @binding(2) var gWorldPos : texture_2d<f32>;
@group(1) @binding(3) var gSampler  : sampler;

// ── Full-screen triangle vertex shader ────────────────────────────────────────

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0)       uv  : vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
  // Generate a full-screen triangle (covers the entire NDC quad)
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  var uvs = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(2.0, 1.0),
    vec2<f32>(0.0, -1.0),
  );
  var out: VSOut;
  out.pos = vec4<f32>(positions[vid], 0.0, 1.0);
  out.uv  = uvs[vid];
  return out;
}

// ── Deferred lighting fragment shader ─────────────────────────────────────────

fn computeDirectionalLight(
  light    : DirectionalLight,
  normal   : vec3<f32>,
  viewDir  : vec3<f32>,
  albedo   : vec3<f32>,
) -> vec3<f32> {
  let L = normalize(-light.direction.xyz);
  let H = normalize(L + viewDir);

  // Lambertian diffuse
  let NdotL = max(dot(normal, L), 0.0);
  let diffuse = albedo * NdotL;

  // Blinn-Phong specular (simple)
  let NdotH = max(dot(normal, H), 0.0);
  let specular = pow(NdotH, 32.0) * 0.3;

  let ambient = albedo * light.ambientIntensity;

  return (diffuse + specular) * light.color.rgb + ambient;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let texCoord = in.uv;

  let albedo4   = textureSample(gAlbedo,   gSampler, texCoord);
  let normal4   = textureSample(gNormal,   gSampler, texCoord);
  let worldPos4 = textureSample(gWorldPos, gSampler, texCoord);

  // Skip background pixels (alpha == 0 in G-Buffer)
  if (worldPos4.w < 0.5) {
    discard;
  }

  let albedo   = albedo4.rgb;

  // Unlit passthrough (T3.1): the unlit G-Buffer fragment entry writes a
  // zero-length normal as a marker.  Bypass lighting, haze, tone mapping and
  // gamma entirely so effect geometry keeps its raw emissive colour.
  if (dot(normal4.xyz, normal4.xyz) < 0.5) {
    return vec4<f32>(albedo, 1.0);
  }

  let normal   = normalize(normal4.xyz);
  let worldPos = worldPos4.xyz;
  let viewDir  = normalize(camera.cameraPosition.xyz - worldPos);

  var color = vec3<f32>(0.0);

  for (var i: u32 = 0u; i < lighting.lightCount; i++) {
    color += computeDirectionalLight(lighting.lights[i], normal, viewDir, albedo);
  }

  // Atmospheric haze: slight blue tint near limb
  let earthCenter = vec3<f32>(0.0, 0.0, 0.0);
  let toSurface   = normalize(worldPos - earthCenter);
  let limbFactor  = 1.0 - abs(dot(toSurface, viewDir));
  let haze = vec3<f32>(0.1, 0.2, 0.4) * pow(limbFactor, 3.0) * 0.8;

  color = color + haze;
`;

// ── LDR 尾段：旧内嵌 Reinhard + gamma（与拆分前逐字节一致）─────────────────
const TAIL_LDR = /* wgsl */`
  // Tone mapping (Reinhard)
  color = color / (color + vec3<f32>(1.0));

  // Gamma correction
  color = pow(color, vec3<f32>(1.0 / 2.2));

  return vec4<f32>(color, 1.0);
}
`;

// ── HDR 尾段：线性直通（后处理 ToneMappingPass 负责映射 + gamma）────────────
const TAIL_HDR = /* wgsl */`
  // P2 HDR passthrough: tone mapping + gamma move to ToneMappingPass
  return vec4<f32>(color, 1.0);
}
`;

export const LIGHTING_SHADER = SHADER_CORE + TAIL_LDR;

export const LIGHTING_SHADER_HDR = SHADER_CORE + TAIL_HDR;

// ── A1 CSSM 级联阴影变体 ─────────────────────────────────────────────────────
//
// 以锚点替换把 SHADOW_RECEIVER_MODULE 注入 SHADER_CORE，形成 4 变体中的
// 带阴影 2 个（LDR / HDR 尾段复用）。无阴影的 LIGHTING_SHADER / _HDR 逐字节
// 不变 —— LightingPass 走原管线变体即零回归。

/** 光照循环锚点：唯一匹配 fs_main 中的 directional 光累加循环。 */
const LOOP_ANCHOR = `  for (var i: u32 = 0u; i < lighting.lightCount; i++) {
    color += computeDirectionalLight(lighting.lights[i], normal, viewDir, albedo);
  }`;

/** 带阴影的循环：仅第 0 盏光（太阳）乘 CSSM 阴影因子。 */
const LOOP_SHADOWED = `  let viewDist = length(worldPos - camera.cameraPosition.xyz);
  for (var i: u32 = 0u; i < lighting.lightCount; i++) {
    var shadowFactor = 1.0;
    if (i == 0u) {
      shadowFactor = cssmShadowFactor(worldPos, normal, viewDist);
    }
    color += shadowFactor * computeDirectionalLight(lighting.lights[i], normal, viewDir, albedo);
  }`;

/** fs_main 前插入锚点：阴影 uniform/采样器/辅助函数声明插入点。 */
const FRAGMENT_ANCHOR = `@fragment
fn fs_main(in: VSOut)`;

/**
 * 把 CSSM receiver 模块注入光照核心。
 * 两个锚点都必须恰好命中，否则抛错（禁止静默产出错误 shader）。
 */
export function injectShadowCore(core: string): string {
  if (!core.includes(FRAGMENT_ANCHOR) || !core.includes(LOOP_ANCHOR)) {
    throw new Error('LightingShader.injectShadowCore: SHADER_CORE anchors drifted — update LOOP_ANCHOR/FRAGMENT_ANCHOR');
  }
  return core
    .replace(FRAGMENT_ANCHOR, SHADOW_RECEIVER_MODULE + FRAGMENT_ANCHOR)
    .replace(LOOP_ANCHOR, LOOP_SHADOWED);
}

/** CSSM 阴影 + LDR 尾段（旧内嵌 Reinhard + gamma）。 */
export const LIGHTING_SHADER_SHADOW = injectShadowCore(SHADER_CORE) + TAIL_LDR;

/** CSSM 阴影 + HDR 直通尾段（配 ToneMappingPass）。 */
export const LIGHTING_SHADER_SHADOW_HDR = injectShadowCore(SHADER_CORE) + TAIL_HDR;

// ── Track C 云影（CloudLayer 烘焙贴图，组 3）变体 ────────────────────────────
//
// injectShadowCore 同款锚点拼接：把 CLOUD_SHADOW_RECEIVER_MODULE（CloudShader.ts
// 导出，含组 3 uniform/纹理/采样器声明 + cloudShadowFactor）注入 SHADER_CORE，
// 仅第 0 盏光（太阳）再乘云影因子。四个既有导出（LIGHTING_SHADER / _HDR /
// _SHADOW / _SHADOW_HDR）逐字节不变 —— LightingPass 未接云影时走原变体，
// 零回归。与 CSSM 正交组合成 8 变体（4 旧 + 4 新）。

/** 原始光照循环注入云影：仅第 0 盏光乘 cloudShadowFactor。 */
const LOOP_CLOUDED = `  for (var i: u32 = 0u; i < lighting.lightCount; i++) {
    var shadowFactor = 1.0;
    if (i == 0u) {
      shadowFactor = cloudShadowFactor(worldPos);
    }
    color += shadowFactor * computeDirectionalLight(lighting.lights[i], normal, viewDir, albedo);
  }`;

/** CSSM 阴影循环（LOOP_SHADOWED）叠加云影：两因子相乘。 */
const LOOP_SHADOW_CLOUDED = `  let viewDist = length(worldPos - camera.cameraPosition.xyz);
  for (var i: u32 = 0u; i < lighting.lightCount; i++) {
    var shadowFactor = 1.0;
    if (i == 0u) {
      shadowFactor = cssmShadowFactor(worldPos, normal, viewDist)
        * cloudShadowFactor(worldPos);
    }
    color += shadowFactor * computeDirectionalLight(lighting.lights[i], normal, viewDir, albedo);
  }`;

/**
 * 把云影 receiver 模块注入光照核心（可链在 injectShadowCore 之后）。
 * 循环锚点二选一（带/不带 CSSM 阴影）必须恰好命中一个，声明锚点沿用
 * FRAGMENT_ANCHOR（注入后文本保留，可多次插入），否则抛错。
 */
export function injectCloudShadowCore(core: string): string {
  const hasPlain = core.includes(LOOP_ANCHOR);
  const hasCssm = core.includes(LOOP_SHADOWED);
  if ((!hasPlain && !hasCssm) || !core.includes(FRAGMENT_ANCHOR) || (hasPlain && hasCssm)) {
    throw new Error('LightingShader.injectCloudShadowCore: anchors drifted — update LOOP_ANCHOR/LOOP_SHADOWED/FRAGMENT_ANCHOR');
  }
  return core
    .replace(FRAGMENT_ANCHOR, CLOUD_SHADOW_RECEIVER_MODULE + FRAGMENT_ANCHOR)
    .replace(hasCssm ? LOOP_SHADOWED : LOOP_ANCHOR, hasCssm ? LOOP_SHADOW_CLOUDED : LOOP_CLOUDED);
}

/** 云影 + LDR 尾段（无 CSSM）。 */
export const LIGHTING_SHADER_CLOUD = injectCloudShadowCore(SHADER_CORE) + TAIL_LDR;

/** 云影 + HDR 直通尾段（无 CSSM）。 */
export const LIGHTING_SHADER_CLOUD_HDR = injectCloudShadowCore(SHADER_CORE) + TAIL_HDR;

/** CSSM 阴影 + 云影 + LDR 尾段。 */
export const LIGHTING_SHADER_SHADOW_CLOUD =
  injectCloudShadowCore(injectShadowCore(SHADER_CORE)) + TAIL_LDR;

/** CSSM 阴影 + 云影 + HDR 直通尾段（配 ToneMappingPass）。 */
export const LIGHTING_SHADER_SHADOW_CLOUD_HDR =
  injectCloudShadowCore(injectShadowCore(SHADER_CORE)) + TAIL_HDR;

export default LIGHTING_SHADER;
