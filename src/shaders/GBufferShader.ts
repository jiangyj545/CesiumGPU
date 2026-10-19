/**
 * G-Buffer shader for opaque objects (Earth surface).
 * Outputs: albedo, normal (encoded), world position to G-Buffer render targets.
 */
export const GBUFFER_SHADER = /* wgsl */`
// ── Uniforms ──────────────────────────────────────────────────────────────────

// 256B camera UBO — layout single source of truth: src/renderer/CameraUniforms.ts
// (uniform vec2 aligns to 16 → viewportSize @208, nearFar @224 — NOT @216)
struct CameraUniforms {
  viewMatrix      : mat4x4<f32>,   // @0   (64B)
  projMatrix      : mat4x4<f32>,   // @64  (64B)
  viewProjMatrix  : mat4x4<f32>,   // @128 (64B)
  cameraPosition  : vec4<f32>,     // @192 (16B) world-space camera position (w = unused)
  viewportSize    : vec2<f32>,     // @208 (8B)
  nearFar         : vec2<f32>,     // @224 (8B) x = near, y = far
  viewRotProj     : mat4x4<f32>,   // @240 (64B) RTC: proj·viewRot（view 平移置零）
  camHi           : vec4<f32>,     // @304 (16B) RTC: 相机位置 double-single 高位
  camLo           : vec4<f32>,     // @320 (16B) RTC: 低位
}

struct ModelUniforms {
  modelMatrix  : mat4x4<f32>,
  normalMatrix : mat4x4<f32>,  // upper-left 3x3 of inverse-transpose model matrix
  originHi     : vec4<f32>,    // @128 RTC: 几何中心高位（真 WGS84 归一化），w=flag(1=RTC)
  originLo     : vec4<f32>,    // @144 RTC: 中心低位；顶点 attribute = 相对中心偏移
}

@group(0) @binding(0) var<uniform> camera : CameraUniforms;
@group(1) @binding(0) var<uniform> model  : ModelUniforms;
@group(1) @binding(1) var baseColorTexture : texture_2d<f32>;
@group(1) @binding(2) var baseColorSampler : sampler;

// ── Vertex I/O ────────────────────────────────────────────────────────────────

struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) normal   : vec3<f32>,
  @location(2) uv       : vec2<f32>,
}

struct VertexOutput {
  @builtin(position) clipPosition : vec4<f32>,
  @location(0) worldPosition      : vec3<f32>,
  @location(1) worldNormal        : vec3<f32>,
  @location(2) uv                 : vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  let worldPos4  = model.modelMatrix * vec4<f32>(in.position, 1.0);
  let worldNorm4 = model.normalMatrix * vec4<f32>(in.normal, 0.0);

  out.worldPosition = worldPos4.xyz;
  out.worldNormal   = normalize(worldNorm4.xyz);
  out.uv            = in.uv;
  if (model.originHi.w > 0.5) {
    // RTC（相对相机渲染，encoded camera）：顶点 attribute = 相对几何中心的
    // 偏移（幅度小 → f32 量化亚毫米）。hi/lo 差值经 Sterbenz 精确相减，
    // δ 相互抵消 → 相对相机坐标 f32 无损；深度不再携带 ±0.38m 顶点量化。
    let dHi = model.originHi.xyz - camera.camHi.xyz;
    let dLo = model.originLo.xyz - camera.camLo.xyz;
    let pRelEye = in.position + dHi + dLo;
    out.worldPosition = model.originHi.xyz + in.position;  // worldPos RT（光照用）
    out.clipPosition  = camera.viewRotProj * vec4<f32>(pRelEye, 1.0);
  } else {
    out.clipPosition  = camera.viewProjMatrix * worldPos4;
  }
  return out;
}

// ── G-Buffer Fragment Output ───────────────────────────────────────────────────
// Target 0: albedo       (rgba8unorm)
// Target 1: normal       (rgba16float) — world-space normal in [−1,1]
// Target 2: worldPos     (rgba32float) — world-space position

struct GBufferOutput {
  @location(0) albedo   : vec4<f32>,
  @location(1) normal   : vec4<f32>,
  @location(2) worldPos : vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> GBufferOutput {
  var out: GBufferOutput;

  // Sample base color texture
  let texColor = textureSample(baseColorTexture, baseColorSampler, in.uv);

  out.albedo   = texColor;
  out.normal   = vec4<f32>(normalize(in.worldNormal), 1.0);
  out.worldPos = vec4<f32>(in.worldPosition, 1.0);

  return out;
}

// ── Unlit fragment entry (effect/emissive geometry, T3.1) ─────────────────────
// Same vertex stage, same G-Buffer targets as fs_main, but writes a
// zero-length normal as the "unlit" marker.  The lighting pass detects
// dot(N,N) < 0.5 and outputs raw albedo (no shading / haze / tonemap),
// so the primitive renders pure emissive colour.
@fragment
fn fs_unlit(in: VertexOutput) -> GBufferOutput {
  var out: GBufferOutput;

  let texColor = textureSample(baseColorTexture, baseColorSampler, in.uv);

  out.albedo   = texColor;
  out.normal   = vec4<f32>(vec3<f32>(0.0), 0.0);  // zero normal = unlit marker
  out.worldPos = vec4<f32>(in.worldPosition, 1.0);

  return out;
}
`;

/** Fragment entry point for the unlit G-Buffer pipeline variant. */
export const GBUFFER_UNLIT_ENTRY_POINT = 'fs_unlit';

export default GBUFFER_SHADER;
