/**
 * HeatmapShader - 热力图叠加 WGSL（对齐 EV-Globe 的 EV_Heatmap）。
 *
 * 地表矩形网格逐顶点投影到 WGS84 椭球，UV 采样 CPU 端 canvas 合成的
 * 热力纹理（渐变圆 'lighter' 累积 + 色带上色），以 additive 发光混合
 * 叠加到最终画面。遮挡：射线-单位球求交剔除被地球挡住的远侧网格
 * （与 DynamicWallShader 同款）。
 */
export const HEATMAP_SHADER = /* wgsl */`
// ── Uniforms ──────────────────────────────────────────────────────────────────

// 256B 相机 UBO — 布局单一来源：src/renderer/CameraUniforms.ts
struct CameraUniforms {
  viewMatrix      : mat4x4<f32>,   // @0   (64B)
  projMatrix      : mat4x4<f32>,   // @64  (64B)
  viewProjMatrix  : mat4x4<f32>,   // @128 (64B)
  cameraPosition  : vec4<f32>,     // @192 (16B)
  viewportSize    : vec2<f32>,     // @208 (8B)
  nearFar         : vec2<f32>,     // @224 (8B) x = near, y = far
}

// 16B：opacity + pad。
struct HeatUniforms {
  opacity : f32,
  _pad0   : f32,
  _pad1   : f32,
  _pad2   : f32,
}

@group(0) @binding(0) var<uniform> camera : CameraUniforms;
@group(0) @binding(1) var<uniform> heat   : HeatUniforms;
@group(0) @binding(2) var heatTex   : texture_2d<f32>;
@group(0) @binding(3) var heatSamp  : sampler;

// 地球半径略小于 1（归一化 ECEF），与 DynamicWallShader 一致。
const GLOBE_OCCLUDE_R2 : f32 = 0.998 * 0.998;

// ── Vertex ────────────────────────────────────────────────────────────────────

struct VSIn {
  @location(0) position : vec3<f32>,
  @location(1) uv       : vec2<f32>,
}

struct VSOut {
  @builtin(position) clipPos  : vec4<f32>,
  @location(0)       worldPos : vec3<f32>,
  @location(1)       uv       : vec2<f32>,
}

@vertex
fn vs_main(input : VSIn) -> VSOut {
  var out : VSOut;
  out.worldPos = input.position;
  out.clipPos  = camera.viewProjMatrix * vec4<f32>(input.position, 1.0);
  out.uv       = input.uv;
  return out;
}

// ── Fragment ──────────────────────────────────────────────────────────────────

@fragment
fn fs_main(input : VSOut) -> @location(0) vec4<f32> {
  // 地球遮挡：相机 → 片段线段与单位球相交且最近交点落在片段之前，
  // 则该片段位于地球远侧，剔除（IPass 画在 swapchain 上、无深度附件）。
  let camPos = camera.cameraPosition.xyz;
  let oc   = input.worldPos - camPos;
  let a    = dot(oc, oc);
  let b    = 2.0 * dot(camPos, oc);
  let c    = dot(camPos, camPos) - GLOBE_OCCLUDE_R2;
  let disc = b * b - 4.0 * a * c;
  if (disc > 0.0) {
    let t = (-b - sqrt(disc)) / (2.0 * a);
    if (t > 1e-3 && t < 1.0 - 1e-3) {
      discard;
    }
  }

  let tex = textureSample(heatTex, heatSamp, input.uv);
  // 纹理 alpha 为热力强度；rgb 已是色带颜色。additive 发光叠加：
  // color = src.rgb * src.a * opacity + dst。
  let intensity = tex.a * heat.opacity;
  return vec4<f32>(tex.rgb * intensity, intensity);
}
`;
