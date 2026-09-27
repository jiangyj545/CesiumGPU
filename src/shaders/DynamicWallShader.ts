/**
 * DynamicWallShader - 动态墙 WGSL（T3.2，对齐 EV-Globe 的 EV_DynamicWall）。
 *
 * 视觉语义：底部亮、向上渐隐、横向光带沿墙体从底往上涌。
 *
 * - 流动：`vShift = fract(v - time * speed)`，v 底 0 顶 1，因此光带随
 *   time 增大向 +v（向上）移动。
 * - 渐隐：竖直包络 `pow(1 - v, 0.8)`（EV 公式基于 v=1 在底部；本实现
 *   几何 v 底 0 / 顶 1，故以 (1-v) 等价改写，保证"底部亮、顶部隐"）。
 * - 光带：`smoothstep(0.0, 0.15, x) * (1 - smoothstep(0.35, 0.5, x))`，
 *   与 EV 公式 `smoothstep(0, 0.15, x) * smoothstep(0.5, 0.35, x)` 数学等价
 *   （WGSL 规范中 edge0 > edge1 结果不确定，故用安全写法）。
 *
 * 遮挡：特效画在 swapchain 上（IPass 无深度附件），用射线-单位球求交
 * 在片元级剔除被地球挡住的远侧墙体。
 */
export const DYNAMIC_WALL_SHADER = /* wgsl */`
// ── Uniforms ──────────────────────────────────────────────────────────────────

// 256B 相机 UBO — 布局单一来源：src/renderer/CameraUniforms.ts
//（uniform 下 vec2 按 16 对齐 → viewportSize @208，nearFar @224 —— 非 @216）
struct CameraUniforms {
  viewMatrix      : mat4x4<f32>,   // @0   (64B)
  projMatrix      : mat4x4<f32>,   // @64  (64B)
  viewProjMatrix  : mat4x4<f32>,   // @128 (64B)
  cameraPosition  : vec4<f32>,     // @192 (16B)
  viewportSize    : vec2<f32>,     // @208 (8B)
  nearFar         : vec2<f32>,     // @224 (8B) x = near, y = far
}

// 32B：color(16) + speed/time/repeat/pad(16)。
struct WallUniforms {
  color  : vec4<f32>,
  speed  : f32,
  time   : f32,
  repeat : f32,
  _pad0  : f32,
}

@group(0) @binding(0) var<uniform> camera : CameraUniforms;
@group(0) @binding(1) var<uniform> wall   : WallUniforms;

// 遮挡球半径²：压扁世界地表 |p|² ≥ polarRatio² ≈ 0.99331（极点最小）。
// 旧值 0.998² = 0.996004 大于该下界——中高纬贴地片元被划进遮挡球内，
// 相机升到球外俯视时射线必穿球 → 整面 discard。取 0.9945² = 0.989030
// < 0.99331，任意纬度贴地片元恒在球外；真正超出地平线的远侧片元仍被
// 正确遮挡（与 OceanFFTShader / OceanCausticsShader 同一常量约定）。
const GLOBE_OCCLUDE_R2 : f32 = 0.9945 * 0.9945;

// ── Vertex ────────────────────────────────────────────────────────────────────

struct VSIn {
  @location(0) position : vec3<f32>,
  @location(1) normal   : vec3<f32>,
  @location(2) uv       : vec2<f32>,
}

struct VSOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0)       worldPos : vec3<f32>,
  @location(1)       v        : f32,
}

@vertex
fn vs_main(input : VSIn) -> VSOut {
  var out : VSOut;
  out.worldPos = input.position;
  out.clipPos  = camera.viewProjMatrix * vec4<f32>(input.position, 1.0);
  out.v        = input.uv.y; // 底 0 / 顶 1
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

  let v = clamp(input.v, 0.0, 1.0);

  // 流动：光带随 time 向上涌（speed 控制流速）。
  let vShift = fract(v - wall.time * wall.speed);
  // 横向光带（EV 公式的规范安全等价写法）。
  let band = smoothstep(0.0, 0.15, vShift) * (1.0 - smoothstep(0.35, 0.5, vShift));

  // 渐隐：底部亮（v=0 → 1）、向上渐隐（v=1 → 0）。
  let fade = pow(1.0 - v, 0.8);

  let lit   = 0.35 + 0.65 * band;
  let rgb   = wall.color.rgb * lit;
  let alpha = wall.color.a * (0.25 + 0.75 * band) * fade;

  return vec4<f32>(rgb, alpha);
}
`;
