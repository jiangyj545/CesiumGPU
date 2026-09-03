/**
 * VideoOverlayShader - 云台视频叠加 WGSL（网格贴图 + 视锥线框，双 entry point）。
 *
 * 两个渲染管线共用本模块与同一 bindGroupLayout：
 * - 网格管线（vs_main / fs_main）：triangle-list，世界坐标经 `texMatrix`
 *   投影到视频纹理 [0,1]² UV 后采样 `rgba8unorm` 视频帧，乘 tint/opacity。
 * - 线框管线（vs_line / fs_line）：line-list，输出视锥线框颜色。
 *
 * 遮挡：特效画在 swap-chain 上（IPass 无深度附件），镜像 DynamicWallShader
 * 的射线-球求交剔除被地球挡住的远侧片段 —— 相机 → 片段线段与半径
 * sqrt(GLOBE_OCCLUDE_R2) 的球相交且最近交点落在片段之前则 discard。
 * 相机位置由 uniform 传入（本 pass 没有 256B 相机 UBO，见 `camPos` @176）。
 *
 * UV y 翻转（NDC_TO_UV_Y_SIGN = -1）：已在 CPU 侧折入 `texMatrix`
 * （buildTextureMatrix 的 scale(0.5, -0.5, 0.5) 项），shader 只做 uv = xy / w。
 */
export const VIDEO_OVERLAY_SHADER = /* wgsl */`
// ── Uniforms ──────────────────────────────────────────────────────────────────

// 192B：mat4x4 16B 对齐，全部成员偏移天然 16 对齐，无隐藏 padding。
struct VideoOverlayUniforms {
  viewProj  : mat4x4<f32>,   // @0   主相机 view * projection
  texMatrix : mat4x4<f32>,   // @64  世界坐标 -> 视频 UV（y 翻转已在 CPU 侧乘入，
                             //        见 GimbalMath.NDC_TO_UV_Y_SIGN）
  color     : vec4<f32>,     // @128 rgb = tint，a = opacity
  params    : vec4<f32>,     // @144 x = time(s)，y = seeking 标志（1 = 尚无视频帧，
                             //        画纯 tint*0.3 占位），z/w 预留
  lineColor : vec4<f32>,     // @160 视锥线框颜色（rgb + a）
  camPos    : vec4<f32>,     // @176 xyz = 主相机世界位置（归一化 ECEF），w 预留
}

@group(0) @binding(0) var<uniform> overlay : VideoOverlayUniforms;
@group(0) @binding(1) var videoTex  : texture_2d<f32>;
@group(0) @binding(2) var videoSamp : sampler;

// 地球遮挡判定半径的平方（略小于 1，归一化 ECEF），与 DynamicWallShader 同一常量约定。
const GLOBE_OCCLUDE_R2 : f32 = 0.998 * 0.998;

// ── Vertex（两管线共用布局：location 0 = position float32x3） ─────────────────

struct VSIn {
  @location(0) position : vec3<f32>
}

struct VSOut {
  @builtin(position) clipPos  : vec4<f32>,
  @location(0)       worldPos : vec3<f32>
}

@vertex
fn vs_main(input : VSIn) -> VSOut {
  var out : VSOut;
  out.worldPos = input.position;
  out.clipPos  = overlay.viewProj * vec4<f32>(input.position, 1.0);
  return out;
}

@vertex
fn vs_line(input : VSIn) -> VSOut {
  var out : VSOut;
  out.worldPos = input.position;
  out.clipPos  = overlay.viewProj * vec4<f32>(input.position, 1.0);
  return out;
}

// ── 地球遮挡（镜像 DynamicWallShader，相机位置改由 uniform 提供） ─────────────

// 相机 → 片段线段与球（半径 sqrt(GLOBE_OCCLUDE_R2)）相交且最近交点落在
// 片段之前（t ∈ (eps, 1-eps)，t 以线段长度归一）→ 片段在地球远侧，剔除。
fn occluded(worldPos : vec3<f32>) -> bool {
  let camPos = overlay.camPos.xyz;
  let oc   = worldPos - camPos;
  let a    = dot(oc, oc);
  let b    = 2.0 * dot(camPos, oc);
  let c    = dot(camPos, camPos) - GLOBE_OCCLUDE_R2;
  let disc = b * b - 4.0 * a * c;
  if (disc > 0.0) {
    let t = (-b - sqrt(disc)) / (2.0 * a);
    if (t > 1e-3 && t < 1.0 - 1e-3) {
      return true;
    }
  }
  return false;
}

// ── Fragment ──────────────────────────────────────────────────────────────────

@fragment
fn fs_main(input : VSOut) -> @location(0) vec4<f32> {
  let projUV = overlay.texMatrix * vec4<f32>(input.worldPos, 1.0);
  // uv = xy / w：y 翻转已折入 texMatrix（CPU 侧 NDC_TO_UV_Y_SIGN），此处不做翻转。
  let uv = projUV.xy / projUV.w;

  // textureSample 需 uniform control flow（隐式导数），
  // 因此先采样、后做依赖插值 worldPos 的 discard（遮挡 / 越界判断）。
  var rgb = overlay.color.rgb;
  var a   = 0.3;                       // seeking 占位：纯 tint * 0.3
  if (overlay.params.y < 0.5) {
    let tex = textureSample(videoTex, videoSamp, uv);
    rgb = tex.rgb * overlay.color.rgb;
    a   = tex.a;
  }

  if (occluded(input.worldPos)) { discard; }
  if (projUV.w <= 0.0001) { discard; }
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { discard; }
  return vec4<f32>(rgb, a * overlay.color.a);
}

@fragment
fn fs_line(input : VSOut) -> @location(0) vec4<f32> {
  if (occluded(input.worldPos)) { discard; }
  return overlay.lineColor;
}
`;
