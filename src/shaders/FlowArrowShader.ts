/**
 * FlowArrowShader - 沿折线累计弧长滚动的程序化箭头（P2-6）。
 * 顶点布局 [x,y,z, arc, v, side.xyz]（8 floats / stride 32 B，位置 = 纯中心线）；
 * 顶点着色器按**屏幕空间恒定像素宽**展开（CesiumJS polyline 语义）：
 * side 投影到屏幕归一化后乘 halfWidthDevicePx（透视正确——除以本
 * 顶点 w），再换回 clip 坐标偏移。
 *
 * uniform（binding 0，128 B）：
 *   viewProj(64B) + baseColor(16B) + arrowColor(16B)
 *   + params(16B: x=time, y=speed, z=gap, w=halfWidthDevicePx)
 *   + viewport(16B: xy=canvas device px)
 *
 * 深度遮挡（binding 1/2）：采样 GBuffer 场景深度（depth32float，
 * non-filtering sampler），片元深度在场景之后（被地形/地球遮挡）即
 * discard——对齐 CesiumJS 折线的遮挡观感。
 *
 * 片元：sPos = fract(arc·gap − t·speed)（箭头向 s 增大方向滚动，相位只依赖
 * 标量弧长——与段方向无关，不翻转）；形状 = 渐亮尾 + 三角头（常数与
 * hexapod FLOW_ARROW_SHADER 同值），叠加在半透明底线上。
 * 混合：标准 src-alpha（EffectManager 在 composite 后叠加到最终目标）。
 */
export const FLOW_ARROW_SHADER = /* wgsl */ `
struct Uniforms {
  viewProj   : mat4x4<f32>,  // @0
  baseColor  : vec4<f32>,    // @64
  arrowColor : vec4<f32>,    // @80
  params     : vec4<f32>,    // @96   x=time y=speed z=gap w=halfWidthDevicePx
  viewport   : vec4<f32>,    // @112  xy=canvas device px
}

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) arc : f32,
  @location(1) v   : f32,
}

@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var sceneDepth : texture_depth_2d;
@group(0) @binding(2) var depthSamp  : sampler;

@vertex
fn vs_main(@location(0) pos  : vec3<f32>,
           @location(1) arc  : f32,
           @location(2) v    : f32,
           @location(3) side : vec3<f32>) -> VSOut {
  // 屏幕空间恒定像素宽展开（透视正确：除以本顶点 w）。
  let clip = u.viewProj * vec4<f32>(pos, 1.0);
  let sideClip = (u.viewProj * vec4<f32>(side, 0.0)).xy;
  let w = max(clip.w, 1e-6);
  let dirPx = sideClip / w * u.viewport.xy * 0.5;
  let dl = max(length(dirPx), 1e-6);
  let offPx = dirPx / dl * u.params.w;
  let sgn = v * 2.0 - 1.0;
  var out : VSOut;
  out.clip = vec4<f32>(clip.xy + sgn * offPx / (u.viewport.xy * 0.5) * clip.w, clip.z, clip.w);
  out.arc = arc;
  out.v = v;
  return out;
}

const TIP = 0.78;
const HEAD_LEN = 0.20;
const TAIL_LEN = 0.48;
const HEAD_HALF_MAX = 0.32;

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
  // 深度遮挡：被 GBuffer 已写深度（地形/地球）遮住即丢弃。
  let fragCoord = in.clip;
  let uv = fragCoord.xy / u.viewport.xy;
  let sd = textureSampleLevel(sceneDepth, depthSamp, uv, 0);
  if (fragCoord.z > sd + 5e-5) { discard; }

  let raw = in.arc * u.params.z - u.params.x * u.params.y;
  let sPos = raw - floor(raw);
  let tc = in.v - 0.5;

  let tailHalf = HEAD_HALF_MAX * 0.30;
  let tailStart = TIP - HEAD_LEN - TAIL_LEN;

  let dHead = TIP - sPos;
  let inHead = select(0.0, 1.0, dHead >= 0.0 && dHead <= HEAD_LEN);
  let inTail = select(0.0, 1.0, sPos >= tailStart && sPos <= TIP - HEAD_LEN);

  let headHalf = HEAD_HALF_MAX * clamp(dHead / HEAD_LEN, 0.0, 1.0);
  let halfW = max(inHead * headHalf, inTail * tailHalf);
  let inside = select(0.0, 1.0, abs(tc) < halfW) * max(inHead, inTail);

  let dTail = sPos - tailStart;
  let tailFade = 0.15 + 0.85 * clamp(dTail / (TAIL_LEN * 0.75), 0.0, 1.0);
  let fade = select(1.0, tailFade, inTail > 0.5);

  let arrow = vec4<f32>(u.arrowColor.rgb, u.arrowColor.a * fade);
  return mix(u.baseColor, arrow, inside);
}
`;
