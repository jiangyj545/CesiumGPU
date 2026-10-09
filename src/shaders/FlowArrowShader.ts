/**
 * FlowArrowShader - 沿折线累计弧长滚动的程序化箭头（P2-6）。
 *
 * 顶点布局 [x,y,z, arcLen, v]；uniform：
 *   binding 0 : viewProj(16) + baseColor(4) + arrowColor(4) + params(4) = 28 floats
 *   params = [time(s), speed(cycle/s), gap(arrows/unit), unused]
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
  params     : vec4<f32>,    // @96  x=time y=speed z=gap
}

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) arc : f32,
  @location(1) v   : f32,
}

@group(0) @binding(0) var<uniform> u : Uniforms;

@vertex
fn vs_main(@location(0) pos : vec3<f32>,
           @location(1) arc : f32,
           @location(2) v   : f32) -> VSOut {
  var out : VSOut;
  out.clip = u.viewProj * vec4<f32>(pos, 1.0);
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
