/**
 * DynamicWaterShader - 动态水面 WGSL（对齐 EV-Globe 的 EV_DynamicWater / EV_Ocean）。
 *
 * 视觉语义：半透明水体覆盖底图，多层行波扰动法线产生波光（glint），
 * 波峰提亮、视角越平（菲涅尔）反射越强。波纹完全程序化（解析正弦行波
 * + value noise），与网格密度无关，湖面小域与海域大域一致。
 *
 * 遮挡：特效画在 swapchain 上（IPass 无深度附件），用射线-单位球求交
 * 在片元级剔除被地球挡住的远侧水面（与 DynamicWallShader 同款）。
 */
export const DYNAMIC_WATER_SHADER = /* wgsl */`
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

// 64B：color(16) + sunDir(16) + time/speed/waveScale/glint(16)。
struct WaterUniforms {
  color      : vec4<f32>,   // 水体基色 RGBA
  sunDir     : vec4<f32>,   // 太阳方向（指向太阳，归一化）
  time       : f32,         // 特效时间（秒）
  speed      : f32,         // 波速倍率
  waveScale  : f32,         // 波高（法线扰动强度）
  glint      : f32,         // 波光高光强度
}

@group(0) @binding(0) var<uniform> camera : CameraUniforms;
@group(0) @binding(1) var<uniform> water  : WaterUniforms;

// 遮挡球半径²：0.998² = 0.996004 大于压扁世界地表模长下界
// polarRatio² ≈ 0.99331——中高纬贴地片元被划进球内，相机升到球外
// 俯视时整面 discard。取 0.9945² < 下界，贴地片元恒在球外
//（对齐 OceanFFTShader / DynamicWallShader 常量约定）。
const GLOBE_OCCLUDE_R2 : f32 = 0.9945 * 0.9945;

// ── 程序化波纹（解析法线 + 波高）───────────────────────────────────────────

fn hash2(p : vec2<f32>) -> f32 {
  let q = fract(p * vec2<f32>(0.3183099, 0.3678794) + vec2<f32>(0.71, 0.113));
  return fract(43758.5453 * dot(q, vec2<f32>(23.14, 71.37)));
}

fn vnoise(p : vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash2(i), hash2(i + vec2<f32>(1.0, 0.0)), u.x),
    mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x),
    u.y
  );
}

// 三层行波叠加：返回 (法线 xyz, 波高 w)。
fn waveField(p : vec2<f32>, t : f32) -> vec4<f32> {
  var dhx = 0.0;
  var dhy = 0.0;
  var h   = 0.0;

  // 行波 1：主浪向
  var k = vec2<f32>(0.92, 0.38);
  var ph = dot(p, k) + t * 1.7;
  dhx += 0.55 * k.x * cos(ph);
  dhy += 0.55 * k.y * cos(ph);
  h   += 0.55 * sin(ph);

  // 行波 2：交叉浪
  k = vec2<f32>(-0.35, 0.94);
  ph = dot(p, k) + t * 2.3;
  dhx += 0.32 * k.x * cos(ph);
  dhy += 0.32 * k.y * cos(ph);
  h   += 0.32 * sin(ph);

  // 涟漪层：value noise 扰动
  let np = p * 2.6 + vec2<f32>(t * 0.8, -t * 0.6);
  let e = 0.08;
  let n0 = vnoise(np);
  dhx += 0.8 * (vnoise(np + vec2<f32>(e, 0.0)) - n0) / e;
  dhy += 0.8 * (vnoise(np + vec2<f32>(0.0, e)) - n0) / e;
  h   += n0 * 0.4;

  let s = water.waveScale;
  let n = normalize(vec3<f32>(-dhx * s, -dhy * s, 1.0));
  return vec4<f32>(n, h);
}

// ── Vertex ────────────────────────────────────────────────────────────────────

struct VSIn {
  @location(0) position : vec3<f32>,
  @location(1) uv       : vec2<f32>,
}

struct VSOut {
  @builtin(position) clipPos  : vec4<f32>,
  @location(0)       worldPos : vec3<f32>,
}

@vertex
fn vs_main(input : VSIn) -> VSOut {
  var out : VSOut;
  out.worldPos = input.position;
  out.clipPos  = camera.viewProjMatrix * vec4<f32>(input.position, 1.0);
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

  // 波纹空间：世界单位 1 ≈ 6378 km，放大约 9000 倍得到视觉波长；
  // 混入 y 分量避免极区 xz 投影退化。
  let waveP = (input.worldPos.xz + 0.6 * input.worldPos.yy) * 9000.0;
  let wf    = waveField(waveP, water.time * max(water.speed, 0.001));
  let n     = normalize(wf.xyz);
  let crest = smoothstep(0.25, 0.9, wf.w);          // 波峰亮斑

  let viewDir = normalize(camPos - input.worldPos);
  let refl    = reflect(-viewDir, n);
  let spec    = pow(max(dot(refl, normalize(water.sunDir.xyz)), 0.0), 64.0) * water.glint;
  let fres    = pow(1.0 - max(dot(viewDir, n), 0.0), 3.0);

  let base  = water.color.rgb * (0.72 + 0.28 * crest);
  let rgb   = base + vec3<f32>(spec) + water.color.rgb * fres * 0.55;
  let alpha = water.color.a * (0.62 + 0.38 * fres);

  return vec4<f32>(rgb, alpha);
}
`;
