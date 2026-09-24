/**
 * FloodWaterShader - 水深/淹没水面 WGSL（基于 DynamicWaterShader 改造）。
 *
 * 与 DynamicWater 的核心差异：水位是 uniform 而非烘焙在顶点里——
 * 顶点位置按**地形高程**径向布置，VS 内 `k = max(kTerrain, waterLevel)`
 * 径向放缩（水位变化零重建）；FS 用顶点插值的归一化地形高还原
 * `depth = waterLevel - terrainH`（米），按水深走 4 段色带
 * （浅青 0m → 青 2m → 蓝 5m → 深蓝 10m+），`depth <= 0.001` 的
 * 岸边片元输出 alpha 0（配合 src-alpha blend 剔除，免 CPU 裁剪）。
 *
 * 遮挡：与 DynamicWaterShader 同款射线-单位球求交（IPass 画在
 * swapchain 上、无深度附件）。
 */
export const FLOOD_WATER_SHADER = /* wgsl */`
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

// 64B：color(16) + sunDir(16) + time/speed/waveScale/glint(16)
//     + waterLevel/earthScale/pad(16)。
struct FloodUniforms {
  color      : vec4<f32>,   // 水体基色 RGBA（alpha 控制整体透明度）
  sunDir     : vec4<f32>,   // 太阳方向（指向太阳，归一化）
  time       : f32,         // 特效时间（秒）
  speed      : f32,         // 波速倍率
  waveScale  : f32,         // 波高（法线扰动强度）
  glint      : f32,         // 波光高光强度
  waterLevel : f32,         // 水位（归一化单位，米 / 6378137）
  earthScale : f32,         // 6378137：归一化单位 → 米
  pad0       : f32,
  pad1       : f32,
}

@group(0) @binding(0) var<uniform> camera : CameraUniforms;
@group(0) @binding(1) var<uniform> flood  : FloodUniforms;

// 地球半径略小于 1（归一化 ECEF），与 DynamicWaterShader 一致。
const GLOBE_OCCLUDE_R2 : f32 = 0.998 * 0.998;

// ── 程序化波纹（简化版：两层行波 + value noise，风格继承 DynamicWater）─────

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

  var k = vec2<f32>(0.92, 0.38);
  var ph = dot(p, k) + t * 1.7;
  dhx += 0.55 * k.x * cos(ph);
  dhy += 0.55 * k.y * cos(ph);
  h   += 0.55 * sin(ph);

  k = vec2<f32>(-0.35, 0.94);
  ph = dot(p, k) + t * 2.3;
  dhx += 0.32 * k.x * cos(ph);
  dhy += 0.32 * k.y * cos(ph);
  h   += 0.32 * sin(ph);

  let np = p * 2.6 + vec2<f32>(t * 0.8, -t * 0.6);
  let e = 0.08;
  let n0 = vnoise(np);
  dhx += 0.8 * (vnoise(np + vec2<f32>(e, 0.0)) - n0) / e;
  dhy += 0.8 * (vnoise(np + vec2<f32>(0.0, e)) - n0) / e;
  h   += n0 * 0.4;

  let s = flood.waveScale;
  let n = normalize(vec3<f32>(-dhx * s, -dhy * s, 1.0));
  return vec4<f32>(n, h);
}

// ── Vertex ────────────────────────────────────────────────────────────────────

struct VSIn {
  @location(0) position : vec3<f32>,
  // uv.x = 归一化地形高（米 / earthScale），uv.y = 地形处径向模长 kTerrain
  // （= 1 + 地形高/earthScale，位置折入模长的引擎约定）。
  @location(1) uv       : vec2<f32>,
}

struct VSOut {
  @builtin(position) clipPos   : vec4<f32>,
  @location(0)       worldPos  : vec3<f32>,
  @location(1)       terrainH  : f32,   // 归一化地形高（插值后 FS 还原水深）
}

@vertex
fn vs_main(input : VSIn) -> VSOut {
  // 水位（归一化）不低于地形处模长 → 径向放缩顶点。
  // 引擎约定：位置 = 单位方向 × k（y/z 已含极区压扁的线性缩放），
  // 径向放缩 = 逐分量乘 (k_new / k_terrain)，与 TerrainLayer stitch 同款
  //（不可用 atan2 地心纬度反推重建——项目坐标约定血泪坑）。
  let kTerrain = max(input.uv.y, 1e-7);
  let k        = max(kTerrain, flood.waterLevel);
  let worldPos = input.position * (k / kTerrain);

  var out : VSOut;
  out.worldPos = worldPos;
  out.terrainH = input.uv.x;
  out.clipPos  = camera.viewProjMatrix * vec4<f32>(worldPos, 1.0);
  return out;
}

// ── Fragment ──────────────────────────────────────────────────────────────────

@fragment
fn fs_main(input : VSOut) -> @location(0) vec4<f32> {
  // 水深（米）：水位 - 地形高（顶点插值 uv.x 还原）。
  let depthM = (flood.waterLevel - input.terrainH) * flood.earthScale;
  if (depthM <= 0.001) {
    return vec4<f32>(0.0); // 岸线以上：alpha 0，src-alpha blend 剔除
  }

  // 地球遮挡（IPass 无深度附件，片元级射线-球求交，与 DynamicWater 同款）。
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

  // 水深色带：浅青(0m) → 青(2m) → 蓝(5m) → 深蓝(10m+)，4 段 lerp。
  var band = vec3<f32>(0.55, 0.95, 0.95);
  band = mix(band, vec3<f32>(0.20, 0.75, 0.85), smoothstep(0.0, 2.0, depthM));
  band = mix(band, vec3<f32>(0.10, 0.50, 0.80), smoothstep(2.0, 5.0, depthM));
  band = mix(band, vec3<f32>(0.05, 0.25, 0.60), smoothstep(5.0, 10.0, depthM));
  let tinted = mix(flood.color.rgb, band, 0.65);

  // 波纹 / 菲涅尔 / 高光（继承 DynamicWater 风格）。
  let waveP = (input.worldPos.xz + 0.6 * input.worldPos.yy) * 9000.0;
  let wf    = waveField(waveP, flood.time * max(flood.speed, 0.001));
  let n     = normalize(wf.xyz);
  let crest = smoothstep(0.25, 0.9, wf.w);

  let viewDir = normalize(camPos - input.worldPos);
  let refl    = reflect(-viewDir, n);
  let spec    = pow(max(dot(refl, normalize(flood.sunDir.xyz)), 0.0), 64.0) * flood.glint;
  let fres    = pow(1.0 - max(dot(viewDir, n), 0.0), 3.0);

  let rgb   = tinted * (0.72 + 0.28 * crest) + vec3<f32>(spec) + tinted * fres * 0.45;
  // 浅水（岸线附近）更透，深水更实；depthM 已通过 0.001 门限。
  let alpha = flood.color.a * (0.55 + 0.45 * fres) * clamp(0.35 + depthM / 2.0, 0.0, 1.0);

  return vec4<f32>(rgb, clamp(alpha, 0.0, 1.0));
}
`;
