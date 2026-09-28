/**
 * BreakSpray shaders — 岸线破碎波喷雾粒子（海洋 L5/T4，架构照抄 BowSprayShader）。
 *
 * 与 BowSpray 的差异只在发射模型与吸附逻辑，其余（状态布局 64B/粒子、
 * instanced billboard、premultiplied alpha、G-Buffer 深度测试、环形发射
 * 游标）逐行同构：
 * - 发射：岸线破碎带内沿 alongDir 均匀撒点（bandCenter ± widthM/2），
 *   CPU 侧 emit = min(round(intensity·cap), cap)，每帧环形窗口重生；
 * - 吸附：发射时 ECEF → 海面局部平面米（east/north 基最小二乘投影，
 *   OceanFFTWater.bindSeabed 同式）→ seabed uv → textureLoad 海床高 +
 *   disp 级联0 波高，出生高度贴合波面；水深门 smoothstep(farDepth,
 *   nearDepth, depth) 概率薄化（浅处多喷、深处少喷）；
 * - 积分：重力沿发射帧 -up + 线性拖拽；h < 0（落回静水面）死亡回收。
 *
 * 模拟空间选型（世界空间归一化 ECEF、f32 ULP 论证）与 BowSprayShader 一致，
 * 详见其文件头，此处不赘述。
 *
 * 状态布局 — 64 B/粒子（16 x f32，与 BowSpray 完全一致，渲染端同构）：
 *   offset  0 : pos   vec3<f32>  归一化 ECEF（地球半径 = 1）
 *   offset 12 : age   f32        秒
 *   offset 16 : vel   vec3<f32>  m/s（世界轴）
 *   offset 28 : life  f32        秒
 *   offset 32 : up    vec3<f32>  发射帧竖直轴（重力轴）
 *   offset 44 : size  f32        米
 *   offset 48 : h     f32        高出静水面的高度（米，< 0 落水回收）
 *   offset 52 : kind  f32        恒 0（预留；BowSpray 的 mist 位）
 *   offset 56 : alpha f32        基础不透明度
 */

export const BREAK_SPRAY_WORKGROUP = 64;
/** 每粒子 f32 数（见文件头状态布局）。 */
export const BREAK_SPRAY_FLOATS_PER_PARTICLE = 16;
/** SimParams：9 x vec4<f32> = 144B。 */
export const BREAK_SPRAY_SIM_PARAMS_BYTES = 144;
/** DrawParams：mat4x4<f32> + 2 x vec4<f32> = 96B。 */
export const BREAK_SPRAY_DRAW_PARAMS_BYTES = 96;

// ── Compute: ring 发射 + 海床/波面吸附 + 米制积分 ────────────────────────────

export const BREAK_SPRAY_COMPUTE_SHADER = /* wgsl */`
struct Spray {
  pos   : vec3<f32>,
  age   : f32,
  vel   : vec3<f32>,
  life  : f32,
  up    : vec3<f32>,
  size  : f32,
  h     : f32,
  kind  : f32,
  alpha : f32,
}

struct SimParams {
  band   : vec4<f32>,  // xyz bandCenter (normalised ECEF), w = dt (s)
  along  : vec4<f32>,  // xyz alongDir (切向单位矢量), w = EARTH_SCALE (m)
  up     : vec4<f32>,  // xyz up axis,            w = emitCount
  east   : vec4<f32>,  // xyz 海面 east 基 (每米归一化 ECEF 增量), w = ring cursor
  north  : vec4<f32>,  // xyz 海面 north 基,                      w = frameSeed
  oceanC : vec4<f32>,  // xyz 海面中心 (归一化 ECEF),             w = widthM (米)
  uvaff  : vec4<f32>,  // xy = seabed uvOffset, zw = uvScale（渲染 FS 同源仿射）
  bandp  : vec4<f32>,  // x = nearDepth (米), y = farDepth (米), z = intensity, w = raiseM
  misc   : vec4<f32>,  // x = gravity (m/s^2), y = dragTau (s), z = 预留, w = count
}

@group(0) @binding(0) var<storage, read_write> sprays : array<Spray>;
@group(0) @binding(1) var<uniform> params : SimParams;
@group(0) @binding(2) var seabedTex : texture_2d<f32>;          // r32float（unfilterable，只 textureLoad）
@group(0) @binding(3) var dispTex : texture_2d_array<f32>;      // 位移纹理（.y = 波高米）

// PCG hash — WGSL replacement for Math.random()
fn pcgHash(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand01(x: u32) -> f32 {
  return f32(pcgHash(x)) * (1.0 / 4294967296.0);
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let count = u32(params.misc.w);
  if (idx >= count) { return; }

  var p = sprays[idx];
  let dt = params.band.w;
  let up = params.up.xyz;

  // ── 发射：CPU 维护的环形窗口 [cursor, cursor+emit) mod count ──────────────
  let cursor = u32(params.east.w);
  let emit   = u32(params.up.w);
  let dist   = (idx + count - cursor) % count;

  if (p.age >= p.life) {
    if (dist >= emit) { return; }  // 死亡且不在发射窗口：驻留

    let seed = u32(params.north.w);
    let h0 = rand01(idx * 9781u  + seed);
    let h1 = rand01(idx * 6271u  + seed + 11u);
    let h2 = rand01(idx * 3253u  + seed + 23u);
    let h3 = rand01(idx * 7919u  + seed + 37u);
    let h4 = rand01(idx * 1543u  + seed + 53u);
    let h5 = rand01(idx * 40503u + seed + 71u);

    let R = params.along.w;
    let A = params.along.xyz;
    let s = (h0 - 0.5) * params.oceanC.w;  // 沿带横向偏移（米）

    // 发射点：bandCenter + alongDir·s（米→归一化）+ up·raise
    let posN = params.band.xyz + A * (s / R) + up * (params.bandp.w / R);

    // ── 海床/波面吸附：ECEF → 海面局部平面米 → seabed uv → textureLoad ────
    // east/north 为「每米归一化 ECEF 增量」（非单位长），米数须按 dot(e,e)
    // 最小二乘除回（OceanFFTWater.bindSeabed 同式，直接点积会差 R 倍）。
    let e = params.east.xyz;
    let n = params.north.xyz;
    let d = posN - params.oceanC.xyz;
    let planeX = dot(d, e) / dot(e, e);
    let planeZ = dot(d, n) / dot(n, n);
    let uv = params.uvaff.xy + params.uvaff.zw * vec2<f32>(planeX, planeZ);
    var sbH = -1000.0;   // 场外 = 极深：无吸附无门（照常喷）
    var waveH = 0.0;
    if (all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0))) {
      let dim = textureDimensions(seabedTex, 0);
      let tc = clamp(vec2<i32>(uv * vec2<f32>(dim)), vec2<i32>(0), vec2<i32>(dim) - vec2<i32>(1));
      sbH = textureLoad(seabedTex, tc, 0).r;
      let ddim = textureDimensions(dispTex);
      let dtc = clamp(vec2<i32>(uv * vec2<f32>(ddim)), vec2<i32>(0), vec2<i32>(ddim) - vec2<i32>(1));
      waveH = textureLoad(dispTex, dtc, 0, 0).y;  // 位移纹理 .y = 波高 Dy（米）
    }

    // 速度门：smoothstep(farDepth, nearDepth, depth) 概率薄化——浅水全量、
    // 深水归零；r 超过门限的槽位保持死亡（本帧不重生）
    let depth = waveH - sbH;
    let gate = smoothstep(params.bandp.y, params.bandp.x, depth);
    if (h1 > gate) { return; }

    // 初速：上抛为主 + 沿带切向随机（破碎白水向上向前翻卷）
    p.pos   = posN;
    p.vel   = up * (1.2 + 1.8 * h2) + A * ((h3 - 0.5) * 1.6);
    p.age   = 0.0;
    p.life  = mix(0.5, 1.1, h4);
    p.up    = up;
    p.size  = mix(0.04, 0.14, h5);
    p.h     = max(waveH, 0.0) + params.bandp.w;  // 出生在波面上方
    p.kind  = 0.0;
    p.alpha = 0.5;
  } else {
    // ── 积分：重力沿 -up（发射帧固定）+ 线性拖拽 + 离面高度跟踪 ─────────────
    let g = params.misc.x;
    p.vel = p.vel + (-p.up * g) * dt;
    p.vel = p.vel + (-p.vel / params.misc.y) * dt;
    p.h   = p.h + dot(p.vel, p.up) * dt;
    p.pos = p.pos + p.vel * (dt / params.along.w);   // m/s → 归一化单位/s
    p.age = p.age + dt;

    if (p.h < 0.0) {
      // 落回静水面以下：标记死亡，等待环形窗口回收
      p.age = p.life;
    }
  }

  sprays[idx] = p;
}
`;

// ── Render: premultiplied alpha 深度测试 billboard（BowSpray 同构，白水色）──

export const BREAK_SPRAY_RENDER_SHADER = /* wgsl */`
struct Spray {
  pos   : vec3<f32>,
  age   : f32,
  vel   : vec3<f32>,
  life  : f32,
  up    : vec3<f32>,
  size  : f32,
  h     : f32,
  kind  : f32,
  alpha : f32,
}

struct DrawParams {
  viewProj : mat4x4<f32>,
  misc     : vec4<f32>,  // x = clip scale X (1/(tanHalfFov*aspect)),
                          // y = clip scale Y (1/tanHalfFov),
                          // z = EARTH_SCALE (m / world unit)
}

@group(0) @binding(0) var<uniform> drawParams : DrawParams;
@group(0) @binding(1) var<storage, read> sprays : array<Spray>;

struct VertexInput {
  @location(0) corner : vec2<f32>,  // quad corner in [-1, 1]
}

struct VertexOutput {
  @builtin(position) clipPos : vec4<f32>,
  @location(0) uv            : vec2<f32>,
  @location(1) @interpolate(flat) color : vec4<f32>,
}

@vertex
fn vs_main(
  in: VertexInput,
  @builtin(instance_index) instance: u32,
) -> VertexOutput {
  let p = sprays[instance];
  let base = drawParams.viewProj * vec4<f32>(p.pos, 1.0);

  // 死亡槽位（含未发射驻留）：零尺寸零 alpha，退化不出片元。
  var alpha = 0.0;
  var sizeNorm = 0.0;
  var dir = vec2<f32>(1.0, 0.0);
  var stretch = 1.0;

  if (p.age < p.life) {
    let t = clamp(p.age / max(p.life, 1e-4), 0.0, 1.0);
    sizeNorm = p.size / drawParams.misc.z;  // 米 → 归一化世界单位
    sizeNorm = sizeNorm * (0.8 + 1.2 * t);  // 随寿命膨胀（白水散开）
    alpha = p.alpha * (1.0 - t) * (1.0 - t);

    // 沿屏幕速度方向拉伸（运动模糊感）
    let velNorm = p.vel / drawParams.misc.z;
    let ahead = drawParams.viewProj * vec4<f32>(p.pos + velNorm * 0.05, 1.0);
    let dv = ahead.xy / max(ahead.w, 1e-5) - base.xy / max(base.w, 1e-5);
    let l = length(dv);
    if (l > 1e-6) { dir = dv / l; }
    stretch = 1.0 + min(length(p.vel) * 0.08, 2.0);
  }

  let perp = vec2<f32>(-dir.y, dir.x);
  let off  = (dir * (in.corner.x * stretch) + perp * in.corner.y)
           * drawParams.misc.xy * sizeNorm;

  var out: VertexOutput;
  // 透视正确的屏幕空间扩展：clip 偏移 = NDC 偏移 * w（ParticleSystem 同式）
  out.clipPos = vec4<f32>(base.xy + off * base.w, base.z, base.w);
  out.uv      = in.corner;
  out.color   = vec4<f32>(vec3<f32>(0.94, 0.97, 1.0), alpha);
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let d = length(in.uv);
  if (d > 1.0) { discard; }
  // edge0 > edge1 的反序 smoothstep 在 WGSL 规范属未定义行为；此处依赖
  // Dawn/Tint 的实测行为（BowSpray / SeabedHeightField 同模式注释背书），
  // 用于把圆盘边缘软化为 1→0 的衰减。
  let soft = smoothstep(1.0, 0.35, d);
  let a = in.color.a * soft;
  // premultiplied alpha：rgb 已乘 a，配合 blend src='one'
  return vec4<f32>(in.color.rgb * a, a);
}
`;
