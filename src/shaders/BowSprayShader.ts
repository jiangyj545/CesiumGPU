/**
 * Bow spray shaders — 船首浪喷雾（海洋 L4，tidewater BoatSpray 直译裁剪版）。
 *
 * ## 模拟空间（选型说明）
 * 粒子在**世界空间（归一化 ECEF）**模拟：pos 存归一化坐标，vel 以 m/s 记、
 * 位置增量除 EARTH_SCALE 转归一化单位，gravityDir = 发射帧的 -up（每粒子
 * 存一份 up）。选它而非"局部 ENU 米制平面 + VS 转回世界"的理由：
 * - 渲染 VS 无需持基，粒子云不随船转向/平移漂移（喷雾应留在船后）；
 * - 引擎端到端就是 f32（viewProj mat4 也是 f32），|pos|≈1 处 f32 ULP ≈ 0.4 m，
 *   局部米制模拟的额外精度在 VS 变换时同样被量化掉，不会更准；
 * - 死亡判定 h（沿 up 的高度标量）逐帧积分即可，无需存发射点。
 *
 * ## 寿命说明
 * droplet/mist 的实际寿命由弹道主导：h < 0（落回发射平面）即死亡回收，
 * 典型约 0.25–0.6 s；life 上限（droplet 0.5–1.3 s / mist 1.1–2.1 s）通常
 * 不触及——与 tidewater BoatSpray 的物理一致。
 *
 * 状态布局 — 64 B/粒子（16 x f32）：
 *   offset  0 : pos   vec3<f32>  归一化 ECEF（地球半径 = 1）
 *   offset 12 : age   f32        秒
 *   offset 16 : vel   vec3<f32>  m/s（世界轴）
 *   offset 28 : life  f32        秒
 *   offset 32 : up    vec3<f32>  发射帧竖直轴（重力轴）
 *   offset 44 : size  f32        米（发射时定，寿命内由 VS 随 age 整形）
 *   offset 48 : h     f32        高出发射平面的高度（米，< 0 落水回收）
 *   offset 52 : kind  f32        0 = droplet / 1 = mist
 *   offset 56 : alpha f32        基础不透明度（发射时定）
 */

export const BOW_SPRAY_WORKGROUP = 64;
/** 每粒子 f32 数（见文件头状态布局）。 */
export const BOW_SPRAY_FLOATS_PER_PARTICLE = 16;
/** SimParams：9 x vec4<f32>。 */
export const BOW_SPRAY_SIM_PARAMS_BYTES = 144;
/** DrawParams：mat4x4<f32> + 2 x vec4<f32>。 */
export const BOW_SPRAY_DRAW_PARAMS_BYTES = 96;

// ── Compute: ring 发射 + 米制积分 ─────────────────────────────────────────────

export const BOW_SPRAY_COMPUTE_SHADER = /* wgsl */`
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
  bow     : vec4<f32>,  // xyz bowPos (normalised ECEF), w = dt (s)
  basisR  : vec4<f32>,  // xyz right axis, w = EARTH_SCALE (m)
  basisU  : vec4<f32>,  // xyz up axis,    w = emitCount
  basisF  : vec4<f32>,  // xyz fwd axis,   w = ring cursor
  misc    : vec4<f32>,  // x = count, y = frameSeed, z = cornerOffset (m), w = mistRatio
  spawn   : vec4<f32>,  // x = speed (m/s), y = raise (m), z = dropletAlpha, w = mistAlpha
  droplet : vec4<f32>,  // x = lifeMin, y = lifeMax, z = sizeMin, w = sizeMax
  mist    : vec4<f32>,  // x = lifeMin, y = lifeMax, z = sizeMin, w = sizeMax
  misc2   : vec4<f32>,  // x = gravity (m/s^2), y = dragTau (s)
}

@group(0) @binding(0) var<storage, read_write> sprays  : array<Spray>;
@group(0) @binding(1) var<uniform> params : SimParams;

// mist 相对 droplet 的裁剪常数（tidewater BoatSpray 同值量级）
const MIST_GRAVITY_SCALE = 0.3;  // 重力 x0.3
const MIST_VEL_SCALE     = 0.5;  // 初速 x0.5

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
  let count = u32(params.misc.x);
  if (idx >= count) { return; }

  var p = sprays[idx];
  let dt = params.bow.w;
  let up = params.basisU.xyz;

  // ── 发射：CPU 维护的环形窗口 [cursor, cursor+emit) mod count ──────────────
  // 窗口内且已死亡的槽位被回收重生成；仍存活的槽位跳过（不硬覆盖，防弹跳消失）。
  let cursor = u32(params.basisF.w);
  let emit   = u32(params.basisU.w);
  let dist   = (idx + count - cursor) % count;

  if (p.age >= p.life) {
    if (dist >= emit) { return; }  // 死亡且不在发射窗口：驻留

    let seed = u32(params.misc.y);
    let h0 = rand01(idx * 9781u  + seed);
    let h1 = rand01(idx * 6271u  + seed + 11u);
    let h2 = rand01(idx * 3253u  + seed + 23u);
    let h3 = rand01(idx * 7919u  + seed + 37u);
    let h4 = rand01(idx * 1543u  + seed + 53u);
    let h5 = rand01(idx * 40503u + seed + 71u);
    let h6 = rand01(idx * 2246822519u + seed + 97u);

    let isMist = h0 < params.misc.w;               // 10% mist / 90% droplet
    let side   = select(-1.0, 1.0, h4 < 0.5);      // 船首左/右舷角二选一
    let jitter = 0.7 + 0.6 * h5;                   // 初速整体 ±30%

    // 局部米制初速：x = 舷外法向, y = 竖直, z = 航向切向（tidewater 公式直译）
    var v = vec3<f32>(
      side * (0.8 * h1 + 0.4),   // 外法向 sideDir*(±1)*(0.8r+0.4) m/s
      0.55 * h2,                 // 竖直 0.55r m/s（沿局部 up）
      params.spawn.x * 0.6,      // 切向 headingDir*speed*0.6 m/s
    ) * jitter;
    if (isMist) { v = v * MIST_VEL_SCALE; }

    let R = params.basisR.xyz;
    let F = params.basisF.xyz;
    let earth = params.basisR.w;
    let corner = params.misc.z;
    let raise  = params.spawn.y;

    let dr = params.droplet;
    let mi = params.mist;
    let life = select(mix(dr.x, dr.y, h3), mix(mi.x, mi.y, h3), isMist);
    let size = select(mix(dr.z, dr.w, h6), mix(mi.z, mi.w, h6), isMist);

    p.pos   = params.bow.xyz + (R * (side * corner) + up * raise) / earth;
    p.vel   = R * v.x + up * v.y + F * v.z;
    p.age   = 0.0;
    p.life  = life;
    p.up    = up;
    p.size  = size;
    p.h     = raise;  // 发射点抬升即离面高度初值
    p.kind  = select(0.0, 1.0, isMist);
    p.alpha = select(params.spawn.z, params.spawn.w, isMist);
  } else {
    // ── 积分：重力沿 -up（发射帧固定）+ 线性拖拽 + 离面高度跟踪 ─────────────
    let g = params.misc2.x * select(1.0, MIST_GRAVITY_SCALE, p.kind > 0.5);
    p.vel = p.vel + (-p.up * g) * dt;
    p.vel = p.vel + (-p.vel / params.misc2.y) * dt;   // tau 线性拖拽
    p.h   = p.h + dot(p.vel, p.up) * dt;
    p.pos = p.pos + p.vel * (dt / params.basisR.w);   // m/s → 归一化单位/s
    p.age = p.age + dt;

    if (p.h < 0.0) {
      // 落回发射平面（水面）以下：标记死亡，等待环形窗口回收
      p.age = p.life;
    }
  }

  sprays[idx] = p;
}
`;

// ── Render: premultiplied alpha 深度测试 billboard ───────────────────────────

export const BOW_SPRAY_RENDER_SHADER = /* wgsl */`
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

const MIST_TINT = vec3<f32>(0.92, 0.97, 1.0);  // mist 略带青

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
  var col = vec3<f32>(1.0);
  var dir = vec2<f32>(1.0, 0.0);   // 运动拉伸方向（屏幕空间）
  var stretch = 1.0;

  if (p.age < p.life) {
    let t = clamp(p.age / max(p.life, 1e-4), 0.0, 1.0);
    let mist = p.kind > 0.5;
    col = select(vec3<f32>(1.0), MIST_TINT, mist);
    sizeNorm = p.size / drawParams.misc.z;  // 米 → 归一化世界单位

    if (mist) {
      // 雾滴：随寿命膨胀、淡入淡出（极低 alpha 的体积感）
      sizeNorm = sizeNorm * (0.7 + 0.9 * t);
      alpha = p.alpha * min(t * 4.0, 1.0) * (1.0 - t);
    } else {
      // 水滴：微缩 + 寿命平方衰减 + 沿屏幕速度方向拉伸（运动模糊感）
      sizeNorm = sizeNorm * (1.0 - 0.3 * t);
      alpha = p.alpha * (1.0 - t) * (1.0 - t);

      let velNorm = p.vel / drawParams.misc.z;
      let ahead = drawParams.viewProj * vec4<f32>(p.pos + velNorm * 0.05, 1.0);
      let dv = ahead.xy / max(ahead.w, 1e-5) - base.xy / max(base.w, 1e-5);
      let l = length(dv);
      if (l > 1e-6) { dir = dv / l; }
      stretch = 1.0 + min(length(p.vel) * 0.06, 2.5);
    }
  }

  let perp = vec2<f32>(-dir.y, dir.x);
  let off  = (dir * (in.corner.x * stretch) + perp * in.corner.y)
           * drawParams.misc.xy * sizeNorm;

  var out: VertexOutput;
  // 透视正确的屏幕空间扩展：clip 偏移 = NDC 偏移 * w（ParticleSystem 同式）
  out.clipPos = vec4<f32>(base.xy + off * base.w, base.z, base.w);
  out.uv      = in.corner;
  out.color   = vec4<f32>(col, alpha);
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let d = length(in.uv);
  if (d > 1.0) { discard; }
  // edge0 > edge1 的反序 smoothstep 在 WGSL 规范属未定义行为；此处依赖
  // Dawn/Tint 的实测行为（CPU 侧 SeabedHeightField.smoothstep 的同模式注释背书），
  // 用于把圆盘边缘软化为 1→0 的衰减。
  let soft = smoothstep(1.0, 0.35, d);
  let a = in.color.a * soft;
  // premultiplied alpha：rgb 已乘 a，配合 blend src='one'
  return vec4<f32>(in.color.rgb * a, a);
}
`;
