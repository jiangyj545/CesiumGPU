/**
 * WakeField compute shader（船迹波场——泄漏波动方程 + Havelock 移动压力）。
 *
 * 网格语义（继承 tidewater WakeSim 的环形寻址设计）：
 * - N×N 纹理即整个跟随窗口，世界格 I 存放在 texel `I mod N`（窗口滚动零拷贝）。
 * - `wfLoad` 做 tidewater `wakeLoadState` 同款世界格一致性检查：本帧 texel 的
 *   世界格若落在上一帧窗口之外，则视为无历史（新水，返回 0）。滚动量小于窗
 *   宽时 texel 保留原有波形（相对世界整体随窗口平移一格，0.x m 级，不可见）；
 *   越过整窗的传送则全部清零重来。
 * - lap / 邻域读取均在 mod 域做 wrap（`& WF_MASK`）。
 *
 * 物理步进（单 kernel，symplectic Euler）：
 *   w' = w + dt·c²·∇²(h + P)      // P = 船体压力头（米），平衡解 h ≈ −P
 *   h' = h + dt·w'
 *   h', w' 同乘 exp(−dt·(damping + sponge²·6))   // 物理阻尼 + 边缘海绵
 * 泡沫（G 通道）：|h| 超阈值 + 船尾 wash 注入，指数衰减生命周期。
 *
 * 输出纹理 rgba16float：R=波高 h（米） G=泡沫(0-1) B=竖直流速 w（调试） A=压力头 P（调试）。
 *
 * 稳定性：显式格式要求 CFL = c·dt/cell < 1/√2；CPU 侧将 c clamp 到
 * 0.45·cell/dt（dt 上限 1/30 s），本 shader 不再做二次钳制。
 */

/** workgroup 边长（8×8=64 线程，gridN 须为 8 的倍数）。 */
export const WAKE_FIELD_WORKGROUP = 8;

/** uniform WakeParams 的字节数（5 × vec4<f32>）。 */
export const WAKE_FIELD_PARAMS_BYTES = 80;

/**
 * 构建 WakeField compute shader 源码（gridN 编译期注入，tidewater 同款模板手法）。
 * @param gridN 网格边长（8 的倍数，64..512）
 */
export function buildWakeFieldShader(gridN: number): string {
  return /* wgsl */`
const WF_N: u32 = ${gridN}u;
const WF_MASK: i32 = ${gridN - 1}i;
const WF_SPONGE: f32 = 10.0; // 海绵层宽度（格）

struct WakeParams {
  origin  : vec4<f32>,  // xy = 窗口最小角（世界格，本帧） zw = 上一帧
  boat    : vec4<f32>,  // xy = 船位（米，局部平面域） z = cos(heading) w = sin(heading)
  grid    : vec4<f32>,  // x = cellM, y = dt(s), z = resetFlag, w = 动压头 K·U²/2g (m)
  phys    : vec4<f32>,  // x = c² (m²/s²), y = damping (1/s), z = foamGain, w = foamThr (m)
  hull    : vec4<f32>,  // x = lengthM, y = beamM, z = speed (m/s), w = 涡粘 nu (m²/s)
}

@group(0) @binding(0) var<uniform> wp : WakeParams;
@group(0) @binding(1) var texPrev : texture_2d<f32>;                              // 上一帧场（ping）
@group(0) @binding(2) var texNext : texture_storage_2d<rgba16float, write>;       // 本帧场（pong）

// 世界格 index：环形寻址——texel c 在窗口（最小角 o）中的世界格
fn wfWorldCell(c: i32, o: i32) -> i32 {
  return o + ((c - o) & WF_MASK);
}

// tidewater wakeLoadState 同款：texel (c,r) 若本帧与上一帧持有同一世界格则读
// 历史状态，否则返回静水（新水）。reset 时全部视为新水。
fn wfLoad(c: i32, r: i32) -> vec4<f32> {
  if (wp.grid.z > 0.5) { return vec4<f32>(0.0); }
  let I = wfWorldCell(c, i32(wp.origin.x));
  let J = wfWorldCell(r, i32(wp.origin.y));
  let keep = I >= i32(wp.origin.z) && I < i32(wp.origin.z) + i32(WF_N)
          && J >= i32(wp.origin.w) && J < i32(wp.origin.w) + i32(WF_N);
  return select(vec4<f32>(0.0), textureLoad(texPrev, vec2<u32>(u32(c), u32(r)), 0u), keep);
}

// 船体压力头 P（米）——Havelock 移动压力思想的 L2 近似：
// 船头正高斯（入水冲击）+ 船尾负高斯（船尾板后空穴），幅值 = K·U²/2g 动压头。
// 真实型线浸没积分（tidewater hullTex）不在本简化范围内。
fn wfPressure(x: f32, z: f32) -> f32 {
  let dyn = wp.grid.w;
  if (dyn <= 0.001) { return 0.0; }
  let rel = vec2<f32>(x, z) - wp.boat.xy;
  let bx = rel.x * wp.boat.z - rel.y * wp.boat.w; // 横向（船体系）
  let bz = rel.x * wp.boat.w + rel.y * wp.boat.z; // 纵向（船体系）
  let L = max(wp.hull.x, 2.0);
  let B = max(wp.hull.y, 0.8);
  let sigA = max(0.20 * L, 1.2);          // 船头纵向 σ
  let sigS = max(0.30 * L, 2.0);          // 船尾空穴纵向 σ（更长）
  let sigC = max(0.55 * B, 0.5);          // 横向 σ
  let db = (bz - 0.42 * L) / sigA;
  let ds = (bz + 0.45 * L) / sigS;
  let bow   = exp(-db * db - (bx / sigC) * (bx / sigC));
  let stern = exp(-ds * ds - (bx / (sigC * 1.8)) * (bx / (sigC * 1.8)));
  return dyn * (bow - 0.9 * stern);
}

@compute @workgroup_size(${WAKE_FIELD_WORKGROUP}, ${WAKE_FIELD_WORKGROUP}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = i32(gid.x);
  let r = i32(gid.y);
  let cell = wp.grid.x;
  let dt = wp.grid.y;
  let cell2 = cell * cell;

  let I = wfWorldCell(c, i32(wp.origin.x));
  let J = wfWorldCell(r, i32(wp.origin.y));
  let p = vec2<f32>(f32(I) + 0.5, f32(J) + 0.5) * cell;

  let s  = wfLoad(c, r);
  let sL = wfLoad((c - 1) & WF_MASK, r);
  let sR = wfLoad((c + 1) & WF_MASK, r);
  let sD = wfLoad(c, (r - 1) & WF_MASK);
  let sU = wfLoad(c, (r + 1) & WF_MASK);

  // 压力经 ∇²P 进入竖直加速度（不直接写高度）；5 点差分同时把源模糊到网格
  // 尺度，避免移动压力激励网格噪声（tidewater 同款理由）
  let P = wfPressure(p.x, p.y);
  let lapP = (wfPressure(p.x + cell, p.y) + wfPressure(p.x - cell, p.y)
            + wfPressure(p.x, p.y + cell) + wfPressure(p.x, p.y - cell) - 4.0 * P) / cell2;
  let lapH = (sL.x + sR.x + sD.x + sU.x - 4.0 * s.x) / cell2;

  // 注意符号：lap 的平面波本征为 −q（负定），波动方程 h_tt = c²∇²h 要求
  // w_t = +c²∇²(h+P)（tidewater 的 −g·L[h] 中 L 是正符号谱算子，实空间即 +c²∇²；
  // 若写成减号则成反扩散，网格模式指数爆炸）。
  // 涡粘 nu·∇²w（通量形式守质量）：~k² 阻尼，压网格尺度噪声，对米级波浪无感
  // （tidewater visc 0.006 + 湍流场的单场简化）
  let lapW = (sL.y + sR.y + sD.y + sU.y - 4.0 * s.y) / cell2;
  var w1 = s.y + dt * wp.phys.x * (lapH + lapP) + dt * wp.hull.w * lapW;
  var h1 = s.x + dt * w1;

  // 重置帧：静水 + 船体压头直接写入平衡位移 h ≈ −P，无启动瞬态（tidewater 同款）
  if (wp.grid.z > 0.5) {
    h1 = -P;
    w1 = 0.0;
  }

  // 物理阻尼 + 边缘海绵（窗口边缘 ~10 格渐变吸收，防滚动/外漏反射）
  let edge = min(min(f32(c), f32(WF_N) - 1.0 - f32(c)),
                 min(f32(r), f32(WF_N) - 1.0 - f32(r)));
  let sig = 1.0 - smoothstep(0.0, WF_SPONGE, edge);
  let k = exp(-dt * (wp.phys.y + sig * sig * 6.0));
  h1 = h1 * k;
  w1 = w1 * k;

  // 泡沫（G 通道）：|h| 超阈值的碎浪累积 + 船尾 prop wash 注入，指数衰减
  // （Jacobian 泡沫同款 accumulate-decay 生命周期）
  let thr = max(wp.phys.w, 0.001);
  let crest = smoothstep(thr, thr * 2.0, abs(h1));
  let rel = p - wp.boat.xy;
  let bx = rel.x * wp.boat.z - rel.y * wp.boat.w;
  let bz = rel.x * wp.boat.w + rel.y * wp.boat.z;
  let behind = max(-bz, 0.0);
  let washW = max(0.8 * max(wp.hull.y, 0.8), 1.2) + behind * 0.08;
  let wash = exp(-behind / 20.0) * exp(-(bx / washW) * (bx / washW))
           * smoothstep(0.3, 1.5, wp.hull.z);
  let gen = wp.phys.z * (crest * 2.0 + wash * 1.2);
  let foam = clamp(s.z * exp(-dt * 0.6) + gen * dt * 1.5, 0.0, 1.0);

  textureStore(texNext, vec2<u32>(gid.x, gid.y), vec4<f32>(h1, foam, w1, P));
}
`;
}
