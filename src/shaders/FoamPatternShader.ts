/**
 * FoamPatternShader - 可平铺程序泡沫纹理 compute WGSL（一次性生成，非每帧）。
 *
 * 忠实移植 tidewater 参考实现（FoamTexture.js 的 ComputeKernel）：
 * GPU 上一次性生成 rgba16float 可平铺泡沫纹理，四个通道语义：
 *   R: 泡沫密度场 —— fbm 密集筏 + 域扭曲撕边 + 三层周期化 worley F1 打孔
 *      （消费端按本地 foam coverage 阈值化，泡沫自然生长/撕裂/消散）
 *   G: 细泡细节 —— 两层高频 worley 小亮点（亮度/法线扰动源）
 *   B: 大尺度斑驳 —— 低频 fbm
 *   A: 条纹 streaks —— 中频 fbm（流动拉丝方向感）
 *
 * 可平铺性：worley / vnoise 的格点 ID 一律经 `fmod2`（GLSL mod 语义，
 * x - y*floor(x/y)，负坐标也正确回绕）周期化，uv 域整数平移不破坏平铺。
 * fbm 按参考实现做常数权重展开（不加循环，权重为编译期字面量）。
 *
 * seed 实现：以正整数 uv 域偏移注入随机性——对周期化噪声而言，整数偏移
 * 等价于格点相位重排（p 平移 K*cells 为整数格步，fmod2/fract 结果逐位不变、
 * 仅特征布局改变），因此任意 seed 都严格保持可平铺。
 */

/**
 * fbm(uv, base, oct) 常数权重展开 —— 与参考实现逐字一致：
 * 逐 octave 累加 `vnoise(uv, base*2^o) * a`（a 折半），再除以权重和归一化。
 * 返回可直接内联进 WGSL 的表达式字符串。
 */
function fbmExpr(uv: string, base: number, oct: number): string {
  let s = '';
  let a = 0.5;
  let n = 0;
  for (let o = 0; o < oct; o++) {
    s += `${s ? ' + ' : ''}vnoise(${uv}, ${(base * Math.pow(2, o)).toFixed(1)}) * ${a}`;
    n += a;
    a *= 0.5;
  }
  return `(( ${s} ) / ${n})`;
}

/**
 * 生成泡沫纹理 compute shader 源码。
 *
 * @param size  纹理边长（像素，2 的幂），uv 归一化分母
 * @param seedX seed 派生的正整数 uv.x 偏移（格点相位重排）
 * @param seedY seed 派生的正整数 uv.y 偏移
 */
export function createFoamPatternShaderCode(
  size: number,
  seedX: number,
  seedY: number,
): string {
  return /* wgsl */ `
// ── 输出：rgba16float write-only storage（仅 level 0；mip 链由 MipmapGenerator 补齐）──

@group(0) @binding(0) var foamOut : texture_storage_2d<rgba16float, write>;

// ── 周期化噪声基元 ────────────────────────────────────────────────────────────

fn hash2(p : vec2<f32>) -> vec2<f32> {
  return fract(sin(vec2<f32>(
    dot(p, vec2<f32>(127.1, 311.7)),
    dot(p, vec2<f32>(269.5, 183.3)),
  )) * 43758.5453);
}

// GLSL mod 语义（参考实现 TSL .mod）：x - y * floor(x / y)。负坐标正确回绕。
fn fmod2(x : vec2<f32>, y : f32) -> vec2<f32> {
  return x - y * floor(x / y);
}

// 周期化 worley F1：jittered cell——格点特征位置由 hash2(cellId) 在单元内抖动，
// 3x3 邻域搜索 + fmod2 回绕保证 uv 与 uv+1 逐位一致（可平铺）。
fn worley(uv : vec2<f32>, cells : f32) -> f32 {
  let p  = uv * cells;
  let ip = floor(p);
  let fp = fract(p);
  var f1 = 8.0;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let o    = vec2<f32>(f32(i), f32(j));
      let cell = fmod2(ip + o, cells);
      let h    = hash2(cell);
      let d    = length(o + h - fp);
      f1 = min(f1, d);
    }
  }
  return f1;
}

// 周期化 value noise（smoothstep 插值，格点值 fmod2 回绕）。
fn vnoise(uv : vec2<f32>, cells : f32) -> f32 {
  let p = uv * cells;
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - f * 2.0);
  let a = hash2(fmod2(i, cells)).x;
  let b = hash2(fmod2(i + vec2<f32>(1.0, 0.0), cells)).x;
  let c = hash2(fmod2(i + vec2<f32>(0.0, 1.0), cells)).x;
  let d = hash2(fmod2(i + vec2<f32>(1.0, 1.0), cells)).x;
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// ── 主 kernel ────────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let px = gid.xy;
  // 像素中心归一化到 0..1 uv；seed 整数偏移做格点相位重排（保持可平铺）。
  let uv = (vec2<f32>(px) + 0.5) / ${size}.0 + vec2<f32>(${seedX}.0, ${seedY}.0);

  // 域扭曲（有机流动形态）
  let w1  = (vec2<f32>(${fbmExpr('uv', 3, 4)}, ${fbmExpr('(uv + 0.43)', 3, 4)}) - 0.5) * 0.14;
  let wuv = uv + w1;

  // R 密度场：撕边密集筏
  let dens = ${fbmExpr('wuv', 4, 6)};

  // 各级孔洞（worley，半径被噪声抖动）：粗 7 格 / 中 19 格 / 细 47 格
  let holeA = smoothstep(0.28, 0.12, worley(wuv, 7.0) + (${fbmExpr('uv', 16, 3)} - 0.5) * 0.25);
  let holeB = smoothstep(0.30, 0.16, worley(wuv + 0.17, 19.0) + (${fbmExpr('uv', 32, 2)} - 0.5) * 0.3);
  let holeC = smoothstep(0.32, 0.18, worley(wuv + 0.61, 47.0));
  let holes = clamp(holeA * 0.9 + holeB * 0.7 + holeC * 0.45, 0.0, 1.0);

  // G 细泡：高频 worley 小亮点（140 / 260 格两级）
  let bub = smoothstep(0.24, 0.08, worley(uv + 0.33, 140.0)) * 0.8
          + smoothstep(0.2, 0.05, worley(uv + 0.71, 260.0)) * 0.5;

  // A 条纹：warp 加强的中频 fbm（流动拉丝）
  let suv    = vec2<f32>(uv.x * 1.0, uv.y * 1.0) + w1 * 2.0;
  let streak = ${fbmExpr('suv', 12, 4)};

  // 密度合成：筏体为主，孔洞撕开，细泡轻微增密
  let foam   = clamp(dens * 1.35 - holes * 0.55 + bub * 0.08, 0.0, 1.0);
  let mottle = ${fbmExpr('uv', 2, 3)};

  textureStore(foamOut, px, vec4<f32>(foam, clamp(bub, 0.0, 1.0), mottle, streak));
}
`;
}
