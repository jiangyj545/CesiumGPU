/**
 * SwashComputeShader - swash 冲刷带浅水 compute（tidewater 追赶 Track W）。
 *
 * 物理模型：**沿岸线法向的 1D Saint-Venant 浅水方程**（每 texel 一根沿本地
 * 岸法向的 1D 水柱，法向 = 海床梯度方向 ∇b，指向岸/高处）。存「水位 η（米
 * 相对静水）+ 法向流速 u」，以水深 d = η - b 守恒：
 *   ∂η/∂t = -∂(d·u)/∂x                    （质量守恒）
 *   ∂u/∂t = -g·∂η/∂x - cf·u·|u|/d         （动量：水面坡度 + Manning 简化摩擦）
 * （任务式的 -g·∂(h+b)/∂x 中 h 为水深：∂(h+b) = ∂η，本 shader 直存 η 故为 ∂η。）
 *
 * 数值格式（单 kernel 显式步进）：
 * - 质量通量用**局部 Lax-Friedrichs（Rusanov）数值通量**沿法向迎风：
 *   F = 0.5(dC·uC + dR·uR) - 0.5·a·(ηR - ηC)，a = max(√(g·dC), √(g·dR))；
 *   相比纯迎风耗散更低、相比中心差分 CFL ≤ 1 内无条件稳定，干湿前沿
 *   表现良好——Rusanov 是浅水干湿问题的教科书选择。
 * - 动量坡度项用中心差分 + **半隐式摩擦**（u/(1 + dt·cf·|u|/d)），摩擦项
 *   无条件稳定，大 cf 不炸。
 * - 横向（沿岸切向）4 格涡粘 ν·∇² 阻网格尺度噪声（法向耗散已由 Rusanov
 *   提供，ν 取小值 0.3 m²/s，对米级 swash 无感）。
 * - **干湿前沿**：水深 < dryDepth 置干（u = 0、η 钳到 ≥ b），界面两侧均干
 *   时通量置零（湿润化阈值防数值振荡）。
 * - **入射波边界**：深水带（b < forceDepth，默认 -4m）松弛到入射波形
 *   η_t = A·sin(ω·t + k·切向坐标) + 次波噪声（确定性 vnoise，幅比 0.35），
 *   波形沿 -∇b 自动向岸传播；松弛同时吸收外行反射波（兼海绵）。
 * - **窗口边缘海绵**（~6 格）：防贴边反射（WakeField 同款手法）。
 *
 * 稳定性：显式格式要求 CFL = c·dt/cell ≤ 1（c = √(g·dmax)）。CPU 侧
 * （SwashField）按海床最深水深算 dtMax 并拆多子步（≤ 4），本 shader 不做
 * 二次钳制。
 *
 * 海床采样：r32float 不可过滤 → textureLoad（unfilterable-float BGL，与
 * CausticsFloor binding 5 同款坑）；swash 窗口与 seabed 纹理同域同尺寸
 * （uv = p/sizeM + 0.5 恒等映射），故无需仿射 uniform。
 *
 * 输出 rgba16float：R=水位 η（米） G=法向流速 u（m/s） B=水深 d（米）
 * A=湿润度（0-1）。rgba16float 不能 read_write storage → ping-pong 双纹理
 * （WakeField 同款，绑定 bg[k]：读 k 写 k^1）。
 */

/** workgroup 边长（8×8=64 线程，gridN 须为 8 的倍数）。 */
export const SWASH_WORKGROUP = 8;

/** uniform SwashParams 的字节数（4 × vec4<f32>）。 */
export const SWASH_PARAMS_BYTES = 64;

/** uniform f32 索引（与 SwashField 的打包代码、测试断言对齐）。 */
export const SWASH_P_GRID = 0;  // cellM, sizeM, dt(s), time(s)
export const SWASH_P_PHYS = 4;  // g, frictionCf, dryDepth(m), reset
export const SWASH_P_WAVE = 8;  // amplitude(m), omega(rad/s), 次波幅比, 沿岸波数(rad/m)
export const SWASH_P_MISC = 12; // 松弛率(1/s), forcing 深度阈(m,负), 海绵宽(格), 涡粘(m²/s)

/**
 * 构建 SwashField compute shader 源码（gridN 编译期注入，WakeField 同款模板手法）。
 * @param gridN 网格边长（8 的倍数，64..512）
 */
export function buildSwashShader(gridN: number): string {
  return /* wgsl */`
const SW_N : u32 = ${gridN}u;

struct SwashParams {
  grid : vec4<f32>,  // x=cellM y=sizeM z=dt(s) w=time(s)
  phys : vec4<f32>,  // x=g y=frictionCf z=dryDepth(m) w=reset
  wave : vec4<f32>,  // x=amplitude(m) y=omega(rad/s) z=次波幅比 w=沿岸波数(rad/m)
  misc : vec4<f32>,  // x=松弛率(1/s) y=forcing深度阈(m,负) z=海绵宽(格) w=涡粘(m²/s)
}

@group(0) @binding(0) var<uniform> sp : SwashParams;
@group(0) @binding(1) var texPrev : texture_2d<f32>;                         // 上一帧场（ping）
@group(0) @binding(2) var texNext : texture_storage_2d<rgba16float, write>;  // 本帧场（pong）
// r32float 不可过滤格式，只允许 textureLoad（CausticsFloor binding 5 同款）
@group(0) @binding(3) var seabedTex : texture_2d<f32>;

// 确定性 hash 噪声（次波相位扰动用，与 CausticsFloor FS 同款）
fn hash21( p : vec2<f32> ) -> f32 {
  var q = fract( p * vec2<f32>( 0.1031, 0.1030 ) );
  q += dot( q, q.yx + vec2<f32>( 33.33, 33.33 ) );
  return fract( ( q.x + q.y ) * q.x );
}
fn vnoise( p : vec2<f32> ) -> f32 {
  let i = floor( p );
  let f = fract( p );
  let u = f * f * ( 3.0 - 2.0 * f );
  let a = hash21( i );
  let b = hash21( i + vec2<f32>( 1.0, 0.0 ) );
  let c = hash21( i + vec2<f32>( 0.0, 1.0 ) );
  let d = hash21( i + vec2<f32>( 1.0, 1.0 ) );
  return mix( mix( a, b, u.x ), mix( c, d, u.x ), u.y );
}

// 平面米（相对 seabed 中心）→ 海床海拔 b + 界内标记（x=b, y=inRange 0/1）。
// swash 窗口与 seabed 纹理同域同尺寸 → uv = p/sizeM + 0.5 恒等映射；界外/
// 未绑定回退极深（-1000，无摩擦无干湿，物理上等效深水）。
fn swSeabed( p : vec2<f32> ) -> vec2<f32> {
  let suv = p / sp.grid.y + vec2<f32>( 0.5 );
  if ( ! ( all( suv >= vec2<f32>( 0.0 ) ) && all( suv <= vec2<f32>( 1.0 ) ) ) ) {
    return vec2<f32>( -1000.0, 0.0 );
  }
  let dim = textureDimensions( seabedTex, 0 );
  let tc = clamp( vec2<i32>( suv * vec2<f32>( dim ) ), vec2<i32>( 0 ), vec2<i32>( i32( SW_N ) - 1 ) );
  return vec2<f32>( textureLoad( seabedTex, tc, 0 ).r, 1.0 );
}

// 平面米 → 场状态（texel 中心对齐 + 边缘钳位；中心在半格点上，±整格偏移
// 远离 floor 边界，f32 量化无风险）
fn swLoad( p : vec2<f32> ) -> vec4<f32> {
  let t = floor( p / sp.grid.x + vec2<f32>( f32( SW_N ) * 0.5 ) );
  let tc = clamp( vec2<i32>( t ), vec2<i32>( 0 ), vec2<i32>( i32( SW_N ) - 1 ) );
  return textureLoad( texPrev, vec2<u32>( tc ), 0u );
}

@compute @workgroup_size(${SWASH_WORKGROUP}, ${SWASH_WORKGROUP}, 1)
fn main( @builtin( global_invocation_id ) gid : vec3<u32> ) {
  let cell = sp.grid.x;
  let e = cell;
  let dt = sp.grid.z;
  let p = ( vec2<f32>( gid.xy ) + vec2<f32>( 0.5 ) ) * cell - vec2<f32>( 0.5 ) * sp.grid.y;

  // 重置帧：全场静水（forcing 带随松弛自然起波，无启动瞬态爆炸）
  if ( sp.phys.w > 0.5 ) {
    textureStore( texNext, gid.xy, vec4<f32>( 0.0 ) );
    return;
  }

  let sb = swSeabed( p );
  let b = sb.x;
  let inRange = sb.y > 0.5;

  // 岸法向 = 海床梯度方向（∇b 指向高处/岸）；±2 texel 中心差分（与
  // CausticsFloor FS 梯度同款，阻单 texel 噪声）；无梯度（平底/界外）回退 +z
  let bR = swSeabed( p + vec2<f32>( 2.0 * e, 0.0 ) ).x;
  let bL = swSeabed( p - vec2<f32>( 2.0 * e, 0.0 ) ).x;
  let bU = swSeabed( p + vec2<f32>( 0.0, 2.0 * e ) ).x;
  let bD = swSeabed( p - vec2<f32>( 0.0, 2.0 * e ) ).x;
  let grad = vec2<f32>( bR - bL, bU - bD ) * 0.25;
  var nrm = vec2<f32>( 0.0, 1.0 );
  if ( length( grad ) > 1e-5 ) {
    nrm = normalize( grad );
  }
  let tanDir = vec2<f32>( - nrm.y, nrm.x );   // 岸线切向（沿岸波相位坐标）

  // 邻域状态：沿法向 ±cell（Rusanov 通量）与切向 ±cell（涡粘）
  let sC = swLoad( p );
  let sR = swLoad( p + nrm * e );
  let sL = swLoad( p - nrm * e );
  let sX = swLoad( p + tanDir * e );
  let sY = swLoad( p - tanDir * e );

  let g = sp.phys.x;
  let dry = max( sp.phys.z, 1e-4 );
  let dC = max( sC.r - b, 0.0 );   // 本格水深（η - 海床）
  let dR = max( sR.r - b, 0.0 );
  let dL = max( sL.r - b, 0.0 );

  // 质量守恒：Rusanov 数值通量沿法向（a = 面两侧波速 max，CFL 内无条件稳定）
  let aR = max( sqrt( g * max( dC, dry ) ), sqrt( g * max( dR, dry ) ) );
  let aL = max( sqrt( g * max( dC, dry ) ), sqrt( g * max( dL, dry ) ) );
  var fR = 0.5 * ( dC * sC.g + dR * sR.g ) - 0.5 * aR * ( sR.r - sC.r );
  var fL = 0.5 * ( dL * sL.g + dC * sC.g ) - 0.5 * aL * ( sC.r - sL.r );
  // 干面：界面两侧均干（水深 < dryDepth）→ 通量置零（湿润化阈值防振荡）
  fR = select( fR, 0.0, max( dC, dR ) < dry );
  fL = select( fL, 0.0, max( dC, dL ) < dry );

  var eta1 = sC.r - dt * ( fR - fL ) / e;

  // 动量：-g·∂η/∂x（中心差分）+ 横向涡粘 + 半隐式 Manning 摩擦
  //（分母 ≥ dry：大 cf 无条件稳定，干格 d≈0 不除零）
  var u1 = sC.g - dt * g * ( sR.r - sL.r ) / ( 2.0 * e );
  u1 = u1 / ( 1.0 + dt * sp.phys.y * abs( u1 ) / max( dC, dry ) );
  let nu = sp.misc.w;
  eta1 = eta1 + dt * nu * ( sX.r + sY.r - 2.0 * sC.r ) / ( e * e );
  u1 = u1 + dt * nu * ( sX.g + sY.g - 2.0 * sC.g ) / ( e * e );

  // 入射波边界：深水带（b < 阈值，界内）松弛到入射波形——主波 A·sin(ωt +
  // k·切向坐标) + 次波噪声；松弛同时吸收外行反射波（辐射开边界近似）
  if ( inRange && b < sp.misc.y ) {
    let phi = dot( p, tanDir ) * sp.wave.w;
    let sec = sp.wave.z * sp.wave.x * ( vnoise( p * 0.02 + vec2<f32>( sp.grid.w * 0.13, -sp.grid.w * 0.11 ) ) * 2.0 - 1.0 );
    // targetEta 而非 target：target 是 WGSL 保留关键字，声明即解析错
    // （真机 "target is a reserved keyword"，2026-09-28 横幅捕获）。
    let targetEta = sp.wave.x * sin( sp.wave.y * sp.grid.w + phi ) + sec;
    let k = 1.0 - exp( - dt * sp.misc.x );
    eta1 = mix( eta1, targetEta, k );
    u1 = u1 * ( 1.0 - k * 0.5 );
  }

  // 窗口边缘海绵（~sp.misc.z 格渐变吸收，防贴边反射；WakeField 同款）
  let edge = min( min( f32( gid.x ), f32( SW_N ) - 1.0 - f32( gid.x ) ),
                  min( f32( gid.y ), f32( SW_N ) - 1.0 - f32( gid.y ) ) );
  let sig = 1.0 - smoothstep( 0.0, sp.misc.z, edge );
  let kd = exp( - dt * sig * sig * 6.0 );
  eta1 = eta1 * kd;
  u1 = u1 * kd;

  // 干湿前沿：水深 < dryDepth 置干（u = 0，η 钳到海床——水不能低于床面）
  if ( eta1 - b < dry ) {
    u1 = 0.0;
    eta1 = max( eta1, b );
  }

  let dOut = max( eta1 - b, 0.0 );
  textureStore( texNext, gid.xy,
    vec4<f32>( eta1, u1, dOut, smoothstep( dry, dry * 2.0, dOut ) ) );
}
`;
}
