/**
 * OceanFFTShader - FFT 海洋水体 WGSL（算法忠实移植 tidewater OceanFFT.js）。
 *
 * 多级联 Tessendorf 谱海洋：JONSWAP×方向谱（Horvath 参数化）生成 h0(k)，
 * 双波浪系统（local 风浪 + swell 涌浪）叠加；每帧 2 个主 compute dispatch：
 *   1. row  - 时间演化 h0→h(k,t)，打包 4 个复数场并按行做 256 点 radix-2 IFFT
 *   2. column - 按列 IFFT + (-1)^(x+y) 符号修正 + Jacobian 泡沫累积，
 *      写出 displacement(Dx,Dy,Dz,foam) / derivatives(dDy/dx,dDy/dz,λDxx,λDzz)
 *      两张 rgba16float 2d-array（N×N×4 层，9 级 mip 由 mipA/mipB kernel 生成）。
 *
 * 打包复数场（两个实场共用一次复 IFFT）：
 *   c0 = Dx + i Dz        c1 = Dy + i dDx/dz
 *   c2 = dDy/dx + i dDy/dz  c3 = dDx/dx + i dDz/dz
 *
 * CesiumGPU 适配：海面是局部平面（归一化 ECEF 球），顶点携带
 * [ecefXYZ, planeXZ_m]；eastDir/northDir/upDir 为中心点「每米归一化
 * ECEF 增量」基向量（含极轴压扁，已在 CPU 侧除以米数），VS 直接
 * ecef + east·Dx + up·Dy + north·Dz。λ(choppiness) 在 column kernel
 * 写纹理时已乘（Dx/Dz 与 Dxx/Dzz），顶点/片元不再重复乘。
 */

/** FFT 边长 N（tidewater FFT_SIZE）。 */
export const OCEAN_FFT_SIZE = 256;
const N = OCEAN_FFT_SIZE;
/** log2(N)：IFFT 级数。 */
const LOG2N = 8;
/** 半宽：workgroup 单线程处理 2 元素。 */
const HALF = N / 2;
/** 级联数。 */
const CASCADES = 4;

// ── 共享：OceanParams uniform（288B）────────────────────────────────────────
// 64+64+32+32 = @192 起标量段；seed 后按 vec3 对齐 16 补齐，
// eastDir/northDir/upDir 落在 @240/@256/@272，struct 总 288B。

const OCEAN_PARAMS_WGSL = /* wgsl */`
struct OceanParams {
  sizes      : array<vec4<f32>, 4>,  // @0   每级联 x = 平铺尺寸（米）
  cuts       : array<vec4<f32>, 4>,  // @64  每级联 x/y = 波数带通 low/high (rad/m)
  sysA       : array<vec4<f32>, 2>,  // @128 [scale, 风向角(rad), spreadBlend, swell]
  sysB       : array<vec4<f32>, 2>,  // @160 [alpha, peakOmega, gamma, shortWavesFade]
  choppiness : f32,                  // @192 λ：水平位移缩放（写纹理时已乘）
  foamBias   : f32,                  // @196 J 低于此值开始生泡沫
  foamGain   : f32,                  // @200 泡沫生成增益
  foamDecay  : f32,                  // @204 泡沫指数衰减率
  foamAdd    : f32,                  // @208 泡沫生成速率
  time       : f32,                  // @212 累计效果时间（秒）
  depth      : f32,                  // @216 水深（米，色散关系用）
  dt         : f32,                  // @220 帧间隔（秒，泡沫累积）
  seed       : u32,                  // @224 谱随机种子
  eastDir    : vec3<f32>,            // @240 每米归一化 ECEF 增量（含极轴压扁）
  northDir   : vec3<f32>,            // @256
  upDir      : vec3<f32>,            // @272
}
`;

// ── compute 公共：常量 + PCG hash + 复数 helper（tidewater FFT_COMMON）──────

const OCEAN_FFT_COMMON_WGSL = /* wgsl */`${OCEAN_PARAMS_WGSL}
@group(0) @binding(0) var<uniform> ocean : OceanParams;

const OCEAN_CASCADES : i32 = ${CASCADES};
const OCEAN_FFT_HALF : u32 = ${HALF}u;
const FFT_G : f32 = 9.81;
const PI : f32 = 3.141592653589793;
const TWO_PI : f32 = 6.283185307179586;

fn sat( x : f32 ) -> f32 { return clamp( x, 0.0, 1.0 ); }

fn fftBitReverse8( v : u32 ) -> u32 {
  var r = v;
  r = ( ( r & 0x55u ) << 1u ) | ( ( r >> 1u ) & 0x55u );
  r = ( ( r & 0x33u ) << 2u ) | ( ( r >> 2u ) & 0x33u );
  r = ( ( r & 0x0Fu ) << 4u ) | ( ( r >> 4u ) & 0x0Fu );
  return r;
}

// 两个打包复数 (v.xy, v.zw) 各乘标量复数 w
fn fftCmul2( v : vec4<f32>, w : vec2<f32> ) -> vec4<f32> {
  return vec4<f32>(
    v.x * w.x - v.y * w.y, v.x * w.y + v.y * w.x,
    v.z * w.x - v.w * w.y, v.z * w.y + v.w * w.x
  );
}

// PCG hash
fn fftPcg( v : u32 ) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ( ( state >> ( ( state >> 28u ) + 4u ) ) ^ state ) * 277803737u;
  return ( word >> 22u ) ^ word;
}
fn fftToUnit( h : u32 ) -> f32 { return f32( h >> 8u ) * ( 1.0 / 16777216.0 ) + ( 0.5 / 16777216.0 ); }
`;

// ── 谱 helpers（tidewater SPECTRUM：JONSWAP + TMA 修正 + 方向谱）────────────

const OCEAN_SPECTRUM_WGSL = /* wgsl */`
fn dispersion( k : f32 ) -> f32 { return sqrt( k * FFT_G * tanh( min( k * ocean.depth, 20.0 ) ) ); }

fn dispersionDerivative( k : f32 ) -> f32 {
  let kd = min( k * ocean.depth, 20.0 );
  let th = tanh( kd );
  let ch = cosh( kd );
  return FFT_G * ( ocean.depth * k / ( ch * ch ) + th ) / dispersion( k ) * 0.5;
}

// TMA 浅水修正
fn tmaCorrection( omega : f32 ) -> f32 {
  let omegaH = omega * sqrt( ocean.depth / FFT_G );
  let a = omegaH * omegaH * 0.5;
  let b = 1.0 - pow( 2.0 - omegaH, 2.0 ) * 0.5;
  return select( select( 1.0, b, omegaH < 2.0 ), a, omegaH <= 1.0 );
}

fn jonswap( omega : f32, sysA : vec4<f32>, sysB : vec4<f32> ) -> f32 {
  let alpha = sysB.x;
  let peakOmega = sysB.y;
  let gamma = sysB.z;
  let sigma = select( 0.09, 0.07, omega <= peakOmega );
  let d = omega - peakOmega;
  let r = exp( - d * d / ( sigma * sigma * peakOmega * peakOmega * 2.0 ) );
  let inv = 1.0 / omega;
  let po = peakOmega * inv;
  return sysA.x * tmaCorrection( omega ) * alpha * ( FFT_G * FFT_G )
    * pow( inv, 5.0 )
    * exp( pow( po, 4.0 ) * -1.25 )
    * pow( abs( gamma ), r );
}

fn normalisationFactor( s : f32 ) -> f32 {
  let s2 = s * s;
  let s3 = s2 * s;
  let s4 = s3 * s;
  let lo = s4 * -0.000564 + s3 * 0.00776 - s2 * 0.044 + s * 0.192 + 0.163;
  let hi = s4 * -4.80e-08 + s3 * 1.07e-05 - s2 * 9.53e-04 + s * 5.90e-02 + 3.93e-01;
  return select( hi, lo, s < 5.0 );
}

fn directionSpectrum( theta : f32, omega : f32, sysA : vec4<f32>, sysB : vec4<f32> ) -> f32 {
  let peakOmega = sysB.y;
  let ratio = omega / peakOmega;
  let spreadPower = select( pow( abs( ratio ), 5.0 ) * 6.97, pow( abs( ratio ), -2.5 ) * 9.77, omega > peakOmega );
  let s = spreadPower + tanh( min( ratio, 20.0 ) ) * 16.0 * sysA.w * sysA.w;
  let dTheta = theta - sysA.y;
  let cos2s = normalisationFactor( s ) * pow( abs( cos( dTheta * 0.5 ) ), s * 2.0 );
  let cosT = cos( dTheta );
  let base = cosT * cosT * ( 2.0 / PI ) * select( 0.0, 1.0, cosT > 0.0 );
  return mix( base, cos2s, sysA.z );
}

fn shortWavesFade( k : f32, sysB : vec4<f32> ) -> f32 { return exp( - sysB.w * sysB.w * k * k ); }
`;

// ── IFFT 8 级蝶形（radix-2，shared memory，编译期展开）──────────────────────

function fftStages(): string {
  let stages = '';
  for (let s = 0; s < LOG2N; s++) {
    const half = 1 << s;
    stages += /* wgsl */`
  {
    let pos = t & ${half - 1}u;
    let i = ( ( t >> ${s}u ) << ${s + 1}u ) | pos;
    let j = i + ${half}u;
    let ang = f32( pos ) * ${(Math.PI / half).toFixed(12)};
    let w = vec2<f32>( cos( ang ), sin( ang ) );
    let i2 = i * 2u;
    let j2 = j * 2u;
    let a0 = fftShared[ i2 ];
    let a1 = fftShared[ i2 + 1u ];
    let b0 = fftCmul2( fftShared[ j2 ], w );
    let b1 = fftCmul2( fftShared[ j2 + 1u ], w );
    fftShared[ i2 ] = a0 + b0;
    fftShared[ i2 + 1u ] = a1 + b1;
    fftShared[ j2 ] = a0 - b0;
    fftShared[ j2 + 1u ] = a1 - b1;
    workgroupBarrier();
  }`;
  }
  return stages;
}

// ── kernel 1：initSpectrum（一次性：h0 与波数/色散查表）─────────────────────

export const OCEAN_FFT_INIT_SPECTRUM_SHADER = /* wgsl */`${OCEAN_FFT_COMMON_WGSL}${OCEAN_SPECTRUM_WGSL}
@group(0) @binding(1) var<storage, read_write> h0 : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> waveData : array<vec4<f32>>;

@compute @workgroup_size(16, 16, 1)
fn main( @builtin( global_invocation_id ) gid : vec3<u32> ) {
  let x = i32( gid.x );
  let y = i32( gid.y );
  let c = i32( gid.z );
  let idx = c * ${N * N} + y * ${N} + x;

  let L = ocean.sizes[ c ].x;
  let dk = TWO_PI / L;
  let kx = f32( x - ${HALF} ) * dk;
  let kz = f32( y - ${HALF} ) * dk;
  let kLen = length( vec2<f32>( kx, kz ) );

  var outH = vec4<f32>( 0.0 );
  var outW = vec4<f32>( kx, kz, 0.0, 0.0 );

  if ( kLen >= ocean.cuts[ c ].x && kLen <= ocean.cuts[ c ].y ) {
    let omega = dispersion( kLen );
    let dOmega = dispersionDerivative( kLen );
    let theta = atan2( kz, kx );

    let sa0 = ocean.sysA[ 0 ];
    let sb0 = ocean.sysB[ 0 ];
    let sa1 = ocean.sysA[ 1 ];
    let sb1 = ocean.sysB[ 1 ];

    let S0 = jonswap( omega, sa0, sb0 ) * directionSpectrum( theta, omega, sa0, sb0 ) * shortWavesFade( kLen, sb0 );
    let S1 = jonswap( omega, sa1, sb1 ) * directionSpectrum( theta, omega, sa1, sb1 ) * shortWavesFade( kLen, sb1 );
    let S = max( S0 + S1, 0.0 );

    // E|h0|^2 = S(k) dk^2 / 2，使 var(height) = sum S(k) dk^2（h 含 ±k 两项）
    let amp = sqrt( S * abs( dOmega ) / kLen * dk * dk ) * 0.5;

    // Box-Muller 高斯随机对
    let seed = u32( idx ) * 4u + ocean.seed * 7919u;
    let u1 = fftToUnit( fftPcg( seed ) );
    let u2 = fftToUnit( fftPcg( seed + 1u ) );
    let r = sqrt( log( u1 ) * -2.0 );
    let g0 = r * cos( u2 * TWO_PI );
    let g1 = r * sin( u2 * TWO_PI );

    outH = vec4<f32>( g0 * amp, g1 * amp, 0.0, 0.0 );
    outW = vec4<f32>( kx, kz, 1.0 / kLen, omega );
  }

  h0[ idx ] = outH;
  waveData[ idx ] = outW;
}
`;

// ── kernel 2：conjugate（tmp = (h0(k), conj(h0(-k))) 打包）──────────────────

export const OCEAN_FFT_CONJUGATE_SHADER = /* wgsl */`${OCEAN_FFT_COMMON_WGSL}
@group(0) @binding(1) var<storage, read_write> h0 : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> tmp : array<vec4<f32>>;

@compute @workgroup_size(16, 16, 1)
fn main( @builtin( global_invocation_id ) gid : vec3<u32> ) {
  let x = i32( gid.x );
  let y = i32( gid.y );
  let c = i32( gid.z );
  let base = c * ${N * N};
  let idx = base + y * ${N} + x;
  let xm = ( ${N} - x ) % ${N};
  let ym = ( ${N} - y ) % ${N};
  let idxm = base + ym * ${N} + xm;
  let hm = h0[ idxm ].xy;
  let cur = h0[ idx ].xy;
  tmp[ idx ] = vec4<f32>( cur.x, cur.y, hm.x, - hm.y );
}
`;

// ── kernel 3：copyH0（h0 ← tmp 打包结果，foam 清零）─────────────────────────

export const OCEAN_FFT_COPY_H0_SHADER = /* wgsl */`${OCEAN_FFT_COMMON_WGSL}
@group(0) @binding(1) var<storage, read_write> h0 : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> tmp : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> foam : array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main( @builtin( global_invocation_id ) gid : vec3<u32> ) {
  let idx = gid.z * ${N * N}u + gid.y * ${N}u + gid.x;
  h0[ idx ] = tmp[ idx ];
  foam[ idx ] = 0.0;
}
`;

// ── kernel 4：fftRow（时间演化 + 4 打包场 + 按行 IFFT）──────────────────────

export const OCEAN_FFT_ROW_SHADER = /* wgsl */`${OCEAN_FFT_COMMON_WGSL}
var<workgroup> fftShared : array<vec4<f32>, ${N * 2}>;

@group(0) @binding(1) var<storage, read_write> h0 : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> waveData : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> tmp : array<vec4<f32>>;

@compute @workgroup_size(${HALF}, 1, 1)
fn main( @builtin( local_invocation_id ) lid : vec3<u32>, @builtin( workgroup_id ) wid : vec3<u32> ) {
  let t = lid.x;
  let row = wid.x;
  let c = wid.y;
  let base = c * ${N * N}u + row * ${N}u;
  let time = ocean.time;

  for ( var e = 0u; e < 2u; e++ ) {
    let x = t + e * OCEAN_FFT_HALF;
    let idx = base + x;
    let w = waveData[ idx ];
    let hv = h0[ idx ];
    let ph = w.w * time;
    let cs = cos( ph );
    let sn = sin( ph );

    // h = h0 * e^{i w t} + conj(h0(-k)) * e^{-i w t}
    let hr = hv.x * cs - hv.y * sn + hv.z * cs + hv.w * sn;
    let hi = hv.x * sn + hv.y * cs - hv.z * sn + hv.w * cs;

    let kx = w.x;
    let kz = w.y;
    let ik = w.z;
    let fx = kx * ik;
    let fz = kz * ik;

    // Dx_hat = i kx/k h, Dz_hat = i kz/k h -> c0 = Dx + i Dz
    let c0 = vec2<f32>( - ( fx * hi + fz * hr ), fx * hr - fz * hi );
    // c1 = Dy + i dDx/dz, dDx/dz_hat = -kx kz / k h
    let q = - ( kx * kz * ik );
    let c1 = vec2<f32>( hr - q * hi, hi + q * hr );
    // c2 = dDy/dx + i dDy/dz
    let c2 = vec2<f32>( - ( kx * hi + kz * hr ), kx * hr - kz * hi );
    // c3 = dDx/dx + i dDz/dz
    let a = - ( kx * kx * ik );
    let b = - ( kz * kz * ik );
    let c3 = vec2<f32>( a * hr - b * hi, a * hi + b * hr );

    let r = fftBitReverse8( x ) * 2u;
    fftShared[ r ] = vec4<f32>( c0, c1 );
    fftShared[ r + 1u ] = vec4<f32>( c2, c3 );
  }

  workgroupBarrier();
${fftStages()}

  for ( var e = 0u; e < 2u; e++ ) {
    let x = t + e * OCEAN_FFT_HALF;
    let o = ( base + x ) * 2u;
    tmp[ o ] = fftShared[ x * 2u ];
    tmp[ o + 1u ] = fftShared[ x * 2u + 1u ];
  }
}
`;

// ── kernel 5：fftColumn（按列 IFFT + 符号修正 + Jacobian 泡沫 → 纹理）────────

export const OCEAN_FFT_COLUMN_SHADER = /* wgsl */`${OCEAN_FFT_COMMON_WGSL}
var<workgroup> fftShared : array<vec4<f32>, ${N * 2}>;

@group(0) @binding(1) var<storage, read_write> tmp : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> foam : array<f32>;
@group(0) @binding(3) var<storage, read_write> mipSrc : array<vec4<f32>>;
@group(0) @binding(4) var dispOut : texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(5) var derivOut : texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(${HALF}, 1, 1)
fn main( @builtin( local_invocation_id ) lid : vec3<u32>, @builtin( workgroup_id ) wid : vec3<u32> ) {
  let t = lid.x;
  let col = wid.x;
  let c = wid.y;
  let base = c * ${N * N}u;

  for ( var e = 0u; e < 2u; e++ ) {
    let y = t + e * OCEAN_FFT_HALF;
    let idx = base + y * ${N}u + col;
    let r = fftBitReverse8( y ) * 2u;
    fftShared[ r ] = tmp[ idx * 2u ];
    fftShared[ r + 1u ] = tmp[ idx * 2u + 1u ];
  }

  workgroupBarrier();
${fftStages()}

  let lambda = ocean.choppiness;
  for ( var e = 0u; e < 2u; e++ ) {
    let y = t + e * OCEAN_FFT_HALF;
    let idx = base + y * ${N}u + col;
    let sgn = select( -1.0, 1.0, ( ( col + y ) & 1u ) == 0u );
    let A = fftShared[ y * 2u ] * sgn;
    let B = fftShared[ y * 2u + 1u ] * sgn;

    let Dx = A.x;
    let Dz = A.y;
    let Dy = A.z;
    let Dxz = A.w;
    let Dyx = B.x;
    let Dyz = B.y;
    let Dxx = B.z;
    let Dzz = B.w;

    let jxx = lambda * Dxx + 1.0;
    let jzz = lambda * Dzz + 1.0;
    let jxz = lambda * Dxz;
    let J = jxx * jzz - jxz * jxz;

    // 持久泡沫：J < bias（波面压缩）处生成，之后指数衰减留下白色尾迹
    let prev = foam[ idx ];
    let gen = sat( ( ocean.foamBias - J ) * ocean.foamGain );
    let f = prev * exp( - ocean.foamDecay * ocean.dt ) + gen * ocean.foamAdd * ocean.dt;
    let fNew = clamp( max( f, gen * 0.5 ), 0.0, 1.5 );
    foam[ idx ] = fNew;

    let uv = vec2<u32>( col, y );
    let vDisp = vec4<f32>( lambda * Dx, Dy, lambda * Dz, fNew );
    let vDeriv = vec4<f32>( Dyx, Dyz, lambda * Dxx, lambda * Dzz );
    textureStore( dispOut, uv, i32( c ), vDisp );
    textureStore( derivOut, uv, i32( c ), vDeriv );
    mipSrc[ idx * 2u ] = vDisp;
    mipSrc[ idx * 2u + 1u ] = vDeriv;
  }
}
`;

// ── mip 链（compute 归约代替渲染 pass；A: level 1-5，B: level 6-8）──────────
// A：16×16 线程 / 32×32 texel 块经 workgroup memory 归约；
// B：每层一个 8×8 workgroup 接力 mipMid。

function mipReduce(from: string, to: string | null, width: number, lvl: number, t: number): string {
  const w2 = width * 2;
  let dst = '';
  if (to) dst = `${to}[ ly * ${width}u + lx ] = v;`;
  else if (lvl === 5) dst = `mipMid[ ( c * 64u + gy * 8u + gx ) * 2u + ${t}u ] = v;`;
  return /* wgsl */`
  if ( lx < ${width}u && ly < ${width}u ) {
    let i = ly * ${2 * w2}u + lx * 2u;
    let v = ( ${from}[ i ] + ${from}[ i + 1u ] + ${from}[ i + ${w2}u ] + ${from}[ i + ${w2 + 1}u ] ) * 0.25;
    textureStore( out${lvl}, vec2u( gx * ${width}u + lx, gy * ${width}u + ly ), i32( c ), v );
    ${dst}
  }`;
}

function mipAShader(t: number): string {
  return /* wgsl */`${OCEAN_FFT_COMMON_WGSL}
var<workgroup> s1 : array<vec4<f32>, 256>;
var<workgroup> s2 : array<vec4<f32>, 64>;
var<workgroup> s3 : array<vec4<f32>, 16>;
var<workgroup> s4 : array<vec4<f32>, 4>;

@group(0) @binding(1) var<storage, read_write> mipSrc : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> mipMid : array<vec4<f32>>;
@group(0) @binding(3) var out1 : texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var out2 : texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(5) var out3 : texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(6) var out4 : texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(7) var out5 : texture_storage_2d_array<rgba16float, write>;

fn src( c : u32, x : u32, y : u32 ) -> vec4<f32> {
  return mipSrc[ ( c * ${N * N}u + y * ${N}u + x ) * 2u + ${t}u ];
}

@compute @workgroup_size(16, 16, 1)
fn main( @builtin( local_invocation_id ) lid : vec3<u32>, @builtin( workgroup_id ) wid : vec3<u32> ) {
  let lx = lid.x;
  let ly = lid.y;
  let gx = wid.x;
  let gy = wid.y;
  let c = wid.z;
  let x1 = gx * 16u + lx;
  let y1 = gy * 16u + ly;
  let x0 = x1 * 2u;
  let y0 = y1 * 2u;
  let v1 = ( src( c, x0, y0 ) + src( c, x0 + 1u, y0 ) + src( c, x0, y0 + 1u ) + src( c, x0 + 1u, y0 + 1u ) ) * 0.25;
  textureStore( out1, vec2u( x1, y1 ), i32( c ), v1 );
  s1[ ly * 16u + lx ] = v1;
  workgroupBarrier();
${mipReduce('s1', 's2', 8, 2, t)}
  workgroupBarrier();
${mipReduce('s2', 's3', 4, 3, t)}
  workgroupBarrier();
${mipReduce('s3', 's4', 2, 4, t)}
  workgroupBarrier();
${mipReduce('s4', null, 1, 5, t)}
}
`;
}

function mipBShader(t: number): string {
  return /* wgsl */`${OCEAN_FFT_COMMON_WGSL}
var<workgroup> s5 : array<vec4<f32>, 64>;
var<workgroup> s6 : array<vec4<f32>, 16>;
var<workgroup> s7 : array<vec4<f32>, 4>;

@group(0) @binding(1) var<storage, read_write> mipMid : array<vec4<f32>>;
@group(0) @binding(2) var out6 : texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var out7 : texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var out8 : texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main( @builtin( local_invocation_id ) lid : vec3<u32>, @builtin( workgroup_id ) wid : vec3<u32> ) {
  let lx = lid.x;
  let ly = lid.y;
  let c = wid.z;
  let gx = 0u;
  let gy = 0u;
  s5[ ly * 8u + lx ] = mipMid[ ( c * 64u + ly * 8u + lx ) * 2u + ${t}u ];
  workgroupBarrier();
${mipReduce('s5', 's6', 4, 6, t)}
  workgroupBarrier();
${mipReduce('s6', 's7', 2, 7, t)}
  workgroupBarrier();
${mipReduce('s7', null, 1, 8, t)}
}
`;
}

/** mipA 两变体：t=0 位移纹理 / t=1 导数纹理（mipSrc 交错排布 [disp, deriv]）。 */
export const OCEAN_FFT_MIP_A_DISP_SHADER = mipAShader(0);
export const OCEAN_FFT_MIP_A_DERIV_SHADER = mipAShader(1);
/** mipB 两变体（读 mipMid 接力 level 6-8）。 */
export const OCEAN_FFT_MIP_B_DISP_SHADER = mipBShader(0);
export const OCEAN_FFT_MIP_B_DERIV_SHADER = mipBShader(1);

// ── 渲染 shader（VS 位移采样 / FS 法线・菲涅尔・高光・泡沫・雾）─────────────

export const OCEAN_FFT_RENDER_SHADER = /* wgsl */`${OCEAN_PARAMS_WGSL}
// 256B 相机 UBO — 布局单一来源：src/renderer/CameraUniforms.ts
struct CameraUniforms {
  viewMatrix      : mat4x4<f32>,   // @0   (64B)
  projMatrix      : mat4x4<f32>,   // @64  (64B)
  viewProjMatrix  : mat4x4<f32>,   // @128 (64B)
  cameraPosition  : vec4<f32>,     // @192 (16B)
  viewportSize    : vec2<f32>,     // @208 (8B)
  nearFar         : vec2<f32>,     // @224 (8B)
}

// 128B 渲染侧参数（B1 flags/船迹 + L3 水下视角 + L4-A 岸线泡沫）
struct OceanRenderParams {
  sunDir      : vec4<f32>,  // 指向太阳（归一化，归一化 ECEF 空间）
  deepColor   : vec4<f32>,  // 深水基色
  skyColor    : vec4<f32>,  // 天空/雾色
  flags       : vec4<f32>,  // x 泡沫图案开 / y 船迹开 / z 折射开 / w 折射强度
  wakeUv      : vec4<f32>,  // 船迹 uv = planeXZ*zw + xy（窗口米 → [0,1)）
  underwater  : vec4<f32>,  // L3 x 水下视角开（从下看水面，其余保留）
  shoreParams : vec4<f32>,  // L4-A x 岸线泡沫开 / y shoreStrength（zw 保留）
  seabedUv    : vec4<f32>,  // L4-A seabed uv = planeXZ*zw + xy（海面局部米 → [0,1]）
}

@group(0) @binding(0) var<uniform> camera : CameraUniforms;
@group(0) @binding(1) var<uniform> ocean : OceanParams;
@group(0) @binding(2) var dispTex : texture_2d_array<f32>;
@group(0) @binding(3) var derivTex : texture_2d_array<f32>;
@group(0) @binding(4) var smp : sampler;
@group(0) @binding(5) var<uniform> renderParams : OceanRenderParams;
@group(0) @binding(6) var foamTex : texture_2d<f32>;        // A3 泡沫图案（未绑定 1×1 占位）
@group(0) @binding(7) var wakeTex : texture_2d<f32>;        // A4 船迹波场（VS 采样高度）
@group(0) @binding(8) var sceneHistTex : texture_2d<f32>;   // 场景颜色历史（折射）
@group(0) @binding(9) var clampSmp : sampler;               // 非平铺纹理用 clamp 采样

// ── A1 CSSM 级联阴影接收（L3：组 0 内 10-12，布局同 LightingPass 组 2）──────
// ShadowUniforms 布局（256B，单一来源 ShadowMap._rxData 写入端）：
//   cascadeMatrix @0（3×光源 view-proj）/ splitEnds @192 / texelSizes @208 /
//   params @224（x=enabled y=depthBias z=normalOffsetScale w=mapSize）。
// 未注入 scene.shadow 时 CPU 侧绑定占位资源（全零 UBO → params.x=0 短路恒 1）。
struct ShadowUniforms {
  cascadeMatrix : array<mat4x4<f32>, 3>,
  splitEnds     : vec4<f32>,
  texelSizes    : vec4<f32>,
  params        : vec4<f32>,
}

@group(0) @binding(10) var<uniform> shadow     : ShadowUniforms;
@group(0) @binding(11) var           shadowTex : texture_depth_2d_array;
@group(0) @binding(12) var           shadowCmp : sampler_comparison;

// L4-A 岸线泡沫：海床高度场（r32float，R = 海拔米，海平面 = 0）。不可过滤
// 格式 → FS 用 textureLoad 取最近 texel；未绑定海床时 CPU 侧挂 1×1 全 -1000
// （极深 = 无泡沫）占位且 shoreParams.x = 0 不采样。
@group(0) @binding(13) var seabedTex : texture_2d<f32>;

// 按视距（归一化单位）选级联；超出覆盖范围返回 -1（无阴影）。
fn cssmSelectCascade(viewDist: f32) -> i32 {
  if (viewDist < shadow.splitEnds.x) { return 0; }
  if (viewDist < shadow.splitEnds.y) { return 1; }
  if (viewDist < shadow.splitEnds.z) { return 2; }
  return -1;
}

// PCF 3×3 comparison 采样（与 LightingShader 注入的 SHADOW_RECEIVER_MODULE
// 同款级联选择 + normal-offset + bias）。返回 1 = 全光照，0 = 全影。
fn cssmShadowFactor(worldPos: vec3<f32>, normal: vec3<f32>, viewDist: f32) -> f32 {
  if (shadow.params.x < 0.5) { return 1.0; }

  let ci = cssmSelectCascade(viewDist);
  if (ci < 0) { return 1.0; }

  let texelW = select(
    select(shadow.texelSizes.x, shadow.texelSizes.y, ci == 1),
    shadow.texelSizes.z,
    ci == 2);
  let offsetPos = worldPos + normal * (texelW * shadow.params.z);

  let lp   = shadow.cascadeMatrix[ci] * vec4<f32>(offsetPos, 1.0);
  let proj = lp.xyz / lp.w;

  if (proj.x < -1.0 || proj.x > 1.0 || proj.y < -1.0 || proj.y > 1.0 || proj.z < 0.0 || proj.z >= 1.0) {
    return 1.0;
  }

  let uv = vec2<f32>(proj.x * 0.5 + 0.5, 0.5 - 0.5 * proj.y);
  // 'ref' 是 WGSL 保留字 —— 比较参考深度命名 refDepth。
  let refDepth = proj.z - shadow.params.y;
  let texelUV  = 1.0 / shadow.params.w;
  var sum = 0.0;
  for (var dy: i32 = -1; dy <= 1; dy++) {
    for (var dx: i32 = -1; dx <= 1; dx++) {
      let tap = uv + vec2<f32>(f32(dx), f32(dy)) * texelUV;
      sum += textureSampleCompareLevel(shadowTex, shadowCmp, tap, ci, refDepth);
    }
  }
  return sum / 9.0;
}

const EARTH_SCALE : f32 = 6378137.0;
const OCEAN_CASCADES : i32 = ${CASCADES};
const FOG_DIST : f32 = 12000.0;
// 泡沫图案世界平铺周期（米）：中频级联量级，图案不糊也不高频闪烁
const FOAM_TILE_M : f32 = 28.0;
// 遮挡球半径²：必须低于海面最小模长²。海面在纬度 φ 处
// |p|² = cos²φ + sin²φ·polarRatio² ≥ polarRatio² ≈ 0.99331（极点最小），
// 旧值 0.998² = 0.996004 会在中高纬（|φ| ≳ 39°）把海面划进球内——相机
// 升到球外俯视时相机→海面射线必穿球 → 海面整体 discard 黑屏。取
// 0.9945² = 0.989030 < 0.99331，任意纬度海面均贴地于球外；远处超出
// 地平线的片元仍被正确遮挡。（FloodWater / DynamicWall 同款问题另线处理）
const GLOBE_OCCLUDE_R2 : f32 = 0.9945 * 0.9945;

fn hash2( p : vec2<f32> ) -> f32 {
  let q = fract( p * vec2<f32>( 0.3183099, 0.3678794 ) + vec2<f32>( 0.71, 0.113 ) );
  return fract( 43758.5453 * dot( q, vec2<f32>( 23.14, 71.37 ) ) );
}

fn vnoise( p : vec2<f32> ) -> f32 {
  let i = floor( p );
  let f = fract( p );
  let u = f * f * ( 3.0 - 2.0 * f );
  return mix(
    mix( hash2( i ), hash2( i + vec2<f32>( 1.0, 0.0 ) ), u.x ),
    mix( hash2( i + vec2<f32>( 0.0, 1.0 ) ), hash2( i + vec2<f32>( 1.0, 1.0 ) ), u.x ),
    u.y
  );
}

struct VSIn {
  @location(0) ecef : vec3<f32>,    // 归一化 ECEF 水面基准点（水位折入模长）
  @location(1) planeXZ : vec2<f32>, // 局部平面坐标（米：x 东 / z 北）
}

struct VSOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0) worldPos : vec3<f32>,
  @location(1) planeXZ : vec2<f32>,
  @location(2) screenUV : vec2<f32>,  // 折射采样场景颜色历史用（NDC → [0,1]）
  @location(3) waveH : f32,           // L4-A 波面高（米，前 3 级联 disp.y + 船迹），岸线水深用
}

@vertex
fn vs_main( input : VSIn ) -> VSOut {
  // 前 3 个大级联位移（最小 7.1m 级联留给片元法线/泡沫）；显式 level 0
  var disp = vec3<f32>( 0.0 );
  for ( var c = 0; c < OCEAN_CASCADES - 1; c++ ) {
    let d = textureSampleLevel( dispTex, smp, input.planeXZ / ocean.sizes[ c ].x, c, 0.0 );
    disp += d.xyz;
  }
  // 船迹波高（A4，米）：planeXZ → 跟随窗口 uv，R 通道波高直接入竖直位移。
  // textureSampleLevel 在分支内合法（仅 textureSample 受 uniform control flow 限制）
  if ( renderParams.flags.y > 0.5 ) {
    let wuv = input.planeXZ * renderParams.wakeUv.zw + renderParams.wakeUv.xy;
    let wh = textureSampleLevel( wakeTex, clampSmp, wuv, 0.0 ).r;
    disp.y += wh;
  }
  // eastDir/northDir/upDir 是「每米归一化 ECEF 增量」（CPU 侧已除米数），
  // 位移纹理单位为米 → 直接线性组合；λ 已在 column kernel 写纹理时乘过
  let dispVec = ocean.eastDir * disp.x + ocean.upDir * disp.y + ocean.northDir * disp.z;

  // 基准位置与位移分别变换后相加：1 米位移 ≈ 1.6e-7 归一化单位，
  // 与 |worldPos|~0.5 直接相加会掉进 f32 ulp——先各自过 viewProj 再相加
  let clipBase = camera.viewProjMatrix * vec4<f32>( input.ecef, 1.0 );
  let clipDisp = camera.viewProjMatrix * vec4<f32>( dispVec, 0.0 );
  let clip = clipBase + clipDisp;

  var out : VSOut;
  out.clipPos = clip;
  out.worldPos = input.ecef + dispVec;
  out.planeXZ = input.planeXZ;
  out.screenUV = ( clip.xy / max( clip.w, 1e-6 ) ) * 0.5 + 0.5;
  out.waveH = disp.y;
  return out;
}

@fragment
fn fs_main( input : VSOut ) -> @location(0) vec4<f32> {
  let camPos = camera.cameraPosition.xyz;

  // 地球遮挡（IPass 无深度附件，片元级射线-球求交，与 FloodWater 同款）
  let oc = input.worldPos - camPos;
  let qa = dot( oc, oc );
  let qb = 2.0 * dot( camPos, oc );
  let qc = dot( camPos, camPos ) - GLOBE_OCCLUDE_R2;
  let disc = qb * qb - 4.0 * qa * qc;
  if ( disc > 0.0 ) {
    let t = ( - qb - sqrt( disc ) ) / ( 2.0 * qa );
    if ( t > 1e-3 && t < 1.0 - 1e-3 ) {
      discard;
    }
  }

  // 4 级联导数求和（斜率）+ 泡沫累计（位移纹理 alpha 通道）
  var foam = 0.0;
  var slopeX = 0.0;
  var slopeZ = 0.0;
  var divg = 0.0;
  for ( var c = 0; c < OCEAN_CASCADES; c++ ) {
    let uv = input.planeXZ / ocean.sizes[ c ].x;
    foam += textureSampleLevel( dispTex, smp, uv, c, 0.0 ).w;
    let g = textureSampleLevel( derivTex, smp, uv, c, 0.0 );
    slopeX += g.x;
    slopeZ += g.y;
    divg += g.z + g.w; // λ(Dxx + Dzz)：水平压缩度 → 粗糙度调制
  }

  // 局部法线 → 归一化 ECEF
  let nLocal = normalize( vec3<f32>( - slopeX, 1.0, - slopeZ ) );
  let n = normalize( ocean.eastDir * nLocal.x + ocean.upDir * nLocal.y + ocean.northDir * nLocal.z );

  let toCam = camPos - input.worldPos;
  let distN = length( toCam );
  let distM = distN * EARTH_SCALE;
  let viewDir = toCam / max( distN, 1e-9 );

  // 水下视角（L3）：从下看水面 → 光照法线翻转（几何法线 n 仍朝上，供阴影采样）
  let uw = renderParams.underwater.x > 0.5;
  let nLit = select( n, -n, uw );

  // 菲涅尔：水上 Schlick（F0 ≈ 0.02）；水下反转近似全内反射（掠射亮带更宽更陡）
  let ndv = clamp( dot( nLit, viewDir ), 0.0, 1.0 );
  let fresnel = select(
    0.02 + 0.98 * pow( 1.0 - ndv, 5.0 ),
    0.06 + 0.94 * pow( 1.0 - ndv, 1.6 ),
    uw );

  // 太阳高光：高指数锐主瓣 + 宽 glitter 次瓣；波面压缩处略增粗糙；水下弱化
  let sunDir = normalize( renderParams.sunDir.xyz );
  let hv = normalize( viewDir + sunDir );
  let ndh = max( dot( nLit, hv ), 0.0 );
  let rough = clamp( 1.0 + divg * 0.35, 0.6, 1.6 );
  var spec = ( pow( ndh, 500.0 / rough ) * 2.2 + pow( ndh, 48.0 / rough ) * 0.30 ) * ( 0.25 + fresnel );
  spec = select( spec, spec * 0.35, uw );

  // CSSM 阴影因子（L3）：太阳直射贡献（太阳高光 + 天光反射）同乘，软边船影；
  // 未绑定 scene.shadow / 已关闭时 params.x=0 短路恒 1（零回归）；水下不采样。
  let shadowF = select( cssmShadowFactor( input.worldPos, n, distN ), 1.0, uw );

  // 船迹泡沫（A4 G 通道）叠加进 foam 累计值
  var wakeFoam = 0.0;
  if ( renderParams.flags.y > 0.5 ) {
    let wuv = input.planeXZ * renderParams.wakeUv.zw + renderParams.wakeUv.xy;
    wakeFoam = textureSampleLevel( wakeTex, clampSmp, wuv, 0.0 ).g;
  }

  // 泡沫遮罩：A3 图案阈值化（绑定图案且开关开启）或原 hash 噪声调制（零回归回退）
  let fN = vnoise( input.planeXZ * 1.7 ) * 0.6 + vnoise( input.planeXZ * 5.3 ) * 0.4;

  // L4-A 岸线泡沫：seabed 高度场（R = 海拔米）+ 波面高 varying → 瞬时水深。
  // flag 关闭时三分量保持初值，下游公式逐项退化回原路径
  // （covTotal = foam*1 + 0 + wakeFoam、edgeFade = 1 → 零回归）。
  var jacW = 1.0;     // 近岸 Jacobian 泡沫减权（surf=1 打 5 折，tidewater:421 语义）
  var shoreCov = 0.0; // 破碎带泡沫增量
  var edgeFade = 1.0; // 浅水 alpha 收边（水膜结束在岸线，露出干沙给海床画）
  if ( renderParams.shoreParams.x > 0.5 ) {
    let suv = input.planeXZ * renderParams.seabedUv.zw + renderParams.seabedUv.xy;
    var seabedH = -1000.0; // 场外 = 极深 → 无泡沫不收边
    if ( all( suv >= vec2<f32>( 0.0 ) ) && all( suv <= vec2<f32>( 1.0 ) ) ) {
      let dim = textureDimensions( seabedTex, 0 );
      let tc = clamp( vec2<i32>( suv * vec2<f32>( dim ) ), vec2<i32>( 0 ), vec2<i32>( dim ) - vec2<i32>( 1 ) );
      seabedH = textureLoad( seabedTex, tc, 0 ).r;
    }
    let depth = input.waveH - seabedH;        // 瞬时水深（米）
    let surf = smoothstep( 7.0, 3.0, depth ); // 破碎带：≤3m 全量、≥7m 归零（tidewater SurfFoam 同款语义）
    jacW = 1.0 - 0.5 * surf;
    shoreCov = surf * renderParams.shoreParams.y * ( 0.75 + 0.5 * fN );
    edgeFade = smoothstep( 0.0, 0.05, depth );
  }

  let covTotal = foam * jacW + shoreCov + wakeFoam;
  let coverage = clamp( covTotal, 0.0, 1.0 );

  var foamMask = clamp( foam * jacW * ( 0.45 + 1.1 * fN ), 0.0, 1.0 ) + shoreCov;
  foamMask = clamp( foamMask, 0.0, 1.0 );
  var fineGlow = 0.0;
  var albedoJit = 1.0;
  if ( renderParams.flags.x > 0.5 ) {
    // A3 阈值方案：coverage 驱动阈值 t，R 密度场 smoothstep 软边；
    // G 细泡亮度 0.3-0.5、B 斑驳反照率 ±10%、A 条纹 ±0.1 偏置
    let p = textureSampleLevel( foamTex, smp, input.planeXZ * ( 1.0 / FOAM_TILE_M ), 0.0 );
    let t = 1.0 - coverage * 0.85;
    foamMask = clamp( smoothstep( t - 0.04, t + 0.04, p.r ) + ( p.a - 0.5 ) * 0.2, 0.0, 1.0 );
    fineGlow = p.g * coverage;
    albedoJit = 1.0 + ( p.b - 0.5 ) * 0.2;
  }

  // 水色：深水 → 地平线浅色（距离混合）；水下切换透射青绿（透光增强）
  let shallow = vec3<f32>( 0.075, 0.19, 0.27 );
  let baseCol = mix( renderParams.deepColor.rgb, shallow, smoothstep( 150.0, 5000.0, distM ) );
  let uwWater = vec3<f32>( 0.10, 0.46, 0.42 );
  let waterTint = select( baseCol, uwWater, uw );
  var col = waterTint * ( 0.82 + 0.18 * clamp( 1.0 - divg * 0.5, 0.0, 1.0 ) ) * albedoJit
    + renderParams.skyColor.rgb * fresnel * 0.9 * shadowF
    + vec3<f32>( spec ) * shadowF;

  // 折射：法线水平分量扰动采样场景颜色历史（水体下方背景），
  // 混合权重 ∝ (1-菲涅尔)；history 缺省绑 1×1 零纹理 → 采样为黑但不走此分支；
  // 水下视角跳过（历史为水上画面，语义不成立）
  if ( renderParams.flags.z > 0.5 && !uw ) {
    let roff = vec2<f32>( nLocal.x, nLocal.z ) * renderParams.flags.w;
    let ruv = clamp( input.screenUV + roff, vec2<f32>( 0.001 ), vec2<f32>( 0.999 ) );
    let under = textureSampleLevel( sceneHistTex, clampSmp, ruv, 0.0 ).rgb;
    let underCol = under * vec3<f32>( 0.60, 0.85, 0.90 ); // 水色吸收
    let refrW = clamp( ( 1.0 - fresnel ) * 0.75, 0.0, 0.60 );
    col = mix( col, underCol + vec3<f32>( spec ), refrW );
  }

  var foamWhite = vec3<f32>( 0.90, 0.94, 0.97 ) * ( 0.75 + 0.25 * fN + fineGlow );
  foamWhite = select( foamWhite, foamWhite * vec3<f32>( 0.35, 0.60, 0.62 ), uw ); // 水下泡沫偏暗青
  col = mix( col, foamWhite * albedoJit, foamMask * 0.92 );

  // 距离雾：远处融入天空色（弱化网格边缘接缝）；水下远景融入水色
  let fogCol = select( renderParams.skyColor.rgb, uwWater, uw );
  let fog = 1.0 - exp( - distM / FOG_DIST );
  col = mix( col, fogCol, fog * 0.95 );

  return vec4<f32>( col, edgeFade );
}
`;

/** OceanParams uniform buffer 字节数（288B，见 struct 注释）。 */
export const OCEAN_PARAMS_UBO_BYTES = 288;
/** OceanRenderParams uniform buffer 字节数（128B：sunDir/deep/sky/flags/wakeUv/underwater/shoreParams/seabedUv）。 */
export const OCEAN_RENDER_PARAMS_UBO_BYTES = 128;
