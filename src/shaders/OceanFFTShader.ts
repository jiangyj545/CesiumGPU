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
import {
  ATMO_PARAMS_STRUCT_WGSL,
  SKY_PARAMS_STRUCT_WGSL,
  ATMO_TRANS_SAMPLE_WGSL,
  ATMO_SKY_REFLECTION_WGSL,
} from './AtmosphereSkyShader';
import { cloudShadowReceiverModule } from './CloudShader';
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
// sat 定义（与 OCEAN_FFT_COMMON_WGSL 同式但渲染段不拼 COMMON——Track P 雾
// 染色调用 sat 而定义缺失，真机解析期 unresolved call target，2026-09-28
// fn 守卫捕获；此处独立声明防重复定义）。
fn sat( x : f32 ) -> f32 { return clamp( x, 0.0, 1.0 ); }

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
  underwater  : vec4<f32>,  // x L3 水下视角开 / y L5 时间秒 / z L5 SSS 总强度 / w L5 featureFlags
  shoreParams : vec4<f32>,  // L4-A x 岸线泡沫开 / y shoreStrength / z 海面雾太阳染色强度（w 保留）
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
// 格式 → textureLoad 取最近 texel；未绑定海床时 CPU 侧挂 1×1 全 -1000
// （极深 = 无泡沫）占位且 shoreParams.x = 0 不采样。T4 起顶点阶段也读
// （破碎波波形变形）→ BGL visibility 须含 VERTEX。
@group(0) @binding(13) var seabedTex : texture_2d<f32>;

// ── L5 视觉升级（tidewater 观感追赶：逐像素大气反射/太阳染色/SSS/破碎波）──
// atmosphereParams/skyParams 布局单一来源 AtmosphereSkyShader.ts（各 64B）。
// 未接 AtmosphereSky 时 CPU 侧挂 fallback（1×1 黑 LUT ×2 + 全零 UBO ×2），
// featureFlags（underwater.w）对应位未置即短路回旧路径（零回归）。
//   bit0 天空反射 / bit1 SSS / bit2 破碎波 / bit3 太阳高光染色
${ATMO_PARAMS_STRUCT_WGSL}
${SKY_PARAMS_STRUCT_WGSL}
struct BreakParams {
  gain : vec4<f32>,  // x=enable / y=振幅缩放 / z=波峰前倾 / w=泡沫花边增益
  band : vec4<f32>,  // x=bandNear(m) / y=bandFar(m) / z=内卷水幕染色 / w=花边阈值(m)
  peel : vec4<f32>,  // T-P x=卷唇开关 / y=卷曲角(rad) / z=卷曲速度 / w=卷唇权重阈值
}

@group(0) @binding(14) var atmSkyViewLUT : texture_2d<f32>;
@group(0) @binding(15) var atmLinearClamp : sampler;
@group(0) @binding(16) var atmTransmittanceLUT : texture_2d<f32>;
@group(0) @binding(17) var<uniform> atmosphereParams : AtmosphereParams;
@group(0) @binding(18) var<uniform> skyParams : SkyParams;
@group(0) @binding(19) var<uniform> breakParams : BreakParams;

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
// 2026-10-09 破平铺：单层 repeat 采样在 35m 俯视下出现整片相同形状的
// 白斑簇（视觉可识别的周期重复，被读作"水下纹理"）。改 3 层异向旋转
// 域采样（不同倍频 + 域旋转 + 异向偏移，加权合成）打散重复形状。
fn rot2( v : vec2<f32>, a : f32 ) -> vec2<f32> {
  let c = cos( a );
  let s = sin( a );
  return vec2<f32>( v.x * c - v.y * s, v.x * s + v.y * c );
}
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
// 3 层加权泡沫图案采样：坐标先加 choppiness 水平位移（图案钉在变形后
// 波面上随浪移动，而非钉在网格平面）；2/3 层再做 vnoise 域扭曲（打破
// 各层自身周期性——旋转+倍频只能打散形状对齐，域扭曲才彻底去平铺）。
fn foamPattern( advXZ : vec2<f32> ) -> vec4<f32> {
  let k = 1.0 / FOAM_TILE_M;
  let w1 = advXZ + vec2<f32>( vnoise( advXZ * 0.05 ), vnoise( advXZ * 0.05 + vec2<f32>( 17.3, 9.1 ) ) ) * 5.0;
  let w2 = advXZ + vec2<f32>( vnoise( advXZ * 0.031 + vec2<f32>( 4.7, 8.2 ) ), vnoise( advXZ * 0.031 + vec2<f32>( 25.1, 3.3 ) ) ) * 7.0;
  let p1 = textureSampleLevel( foamTex, smp, rot2( advXZ, 0.757 ) * k, 0.0 );
  let p2 = textureSampleLevel( foamTex, smp, rot2( w1 + vec2<f32>( 13.7, 5.1 ), -0.532 ) * ( k * 0.61 ), 0.0 );
  let p3 = textureSampleLevel( foamTex, smp, rot2( w2 + vec2<f32>( -7.3, 11.9 ), 1.169 ) * ( k * 1.47 ), 0.0 );
  return p1 * 0.55 + p2 * 0.30 + p3 * 0.15;
}

// L5 大气采样函数（AtmosphereSkyShader 同源拼接：透射 LUT / skyView 反射）。
// 仅在片元阶段调用（textureSampleLevel 合法；顶点阶段不可用）。
${ATMO_TRANS_SAMPLE_WGSL}
${ATMO_SKY_REFLECTION_WGSL}

// ── Track C 云影接收（海面版：组 0 内 20-22，布局同 LightingPass 组 3）──────
// binding 20 = 影贴图（rgba8unorm，R = 云密度）/ 21 = CloudShadowParams UBO
//（96B，布局单一来源 CloudShader.ts）/ 22 = filtering 采样器。未绑定时 CPU
// 侧挂 1×1 黑占位 + 全零 UBO（intensity = 0 短路恒 1，零回归）。
${cloudShadowReceiverModule(0, 21, 20, 22)}

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
  @location(4) breakW : f32,          // T4 破碎权重（smoothstep(bandFar, bandNear, 水深)）
  @location(5) dispXZ : vec2<f32>,    // 破平铺：choppiness 水平位移（米），泡沫图案随波坐标
}

@vertex
fn vs_main( input : VSIn ) -> VSOut {
  // 前 3 个大级联位移（最小 7.1m 级联留给片元法线/泡沫）。
  // 掠射防混叠：texel ≈ extent/256（~10m），mip0 在远距欠采样出 moiré——
  // VS 无隐式微分，按相机距离手动选 mip（tidewater 同款手动 LOD fetch；
  // 位移基准用未变形 ecef 近似，位移 <3m 对百米级距离无感）。
  var disp = vec3<f32>( 0.0 );
  let vsDistM = length( input.ecef - camera.cameraPosition.xyz ) * 6378137.0;
  for ( var c = 0; c < OCEAN_CASCADES - 1; c++ ) {
    let texelM = ocean.sizes[ c ].x / 256.0;
    let lod = clamp( log2( max( vsDistM / ( texelM * 700.0 ), 1.0 ) ), 0.0, 8.0 );
    let d = textureSampleLevel( dispTex, smp, input.planeXZ / ocean.sizes[ c ].x, c, lod );
    disp += d.xyz;
  }
  // 船迹波高（A4，米）：planeXZ → 跟随窗口 uv，R 通道波高直接入竖直位移。
  // textureSampleLevel 在分支内合法（仅 textureSample 受 uniform control flow 限制）
  if ( renderParams.flags.y > 0.5 ) {
    let wuv = input.planeXZ * renderParams.wakeUv.zw + renderParams.wakeUv.xy;
    let wh = textureSampleLevel( wakeTex, clampSmp, wuv, 0.0 ).r;
    disp.y += wh;
  }
  // T4 深度感知破碎波：海床高（VS 只能 textureLoad——r32float 不可过滤且顶点
  // 阶段禁 textureSample）→ 瞬时水深 → 破碎权重；再做振幅调制与波峰前倾挤压。
  var breakW = 0.0;
  if ( breakParams.gain.x > 0.5 && renderParams.shoreParams.x > 0.5 ) {
    let buv = input.planeXZ * renderParams.seabedUv.zw + renderParams.seabedUv.xy;
    var sbH = -1000.0; // 场外 = 极深 → 破碎权重 0
    if ( all( buv >= vec2<f32>( 0.0 ) ) && all( buv <= vec2<f32>( 1.0 ) ) ) {
      let bdim = textureDimensions( seabedTex, 0 );
      let btc = clamp( vec2<i32>( buv * vec2<f32>( bdim ) ), vec2<i32>( 0 ), vec2<i32>( bdim ) - vec2<i32>( 1 ) );
      sbH = textureLoad( seabedTex, btc, 0 ).r;
    }
    breakW = smoothstep( breakParams.band.y, breakParams.band.x, disp.y - sbH );
    let g0 = textureSampleLevel( derivTex, smp, input.planeXZ / ocean.sizes[ 0 ].x, 0, 0.0 );
    disp.y *= mix( 1.0, breakParams.gain.y, breakW );                    // 振幅调制
    // 波峰前倾。不用 disp.xz -=（多分量 swizzle 复合赋值）：naga 系校验器
    // 拒绝（2026-09-30 用户实机 "no matching overload for operator -="，
    // Tint 恰好放行）——整向量重建，两校验器都合法。
    let leanXY = g0.xy * max( disp.y, 0.0 ) * breakParams.gain.z * breakW;
    disp = vec3<f32>( disp.x - leanXY.x, disp.y, disp.z - leanXY.y );
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
  out.dispXZ = disp.xz;
  out.breakW = breakW;
  return out;
}

@fragment
fn fs_main( input : VSOut ) -> @location(0) vec4<f32> {
  let camPos = camera.cameraPosition.xyz;

  // 掠射各向异性 tap 间距（dpdx/dpdy 只能 uniform control flow——必须排在
  // 下方 occlusion discard 之前求值；discard 后调用真机解析错，守卫同款）
  let stepMPix = max( length( dpdx( input.planeXZ ) ), length( dpdy( input.planeXZ ) ) );

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
  // 掠射各向异性近似（2026-09-29 用户截图彩色竖条根因）：quad 隐式 LOD 是
  // 各向同性的——掠射方向（视线的平面投影）纹理被极度拉伸时沿拉伸向欠采样，
  // 8-11px 周期彩条（B 通道 ±25 摆动）。4-tap 沿视线平面方向平均 = 手动
  // aniso（tidewater 同款手法）；tap 间距按该级联 texel 步长缩放。
  //（viewDir 在 fs 后段才声明——此处先从 camPos 自算视线方向）
  let toCamA = camPos - input.worldPos;
  let viewDirA = toCamA / max( length( toCamA ), 1e-9 );
  var viewXZn = vec2<f32>( dot( viewDirA, ocean.eastDir ), dot( viewDirA, ocean.northDir ) );
  viewXZn = viewXZn / max( length( viewXZn ), 1e-5 );
  for ( var c = 0; c < OCEAN_CASCADES; c++ ) {
    let uv = input.planeXZ / ocean.sizes[ c ].x;
    // tap 间距 = 屏幕导数（每像素世界米数，discard 前预计算）换算到该级联
    // uv——对准混叠频率（texel×N 固定间距远大于条纹世界周期 0.5-5m，无效）。
    let duv = viewXZn * ( stepMPix * 1.5 / ocean.sizes[ c ].x );
    let g0 = textureSample( derivTex, smp, uv, c );
    let ga = textureSample( derivTex, smp, uv + duv, c );
    let gb = textureSample( derivTex, smp, uv + duv * 2.0, c );
    let gc = textureSample( derivTex, smp, uv - duv, c );
    let g = ( g0 + ga + gb + gc ) * 0.25;
    let f0 = textureSample( dispTex, smp, uv, c ).w;
    let fa = textureSample( dispTex, smp, uv + duv * 1.5, c ).w;
    let fb = textureSample( dispTex, smp, uv - duv * 1.5, c ).w;
    foam += ( f0 + fa + fb ) / 3.0;
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
  // Round3 镜面抗混叠（水下仰视远区高光带「离散方形色块 + 可数色阶」
  // 根因，检查员原生裁剪实锤）：pow(ndh,500) 锐瓣在远场屏幕足迹
  //（stepMPix 米/像素）接近 glitter 波长（~0.3-2m）时欠采样——亮点塌成
  // 1-3px 硬边小方块、亮度走可数色阶。足迹加宽（标准 specular AA）：
  // 指数随足迹降（lobe 变宽摊到多像素）、锐瓣峰值按 1/widen 稀释（太阳
  // 能量摊到更宽 lobe，能量近似守恒）。stepMPix ≤ 0.1m（≈200m 内）
  // widen=1，近中场逐字节零回归；水上/水下同规则（远场 glitter 同源）。
  let widen = min( max( 1.0 + ( stepMPix - 0.025 ) * 8.0, 1.0 ), 12.0 );
  var spec = ( pow( ndh, 500.0 / ( rough * widen ) ) * ( 2.8 / sqrt( widen ) )
             + pow( ndh, 48.0 / ( rough * sqrt( widen ) ) ) * 0.38 ) * ( 0.25 + fresnel );
  spec = select( spec, spec * 0.35, uw );

  // CSSM 阴影因子（L3）：太阳直射贡献（太阳高光 + 天光反射）同乘，软边船影；
  // 未绑定 scene.shadow / 已关闭时 params.x=0 短路恒 1（零回归）；水下不采样。
  let shadowF = select( cssmShadowFactor( input.worldPos, n, distN ), 1.0, uw );

  // Track C 云影因子（海面 T6 接线）：云挡太阳 → 天光反射与太阳高光 glitter
  // 同乘衰减。未绑定 / intensity<=0 时短路恒 1（零回归）；水下视角恒 1。
  let cloudF = select( cloudShadowFactor( input.worldPos ), 1.0, uw );

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
  var shoreDepth = 1000.0; // 瞬时水深（米；岸线分支内更新，浅水色渐变消费）
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
    shoreDepth = depth;
    // 2026-09-30 差距8「泡沫大片团块」：破碎带 7→3m 收窄到 5→1.2m 贴岸，
    // 噪声调制 0.75+0.5fN → 0.45+0.55fN（低基高调）——碎浪线细碎断续
    let surf = smoothstep( 5.0, 1.2, depth );
    jacW = 1.0 - 0.5 * surf;
    // 2026-10-06 Round5（前景 carpet 白基线实锤 whitePct 85%/dark 0%）：噪声底
    // 0.45→0.15——fN 谷处白沫真正熄灭，带内纹理不再托亮全域
    shoreCov = surf * renderParams.shoreParams.y * ( 0.15 + 0.85 * fN );
    edgeFade = smoothstep( 0.0, 0.05, depth );
  }

  // T4 泡沫花边（foam lace）：与主破碎带不同参数的独立 smoothstep——比
  // surf 更贴岸的细碎白沫带；breakParams 未启用时 laceCov = 0（零回归）。
  var laceCov = 0.0;
  if ( breakParams.gain.x > 0.5 && breakParams.gain.w > 0.0 ) {
    let laceW = smoothstep( breakParams.band.w, breakParams.band.w * 0.35, shoreDepth );
    laceCov = laceW * breakParams.gain.w * ( 0.6 + 0.4 * fN );
  }
  shoreCov = shoreCov + laceCov;

  // 大尺度破碎波带（2026-10-06 检查差距4）：foam 场本身无大空间尺度结构、
  // fN 噪声（0.59/0.19 m 周期）只到米级 → 白沫呈均匀细碎小斑。涌浪以群
  // 推进，真实白沫聚在群峰带成片破碎、群谷仅零星：低频平面噪声
  // （vnoise ×0.02 ≈ 50 m 周期）+ 当前波高联合门控 foam。2026-10-06
  // Round3 检查差距2（前景 ~80% 近地毯、带间隙不可辨）：阶跃对比加大——
  // 群谷 0.40→0.10、过渡带 (-0.55,0.55)→(-0.20,0.40) 收窄：群峰带全量
  // 白沫、群谷近乎露底（深色水面 ≥30%），~50 m 周期离散破碎带可辨。
  // covTotal 与 foamMask 同吃（A3 图案阈值分支经 coverage 同步生效，带状
  // 结构保留）。船迹 wakeFoam 不经门控（连续性要求）。
  // 2026-10-06 Round4（阶跃两轮仍 bandPeaks=0 的归因实锤）：前景左下地毯的
  // 主导项是 shoreCov（贴岸破碎带 surf 在浅水区全开、不经 swellBand）——
  // uw-diag-r4 开关隔离 + 前景 crop 实锤左下碎斑地毯。
  // 2026-10-06 Round5（0.25 谷底保底仍 carpet：前景 crop whitePct 85.4/dark
  // 0.0/teal 14.6 实锤）：三处彻底去底噪仍无带。R5-probe（swellBand 直接
  // 上屏可视化）实锤根因：vnoise 单瓣（50m/36m 周期）大于前景 ~30m 视域，
  // 前景 swellBand 恒≈1——门控退化为常数 1，调底噪/阈值全部无效。
  // Round5 v3（最终案）：岸带改确定性 20m 正弦条带（沿岸平行、噪声扰动
  // 相位致条带不规则、随 ocean.time 以 ~1.1 m/s 漂移模拟群波推进），条带
  // 占空比 ~50%、谷值真正到 0.02；6-30m 水深过渡回 Round4 原式
  // （shoreDepth 场外=1000 → 全权回原式，中远场零回归）。
  let bandOffshore = 0.10 + 0.90 * smoothstep( -0.20, 0.40,
    input.waveH * 0.65 + vnoise( input.planeXZ * 0.02 ) * 1.6 - 0.8 );
  let grating = 0.5 + 0.5 * sin( dot( input.planeXZ, vec2<f32>( 0.28, 0.96 ) ) * 0.314
    + vnoise( input.planeXZ * 0.035 ) * 2.2 + ocean.time * 0.35 );
  let bandShore = 0.02 + 0.98 * smoothstep( 0.42, 0.85,
    grating * ( 0.55 + 0.45 * clamp( input.waveH * 1.6 + 0.5, 0.0, 1.0 ) ) );
  // Round5 v4：grating 主导域 6-30m→15-45m——probe4 实锤前景浅带（surf 全开
  // 的 <6m 区）曾被 bandOffshore 混回（smoothstep(6,30) 在 8-15m 已大权重），
  // 条带出不来；15m 内纯 grating、45m 外纯原式。
  var swellBand = mix( bandShore, bandOffshore, smoothstep( 15.0, 45.0, shoreDepth ) );
  // Round6 V4（四项改动上屏但前景零变化的归因实锤：前景左下在 seabed 场外
  // ——沙洲场心在相机西南 ~760m、场外哨兵 seabedH=-1000 → shoreDepth≈1000
  // → swellBand=bandOffshore，且该分支低频噪声单瓣（50m 周期）大于近场
  // ~30m 视域 → swellBand 前景常数化 0.4-0.8 无谷，bandCalm/wakeBand/foam
  // 门全部失去调制对象——R5 注释「前景 swellBand 恒≈1」的场外成因补全）。
  // 修复：近场（30-110m 渐变）叠确定性 grating 条带（与 bandShore 同参数
  // 无缝衔接），谷 0.02 让 bandCalm/wakeBand/foam 门在前景重新生效；110m
  // 外=swellBand 原式零回归（中景沙洲条带语义不动）。
  let nearW2 = 1.0 - smoothstep( 30.0, 110.0, distM );
  let bandNear = 0.02 + 0.98 * smoothstep( 0.42, 0.85,
    grating * ( 0.55 + 0.45 * clamp( input.waveH * 1.6 + 0.5, 0.0, 1.0 ) ) );
  swellBand = mix( bandNear, swellBand, 1.0 - nearW2 );
  let shoreBand = shoreCov * swellBand;

  // 2026-10-06 Round6（前景白毯真凶实锤，推翻 R4/R5 的 shoreCov 归因——
  // 开关因果链：关 wakeToggle 前景 crop white200 23-28%→0.3% 清零；关
  // shoreFoam/swash/swashSim/wetSand/spray/breakSpray/peel/sss 八项均不降
  // 16-28% 随机波动；35s→100s 时间序列白毯随船 wake 尾迹推进铺满——圆周
  // 航行反复扫过前景，G 通道泡沫 τ≈1.7s 衰减但 crest/wash 持续 refill）：
  // wakeFoam 由「不经门控」改为贴 swellBand 群相位门窄带化（wakeBand 峰带
  // 全保留、谷压 15%——与 foam/jacW 同门同带），前景深水波谷露青绿水体。
  // swellBand 本身不动 → 中景已达标条带语义零回归（wake 只被调制，只降
  // 不升）；VS 侧 wake R 波高位移保留（真实船迹波形语义，只增波纹不增白）。
  let wakeBand = 0.15 + 0.85 * swellBand;
  let covTotal = foam * jacW * swellBand + shoreBand + wakeFoam * wakeBand;
  let coverage = clamp( covTotal, 0.0, 1.0 );

  var foamMask = clamp( foam * jacW * swellBand * ( 0.45 + 1.1 * fN ), 0.0, 1.0 ) + shoreBand + wakeFoam * wakeBand;
  foamMask = clamp( foamMask, 0.0, 1.0 );
  var fineGlow = 0.0;
  var albedoJit = 1.0;
  if ( renderParams.flags.x > 0.5 ) {
    // A3 阈值方案：coverage 驱动阈值 t，R 密度场 smoothstep 软边；
    // G 细泡亮度 0.3-0.5、B 斑驳反照率 ±10%、A 条纹 ±0.1 偏置。
    // 2026-10-09：①3 层旋转域采样破平铺（见 foamPattern）；②采样坐标加
    // dispXZ（随波）；③阈值 0.85→0.70 覆盖回落（平静海面白斑大幅减少）。
    let advXZ = input.planeXZ + input.dispXZ * 0.85;
    let p = foamPattern( advXZ );
    let t = 1.0 - coverage * 0.70;
    foamMask = clamp( smoothstep( t - 0.04, t + 0.04, p.r ) + ( p.a - 0.5 ) * 0.2, 0.0, 1.0 );
    fineGlow = p.g * coverage;
    albedoJit = 1.0 + ( p.b - 0.5 ) * 0.2;
  }

  // 水色：近岸清透青蓝 → 远海深蓝（距离混合，2026-09-30 差距7：旧式
  // mix(deep, shallow, dist) 方向反了——近处死蓝、远处泛青）；水下切换
  // 透射蓝绿。
  // 2026-10-06 检查差距3（海面列均值 rgb≈[103,119,114]，R≈G 浑浊灰绿）：
  // 旧 shallow [0.10,0.32,0.30] / lime [0.17,0.46,0.38] / uwWater
  // [0.10,0.46,0.42] 三块色板全 G≥B（绿主导），叠 SSS 绿斑与天空反射后
  // 整体灰绿。晴天深蓝要求 R 显著低于 G/B 且 B≥G——三板重标为 B 主导，
  // lime 碧绿→碧青、浅水权重 0.68→0.55（保浅水辨识度、去绿罩）。
  let shallow = vec3<f32>( 0.05, 0.18, 0.26 );
  var baseCol = mix( shallow, renderParams.deepColor.rgb, smoothstep( 150.0, 5000.0, distM ) );
  let lime = vec3<f32>( 0.09, 0.30, 0.38 );
  // Round6 V3（前景 dark90=0 的第二根因：lime 浅水混色域 8-1m 把前景 2-5m
  // 沙洲全域抬亮到 L≈100-115——Lime 本身 L≈105，无论反射怎么压都过不了
  // L<90；off-wake 对照 dark90 仍 0.2% 实锤基色贡献）。混色域收窄到真水线
  // 2.2-0.5m（waterline 泛绿只在极浅），权重 0.55→0.45；中景沙洲外缘水深
  // >2.2m 区域零回归（原本就不该 lime，青蓝 shallow 基色还原）。
  baseCol = mix( baseCol, lime, smoothstep( 2.2, 0.5, shoreDepth ) * 0.45 );
  let uwWater = vec3<f32>( 0.05, 0.24, 0.34 );
  let waterTint = select( baseCol, uwWater, uw );

  // ── L5 物大气视觉：逐像素天空反射（T1）/ 太阳高光染色（T2）/ SSS（T3）──
  // 反射基用相机天顶（skyView LUT 在相机本地系预计算，基必须同源——用海面
  // upDir 会引入方位偏差）；featureFlags 对应位未置即短路回旧平色路径。
  let atmUp = normalize( camPos );
  let ffb = i32( renderParams.underwater.w + 0.5 );
  let flagSky      = select( 0.0, 1.0, ( ffb & 1 ) != 0 );
  let flagSss      = select( 0.0, 1.0, ( ffb & 2 ) != 0 );
  let flagSpecTint = select( 0.0, 1.0, ( ffb & 8 ) != 0 );

  // T2 太阳透射染色：太阳方向过大气后的透射色染高光/SSS；未启用恒 1（零回归）
  var sunTint = vec3<f32>( 1.0 );
  if ( flagSpecTint > 0.5 || flagSss > 0.5 ) {
    sunTint = atmosphereSampleTransmittance( atmosphereParams.viewHeight, clamp( dot( sunDir, atmUp ), -1.0, 1.0 ) );
  }

  // T1 逐像素天空反射：反射向量（ECEF）直接与相机天顶基点积转本地系
  var skyRad = renderParams.skyColor.rgb;
  if ( flagSky > 0.5 ) {
    let refl = reflect( - viewDir, nLit );
    skyRad = atmSkyReflectionRadiance(
      refl, atmUp, normalize( skyParams.sunDir ), normalize( skyParams.moonDir ),
      renderParams.underwater.y );
  }

  // 2026-10-06 Round5 v2（iso2 全源关断实锤：六泡沫源全关后前景仍 white 46%/
  // dark 0%——残留亮源是天空云镜像反射 + 太阳镜面耀光柱，非泡沫）：群谷未破
  // 碎水面压近场镜像与耀光（近岸破碎带外侧波面糙度打碎镜像反射），谷处露出
  // 深色青绿水体。nearW 限制 40-120m 过渡、120m 外全还原（中远场零回归）。
  let nearW = 1.0 - smoothstep( 40.0, 120.0, distM );
  // Round6 V2（wake off 对照实锤：前景 dark90 仍 0.2%/white160 52.9%——白毯
  // 主体是近场掠射 fresnel≈1 的天空反射：swellBand 谷 bandCalm=0.235 时残留
  // skyRad*0.9*0.235 恰把前景拖过 L90，泡沫清零也无深色区）。近场谷压深
  // 0.78→0.92（bandCalm 谷 0.235→0.08，反射 ≈0.05、水体 L≈73<90）；峰带
  // 全量不动（条带白亮节奏保留）；40-120m 过渡带同门渐变、120m 外零回归。
  let bandCalm = 1.0 - 0.92 * nearW * ( 1.0 - swellBand );
  var col = waterTint * ( 0.88 + 0.12 * clamp( 1.0 - divg * 0.5, 0.0, 1.0 ) ) * albedoJit
    + skyRad * fresnel * 0.9 * bandCalm * shadowF * cloudF
    + vec3<f32>( spec ) * mix( vec3<f32>( 1.0 ), sunTint, flagSpecTint ) * bandCalm * shadowF * cloudF;

  // T3 波峰 SSS 透光：波面高×逆光×掠射的青绿透射，随太阳透射色偏移；
  // 波峰自遮蔽微降。RP[22]（underwater.z）总强度 0 = 关。
  if ( flagSss > 0.5 && renderParams.underwater.z > 0.0 ) {
    let upY  = clamp( dot( sunDir, atmUp ), 0.0, 1.0 );
    let H    = clamp( input.waveH * 0.28, 0.0, 1.0 );                  // ≈3.5m 封顶
    let back = clamp( dot( viewDir, - sunDir ) * 0.5 + 0.5, 0.0, 1.0 ); // 逆光因子
    let sssW = H * ( 0.35 + 0.65 * pow( 1.0 - ndv, 2.0 ) )
             * ( 0.25 + 0.75 * back * back ) * upY * renderParams.underwater.z * shadowF;
    // 2026-10-06 差距3：透射色 [0.06,0.42,0.34] 绿斑 → 偏青蓝
    col += vec3<f32>( 0.05, 0.30, 0.38 ) * sssW * sunTint;
    col = mix( col, col * ( 1.0 - 0.25 * H ), 0.5 );
  }

  // 折射：法线水平分量扰动采样场景颜色历史（水体下方背景），
  // 混合权重 ∝ (1-菲涅尔)；history 缺省绑 1×1 零纹理 → 采样为黑但不走此分支；
  // 水下视角跳过（历史为水上画面，语义不成立）
  if ( renderParams.flags.z > 0.5 && !uw ) {
    let roff = vec2<f32>( nLocal.x, nLocal.z ) * renderParams.flags.w;
    let ruv = clamp( input.screenUV + roff, vec2<f32>( 0.001 ), vec2<f32>( 0.999 ) );
    let under = textureSampleLevel( sceneHistTex, clampSmp, ruv, 0.0 ).rgb;
    let underCol = under * vec3<f32>( 0.60, 0.85, 0.90 ); // 水色吸收
    // 2026-10-06 Round6 V5（前景白毯末段实锤：L5 逐项隔离 refractor 是唯一
    // 大跌项——关折射前景 white140 87.9%→28.6%。病理：refrW=(1-fresnel)*0.75
    // 在前景掠射 fresnel≈0.6 时仍透射 0.30，而 under 采样的是屏幕空间历史帧
    // ——掠射视线到不了海床、采到的是天空亮色 L≈130-170，混 30% 把前景整片
    // 拖过 L140。修复：(1-f)² 更陡的视角衰减（掠射 0.30→0.12、俯视不受损）
    // × 水深门（透见的是海床，shoreDepth>6m 渐灭、场外哨兵 1000 归零——
    // 深水/场外本无底可见，浅滩沙洲 <6m 折射语义全保留、中景零回归）。
    // 2026-10-08 水上网纹收敛：水深门 6→30m 收窄到 4.5→8m——历史帧含
    // CausticsFloor 叠画（上一帧合成色），旧窗口把 tile 深底外环（-7m，
    // shoreDepth≈7-9）也按 ~0.3-0.9 权重混进水色，网纹经屏幕空间偏移平抹
    // 过波面、且 tile 边界两侧折射权重 1↔0 形成深水色阶缝。新窗口与
    // floor 见底域（≤6m，见 OceanCausticsShader visK）对齐：浅滩透底语义
    // 全保留，深水/场外（哨兵 1000）恒零。
    let refrW = clamp( ( 1.0 - fresnel ) * ( 1.0 - fresnel ) * 0.75, 0.0, 0.60 )
      * ( 1.0 - smoothstep( 4.5, 8.0, shoreDepth ) );
    col = mix( col, underCol + vec3<f32>( spec ), refrW );
  }

  // T4 内卷水幕（A 案务实近似）：视线穿波峰背面（nLit·viewDir < 0）且破碎
  // 权重高时，按 curl 混入半透明水幕色并增补 foamMask。真 peeling（二次
  // draw + 深度翻转）二期；curl 取级联0 梯度与视线水平分量（局部平面投影）。
  if ( breakParams.gain.x > 0.5 && input.breakW > 0.0 ) {
    let g0 = textureSampleLevel( derivTex, smp, input.planeXZ / ocean.sizes[ 0 ].x, 0, 0.0 );
    let viewXZ = vec2<f32>( dot( viewDir, ocean.eastDir ), dot( viewDir, ocean.northDir ) );
    let curl = dot( g0.xy, viewXZ );
    let backFace = select( 0.0, 1.0, dot( nLit, viewDir ) < 0.0 );
    let foamW = clamp( abs( curl ), 0.0, 1.0 ) * input.breakW;
    let curtain = clamp( - curl, 0.0, 1.0 ) * backFace * input.breakW * breakParams.band.z;
    let curtainCol = mix( vec3<f32>( 0.16, 0.42, 0.45 ), vec3<f32>( 0.85, 0.93, 0.96 ), foamW );
    col = mix( col, curtainCol, clamp( curtain, 0.0, 0.8 ) );
    foamMask = clamp( foamMask + foamW * 0.5, 0.0, 1.0 );
  }

  var foamWhite = vec3<f32>( 0.90, 0.94, 0.97 ) * ( 0.75 + 0.25 * fN + fineGlow );
  foamWhite = select( foamWhite, foamWhite * vec3<f32>( 0.35, 0.60, 0.62 ), uw ); // 水下泡沫偏暗青
  col = mix( col, foamWhite * albedoJit, foamMask * 0.92 );

  // 距离雾：远处融入天空色（弱化网格边缘接缝）；水下远景融入水色。
  // T-P 海面雾太阳染色（aerial perspective 海面侧）：朝太阳方向的雾色染
  // 大气透射色（落日把远方海面雾染橙）。apStrength（shoreParams.z）= 0 恒
  // 旧值（零回归）；bit3 未开时 sunTint 未计算（fallback 黑 LUT 会染黑雾）
  // → 双门短路。1.4 为透射染色增益（透射色偏暗，乘增益补回雾亮度）。
  var fogCol = select( renderParams.skyColor.rgb, uwWater, uw );
  let apStrength = renderParams.shoreParams.z;
  if ( apStrength > 0.0 && flagSpecTint > 0.5 && !uw ) {
    let toSun = sat( dot( viewDir, sunDir ) * 0.5 + 0.5 );
    fogCol = mix( fogCol, fogCol * sunTint * 1.4, toSun * apStrength );
  }
  let fog = 1.0 - exp( - distM / FOG_DIST );
  col = mix( col, fogCol, fog * 0.88 );

  // 水下仰视 Snell 窗：视线与竖直夹角 < 临界角 ~48.6°（ndv > ~0.66）区域
  // 透明天空（亮窗），窗外全反射暗带；alpha 同步降——此前海面 alpha 恒
  // ≈1，水下仰视是一堵不透明水墙（2026-09-29 用户"水下不透明"实测）。
  // 透出的背景是先画的天空 pass（水下相机向上射线的天空辐射）。
  if ( uw ) {
    // Round3 仰视远区亮带「离散方形色块 + 可数色阶」修复：固定 0.60-0.74
    // smoothstep 门在远场（屏幕足迹≈波周期，ndv 逐像素在门两侧跳变）把
    // 透明天空/全反射暗带二值化——亮暗条纹沿波纹方向、硬边锯齿小方块、
    // 亮度可数色阶（检查员原生裁剪实锤）。门宽随屏幕足迹（stepMPix 米/
    // 像素）加宽：掠射波面统计上本就软过渡；近场足迹小，0.07 半宽还原
    // 原 0.60-0.74 窗缘锐度（逐字节零回归）。
    let gateW = clamp( 0.07 + stepMPix * 1.2, 0.07, 0.42 );
    let snell = smoothstep( 0.67 - gateW, 0.67 + gateW, ndv );
    // 2026-09-30 用户"水下天空效果不对"：旧版窗内填 CPU 平色 skyColor×1.1
    // ——无渐变无日盘，一堵假色墙。绑定物大气（flagSky）时改为真实折射：
    // 视线过水-气界面（水→空气 eta=1.33，TIR 区 refract 归零、snell 门同步
    // 压零），沿透射向采样 skyView LUT 辐射——窗内见真实天空渐变与太阳
    // 亮斑；未绑定时回退平色（零回归）。
    var skyThrough = renderParams.skyColor.rgb * 1.1;
    if ( flagSky > 0.5 && snell > 0.001 ) {
      let refrDir = refract( - viewDir, nLit, 1.33 );
      // TIR 边带（ndv 0.60~0.66）refract 归零——零向量不查 LUT（NaN 防护）
      if ( dot( refrDir, refrDir ) > 0.0 ) {
        skyThrough = atmSkyReflectionRadiance(
          refrDir, atmUp, normalize( skyParams.sunDir ), normalize( skyParams.moonDir ),
          renderParams.underwater.y ) * 1.35;
      }
    }
    col = mix( col, skyThrough, snell * 0.75 );
    edgeFade = mix( edgeFade, edgeFade * 0.3, snell );
  }

  return vec4<f32>( col, edgeFade );
}

// ── T-P 真 peeling 卷唇（破碎波水舌卷曲面：同一海面网格第二个 draw）─────────
// peel_vs 在 vs_main 同款位移之后，对破碎带波峰顶点绕「波峰线」（级联0 梯度
// 垂直方向）卷曲：波前进方向 = 梯度反方向，卷曲角 roll ∝ peel.y(卷曲角) ×
// 破碎权重 × 动画相位（sin(time×speed + 沿前进方向空间相位)）。卷曲几何 =
// 波峰上部高度 h 绕波峰轴转 roll：水平前移 h·sin(roll)、垂直下压 h·(1−cos(roll))
// ——顶点翻过波峰形成向前的水舌面片。peel.x=0 时 roll 恒 0（退化回主面），
// CPU 侧在此情形直接跳过第二个 draw（零回归）。
struct PeelVSOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0) worldPos : vec3<f32>,
  @location(1) planeXZ : vec2<f32>,   // 泡沫密度图案采样（与主 FS 同一 A3 图案源）
  @location(2) curlW : f32,           // 卷唇权重（roll 归一，FS 白沫密度/alpha 门）
  @location(3) nTongue : vec3<f32>,   // 卷曲面法线（up 绕波峰轴转 roll）
  @location(4) dispXZ : vec2<f32>,    // 与主 FS 同款：泡沫图案随波采样坐标
}

@vertex
fn peel_vs( input : VSIn ) -> PeelVSOut {
  // 位移采样与 vs_main 完全同款（前 3 级联 + 船迹 + 破碎变形，含手动 LOD
  // 掠射防混叠）——保持两 draw 的基准面一致，卷曲面片才贴合主水面波峰
  var disp = vec3<f32>( 0.0 );
  let peelDistM = length( input.ecef - camera.cameraPosition.xyz ) * 6378137.0;
  for ( var c = 0; c < OCEAN_CASCADES - 1; c++ ) {
    let texelM = ocean.sizes[ c ].x / 256.0;
    let lod = clamp( log2( max( peelDistM / ( texelM * 700.0 ), 1.0 ) ), 0.0, 8.0 );
    let d = textureSampleLevel( dispTex, smp, input.planeXZ / ocean.sizes[ c ].x, c, lod );
    disp += d.xyz;
  }
  if ( renderParams.flags.y > 0.5 ) {
    let wuv = input.planeXZ * renderParams.wakeUv.zw + renderParams.wakeUv.xy;
    let wh = textureSampleLevel( wakeTex, clampSmp, wuv, 0.0 ).r;
    disp.y += wh;
  }
  var breakW = 0.0;
  var g0 = vec4<f32>( 0.0 );
  if ( breakParams.gain.x > 0.5 && renderParams.shoreParams.x > 0.5 ) {
    let buv = input.planeXZ * renderParams.seabedUv.zw + renderParams.seabedUv.xy;
    var sbH = -1000.0;
    if ( all( buv >= vec2<f32>( 0.0 ) ) && all( buv <= vec2<f32>( 1.0 ) ) ) {
      let bdim = textureDimensions( seabedTex, 0 );
      let btc = clamp( vec2<i32>( buv * vec2<f32>( bdim ) ), vec2<i32>( 0 ), vec2<i32>( bdim ) - vec2<i32>( 1 ) );
      sbH = textureLoad( seabedTex, btc, 0 ).r;
    }
    breakW = smoothstep( breakParams.band.y, breakParams.band.x, disp.y - sbH );
    g0 = textureSampleLevel( derivTex, smp, input.planeXZ / ocean.sizes[ 0 ].x, 0, 0.0 );
    disp.y *= mix( 1.0, breakParams.gain.y, breakW );
    // 同 vs_main：避免 disp.xz -=（naga 拒绝多分量 swizzle 复合赋值）
    let leanXY = g0.xy * max( disp.y, 0.0 ) * breakParams.gain.z * breakW;
    disp = vec3<f32>( disp.x - leanXY.x, disp.y, disp.z - leanXY.y );
  }

  // 卷唇变形：breakW 越过 peel.w 阈值且梯度非退化的波峰顶点才翻卷
  var roll = 0.0;
  var advEcef = vec3<f32>( 0.0 );
  let gLen = length( g0.xy );
  if ( breakParams.peel.x > 0.5 && breakW > breakParams.peel.w && gLen > 1e-5 ) {
    let adv2 = - g0.xy / vec2<f32>( gLen );  // 波前进方向（局部平面）
    let phase = dot( input.planeXZ, adv2 ) * 0.35 + ocean.time * breakParams.peel.z;
    let w = smoothstep( breakParams.peel.w, breakParams.peel.w + 0.25, breakW );
    roll = breakParams.peel.y * w * ( 0.65 + 0.35 * sin( phase ) )
         * smoothstep( 0.0, 0.6, disp.y );   // 只卷波峰之上（波谷不翻）
    let h = max( disp.y, 0.0 );
    let advXY = adv2 * ( h * sin( roll ) );   // 绕波峰线前移（无 swizzle 复合赋值）
    disp = vec3<f32>( disp.x + advXY.x, disp.y, disp.z + advXY.y );
    disp.y    -= h * ( 1.0 - cos( roll ) );  // 绕波峰线下压（舌面翻过波峰）
    advEcef = ocean.eastDir * adv2.x + ocean.northDir * adv2.y;
  }
  let dispVec = ocean.eastDir * disp.x + ocean.upDir * disp.y + ocean.northDir * disp.z;

  // 基准/位移分别过 viewProj 再相加（与 vs_main 同款精度处理）
  let clipBase = camera.viewProjMatrix * vec4<f32>( input.ecef, 1.0 );
  let clipDisp = camera.viewProjMatrix * vec4<f32>( dispVec, 0.0 );

  var out : PeelVSOut;
  out.clipPos = clipBase + clipDisp;
  out.worldPos = input.ecef + dispVec;
  out.planeXZ = input.planeXZ;
  out.curlW = clamp( roll * 1.4, 0.0, 1.0 );
  out.dispXZ = disp.xz;
  // 卷曲面法线：up 绕波峰轴转 roll → 舌面翻卷后背面朝后上（roll→π/2 时
  // 法线水平指回观察者，即经典卷唇内壁）
  out.nTongue = normalize( ocean.upDir * cos( roll ) - advEcef * sin( roll ) );
  return out;
}

// ── T-P peeling FS：白沫材质 + 半透明叠加 + 边缘 fresnel 增亮 ───────────────
// 无深度写入（loadOp load alpha 叠加）+ cullMode none 双面：舌面内外壁都出片；
// 未卷区域（curlW≈0）discard 避免与主水面 z 争抢闪烁。
@fragment
fn peel_fs( input : PeelVSOut ) -> @location(0) vec4<f32> {
  if ( input.curlW < 0.02 ) { discard; }

  let camPos = camera.cameraPosition.xyz;

  // 地球遮挡（IPass 无深度附件，片元级射线-球求交，与 fs_main 同款）
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

  let toCam = camPos - input.worldPos;
  let distN = length( toCam );
  let distM = distN * EARTH_SCALE;
  let viewDir = toCam / max( distN, 1e-9 );

  // 法线内翻（-n）：舌面内壁朝向观察者 → fresnel 在卷唇边缘增亮
  let n = - normalize( input.nTongue );
  let ndv = clamp( dot( n, viewDir ), 0.0, 1.0 );
  let fres = pow( 1.0 - ndv, 3.0 );

  // 白沫材质：A3 密度图案（开图案时）阈值化或 vnoise 兜底；白 0.95 基色 +
  // 卷唇边缘 fresnel 增亮
  let fN = vnoise( input.planeXZ * 1.7 ) * 0.6 + vnoise( input.planeXZ * 5.3 ) * 0.4;
  var dens = 0.9 + 0.1 * fN;
  if ( renderParams.flags.x > 0.5 ) {
    let p = foamPattern( input.planeXZ + input.dispXZ * 0.85 );
    dens = clamp( p.r + p.g * 0.3 * input.curlW, 0.35, 1.0 );
  }
  var col = mix( vec3<f32>( 0.9, 0.94, 0.97 ),
                 vec3<f32>( 0.95 ) * ( 0.8 + 0.2 * fN ) + vec3<f32>( fres * 0.35 ),
                 dens );

  // 距离雾融入天空色（与主 FS 同款）：防远处卷唇片跳色
  let fog = 1.0 - exp( - distM / FOG_DIST );
  col = mix( col, renderParams.skyColor.rgb, fog * 0.95 );

  // 半透明 0.85 封顶：alpha ∝ 卷唇权重 × 密度
  return vec4<f32>( col, clamp( input.curlW * 0.85 * dens, 0.0, 0.85 ) );
}
`;

/** OceanParams uniform buffer 字节数（288B，见 struct 注释）。 */
export const OCEAN_PARAMS_UBO_BYTES = 288;
/** OceanRenderParams uniform buffer 字节数（128B：sunDir/deep/sky/flags/wakeUv/underwater/shoreParams/seabedUv）。 */
export const OCEAN_RENDER_PARAMS_UBO_BYTES = 128;
/** BreakParams uniform buffer 字节数（96B：gain/band/peel 三 vec4）。 */
export const OCEAN_BREAK_PARAMS_UBO_BYTES = 96;
