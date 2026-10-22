import { beforeAll, describe, expect, it } from 'vitest';
import { OceanFFTWater, oceanSrgbToLinear } from '../../src/effect/OceanFFTWater';
import {
  OCEAN_FFT_RENDER_SHADER,
  OCEAN_RENDER_PARAMS_UBO_BYTES,
} from '../../src/shaders/OceanFFTShader';
import type { Camera } from '../../src/scene/Camera';
import type { Context } from '../../src/renderer/Context';

/**
 * T-TROPICAL 观感档验证（无 GPU adapter，全 mock device；复刻
 * threejswaterpro TROPICAL 的 UBO 参数注入 + 渲染源串结构）：
 *   1. oceanSrgbToLinear：sRGB hex → 线性 RGB 数值断言（手算基准）
 *   2. look='tropical'：RP[32..53] 写入正确（开关/三色线性值/标量参数/hazeScale）
 *   3. 默认档回归保护：RP[32..55] 全零 + 前 128B 关键字段与旧布局逐字段一致
 *   4. 渲染 shader 源串：tropical 分支关键式 + 旧 default 路径钉串不动
 * look=0 时 shader `if (renderParams.lookShallow.x > 0.5 ...)` 全部短路，
 * default 档 col 逐字节零回归。
 */

/** RenderParams UBO 扩容段 f32 索引（T-TROPICAL：lookShallow@32 起 6 个 vec4）。 */
const RP_LOOK_ON = 32;
const RP_SHALLOW = 33; // 33..35 线性 RGB
const RP_DEEP = 37;    // 37..39
const RP_SPARK_SUN = 40;
const RP_TRANS = 41;   // 41..43
const RP_FRESNEL_POW = 44;
const RP_SPARKLE_POW = 45;
const RP_SPARKLE_INT = 46;
const RP_FOAM_SCALE = 47;
const RP_SSS_INT = 48;
const RP_SSS_POW = 49;
const RP_DEPTH_OFF = 50;
const RP_DEPTH_FALL = 51;
const RP_HAZE_SCALE = 52;     // lookParamsC.x：俯视雾压制（R3）
const RP_RIPPLE_STRENGTH = 53; // lookParamsC.y：波痕明暗强度（R3）

/** GPUShaderStage 常量（mock 环境值，与 beforeAll 注入一致）。 */
const STAGE_FRAGMENT = 2;
const STAGE_VERTEX = 1;

interface UboWrite {
  label: string;
  f32: Float32Array;
}

interface BglEntry {
  binding: number;
  visibility?: number;
  texture?: { sampleType?: string };
  buffer?: { type?: string; minBindingSize?: number };
  sampler?: { type?: string };
}

interface MockHarness {
  ctx: Context;
  uboWrites: UboWrite[];
  texWrites: { label: string; data: Float32Array }[];
  bindGroups: { entries: { binding: number; resource: unknown }[] }[];
  layouts: { label: string; entries: BglEntry[] }[];
}

/** 记录型 mock Context（与 oceanShoreFoam.spec 同构：UBO 写入快照捕获）。 */
function makeHarness(): MockHarness {
  const uboWrites: UboWrite[] = [];
  const texWrites: { label: string; data: Float32Array }[] = [];
  const bindGroups: { entries: { binding: number; resource: unknown }[] }[] = [];
  const layouts: { label: string; entries: BglEntry[] }[] = [];

  const device = {
    createBuffer: ( d: { label: string; size: number } ) => ( { label: d.label, size: d.size, destroy() {} } ),
    createTexture: ( d: { label: string } ) => ( {
      label: d.label,
      createView: ( opts?: { label?: string } ) => ( {
        label: opts?.label ?? `${ d.label }.View`,
        textureLabel: d.label,
      } ),
      destroy() {},
    } ),
    createSampler: () => ( {} ),
    createShaderModule: ( code: string, label?: string ) => ( { label, codeChars: code.length } ),
    createBindGroupLayout: ( d: { label: string; entries: BglEntry[] } ) => {
      const layout = { label: d.label, entries: d.entries };
      layouts.push( layout );
      return layout;
    },
    createPipelineLayout: ( d: { bindGroupLayouts: unknown[] } ) => ( { bindGroupLayouts: d.bindGroupLayouts } ),
    createComputePipeline: ( d: { label: string } ) => ( { label: d.label } ),
    createRenderPipeline: ( d: { label: string } ) => ( { label: d.label } ),
    createBindGroup: ( d: { entries: { binding: number; resource: unknown }[] } ) => {
      const bg = { entries: d.entries };
      bindGroups.push( bg );
      return bg;
    },
    queue: {
      writeTexture: ( dest: { texture: { label: string } }, data: Float32Array ) => {
        texWrites.push( { label: dest.texture.label, data: new Float32Array( data ) } );
      },
    },
  };

  const ctx = {
    initialized: true,
    device,
    preferredFormat: 'bgra8unorm',
    drawingBufferWidth: 800,
    drawingBufferHeight: 600,
    createSampler: () => ( {} ),
    createShaderModule: ( code: string, label?: string ) => ( { label, codeChars: code.length } ),
    writeBuffer: (
      buffer: { label: string }, _offset: number,
      data: ArrayBufferView | ArrayBuffer,
    ) => {
      // OceanUBO 走裸 ArrayBuffer（OceanFFTWater.ts:832 传 _paramsBytes），
      // 其余走 _asBytes() 的 Uint8Array——两种形态统一到 Float32Array 视图。
      const buf = data instanceof ArrayBuffer ? data : ( data.buffer as ArrayBuffer );
      const off = data instanceof ArrayBuffer ? 0 : data.byteOffset;
      const len = data instanceof ArrayBuffer ? data.byteLength : data.byteLength;
      const src = new Float32Array( buf, off, len / 4 );
      uboWrites.push( { label: buffer.label, f32: new Float32Array( src ) } );
    },
  } as unknown as Context;

  return { ctx, uboWrites, texWrites, bindGroups, layouts };
}

function makeCamera(): Camera {
  return {
    position: { x: 0, y: 0, z: 2 },
    viewMatrix: { values: new Float32Array( 16 ) },
    projectionMatrix: { values: new Float32Array( 16 ) },
    viewProjectionMatrix: { values: new Float32Array( 16 ) },
    frustum: { near: 0.1, far: 1e7 },
  } as unknown as Camera;
}

const computePass = {
  setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {},
};
const renderPass = {
  setPipeline() {}, setBindGroup() {}, setVertexBuffer() {}, setIndexBuffer() {},
  drawIndexed() {}, end() {},
};
const encoder = {
  beginComputePass: () => computePass,
  beginRenderPass: () => renderPass,
} as unknown as GPUCommandEncoder;

function makeOcean( ctx: Context, look?: 'tropical' ): OceanFFTWater {
  return new OceanFFTWater( ctx, {
    center: [ 121.0, 31.0 ],
    extentKm: 0.5,
    segments: 4,
    ...( look ? { look } : {} ),
  } );
}

/** 第 n 帧（负数 = 从尾部倒数）写入的 RenderParams UBO 内容。 */
function rpAt( h: MockHarness, n: number ): Float32Array {
  const writes = h.uboWrites.filter( ( w ) => w.label === 'OceanFFTRenderParamsUBO' );
  let idx = n;
  if ( idx < 0 ) idx = writes.length + idx;
  return writes[ idx ].f32;
}

beforeAll( () => {
  const g = globalThis as unknown as Record<string, unknown>;
  g.GPUBufferUsage ??= {
    MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32,
    UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512,
  };
  g.GPUTextureUsage ??= {
    COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16,
  };
  g.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
} );

describe( 'oceanSrgbToLinear（T-TROPICAL 色域换算）', () => {
  it( 'tropical preset 三色 sRGB → 线性 RGB（手算基准，1e-3 级）', () => {
    // #117778：R=17/255→((0.0667+0.055)/1.055)^2.4、G=119/255、B=120/255
    const shallow = oceanSrgbToLinear( '#117778' );
    expect( shallow[0] ).toBeCloseTo( 0.0056, 3 );
    expect( shallow[1] ).toBeCloseTo( 0.1845, 3 );
    expect( shallow[2] ).toBeCloseTo( 0.1878, 3 );
    // #084c59：R=8/255 ≤ 0.04045 走线性段 /12.92
    const deep = oceanSrgbToLinear( '#084c59' );
    expect( deep[0] ).toBeCloseTo( 0.0024, 3 );
    expect( deep[1] ).toBeCloseTo( 0.0723, 3 );
    expect( deep[2] ).toBeCloseTo( 0.0999, 3 );
    // #2ab29d
    const trans = oceanSrgbToLinear( '#2ab29d' );
    expect( trans[0] ).toBeCloseTo( 0.0232, 3 );
    expect( trans[1] ).toBeCloseTo( 0.4452, 3 );
    expect( trans[2] ).toBeCloseTo( 0.3372, 3 );
  } );

  it( '黑/白边界 + 分量入参 + 非法格式抛错', () => {
    expect( oceanSrgbToLinear( '#000000' ) ).toEqual( [ 0, 0, 0 ] );
    expect( oceanSrgbToLinear( '#ffffff' )[0] ).toBeCloseTo( 1.0, 6 );
    // 0.5 sRGB → ((0.555)/1.055)^2.4 ≈ 0.2140
    expect( oceanSrgbToLinear( [ 0.5, 0.5, 0.5 ] )[0] ).toBeCloseTo( 0.2140, 3 );
    expect( () => oceanSrgbToLinear( '117778' ) ).toThrow( /#rrggbb/ );
    expect( () => oceanSrgbToLinear( '#12345' ) ).toThrow( /#rrggbb/ );
  } );
} );

describe( 'OceanFFTWater look=tropical UBO 注入', () => {
  it( 'RP[32..53] 写入：开关=1、三色线性值、标量参数（默认 preset 值）', () => {
    const h = makeHarness();
    const ocean = makeOcean( h.ctx, 'tropical' );
    ocean.updateCamera( makeCamera() );
    ocean.execute( encoder, {} as GPUTextureView, 0.016 );

    const rp = rpAt( h, -1 );
    expect( rp.length ).toBe( OCEAN_RENDER_PARAMS_UBO_BYTES / 4 ); // 56 f32
    expect( rp[RP_LOOK_ON] ).toBe( 1 );
    // shallow #117778 线性
    expect( rp[RP_SHALLOW] ).toBeCloseTo( 0.0056, 3 );
    expect( rp[RP_SHALLOW + 1] ).toBeCloseTo( 0.1845, 3 );
    expect( rp[RP_SHALLOW + 2] ).toBeCloseTo( 0.1878, 3 );
    // deep #084c59 线性
    expect( rp[RP_DEEP] ).toBeCloseTo( 0.0024, 3 );
    expect( rp[RP_DEEP + 1] ).toBeCloseTo( 0.0723, 3 );
    expect( rp[RP_DEEP + 2] ).toBeCloseTo( 0.0999, 3 );
    // transmission #2ab29d 线性 + sparkle 太阳增益 0.4
    expect( rp[RP_SPARK_SUN] ).toBeCloseTo( 0.4, 6 );
    expect( rp[RP_TRANS] ).toBeCloseTo( 0.0232, 3 );
    expect( rp[RP_TRANS + 1] ).toBeCloseTo( 0.4452, 3 );
    expect( rp[RP_TRANS + 2] ).toBeCloseTo( 0.3372, 3 );
    // 标量参数（surfing tropical preset）
    expect( rp[RP_FRESNEL_POW] ).toBeCloseTo( 2.2, 6 );
    expect( rp[RP_SPARKLE_POW] ).toBeCloseTo( 600, 6 );
    expect( rp[RP_SPARKLE_INT] ).toBeCloseTo( 1.8, 6 );
    expect( rp[RP_FOAM_SCALE] ).toBeCloseTo( 0.25, 6 );
    expect( rp[RP_SSS_INT] ).toBeCloseTo( 1.2, 6 );
    expect( rp[RP_SSS_POW] ).toBeCloseTo( 3.5, 6 );
    expect( rp[RP_DEPTH_OFF] ).toBeCloseTo( 57, 6 );
    expect( rp[RP_DEPTH_FALL] ).toBeCloseTo( 200, 6 );
    // lookParamsC（R3）：俯视雾压制 / 波痕明暗（tropical 默认）
    expect( rp[RP_HAZE_SCALE] ).toBeCloseTo( 0.55, 6 );
    expect( rp[RP_RIPPLE_STRENGTH] ).toBeCloseTo( 0.45, 6 );
  } );

  it( '细粒度覆盖：sRGB 分量覆盖 + 标量覆盖写入 UBO', () => {
    const h = makeHarness();
    const ocean = new OceanFFTWater( h.ctx, {
      center: [ 121.0, 31.0 ], extentKm: 0.5, segments: 4,
      look: 'tropical',
      shallowColor: [ 0.1, 0.2, 0.3 ],
      sparkleSunGain: 0.5,
      foamScale: 0.4,
      depthOffsetM: 30,
      depthFalloffM: 120,
      hazeScale: 0.7,
      rippleStrength: 0.2,
    } );
    ocean.updateCamera( makeCamera() );
    ocean.execute( encoder, {} as GPUTextureView, 0.016 );

    const rp = rpAt( h, -1 );
    const lin = oceanSrgbToLinear( [ 0.1, 0.2, 0.3 ] );
    expect( rp[RP_SHALLOW] ).toBeCloseTo( lin[0], 6 );
    expect( rp[RP_SHALLOW + 1] ).toBeCloseTo( lin[1], 6 );
    expect( rp[RP_SHALLOW + 2] ).toBeCloseTo( lin[2], 6 );
    expect( rp[RP_SPARK_SUN] ).toBeCloseTo( 0.5, 6 );
    expect( rp[RP_FOAM_SCALE] ).toBeCloseTo( 0.4, 6 );
    expect( rp[RP_DEPTH_OFF] ).toBeCloseTo( 30, 6 );
    expect( rp[RP_DEPTH_FALL] ).toBeCloseTo( 120, 6 );
    expect( rp[RP_HAZE_SCALE] ).toBeCloseTo( 0.7, 6 );
    expect( rp[RP_RIPPLE_STRENGTH] ).toBeCloseTo( 0.2, 6 );
  } );

  it( '默认档回归保护：RP[32..55] 全零 + 前 128B 关键字段与旧布局逐字段一致', () => {
    const h = makeHarness();
    const ocean = makeOcean( h.ctx ); // 不传 look
    ocean.updateCamera( makeCamera() );
    ocean.execute( encoder, {} as GPUTextureView, 0.016 );

    const rp = rpAt( h, -1 );
    expect( rp.length ).toBe( OCEAN_RENDER_PARAMS_UBO_BYTES / 4 );
    // 扩容段全零 → shader look 开关短路（零回归的关键前提）
    for ( let i = 32; i < 56; i++ ) {
      expect( rp[ i ] ).toBe( 0 );
    }
    // 旧布局静态段：sunDir（构造默认 [0.4,0.5,0.77] 归一化）/ deepColor / skyColor
    // （_prepareRender 固定值，2026-09-30 差距7 深蓝 + 天空雾色）
    const len = Math.hypot( 0.4, 0.5, 0.77 );
    expect( rp[0] ).toBeCloseTo( 0.4 / len, 6 );
    expect( rp[1] ).toBeCloseTo( 0.5 / len, 6 );
    expect( rp[2] ).toBeCloseTo( 0.77 / len, 6 );
    expect( rp[3] ).toBe( 0 );
    expect( rp[4] ).toBeCloseTo( 0.014, 6 );  // deepColor #0a1647 系
    expect( rp[5] ).toBeCloseTo( 0.088, 6 );
    expect( rp[6] ).toBeCloseTo( 0.140, 6 );
    expect( rp[7] ).toBe( 1 );
    expect( rp[8] ).toBeCloseTo( 0.55, 6 );   // skyColor
    expect( rp[9] ).toBeCloseTo( 0.72, 6 );
    expect( rp[10] ).toBeCloseTo( 0.88, 6 );
    expect( rp[11] ).toBe( 1 );
  } );
} );

describe( 'OCEAN_FFT_RENDER_SHADER T-TROPICAL 源串', () => {
  const s = OCEAN_FFT_RENDER_SHADER.replace( /\s+/g, '' );

  it( 'UBO 扩容 + struct 新字段（前 128B 布局字段全部保留）', () => {
    expect( OCEAN_RENDER_PARAMS_UBO_BYTES ).toBe( 224 );
    expect( s ).toContain( 'sunDir:vec4<f32>' );
    expect( s ).toContain( 'deepColor:vec4<f32>' );
    expect( s ).toContain( 'skyColor:vec4<f32>' );
    expect( s ).toContain( 'flags:vec4<f32>' );
    expect( s ).toContain( 'wakeUv:vec4<f32>' );
    expect( s ).toContain( 'underwater:vec4<f32>' );
    expect( s ).toContain( 'shoreParams:vec4<f32>' );
    expect( s ).toContain( 'seabedUv:vec4<f32>' );
    expect( s ).toContain( 'lookShallow:vec4<f32>' );
    expect( s ).toContain( 'lookDeep:vec4<f32>' );
    expect( s ).toContain( 'lookTrans:vec4<f32>' );
    expect( s ).toContain( 'lookParamsA:vec4<f32>' );
    expect( s ).toContain( 'lookParamsB:vec4<f32>' );
    expect( s ).toContain( 'lookParamsC:vec4<f32>' );
  } );

  it( 'tropical 分支关键式：三色水深 / wrap-lighting SSS / sparkle / 宽带 fresnel / 泡沫弱化', () => {
    // T-C1 三色水深混合（seabed 真实水深优先 + 掠射近似封顶）
    expect( s ).toContain(
      'mix(renderParams.lookShallow.yzw,renderParams.lookDeep.yzw,depthF)' );
    expect( s ).toContain( 'smoothstep(renderParams.lookParamsB.z,renderParams.lookParamsB.w,effDepth)' );
    // T-C2 wrap-lighting SSS（surfing wrap 0.5 / mix 向透射色）。R4 行为等价
    // 更新：waveTr 由 sat(dot(nLit,-sunDir)+0.3)（平面水面恒 0 → 波峰无透光）
    // 改「法线相对平面基线的倾倒量」——平面基线透光 0.45、逆光坡增强顺光坡压暗
    expect( s ).toContain( 'letbacklit=sat((dot(viewDir,-sunDir)+0.5)/1.5)' );
    expect( s ).toContain( 'letwaveTr=sat((dot(nLit,-sunDir)-flatDot)*0.9+0.45)' );
    expect( s ).toContain( 'letflatDot=-dot(sunDir,atmUp)' );
    expect( s ).toContain( 'mix(waterCol,renderParams.lookTrans.yzw,sssF)' );
    // R4 差距3：SSS distanceFade 放宽 fadeEnd 600→1500（透光延伸到中景）
    expect( s ).toContain( '1.0-smoothstep(80.0,1500.0,distM)' );
    // T-C3 sparkle（pow 600 锐瓣 + 泡沫抑制）+ R4 坡度调制（亮带扭曲断续）
    expect( s ).toContain( 'pow(max(dot(nLit,hvT),0.0),renderParams.lookParamsA.y)' );
    expect( s ).toContain( '(1.0-foamMask*0.7)' );
    // R5 回调：0.55+0.9×K2.0 杀闪点过头（c>200 17008→96），基线 0.82 温和调制
    expect( s ).toContain( '(0.82+0.6*sat(slopeAmp*1.4))' );
    // T-C4 宽带 fresnel（power 2.2 可调）+ 旧 spec 清零（双高光防叠加）
    expect( s ).toContain( 'fresnel=0.02+0.98*pow(1.0-ndv,renderParams.lookParamsA.x)' );
    expect( s ).toContain( 'spec=0.0' );
    // T-C5 合成对齐 surfing：mix(waterSSS, 天空反射, fresnel) → +sparkle
    expect( s ).toContain( 'col=mix(waterSSS,skyRad*bandCalm*shadowF*cloudF,fresnel)' );
    expect( s ).toContain( '+vec3<f32>(spark)*mix(vec3<f32>(1.0),sunTint,flagSpecTint)*shadowF*cloudF' );
    // 泡沫 ×foamScale ×pattern 侵蚀（R4）
    expect( s ).toContain( 'letfoamScaleT=select(1.0,renderParams.lookParamsA.w,renderParams.lookShallow.x>0.5)' );
    expect( s ).toContain( 'foamMask*0.92*foamScaleT*foamErode' );
    expect( s ).toContain( 'varfoamErode=1.0;' );
    expect( s ).toContain( 'smoothstep(0.30,0.72,p.r*0.6+p.g*0.4)*0.85+0.15' );
    // R3 俯视治疗：hazeScale 视线天顶角加权雾压制 + rippleShade 波痕明暗
    expect( s ).toContain(
      'select(1.0,renderParams.lookParamsC.x,renderParams.lookShallow.x>0.5)' );
    expect( s ).toContain( 'col=mix(col,fogCol,fog*0.96*hazeScale)' );
    expect( s ).toContain( 'slopeToSun*renderParams.lookParamsC.y' );
    expect( s ).toContain( 'mix(renderParams.lookShallow.yzw,renderParams.lookDeep.yzw,depthF)*rippleShade' );
  } );

  it( 'default 档零回归：旧路径钉串全部保留 + T3 SSS tropical 短路门', () => {
    expect( s ).toContain( 'varfresnel=select(' );
    expect( s ).toContain( '0.02+0.98*pow(1.0-ndv,5.0)' );
    expect( s ).toContain( '+skyRad*fresnel*0.9*bandCalm*shadowF*cloudF' );
    expect( s ).toContain( 'vec3<f32>(spec)*mix(vec3<f32>(1.0),sunTint,flagSpecTint)*bandCalm*shadowF*cloudF' );
    // T3 SSS 块体不动（oceanSkyReflection 钉串），条件追加热带短路
    expect( s ).toContain( 'flagSss>0.5&&renderParams.underwater.z>0.0&&renderParams.lookShallow.x<=0.5' );
    expect( s ).toContain( 'col+=vec3<f32>(0.05,0.30,0.38)*sssW*sunTint' );
  } );
} );

describe( 'render BGL binding 5（renderParams UBO）扩容同步', () => {
  it( 'minBindingSize 引用 OCEAN_RENDER_PARAMS_UBO_BYTES（224）', () => {
    const h = makeHarness();
    const ocean = makeOcean( h.ctx, 'tropical' );
    ocean.updateCamera( makeCamera() );
    ocean.execute( encoder, {} as GPUTextureView, 0.016 );

    const renderBgl = h.layouts.find( ( l ) => l.label === 'OceanFFTRenderBGL' );
    expect( renderBgl ).toBeDefined();
    const e5 = renderBgl!.entries.find( ( e ) => e.binding === 5 );
    expect( e5?.buffer?.type ).toBe( 'uniform' );
    expect( e5?.buffer?.minBindingSize ).toBe( 224 );
    expect( e5?.visibility ).toBe( STAGE_VERTEX | STAGE_FRAGMENT );
    expect( STAGE_FRAGMENT ).toBe( 2 );
  } );
} );

describe( 'R4 波谱覆盖（OceanParams SYSA 段）', () => {
  it( 'spreadBlend/swellWindDirection 写入谱 UBO（local@34 / swell windDirection@37 rad）', () => {
    const h = makeHarness();
    const ocean = new OceanFFTWater( h.ctx, {
      center: [ 121.0, 31.0 ], extentKm: 0.5, segments: 4,
      look: 'tropical',
      spreadBlend: 0.62,
      swellWindDirection: -25,
    } );
    ocean.updateCamera( makeCamera() );
    ocean.execute( encoder, {} as GPUTextureView, 0.016 );

    // _updateSpectrumUniforms：P[P_SYSA(32) + i*4 + 2] = spreadBlend、
    // P[P_SYSA + i*4 + 1] = windDirection(rad)；i=0 local / i=1 swell。
    const ops = h.uboWrites.filter( ( w ) => w.label === 'OceanFFTOceanUBO' );
    expect( ops.length ).toBeGreaterThan( 0 );
    const p = ops[ ops.length - 1 ].f32;
    expect( p[ 34 ] ).toBeCloseTo( 0.62, 6 );                      // local.spreadBlend
    expect( p[ 37 ] ).toBeCloseTo( -25 * Math.PI / 180, 5 );      // swell.windDirection
    // 未覆盖字段维持现值（零回归对照：local windDirection 默认 20°@33）
    expect( p[ 33 ] ).toBeCloseTo( 20 * Math.PI / 180, 5 );
  } );
} );
