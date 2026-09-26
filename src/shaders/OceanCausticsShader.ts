/**
 * OceanCausticsShader - 海面焦散 WGSL（光子溅射光栅化，移植 tidewater
 * Caustics.js——Evan Wallace "WebGL Water" 同法）。
 *
 * 生成 pass（OceanCaustics 持有）：细网格盖住焦散纹理覆盖域 + 边缘 margin，
 * 每顶点取波面一点：采样 OceanFFTWater 的导数纹理（rgba16float 2d-array，
 * (dDy/dx, dDy/dz, dDx/dx, dDz/dz)）重建法线，太阳光线折射后落到 D 米深
 * 平面，顶点改置于落点；片元写（波面元面积 / 海底像元面积）= 光集中度，
 * 加性混合自然叠出聚焦亮网。局部 ENU 框架（x 东 / y 上 / z 北），与
 * OceanFFTShader 的 nLocal = (-slopeX, 1, -slopeZ) 一致。
 *
 * 两个焦平面：R=浅(~2.5m) G=深(~6m)，查询时按真实深度线性混合（wD）。
 * 宽级联（级联 2）的宽聚焦以 0.6*(I-1) 加性烘焙进同一纹理——查询期
 * `c * mix(1, br, 0.6) ≈ c + 0.6*(br-1)`（c 均值 ~1），均值不被抬高。
 *
 * 平折射偏移 T0 从落点中扣除（图案登记在入水点），查询侧按真实深度重加
 * 回溯偏移 + 长波斜率摆动（CausticsFloor 片元，tidewater causticsSample
 * 的 slope 参数语义：涌浪斜率使整张焦散网随波摇摆）。
 */

// ── 生成 pass 共享 uniform（64B）────────────────────────────────────────────

const CAUSTICS_PARAMS_WGSL = /* wgsl */`
struct CausticsParams {
  sunLocal : vec4<f32>,  // @0  指向太阳，局部 ENU（x 东 / y 上 / z 北）
  tiles    : vec4<f32>,  // @16 x=细级联平铺(m) y=宽级联平铺(m) z=纹理覆盖(m) w=fftSize
  depths   : vec4<f32>,  // @32 x=浅焦面 D y=深焦面 D z=宽层浅 D w=宽层深 D
}
`;

/** 生成 pass 公共头（uniform + 采样资源 + 顶点输出布局）。 */
const CAU_GEN_COMMON_WGSL = /* wgsl */`${CAUSTICS_PARAMS_WGSL}
@group(0) @binding(0) var<uniform> caustics : CausticsParams;
@group(0) @binding(1) var derivTex : texture_2d_array<f32>;
@group(0) @binding(2) var smp : sampler;

struct CauOut {
  @builtin(position) pos  : vec4<f32>,
  @location(0)       vOld : vec2<f32>,   // 波面点（平面米）
  @location(1)       vNew : vec2<f32>,   // 海底落点（平面米）
}
`;

/** 单个生成层配置（tidewater CausticLayer 的 depths[k] × cascade 组合）。 */
interface GenLayer {
  cascadeIdx: number;    // 采样的导数级联（3=细 7.1m，2=宽 33.3m）
  slopeLevel: number;    // 采样 mip（粗滤波掉细波纹，防立即散焦）
  tileSwizzle: string;   // caustics.tiles 分量（平铺尺寸，米）
  depthSwizzle: string;  // caustics.depths 分量（焦平面深度，米）
  rExpr: string;         // 片元 R 通道表达式
  gExpr: string;         // 片元 G 通道表达式
  grid: number;          // 网格边长（四边形数）
  margin: number;        // 相对覆盖域的边缘 margin（邻域波面折射进图内）
}

function genLayerShader(g: GenLayer): string {
  const span = (1 + 2 * g.margin).toFixed(4);
  const margin = g.margin.toFixed(4);
  return /* wgsl */`${CAU_GEN_COMMON_WGSL}
// ${g.grid}×${g.grid} 四边形网格盖住 [-${margin}, 1+${margin}]²（索引化：
// 每波面顶点只执行一次 VS，后变换缓存在 6 个三角形间共享）
const VERTS : u32 = ${g.grid + 1}u;
const GRIDF : f32 = ${g.grid}.0;
const SPAN : f32 = ${span};
const MARGIN : f32 = ${margin};

@vertex
fn vs_main( @builtin( vertex_index ) vi : u32 ) -> CauOut {
  let cx = vi % VERTS;
  let cy = vi / VERTS;
  let uv = vec2<f32>( vec2<u32>( cx, cy ) ) / GRIDF * SPAN - MARGIN;
  let p = uv * caustics.tiles.z;   // 平面米（沙洲中心为原点）
  // 波面法线：导数纹理 (dDy/dx, dDy/dz, dDx/dx, dDz/dz) → 斜率 → 法线
  let d = textureSampleLevel( derivTex, smp, p / caustics.tiles.${g.tileSwizzle}, ${g.cascadeIdx}, ${g.slopeLevel} );
  let s = vec2<f32>( d.x / max( d.z + 1.0, 0.3 ), d.y / max( d.w + 1.0, 0.3 ) );
  let n = normalize( vec3<f32>( - s.x, 1.0, - s.y ) );
  // 太阳光线经波面折射落到 D 米深平面；平折射偏移 T0 扣除
  // （图案登记在入水点，查询侧按真实深度重加）
  let T = refract( - caustics.sunLocal.xyz, n, 1.0 / 1.333 );
  let tDown = max( - T.y, 0.15 );
  let T0 = refract( - caustics.sunLocal.xyz, vec3<f32>( 0.0, 1.0, 0.0 ), 1.0 / 1.333 );
  let off = ( T.xz / tDown - T0.xz / max( - T0.y, 0.15 ) ) * caustics.depths.${g.depthSwizzle};
  let q = p + off;
  var o : CauOut;
  o.vOld = p;
  o.vNew = q;
  // 正交投射：落点平面米 → 纹理 NDC（与相机无关的离屏 pass）
  let ndc = q / caustics.tiles.z * 2.0 - 1.0;
  o.pos = vec4<f32>( ndc.x, ndc.y, 0.0, 1.0 );
  return o;
}

@fragment
fn fs_main( input : CauOut ) -> @location(0) vec4<f32> {
  // 波面元与海底像元的面积比 = 光集中度；近折叠单元软限幅
  // （有限太阳盘本就把真实焦散峰值摊到均值数倍，防单点全白）
  let ao = abs( dpdx( input.vOld ).x * dpdy( input.vOld ).y - dpdx( input.vOld ).y * dpdy( input.vOld ).x );
  let an = abs( dpdx( input.vNew ).x * dpdy( input.vNew ).y - dpdx( input.vNew ).y * dpdy( input.vNew ).x );
  let I = ao / max( an + ao * ( 1.0 / 8.0 ), 1e-9 );
  return vec4<f32>( ${g.rExpr}, ${g.gExpr}, 0.0, 1.0 );
}
`;
}

/** 细级联浅焦面 → R（tidewater fine 层 depths[0]）。 */
export const OCEAN_CAUSTICS_GEN_FINE_SHALLOW = genLayerShader({
  cascadeIdx: 3, slopeLevel: 1, tileSwizzle: 'x', depthSwizzle: 'x',
  rExpr: 'I', gExpr: '0.0', grid: 512, margin: 0.05,
});

/** 细级联深焦面 → G（tidewater fine 层 depths[1]）。 */
export const OCEAN_CAUSTICS_GEN_FINE_DEEP = genLayerShader({
  cascadeIdx: 3, slopeLevel: 1, tileSwizzle: 'x', depthSwizzle: 'y',
  rExpr: '0.0', gExpr: 'I', grid: 512, margin: 0.05,
});

/** 宽级联浅焦面 → R（基线 -1：查询期 mix(1,br,0.6) 的加性等价，均值保持 ~1）。 */
export const OCEAN_CAUSTICS_GEN_BROAD_SHALLOW = genLayerShader({
  cascadeIdx: 2, slopeLevel: 0.5, tileSwizzle: 'y', depthSwizzle: 'z',
  rExpr: '0.6 * ( I - 1.0 )', gExpr: '0.0', grid: 128, margin: 0.05,
});

/** 宽级联深焦面 → G（同上基线 -1）。 */
export const OCEAN_CAUSTICS_GEN_BROAD_DEEP = genLayerShader({
  cascadeIdx: 2, slopeLevel: 0.5, tileSwizzle: 'y', depthSwizzle: 'w',
  rExpr: '0.0', gExpr: '0.6 * ( I - 1.0 )', grid: 128, margin: 0.05,
});

// ── 沙洲接收面（CausticsFloor）──────────────────────────────────────────────

/**
 * CausticsFloor - 浅滩沙洲接收面 WGSL。
 *
 * 贴地经纬网格（海平面基准 ECEF + VS 内 upDir·(-depthM) 下移——米级位移
 * 与基准分别过 viewProj 再相加，f32 精度与 OceanFFTWater 同款）。着色 =
 * 沙色 × 焦散采样（双焦平面按沙面真实深度混合，长波斜率使焦散网摇摆），
 * 叠深水衰减蓝；半透明混合 + 距离雾 + 网格边缘 alpha 淡出免硬边。
 */
export const CAUSTICS_FLOOR_SHADER = /* wgsl */`
// 256B 相机 UBO — 布局单一来源：src/renderer/CameraUniforms.ts
struct CameraUniforms {
  viewMatrix      : mat4x4<f32>,   // @0   (64B)
  projMatrix      : mat4x4<f32>,   // @64  (64B)
  viewProjMatrix  : mat4x4<f32>,   // @128 (64B)
  cameraPosition  : vec4<f32>,     // @192 (16B)
  viewportSize    : vec2<f32>,     // @208 (8B)
  nearFar         : vec2<f32>,     // @224 (8B)
}

// 96B 沙洲参数
struct FloorParams {
  sunLocal : vec4<f32>,  // @0  指向太阳，局部 ENU（x 东 / y 上 / z 北）
  upDir    : vec4<f32>,  // @16 每米归一化 ECEF 增量（向上，含极轴压扁）
  spacing  : vec4<f32>,  // @32 x=焦散纹理覆盖(m) y=沙洲边长(m) z=沙面深度(m) w=级联0平铺(m)
  depths   : vec4<f32>,  // @48 x=浅焦面 D y=深焦面 D z=级联1平铺(m) w=基础 alpha
  sandCol  : vec4<f32>,  // @64 沙色 rgb
  skyCol   : vec4<f32>,  // @80 天空/雾色 rgb
}

@group(0) @binding(0) var<uniform> camera : CameraUniforms;
@group(0) @binding(1) var<uniform> floorParams : FloorParams;
@group(0) @binding(2) var causticsTex : texture_2d<f32>;
@group(0) @binding(3) var derivTex : texture_2d_array<f32>;
@group(0) @binding(4) var smp : sampler;

const EARTH_SCALE : f32 = 6378137.0;
const FOG_DIST : f32 = 12000.0;
// 球体遮挡半径²：与 OceanFFT 渲染一致（< 极点海面最小模长²，见其注释）
const GLOBE_OCCLUDE_R2 : f32 = 0.9945 * 0.9945;

struct VSIn {
  @location(0) ecef : vec3<f32>,    // 海平面基准点（归一化 ECEF）
  @location(1) planeXZ : vec2<f32>, // 局部平面坐标（米：x 东 / z 北）
}

struct VSOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0) worldPos : vec3<f32>,
  @location(1) planeXZ : vec2<f32>,
}

@vertex
fn vs_main( input : VSIn ) -> VSOut {
  // 沙面在海面下 depthM：米级下移与基准分别过 viewProj 再相加
  // （6m ≈ 9.4e-7 归一化单位，与 |ecef|~0.5 直接相加会掉进 f32 ulp）
  let drop = floorParams.upDir.xyz * ( - floorParams.spacing.z );
  let clipBase = camera.viewProjMatrix * vec4<f32>( input.ecef, 1.0 );
  let clipDisp = camera.viewProjMatrix * vec4<f32>( drop, 0.0 );
  var out : VSOut;
  out.clipPos = clipBase + clipDisp;
  out.worldPos = input.ecef + drop;
  out.planeXZ = input.planeXZ;
  return out;
}

@fragment
fn fs_main( input : VSOut ) -> @location(0) vec4<f32> {
  let camPos = camera.cameraPosition.xyz;

  // 球体遮挡（IPass 无深度附件：片元级射线-球求交，OceanFFT/FloodWater 同款）
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

  // 长波斜率（级联 0+1 涌浪）：焦散网随波摆动（tidewater causticsSample 的 slope 语义）
  let g0 = textureSampleLevel( derivTex, smp, input.planeXZ / floorParams.spacing.w, 0, 0.0 );
  let g1 = textureSampleLevel( derivTex, smp, input.planeXZ / floorParams.depths.z, 1, 0.0 );
  let slope = g0.xy + g1.xy;
  let n = normalize( vec3<f32>( - slope.x, 1.0, - slope.y ) );
  let Ls = refract( - floorParams.sunLocal.xyz, n, 1.0 / 1.333 );
  let tDown = max( - Ls.y, 0.15 );

  // 入水点回溯到焦散纹理（生成期已扣平折射偏移 T0，图案登记在入水点）
  let depthM = floorParams.spacing.z;
  let entry = input.planeXZ - Ls.xz * ( depthM / tDown );
  let uv = entry / floorParams.spacing.x + 0.5;
  let c = textureSampleLevel( causticsTex, smp, uv, 0.0 );

  // 双焦平面按沙面真实深度混合（tidewater wD 同款）
  let wD = clamp( ( depthM - floorParams.depths.x ) / max( floorParams.depths.y - floorParams.depths.x, 1e-4 ), 0.0, 1.0 );
  let caustic = max( mix( c.r, c.g, wD ), 0.0 );

  // 沙色 × 焦散 + 深水衰减蓝（红先被吸收，越深越偏蓝）
  var col = floorParams.sandCol.rgb * caustic;
  let deepK = clamp( depthM * 0.08, 0.0, 0.5 );
  col = mix( col, col * vec3<f32>( 0.45, 0.72, 1.0 ), deepK );
  col *= 0.35 + 0.75 * clamp( floorParams.sunLocal.y, 0.0, 1.0 );

  // 距离雾 + 沙洲边缘 alpha 淡出（免 400m 网格硬边）
  let distM = max( length( camPos - input.worldPos ), 1e-9 ) * EARTH_SCALE;
  let fog = 1.0 - exp( - distM / FOG_DIST );
  col = mix( col, floorParams.skyCol.rgb, fog * 0.9 );
  let b = max( abs( input.planeXZ.x ), abs( input.planeXZ.y ) ) / ( floorParams.spacing.y * 0.5 );
  let edge = 1.0 - smoothstep( 0.85, 1.0, b );
  let alpha = clamp( floorParams.depths.w * edge * ( 1.0 - fog * 0.85 ), 0.0, 1.0 );
  return vec4<f32>( col, alpha );
}
`;

/** CausticsParams uniform buffer 字节数（64B，见 struct 注释）。 */
export const CAUSTICS_PARAMS_UBO_BYTES = 64;
/** FloorParams uniform buffer 字节数（96B，见 struct 注释）。 */
export const FLOOR_PARAMS_UBO_BYTES = 96;
