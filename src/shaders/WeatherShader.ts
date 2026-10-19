/**
 * WeatherShader - Camera-anchored rain / snow particles (T3.5, EV_Rain / EV_Snow).
 *
 * Static per-instance point cloud animated entirely in the vertex stage:
 * each instance stores a random seed; the particle position is derived as
 *
 *   local = wmod(seed * boxSize + velocity * time + sway, boxSize) - boxSize/2
 *   world = boxOrigin + right*local.x + up*local.y + forward*local.z
 *
 * where boxOrigin is re-uploaded every frame as `camPos + forward * depth/2`,
 * so the volume follows the camera (world-space translation, no accumulation).
 *
 * Rain    - one instance = one elongated screen-facing quad stretched along the
 *           fall velocity (length = speed * 0.06 s), additive blend.
 * Snow    - one instance = one camera-facing soft round point (radial alpha),
 *           with a horizontal sinusoidal sway, standard alpha blend.
 *
 * Occlusion is resolved by the GPU depth test against the G-Buffer depth
 * (depthCompare 'less', depthWriteEnabled false) - particles behind terrain
 * fail the test and are never rasterised.
 *
 * Uniform layout (WeatherParams, 192 bytes):
 *   0   mat4x4 viewProj
 *   64  vec4  camPos          (normalised ECEF)
 *   80  vec4  right           camera basis (unit)
 *   96  vec4  up
 *   112 vec4  forward
 *   128 vec4  boxOrigin       world centre of the volume, in front of camera
 *   144 vec4  boxSize.xyz + time (s)
 *   160 vec4  motion = fallSpeed, swayAmp, streakLength, windSpeed (all normalised)
 *   176 vec4  style  = elementSize, alpha, -, -
 *
 * Instance attributes: loc 0 (per-vertex) quad corner vec2 in [-1,1];
 *                      loc 1 (per-instance) seed vec4 (xyz position seed, w phase).
 */
export const WEATHER_SHADER = /* wgsl */ `
struct WeatherParams {
  viewProj  : mat4x4<f32>,   // 0
  camPos    : vec4<f32>,     // 64
  right     : vec4<f32>,     // 80
  up        : vec4<f32>,     // 96
  forward   : vec4<f32>,     // 112
  boxOrigin : vec4<f32>,     // 128
  boxSize   : vec4<f32>,     // 144  xyz = size, w = time (s)
  motion    : vec4<f32>,     // 160  fallSpeed, swayAmp, streakLength, windSpeed
  style     : vec4<f32>,     // 176  elementSize, alpha, pad, pad
};

@group(0) @binding(0) var<uniform> P : WeatherParams;

struct VSOut {
  @builtin(position) pos    : vec4<f32>,
  @location(0)       corner : vec2<f32>,
  @location(1)       edge   : f32,   // min(局部盒内高度到上/下边界的归一距离)，0=贴边
};

// Rain tint (pale blue-white) and snow tint (near white).
const RAIN_COLOR = vec3<f32>(0.62, 0.72, 0.88);
const SNOW_COLOR = vec3<f32>(0.95, 0.97, 1.00);

/** Always-positive float modulo (wraps the particle inside the box). */
fn wmod(x : f32, y : f32) -> f32 {
  return x - y * floor(x / y);
}

/**
 * Axis perpendicular to "v" and roughly facing the camera, for building a
 * screen-facing quad around "v".  Falls back to "fallback" projected into the
 * view plane when the cross product degenerates (view parallel to v).
 */
fn screenAxis(v : vec3<f32>, toCam : vec3<f32>, fallback : vec3<f32>) -> vec3<f32> {
  let c = cross(v, toCam);
  let l = length(c);
  if (l > 1e-6) {
    return c / l;
  }
  let f = fallback - toCam * dot(fallback, toCam);
  let fl = length(f);
  return select(vec3<f32>(1.0, 0.0, 0.0), f / fl, fl > 1e-6);
}

/** Wrapped position inside the camera-anchored box, centred on boxOrigin. */
fn boxLocal(seed : vec3<f32>, phase : f32, sway : f32) -> vec3<f32> {
  let t    = P.boxSize.w;
  let size = P.boxSize.xyz;
  var off  = seed * size;
  off.y = wmod(off.y - t * P.motion.x, size.y);
  off.x = wmod(off.x + t * P.motion.w + sin(t * 1.7 + phase * 6.2831853) * sway, size.x);
  off.z = wmod(off.z + t * P.motion.w * 0.6 + cos(t * 1.3 + phase * 6.2831853) * sway, size.z);
  return off - size * 0.5;
}

fn toWorld(local : vec3<f32>) -> vec3<f32> {
  return P.boxOrigin.xyz
       + P.right.xyz   * local.x
       + P.up.xyz      * local.y
       + P.forward.xyz * local.z;
}

@vertex
fn vs_rain(
  @location(0) corner : vec2<f32>,
  @location(1) seed   : vec4<f32>,
) -> VSOut {
  let local  = boxLocal(seed.xyz, seed.w, 0.0);
  let world  = toWorld(local);

  // Fall velocity in world space (same recipe as the wrap offsets above).
  let vLocal = vec3<f32>(P.motion.w, -P.motion.x, P.motion.w * 0.6);
  var vel    = P.right.xyz * vLocal.x + P.up.xyz * vLocal.y + P.forward.xyz * vLocal.z;
  vel        = normalize(vel);

  let toCam  = normalize(P.camPos.xyz - world);
  let side   = screenAxis(vel, toCam, P.right.xyz);

  let halfLen = P.motion.z * 0.5;                    // streak length / 2
  let halfW   = P.style.x * 0.5;                     // streak thickness / 2
  let p = world + vel * (corner.y * halfLen) + side * (corner.x * halfW);

  var out : VSOut;
  out.pos    = P.viewProj * vec4<f32>(p, 1.0);
  out.corner = corner;
  // 顶/底 wrap 边界羽化系数（8% 高度带内淡入淡出，消除分层突变亮带）
  out.edge   = min(local.y / P.boxSize.y + 0.5, 0.5 - local.y / P.boxSize.y) * 2.0;
  return out;
}

@vertex
fn vs_snow(
  @location(0) corner : vec2<f32>,
  @location(1) seed   : vec4<f32>,
) -> VSOut {
  let local = boxLocal(seed.xyz, seed.w, P.motion.y);
  let world = toWorld(local);

  let toCam = normalize(P.camPos.xyz - world);
  let side  = screenAxis(cross(P.forward.xyz, toCam), toCam, P.right.xyz);
  let upAx  = cross(side, toCam);

  let half  = P.style.x * 0.5;                       // flake radius
  let p = world + (side * corner.x + upAx * corner.y) * half;

  var out : VSOut;
  out.pos    = P.viewProj * vec4<f32>(p, 1.0);
  out.corner = corner;
  out.edge   = min(local.y / P.boxSize.y + 0.5, 0.5 - local.y / P.boxSize.y) * 2.0;
  return out;
}

@fragment
fn fs_rain(@location(0) corner : vec2<f32>, @location(1) edge : f32) -> @location(0) vec4<f32> {
  // 条带尖端 + 横向边缘双羽化（硬矩形边不可见）。
  let fade = (1.0 - abs(corner.y) * 0.2) * (1.0 - abs(corner.x) * 0.3);
  // 盒边界羽化：wrap 瞬移的粒子淡入，消除水平分层亮带。
  let edgeFade = smoothstep(0.0, 0.08, edge);
  return vec4<f32>(RAIN_COLOR, P.style.y * fade * edgeFade);
}

@fragment
fn fs_snow(@location(0) corner : vec2<f32>, @location(1) edge : f32) -> @location(0) vec4<f32> {
  // Soft round flake: corner is quad-space [-1,1].
  let d     = length(corner);
  let alpha = (1.0 - smoothstep(0.55, 1.0, d)) * P.style.y * smoothstep(0.0, 0.08, edge);
  if (alpha <= 0.004) {
    discard;
  }
  return vec4<f32>(SNOW_COLOR, alpha);
}
`;

export default WEATHER_SHADER;
