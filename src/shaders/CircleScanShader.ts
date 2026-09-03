/**
 * CircleScanShader - Expanding circular radar-scan effect (T3.3, EV_CircleScan).
 *
 * Drawn as an additive overlay (`one, one` blending) onto the swap-chain over
 * the already-composited scene. The geometry is a polar-grid disc projected
 * onto the ellipsoid surface; `u` (vertex attribute, location 1) is the
 * normalised radial distance 0 (centre) .. 1 (rim).
 *
 * Uniform layout (96 bytes, one uniform buffer):
 *   offset  0 : viewProj mat4x4<f32>
 *   offset 64 : color    vec4<f32>  (rgb tint + base alpha)
 *   offset 80 : params   vec4<f32>  (x = time s, y = period s, z = rings, w = unused)
 */
export const CIRCLE_SCAN_SHADER = /* wgsl */ `
struct ScanUniforms {
  viewProj : mat4x4<f32>,
  color    : vec4<f32>,
  params   : vec4<f32>,   // x = time (s), y = speed period (s), z = rings, w = unused
};

@group(0) @binding(0) var<uniform> scan : ScanUniforms;

struct VSOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0)       u       : f32,
};

@vertex
fn vs_main(
  @location(0) position : vec3<f32>,   // normalised ECEF surface point
  @location(1) u        : f32,         // radial distance 0..1
) -> VSOut {
  var out : VSOut;
  out.clipPos = scan.viewProj * vec4<f32>(position, 1.0);
  out.u       = u;
  return out;
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
  let time  = scan.params.x;
  let speed = max(scan.params.y, 0.001);
  let rings = max(scan.params.z, 1.0);
  let u     = clamp(in.u, 0.0, 1.0);

  // Expanding wavefront: one fract cell per ring, sweeping once per 'speed'
  // seconds. (WGSL smoothstep is undefined for edge0 >= edge1, so the band is
  // expressed as an inverted leading-edge ramp.)
  let r       = fract(u * rings - time / speed);
  let wave    = 1.0 - smoothstep(0.0, 0.15, r);   // bright band at the wavefront
  let falloff = 1.0 - u;                          // centre-outward attenuation
  let glow    = exp(-u * 8.0) * 0.5;              // constant bright centre

  let alpha = clamp(scan.color.a * (wave * falloff + glow), 0.0, 1.0);
  return vec4<f32>(scan.color.rgb * alpha, alpha);
}
`;

export default CIRCLE_SCAN_SHADER;
