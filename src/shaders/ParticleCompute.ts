/**
 * Particle system shaders (GPU compute simulation, T3.4 production version).
 *
 * Particle state lives in a single storage buffer updated in-place by the
 * compute pass (particles are fully independent, so no intra-workgroup
 * synchronisation is required) and read back in the vertex stage of the
 * billboard render pass.
 *
 * State layout — 32 bytes per particle:
 *   offset  0 : pos  vec3<f32>  (normalised ECEF, Earth radius = 1)
 *   offset 12 : age  f32        (seconds since (re)spawn)
 *   offset 16 : vel  vec3<f32>  (normalised units / s)
 *   offset 28 : life f32        (total lifetime in seconds, randomised per
 *                               respawn within the preset's lifeRange)
 */

// ── Compute: integrate + respawn ─────────────────────────────────────────────

export const PARTICLE_COMPUTE_SHADER = /* wgsl */`
struct Particle {
  pos  : vec3<f32>,
  age  : f32,
  vel  : vec3<f32>,
  life : f32,
}

struct SimParams {
  origin  : vec4<f32>,  // xyz = emitter position (normalised ECEF), w = dt (s)
  config  : vec4<f32>,  // x = speedMin, y = speedMax, z = coneHalfAngle (rad), w = gravity
  misc    : vec4<f32>,  // x = particleCount, y = frameSeed, z = activeCount, w = simTime
  emitter : vec4<f32>,  // x = shape (0 = cone, 1 = sphere), y = lifeMin (s), z = lifeMax (s), w = unused
}

@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(1) var<uniform> params : SimParams;

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
  if (idx >= u32(params.misc.x)) { return; }

  var p = particles[idx];
  let dt = params.origin.w;

  if (p.age >= p.life) {
    if (idx >= u32(params.misc.z)) {
      // ── Parked: beyond the emissionRate budget. Stays dead until the ────
      // active count is raised again (keeps the pool allocation fixed).
      p.age = p.life;
    } else {
      // ── Respawn at the emitter ──────────────────────────────────────────
      let up = normalize(params.origin.xyz);
      let seed = u32(params.misc.y);
      let h0 = rand01(idx * 9781u + seed);
      let h1 = rand01(idx * 6271u + seed + 1u);
      let h2 = rand01(idx * 3253u + seed + 2u);
      let h3 = rand01(idx * 7919u + seed + 3u);

      var dir: vec3<f32>;
      if (u32(params.emitter.x) == 1u) {
        // Spherical burst (fireworks): uniform direction on the unit sphere.
        let z = h0 * 2.0 - 1.0;
        let a = h1 * 6.2831853;
        let r = sqrt(max(0.0, 1.0 - z * z));
        dir = vec3<f32>(r * cos(a), r * sin(a), z);
      } else {
        // Cone around the local surface normal (fire / fountain / smoke).
        let helper = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(up.z) > 0.9);
        let t1 = normalize(cross(up, helper));
        let t2 = cross(up, t1);

        let theta = h0 * 6.2831853;
        let cosA  = mix(1.0, cos(params.config.z), h1);
        let sinA  = sqrt(max(0.0, 1.0 - cosA * cosA));
        dir = up * cosA + (t1 * cos(theta) + t2 * sin(theta)) * sinA;
      }

      let speed = params.config.x + h2 * (params.config.y - params.config.x);
      p.vel  = dir * speed;
      p.pos  = params.origin.xyz;
      p.age  = 0.0;
      p.life = mix(params.emitter.y, params.emitter.z, h3);
    }
  } else {
    // ── Integrate: gravity toward Earth centre + linear velocity ────────
    let g = -normalize(params.origin.xyz) * params.config.w;
    p.vel = p.vel + g * dt;
    p.pos = p.pos + p.vel * dt;
    p.age = p.age + dt;
  }

  particles[idx] = p;
}
`;

// ── Render: depth-tested instanced screen-aligned billboards ─────────────────

export const PARTICLE_RENDER_SHADER = /* wgsl */`
struct Particle {
  pos  : vec3<f32>,
  age  : f32,
  vel  : vec3<f32>,
  life : f32,
}

struct DrawParams {
  viewProj : mat4x4<f32>,  // view-projection matrix
  misc     : vec4<f32>,    // x = clip scale X base (1/(tanHalfFov*aspect)),
                            // y = clip scale Y base (1/tanHalfFov),
                            // z = colorMode (0 fire / 1 fountain / 2 fireworks / 3 smoke)
  misc2    : vec4<f32>,    // x = sizeMin (m), y = sizeMax (m), z = EARTH_SCALE (m / world unit)
}

@group(0) @binding(0) var<uniform> drawParams : DrawParams;
@group(0) @binding(1) var<storage, read> particles : array<Particle>;

// PCG hash — stable per-slot randomness (size / hue); no time seed here so a
// particle keeps its size and hue across frames.
fn pcgHash(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand01(x: u32) -> f32 {
  return f32(pcgHash(x)) * (1.0 / 4294967296.0);
}

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
  let p = particles[instance];
  let mode = u32(drawParams.misc.z);

  // Stable per-slot randomness.
  let hs = rand01(instance * 2654435761u + 7u);   // size factor
  let hh = rand01(instance * 40503u    + 13u);    // hue factor (fireworks)

  let t = clamp(p.age / max(p.life, 1e-4), 0.0, 1.0);

  // Billboard size: random in [sizeMin, sizeMax] metres, converted to
  // normalised world units, then shaped per mode.
  var sizeNorm = mix(drawParams.misc2.x, drawParams.misc2.y, hs) / drawParams.misc2.z;
  var col: vec3<f32>;
  var alpha: f32;

  if (mode == 0u) {
    // Fire: white-yellow -> orange -> red over lifetime.
    let c0 = vec3<f32>(1.0, 0.95, 0.55);
    let c1 = vec3<f32>(1.0, 0.45, 0.05);
    let c2 = vec3<f32>(0.55, 0.05, 0.02);
    col   = mix(c0, c1, smoothstep(0.0, 0.4, t));
    col   = mix(col, c2, smoothstep(0.4, 1.0, t));
    alpha = (1.0 - t) * (1.0 - t);
  } else if (mode == 1u) {
    // Fountain: white core -> water blue.
    let c0 = vec3<f32>(0.85, 0.97, 1.0);
    let c1 = vec3<f32>(0.25, 0.60, 0.95);
    col   = mix(c0, c1, smoothstep(0.0, 0.8, t));
    alpha = (1.0 - t) * (1.0 - t);
  } else if (mode == 2u) {
    // Fireworks: saturated pseudo-random hue per particle, brightness fades
    // and the billboard shrinks with age (trailing feel).
    col   = (0.55 + 0.45 * cos(6.2831853 * (vec3<f32>(0.0, 0.33, 0.67) + hh))) * (1.0 - 0.4 * t);
    alpha = (1.0 - t) * (1.0 - t);
    sizeNorm *= 1.0 - 0.85 * t;
  } else {
    // Smoke: soft grey, low alpha, fades in then out; billboards expand.
    let g = mix(0.50, 0.28, t);
    col   = vec3<f32>(g);
    alpha = 0.22 * min(t * 5.0, 1.0) * (1.0 - t);
    sizeNorm *= 0.7 + 0.6 * t;
  }

  var out: VertexOutput;

  // Perspective-correct screen-aligned expansion: a clip-space offset K
  // becomes an NDC offset of K / clip.w, i.e. worldSize / (dist * tan(fov/2)).
  let base = drawParams.viewProj * vec4<f32>(p.pos, 1.0);
  out.clipPos = vec4<f32>(
    base.xy + in.corner * drawParams.misc.xy * sizeNorm * base.w,
    base.z,   // particle-centre depth for every fragment -> depth test vs G-Buffer
    base.w,
  );
  out.uv    = in.corner;
  out.color = vec4<f32>(col, alpha);

  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let d = length(in.uv);
  if (d > 1.0) { discard; }
  let soft = smoothstep(1.0, 0.35, d);
  return vec4<f32>(in.color.rgb, in.color.a * soft);
}
`;
