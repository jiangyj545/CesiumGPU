/**
 * Pick shader — id-encoding fragment for GPU picking (spike).
 *
 * Reuses the G-Buffer uniform layout so existing Primitive bind groups remain
 * bind-group compatible (group 0 camera UBO, group 1 model UBO + texture +
 * sampler). The per-primitive id is supplied via an extra group 2 uniform.
 *
 * id encoding (24-bit):
 *   r = (id      ) & 0xFF / 255
 *   g = (id >>  8) & 0xFF / 255
 *   b = (id >> 16) & 0xFF / 255
 *   a = 255
 * id = 0 renders as (0,0,0,0) — the clear value means "background / no hit".
 */
export const PICK_SHADER = /* wgsl */`
struct CameraUniforms {
  viewMatrix      : mat4x4<f32>,
  projMatrix      : mat4x4<f32>,
  viewProjMatrix  : mat4x4<f32>,
  cameraPosition  : vec4<f32>,
  viewportSize    : vec2<f32>,
  nearFar         : vec2<f32>,
}

struct ModelUniforms {
  modelMatrix  : mat4x4<f32>,
  normalMatrix : mat4x4<f32>,
}

struct PickIdUniform {
  id : u32,
}

@group(0) @binding(0) var<uniform> camera : CameraUniforms;
@group(1) @binding(0) var<uniform> model  : ModelUniforms;
@group(2) @binding(0) var<uniform> pickId : PickIdUniform;

struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) normal   : vec3<f32>,
  @location(2) uv       : vec2<f32>,
}

struct VertexOutput {
  @builtin(position) clipPosition : vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  let worldPos = model.modelMatrix * vec4<f32>(in.position, 1.0);
  out.clipPosition = camera.viewProjMatrix * worldPos;
  return out;
}

fn encodeId(id: u32) -> vec4<f32> {
  return vec4<f32>(
    f32( id        & 0xFFu) / 255.0,
    f32((id >>  8) & 0xFFu) / 255.0,
    f32((id >> 16) & 0xFFu) / 255.0,
    1.0
  );
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
  return encodeId(pickId.id);
}
`;

/**
 * Id-encode precision test shader — draws a full-screen triangle that outputs
 * the encoded id of a uniform, used to verify a rgba8unorm round-trip is
 * lossless (fragment → render target → copyTextureToBuffer → decode).
 */
export const PICK_ID_TEST_SHADER = /* wgsl */`
struct PickIdUniform {
  id : u32,
}

@group(0) @binding(0) var<uniform> pickId : PickIdUniform;

struct VertexOutput {
  @builtin(position) clipPosition : vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var corners = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  var out: VertexOutput;
  out.clipPosition = vec4<f32>(corners[vertexIndex], 0.0, 1.0);
  return out;
}

fn encodeId(id: u32) -> vec4<f32> {
  return vec4<f32>(
    f32( id        & 0xFFu) / 255.0,
    f32((id >>  8) & 0xFFu) / 255.0,
    f32((id >> 16) & 0xFFu) / 255.0,
    1.0
  );
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
  return encodeId(pickId.id);
}
`;
