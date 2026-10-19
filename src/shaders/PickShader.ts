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
// 336B camera UBO — layout single source of truth: src/renderer/CameraUniforms.ts
// (uniform vec2 aligns to 16 → viewportSize @208, nearFar @224 — NOT @216)
struct CameraUniforms {
  viewMatrix      : mat4x4<f32>,   // @0   (64B)
  projMatrix      : mat4x4<f32>,   // @64  (64B)
  viewProjMatrix  : mat4x4<f32>,   // @128 (64B)
  cameraPosition  : vec4<f32>,     // @192 (16B)
  viewportSize    : vec2<f32>,     // @208 (8B)
  nearFar         : vec2<f32>,     // @224 (8B) x = near, y = far
  viewRotProj     : mat4x4<f32>,   // @240 (64B) RTC: proj·viewRot（view 平移置零）
  camHi           : vec4<f32>,     // @304 (16B) RTC: 相机位置 double-single 高位
  camLo           : vec4<f32>,     // @320 (16B) RTC: 低位
}

struct ModelUniforms {
  modelMatrix  : mat4x4<f32>,
  normalMatrix : mat4x4<f32>,
  originHi     : vec4<f32>,    // @128 RTC: 几何中心高位，w=flag(1=RTC)
  originLo     : vec4<f32>,    // @144 RTC: 中心低位；顶点 = 相对中心偏移
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

// RTC（相对相机渲染）裁剪位置：与 GBufferShader.vs_main 同款 hi/lo 差分，
// 保证 pick id 平面与深度平面逐像素同面（seg↔depth 配对不破）。
fn rtcClipPos(posRel: vec3<f32>) -> vec4<f32> {
  let dHi = model.originHi.xyz - camera.camHi.xyz;
  let dLo = model.originLo.xyz - camera.camLo.xyz;
  return camera.viewRotProj * vec4<f32>(posRel + dHi + dLo, 1.0);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  let worldPos = model.modelMatrix * vec4<f32>(in.position, 1.0);
  out.clipPosition = select(
    camera.viewProjMatrix * worldPos,
    rtcClipPos(in.position),
    model.originHi.w > 0.5
  );
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

// ── F12 实例级拾取（per-vertex instance id override）─────────────────────────
// 第二顶点缓冲（stride 4, format 'uint32'，shaderLocation 3）携带每顶点实例
// id（CityBlockBuilder 每建筑唯一序号，23-bit）。命中实例的像素输出
// "INSTANCE_FLAG | instanceId" —— bit23 置位把实例 id 与 primitive pickId
// 空间隔离（消费侧 SensorCam / bbox-from-seg 以 INSTANCE_FLAG 判别实例
// 像素；pickId 分配器为 24-bit 递增值，与 bit23 置位空间的碰撞仅在 >8M
// 图元的场景理论可达，实际不出现）。instanceId == 0（未赋值顶点）回退
// primitive pickId，保证部分顶点缺失 id 时拾取语义仍正确。

struct InstancedVertexInput {
  @location(0) position   : vec3<f32>,
  @location(1) normal     : vec3<f32>,
  @location(2) uv         : vec2<f32>,
  @location(3) instanceId : u32,
}

struct InstancedVertexOutput {
  @builtin(position) clipPosition : vec4<f32>,
  // WGSL 规则：整型 user-defined 顶点输出必须显式 @interpolate(flat)，
  // 否则整个 ShaderModule 解析失败（2026-09-22 headless 实测：缺此属性时
  // vs_main/vs_instanced 一并 invalid → pick id 平面全 0）。
  @location(0) @interpolate(flat) instanceId : u32,
}

@vertex
fn vs_instanced(in: InstancedVertexInput) -> InstancedVertexOutput {
  var out: InstancedVertexOutput;
  let worldPos = model.modelMatrix * vec4<f32>(in.position, 1.0);
  out.clipPosition = select(
    camera.viewProjMatrix * worldPos,
    rtcClipPos(in.position),
    model.originHi.w > 0.5
  );
  out.instanceId = in.instanceId;
  return out;
}

const INSTANCE_FLAG: u32 = 0x800000u;

@fragment
fn fs_instanced(in: InstancedVertexOutput) -> @location(0) vec4<f32> {
  let id = select(INSTANCE_FLAG | in.instanceId, pickId.id, in.instanceId == 0u);
  return encodeId(id);
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
