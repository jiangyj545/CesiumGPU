/**
 * OverlayShader - Screen-space billboards (entity points and text labels).
 *
 * Vertices are pre-projected on the CPU (world position → NDC → pixels) and
 * expressed in pixel coordinates; the vertex stage only converts pixels back
 * to NDC.  Occlusion is resolved in the fragment stage by loading the G-Buffer
 * depth (standard projection depth, 0..1) at the fragment's pixel and
 * discarding when the opaque scene is closer than the billboard.
 *
 * Vertex layout (36 bytes / vertex, one quad = 6 vertices):
 *   loc 0 - position : vec2<f32>  pixel coordinates (top-left origin)
 *   loc 1 - uv       : vec2<f32>  quad-space 0..1
 *   loc 2 - color    : vec4<f32>  tint colour (straight alpha)
 *   loc 3 - depth    : f32        projected clip z/w of the anchor position
 */
export const OVERLAY_SHADER = /* wgsl */ `
struct OverlayCamera {
  viewProj  : mat4x4<f32>,   // offset 0   (unused on CPU-projected quads, kept for layout parity with other passes)
  viewport  : vec2<f32>,     // offset 64  canvas size in pixels
  depthBias : f32,           // offset 72  occlusion bias (normalised depth)
  pad       : f32,           // offset 76
};

@group(0) @binding(0) var<uniform> camera : OverlayCamera;
@group(1) @binding(0) var gbufDepth : texture_2d<f32>;
@group(1) @binding(1) var labelTex  : texture_2d<f32>;
@group(1) @binding(2) var labelSamp : sampler;

struct VSOut {
  @builtin(position) position : vec4<f32>,
  @location(0) uv             : vec2<f32>,
  @location(1) color          : vec4<f32>,
  @location(2) anchorDepth    : f32,
};

@vertex
fn vs_main(
  @location(0) pixelPos : vec2<f32>,
  @location(1) uv       : vec2<f32>,
  @location(2) color    : vec4<f32>,
  @location(3) depth    : f32,
) -> VSOut {
  var out : VSOut;
  let ndc = vec2<f32>(
    pixelPos.x / camera.viewport.x * 2.0 - 1.0,
    1.0 - pixelPos.y / camera.viewport.y * 2.0
  );
  // Constant z across the quad: billboards are depth-tested (against the
  // G-Buffer) as a flat disc at the anchor depth, not per-fragment geometry.
  out.position     = vec4<f32>(ndc, depth, 1.0);
  out.uv           = uv;
  out.color        = color;
  out.anchorDepth  = depth;
  return out;
}

/**
 * Load the G-Buffer depth at this pixel; true when the opaque scene is
 * closer than the billboard anchor (billboard is occluded).
 */
fn isOccluded(fragCoord : vec4<f32>, anchorDepth : f32) -> bool {
  let coords = vec2<i32>(fragCoord.xy);
  let sceneDepth = textureLoad(gbufDepth, coords, 0).r;
  return sceneDepth < anchorDepth - camera.depthBias;
}

@fragment
fn fs_point(@builtin(position) fragCoord : vec4<f32>,
            @location(0) uv : vec2<f32>,
            @location(1) color : vec4<f32>,
            @location(2) anchorDepth : f32) -> @location(0) vec4<f32> {
  if (isOccluded(fragCoord, anchorDepth)) {
    discard;
  }

  // Round point: uv is quad-space 0..1, centre at 0.5.
  let d = distance(uv, vec2<f32>(0.5, 0.5)) * 2.0;
  let alpha = 1.0 - smoothstep(0.9, 1.0, d);
  if (alpha <= 0.004) {
    discard;
  }
  return vec4<f32>(color.rgb, color.a * alpha);
}

@fragment
fn fs_label(@builtin(position) fragCoord : vec4<f32>,
            @location(0) uv : vec2<f32>,
            @location(1) color : vec4<f32>,
            @location(2) anchorDepth : f32) -> @location(0) vec4<f32> {
  if (isOccluded(fragCoord, anchorDepth)) {
    discard;
  }

  // Text is baked white on transparent; tint with the entity fill colour.
  let texel = textureSample(labelTex, labelSamp, uv);
  let alpha = texel.a * color.a;
  if (alpha <= 0.004) {
    discard;
  }
  return vec4<f32>(color.rgb * texel.rgb, alpha);
}
`;

export default OVERLAY_SHADER;
