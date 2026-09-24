/**
 * 全屏三角形公共 WGSL（P2 后处理 pass 共用）。
 *
 * 与 LightingShader/LightingPass.execute 的 draw(3) 全屏三角一致：
 * 3 顶点覆盖 NDC 平面，uv 约定 pos.y=-1（NDC 底）→ uv.y=1（纹理顶）。
 * 所有后处理 FS 输出 vec4（alpha 恒 1）。
 */
export const FULLSCREEN_VS = /* wgsl */`
struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0)       uv  : vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  var uvs = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(2.0, 1.0),
    vec2<f32>(0.0, -1.0),
  );
  var out: VSOut;
  out.pos = vec4<f32>(positions[vid], 0.0, 1.0);
  out.uv  = uvs[vid];
  return out;
}
`;
