/**
 * CameraUniforms — 单一来源（single source of truth）：所有 pass 共享的
 * 256 字节相机 uniform buffer 布局。
 *
 * WGSL 侧：每个 shader 声明完全相同的 `CameraUniforms` struct：
 *
 *   struct CameraUniforms {
 *     viewMatrix     : mat4x4<f32>,   // @0   (64 B)
 *     projMatrix     : mat4x4<f32>,   // @64  (64 B)
 *     viewProjMatrix : mat4x4<f32>,   // @128 (64 B)
 *     cameraPosition : vec4<f32>,     // @192 (16 B) 世界空间相机位置，w 未用
 *     viewportSize   : vec2<f32>,     // @208 (8 B)
 *     nearFar        : vec2<f32>,     // @224 (8 B) x = near, y = far
 *   }
 *
 * 对齐说明（WGSL 规范 §14.4.5 RequiredAlignOf）：uniform 地址空间下
 * `vec2<f32>` 要求 16 字节对齐（除非 shader 显式
 * `enable uniform_buffer_standard_layout`）。因此 viewportSize 落在
 * @208（13×16），nearFar 被推到 @224 —— 而**不是**紧跟其后的 @216。
 * struct 实际大小 232 B，按 struct 对齐（64 B）向上取整为 256 B 整。
 *
 * 修改布局时：先改此文件与各 shader 的 struct（保持一致），再运行渲染验证。
 */

/** 共享相机 uniform buffer 的字节大小。 */
export const CAMERA_UBO_SIZE = 336;

/** 相机 UBO 各成员的字节偏移（必须与各 shader 的 CameraUniforms struct 一致）。 */
export const CAMERA_UBO_OFFSETS = {
  view: 0,
  proj: 64,
  viewProj: 128,
  position: 192,
  viewportSize: 208,
  nearFar: 224,
  // RTC（相对相机渲染，encoded camera）尾段：仅 GBuffer/Pick 深度精确路径
  // 消费；旧 shader 的 CameraUniforms struct 未声明这些字段——uniform 绑定
  // 允许 buffer 大于 struct，缺省写入 0 即可（RTC 图元仅 Scene 主相机启用）。
  viewRotProj: 240,
  camHi: 304,
  camLo: 320,
} as const;

/** 每帧写入相机 UBO 的字段值。矩阵为 16 元素列主序 Float32Array。 */
export interface CameraUniformValues {
  viewMatrix: Float32Array;
  projectionMatrix: Float32Array;
  viewProjectionMatrix: Float32Array;
  position: { x: number; y: number; z: number };
  viewportWidth: number;
  viewportHeight: number;
  near: number;
  far: number;
  /** RTC 深度路径：proj·viewRot（view 平移列置零，f64 预乘）。缺省写 0。 */
  viewRotProjMatrix?: Float32Array;
  /** RTC：相机位置 double-single 高/低位（f64 → fround 拆分）。缺省写 0。 */
  positionHi?: { x: number; y: number; z: number };
  positionLo?: { x: number; y: number; z: number };
}

/**
 * 将完整相机 UBO 载荷（256 B = 64 floats）写入 `data`。
 *
 * 调用方分配（或复用）Float32Array，连同本帧相机值一并传入；
 * 布局细节只存在于本模块，禁止在调用方手写偏移。
 */
export function writeCameraUniforms(data: Float32Array, camera: CameraUniformValues): void {
  data.set(camera.viewMatrix,           CAMERA_UBO_OFFSETS.view     >> 2);
  data.set(camera.projectionMatrix,     CAMERA_UBO_OFFSETS.proj     >> 2);
  data.set(camera.viewProjectionMatrix, CAMERA_UBO_OFFSETS.viewProj >> 2);

  const p = CAMERA_UBO_OFFSETS.position >> 2;
  data[p]     = camera.position.x;
  data[p + 1] = camera.position.y;
  data[p + 2] = camera.position.z;
  data[p + 3] = 1.0;

  const vp = CAMERA_UBO_OFFSETS.viewportSize >> 2;
  data[vp]     = camera.viewportWidth;
  data[vp + 1] = camera.viewportHeight;

  const nf = CAMERA_UBO_OFFSETS.nearFar >> 2; // = 56，见上方对齐说明
  data[nf]     = camera.near;
  data[nf + 1] = camera.far;

  // RTC 尾段（缺省 0）：f32 相对量在 ~1e-3 量级，深度绝对误差 ~亚毫米
  if (camera.viewRotProjMatrix) {
    data.set(camera.viewRotProjMatrix, CAMERA_UBO_OFFSETS.viewRotProj >> 2);
  }
  if (camera.positionHi) {
    const hi = CAMERA_UBO_OFFSETS.camHi >> 2;
    data[hi] = camera.positionHi.x;
    data[hi + 1] = camera.positionHi.y;
    data[hi + 2] = camera.positionHi.z;
    data[hi + 3] = 0.0;
  }
  if (camera.positionLo) {
    const lo = CAMERA_UBO_OFFSETS.camLo >> 2;
    data[lo] = camera.positionLo.x;
    data[lo + 1] = camera.positionLo.y;
    data[lo + 2] = camera.positionLo.z;
    data[lo + 3] = 0.0;
  }
}
