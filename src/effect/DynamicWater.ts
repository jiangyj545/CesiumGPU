import type { IPass } from '../scene/passes/IPass';
import type { Camera } from '../scene/Camera';
import type { Context } from '../renderer/Context';
import { Cartesian3 } from '../math/Cartesian3';
import { DYNAMIC_WATER_SHADER } from '../shaders/DynamicWaterShader';
import { CAMERA_UBO_SIZE, writeCameraUniforms } from '../renderer/CameraUniforms';

export interface DynamicWaterOptions {
  /**
   * 水面多边形外环顶点，归一化 ECEF（地表，Earth radius = 1）。
   * 按扇形三角化，需为凸多边形且至少 3 个点（顺/逆时针均可，cullMode none）。
   */
  positions: Cartesian3[];
  /** 水体基色 RGBA。Default: `[0.05, 0.22, 0.38, 0.85]`（深湖蓝） */
  color?: [number, number, number, number];
  /** 波速倍率。Default: 1.0 */
  speed?: number;
  /** 波高（法线扰动强度，0-1）。Default: 0.35 */
  waveScale?: number;
  /** 波光高光强度。Default: 0.9 */
  glint?: number;
  /** 太阳方向（指向太阳，归一化 ECEF），驱动镜面高光。Default: `[0.4, 0.5, 0.77]` */
  sunDirection?: [number, number, number];
  /**
   * 相机引用（必填）：`IPass.execute` 不携带相机，水面每帧自行从该相机
   * 读取 viewProj / camPos 上传 UBO。通常传 `viewer.scene.camera`。
   */
  camera: Camera;
}

/**
 * DynamicWater - 动态水面特效（对齐 EV-Globe 的 EV_DynamicWater / EV_Ocean）。
 *
 * 将地表凸多边形铺为半透明水体面，片元内以三层行波 + value noise 程序化
 * 扰动法线：波峰提亮、镜面高光（波光粼粼）、菲涅尔边缘增强。波纹频率以
 * 世界坐标为空间基准，湖面小域与海域大域视觉一致。
 *
 * 作为 {@link IPass} 挂在 `scene.effects`（EffectManager）上，绘制在
 * Transparent composite 之后的 swapchain 上（normal alpha blend 覆盖底图，
 * 高光叠加进 rgb）；因 IPass 无深度附件，片元内用射线-单位球求交剔除被
 * 地球遮挡的远侧水面。
 *
 * @example
 * ```typescript
 * const water = scene.effects.add(new DynamicWater(scene.context, {
 *   positions: ringPositions,       // 归一化 ECEF（Earth radius = 1）
 *   color: [0.05, 0.22, 0.38, 0.85],
 *   speed: 1.0,
 *   camera: scene.camera,
 * }));
 * water.show = false; // 关闭
 * ```
 */
export class DynamicWater implements IPass {
  readonly label = 'DynamicWater';

  /** 可见性开关（EffectManager 每帧检查）。 */
  show = true;

  private _context: Context;
  private _camera: Camera;
  private _color: [number, number, number, number];
  private _speed: number;
  private _waveScale: number;
  private _glint: number;
  private _sunDir: [number, number, number];
  private readonly _geometry: { vertices: Float32Array; indices: Uint32Array; indexCount: number };

  private _pipeline: GPURenderPipeline | null = null;
  private _bindGroup: GPUBindGroup | null = null;
  private _vertexBuffer: GPUBuffer | null = null;
  private _indexBuffer: GPUBuffer | null = null;
  private _cameraUBO: GPUBuffer | null = null;
  private _paramsUBO: GPUBuffer | null = null;
  /** T5 真 HDR：场景目标格式（null = 未下发跟随 preferredFormat 旧行为）。 */
  private _targetFormat: GPUTextureFormat | null = null;

  /** 与 Scene._uploadCameraUBO 相同的 256B 布局。 */
  private static readonly CAMERA_UBO_BYTES = CAMERA_UBO_SIZE;
  /** color(16) + sunDir(16) + time/speed/waveScale/glint(16) = 48B。 */
  private static readonly PARAMS_UBO_BYTES = 48;

  constructor(context: Context, options: DynamicWaterOptions) {
    this._context = context;
    this._camera = options.camera;
    this._color = options.color ?? [0.05, 0.22, 0.38, 0.85];
    this._speed = options.speed ?? 1.0;
    this._waveScale = options.waveScale ?? 0.35;
    this._glint = options.glint ?? 0.9;
    this._sunDir = DynamicWater._normalize(options.sunDirection ?? [0.4, 0.5, 0.77]);
    this._geometry = DynamicWater._buildGeometry(options.positions);
  }

  /** 波速倍率（实时可调，下一帧生效）。 */
  get speed(): number { return this._speed; }
  set speed(v: number) { this._speed = Math.max(0, v); }

  /** 波高（法线扰动强度 0-1，实时可调）。 */
  get waveScale(): number { return this._waveScale; }
  set waveScale(v: number) { this._waveScale = Math.min(1, Math.max(0, v)); }

  /** 波光高光强度（实时可调）。 */
  get glint(): number { return this._glint; }
  set glint(v: number) { this._glint = Math.max(0, v); }

  /**
   * T5 真 HDR：目标格式下发（EffectManager 可选协议，Scene.render 逐帧调用
   * 当前渲染目标格式）。变更时置空管线/缓冲由 prepare() 惰性守卫整体重建
   * （prepare 为单体式——须一并销毁旧缓冲防泄漏）；同格式幂等早退。未接线
   * 的调用路径保持 preferredFormat 旧行为。
   */
  setTargetFormat(format: GPUTextureFormat): void {
    if (format === (this._targetFormat ?? this._context.preferredFormat)) return;
    this._targetFormat = format;
    this._pipeline = null;
    this._bindGroup = null;
    this._cameraUBO?.destroy();
    this._paramsUBO?.destroy();
    this._vertexBuffer?.destroy();
    this._indexBuffer?.destroy();
    this._cameraUBO = null;
    this._paramsUBO = null;
    this._vertexBuffer = null;
    this._indexBuffer = null;
  }

  /** @inheritdoc IPass - 幂等初始化（管线 / 缓冲 / bindGroup）。 */
  prepare(): void {
    if (this._pipeline) return;
    const device = this._context.device;

    const module = this._context.createShaderModule(DYNAMIC_WATER_SHADER, 'DynamicWaterShader');

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this._pipeline = device.createRenderPipeline({
      label: 'DynamicWaterPipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: {
        module,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 20, // 5 floats × 4 bytes：[px,py,pz,u,v]（uv 预留）
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },  // position
            { shaderLocation: 1, offset: 12, format: 'float32x2' }, // uv（预留，当前置 0）
          ],
        }],
      },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [
          {
            format: this._targetFormat ?? this._context.preferredFormat,
            // 半透明覆盖底图；高光/菲涅尔加亮已并入 rgb，随 alpha 混合生效。
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      // 无深度附件：遮挡在片元内用射线-球求交处理。
    });

    this._cameraUBO = device.createBuffer({
      label: 'DynamicWaterCameraUBO',
      size: DynamicWater.CAMERA_UBO_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this._paramsUBO = device.createBuffer({
      label: 'DynamicWaterParamsUBO',
      size: DynamicWater.PARAMS_UBO_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const vertices = this._geometry.vertices;
    this._vertexBuffer = device.createBuffer({
      label: 'DynamicWaterVertices',
      size: Math.max(4, vertices.byteLength),
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    if (vertices.byteLength > 0) {
      this._context.writeBuffer(this._vertexBuffer, 0, DynamicWater._asBytes(vertices));
    }

    const indices = this._geometry.indices;
    this._indexBuffer = device.createBuffer({
      label: 'DynamicWaterIndices',
      size: Math.max(4, indices.byteLength),
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    if (indices.byteLength > 0) {
      this._context.writeBuffer(this._indexBuffer, 0, DynamicWater._asBytes(indices));
    }

    this._bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this._cameraUBO } },
        { binding: 1, resource: { buffer: this._paramsUBO } },
      ],
    });
  }

  /**
   * 每帧执行：上传相机 / 参数 UBO 后在 `target`（swapchain view）上
   * 以 `loadOp: 'load'` 叠加绘制水面（不清除已合成的场景）。
   */
  execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void {
    this.prepare();
    if (!this._pipeline || !this._bindGroup || this._geometry.indexCount === 0) return;

    // ── Camera UBO（布局单一来源：src/renderer/CameraUniforms.ts）─────────
    const cam = this._camera;
    const camData = new Float32Array(CAMERA_UBO_SIZE / 4);
    writeCameraUniforms(camData, {
      viewMatrix:           cam.viewMatrix.values,
      projectionMatrix:     cam.projectionMatrix.values,
      viewProjectionMatrix: cam.viewProjectionMatrix.values,
      position:             cam.position,
      viewportWidth:        this._context.drawingBufferWidth,
      viewportHeight:       this._context.drawingBufferHeight,
      near:                 cam.frustum.near,
      far:                  cam.frustum.far,
    });
    this._context.writeBuffer(this._cameraUBO!, 0, camData);

    // ── Params UBO（48B）─────────────────────────────────────────────────
    const params = new Float32Array([
      this._color[0], this._color[1], this._color[2], this._color[3],
      this._sunDir[0], this._sunDir[1], this._sunDir[2], 0,
      time, this._speed, this._waveScale, this._glint,
    ]);
    this._context.writeBuffer(this._paramsUBO!, 0, params);

    const pass = encoder.beginRenderPass({
      label: this.label,
      colorAttachments: [
        { view: target, loadOp: 'load', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
      ],
    });
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, this._bindGroup);
    pass.setVertexBuffer(0, this._vertexBuffer!);
    pass.setIndexBuffer(this._indexBuffer!, 'uint32');
    pass.drawIndexed(this._geometry.indexCount);
    pass.end();
  }

  /** 释放 GPU 资源；调用后本 pass 不可再用。 */
  destroy(): void {
    this._vertexBuffer?.destroy();
    this._indexBuffer?.destroy();
    this._cameraUBO?.destroy();
    this._paramsUBO?.destroy();
    this._vertexBuffer = null;
    this._indexBuffer = null;
    this._cameraUBO = null;
    this._paramsUBO = null;
    this._bindGroup = null;
    this._pipeline = null;
  }

  /**
   * 多边形扇形三角化（凸多边形）：顶点 0 为扇心，(0, i, i+1) 为扇叶。
   * 每顶点 [px,py,pz,u,v]，uv 置 0（波纹以世界坐标为空间基准）。
   */
  private static _buildGeometry(positions: Cartesian3[]): {
    vertices: Float32Array;
    indices: Uint32Array;
    indexCount: number;
  } {
    const n = positions.length;
    if (n < 3) {
      return { vertices: new Float32Array(0), indices: new Uint32Array(0), indexCount: 0 };
    }

    const vertices = new Float32Array(n * 5);
    for (let i = 0; i < n; i++) {
      const p = positions[i];
      vertices[i * 5 + 0] = p.x;
      vertices[i * 5 + 1] = p.y;
      vertices[i * 5 + 2] = p.z;
      // uv 置 0（预留）
    }

    const triCount = n - 2;
    const indices = new Uint32Array(triCount * 3);
    for (let i = 0; i < triCount; i++) {
      indices[i * 3 + 0] = 0;
      indices[i * 3 + 1] = i + 1;
      indices[i * 3 + 2] = i + 2;
    }

    return { vertices, indices, indexCount: triCount * 3 };
  }

  /** 归一化方向向量。 */
  private static _normalize(v: [number, number, number]): [number, number, number] {
    const m = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / m, v[1] / m, v[2] / m];
  }

  /** 以 Uint8Array 视图暴露 typed array 的可拷贝字节（TS 5.7+ BufferSource 兼容）。 */
  private static _asBytes(src: ArrayBufferView): Uint8Array<ArrayBuffer> {
    return new Uint8Array(src.buffer as ArrayBuffer, src.byteOffset, src.byteLength);
  }
}

export default DynamicWater;
