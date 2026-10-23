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
  /** 赤道半径（米）：归一化 ECEF → 米。 */
  private static readonly EARTH_RADIUS_M = 6378137.0;
  /** WGS84 极半径 / 赤道半径：椭球面 x²+y²+(z/ρ)²=1。 */
  private static readonly POLAR_RATIO = 6356752.3142 / 6378137.0;
  /** 球面细分目标边长（米）：20 km 弦垂距 ≈ 8 m，低空机位片元贴合海面。 */
  private static readonly GRID_EDGE_M = 20_000;
  /** 单片扇叶细分上限（96² ≈ 9.2k 三角形/扇叶，兼顾大域与建网格耗时）。 */
  private static readonly MAX_GRID_STEPS = 96;

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

  /** 水体基色 RGBA（实时可调，下一帧生效；a = 近景不透明度）。 */
  get color(): [number, number, number, number] { return this._color; }
  set color(v: [number, number, number, number]) { this._color = [v[0], v[1], v[2], v[3]]; }

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
   * 多边形扇形三角化（凸多边形）+ 球面自适应细分：顶点 0 为扇心，
   * (0, i, i+1) 为扇叶，每片扇叶按最长边细分成重心格点。
   *
   * 为什么必须细分：扇叶是**弦平面**，横跨 15°×20° 海域时平面中心比
   * 椭球面低 ~100 km。低空机位（数百米）下，片元世界坐标落在地球内部，
   * 射线-球遮挡把整面水判成"远侧"剔除（48 页 400 m 实测整屏无水）；
   * 即使不剔除，平面投影也只会压在画面最底部一条。细分后每个格点沿
   * 径向抬回椭球面（保留调用方给的统一水位偏移），20 km 弦垂距 ≈ 8 m。
   *
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

    // 统一水位偏移（环顶点相对椭球面的平均高度）：常规输入为 0；
    // 传入整体抬高的水体（如 +100 m 湖面）时保持其水位不变。
    let levelOffset = 0;
    for (let i = 0; i < n; i++) {
      levelOffset += Cartesian3.magnitude(positions[i]) - DynamicWater._surfaceRadius(positions[i]);
    }
    levelOffset /= n;

    const verts: number[] = [];
    const idx: number[] = [];
    const lift = (x: number, y: number, z: number): void => {
      const r = Math.hypot(x, y, z) || 1;
      const k = (DynamicWater._surfaceRadiusAt(x / r, y / r, z / r) + levelOffset) / r;
      verts.push(x * k, y * k, z * k, 0, 0);
    };

    // 先把环点抬到水面（输入已是地表点时 k≈1）
    const ring: Cartesian3[] = [];
    for (let i = 0; i < n; i++) {
      const p = positions[i];
      const r = Cartesian3.magnitude(p) || 1;
      const k = (DynamicWater._surfaceRadius(p) + levelOffset) / r;
      ring.push(new Cartesian3(p.x * k, p.y * k, p.z * k));
    }

    for (let t = 0; t < n - 2; t++) {
      const A = ring[0], B = ring[t + 1], C = ring[t + 2];
      const eMax = Math.max(
        Cartesian3.distance(A, B),
        Cartesian3.distance(B, C),
        Cartesian3.distance(C, A),
      ) * DynamicWater.EARTH_RADIUS_M;
      const steps = Math.min(
        DynamicWater.MAX_GRID_STEPS,
        Math.max(1, Math.ceil(eMax / DynamicWater.GRID_EDGE_M)),
      );
      const base = verts.length / 5;
      // 重心格点：行 iy 有 steps-iy+1 个点，沿弦平面线性插值后径向抬回球面
      for (let iy = 0; iy <= steps; iy++) {
        const c = iy / steps;
        for (let ix = 0; ix <= steps - iy; ix++) {
          const b = ix / steps;
          const a = 1 - b - c;
          lift(a * A.x + b * B.x + c * C.x,
               a * A.y + b * B.y + c * C.y,
               a * A.z + b * B.z + c * C.z);
        }
      }
      let rowStart = base;
      for (let iy = 0; iy < steps; iy++) {
        const rowLen = steps - iy + 1;
        const cells = steps - iy;
        for (let ix = 0; ix < cells; ix++) {
          const v0 = rowStart + ix;
          const v1 = v0 + 1;
          const v2 = rowStart + rowLen + ix;
          idx.push(v0, v1, v2);
          if (ix < cells - 1) idx.push(v1, v2 + 1, v2);
        }
        rowStart += rowLen;
      }
    }

    return {
      vertices: new Float32Array(verts),
      indices: new Uint32Array(idx),
      indexCount: idx.length,
    };
  }

  /**
   * WGS84 椭球面（赤道半径归一化 1、z 轴压扁 ρ）沿某方向的半径。
   * 与 `Camera.surfaceRadius` / `EllipsoidGeometry` 同一压扁系约定。
   */
  private static _surfaceRadiusAt(ux: number, uy: number, uz: number): number {
    const inv = 1 / DynamicWater.POLAR_RATIO;
    return 1 / Math.sqrt(ux * ux + uy * uy + uz * uz * inv * inv);
  }

  private static _surfaceRadius(p: Cartesian3): number {
    const r = Cartesian3.magnitude(p) || 1;
    return DynamicWater._surfaceRadiusAt(p.x / r, p.y / r, p.z / r);
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
