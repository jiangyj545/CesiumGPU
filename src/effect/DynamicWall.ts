import type { IPass } from '../scene/passes/IPass';
import type { Camera } from '../scene/Camera';
import type { Context } from '../renderer/Context';
import { Cartesian3 } from '../math/Cartesian3';
import { GeometryPipeline } from '../geometry/GeometryPipeline';
import { DYNAMIC_WALL_SHADER } from '../shaders/DynamicWallShader';
import { CAMERA_UBO_SIZE, writeCameraUniforms } from '../renderer/CameraUniforms';

/** 与 PolylineGeometry 一致的退化段阈值（长度平方）。 */
const DEGENERATE_EPSILON_SQ = 1e-20;

export interface DynamicWallOptions {
  /** 底边路径点，归一化 ECEF（地表，Earth radius = 1）。至少 2 个点。 */
  positions: Cartesian3[];
  /**
   * 墙高，归一化单位（1 = 地球半径 ≈ 6378 km）。
   * Default: 0.02（≈ 127 km）
   */
  height?: number;
  /** 墙体颜色 RGBA。Default: `[0.25, 0.75, 1.0, 0.9]`（青蓝色） */
  color?: [number, number, number, number];
  /** 流速（光带上涌速度倍率）。Default: 1.0 */
  speed?: number;
  /** u 沿路径累积长度的重复倍数。Default: 3 */
  repeat?: number;
  /**
   * 相机引用（必填）：`IPass.execute` 不携带相机，Scene 也不向特效推送
   * 相机，因此墙体每帧自行从该相机读取 viewProj / camPos 上传 UBO。
   * 通常传 `viewer.scene.camera`。
   */
  camera: Camera;
}

interface WallGeometry {
  vertices: Float32Array;
  indices: Uint32Array;
  vertexCount: number;
  indexCount: number;
}

/**
 * DynamicWall - 动态墙特效（T3.2，对齐 EV-Globe 的 EV_DynamicWall）。
 *
 * 将地表路径挤出为垂直墙体（底排贴地 + 顶排沿地表法线抬高 `height`），
 * 片元级渲染"底部亮、向上渐隐、横向光带从底往上涌"。
 *
 * 作为 {@link IPass} 挂在 `scene.effects`（EffectManager）上，绘制在
 * Transparent composite 之后的 swapchain 上，混合为 src-alpha / one 的
 * 叠加发光；因 IPass 无深度附件，片元内用射线-单位球求交剔除被地球
 * 遮挡的远侧墙体。
 *
 * @example
 * ```typescript
 * const wall = scene.effects.add(new DynamicWall(scene.context, {
 *   positions: ringPositions,      // 归一化 ECEF（Earth radius = 1）
 *   height: 0.02,                  // ≈ 127 km
 *   color: [0.25, 0.75, 1.0, 0.9],
 *   speed: 0.5,
 *   repeat: 3,
 *   camera: scene.camera,
 * }));
 * wall.show = false; // 关闭
 * ```
 */
export class DynamicWall implements IPass {
  readonly label = 'DynamicWall';

  /** 可见性开关（EffectManager 每帧检查）。 */
  show = true;

  private _context: Context;
  private _camera: Camera;
  private _color: [number, number, number, number];
  private _speed: number;
  private _repeat: number;
  private readonly _geometry: WallGeometry;

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
  /** color(16) + speed/time/repeat/pad(16)。 */
  private static readonly PARAMS_UBO_BYTES = 32;

  constructor(context: Context, options: DynamicWallOptions) {
    this._context = context;
    this._camera = options.camera;
    this._color = options.color ?? [0.25, 0.75, 1.0, 0.9];
    this._speed = options.speed ?? 1.0;
    this._repeat = options.repeat ?? 3;
    this._geometry = DynamicWall._buildGeometry(
      options.positions,
      options.height ?? 0.02,
      this._repeat
    );
  }

  /** 流速倍率（实时可调，下一帧生效）。 */
  get speed(): number { return this._speed; }
  set speed(v: number) { this._speed = Math.max(0, v); }

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

    const module = this._context.createShaderModule(DYNAMIC_WALL_SHADER, 'DynamicWallShader');

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this._pipeline = device.createRenderPipeline({
      label: 'DynamicWallPipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: {
        module,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 32, // 8 floats × 4 bytes：[px,py,pz,nx,ny,nz,u,v]
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },  // position
            { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal（地表径向）
            { shaderLocation: 2, offset: 24, format: 'float32x2' }, // uv（u 沿路径，v 底 0 顶 1）
          ],
        }],
      },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [
          {
            format: this._targetFormat ?? this._context.preferredFormat,
            // 叠加发光：color = src.rgb * src.a + dst（premultiplied-ish additive）。
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' }, // 双面墙
      // 无深度附件：遮挡在片元内用射线-球求交处理。
    });

    this._cameraUBO = device.createBuffer({
      label: 'DynamicWallCameraUBO',
      size: DynamicWall.CAMERA_UBO_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this._paramsUBO = device.createBuffer({
      label: 'DynamicWallParamsUBO',
      size: DynamicWall.PARAMS_UBO_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const vertices = this._geometry.vertices;
    this._vertexBuffer = device.createBuffer({
      label: 'DynamicWallVertices',
      size: Math.max(4, vertices.byteLength),
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    if (vertices.byteLength > 0) {
      this._context.writeBuffer(this._vertexBuffer, 0, DynamicWall._asBytes(vertices));
    }

    const indices = this._geometry.indices;
    this._indexBuffer = device.createBuffer({
      label: 'DynamicWallIndices',
      size: Math.max(4, indices.byteLength),
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    if (indices.byteLength > 0) {
      this._context.writeBuffer(this._indexBuffer, 0, DynamicWall._asBytes(indices));
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
   * 以 `loadOp: 'load'` 叠加绘制墙体（不清除已合成的场景）。
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

    // ── Params UBO（32B）──────────────────────────────────────────────────
    const params = new Float32Array([
      this._color[0], this._color[1], this._color[2], this._color[3],
      this._speed, time, this._repeat, 0,
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
   * 底边路径 → 垂直墙体三角带：每段 4 顶点 2 三角
   * （角序 [底0, 顶0, 底1, 顶1]，顶点沿地表径向抬高 `height`）。
   * uv：u = 累积线长 × repeat；v 底 0 / 顶 1。
   */
  private static _buildGeometry(
    positions: Cartesian3[],
    height: number,
    repeat: number
  ): WallGeometry {
    const n = positions.length;
    if (n < 2) {
      return { vertices: new Float32Array(0), indices: new Uint32Array(0), vertexCount: 0, indexCount: 0 };
    }

    const maxSegments = n - 1;
    const pos = new Float32Array(maxSegments * 4 * 3);
    const nrm = new Float32Array(maxSegments * 4 * 3);
    const uvs = new Float32Array(maxSegments * 4 * 2);
    const idx = new Uint32Array(maxSegments * 6);

    let seg = 0;      // 有效段数（跳过退化段）
    let cum = 0;      // 累积线长（归一化单位）
    let pi = 0;
    let ti = 0;
    let ii = 0;

    for (let i = 0; i < n - 1; i++) {
      const p0 = positions[i];
      const p1 = positions[i + 1];
      const segLen = Cartesian3.distance(p0, p1);
      if (segLen * segLen < DEGENERATE_EPSILON_SQ) continue;

      const r0 = DynamicWall._radial(p0);
      const r1 = DynamicWall._radial(p1);
      const u0 = cum * repeat;
      const u1 = (cum + segLen) * repeat;
      cum += segLen;

      // 底排（贴地）与顶排（沿地表法线抬高 height）。
      const corners = [
        { p: p0, r: r0, u: u0, v: 0 },
        { p: DynamicWall._raise(p0, r0, height), r: r0, u: u0, v: 1 },
        { p: p1, r: r1, u: u1, v: 0 },
        { p: DynamicWall._raise(p1, r1, height), r: r1, u: u1, v: 1 },
      ];

      const base = seg * 4;
      for (let cIdx = 0; cIdx < 4; cIdx++) {
        const c = corners[cIdx];
        pos[pi] = c.p.x; pos[pi + 1] = c.p.y; pos[pi + 2] = c.p.z;
        nrm[pi] = c.r.x; nrm[pi + 1] = c.r.y; nrm[pi + 2] = c.r.z;
        uvs[ti] = c.u; uvs[ti + 1] = c.v;
        pi += 3;
        ti += 2;
      }
      idx[ii++] = base;
      idx[ii++] = base + 1;
      idx[ii++] = base + 2;
      idx[ii++] = base + 1;
      idx[ii++] = base + 3;
      idx[ii++] = base + 2;
      seg++;
    }

    if (seg === 0) {
      return { vertices: new Float32Array(0), indices: new Uint32Array(0), vertexCount: 0, indexCount: 0 };
    }

    const vertexCount = seg * 4;
    return {
      vertices: GeometryPipeline.interleave(
        pos.subarray(0, vertexCount * 3),
        nrm.subarray(0, vertexCount * 3),
        uvs.subarray(0, vertexCount * 2)
      ),
      indices: idx.subarray(0, seg * 6),
      vertexCount,
      indexCount: seg * 6,
    };
  }

  /** 单位地表径向（地表法线）方向。 */
  private static _radial(p: Cartesian3): Cartesian3 {
    const magSq = Cartesian3.magnitudeSquared(p);
    if (magSq < 1e-20) return Cartesian3.UNIT_Z.clone();
    return Cartesian3.multiplyByScalar(p, 1 / Math.sqrt(magSq), new Cartesian3());
  }

  /** 沿径向抬高：p + r × amount。 */
  private static _raise(p: Cartesian3, radial: Cartesian3, amount: number): Cartesian3 {
    return new Cartesian3(
      p.x + radial.x * amount,
      p.y + radial.y * amount,
      p.z + radial.z * amount
    );
  }

  /** 以 Uint8Array 视图暴露 typed array 的可拷贝字节（TS 5.7+ BufferSource 兼容）。 */
  private static _asBytes(src: ArrayBufferView): Uint8Array<ArrayBuffer> {
    return new Uint8Array(src.buffer as ArrayBuffer, src.byteOffset, src.byteLength);
  }
}

export default DynamicWall;
