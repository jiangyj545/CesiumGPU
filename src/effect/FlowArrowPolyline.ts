import type { IPass } from '../scene/passes/IPass';
import type { Context } from '../renderer/Context';
import type { Camera } from '../scene/Camera';
import { Cartesian3 } from '../math/Cartesian3';
import { FLOW_ARROW_SHADER } from '../shaders/FlowArrowShader';
import {
  buildFlowArrowGeometry,
  FLOW_ARROW_DEFAULTS,
  type FlowArrowPolylineMaterial,
} from './FlowArrowMath';

/** 顶点布局 [x,y,z, arcLen, v] = 5 floats = 20 B。 */
const FLOATS_PER_VERTEX = 5;
/** viewProj(16) + baseColor(4) + arrowColor(4) + params(4) = 28 floats。 */
const UNIFORM_BYTES = 28 * 4;

/**
 * FlowArrowPolyline - 流动箭头折线特效（P2-6，对齐 hexapod
 * flowArrowMaterial / flightManager·waypointEditor 的航线可视化用法）。
 *
 * 半透明底线 + 沿累计弧长滚动的「渐亮尾 + 三角头」箭头；相位 = GPU 侧
 * fract(arc·gap − t·speed)，几何只在 positions 引用变化时重建（配合 P0-A
 * CallbackProperty 动态 positions 逐帧 setPositions，引用不同即重建）。
 *
 * 通过 EffectManager 挂载（scene.effects.add(flowArrow)）——特效在挂载即
 * 满足 P2-5 按需渲染的持续脏条件；setCamera 提供投影矩阵。
 *
 * ```ts
 * const flow = new FlowArrowPolyline(viewer.context, {
 *   positions: [p0, p1, p2],          // 归一化 ECEF
 *   width: 6, speed: 0.35, gap: 24,
 * });
 * flow.setCamera(viewer.camera);
 * viewer.scene.effects.add(flow);
 * // 动态：flow.setPositions(newPositions) 每帧（引用变化触发重建）
 * ```
 */
export class FlowArrowPolyline implements IPass {
  readonly label = 'FlowArrowPolyline';

  /** 可见性开关（Effect 类型要求）。 */
  show = true;

  private _context: Context;
  private _camera: Camera | null = null;

  private _positions: readonly Cartesian3[] | null = null;
  private _halfWidth: number;
  private _baseColor: [number, number, number, number];
  private _arrowColor: [number, number, number, number];
  private _speed: number;
  private _gap: number;

  private _vertexBuffer: GPUBuffer | null = null;
  private _indexBuffer: GPUBuffer | null = null;
  private _uniformBuffer: GPUBuffer | null = null;
  private _pipeline: GPURenderPipeline | null = null;
  private _bindGroup: GPUBindGroup | null = null;
  private _indexCount = 0;
  private _destroyed = false;
  /** T5 真 HDR：目标格式（setTargetFormat 协议）。 */
  private _targetFormat: GPUTextureFormat | null = null;
  /** 几何重建计数（消费方判定「本帧是否变化」的脏源）。 */
  private _version = 0;
  private _uniformData = new Float32Array(UNIFORM_BYTES / 4);

  constructor(
    context: Context,
    options: {
      positions?: readonly Cartesian3[];
      /** 线宽（CSS 像素，×PIXELS_TO_WORLD 换算世界半宽）。 */
      width?: number;
      material?: FlowArrowPolylineMaterial;
    } = {}
  ) {
    this._context = context;
    const m: Partial<FlowArrowPolylineMaterial> = options.material ?? {};
    this._baseColor = m.color ?? FLOW_ARROW_DEFAULTS.color;
    this._arrowColor = m.arrowColor ?? FLOW_ARROW_DEFAULTS.arrowColor;
    this._speed = m.speed ?? FLOW_ARROW_DEFAULTS.speed;
    this._gap = m.gap ?? FLOW_ARROW_DEFAULTS.gap;
    this._halfWidth = ((options.width ?? 6) * 1e-5) / 2; // 与 EntityVisualizer PIXELS_TO_WORLD_WIDTH 同系数
    if (options.positions) this.setPositions(options.positions);
  }

  /** 几何重建计数（每次 positions 引用变化重建时 +1）。 */
  get version(): number {
    return this._version;
  }

  setCamera(camera: Camera): void {
    this._camera = camera;
  }

  /**
   * 更新折线（归一化 ECEF）。引用不同（或长度变化）才重建几何——
   * CallbackProperty 每帧新数组即逐帧重建；静态引用零开销。
   */
  setPositions(positions: readonly Cartesian3[]): void {
    if (this._destroyed) return;
    if (this._positions === positions) return;
    this._positions = positions;
    this._rebuildGeometry();
  }

  setTargetFormat(format: GPUTextureFormat): void {
    if (this._destroyed || format === (this._targetFormat ?? this._context.preferredFormat)) return;
    this._targetFormat = format;
    this._pipeline = null;
    this._bindGroup = null;
    this._vertexBuffer?.destroy();
    this._indexBuffer?.destroy();
    this._uniformBuffer?.destroy();
    this._vertexBuffer = null;
    this._indexBuffer = null;
    this._uniformBuffer = null;
    // 几何在 prepare 惰性重建（_positions 仍保留）。
    this._version++;
  }

  prepare(): void {
    if (!this._destroyed) this._prepare();
  }

  execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void {
    if (this._destroyed || !this.show) return;
    this._prepare();
    const pipeline = this._pipeline;
    if (!pipeline || !this._bindGroup || !this._vertexBuffer || !this._indexBuffer || !this._uniformBuffer) return;
    if (this._indexCount === 0) return;

    const d = this._uniformData;
    if (this._camera) d.set(this._camera.viewProjectionMatrix.values, 0);
    d[21] = this._speed;
    d[22] = this._gap;
    d[20] = time;
    this._context.writeBuffer(this._uniformBuffer, 0, d);

    const pass = encoder.beginRenderPass({
      label: this.label,
      colorAttachments: [{
        view: target,
        loadOp: 'load',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this._bindGroup);
    pass.setVertexBuffer(0, this._vertexBuffer);
    pass.setIndexBuffer(this._indexBuffer, 'uint32');
    pass.drawIndexed(this._indexCount);
    pass.end();
  }

  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._vertexBuffer?.destroy();
    this._indexBuffer?.destroy();
    this._uniformBuffer?.destroy();
    this._vertexBuffer = null;
    this._indexBuffer = null;
    this._uniformBuffer = null;
    this._pipeline = null;
    this._bindGroup = null;
  }

  // ── Internals ─────────────────────────────────────────────────────────────

  private _rebuildGeometry(): void {
    // 释放旧几何缓冲；pipeline/bindGroup 保留（与几何无关）。
    this._vertexBuffer?.destroy();
    this._indexBuffer?.destroy();
    this._vertexBuffer = null;
    this._indexBuffer = null;
    this._indexCount = 0;
    this._version++;
  }

  private _prepare(): void {
    const device = this._context.device;

    if (!this._uniformBuffer) {
      const d = this._uniformData;
      d[16] = this._baseColor[0];
      d[17] = this._baseColor[1];
      d[18] = this._baseColor[2];
      d[19] = this._baseColor[3];
      d[23] = this._arrowColor[0];
      d[24] = this._arrowColor[1];
      d[25] = this._arrowColor[2];
      d[26] = this._arrowColor[3];
      this._uniformBuffer = device.createBuffer({
        label: 'FlowArrow.Uniforms',
        size: UNIFORM_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }

    if (!this._vertexBuffer && this._positions) {
      const geo = buildFlowArrowGeometry(this._positions, this._halfWidth);
      if (geo) {
        const vb = device.createBuffer({
          label: 'FlowArrow.Vertices',
          size: geo.vertices.byteLength,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this._context.writeBuffer(vb, 0, new Float32Array(geo.vertices));
        this._vertexBuffer = vb;
        const ib = device.createBuffer({
          label: 'FlowArrow.Indices',
          size: geo.indices.byteLength,
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        this._context.writeBuffer(ib, 0, new Uint32Array(geo.indices));
        this._indexBuffer = ib;
        this._indexCount = geo.indices.length;
      }
    }

    if (!this._pipeline) {
      const module = this._context.createShaderModule(FLOW_ARROW_SHADER, 'FlowArrow.Module');
      const bgl = device.createBindGroupLayout({
        label: 'FlowArrow.BGL',
        entries: [{
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        }],
      });
      this._pipeline = device.createRenderPipeline({
        label: 'FlowArrow.Pipeline',
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        vertex: {
          module,
          entryPoint: 'vs_main',
          buffers: [{
            arrayStride: FLOATS_PER_VERTEX * 4,
            attributes: [
              { shaderLocation: 0, offset: 0,  format: 'float32x3' }, // position
              { shaderLocation: 1, offset: 12, format: 'float32' },   // arcLen
              { shaderLocation: 2, offset: 16, format: 'float32' },   // v
            ],
          }],
        },
        fragment: {
          module,
          entryPoint: 'fs_main',
          targets: [{
            format: this._targetFormat ?? this._context.preferredFormat,
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
      });
      this._bindGroup = device.createBindGroup({
        layout: bgl,
        entries: [{ binding: 0, resource: { buffer: this._uniformBuffer } }],
      });
    }
  }
}

export default FlowArrowPolyline;
