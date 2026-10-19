import type { IPass } from '../scene/passes/IPass';
import type { Context } from '../renderer/Context';
import type { Camera } from '../scene/Camera';
import { Cartesian3 } from '../math/Cartesian3';
import { FLOW_ARROW_SHADER } from '../shaders/FlowArrowShader';
import {
  buildFlowArrowGeometry,
  effectiveFlowArrowGap,
  FLOW_ARROW_DEFAULTS,
  type FlowArrowPolylineMaterial,
} from './FlowArrowMath';

/** 顶点布局 [x,y,z, arc, v, side.xyz] = 8 floats = 32 B。 */
const FLOATS_PER_VERTEX = 8;
/** viewProj(16) + baseColor(4) + arrowColor(4) + params(4) + viewport(4) = 32 floats。 */
const UNIFORM_BYTES = 32 * 4;

/**
 * FlowArrowPolyline - 流动箭头折线特效（P2-6，对齐 hexapod
 * flowArrowMaterial / flightManager·waypointEditor 的航线可视化用法）。
 *
 * 半透明底线 + 沿累计弧长滚动的「渐亮尾 + 三角头」箭头；相位 = GPU 侧
 * fract(arc·gap − t·speed)，几何只在 positions 引用变化时重建（配合 P0-A
 * CallbackProperty 动态 positions 逐帧 setPositions，引用不同即重建）。
 *
 * - **width = 恒定屏幕 CSS 像素**（CesiumJS polyline 语义）：顶点着色器
 *   按透视正确的屏幕空间展开（halfWidthDevicePx = width/2·dpr）。
 * - **箭头密度**：material.repeatCount（hexapod 语义，整条线箭头数，默认
 *   自适应 ~每 100 m 一支）在几何重建时经 effectiveFlowArrowGap 换算为
 *   gap；material.gap 为高级覆盖（优先级更高）。
 * - **深度遮挡**：binding 1 采样 GBuffer 场景深度（depthView provider，
 *   惰性解析；无 provider 时退化为自建 1×1 清 1 dummy，等效不遮挡）。
 *
 * 通过 EffectManager 挂载（scene.effects.add(flowArrow)）——特效在挂载即
 * 满足 P2-5 按需渲染的持续脏条件；setCamera 提供投影矩阵。
 *
 * ```ts
 * const flow = new FlowArrowPolyline(viewer.context, {
 *   positions: [p0, p1, p2],       // 归一化 ECEF
 *   width: 6,                      // CSS 像素（屏幕恒定宽）
 *   material: { type: 'flow', repeatCount: 30 },
 *   depthView: () => scene.sceneDepthView(),
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
  /** 线宽（CSS 像素；设备像素半宽在 execute 按当前 dpr 换算）。 */
  private _width: number;
  /** 材质引用（几何重建时刷新自适应箭头密度）。 */
  private _material: Partial<FlowArrowPolylineMaterial>;
  private _baseColor: [number, number, number, number];
  private _arrowColor: [number, number, number, number];
  private _speed: number;
  private _gap: number;

  /** 惰性场景深度视图 provider（null → 自建 dummy，等效不遮挡）。 */
  private _depthViewProvider: (() => GPUTextureView | null) | null;

  private _vertexBuffer: GPUBuffer | null = null;
  private _indexBuffer: GPUBuffer | null = null;
  private _uniformBuffer: GPUBuffer | null = null;
  private _pipeline: GPURenderPipeline | null = null;
  private _bindGroupLayout: GPUBindGroupLayout | null = null;
  private _bindGroup: GPUBindGroup | null = null;
  /** bindGroup 当前绑定的深度视图（视图变化仅重建 bindGroup，pipeline 复用）。 */
  private _bindGroupDepthView: GPUTextureView | null = null;
  private _depthSampler: GPUSampler | null = null;
  /** 无 provider 时的 1×1 depth32float dummy（清 1 → 永不遮挡）。 */
  private _dummyDepth: GPUTexture | null = null;
  private _dummyDepthView: GPUTextureView | null = null;
  private _dummyDepthCleared = false;
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
      /** 线宽（CSS 像素，屏幕空间恒定——CesiumJS polyline 语义）。 */
      width?: number;
      material?: FlowArrowPolylineMaterial;
      /** 场景深度视图惰性 provider（GBuffer 深度，遮挡用）。 */
      depthView?: () => GPUTextureView | null;
    } = {}
  ) {
    this._context = context;
    const m: Partial<FlowArrowPolylineMaterial> = options.material ?? {};
    this._material = m;
    this._baseColor = m.color ?? FLOW_ARROW_DEFAULTS.color;
    this._arrowColor = m.arrowColor ?? FLOW_ARROW_DEFAULTS.arrowColor;
    this._speed = m.speed ?? FLOW_ARROW_DEFAULTS.speed;
    // 自适应密度在几何重建时刷新；未重建前的安全值（effectiveFlowArrowGap
    // 对 totalArcNorm ≤ 0 同样返回 1）。
    this._gap = m.gap ?? 1;
    this._width = options.width ?? 6;
    this._depthViewProvider = options.depthView ?? null;
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
    this._bindGroupLayout = null;
    this._bindGroup = null;
    this._bindGroupDepthView = null;
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
    if (!pipeline || !this._bindGroupLayout || !this._vertexBuffer || !this._indexBuffer || !this._uniformBuffer) return;
    if (this._indexCount === 0) return;

    const d = this._uniformData;
    if (this._camera) d.set(this._camera.viewProjectionMatrix.values, 0);
    const canvas = this._context.canvas;
    const dpr = canvas.width / Math.max(canvas.clientWidth, 1);
    d[24] = time;
    d[25] = this._speed;
    d[26] = this._gap;
    d[27] = (this._width / 2) * dpr; // halfWidthDevicePx
    d[28] = Math.max(1, canvas.width);
    d[29] = Math.max(1, canvas.height);
    this._context.writeBuffer(this._uniformBuffer, 0, d);

    // 深度遮挡：惰性解析当前视图（GBuffer resize 会换视图），变化仅重建
    // bindGroup——pipeline 不含深度信息可复用。
    const depthView = this._resolveDepthView(encoder);
    if (!this._bindGroup || this._bindGroupDepthView !== depthView) {
      if (!this._depthSampler) {
        // depth32float 不可过滤 → non-filtering（nearest/nearest）。
        this._depthSampler = this._context.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });
      }
      this._bindGroup = this._context.device.createBindGroup({
        label: 'FlowArrow.BindGroup',
        layout: this._bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this._uniformBuffer } },
          { binding: 1, resource: depthView },
          { binding: 2, resource: this._depthSampler },
        ],
      });
      this._bindGroupDepthView = depthView;
    }

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
    this._dummyDepth?.destroy();
    this._dummyDepth = null;
    this._dummyDepthView = null;
    this._pipeline = null;
    this._bindGroupLayout = null;
    this._bindGroup = null;
    this._bindGroupDepthView = null;
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

  /** 解析遮挡用深度视图：provider 优先；null → 1×1 清 1 dummy（不遮挡）。 */
  private _resolveDepthView(encoder: GPUCommandEncoder): GPUTextureView {
    const provided = this._depthViewProvider?.() ?? null;
    if (provided) return provided;

    if (!this._dummyDepth) {
      this._dummyDepth = this._context.device.createTexture({
        label: 'FlowArrow.DummyDepth',
        size: { width: 1, height: 1 },
        format: 'depth32float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this._dummyDepthView = this._dummyDepth.createView();
      this._dummyDepthCleared = false;
    }
    if (!this._dummyDepthCleared) {
      // 首次需要时在当前 encoder 上用一个只有 depthAttachment 的空 pass 清 1。
      const clearPass = encoder.beginRenderPass({
        label: 'FlowArrow.DummyDepthClear',
        colorAttachments: [],
        depthStencilAttachment: {
          view: this._dummyDepthView!,
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      });
      clearPass.end();
      this._dummyDepthCleared = true;
    }
    return this._dummyDepthView!;
  }

  private _prepare(): void {
    const device = this._context.device;

    if (!this._uniformBuffer) {
      const d = this._uniformData;
      d[16] = this._baseColor[0];
      d[17] = this._baseColor[1];
      d[18] = this._baseColor[2];
      d[19] = this._baseColor[3];
      d[20] = this._arrowColor[0];
      d[21] = this._arrowColor[1];
      d[22] = this._arrowColor[2];
      d[23] = this._arrowColor[3];
      this._uniformBuffer = device.createBuffer({
        label: 'FlowArrow.Uniforms',
        size: UNIFORM_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }

    if (!this._vertexBuffer && this._positions) {
      const geo = buildFlowArrowGeometry(this._positions);
      if (geo) {
        // 箭头密度：repeatCount（hexapod 语义，默认自适应 ~每 100 m 一支）
        // → 每归一化弧长单位 gap（material.gap 显式给定时优先）。
        this._gap = effectiveFlowArrowGap(this._material, geo.arcLength);
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
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform' },
          },
          {
            binding: 1,
            visibility: GPUShaderStage.FRAGMENT,
            texture: { sampleType: 'depth' },
          },
          {
            binding: 2,
            visibility: GPUShaderStage.FRAGMENT,
            sampler: { type: 'non-filtering' },
          },
        ],
      });
      this._bindGroupLayout = bgl;
      this._pipeline = device.createRenderPipeline({
        label: 'FlowArrow.Pipeline',
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        vertex: {
          module,
          entryPoint: 'vs_main',
          buffers: [{
            arrayStride: FLOATS_PER_VERTEX * 4,
            attributes: [
              { shaderLocation: 0, offset: 0,  format: 'float32x3' }, // position（中心线）
              { shaderLocation: 1, offset: 12, format: 'float32' },   // arc
              { shaderLocation: 2, offset: 16, format: 'float32' },   // v
              { shaderLocation: 3, offset: 20, format: 'float32x3' }, // side（本段方向）
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
      // bindGroup 在 execute 按当前深度视图构建（见 _resolveDepthView）。
    }
  }
}

export default FlowArrowPolyline;
