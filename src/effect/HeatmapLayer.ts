import type { IPass } from '../scene/passes/IPass';
import type { Camera } from '../scene/Camera';
import type { Context } from '../renderer/Context';
import { HEATMAP_SHADER } from '../shaders/HeatmapShader';
import { CAMERA_UBO_SIZE, writeCameraUniforms } from '../renderer/CameraUniforms';

/** 热力输入点：经纬度（度）+ 权重（0-1）。 */
export interface HeatPoint {
  lon: number;
  lat: number;
  weight: number;
}

/** 色带停靠点：[stop 0-1, [r, g, b] 0-1]。 */
export type HeatGradientStop = [number, [number, number, number]];

export interface HeatmapLayerOptions {
  /** 热力点集合（经纬度，度）。 */
  points: HeatPoint[];
  /** 单点扩散半径（米）。Default: 60_000 */
  radius?: number;
  /** 色带停靠点（stop 升序，≥2 个）。Default: 蓝→青→黄→橙→红 */
  gradient?: HeatGradientStop[];
  /** 全局不透明度（0-1）。Default: 0.85 */
  maxOpacity?: number;
  /** 贴地网格细分（每轴格数）。Default: 64 */
  gridN?: number;
  /** 热力纹理最长边像素。Default: 768 */
  textureSize?: number;
  /**
   * 相机引用（必填）：每帧从该相机读取 viewProj / camPos 上传 UBO。
   * 通常传 `viewer.scene.camera`。
   */
  camera: Camera;
}

/* WGS84 参数（与 RectangleGeometry / CircleScan 一致）。 */
const SEMI_MAJOR = 6378137.0;
const SEMI_MINOR = 6356752.3142;
const ECCENTRICITY_SQ = (SEMI_MAJOR * SEMI_MAJOR - SEMI_MINOR * SEMI_MINOR) / (SEMI_MAJOR * SEMI_MAJOR);
const DEG2RAD = Math.PI / 180;

/**
 * WGS84 经纬度（度）+ 高程（米）→ 归一化 ECEF（Earth radius = 1）。
 */
function lonLatToEcef(lonDeg: number, latDeg: number, hMeters: number): [number, number, number] {
  const lon = lonDeg * DEG2RAD;
  const lat = latDeg * DEG2RAD;
  const sinLat = Math.sin(lat);
  const n = SEMI_MAJOR / Math.sqrt(1 - ECCENTRICITY_SQ * sinLat * sinLat);
  const xy = (n + hMeters) * Math.cos(lat);
  return [
    (xy * Math.cos(lon)) / SEMI_MAJOR,
    (xy * Math.sin(lon)) / SEMI_MAJOR,
    ((n * (1 - ECCENTRICITY_SQ) + hMeters) * sinLat) / SEMI_MAJOR,
  ];
}

const DEFAULT_GRADIENT: HeatGradientStop[] = [
  [0.0,  [0.02, 0.06, 0.55]],
  [0.25, [0.03, 0.42, 0.9]],
  [0.45, [0.05, 0.85, 0.72]],
  [0.65, [0.85, 0.93, 0.12]],
  [0.82, [0.98, 0.55, 0.05]],
  [1.0,  [0.93, 0.08, 0.08]],
];

/**
 * HeatmapLayer - 热力图贴地叠加特效（对齐 EV-Globe 的 EV_Heatmap）。
 *
 * CPU 端离屏 canvas 两步合成（经典 heatmap 算法）：先以
 * `globalCompositeOperation: 'lighter'` 逐点累积径向渐变灰度强度，再把
 * alpha 通道映射到色带生成彩色纹理；GPU 端将点集包围盒展开为
 * `gridN × gridN` 地表网格（逐顶点投影 WGS84 椭球），UV 采样该纹理，
 * 以 additive 发光叠加到最终画面。
 *
 * 作为 {@link IPass} 挂在 `scene.effects`（EffectManager）上；因 IPass
 * 无深度附件，片元内用射线-单位球求交剔除被地球遮挡的远侧网格。
 *
 * ```ts
 * const heat = new HeatmapLayer(viewer.context, {
 *   points: [{ lon: 116.4, lat: 39.9, weight: 0.8 }, ...],
 *   radius: 60_000,
 *   camera: viewer.scene.camera,
 * });
 * viewer.scene.effects.add(heat);
 * heat.setRadius(120_000);   // 动态调半径（重画纹理）
 * heat.setOpacity(0.5);      // 动态调不透明度
 * ```
 */
export class HeatmapLayer implements IPass {
  readonly label = 'HeatmapLayer';

  /** 可见性开关（EffectManager 每帧检查）。 */
  show = true;

  private _context: Context;
  private _camera: Camera;
  private _points: HeatPoint[];
  private _radius: number;
  private _gradient: HeatGradientStop[];
  private _maxOpacity: number;
  private readonly _gridN: number;
  private readonly _textureSize: number;
  private readonly _indexCount: number;

  private _pipeline: GPURenderPipeline | null = null;
  private _bindGroup: GPUBindGroup | null = null;
  private _vertexBuffer: GPUBuffer | null = null;
  private _indexBuffer: GPUBuffer | null = null;
  private _cameraUBO: GPUBuffer | null = null;
  private _paramsUBO: GPUBuffer | null = null;
  private _texture: GPUTexture | null = null;
  private _sampler: GPUSampler | null = null;

  /** 当前纹理覆盖的经纬度包围盒 [minLon, minLat, maxLon, maxLat]（度）。 */
  private _bounds: [number, number, number, number] | null = null;

  private static readonly CAMERA_UBO_BYTES = CAMERA_UBO_SIZE;
  /** opacity + pad。 */
  private static readonly PARAMS_UBO_BYTES = 16;

  constructor(context: Context, options: HeatmapLayerOptions) {
    this._context = context;
    this._camera = options.camera;
    this._points = options.points;
    this._radius = options.radius ?? 60_000;
    this._gradient = options.gradient ?? DEFAULT_GRADIENT;
    this._maxOpacity = options.maxOpacity ?? 0.85;
    this._gridN = Math.max(2, options.gridN ?? 64);
    this._textureSize = Math.max(64, options.textureSize ?? 768);
    this._indexCount = this._gridN * this._gridN * 6;

    if (this._points.length === 0) {
      throw new Error('HeatmapLayer: points 不能为空');
    }
  }

  /** 动态调整单点扩散半径（米）——重画 CPU 纹理并重新上传。 */
  setRadius(meters: number): void {
    this._radius = Math.max(1000, meters);
    this._uploadTexture();
  }

  /** 当前扩散半径（米，只读）。 */
  get radius(): number { return this._radius; }

  /** 当前全局不透明度（只读）。 */
  get opacity(): number { return this._maxOpacity; }

  /** 动态调整全局不透明度（0-1）。 */
  setOpacity(v: number): void {
    this._maxOpacity = Math.min(1, Math.max(0, v));
  }

  /** 替换热力点集并重画纹理。 */
  setPoints(points: HeatPoint[]): void {
    if (points.length === 0) return;
    this._points = points;
    this._uploadTexture();
  }

  /** @inheritdoc IPass - 幂等初始化（管线 / 网格 / 纹理 / bindGroup）。 */
  prepare(): void {
    if (this._pipeline) return;
    const device = this._context.device;

    const module = this._context.createShaderModule(HEATMAP_SHADER, 'HeatmapShader');

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });

    this._pipeline = device.createRenderPipeline({
      label: 'HeatmapLayerPipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: {
        module,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 20, // [px,py,pz,u,v]
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },  // position（归一化 ECEF）
            { shaderLocation: 1, offset: 12, format: 'float32x2' }, // uv（热力纹理）
          ],
        }],
      },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [
          {
            format: this._context.preferredFormat,
            // additive 发光叠加（与 DynamicWall 同款 src-alpha / one）。
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });

    this._cameraUBO = device.createBuffer({
      label: 'HeatmapCameraUBO',
      size: HeatmapLayer.CAMERA_UBO_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._paramsUBO = device.createBuffer({
      label: 'HeatmapParamsUBO',
      size: HeatmapLayer.PARAMS_UBO_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this._sampler = device.createSampler({
      label: 'HeatmapSampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    // 首次上传：建纹理 + 地表网格 + 顶点缓冲。
    this._uploadTexture();
    const grid = this._buildGrid(this._bounds!);

    this._vertexBuffer = device.createBuffer({
      label: 'HeatmapVertices',
      size: Math.max(4, grid.vertices.byteLength),
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this._context.writeBuffer(this._vertexBuffer, 0, HeatmapLayer._asBytes(grid.vertices));

    this._indexBuffer = device.createBuffer({
      label: 'HeatmapIndices',
      size: Math.max(4, grid.indices.byteLength),
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    this._context.writeBuffer(this._indexBuffer, 0, HeatmapLayer._asBytes(grid.indices));

    this._bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this._cameraUBO } },
        { binding: 1, resource: { buffer: this._paramsUBO } },
        { binding: 2, resource: this._texture!.createView() },
        { binding: 3, resource: this._sampler },
      ],
    });
  }

  /** 每帧执行：上传相机 / 参数 UBO 后在 swapchain 上叠加绘制热力层。 */
  execute(encoder: GPUCommandEncoder, target: GPUTextureView, _time: number): void {
    this.prepare();
    if (!this._pipeline || !this._bindGroup || !this._indexBuffer) return;

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

    const params = new Float32Array([this._maxOpacity, 0, 0, 0]);
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
    pass.setIndexBuffer(this._indexBuffer, 'uint32');
    pass.drawIndexed(this._indexCount);
    pass.end();
  }

  /** 释放 GPU 资源；调用后本 pass 不可再用。 */
  destroy(): void {
    this._vertexBuffer?.destroy();
    this._indexBuffer?.destroy();
    this._cameraUBO?.destroy();
    this._paramsUBO?.destroy();
    this._texture?.destroy();
    this._vertexBuffer = null;
    this._indexBuffer = null;
    this._cameraUBO = null;
    this._paramsUBO = null;
    this._texture = null;
    this._bindGroup = null;
    this._pipeline = null;
  }

  // ── CPU 合成：canvas 渐变圆 'lighter' 累积 + 色带上色 ─────────────────────

  /** 计算点集包围盒（度），外扩一个半径（米→度）。 */
  private _computeBounds(): [number, number, number, number] {
    let minLon = Infinity;
    let minLat = Infinity;
    let maxLon = -Infinity;
    let maxLat = -Infinity;
    for (const p of this._points) {
      if (p.lon < minLon) minLon = p.lon;
      if (p.lon > maxLon) maxLon = p.lon;
      if (p.lat < minLat) minLat = p.lat;
      if (p.lat > maxLat) maxLat = p.lat;
    }
    const latMid = (minLat + maxLat) * 0.5;
    const dLat = this._radius / 111_320;
    const dLon = this._radius / (111_320 * Math.max(0.2, Math.cos(latMid * DEG2RAD)));
    return [minLon - dLon, minLat - dLat, maxLon + dLon, maxLat + dLat];
  }

  /** 重建热力纹理（半径/点集变化时调用）。 */
  private _uploadTexture(): void {
    const bounds = this._computeBounds();
    this._bounds = bounds;
    const [minLon, minLat, maxLon, maxLat] = bounds;
    const dLon = Math.max(maxLon - minLon, 1e-6);
    const dLat = Math.max(maxLat - minLat, 1e-6);

    // 画布按包围盒纵横比取尺寸（最长边 = textureSize）。
    const aspect = dLon / dLat;
    const w = aspect >= 1 ? this._textureSize : Math.max(64, Math.round(this._textureSize * aspect));
    const h = aspect >= 1 ? Math.max(64, Math.round(this._textureSize / aspect)) : this._textureSize;

    // ── 步骤 1：灰度强度累积（径向渐变圆 + 'lighter'）─────────────────
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('HeatmapLayer: 无法创建 2D context');
    ctx.globalCompositeOperation = 'lighter';

    const latMid = (minLat + maxLat) * 0.5;
    const metersPerDegLon = 111_320 * Math.max(0.2, Math.cos(latMid * DEG2RAD));
    const rPxY = (this._radius / 111_320) / dLat * h;          // 纬度方向像素半径
    const rPxX = (this._radius / metersPerDegLon) / dLon * w;  // 经度方向像素半径

    for (const p of this._points) {
      const x = ((p.lon - minLon) / dLon) * w;
      const y = (1 - (p.lat - minLat) / dLat) * h;
      const weight = Math.min(1, Math.max(0, p.weight));
      // 经纬度量纲不同 → 椭圆缩放把圆变形成正确的大地半径椭圆。
      ctx.save();
      ctx.translate(x, y);
      ctx.scale(rPxX / rPxY, 1);
      const grad = ctx.createRadialGradient(0, 0, 0, 0, 0, rPxY);
      grad.addColorStop(0, `rgba(0,0,0,${weight})`);
      grad.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = grad;
      ctx.fillRect(-rPxY, -rPxY, rPxY * 2, rPxY * 2);
      ctx.restore();
    }

    // ── 步骤 2：色带映射（alpha 通道 → palette 查色）──────────────────
    const palette = this._palette(ctx);
    const img = ctx.getImageData(0, 0, w, h);
    const data = img.data;
    for (let i = 0; i < data.length; i += 4) {
      const a = data[i + 3];
      if (a === 0) continue;
      const pi = a * 4;
      data[i + 0] = palette[pi + 0];
      data[i + 1] = palette[pi + 1];
      data[i + 2] = palette[pi + 2];
      // a 保留为累积强度（shader 里再乘全局 opacity）
    }
    ctx.putImageData(img, 0, 0);

    // ── 步骤 3：上传 GPU 纹理 ─────────────────────────────────────────
    const device = this._context.device;
    if (this._texture) {
      this._texture.destroy();
    }
    this._texture = device.createTexture({
      label: 'HeatmapTexture',
      size: [w, h],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture(
      { source: canvas },
      { texture: this._texture },
      [w, h]
    );

    // 纹理重建后 bindGroup 需要重建（纹理视图变化）。
    if (this._pipeline && this._bindGroup) {
      this._bindGroup = device.createBindGroup({
        layout: this._pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this._cameraUBO! } },
          { binding: 1, resource: { buffer: this._paramsUBO! } },
          { binding: 2, resource: this._texture.createView() },
          { binding: 3, resource: this._sampler! },
        ],
      });
    }
  }

  /** 色带停靠点 → 256×1 RGBA 像素表（写入传入 ctx，随后由调用方重新填充主画布）。 */
  private _palette(ctx: CanvasRenderingContext2D): Uint8ClampedArray {
    const grad = ctx.createLinearGradient(0, 0, 256, 0);
    const list = this._gradient.length >= 2 ? this._gradient : DEFAULT_GRADIENT;
    for (const [stop, [r, g, b]] of list) {
      grad.addColorStop(
        Math.min(1, Math.max(0, stop)),
        `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`
      );
    }
    ctx.save();
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 256, 1);
    const data = ctx.getImageData(0, 0, 256, 1).data;
    ctx.restore();
    return data;
  }

  // ── GPU 网格：包围盒 → gridN×gridN 地表网格（WGS84 椭球投影）────────────

  private _buildGrid(bounds: [number, number, number, number]): {
    vertices: Float32Array;
    indices: Uint32Array;
  } {
    const [minLon, minLat, maxLon, maxLat] = bounds;
    const n = this._gridN;
    const vertices = new Float32Array((n + 1) * (n + 1) * 5);
    let vi = 0;
    for (let j = 0; j <= n; j++) {
      const v = j / n;
      const lat = minLat + (maxLat - minLat) * v;
      for (let i = 0; i <= n; i++) {
        const u = i / n;
        const lon = minLon + (maxLon - minLon) * u;
        const [x, y, z] = lonLatToEcef(lon, lat, 0);
        vertices[vi++] = x;
        vertices[vi++] = y;
        vertices[vi++] = z;
        vertices[vi++] = u;
        vertices[vi++] = 1 - v;
      }
    }

    const indices = new Uint32Array(n * n * 6);
    let ii = 0;
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const a = j * (n + 1) + i;
        const b = a + 1;
        const c = a + (n + 1);
        const d = c + 1;
        indices[ii++] = a; indices[ii++] = c; indices[ii++] = b;
        indices[ii++] = b; indices[ii++] = c; indices[ii++] = d;
      }
    }
    return { vertices, indices };
  }

  /** 以 Uint8Array 视图暴露 typed array 的可拷贝字节（TS 5.7+ BufferSource 兼容）。 */
  private static _asBytes(src: ArrayBufferView): Uint8Array<ArrayBuffer> {
    return new Uint8Array(src.buffer as ArrayBuffer, src.byteOffset, src.byteLength);
  }
}

export default HeatmapLayer;
