import type { IPass } from '../scene/passes/IPass';
import type { Context } from '../renderer/Context';
import type { Camera } from '../scene/Camera';
import { Cartesian3 } from '../math/Cartesian3';
import { CesiumMath } from '../math/CesiumMath';
import { CIRCLE_SCAN_SHADER } from '../shaders/CircleScanShader';

/**
 * CircleScan - 地表圆形扫描特效（T3.3，对齐 EV-Globe 的 EV_CircleScan）。
 *
 * 在以 `center` 为圆心、`radius`（米）为半径的地表圆盘上绘制从中心周期性
 * 向外扩散的扫描波前（同心 `rings` 圈，每 `speed` 秒扫完一轮），附加一个
 * 中心恒亮光斑。圆盘为极坐标网格（径向 32 × 周向 64），逐顶点投影到 WGS84
 * 椭球面，因此大半径下依然贴合地表、无悬浮。
 *
 * 通过 `EffectManager` 挂载（`scene.effects.add(scan)`），在 Transparent
 * composite 之后以 additive（one, one）混合直接画到 swap-chain 上，无深度
 * 测试。相机矩阵需通过 `setCamera` 提供（每帧在 `execute` 中重传）。
 *
 * ```ts
 * const scan = new CircleScan(viewer.context, {
 *   center: { lon: 116.4, lat: 39.9 },
 *   radius: 300_000,
 *   color: [0.2, 0.9, 1.0, 0.85],
 *   speed: 2.0,
 *   rings: 3,
 * });
 * scan.setCamera(viewer.camera);
 * viewer.scene.effects.add(scan);
 * ```
 */
export interface CircleScanOptions {
  /** 圆心经纬度（度）。 */
  center: { lon: number; lat: number };
  /** 扫描半径（米）。默认 100_000。 */
  radius?: number;
  /** 扫描颜色（线性 RGBA，a 为基准不透明度）。默认青蓝色。 */
  color?: [number, number, number, number];
  /** 扫描周期（秒/轮）。默认 2.0。 */
  speed?: number;
  /** 同心环数（同时可见的波前圈数）。默认 3。 */
  rings?: number;
}

/* WGS84 parameters (matching RectangleGeometry / Cartesian3.fromRadians). */
const SEMI_MAJOR = 6378137.0;
const SEMI_MINOR = 6356752.3142;
const ECCENTRICITY_SQ = 1.0 - (SEMI_MINOR * SEMI_MINOR) / (SEMI_MAJOR * SEMI_MAJOR);

const RADIAL_SEGMENTS = 32;        // 径向网格段数（圆心 → 边缘）
const CIRCUMFERENCE_SEGMENTS = 64; // 周向网格段数
const FLOATS_PER_VERTEX = 4;       // [position(3), u(1)] = 16 字节
const UNIFORM_BYTES = 96;          // mat4x4 + vec4(color) + vec4(params)

const DEFAULT_RADIUS_METERS = 100_000;
const DEFAULT_SPEED_SECONDS = 2.0;
const DEFAULT_RINGS = 3;
const DEFAULT_COLOR: [number, number, number, number] = [0.2, 0.9, 1.0, 0.85];

const TWO_PI = Math.PI * 2;

export class CircleScan implements IPass {
  readonly label = 'CircleScan';

  /** 可见性开关（Effect 类型要求）。 */
  show = true;

  private _context: Context;
  private _camera: Camera | null = null;

  private _centerLonRad: number;
  private _centerLatRad: number;
  private _radiusMeters: number;
  private _color: [number, number, number, number];
  private _speed: number;
  private _rings: number;

  private _vertexBuffer: GPUBuffer | null = null;
  private _indexBuffer: GPUBuffer | null = null;
  private _uniformBuffer: GPUBuffer | null = null;
  private _pipeline: GPURenderPipeline | null = null;
  private _bindGroup: GPUBindGroup | null = null;
  private _indexCount = 0;
  private _destroyed = false;

  /** 每帧上传的 uniform 暂存：[viewProj(16), color(4), params(4)]。 */
  private _uniformData = new Float32Array(UNIFORM_BYTES / 4);

  constructor(context: Context, options: CircleScanOptions) {
    this._context = context;
    this._centerLonRad = CesiumMath.toRadians(options.center.lon);
    this._centerLatRad = CesiumMath.toRadians(options.center.lat);
    this._radiusMeters = Math.max(1, options.radius ?? DEFAULT_RADIUS_METERS);
    const c = options.color ?? DEFAULT_COLOR;
    this._color = [c[0], c[1], c[2], c[3]];
    this._speed = Math.max(0.01, options.speed ?? DEFAULT_SPEED_SECONDS);
    this._rings = Math.max(1, Math.round(options.rings ?? DEFAULT_RINGS));

    // 静态部分预填，execute 中只更新 viewProj 与 time。
    const d = this._uniformData;
    d[16] = this._color[0];
    d[17] = this._color[1];
    d[18] = this._color[2];
    d[19] = this._color[3];
    d[21] = this._speed;
    d[22] = this._rings;
  }

  /** 提供用于投影的相机（viewProj 每帧重传）。挂载后调用一次即可。 */
  setCamera(camera: Camera): void {
    this._camera = camera;
  }

  /** 幂等初始化（管线 / 缓冲 / bindGroup）。 */
  prepare(): void {
    if (!this._destroyed) this._prepare();
  }

  execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void {
    if (this._destroyed || !this.show) return;
    this._prepare();

    const pipeline     = this._pipeline;
    const bindGroup    = this._bindGroup;
    const vertexBuffer = this._vertexBuffer;
    const indexBuffer  = this._indexBuffer;
    const uniformBuffer = this._uniformBuffer;
    if (!pipeline || !bindGroup || !vertexBuffer || !indexBuffer || !uniformBuffer) return;

    const d = this._uniformData;
    if (this._camera) {
      d.set(this._camera.viewProjectionMatrix.values, 0); // viewProj @ offset 0
    }
    d[20] = time; // params.x
    this._context.writeBuffer(uniformBuffer, 0, d);

    const pass = encoder.beginRenderPass({
      label: this.label,
      colorAttachments: [{
        view: target,        // swap-chain，叠加在已合成的场景之上
        loadOp: 'load',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setIndexBuffer(indexBuffer, 'uint32');
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

  private _prepare(): void {
    if (this._pipeline) return;
    const device = this._context.device;

    // ── Geometry: polar-grid disc projected onto the WGS84 ellipsoid ────────
    // Vertices: 1 centre + radial rings x (circumference + 1); per-vertex
    // layout [x, y, z, u] with u = normalised radial distance [0, 1].
    const radial = RADIAL_SEGMENTS;
    const circum = CIRCUMFERENCE_SEGMENTS;
    const vertexCount = 1 + radial * (circum + 1);
    const vertices = new Float32Array(vertexCount * FLOATS_PER_VERTEX);
    const indexCount = circum + (radial - 1) * circum * 6;
    const indices = new Uint32Array(indexCount);

    const centre = this._geodeticSurfacePoint(this._centerLonRad, this._centerLatRad);
    vertices[0] = centre[0];
    vertices[1] = centre[1];
    vertices[2] = centre[2];
    vertices[3] = 0; // u at the centre

    // ENU 基底（米制 ECEF 方向），局部平面偏移量投到椭球面。
    const sinLon = Math.sin(this._centerLonRad);
    const cosLon = Math.cos(this._centerLonRad);
    const sinLat = Math.sin(this._centerLatRad);
    const cosLat = Math.cos(this._centerLatRad);
    const origin = Cartesian3.fromRadians(this._centerLonRad, this._centerLatRad, 0); // 米
    const eastX = -sinLon,           eastY = cosLon,                eastZ = 0;
    const northX = -sinLat * cosLon, northY = -sinLat * sinLon,     northZ = cosLat;

    let vi = FLOATS_PER_VERTEX;
    for (let i = 1; i <= radial; i++) {
      const f = i / radial;                       // 归一化半径 → u
      const r = f * this._radiusMeters;           // 米
      for (let j = 0; j <= circum; j++) {
        const theta = (j / circum) * TWO_PI;
        const dx = r * Math.cos(theta);
        const dy = r * Math.sin(theta);
        // 局部平面点（米制 ECEF）→ 沿地心射线与归一化椭球求交 → 椭球面点。
        // 注意不能用 atan2 反推纬度再喂给 _geodeticSurfacePoint：atan2 得到
        // 的是地心纬度，而 _geodeticSurfacePoint 按大地纬度实现，中纬度大
        // 半径下贴地误差可达 ~20 km；直接射线求交则天然一致。
        const px = origin.x + eastX * dx + northX * dy;
        const py = origin.y + eastY * dx + northY * dy;
        const pz = origin.z + eastZ * dx + northZ * dy;
        const len = Math.hypot(px, py, pz) || 1;
        const dxU = px / len, dyU = py / len, dzU = pz / len;
        // 椭球（归一化，赤道半径 = 1）：x² + y² + z²/b² = 1，b = SEMI_MINOR / SEMI_MAJOR。
        // 射线 p = t·d 代入解得 t = 1 / sqrt(dx² + dy² + dz²/b²)。
        const bNorm = SEMI_MINOR / SEMI_MAJOR;
        const t = 1 / Math.sqrt(
          dxU * dxU + dyU * dyU + (dzU * dzU) / (bNorm * bNorm),
        );
        vertices[vi]     = dxU * t;
        vertices[vi + 1] = dyU * t;
        vertices[vi + 2] = dzU * t;
        vertices[vi + 3] = f;
        vi += FLOATS_PER_VERTEX;
      }
    }

    let ii = 0;
    const firstRing = 1;
    for (let j = 0; j < circum; j++) { // 中心扇形
      indices[ii++] = 0;
      indices[ii++] = firstRing + j + 1;
      indices[ii++] = firstRing + j;
    }
    for (let i = 1; i < radial; i++) { // 环间四边形
      const a0 = 1 + (i - 1) * (circum + 1);
      const b0 = a0 + circum + 1;
      for (let j = 0; j < circum; j++) {
        const a = a0 + j;
        const b = b0 + j;
        const c = b + 1;
        const d = a + 1;
        indices[ii++] = a;
        indices[ii++] = b;
        indices[ii++] = d;
        indices[ii++] = b;
        indices[ii++] = c;
        indices[ii++] = d;
      }
    }
    this._indexCount = indexCount;

    const vertexBuffer = device.createBuffer({
      label: 'CircleScan.Vertices',
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this._context.writeBuffer(vertexBuffer, 0, vertices);
    this._vertexBuffer = vertexBuffer;

    const indexBuffer = device.createBuffer({
      label: 'CircleScan.Indices',
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    this._context.writeBuffer(indexBuffer, 0, indices);
    this._indexBuffer = indexBuffer;

    const uniformBuffer = device.createBuffer({
      label: 'CircleScan.Uniforms',
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._uniformBuffer = uniformBuffer;

    // ── Pipeline: additive (one, one) onto the swap-chain, no depth test ───
    const module = this._context.createShaderModule(CIRCLE_SCAN_SHADER, 'CircleScan.Module');
    const bgl = device.createBindGroupLayout({
      label: 'CircleScan.BGL',
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      }],
    });
    this._pipeline = device.createRenderPipeline({
      label: 'CircleScan.Pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex: {
        module,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: FLOATS_PER_VERTEX * 4,
          attributes: [
            { shaderLocation: 0, offset: 0,  format: 'float32x3' }, // position
            { shaderLocation: 1, offset: 12, format: 'float32' },   // u (radial)
          ],
        }],
      },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [{
          format: this._context.preferredFormat,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' }, // cullMode 默认 none（圆盘两面可见）
    });
    this._bindGroup = device.createBindGroup({
      layout: bgl,
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
    });
  }

  /**
   * 大地坐标（弧度）→ 归一化 ECEF 椭球面点（赤道半径 = 1，极轴按 b/a 压缩，
   * 与 RectangleGeometry / EllipsoidGeometry 同一约定）。
   */
  private _geodeticSurfacePoint(lonRad: number, latRad: number): [number, number, number] {
    const cosLat = Math.cos(latRad);
    const sinLat = Math.sin(latRad);
    // 卯酉圈曲率半径（米）：n = a / sqrt(1 - e² sin²lat)
    const n = SEMI_MAJOR / Math.sqrt(1.0 - ECCENTRICITY_SQ * sinLat * sinLat);
    return [
      (n * cosLat * Math.cos(lonRad)) / SEMI_MAJOR,
      (n * cosLat * Math.sin(lonRad)) / SEMI_MAJOR,
      (n * (1.0 - ECCENTRICITY_SQ) * sinLat) / SEMI_MAJOR,
    ];
  }
}

export default CircleScan;
