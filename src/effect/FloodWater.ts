import type { IPass } from '../scene/passes/IPass';
import type { Camera } from '../scene/Camera';
import type { Context } from '../renderer/Context';
import type { TerrainLayer } from '../terrain/TerrainLayer';
import type { HeightSampler } from '../media/GimbalMath';
import type { RectangleDeg } from '../analysis/FloodAnalysis';
import { ecefFromLonLatHeight } from '../terrain/TerrainProvider';
import { FLOOD_WATER_SHADER } from '../shaders/FloodWaterShader';
import { CAMERA_UBO_SIZE, writeCameraUniforms } from '../renderer/CameraUniforms';

/** 高程网格注入数据（行主序 n×n，覆盖 bounds，同 FloodAnalysis.heightGrid 输出）。 */
export interface HeightGridData {
  data: Float64Array;
  n: number;
}

export interface FloodWaterOptions {
  /** WebGPU 上下文（通常 `viewer.scene.context`）。 */
  context: Context;
  /**
   * 相机引用（必填）：`IPass.execute` 不携带相机，水面每帧自行从该相机
   * 读取 viewProj / camPos 上传 UBO。通常传 `viewer.scene.camera`。
   */
  camera: Camera;
  /** 水面覆盖的经纬度矩形（度）。 */
  bounds: RectangleDeg;
  /**
   * 地形层：无 `heightSampler` / `heightGrid` 时用其 `sampleHeight` 逐格
   * 采样地形高。采样优先级：heightSampler > heightGrid > terrainLayer
   * > 恒 0 平面（无任何地形源时的回退）。
   */
  terrainLayer?: TerrainLayer;
  /** 显式高程采样回调（优先级最高，供测试 / 合成 DEM 注入）。 */
  heightSampler?: HeightSampler;
  /** 预采样高程网格（FloodAnalysis.heightGrid 的输出）。 */
  heightGrid?: HeightGridData;
  /** 初始水位（米）。Default: `0` */
  waterLevel?: number;
  /** 经向网格数（>= 2）。Default: `48` */
  gridNX?: number;
  /** 纬向网格数（>= 2）。Default: `32` */
  gridNY?: number;
  /** 水体基色 RGBA。Default: `[0.10, 0.35, 0.55, 0.80]` */
  color?: [number, number, number, number];
  /** 波速倍率。Default: `1.0` */
  speed?: number;
  /** 波高（法线扰动强度 0-1）。Default: `0.3` */
  waveScale?: number;
  /** 波光高光强度。Default: `0.8` */
  glint?: number;
  /** 太阳方向（指向太阳，归一化 ECEF）。Default: `[0.4, 0.5, 0.77]` */
  sunDirection?: [number, number, number];
}

/**
 * FloodWater - 水深/淹没水面特效（基于 DynamicWater 改造）。
 *
 * 经纬规则网格（默认 48×32）逐格采样地形高后投影到归一化 ECEF：
 * 顶点位置按**地形高程**径向布置（`|p| = 1 + terrainH/6378137`，
 * 引擎「高度折入模长」约定），`uv.x` 存归一化地形高、`uv.y` 存地形处
 * 径向模长（stride 20B 与 DynamicWater 一致）。水位走 uniform：
 * VS 内 `k = max(kTerrain, waterLevel)` 径向放缩——水位变化零重建；
 * FS 内 `depth = waterLevel - terrainH` 还原水深走 4 段色带，
 * 岸线以上片元 alpha 0 剔除。
 *
 * 坐标约定（血泪坑）：经纬度 → 归一化 ECEF 用
 * `ecefFromLonLatHeight`（径向单位向量 × (1 + h/EARTH_SCALE)），
 * **不可用 atan2 地心纬度反推**（见 CircleScan._prepare / GimbalMath 注释）。
 *
 * @example
 * ```typescript
 * const water = scene.effects.add(new FloodWater(scene.context, {
 *   bounds: { west: 110.3, south: 31.0, east: 111.0, north: 31.5 },
 *   terrainLayer: viewer.globe.terrainLayer,
 *   waterLevel: 500,
 *   camera: scene.camera,
 * }));
 * water.setWaterLevel(600); // 只更新 uniform，下一帧生效
 * ```
 */
export class FloodWater implements IPass {
  readonly label = 'FloodWater';

  /** 可见性开关（EffectManager 每帧检查）。 */
  show = true;

  private _context: Context;
  private _camera: Camera;
  private _bounds: RectangleDeg;
  private readonly _gridNX: number;
  private readonly _gridNY: number;
  private _color: [number, number, number, number];
  private _speed: number;
  private _waveScale: number;
  private _glint: number;
  private _sunDir: [number, number, number];
  /** 水位（米）。 */
  private _waterLevel: number;
  /** 逐格地形高采样器（构造期绑定，setBounds 重建几何时复用）。 */
  private readonly _sampleHeight: HeightSampler;
  private _geometry: { vertices: Float32Array; indices: Uint32Array; indexCount: number };

  private _pipeline: GPURenderPipeline | null = null;
  private _bindGroup: GPUBindGroup | null = null;
  private _vertexBuffer: GPUBuffer | null = null;
  private _indexBuffer: GPUBuffer | null = null;
  private _cameraUBO: GPUBuffer | null = null;
  private _paramsUBO: GPUBuffer | null = null;

  /** 与 Scene._uploadCameraUBO 相同的 256B 布局。 */
  private static readonly CAMERA_UBO_BYTES = CAMERA_UBO_SIZE;
  /** color(16) + sunDir(16) + time/speed/waveScale/glint(16) + waterLevel/earthScale/pad(16) = 64B。 */
  private static readonly PARAMS_UBO_BYTES = 64;

  constructor(context: Context, options: FloodWaterOptions) {
    const nx = options.gridNX ?? DEFAULT_GRID_X;
    const ny = options.gridNY ?? DEFAULT_GRID_Y;
    if (!Number.isFinite(nx) || nx < 2 || !Number.isFinite(ny) || ny < 2) {
      throw new Error(`FloodWater: gridNX/gridNY 必须 >= 2，收到 ${nx}×${ny}`);
    }

    this._context   = context;
    this._camera    = options.camera;
    this._bounds    = { ...options.bounds };
    this._gridNX    = Math.floor(nx);
    this._gridNY    = Math.floor(ny);
    this._color     = options.color ?? [0.10, 0.35, 0.55, 0.80];
    this._speed     = options.speed ?? 1.0;
    this._waveScale = options.waveScale ?? 0.3;
    this._glint     = options.glint ?? 0.8;
    this._sunDir    = FloodWater._normalize(options.sunDirection ?? [0.4, 0.5, 0.77]);
    this._waterLevel = options.waterLevel ?? 0;

    // 采样优先级：heightSampler > heightGrid（双线性）> terrainLayer > 恒 0
    this._sampleHeight = options.heightSampler
      ?? (options.heightGrid ? gridSampler(this._bounds, options.heightGrid) : null)
      ?? (options.terrainLayer
        ? (lon: number, lat: number) => options.terrainLayer!.sampleHeight(lon, lat)
        : null)
      ?? (() => 0);

    this._geometry = FloodWater._buildGeometry(
      this._bounds, this._gridNX, this._gridNY, this._sampleHeight
    );
  }

  /** 当前水位（米）。 */
  get waterLevel(): number {
    return this._waterLevel;
  }

  /** 水面覆盖范围（度，副本）。 */
  get bounds(): RectangleDeg {
    return { ...this._bounds };
  }

  /**
   * 设置水位（米）。只更新 uniform（execute 每帧上传），下一帧生效，
   * 零几何重建。
   */
  setWaterLevel(levelMeters: number): void {
    if (!Number.isFinite(levelMeters)) {
      throw new Error(`FloodWater.setWaterLevel: 水位非法 ${levelMeters}`);
    }
    this._waterLevel = levelMeters;
  }

  /**
   * 更新覆盖范围（重新采样地形 + 重建网格与顶/索引缓冲）。水位保持不变。
   * GPU 缓冲重建走与 destroy/prepare 相同的幂等路径（下帧 prepare 重入）。
   */
  setBounds(bounds: RectangleDeg): void {
    this._bounds   = { ...bounds };
    this._geometry = FloodWater._buildGeometry(
      this._bounds, this._gridNX, this._gridNY, this._sampleHeight
    );
    // 清掉顶/索引缓冲（管线/UBO/bindGroup 不依赖几何尺寸，保留复用）；
    // 顶/索引缓冲为 null 时 prepare 会重建它们。
    this._vertexBuffer?.destroy();
    this._indexBuffer?.destroy();
    this._vertexBuffer = null;
    this._indexBuffer = null;
  }

  /** @inheritdoc IPass - 幂等初始化（管线 / 缓冲 / bindGroup）。 */
  prepare(): void {
    if (this._pipeline && this._vertexBuffer && this._indexBuffer && this._bindGroup) return;
    const device = this._context.device;

    if (!this._pipeline) {
      const module = this._context.createShaderModule(FLOOD_WATER_SHADER, 'FloodWaterShader');

      const bindGroupLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          // vs_main 里读 flood.waterLevel 布置顶点半径（k = max(kTerrain, waterLevel)），
          // visibility 必须含 VERTEX，否则管线验证失败并污染整帧 Frame Encoder
          { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        ],
      });

      this._pipeline = device.createRenderPipeline({
        label: 'FloodWaterPipeline',
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        vertex: {
          module,
          entryPoint: 'vs_main',
          buffers: [{
            arrayStride: 20, // 5 floats × 4 bytes：[px,py,pz,hNorm,kTerrain]
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },  // position
              { shaderLocation: 1, offset: 12, format: 'float32x2' }, // uv = (hNorm, kTerrain)
            ],
          }],
        },
        fragment: {
          module,
          entryPoint: 'fs_main',
          targets: [
            {
              format: this._context.preferredFormat,
              // 半透明覆盖底图；岸线以上片元 FS 输出 alpha 0 随混合剔除。
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
        label: 'FloodWaterCameraUBO',
        size: FloodWater.CAMERA_UBO_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      this._paramsUBO = device.createBuffer({
        label: 'FloodWaterParamsUBO',
        size: FloodWater.PARAMS_UBO_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      this._bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this._cameraUBO } },
          { binding: 1, resource: { buffer: this._paramsUBO } },
        ],
      });
    }

    if (!this._vertexBuffer) {
      const vertices = this._geometry.vertices;
      this._vertexBuffer = device.createBuffer({
        label: 'FloodWaterVertices',
        size: Math.max(4, vertices.byteLength),
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      if (vertices.byteLength > 0) {
        this._context.writeBuffer(this._vertexBuffer, 0, FloodWater._asBytes(vertices));
      }
    }

    if (!this._indexBuffer) {
      const indices = this._geometry.indices;
      this._indexBuffer = device.createBuffer({
        label: 'FloodWaterIndices',
        size: Math.max(4, indices.byteLength),
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      if (indices.byteLength > 0) {
        this._context.writeBuffer(this._indexBuffer, 0, FloodWater._asBytes(indices));
      }
    }
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

    // ── Params UBO（64B）────────────────────────────────────────────────
    const params = new Float32Array([
      this._color[0], this._color[1], this._color[2], this._color[3],
      this._sunDir[0], this._sunDir[1], this._sunDir[2], 0,
      time, this._speed, this._waveScale, this._glint,
      this._waterLevel / EARTH_SCALE, EARTH_SCALE, 0, 0,
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

  // ── Private ───────────────────────────────────────────────────────────────

  /**
   * 经纬规则网格 → 归一化 ECEF 顶点：每顶点 [px,py,pz,hNorm,kTerrain]。
   *
   * 位置按地形高程径向布置（ecefFromLonLatHeight：径向单位向量 ×
   * (1 + h/EARTH_SCALE)，y/z 含极区压扁）；hNorm = h/EARTH_SCALE 供 FS
   * 还原水深，kTerrain = 1 + h/EARTH_SCALE 供 VS 做水位径向放缩。
   * 索引为 (nx-1)×(ny-1) 个 quad 的两三角。
   */
  private static _buildGeometry(
    bounds: RectangleDeg,
    nx: number,
    ny: number,
    sampleHeight: (lon: number, lat: number) => number
  ): { vertices: Float32Array; indices: Uint32Array; indexCount: number } {
    const vertices = new Float32Array(nx * ny * 5);
    const tmp: [number, number, number] = [0, 0, 0];
    const dLon = (bounds.east - bounds.west) / (nx - 1);
    const dLat = (bounds.north - bounds.south) / (ny - 1);

    for (let iy = 0; iy < ny; iy++) {
      const lat = bounds.south + dLat * iy;
      for (let ix = 0; ix < nx; ix++) {
        const lon = bounds.west + dLon * ix;
        const h   = sampleHeight(lon, lat);
        const hN  = h / EARTH_SCALE;
        ecefFromLonLatHeight(lon, lat, h, tmp);
        const v = (iy * nx + ix) * 5;
        vertices[v + 0] = tmp[0];
        vertices[v + 1] = tmp[1];
        vertices[v + 2] = tmp[2];
        vertices[v + 3] = hN;          // uv.x：归一化地形高
        vertices[v + 4] = 1.0 + hN;    // uv.y：地形处径向模长
      }
    }

    const quads = (nx - 1) * (ny - 1);
    const indices = new Uint32Array(quads * 6);
    let o = 0;
    for (let iy = 0; iy < ny - 1; iy++) {
      for (let ix = 0; ix < nx - 1; ix++) {
        const r0 = iy * nx + ix;
        const r1 = (iy + 1) * nx + ix;
        indices[o++] = r0;
        indices[o++] = r1;
        indices[o++] = r0 + 1;
        indices[o++] = r0 + 1;
        indices[o++] = r1;
        indices[o++] = r1 + 1;
      }
    }
    return { vertices, indices, indexCount: indices.length };
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

/**
 * 高程网格（行主序 n×n，覆盖 bounds，网格点落在 bounds 四角上）的
 * 双线性采样器；出界点 clamp 到边缘值。
 */
function gridSampler(bounds: RectangleDeg, grid: HeightGridData): (lon: number, lat: number) => number {
  const { data, n } = grid;
  if (!Number.isFinite(n) || n < 2) {
    throw new Error(`FloodWater: heightGrid.n 必须 >= 2，收到 ${n}`);
  }
  return (lon: number, lat: number): number => {
    const fx = Math.min(1, Math.max(0, (lon - bounds.west) / (bounds.east - bounds.west))) * (n - 1);
    const fy = Math.min(1, Math.max(0, (lat - bounds.south) / (bounds.north - bounds.south))) * (n - 1);
    const x0 = Math.min(n - 2, Math.floor(fx));
    const y0 = Math.min(n - 2, Math.floor(fy));
    const tx = fx - x0;
    const ty = fy - y0;
    const h00 = data[y0 * n + x0];
    const h10 = data[y0 * n + x0 + 1];
    const h01 = data[(y0 + 1) * n + x0];
    const h11 = data[(y0 + 1) * n + x0 + 1];
    return (h00 * (1 - tx) + h10 * tx) * (1 - ty) + (h01 * (1 - tx) + h11 * tx) * ty;
  };
}

/** WGS84 赤道半轴（米）：米 ↔ 归一化单位换算（与 GimbalMath.EARTH_SCALE 同值）。 */
const EARTH_SCALE = 6378137;

/** 默认经向网格数。 */
const DEFAULT_GRID_X = 48;
/** 默认纬向网格数。 */
const DEFAULT_GRID_Y = 32;

export default FloodWater;
