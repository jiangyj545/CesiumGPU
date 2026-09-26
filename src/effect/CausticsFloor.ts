import type { IPass } from '../scene/passes/IPass';
import type { Camera } from '../scene/Camera';
import type { Context } from '../renderer/Context';
import type { FFTDataSource, OceanCaustics } from './OceanCaustics';
import { ecefFromLonLatHeight } from '../terrain/TerrainProvider';
import { CAMERA_UBO_SIZE, writeCameraUniforms } from '../renderer/CameraUniforms';
import { CAUSTICS_FLOOR_SHADER, FLOOR_PARAMS_UBO_BYTES } from '../shaders/OceanCausticsShader';

export interface CausticsFloorOptions {
  /** WebGPU 上下文（通常 `viewer.scene.context`）。 */
  context: Context;
  /** 沙洲中心经纬度（度），必填。 */
  center: [number, number];
  /** 沙洲边长（米，贴地经纬网格）。Default: `400` */
  sizeM?: number;
  /** 沙洲面在水下的深度（米，向下偏移 ECEF 模长）。Default: `6` */
  depthM?: number;
  /** 太阳方向（指向太阳，归一化 ECEF）。Default: `[0.4, 0.5, 0.77]` */
  sunDirection?: [number, number, number];
}

/** 每纬度度数对应米数（小范围线性换算，与 OceanFFTWater 一致）。 */
const METERS_PER_DEG = 111320;
/** 沙洲网格每边段数（3.1m/quad @400m）。 */
const FLOOR_SEGMENTS = 128;
/** 沙色（暖沙滩色）与天空/雾色（与 OceanFFTWater 渲染参数一致）。 */
const SAND_COLOR: [number, number, number] = [0.78, 0.70, 0.52];
const SKY_COLOR: [number, number, number] = [0.55, 0.72, 0.88];

/**
 * CausticsFloor - 浅滩沙洲接收面（OceanCaustics 配套小类）。
 *
 * 贴地经纬网格（~400×400m，海平面基准 ECEF + VS 内按 depthM 下移——米级
 * 位移与基准分别过 viewProj 再相加，f32 精度血泪坑同款规避）。着色 = 沙色
 * × 焦散采样（OceanCaustics 生成的双焦平面纹理按真实深度混合，长波斜率
 * 使焦散网随涌浪摇摆）+ 一点深水衰减蓝；半透明混合画在 target 上
 * （loadOp load，无深度附件，片元级射线-球遮挡）。
 *
 * @example
 * ```typescript
 * const floor = scene.effects.add(new CausticsFloor(scene.context, {
 *   center: [121.5, 31.0], sizeM: 400, depthM: 6,
 * }));
 * floor.bindFFT(ocean);        // 长波斜率摆动数据源（bindFFT 同款）
 * floor.bindCaustics(caustics); // 焦散纹理来源（OceanCaustics 实例）
 * ```
 */
export class CausticsFloor implements IPass {
  readonly label = 'CausticsFloor';

  /** 可见性开关（EffectManager 每帧检查）。 */
  show = true;

  private _context: Context;
  private _camera: Camera | null = null;
  private _fft: FFTDataSource | null = null;
  private _source: OceanCaustics | null = null;

  private readonly _sizeM: number;
  private readonly _depthM: number;
  private readonly _sunLocal: [number, number, number];
  private readonly _upDir: [number, number, number];
  private readonly _geometry: { vertices: Float32Array; indices: Uint32Array; indexCount: number };

  // ── GPU 资源 ──
  private _cameraUBO: GPUBuffer | null = null;
  private _paramsUBO: GPUBuffer | null = null;
  private readonly _paramsF32 = new Float32Array(FLOOR_PARAMS_UBO_BYTES / 4);
  private _vertexBuffer: GPUBuffer | null = null;
  private _indexBuffer: GPUBuffer | null = null;
  private _pipeline: GPURenderPipeline | null = null;
  private _bgl: GPUBindGroupLayout | null = null;
  private _bindGroup: GPUBindGroup | null = null;
  /** 缓存当前绑定的两个纹理视图，变更时重建 bind group。 */
  private _boundCaustics: GPUTextureView | null = null;
  private _boundDeriv: GPUTextureView | null = null;
  /** 初始化失败闸：一次 console.error 后禁用，防每帧刷屏（OceanFFTWater 同款）。 */
  private _failed = false;

  constructor(context: Context, options: CausticsFloorOptions) {
    const [lon, lat] = options.center;
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
      throw new Error(`CausticsFloor: center 非法 [${lon}, ${lat}]`);
    }

    this._context = context;
    this._sizeM = options.sizeM ?? 400;
    this._depthM = options.depthM ?? 6;

    // 太阳方向 ECEF → 局部 ENU；upDir 为「每米归一化 ECEF 增量」（数值差分）
    const enu = CausticsFloor._enuBasis(lon, lat);
    const sun = CausticsFloor._normalize(options.sunDirection ?? [0.4, 0.5, 0.77]);
    const dot = (a: [number, number, number], b: [number, number, number]): number =>
      a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    this._sunLocal = [dot(sun, enu.east), dot(sun, enu.up), dot(sun, enu.north)];
    this._upDir = enu.up;

    this._geometry = CausticsFloor._buildGeometry(lon, lat, this._sizeM, FLOOR_SEGMENTS);
  }

  // ── 公开 API ─────────────────────────────────────────────────────────────

  /** 绑定 FFT 波场数据源（长波斜率摆动 + 级联尺寸）。可重复调用。 */
  bindFFT(fft: FFTDataSource): void {
    this._fft = fft;
  }

  /** 绑定焦散纹理来源（OceanCaustics 实例，读其 causticsTexture / 焦面深度配置）。 */
  bindCaustics(source: OceanCaustics | null): void {
    this._source = source;
  }

  /** @inheritdoc IPass - 由 Scene 在 execute 前注入相机（IPass 可选协议）。 */
  updateCamera(camera: Camera): void {
    this._camera = camera;
  }

  /** @inheritdoc IPass - 幂等初始化（管线 / bindGroup / 网格缓冲）。 */
  prepare(): void {
    if (this._failed) return;
    try {
      if (this._pipeline && this._vertexBuffer && this._indexBuffer && this._bindGroup) return;
      const device = this._context.device;

      this._cameraUBO ??= device.createBuffer({
        label: 'CausticsFloorCameraUBO',
        size: CAMERA_UBO_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this._paramsUBO ??= device.createBuffer({
        label: 'CausticsFloorParamsUBO',
        size: FLOOR_PARAMS_UBO_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      this._bgl ??= device.createBindGroupLayout({
        label: 'CausticsFloorBGL',
        entries: [
          // visibility 按实际着色阶段给全（缺 VERTEX/FRAGMENT 均会验证失败）
          {
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform', minBindingSize: CAMERA_UBO_SIZE },
          },
          {
            binding: 1,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform', minBindingSize: FLOOR_PARAMS_UBO_BYTES },
          },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
          { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
          { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        ],
      });

      if (!this._pipeline) {
        const module = this._context.createShaderModule(CAUSTICS_FLOOR_SHADER, 'CausticsFloorShader');
        this._pipeline = device.createRenderPipeline({
          label: 'CausticsFloorPipeline',
          layout: device.createPipelineLayout({ bindGroupLayouts: [this._bgl] }),
          vertex: {
            module,
            entryPoint: 'vs_main',
            buffers: [{
              arrayStride: 20, // 5 floats：[ecefX, ecefY, ecefZ, planeX_m, planeZ_m]
              attributes: [
                { shaderLocation: 0, offset: 0, format: 'float32x3' },  // 海平面基准 ecef
                { shaderLocation: 1, offset: 12, format: 'float32x2' }, // planeXZ（米）
              ],
            }],
          },
          fragment: {
            module,
            entryPoint: 'fs_main',
            targets: [
              {
                format: this._context.preferredFormat,
                blend: {
                  color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                  alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                },
              },
            ],
          },
          primitive: { topology: 'triangle-list', cullMode: 'none' },
          // 无深度附件：遮挡在片元内用射线-球求交处理（与 OceanFFT/FloodWater 一致）
        });
      }

      if (!this._vertexBuffer) {
        const vertices = this._geometry.vertices;
        this._vertexBuffer = device.createBuffer({
          label: 'CausticsFloorVertices',
          size: Math.max(4, vertices.byteLength),
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        if (vertices.byteLength > 0) {
          this._context.writeBuffer(this._vertexBuffer, 0, CausticsFloor._asBytes(vertices));
        }
      }

      if (!this._indexBuffer) {
        const indices = this._geometry.indices;
        this._indexBuffer = device.createBuffer({
          label: 'CausticsFloorIndices',
          size: Math.max(4, indices.byteLength),
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        if (indices.byteLength > 0) {
          this._context.writeBuffer(this._indexBuffer, 0, CausticsFloor._asBytes(indices));
        }
      }
    } catch (e) {
      // execute 每帧重入 prepare：初始化抛错后若不设闸会每帧重试刷屏，
      // 且 GPU 资源已部分创建，重试必然同样失败——一次报错并禁用本效果。
      this._failed = true;
      console.error(`[${this.label}] 初始化失败，效果已禁用:`, e);
    }
  }

  /**
   * 每帧执行：上传相机 / 参数 UBO 后在 `target`（swapchain view）上以
   * `loadOp: 'load'` 叠加绘制沙面（不清除已合成的场景）。
   * 未绑定焦散纹理或 FFT 数据源时跳过（B1 集成两 bind 均需就位）。
   */
  execute(encoder: GPUCommandEncoder, target: GPUTextureView, _time: number): void {
    this.prepare();
    if (this._failed) return;

    const cam = this._camera;
    const caustics = this._source?.causticsTexture ?? null;
    const deriv = this._fft?.derivativeTexture ?? null;
    const sizes = this._fft?.cascadeSizes;
    if (!cam || !caustics || !deriv || !sizes || sizes.length < 2
      || !this._pipeline || !this._cameraUBO || !this._paramsUBO
      || !this._vertexBuffer || !this._indexBuffer || this._geometry.indexCount === 0) {
      return;
    }

    // ── 相机 UBO（布局单一来源：src/renderer/CameraUniforms.ts）──────────
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
    this._context.writeBuffer(this._cameraUBO, 0, camData);

    // ── 参数 UBO（96B；焦面深度/覆盖域与 OceanCaustics 配置保持一致）────
    const P = this._paramsF32;
    P.set(this._sunLocal, 0);
    P.set(this._upDir, 4);
    P[8]  = this._source?.coverMeters ?? this._sizeM;      // spacing.x 焦散纹理覆盖
    P[9]  = this._sizeM;                                   // spacing.y 沙洲边长
    P[10] = this._depthM;                                  // spacing.z 沙面深度
    P[11] = sizes[0];                                      // spacing.w 级联0平铺
    P[12] = this._source?.shallowDepthMeters ?? 2.5;       // depths.x 浅焦面
    P[13] = this._source?.deepDepthMeters ?? this._depthM; // depths.y 深焦面
    P[14] = sizes[1];                                      // depths.z 级联1平铺
    P[15] = 0.92;                                          // depths.w 基础 alpha
    P.set(SAND_COLOR, 16);
    P.set(SKY_COLOR, 20);

    if (this._boundCaustics !== caustics || this._boundDeriv !== deriv || !this._bindGroup) {
      this._bindGroup = this._context.device.createBindGroup({
        layout: this._bgl!,
        entries: [
          { binding: 0, resource: { buffer: this._cameraUBO } },
          { binding: 1, resource: { buffer: this._paramsUBO } },
          { binding: 2, resource: caustics },
          { binding: 3, resource: deriv },
          { binding: 4, resource: this._sampler() },
        ],
      });
      this._boundCaustics = caustics;
      this._boundDeriv = deriv;
    }

    this._context.writeBuffer(this._paramsUBO, 0, this._paramsF32);

    const pass = encoder.beginRenderPass({
      label: this.label,
      colorAttachments: [
        { view: target, loadOp: 'load', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
      ],
    });
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, this._bindGroup);
    pass.setVertexBuffer(0, this._vertexBuffer);
    pass.setIndexBuffer(this._indexBuffer, 'uint32');
    pass.drawIndexed(this._geometry.indexCount);
    pass.end();
  }

  /** 释放全部 GPU 资源；调用后本 pass 不可再用。 */
  destroy(): void {
    for (const b of [this._vertexBuffer, this._indexBuffer, this._cameraUBO, this._paramsUBO]) {
      b?.destroy();
    }
    this._vertexBuffer = null;
    this._indexBuffer = null;
    this._cameraUBO = null;
    this._paramsUBO = null;
    this._pipeline = null;
    this._bgl = null;
    this._bindGroup = null;
    this._boundCaustics = null;
    this._boundDeriv = null;
  }

  // ── 内部 ─────────────────────────────────────────────────────────────────

  /** repeat 寻址过滤采样器（导数级联平铺 + 焦散纹理共用；重建 bind group 时按需创建）。 */
  private _sampler(): GPUSampler {
    return this._context.device.createSampler({
      label: 'CausticsFloorSampler',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    });
  }

  /**
   * 贴地经纬正方形网格 → 归一化 ECEF 顶点：每顶点 [ecefX,ecefY,ecefZ,planeX,planeZ]
   * （ecef 为海平面基准；沙面深度在 VS 内经 upDir 偏移，规避 f32 精度损失）。
   * planeX=东向米 / planeZ=北向米（与焦散纹理/导数采样坐标同系）；
   * 经纬增量用小范围线性换算（Δlon = 米 / (111320·cos lat)）。
   */
  private static _buildGeometry(
    centerLon: number,
    centerLat: number,
    sizeM: number,
    segments: number
  ): { vertices: Float32Array; indices: Uint32Array; indexCount: number } {
    const n = segments + 1;
    const metersPerDegLon = METERS_PER_DEG * Math.cos((centerLat * Math.PI) / 180);
    const vertices = new Float32Array(n * n * 5);
    const tmp: [number, number, number] = [0, 0, 0];

    for (let iy = 0; iy < n; iy++) {
      const planeZ = (iy / segments - 0.5) * sizeM;
      const lat = centerLat + planeZ / METERS_PER_DEG;
      for (let ix = 0; ix < n; ix++) {
        const planeX = (ix / segments - 0.5) * sizeM;
        const lon = centerLon + planeX / metersPerDegLon;
        ecefFromLonLatHeight(lon, lat, 0, tmp);
        const v = (iy * n + ix) * 5;
        vertices[v + 0] = tmp[0];
        vertices[v + 1] = tmp[1];
        vertices[v + 2] = tmp[2];
        vertices[v + 3] = planeX;
        vertices[v + 4] = planeZ;
      }
    }

    const quads = segments * segments;
    const indices = new Uint32Array(quads * 6);
    let o = 0;
    for (let iy = 0; iy < segments; iy++) {
      for (let ix = 0; ix < segments; ix++) {
        const r0 = iy * n + ix;
        const r1 = (iy + 1) * n + ix;
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

  /**
   * 中心点 ENU 基，单位为「每米归一化 ECEF 增量」（数值差分自动含极轴压扁
   * ——不可用 atan2 地心纬度反推重建，项目坐标约定血泪坑）。
   */
  private static _enuBasis(
    lon: number,
    lat: number
  ): { east: [number, number, number]; north: [number, number, number]; up: [number, number, number] } {
    const p0 = CausticsFloor._ecef(lon, lat, 0);

    const dLon = 1e-4; // 度
    const pE = CausticsFloor._ecef(lon + dLon, lat, 0);
    const stepLonM = dLon * METERS_PER_DEG * Math.cos((lat * Math.PI) / 180);
    const east = CausticsFloor._scale(CausticsFloor._sub(pE, p0), 1 / stepLonM);

    const dLat = 1e-4; // 度
    const pN = CausticsFloor._ecef(lon, lat + dLat, 0);
    const north = CausticsFloor._scale(CausticsFloor._sub(pN, p0), 1 / (dLat * METERS_PER_DEG));

    const pU = CausticsFloor._ecef(lon, lat, 1.0); // +1 米
    const up = CausticsFloor._sub(pU, p0);

    return { east, north, up };
  }

  private static _ecef(lon: number, lat: number, h: number): [number, number, number] {
    const out: [number, number, number] = [0, 0, 0];
    ecefFromLonLatHeight(lon, lat, h, out);
    return out;
  }

  private static _sub(a: [number, number, number], b: [number, number, number]): [number, number, number] {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  }

  private static _scale(v: [number, number, number], s: number): [number, number, number] {
    return [v[0] * s, v[1] * s, v[2] * s];
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

export default CausticsFloor;
