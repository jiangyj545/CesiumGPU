import type { IPass } from '../scene/passes/IPass';
import type { Camera } from '../scene/Camera';
import type { Context } from '../renderer/Context';
import type { FFTDataSource, OceanCaustics } from './OceanCaustics';
import type { SeabedHeightField } from './SeabedHeightField';
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
  /** 沙洲面在水下的深度（米，向下偏移 ECEF 模长，仅平底模式）。Default: `6` */
  depthM?: number;
  /** 太阳方向（指向太阳，归一化 ECEF）。Default: `[0.4, 0.5, 0.77]` */
  sunDirection?: [number, number, number];
  /** 湿沙线强度（0=关；bindSeabed 绑定高度场后生效）。Default: `0` */
  shoreStrength?: number;
  /** swash 岸线冲刷强度（0=关，1=全效果；setSwash 同款语义）。Default: `0` */
  swash?: number;
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
 * floor.setSwash(0.8);         // swash 岸线冲刷（run-up 涨落 + 退潮白沫）
 * ```
 */
export class CausticsFloor implements IPass {
  readonly label = 'CausticsFloor';

  /** 可见性开关（EffectManager 每帧检查）。 */
  show = true;

  /** 湿沙线强度（0=关；每帧写入 UBO P[24]，仅 bindSeabed 绑定高度场后有视觉意义）。 */
  shoreStrength: number;

  /** swash 岸线冲刷强度（0=关，1=全效果；每帧写入 UBO P[27]，shader 侧 wet.w>0.001 门控）。 */
  swash: number;

  private _context: Context;
  private _camera: Camera | null = null;
  private _fft: FFTDataSource | null = null;
  private _source: OceanCaustics | null = null;
  private _seabed: SeabedHeightField | null = null;
  /** seabedUv 仿射（P[28..31]）：scale 0 = 未绑定 → suv 恒 0 落界外，FS 回退顶点值。 */
  private _seabedUvOffset: [number, number] = [0.5, 0.5];
  private _seabedUvScale: [number, number] = [0, 0];

  private readonly _centerLon: number;
  private readonly _centerLat: number;
  private readonly _sizeM: number;
  private readonly _depthM: number;
  private readonly _sunLocal: [number, number, number];
  private readonly _upDir: [number, number, number];
  /** ENU east/north 基（每米归一化 ECEF 增量，构造期缓存，bindSeabed uv 仿射用）。 */
  private readonly _eastDir: [number, number, number];
  private readonly _northDir: [number, number, number];
  private _geometry: { vertices: Float32Array; indices: Uint32Array; indexCount: number };

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
  /** 缓存 binding 5 海床纹理视图身份（含 fallback），变更时重建 bind group。 */
  private _bgSeabed: GPUTextureView | null = null;
  /** 湿沙占位纹理（1×1 r32float 全 -1000 极深 = 无湿沙）：惰性创建。 */
  private _seabedFallbackTex: GPUTexture | null = null;
  private _seabedFallbackViewCache: GPUTextureView | null = null;
  /** 初始化失败闸：一次 console.error 后禁用，防每帧刷屏（OceanFFTWater 同款）。 */
  private _failed = false;
  /** T5 真 HDR：当前管线目标格式（构造期 = preferredFormat，Scene 逐帧下发后跟随）。 */
  private _targetFormat: GPUTextureFormat;

  constructor(context: Context, options: CausticsFloorOptions) {
    const [lon, lat] = options.center;
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
      throw new Error(`CausticsFloor: center 非法 [${lon}, ${lat}]`);
    }

    this._context = context;
    this._targetFormat = context.preferredFormat;
    this._centerLon = lon;
    this._centerLat = lat;
    this._sizeM = options.sizeM ?? 400;
    this._depthM = options.depthM ?? 6;
    this.shoreStrength = options.shoreStrength ?? 0;
    this.swash = options.swash ?? 0;

    // 太阳方向 ECEF → 局部 ENU；upDir 为「每米归一化 ECEF 增量」（数值差分）
    const enu = CausticsFloor._enuBasis(lon, lat);
    const sun = CausticsFloor._normalize(options.sunDirection ?? [0.4, 0.5, 0.77]);
    const dot = (a: [number, number, number], b: [number, number, number]): number =>
      a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    this._sunLocal = [dot(sun, enu.east), dot(sun, enu.up), dot(sun, enu.north)];
    this._upDir = enu.up;
    this._eastDir = enu.east;
    this._northDir = enu.north;

    this._geometry = CausticsFloor._buildGeometry(lon, lat, this._sizeM, FLOOR_SEGMENTS, null);
  }

  // ── 公开 API ─────────────────────────────────────────────────────────────

  /**
   * 绑定海床高度场（海洋 L4-B 起伏几何）：立即 CPU 重建网格，顶点 ECEF 改为
   * `ecefFromLonLatHeight(lon, lat, seabedH)` 海拔投影（引擎单一来源换算），
   * 高度米随顶点属性进 VS/FS（湿沙线）。传 `null` 恢复平底（零回归）。
   * 同一 field 引用重复调用为无操作（幂等，不重写 GPU 缓冲）。
   */
  bindSeabed(field: SeabedHeightField | null): void {
    if (this._seabed === field) return;
    this._seabed = field;
    this._geometry = CausticsFloor._buildGeometry(
      this._centerLon, this._centerLat, this._sizeM, FLOOR_SEGMENTS, field
    );
    // prepare 已建过顶点缓冲时原位重写（几何拓扑固定 → 字节数恒定，无需重建）
    if (this._vertexBuffer) {
      this._context.writeBuffer(this._vertexBuffer, 0, CausticsFloor._asBytes(this._geometry.vertices));
    }

    // seabedUv 仿射（P[28..31]，OceanFFTWater.bindSeabed 同款原点差投影）：
    // seabed 局部平面与沙面 planeXZ 同为 ENU 米域但原点不同，须做原点差换算
    // ——基向量为「每米归一化增量」，需最小二乘投影还原米数（直接点积差 R 倍），
    // uv = 原点差 / sizeMeters + 0.5。scale 0 = 未绑定 → FS 回退顶点 seabedH。
    if (field) {
      const p0 = CausticsFloor._ecef(this._centerLon, this._centerLat, 0);
      const c = field.centerRadial;
      const dx = p0[0] - c.x, dy = p0[1] - c.y, dz = p0[2] - c.z;
      const e = this._eastDir;
      const n = this._northDir;
      const eastM = (dx * e[0] + dy * e[1] + dz * e[2]) / (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
      const northM = (dx * n[0] + dy * n[1] + dz * n[2]) / (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      const s = 1 / field.sizeMeters;
      this._seabedUvOffset = [eastM * s + 0.5, northM * s + 0.5];
      this._seabedUvScale = [s, s];
    } else {
      this._seabedUvOffset = [0.5, 0.5];
      this._seabedUvScale = [0, 0];
    }
  }

  /** 绑定 FFT 波场数据源（长波斜率摆动 + 级联尺寸）。可重复调用。 */
  bindFFT(fft: FFTDataSource): void {
    this._fft = fft;
  }

  /**
   * swash 岸线动态冲刷强度（写 UBO wet.w，与 shoreStrength 独立叠加；默认
   * 0 = 逐字节零回归）。非法值钳 [0,1]：NaN/负数 → 0，>1 → 1（T6 接线入口）。
   */
  setSwash(intensity: number): void {
    this.swash = Number.isFinite(intensity) ? Math.min(1, Math.max(0, intensity)) : 0;
  }

  /** 绑定焦散纹理来源（OceanCaustics 实例，读其 causticsTexture / 焦面深度配置）。 */
  bindCaustics(source: OceanCaustics | null): void {
    this._source = source;
  }

  /**
   * T5 真 HDR：目标格式下发（EffectManager 可选协议，Scene.render 逐帧调用
   * 当前渲染目标格式）。与当前格式相同则幂等早退；不同则置空管线由
   * prepare() 惰性重建（幂等——顶点/索引缓冲与格式无关原位复用，bind
   * group 不含目标格式不重建）。未接线的调用路径保持 preferredFormat
   * 旧行为。
   */
  setTargetFormat(format: GPUTextureFormat): void {
    if (this._failed || format === this._targetFormat) return;
    this._targetFormat = format;
    this._pipeline = null;
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
          // r32float 不可过滤高度场（湿沙 FS textureLoad），OceanFFTShader binding 13 同款
          { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
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
              arrayStride: 24, // 6 floats：[ecefX, ecefY, ecefZ, planeX_m, planeZ_m, seabedH_m]
              attributes: [
                { shaderLocation: 0, offset: 0, format: 'float32x3' },  // 海平面基准 ecef
                { shaderLocation: 1, offset: 12, format: 'float32x2' }, // planeXZ（米）
                { shaderLocation: 2, offset: 20, format: 'float32' },   // 海拔米（湿沙线）
              ],
            }],
          },
          fragment: {
            module,
            entryPoint: 'fs_main',
            targets: [
              {
                // T5 真 HDR：跟随 setTargetFormat 下发的场景目标格式
                //（rgba16float FP16 可混合，src-alpha 混合合法）
                format: this._targetFormat,
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
  execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void {
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

    // ── 参数 UBO（128B；P[0..23] 焦面深度/覆盖域与 OceanCaustics 保持一致）──
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
    P[24] = Number.isFinite(this.shoreStrength) ? Math.max(0, this.shoreStrength) : 0; // wet.x 湿沙强度（0=关）
    P[25] = time;                                          // wet.y 效果时间（s，浪痕呼吸）
    P[26] = this._seabed ? 1.0 : 0.0;                      // wet.z 高度模式（1=ECEF 已含海拔）
    P[27] = Number.isFinite(this.swash) ? Math.min(1, Math.max(0, this.swash)) : 0; // wet.w swash 冲刷强度（0=关）
    P[28] = this._seabedUvOffset[0];                       // seabedUv 仿射 offset.xy
    P[29] = this._seabedUvOffset[1];
    P[30] = this._seabedUvScale[0];                        // seabedUv 仿射 scale.zw（0=未绑定）
    P[31] = this._seabedUvScale[1];

    const seabedView = this._seabed?.seabedTexture ?? this._seabedFallbackView();
    if (this._boundCaustics !== caustics || this._boundDeriv !== deriv
      || this._bgSeabed !== seabedView || !this._bindGroup) {
      this._bindGroup = this._context.device.createBindGroup({
        layout: this._bgl!,
        entries: [
          { binding: 0, resource: { buffer: this._cameraUBO } },
          { binding: 1, resource: { buffer: this._paramsUBO } },
          { binding: 2, resource: caustics },
          { binding: 3, resource: deriv },
          { binding: 4, resource: this._sampler() },
          { binding: 5, resource: seabedView },
        ],
      });
      this._boundCaustics = caustics;
      this._boundDeriv = deriv;
      this._bgSeabed = seabedView;
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
    this._seabedFallbackTex?.destroy();
    this._vertexBuffer = null;
    this._indexBuffer = null;
    this._cameraUBO = null;
    this._paramsUBO = null;
    this._pipeline = null;
    this._bgl = null;
    this._bindGroup = null;
    this._boundCaustics = null;
    this._boundDeriv = null;
    this._bgSeabed = null;
    this._seabedFallbackTex = null;
    this._seabedFallbackViewCache = null;
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
   * binding 5 占位纹理视图（惰性创建，视图身份恒一防每帧重建 bind group）：
   * 1×1 r32float 全 -1000（极深 = 无湿沙；仅未绑 field 时使用，湿沙受
   * shoreStrength/wet.z 门控走原路径，OceanFFTSeabedFallback 同款模式）。
   */
  private _seabedFallbackView(): GPUTextureView {
    if (!this._seabedFallbackViewCache) {
      const device = this._context.device;
      this._seabedFallbackTex = device.createTexture({
        label: 'CausticsFloorSeabedFallback',
        size: [1, 1],
        format: 'r32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture(
        { texture: this._seabedFallbackTex },
        new Float32Array([-1000]),
        { bytesPerRow: 4, rowsPerImage: 1 },
        [1, 1, 1],
      );
      this._seabedFallbackViewCache = this._seabedFallbackTex.createView();
    }
    return this._seabedFallbackViewCache;
  }

  /**
   * 贴地经纬正方形网格 → 归一化 ECEF 顶点：每顶点 [ecefX,ecefY,ecefZ,planeX,planeZ,seabedH]。
   *
   * 平底模式（field=null，零回归）：ecef 为海平面基准，seabedH 写 0；
   * 沙面深度在 VS 内经 upDir·(-depthM) 偏移，规避 f32 精度损失。
   *
   * 高度场模式（field≠null，海洋 L4-B）：seabedH = field.sampleHeight(planeX,planeZ)
   * 海拔米，ecef 经引擎 `ecefFromLonLatHeight(lon,lat,seabedH)` 单一来源投影回
   * 压扁世界（禁止手写坐标换算——项目血泪铁律），VS 内 drop=0（wet.z 门）。
   * planeX=东向米 / planeZ=北向米（与焦散纹理/导数采样坐标同系）；
   * 经纬增量用小范围线性换算（Δlon = 米 / (111320·cos lat)）。
   */
  private static _buildGeometry(
    centerLon: number,
    centerLat: number,
    sizeM: number,
    segments: number,
    field: SeabedHeightField | null
  ): { vertices: Float32Array; indices: Uint32Array; indexCount: number } {
    const n = segments + 1;
    const metersPerDegLon = METERS_PER_DEG * Math.cos((centerLat * Math.PI) / 180);
    const vertices = new Float32Array(n * n * 6);
    const tmp: [number, number, number] = [0, 0, 0];

    for (let iy = 0; iy < n; iy++) {
      const planeZ = (iy / segments - 0.5) * sizeM;
      const lat = centerLat + planeZ / METERS_PER_DEG;
      for (let ix = 0; ix < n; ix++) {
        const planeX = (ix / segments - 0.5) * sizeM;
        const lon = centerLon + planeX / metersPerDegLon;
        const h = field ? field.sampleHeight(planeX, planeZ) : 0;
        ecefFromLonLatHeight(lon, lat, h, tmp);
        const v = (iy * n + ix) * 6;
        vertices[v + 0] = tmp[0];
        vertices[v + 1] = tmp[1];
        vertices[v + 2] = tmp[2];
        vertices[v + 3] = planeX;
        vertices[v + 4] = planeZ;
        vertices[v + 5] = h;
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
