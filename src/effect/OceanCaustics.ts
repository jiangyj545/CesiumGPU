import type { IPass } from '../scene/passes/IPass';
import type { Context } from '../renderer/Context';
import { ecefFromLonLatHeight } from '../terrain/TerrainProvider';
import {
  CAUSTICS_PARAMS_UBO_BYTES,
  OCEAN_CAUSTICS_GEN_FINE_SHALLOW,
  OCEAN_CAUSTICS_GEN_FINE_DEEP,
  OCEAN_CAUSTICS_GEN_BROAD_SHALLOW,
  OCEAN_CAUSTICS_GEN_BROAD_DEEP,
} from '../shaders/OceanCausticsShader';

/**
 * FFTDataSource - 焦散对 FFT 波场的解耦视图（B1 集成由 OceanFFTWater 适配，
 * 本模块不得 import/修改 OceanFFTWater）。
 */
export interface FFTDataSource {
  /** 导数纹理视图：rgba16float 2d-array，(dDy/dx, dDy/dz, dDx/dx, dDz/dz)。 */
  readonly derivativeTexture: GPUTextureView | null;
  /** 每级联平铺尺寸（米），如 [733, 157, 33.3, 7.1]。 */
  readonly cascadeSizes: readonly number[];
  /** FFT 边长（如 256）。 */
  readonly fftSize: number;
}

export interface OceanCausticsOptions {
  /** WebGPU 上下文（通常 `viewer.scene.context`）。 */
  context: Context;
  /** 沙洲中心经纬度（度），必填。 */
  center: [number, number];
  /** 沙洲边长（米，焦散纹理覆盖域）。Default: `400` */
  floorSizeM?: number;
  /** 沙洲面在水下的深度（米，向下偏移 ECEF 模长）。Default: `6` */
  floorDepthM?: number;
  /** 太阳方向（指向太阳，归一化 ECEF）。Default: `[0.4, 0.5, 0.77]` */
  sunDirection?: [number, number, number];
  /** 浅焦面深度（米）。Default: `2.5` */
  shallowDepthM?: number;
  /** 深焦面深度（米）。Default: `floorDepthM` */
  deepDepthM?: number;
}

/** 一个生成层：管线 + 网格索引缓冲。 */
interface GenLayerUnit {
  pipeline: GPURenderPipeline;
  indexBuffer: GPUBuffer;
  indexCount: number;
}

/** 每纬度度数对应米数（小范围线性换算，与 OceanFFTWater 一致）。 */
const METERS_PER_DEG = 111320;
/** 焦散纹理边长（texel）。400m 覆盖 → ~20cm/texel，可分辨米级焦散网。 */
const CAUSTICS_RES = 2048;
/** 细/宽级联索引（tidewater fine=级联数-1、broad=再上一层）。 */
const FINE_CASCADE = 3;
const BROAD_CASCADE = 2;

/**
 * OceanCaustics - 海面焦散生成 pass（移植 tidewater Caustics.js——光子溅射
 * 光栅化，Evan Wallace 法）。
 *
 * 覆盖「焦散纹理覆盖域 + 边缘 margin」的细网格离屏 render pass：顶点即波面
 * 点，太阳光线经波浪法线折射落到 D 米深平面，顶点改置于落点；片元加性混合
 * 写（波面元面积 / 海底像元面积）= 光集中度，聚焦褶皱自然形成亮焦散网。
 * 两个焦平面（R=浅、G=深）写进同一张 rgba16float 纹理，宽级联的宽聚焦以
 * 0.6*(I-1) 加性烘焙（查询期 mix(1,br,0.6) 的等价，均值保持 ~1）。
 *
 * 与相机无关（正交投射平面米 → NDC），故不实现 updateCamera；导数纹理经
 * FFTDataSource 解耦获取，每帧随 OceanFFTWater 的波场同步更新。
 *
 * T5 真 HDR 注：本 pass 渲染目标是自持的离屏 rgba16float 焦散纹理（execute
 * 忽略 Scene 下发的 target），场景目标格式与其管线无关，故不接 EffectManager
 * 的 setTargetFormat 协议（CausticsFloor/BowSpray 已接）。
 *
 * @example
 * ```typescript
 * const caustics = scene.effects.add(new OceanCaustics(scene.context, {
 *   center: [121.5, 31.0],
 * }));
 * caustics.bindFFT(ocean); // ocean 需满足 FFTDataSource（B1 集成）
 * ```
 */
export class OceanCaustics implements IPass {
  readonly label = 'OceanCaustics';

  /** 可见性开关（EffectManager 每帧检查）。 */
  show = true;

  private _context: Context;
  private _fft: FFTDataSource | null = null;

  private readonly _floorSizeM: number;
  private readonly _floorDepthM: number;
  private readonly _shallowD: number;
  private readonly _deepD: number;
  private readonly _sunLocal: [number, number, number];

  // ── GPU 资源 ──
  private _paramsUBO: GPUBuffer | null = null;
  /** sunLocal(+depths) 静态段构造期填好；tiles 段每帧随绑定 FFT 更新。 */
  private readonly _paramsF32 = new Float32Array(CAUSTICS_PARAMS_UBO_BYTES / 4);
  private _target: GPUTexture | null = null;
  private _targetView: GPUTextureView | null = null;
  private _sampler: GPUSampler | null = null;
  private _bgl: GPUBindGroupLayout | null = null;
  private _bindGroup: GPUBindGroup | null = null;
  private _boundDeriv: GPUTextureView | null = null;
  private readonly _layers: GenLayerUnit[] = [];
  /** 初始化失败闸：一次 console.error 后禁用，防每帧刷屏（OceanFFTWater 同款）。 */
  private _failed = false;

  constructor(context: Context, options: OceanCausticsOptions) {
    const [lon, lat] = options.center;
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
      throw new Error(`OceanCaustics: center 非法 [${lon}, ${lat}]`);
    }

    this._context = context;
    this._floorSizeM = options.floorSizeM ?? 400;
    this._floorDepthM = options.floorDepthM ?? 6;
    this._shallowD = options.shallowDepthM ?? 2.5;
    this._deepD = options.deepDepthM ?? options.floorDepthM ?? 6;

    // 太阳方向 ECEF → 局部 ENU（x 东 / y 上 / z 北）。
    // 2026-09-30 实锤修复：_enuBasis 返回「每米归一化增量」（模长 ~1.6e-7），
    // 单位太阳向量与之点积 ≈ 1e-7 ≈ 0——sunLocal 全零使折射向量归零、焦散
    // 生成四层全部画不出（纹理读回全 0）、水底自始全黑。点积前须归一化。
    const sun = OceanCaustics._normalize(options.sunDirection ?? [0.4, 0.5, 0.77]);
    const enu = OceanCaustics._enuBasis(lon, lat);
    const nrm = (v: [number, number, number]): [number, number, number] => {
      const m = Math.hypot(v[0], v[1], v[2]) || 1;
      return [v[0] / m, v[1] / m, v[2] / m];
    };
    const dot = (a: [number, number, number], b: [number, number, number]): number =>
      a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    this._sunLocal = [
      dot(sun, nrm(enu.east)), dot(sun, nrm(enu.up)), dot(sun, nrm(enu.north)),
    ];

    // 静态 uniform 段（tiles.x/y/w 依赖绑定 FFT，execute 时填）
    const P = this._paramsF32;
    P.set(this._sunLocal, 0);
    P[8] = this._shallowD;
    P[9] = this._deepD;
    // 宽层焦平面：tidewater fine [1.2,4] → broad [3,9] 的比例
    P[10] = Math.min(this._shallowD * 2.5, this._deepD);
    P[11] = this._deepD * 2.25;
  }

  // ── 公开 API ─────────────────────────────────────────────────────────────

  /** 绑定 FFT 波场数据源（导数纹理 / 级联尺寸）。可重复调用。 */
  bindFFT(fft: FFTDataSource): void {
    this._fft = fft;
  }

  /** 生成的双焦平面纹理（R=浅焦面 G=深焦面，rgba16float 2d），供外部水下着色采样。 */
  get causticsTexture(): GPUTextureView | null {
    return this._targetView;
  }

  /** 焦散纹理覆盖域边长（米，纹理 uv [0,1] ↔ 平面米 [-cover/2, cover/2]）。 */
  get coverMeters(): number {
    return this._floorSizeM;
  }

  /** 沙洲面在水下的深度（米）。 */
  get floorDepthMeters(): number {
    return this._floorDepthM;
  }

  /** 浅焦面深度（米）。 */
  get shallowDepthMeters(): number {
    return this._shallowD;
  }

  /** 深焦面深度（米）。 */
  get deepDepthMeters(): number {
    return this._deepD;
  }

  /** @inheritdoc IPass - 幂等初始化（纹理 / 管线族 / 索引缓冲）。 */
  prepare(): void {
    if (this._failed) return;
    try {
      if (this._target && this._layers.length === 4 && this._bgl && this._paramsUBO) return;
      const device = this._context.device;

      this._paramsUBO ??= device.createBuffer({
        label: 'OceanCausticsParamsUBO',
        size: CAUSTICS_PARAMS_UBO_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      // rgba16float 渲染附件 + 采样；HDR 加性混合允许 I > 1 的聚焦峰值
      this._target ??= device.createTexture({
        label: 'OceanCausticsTexture',
        size: { width: CAUSTICS_RES, height: CAUSTICS_RES },
        format: 'rgba16float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this._targetView ??= this._target.createView({
        label: 'OceanCausticsTextureView',
        dimension: '2d',
      });

      this._sampler ??= device.createSampler({
        label: 'OceanCausticsSampler',
        addressModeU: 'repeat',
        addressModeV: 'repeat',
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: 'linear',
      });

      this._bgl ??= device.createBindGroupLayout({
        label: 'OceanCausticsBGL',
        entries: [
          // visibility 按实际着色阶段给全：VS 采样导数纹理 / 读 uniform，
          // 缺 VERTEX 会导致管线验证失败并污染整帧（FloodWater 前车之鉴）
          {
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform', minBindingSize: CAUSTICS_PARAMS_UBO_BYTES },
          },
          {
            binding: 1,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            texture: { sampleType: 'float', viewDimension: '2d-array' },
          },
          {
            binding: 2,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            sampler: { type: 'filtering' },
          },
        ],
      });

      if (this._layers.length === 0) {
        const specs = [
          { code: OCEAN_CAUSTICS_GEN_FINE_SHALLOW, grid: 512 },
          { code: OCEAN_CAUSTICS_GEN_FINE_DEEP, grid: 512 },
          { code: OCEAN_CAUSTICS_GEN_BROAD_SHALLOW, grid: 128 },
          { code: OCEAN_CAUSTICS_GEN_BROAD_DEEP, grid: 128 },
        ];
        for (const s of specs) {
          const module = this._context.createShaderModule(s.code, 'OceanCausticsGen');
          const pipeline = device.createRenderPipeline({
            label: 'OceanCausticsGenPipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this._bgl] }),
            // 无顶点缓冲：网格顶点由 vertex_index 展开（tidewater 同款）
            vertex: { module, entryPoint: 'vs_main' },
            fragment: {
              module,
              entryPoint: 'fs_main',
              targets: [
                {
                  format: 'rgba16float',
                  blend: {
                    color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                  },
                },
              ],
            },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
          });
          this._layers.push({ pipeline, ...this._buildIndexBuffer(s.grid) });
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
   * 每帧执行：一个离屏 render pass（clear + 4 层加性 draw），生成双焦平面
   * 焦散纹理。未绑定 FFT / 导数纹理未就绪时跳过（纹理保留上一帧内容）。
   */
  execute(encoder: GPUCommandEncoder, _target: GPUTextureView, _time: number): void {
    this.prepare();
    if (this._failed) return;

    const fft = this._fft;
    const deriv = fft?.derivativeTexture ?? null;
    const sizes = fft?.cascadeSizes;
    if (!deriv || !sizes || sizes.length < 4 || !this._targetView
      || !this._bgl || !this._paramsUBO || this._layers.length === 0) {
      return;
    }
    const fineSize = sizes[FINE_CASCADE];
    const broadSize = sizes[BROAD_CASCADE];
    if (!Number.isFinite(fineSize) || fineSize <= 0
      || !Number.isFinite(broadSize) || broadSize <= 0) {
      return;
    }

    // tiles 段随绑定的 FFT 更新（bindFFT 可晚于构造）
    const P = this._paramsF32;
    P[4] = fineSize;
    P[5] = broadSize;
    P[6] = this._floorSizeM;
    P[7] = fft.fftSize;

    if (this._boundDeriv !== deriv || !this._bindGroup) {
      this._bindGroup = this._context.device.createBindGroup({
        layout: this._bgl,
        entries: [
          { binding: 0, resource: { buffer: this._paramsUBO } },
          { binding: 1, resource: deriv },
          { binding: 2, resource: this._sampler! },
        ],
      });
      this._boundDeriv = deriv;
    }

    this._context.writeBuffer(this._paramsUBO, 0, this._paramsF32);

    const pass = encoder.beginRenderPass({
      label: this.label,
      colorAttachments: [
        {
          view: this._targetView,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
        },
      ],
    });
    pass.setBindGroup(0, this._bindGroup);
    for (const layer of this._layers) {
      pass.setPipeline(layer.pipeline);
      pass.setIndexBuffer(layer.indexBuffer, 'uint32');
      pass.drawIndexed(layer.indexCount);
    }
    pass.end();
  }

  /** 释放全部 GPU 资源；调用后本 pass 不可再用。 */
  destroy(): void {
    this._paramsUBO?.destroy();
    this._target?.destroy();
    for (const layer of this._layers) {
      layer.indexBuffer.destroy();
    }
    this._paramsUBO = null;
    this._target = null;
    this._targetView = null;
    this._sampler = null;
    this._bgl = null;
    this._bindGroup = null;
    this._boundDeriv = null;
    this._layers.length = 0;
  }

  // ── 内部 ─────────────────────────────────────────────────────────────────

  /** grid×grid 四边形索引缓冲（tidewater CausticLayer 同款布局）。 */
  private _buildIndexBuffer(grid: number): { indexBuffer: GPUBuffer; indexCount: number } {
    const row = grid + 1;
    const idx = new Uint32Array(grid * grid * 6);
    let i = 0;
    for (let y = 0; y < grid; y++) {
      for (let x = 0; x < grid; x++) {
        const a = y * row + x;
        idx[i++] = a;
        idx[i++] = a + 1;
        idx[i++] = a + row;
        idx[i++] = a + row;
        idx[i++] = a + 1;
        idx[i++] = a + row + 1;
      }
    }
    const buffer = this._context.device.createBuffer({
      label: 'OceanCausticsIndices',
      size: idx.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    this._context.writeBuffer(buffer, 0, idx);
    return { indexBuffer: buffer, indexCount: idx.length };
  }

  /**
   * 中心点 ENU 基，单位为「每米归一化 ECEF 增量」（数值差分自动含极轴压扁
   * ——不可用 atan2 地心纬度反推重建，项目坐标约定血泪坑）。
   */
  private static _enuBasis(
    lon: number,
    lat: number
  ): { east: [number, number, number]; north: [number, number, number]; up: [number, number, number] } {
    const p0 = OceanCaustics._ecef(lon, lat, 0);

    const dLon = 1e-4; // 度
    const pE = OceanCaustics._ecef(lon + dLon, lat, 0);
    const stepLonM = dLon * METERS_PER_DEG * Math.cos((lat * Math.PI) / 180);
    const east = OceanCaustics._scale(OceanCaustics._sub(pE, p0), 1 / stepLonM);

    const dLat = 1e-4; // 度
    const pN = OceanCaustics._ecef(lon, lat + dLat, 0);
    const north = OceanCaustics._scale(OceanCaustics._sub(pN, p0), 1 / (dLat * METERS_PER_DEG));

    const pU = OceanCaustics._ecef(lon, lat, 1.0); // +1 米
    const up = OceanCaustics._sub(pU, p0);

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
}

export default OceanCaustics;
