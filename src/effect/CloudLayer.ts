import type { IPass } from '../scene/passes/IPass';
import type { Context } from '../renderer/Context';
import { Camera } from '../scene/Camera';
import { Matrix4 } from '../math/Matrix4';
import { Cartesian3 } from '../math/Cartesian3';
import { ecefFromLonLatHeight } from '../terrain/TerrainProvider';
import {
  CLOUD_RENDER_WGSL,
  CLOUD_SHADOW_BAKE_WGSL,
  CLOUD_CAMERA_UBO_BYTES,
  CLOUD_PARAMS_UBO_BYTES,
  CLOUD_SHADOW_PARAMS_UBO_BYTES,
} from '../shaders/CloudShader';

/**
 * 云影消费信息（{@link CloudLayer.cloudShadow}）。
 * 数值均为归一化 ECEF；eastDir/northDir 为单位方向基，
 * 米数 = dot(d, 基) * Camera.EARTH_SCALE。
 */
export interface CloudShadowInfo {
  /** 影贴图视图（rgba8unorm，R 通道 = 云密度，uv [0,1]²）。 */
  view: GPUTextureView;
  /** 云高平面中心（归一化 ECEF）。 */
  centerEcef: readonly [number, number, number];
  /** ENU 东基（单位方向）。 */
  eastDir: readonly [number, number, number];
  /** ENU 北基（单位方向）。 */
  northDir: readonly [number, number, number];
  /** 云高球半径（归一化，worldPos 沿太阳方向上投的目标球面）。 */
  cloudRadiusNorm: number;
  /** 影贴图覆盖边长（米）。 */
  extentM: number;
  /** 指向太阳（归一化）。 */
  sunDirection: readonly [number, number, number];
}

/**
 * 云影接线资源（{@link CloudLayer.cloudShadowSource}，仿
 * OceanAtmosphereBinding 的「身份稳定」契约——prepare 后同一组对象引用，
 * LightingPass 的 bind group 以身份做缓存键）。
 * ubo 为 96B CloudShadowParams UBO（布局单一来源 CloudShader.ts），
 * CloudLayer 每帧写入；消费方只读。
 */
export interface CloudShadowSourceBinding {
  view: GPUTextureView;
  ubo: GPUBuffer;
  sampler: GPUSampler;
}

export interface CloudLayerOptions {
  /** WebGPU 上下文（通常 `viewer.scene.context`）。 */
  context: Context;
  /** 场景中心经纬度（度）——云场 / 云影覆盖区锚点。 */
  center: [number, number];
  /** 指向太阳（归一化前任意长度，内部归一化）。Default: `[0.4, 0.5, 0.77]` */
  sunDirection?: [number, number, number];
  /** 云底离地高度（米）。Default: `2200` */
  cloudHeightM?: number;
  /** 云量 0-1（0 晴空，1 满天）。Default: `0.5` */
  coverage?: number;
  /** 云体浓度乘数（α 系数）。Default: `0.92` */
  density?: number;
  /** 风向（度，云漂移「去往」方向，0 = 东，90 = 北）。Default: `0` */
  windDirectionDeg?: number;
  /** 风速（米/秒）。Default: `9` */
  windSpeedMps?: number;
  /** 云底色（rgb 0-1）。Default: `[0.62, 0.64, 0.7]` */
  tintLow?: [number, number, number];
  /** 云顶色（rgb 0-1）。Default: `[1.02, 1.0, 0.97]` */
  tintHigh?: [number, number, number];
  /** 云影覆盖边长（米）。Default: `8192` */
  shadowExtentM?: number;
  /** 影贴图边长（像素，8 的倍数）。Default: `512` */
  shadowResolution?: number;
  /** 云影强度 0-1（0 = 无影）。Default: `0.65` */
  shadowIntensity?: number;
  /** 云场跟随相机（覆盖区中心 = 相机星下点）。Default: `false` */
  followCamera?: boolean;
  /** 地平线渐隐强度 0-1（0 = 关闭）。Default: `1` */
  horizonFade?: number;
  /** 是否绘制。Default: `true` */
  show?: boolean;
}

/**
 * CloudLayer - 体积云层（全屏三角形 × 云高球面求交）+ 云影烘焙
 * （tidewater 观感追赶 Track C）。
 *
 * 渲染：逆 viewProj 重建相机射线，与云高球面（|p| = 地表半径 + 云高）
 * 求交，交点投影到中心切平面取 FBM 云密度；朝太阳 2D 密度差分做
 * 迎光/背光，太阳高度驱动黄昏橙红染色；预乘 alpha（loadOp load）
 * 叠加在已合成场景之上，depthCompare equal 只写背景像素（同
 * AtmosphereSky 安全闸——未 setDepthView 不绘制，绝不盖球体 imagery）。
 *
 * 云影：compute 逐帧把同款密度场正交烘焙进 rgba8unorm（r8unorm 不可作
 * storage），覆盖区以中心（或相机星下点）为中心 extentM² 米；经
 * {@link cloudShadowSource}（身份稳定）接 LightingPass 组 3（地形），
 * 海面接线由主会话 T6 按 {@link cloudShadow} 契约消费。
 */
export class CloudLayer implements IPass {
  readonly label = 'CloudLayer';

  private readonly _context: Context;

  // ── 参数状态（setter 校验后生效）──
  private _centerLon: number;
  private _centerLat: number;
  private _sun: [number, number, number];
  private _cloudHeightM: number;
  private _coverage: number;
  private _density: number;
  private _windDeg: number;
  private _windSpeedMps: number;
  private _tintLow: [number, number, number];
  private _tintHigh: [number, number, number];
  private _shadowExtentM: number;
  private readonly _shadowRes: number;
  private _shadowIntensity: number;
  private _followCamera: boolean;
  private _horizonFade: number;
  private _show: boolean;

  // ── 派生几何（_refreshDerived 维护）──
  private _planeCenter: [number, number, number] = [0, 0, 1];
  private _eastDir: [number, number, number] = [1, 0, 0];
  private _northDir: [number, number, number] = [0, 1, 0];
  private _rcNorm = 1.001;

  // ── 运行状态 ──
  private _windKm: [number, number] = [0, 0];
  private _prevTime = -1;
  private _camera: Camera | null = null;
  private _failed = false;
  private _destroyed = false;
  private _warned = false;
  private _targetFormat: GPUTextureFormat | null = null;
  private _depthSource: GPUTextureView | (() => GPUTextureView | null) | null = null;

  // ── GPU 资源 ──
  private _cameraUBO: GPUBuffer | null = null;
  private _paramsUBO: GPUBuffer | null = null;
  private _shadowUBO: GPUBuffer | null = null;
  private _shadowTex: GPUTexture | null = null;
  private _shadowView: GPUTextureView | null = null;
  private _shadowSampler: GPUSampler | null = null;
  private _bgl: GPUBindGroupLayout | null = null;
  private _pipeline: GPURenderPipeline | null = null;
  private _renderBindGroup: GPUBindGroup | null = null;
  private _bakeBgl: GPUBindGroupLayout | null = null;
  private _bakePipeline: GPUComputePipeline | null = null;
  private _bakeBindGroup: GPUBindGroup | null = null;
  private _cloudShadowSource: CloudShadowSourceBinding | null = null;

  // ── CPU 侧 UBO 载荷（F32 视图）──
  private readonly _camF32 = new Float32Array(CLOUD_CAMERA_UBO_BYTES / 4);
  private readonly _paramsF32 = new Float32Array(CLOUD_PARAMS_UBO_BYTES / 4);
  private readonly _shadowF32 = new Float32Array(CLOUD_SHADOW_PARAMS_UBO_BYTES / 4);
  private readonly _scratchInv = new Matrix4();

  constructor(context: Context, options: CloudLayerOptions) {
    this._context = context;
    this._centerLon = options.center[0];
    this._centerLat = options.center[1];
    this._sun = CloudLayer._normalize(options.sunDirection ?? [0.4, 0.5, 0.77], 'sunDirection');
    this._cloudHeightM = options.cloudHeightM ?? 2200;
    this._coverage = CloudLayer._clamp01(options.coverage ?? 0.5);
    this._density = Math.min(Math.max(options.density ?? 0.92, 0), 4);
    this._windDeg = options.windDirectionDeg ?? 0;
    this._windSpeedMps = options.windSpeedMps ?? 9;
    this._tintLow = options.tintLow ?? [0.62, 0.64, 0.7];
    this._tintHigh = options.tintHigh ?? [1.02, 1.0, 0.97];
    this._shadowExtentM = Math.max(options.shadowExtentM ?? 8192, 256);
    const res = options.shadowResolution ?? 512;
    this._shadowRes = Math.max(Math.ceil(res / 8) * 8, 8);
    this._shadowIntensity = CloudLayer._clamp01(options.shadowIntensity ?? 0.65);
    this._followCamera = options.followCamera ?? false;
    this._horizonFade = CloudLayer._clamp01(options.horizonFade ?? 1);
    this._show = options.show ?? true;
    this._refreshDerived();
  }

  // ── 公开 API ─────────────────────────────────────────────────────────────

  /** 云量 0-1（非法值抛错，仿 OceanFFTWater setter 风格）。 */
  setCoverage(v: number): void {
    if (!Number.isFinite(v)) throw new Error(`CloudLayer.setCoverage: 非法值 ${v}`);
    this._coverage = CloudLayer._clamp01(v);
    this._flushIfReady();
  }

  /** 云体浓度乘数（0-4，α 系数）。 */
  setDensity(v: number): void {
    if (!Number.isFinite(v)) throw new Error(`CloudLayer.setDensity: 非法值 ${v}`);
    this._density = Math.min(Math.max(v, 0), 4);
    this._flushIfReady();
  }

  /** 云底离地高度（米，>= 50）。 */
  setCloudHeightM(v: number): void {
    if (!Number.isFinite(v) || v < 50) throw new Error(`CloudLayer.setCloudHeightM: 非法值 ${v}`);
    this._cloudHeightM = v;
    this._refreshDerived();
    this._flushIfReady();
  }

  /** 风向（度，0 = 东）+ 风速（米/秒）。 */
  setWind(windDirectionDeg: number, windSpeedMps: number): void {
    if (!Number.isFinite(windDirectionDeg) || !Number.isFinite(windSpeedMps)) {
      throw new Error(`CloudLayer.setWind: 非法值 ${windDirectionDeg}, ${windSpeedMps}`);
    }
    this._windDeg = windDirectionDeg;
    this._windSpeedMps = windSpeedMps;
    this._flushIfReady();
  }

  /** 指向太阳（任意长度，内部归一化；零向量抛错）。 */
  setSunDirection(v: [number, number, number]): void {
    this._sun = CloudLayer._normalize(v, 'sunDirection');
    this._flushIfReady();
  }

  /** 云影强度 0-1（0 = 无影，消费端短路恒 1）。 */
  setShadowIntensity(v: number): void {
    if (!Number.isFinite(v)) throw new Error(`CloudLayer.setShadowIntensity: 非法值 ${v}`);
    this._shadowIntensity = CloudLayer._clamp01(v);
    this._flushIfReady();
  }

  /** 云影覆盖边长（米，>= 256；只改仿射系数，不重建纹理）。 */
  setShadowExtentM(v: number): void {
    if (!Number.isFinite(v) || v < 256) throw new Error(`CloudLayer.setShadowExtentM: 非法值 ${v}`);
    this._shadowExtentM = v;
    this._flushIfReady();
  }

  /** 云场跟随相机开关。 */
  setFollowCamera(v: boolean): void {
    this._followCamera = v;
    this._refreshDerived();
    this._flushIfReady();
  }

  /** 地平线渐隐强度 0-1（0 = 关闭渐隐）。 */
  setHorizonFade(v: number): void {
    if (!Number.isFinite(v)) throw new Error(`CloudLayer.setHorizonFade: 非法值 ${v}`);
    this._horizonFade = CloudLayer._clamp01(v);
    this._flushIfReady();
  }

  setShow(v: boolean): void {
    this._show = v;
  }

  /**
   * 深度剔除源（GBufferPass depth32float；同 AtmosphereSky 安全闸）。
   * 未调用时本 pass 不绘制（execute 告警一次后跳过）。
   */
  setDepthView(source: GPUTextureView | (() => GPUTextureView | null)): void {
    this._depthSource = source;
  }

  /**
   * @inheritdoc TargetFormatAware - 场景目标格式切换（T5 真 HDR，照抄
   * CausticsFloor L186 幂等早退模式）：置空渲染管线，prepare() 惰性重建
   * （UBO/纹理/烘焙管线与格式无关原位复用，仅渲染管线含目标格式）。
   */
  setTargetFormat(format: GPUTextureFormat): void {
    if (this._failed || this._destroyed
      || format === (this._targetFormat ?? this._context.preferredFormat)) return;
    this._targetFormat = format;
    this._pipeline = null;
    this._bgl = null;
    this._renderBindGroup = null;
  }

  /** @inheritdoc IPass - 幂等初始化（UBO / 影纹理 / 渲染+烘焙管线）。 */
  prepare(): void {
    if (this._failed || this._destroyed) return;
    try {
      this._prepare();
    } catch (e) {
      // execute 每帧重入 prepare：置闸禁用本 pass（AtmosphereSky 同款）。
      this._failed = true;
      console.error(`[${this.label}] 初始化失败，效果已禁用:`, e);
    }
  }

  /** @inheritdoc IPass - 由 Scene 在 execute 前注入相机。 */
  updateCamera(camera: Camera): void {
    this._camera = camera;
  }

  /**
   * @inheritdoc IPass - 每帧：风漂移推进 → 云影 compute 烘焙 → 渲染
   * （loadOp load 叠加已合成场景，depth equal 只写背景像素）。
   */
  execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void {
    if (this._destroyed) return;
    this.prepare();
    if (this._failed || !this._show) return;
    if (!this._pipeline || !this._renderBindGroup || !this._bakePipeline || !this._bakeBindGroup
      || !this._cameraUBO || !this._paramsUBO || !this._shadowUBO) {
      if (!this._warned) {
        this._warned = true;
        console.warn(`[${this.label}] 管线未就绪，跳过绘制`);
      }
      return;
    }
    const cam = this._camera;
    if (!cam) {
      if (!this._warned) {
        this._warned = true;
        console.warn(`[${this.label}] 未调用 updateCamera，跳过绘制`);
      }
      return;
    }
    const depthView = typeof this._depthSource === 'function' ? this._depthSource() : this._depthSource;
    if (!depthView) {
      if (!this._warned) {
        this._warned = true;
        console.warn(
          `[${this.label}] 未调用 setDepthView（GBuffer 深度），` +
          '跳过绘制：无深度剔除的云层会覆盖球体 imagery。',
        );
      }
      return;
    }

    // 风漂移推进（米 → 公里，FS/烘焙共用偏移）。
    const dt = this._prevTime < 0 ? 0 : Math.max(0, time - this._prevTime);
    this._prevTime = time;
    const rad = this._windDeg * Math.PI / 180;
    this._windKm[0] += Math.cos(rad) * this._windSpeedMps * dt / 1000;
    this._windKm[1] += Math.sin(rad) * this._windSpeedMps * dt / 1000;

    if (this._followCamera) this._refreshDerived(); // 相机跟随：中心逐帧重算
    this._flushUniforms();
    this._writeCameraUniforms(cam);

    // 云影正交烘焙（rgba8unorm storage，R = 密度）。
    const cpass = encoder.beginComputePass({ label: `${this.label}.ShadowBake` });
    cpass.setPipeline(this._bakePipeline);
    cpass.setBindGroup(0, this._bakeBindGroup);
    const wg = this._shadowRes / 8;
    cpass.dispatchWorkgroups(wg, wg);
    cpass.end();

    const pass = encoder.beginRenderPass({
      label: this.label,
      colorAttachments: [{
        view: target,
        loadOp: 'load', // 叠加在已合成场景之上（天空之后）
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
      depthStencilAttachment: {
        view: depthView,
        depthLoadOp: 'load',
        depthStoreOp: 'store',
      },
    });
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, this._renderBindGroup);
    pass.draw(3); // 全屏三角形
    pass.end();
  }

  /** @inheritdoc IPass - 释放全部 GPU 资源；getter 此后返回 null。 */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._cameraUBO?.destroy();
    this._paramsUBO?.destroy();
    this._shadowUBO?.destroy();
    this._shadowTex?.destroy();
    this._cameraUBO = null;
    this._paramsUBO = null;
    this._shadowUBO = null;
    this._shadowTex = null;
    this._shadowView = null;
    this._shadowSampler = null;
    this._bgl = null;
    this._bakeBgl = null;
    this._pipeline = null;
    this._bakePipeline = null;
    this._renderBindGroup = null;
    this._bakeBindGroup = null;
    this._cloudShadowSource = null;
  }

  /**
   * 云影消费信息（主会话 T6 接海面用）。prepare 前失、destroy 后 /
   * 失败闸置位后为 null。仿射公式见 CloudShader.ts 头注释。
   */
  get cloudShadow(): CloudShadowInfo | null {
    if (this._destroyed || this._failed || !this._shadowView) return null;
    return {
      view: this._shadowView,
      centerEcef: [this._planeCenter[0], this._planeCenter[1], this._planeCenter[2]],
      eastDir: [this._eastDir[0], this._eastDir[1], this._eastDir[2]],
      northDir: [this._northDir[0], this._northDir[1], this._northDir[2]],
      cloudRadiusNorm: this._rcNorm,
      extentM: this._shadowExtentM,
      sunDirection: [this._sun[0], this._sun[1], this._sun[2]],
    };
  }

  /**
   * 云影接线资源（接 LightingPass.setCloudShadowSource）。身份稳定契约：
   * prepare 后同组对象引用（bind group 缓存键），destroy 后 null。
   */
  get cloudShadowSource(): CloudShadowSourceBinding | null {
    if (this._destroyed || this._failed) return null;
    return this._cloudShadowSource;
  }

  get show(): boolean {
    return this._show;
  }

  // ── 内部 ─────────────────────────────────────────────────────────────────

  private _prepare(): void {
    if (this._pipeline && this._bakePipeline && this._cloudShadowSource) return;
    const device = this._context.device;
    const usage = GPUTextureUsage;

    this._cameraUBO ??= device.createBuffer({
      label: 'CloudLayer.CameraUBO',
      size: CLOUD_CAMERA_UBO_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._paramsUBO ??= device.createBuffer({
      label: 'CloudLayer.ParamsUBO',
      size: CLOUD_PARAMS_UBO_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._shadowUBO ??= device.createBuffer({
      label: 'CloudLayer.ShadowUBO',
      size: CLOUD_SHADOW_PARAMS_UBO_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    if (!this._shadowTex) {
      this._shadowTex = device.createTexture({
        label: 'CloudLayer.ShadowTex',
        size: [this._shadowRes, this._shadowRes, 1],
        format: 'rgba8unorm',
        usage: usage.TEXTURE_BINDING | usage.STORAGE_BINDING,
      });
      this._shadowView = this._shadowTex.createView();
      this._shadowSampler = device.createSampler({
        label: 'CloudLayer.ShadowSampler',
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
      });
    }

    // ── 渲染管线（全屏三角形；预乘 alpha；depth equal 只写背景像素）──
    if (!this._pipeline) {
      this._bgl = device.createBindGroupLayout({
        label: 'CloudLayer.Render.BGL',
        entries: [
          { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        ],
      });
      const module = this._context.createShaderModule(CLOUD_RENDER_WGSL, 'CloudLayer.Render.Module');
      this._pipeline = device.createRenderPipeline({
        label: 'CloudLayer.Render',
        layout: device.createPipelineLayout({ bindGroupLayouts: [this._bgl] }),
        vertex: { module, entryPoint: 'vs_main' },
        fragment: {
          module,
          entryPoint: 'fs_main',
          targets: [{
            format: this._targetFormat ?? this._context.preferredFormat,
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          }],
        },
        primitive: { topology: 'triangle-list' },
        // GBufferPass 深度格式；VS 输出 NDC z=1 恰与背景深度相等。
        depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'equal' },
      });
      this._renderBindGroup = device.createBindGroup({
        label: 'CloudLayer.Render.BG',
        layout: this._bgl,
        entries: [
          { binding: 0, resource: { buffer: this._cameraUBO } },
          { binding: 1, resource: { buffer: this._paramsUBO } },
        ],
      });
    }

    // ── 云影烘焙 compute ──
    if (!this._bakePipeline) {
      this._bakeBgl = device.createBindGroupLayout({
        label: 'CloudLayer.Bake.BGL',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            storageTexture: { access: 'write-only', format: 'rgba8unorm' },
          },
        ],
      });
      const module = this._context.createShaderModule(CLOUD_SHADOW_BAKE_WGSL, 'CloudLayer.Bake.Module');
      this._bakePipeline = device.createComputePipeline({
        label: 'CloudLayer.Bake',
        layout: device.createPipelineLayout({ bindGroupLayouts: [this._bakeBgl] }),
        compute: { module, entryPoint: 'main' },
      });
      this._bakeBindGroup = device.createBindGroup({
        label: 'CloudLayer.Bake.BG',
        layout: this._bakeBgl,
        entries: [
          { binding: 0, resource: { buffer: this._shadowUBO } },
          { binding: 1, resource: this._shadowView! }, // 上方纹理块已同步创建视图
        ],
      });
    }

    // 身份稳定接线资源（LightingPass bind group 缓存键）。
    this._cloudShadowSource ??= {
      view: this._shadowView!,
      ubo: this._shadowUBO,
      sampler: this._shadowSampler!,
    };
  }

  /** 派生几何：云高平面中心 / ENU 基 / 云球半径。 */
  private _refreshDerived(): void {
    // 中心方向：场景中心星下点，或相机跟随时的相机方向。
    let dir: [number, number, number];
    let surfaceR: number;
    if (this._followCamera && this._camera) {
      const p = this._camera.position;
      const m = Cartesian3.magnitude(p);
      if (m < 1e-9) throw new Error('CloudLayer: 相机位置零向量');
      dir = [p.x / m, p.y / m, p.z / m];
      surfaceR = Camera.surfaceRadius(this._camera.position);
    } else {
      const ground: [number, number, number] = [0, 0, 0];
      ecefFromLonLatHeight(this._centerLon, this._centerLat, 0, ground);
      const m = Math.hypot(ground[0], ground[1], ground[2]);
      dir = [ground[0] / m, ground[1] / m, ground[2] / m];
      surfaceR = m; // 中心星下点椭球半径（归一化）
    }

    this._rcNorm = Math.max(surfaceR, 1e-6) + this._cloudHeightM / Camera.EARTH_SCALE;
    this._planeCenter = [
      dir[0] * this._rcNorm,
      dir[1] * this._rcNorm,
      dir[2] * this._rcNorm,
    ];

    // ENU 基（单位方向，×EARTH_SCALE = 米）：east = ẑ×up，north = up×east。
    const e = [dir[2], 0, -dir[0]];
    const eLen = Math.hypot(e[0], e[1], e[2]);
    if (eLen < 1e-6) { // 极点退化兜底
      this._eastDir = [1, 0, 0];
    } else {
      this._eastDir = [e[0] / eLen, e[1] / eLen, e[2] / eLen];
    }
    const u = this._eastDir;
    this._northDir = [
      dir[1] * u[2] - dir[2] * u[1],
      dir[2] * u[0] - dir[0] * u[2],
      dir[0] * u[1] - dir[1] * u[0],
    ];
  }

  /** UBO 全量上传（params 128B + shadow 96B；布局单一来源 CloudShader.ts）。 */
  private _flushUniforms(): void {
    const p = this._paramsF32;
    // CloudParams
    p[0] = this._sun[0]; p[1] = this._sun[1]; p[2] = this._sun[2]; p[3] = 0;
    p[4] = this._planeCenter[0]; p[5] = this._planeCenter[1];
    p[6] = this._planeCenter[2]; p[7] = this._rcNorm;
    p[8] = this._eastDir[0]; p[9] = this._eastDir[1]; p[10] = this._eastDir[2]; p[11] = 0;
    p[12] = this._northDir[0]; p[13] = this._northDir[1]; p[14] = this._northDir[2]; p[15] = 0;
    p[16] = this._coverage; p[17] = this._density; p[18] = this._horizonFade; p[19] = 0;
    p[20] = this._windKm[0]; p[21] = this._windKm[1]; p[22] = 0; p[23] = 0;
    p[24] = this._tintLow[0]; p[25] = this._tintLow[1]; p[26] = this._tintLow[2]; p[27] = 0;
    p[28] = this._tintHigh[0]; p[29] = this._tintHigh[1]; p[30] = this._tintHigh[2]; p[31] = 0;

    const s = this._shadowF32;
    // CloudShadowParams
    s[0] = this._planeCenter[0]; s[1] = this._planeCenter[1];
    s[2] = this._planeCenter[2]; s[3] = this._rcNorm;
    s[4] = this._eastDir[0]; s[5] = this._eastDir[1]; s[6] = this._eastDir[2]; s[7] = 0;
    s[8] = this._northDir[0]; s[9] = this._northDir[1]; s[10] = this._northDir[2]; s[11] = 0;
    s[12] = Camera.EARTH_SCALE / this._shadowExtentM;
    s[13] = this._shadowIntensity;
    s[14] = this._coverage;
    s[15] = 0;
    s[16] = this._sun[0]; s[17] = this._sun[1]; s[18] = this._sun[2]; s[19] = 0;
    s[20] = this._windKm[0]; s[21] = this._windKm[1]; s[22] = 0; s[23] = 0;

    if (this._paramsUBO) this._context.writeBuffer(this._paramsUBO, 0, p);
    if (this._shadowUBO) this._context.writeBuffer(this._shadowUBO, 0, s);
  }

  /** setter 后即时生效（GPU 资源已创建时）。 */
  private _flushIfReady(): void {
    if (this._destroyed || !this._paramsUBO) return;
    if (!this._followCamera || this._camera) this._refreshDerived();
    this._flushUniforms();
  }

  private _writeCameraUniforms(cam: Camera): void {
    Matrix4.inverse(cam.viewProjectionMatrix, this._scratchInv);
    const f = this._camF32;
    f.set(this._scratchInv.values, 0);
    f[16] = cam.position.x; f[17] = cam.position.y; f[18] = cam.position.z; f[19] = 1;
    f[20] = this._context.drawingBufferWidth;
    f[21] = this._context.drawingBufferHeight;
    f[22] = 0; f[23] = 0;
    this._context.writeBuffer(this._cameraUBO!, 0, f);
  }

  private static _clamp01(v: number): number {
    return Math.min(Math.max(v, 0), 1);
  }

  private static _normalize(v: [number, number, number], name: string): [number, number, number] {
    const len = Math.hypot(v[0], v[1], v[2]);
    if (!Number.isFinite(len) || len < 1e-9) {
      throw new Error(`CloudLayer: ${name} 零向量或非法`);
    }
    return [v[0] / len, v[1] / len, v[2] / len];
  }
}
