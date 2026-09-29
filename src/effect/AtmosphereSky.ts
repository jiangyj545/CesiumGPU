import type { IPass } from '../scene/passes/IPass';
import type { Context } from '../renderer/Context';
import { Camera } from '../scene/Camera';
import { CAMERA_UBO_SIZE, writeCameraUniforms } from '../renderer/CameraUniforms';
import {
  ATMO_PLANET_RADIUS_KM,
  ATMO_SUN_ILLUMINANCE,
  ATMO_TRANSMITTANCE_SIZE,
  ATMO_MULTISCAT_SIZE,
  ATMO_SKYVIEW_SIZE,
  ATMO_PARAMS_UBO_BYTES,
  SKY_PARAMS_UBO_BYTES,
  ATMO_TRANSMITTANCE_KERNEL_WGSL,
  ATMO_MULTISCAT_KERNEL_WGSL,
  ATMO_SKYVIEW_KERNEL_WGSL,
  ATMO_SKY_RENDER_WGSL,
  ATMO_SKY_REFLECTION_WGSL,
} from '../shaders/AtmosphereSkyShader';

/**
 * AtmosphereSky - Hillaire 2020「A Scalable and Production Ready Sky and
 * Atmosphere Rendering Technique」物理大气天空背景（忠实移植 tidewater
 * Atmosphere.js + Sky.js）。
 *
 * 三级 compute LUT 链（rgba16float write-only storage）：
 *   transmittance 256×64 → multiScat 32×32 → skyView 192×108。
 * 重算策略与 tidewater 一致：transmittance/multiScat 只在静态参数变化时跑一次；
 * skyView 仅当参数 UBO 字节（viewHeight / 太阳方向本地系 / 照度 / 散射系数）
 * 相对上次 dispatch 发生变化时重跑——viewHeight 按「近地 2 m、高空 2 %」量化，
 * 相机小幅运动不触发 LUT 重建。
 *
 * 归一化 ECEF 球适配（tidewater 是局部平地世界）：
 *  - 天空背景按方向渲染：dir = 全屏像素重建的归一化 ECEF 世界方向；
 *  - viewHeight（km）≈ (|camPos| - Camera.surfaceRadius(camPos)) × 6378.137 km。
 *    surfaceRadius 给出该纬度椭球地表模长（归一化），乘赤道半径即「相对地表
 *    高度 km」——椭球高度当球壳高度的近似，注明即可（误差 ≤ 0.34 %）；
 *  - LUT kernel 全在「局部平地系」工作（y = 相机天顶方向）。CPU 每帧把 ECEF
 *    世界太阳方向投影成本地系写入 AtmosphereParams.sunDir，kernel 与 tidewater
 *    逐行一致；星空 cube 映射保持 ECEF 惯性系（天球不随相机转动）。
 *
 * **挂载机制调研结论与集成方式**：
 *  - 现有 `SkyboxShader.ts` 只是孤立 shader 常量，全仓库无挂载点引用；
 *  - `Scene._renderPasses`（gbuffer → lighting → overlay → transparent）为私有
 *    数组，无公开插入口（Scene.ts 属禁改文件）；
 *  - 唯一公开挂载点是 `scene.effects`（EffectManager），执行于 Transparent
 *    composite **之后**——直接全屏画会盖住球体 imagery。
 *  - 因此本 pass 绑定 GBufferPass 的 `depth32float` 深度纹理（背景像素深度 ==
 *    1.0），管线 `depthCompare:'equal'` + `depthWriteEnabled:false`：只在
 *    「无几何体」的背景像素写天空色，**画在最后也绝不遮挡球体/实体**。
 *
 * 集成示例（推荐，getter 防 resize 后 GBuffer 重建纹理导致视图失效）：
 * ```ts
 * const sky = new AtmosphereSky(scene.context, {
 *   sunDirection: [0.4, 0.5, 0.77],
 *   starIntensity: 1,
 *   moonDirection: null,
 * });
 * sky.setDepthView(() => scene.gbufferPass?.depthView ?? null);
 * scene.effects.add(sky);
 * ```
 * 未调用 `setDepthView` 时本 pass 不绘制（安全闸，杜绝盖球）。
 * 局限：OverlayPass 的屏幕空间标注画在天空之后，落在纯背景区的标注会被天空
 * 覆盖；若需要标注最上层，需将本 pass 移入 Scene 的 renderPasses 链（属禁改
 * 范围，留给 B1 集成决策）。
 *
 * HDR 后处理链集成：主路径 target 为 LDR swap-chain，默认 `toneMapped = true`
 * （Reinhard + gamma，与 LightingPass 旧路径一致）；接入 P2 HDR 链时置 false
 * 输出线性辐射。
 *
 * @example
 * ```typescript
 * const sky = new AtmosphereSky(viewer.scene.context, {});
 * sky.setDepthView(() => viewer.scene.gbufferPass?.depthView ?? null);
 * viewer.scene.effects.add(sky);
 * sky.setSunDirection([0.3, 0.2, 0.93]); // 运行时更新，自动重算 skyView
 * ```
 */
export interface AtmosphereSkyOptions {
  /** 太阳方向（指向太阳，归一化 ECEF）。Default: `[0.4, 0.5, 0.77]` */
  sunDirection?: [number, number, number];
  /** 太阳照度缩放（大气外辐照度，场景单位）。Default: `11` */
  sunIntensity?: number;
  /** Rayleigh 散射标高缩放。Default: `1` */
  rayleighScale?: number;
  /** Mie 散射标高缩放。Default: `1` */
  mieScale?: number;
  /** Mie 各向异性系数 g。Default: `0.8` */
  mieG?: number;
  /** 臭氧吸收标高缩放。Default: `1` */
  ozoneScale?: number;
  /** 地面反照率（灰度，折入 RGB）。Default: `0.08` */
  groundAlbedo?: number;
  /** 星空强度，0 关星星。Default: `1` */
  starIntensity?: number;
  /** 太阳盘+Mie 光晕强度（tidewater 观感 1.5-2；0 关日盘）。Default: `1` */
  sunDiskIntensity?: number;
  /** 月亮方向（归一化 ECEF），null 关月亮。Default: `null` */
  moonDirection?: [number, number, number] | null;
}

/** 一个 compute pipeline + 配套 bind group（每 kernel 一份）。 */
interface ComputeUnit {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
}

/** tidewater viewHeight 量化步长（高空 2 %）。 */
const LOG_STEP = Math.log(1.02);
/** 归一化 ECEF 单位 → km 的换算（赤道半径，近似口径见类注释）。 */
const EARTH_RADIUS_KM = 6378.137;

export class AtmosphereSky implements IPass {
  readonly label = 'AtmosphereSky';

  /** 可见性开关（EffectManager 每帧检查）。 */
  show = true;

  /** LDR 主路径默认开 Reinhard+gamma；接入 HDR 后处理链时置 false。 */
  toneMapped = true;

  private _context: Context;
  private _camera: Camera | null = null;
  private _sunDir: [number, number, number];
  private _moonDir: [number, number, number] | null;
  private readonly _sunIntensity: number;
  private readonly _rayleighScale: number;
  private readonly _mieScale: number;
  private readonly _mieG: number;
  private readonly _ozoneScale: number;
  private readonly _groundAlbedo: number;
  private readonly _starIntensity: number;
  /** 太阳盘+Mie 光晕强度（SkyParams @32；tidewater 观感 1.5-2）。 */
  private readonly _sunDiskIntensity: number;

  /** 深度源：view 或 getter（gbuffer resize 会重建纹理，推荐 getter）。 */
  private _depthSource: GPUTextureView | (() => GPUTextureView | null) | null = null;
  private _warnedNoDepth = false;

  /** AtmosphereParams 载荷（64B）：@7 viewHeight / @12 sunDirLocal 每帧更新。 */
  private readonly _atmoData = new Float32Array(ATMO_PARAMS_UBO_BYTES / 4);
  /** SkyParams 载荷（64B）：time/nightK/tonemapK 每帧更新。 */
  private readonly _skyData = new Float32Array(SKY_PARAMS_UBO_BYTES / 4);
  /** 上次 skyView dispatch 时的 atmo 快照（字节级 dirty 检测）。 */
  private readonly _svLast = new Float32Array(ATMO_PARAMS_UBO_BYTES / 4);
  private _needsStatic = true;
  private _svDirty = true;

  private _atmoUBO: GPUBuffer | null = null;
  private _skyUBO: GPUBuffer | null = null;
  private _cameraUBO: GPUBuffer | null = null;
  private _transLUT: GPUTexture | null = null;
  private _multiScatLUT: GPUTexture | null = null;
  private _skyViewLUT: GPUTexture | null = null;
  private _sampler: GPUSampler | null = null;
  private _transUnit: ComputeUnit | null = null;
  private _multiScatUnit: ComputeUnit | null = null;
  private _skyViewUnit: ComputeUnit | null = null;
  private _renderPipeline: GPURenderPipeline | null = null;
  /** T5 真 HDR：Scene 经 TargetFormatAware 协议下发的场景目标格式。
   *  null = 未下发（跟随 preferredFormat 旧行为）。 */
  private _targetFormat: GPUTextureFormat | null = null;
  private _renderBindGroup: GPUBindGroup | null = null;
  /** 外部消费方（海面反射）复用的缓存视图：getter 重复调用须返回同一身份，
   *  否则消费方按「绑定身份变化」逐帧重建 bind group。 */
  private _skyViewExternalView: GPUTextureView | null = null;
  private _transExternalView: GPUTextureView | null = null;
  private _failed = false;
  private _destroyed = false;

  constructor(context: Context, opts: AtmosphereSkyOptions = {}) {
    this._context = context;
    this._sunDir = this._normalize(opts.sunDirection ?? [0.4, 0.5, 0.77]);
    this._moonDir = opts.moonDirection ? this._normalize(opts.moonDirection) : null;
    this._sunIntensity = opts.sunIntensity ?? ATMO_SUN_ILLUMINANCE;
    this._rayleighScale = opts.rayleighScale ?? 1;
    this._mieScale = opts.mieScale ?? 1;
    this._mieG = opts.mieG ?? 0.8;
    this._ozoneScale = opts.ozoneScale ?? 1;
    this._groundAlbedo = opts.groundAlbedo ?? 0.08;
    this._starIntensity = opts.starIntensity ?? 1;
    this._sunDiskIntensity = opts.sunDiskIntensity ?? 1;

    // 静态参数预填（每帧只更新 viewHeight / sunDirLocal）
    const a = this._atmoData;
    a[0] = this._rayleighScale;
    a[1] = this._mieScale;
    a[2] = this._mieG;
    a[3] = this._ozoneScale;
    a[4] = this._groundAlbedo; a[5] = this._groundAlbedo; a[6] = this._groundAlbedo;
    a[7] = ATMO_PLANET_RADIUS_KM + 0.002; // 默认海平面 2 m
    a[8] = this._sunIntensity; a[9] = this._sunIntensity; a[10] = this._sunIntensity;
    this._packSunLocal([0, 1, 0]); // 无相机时退化 up=+y

    const s = this._skyData;
    s[3] = this._starIntensity;
    s[4] = this._sunDir[0]; s[5] = this._sunDir[1]; s[6] = this._sunDir[2];
    s[7] = this._moonDir ? 1 : 0;
    s[8] = this._sunDiskIntensity;
    if (this._moonDir) { s[0] = this._moonDir[0]; s[1] = this._moonDir[1]; s[2] = this._moonDir[2]; }
    else { s[0] = 0; s[1] = 1; s[2] = 0; }
    s[11] = this.toneMapped ? 1 : 0;
  }

  // ── 公开 API ───────────────────────────────────────────────────────────────

  /** 运行时更新太阳方向（归一化 ECEF），下一帧自动重算 skyView LUT。 */
  setSunDirection(d: [number, number, number]): void {
    this._sunDir = this._normalize(d);
    const s = this._skyData;
    s[4] = this._sunDir[0]; s[5] = this._sunDir[1]; s[6] = this._sunDir[2];
  }

  /**
   * 提供深度剔除源（GBufferPass 深度，背景像素 == 1.0）。
   * 传 getter 更稳：GBufferPass resize 会重建纹理，旧 GPUTextureView 随之失效。
   * 未设置时本 pass 不绘制（安全闸：effects 链画在场景之后，无深度剔除会盖球）。
   */
  setDepthView(source: GPUTextureView | (() => GPUTextureView | null)): void {
    this._depthSource = source;
  }

  /** skyView LUT 纹理视图（外部反射采样用；destroy 后为 null）。重复调用返回同一视图身份。 */
  get skyViewTexture(): GPUTextureView | null {
    if (this._destroyed || !this._skyViewLUT) return null;
    this._skyViewExternalView ??= this._skyViewLUT.createView({ label: 'AtmosphereSky.SkyViewExternal' });
    return this._skyViewExternalView;
  }

  /** transmittance LUT 纹理视图（外部透射采样，如海面太阳染色；destroy 后 null）。 */
  get transmittanceTexture(): GPUTextureView | null {
    if (this._destroyed || !this._transLUT) return null;
    this._transExternalView ??= this._transLUT.createView({ label: 'AtmosphereSky.TransmittanceExternal' });
    return this._transExternalView;
  }

  /**
   * AtmosphereParams UBO（64B，prepare 后有效；destroy 后 null）——海面反射
   * 直接绑同一 buffer：同帧内本 pass execute 先写入最新参数，消费方读到一致数据。
   */
  get atmosphereParamsBuffer(): GPUBuffer | null {
    if (this._destroyed) return null;
    return this._atmoUBO;
  }

  /** SkyParams UBO（64B：moonDir/starIntensity/sunDir/time 等；destroy 后 null）。 */
  get skyParamsBuffer(): GPUBuffer | null {
    if (this._destroyed) return null;
    return this._skyUBO;
  }

  /**
   * 归一化 ECEF 太阳方向（构造 `sunDirection` / {@link setSunDirection} 同源，
   * 已归一化）。LightingPass AP（AerialParams.sunDir）等消费方直接引用，
   * 保证与天空渲染同一太阳。
   */
  get sunDirection(): [number, number, number] {
    return this._sunDir;
  }

  /**
   * 可选导出：供海面反射（B1）拼接的 WGSL 采样函数文本。
   * 依赖四个外部绑定符号（名字必须一致）：`atmosphereParams`（64B uniform，
   * 布局同 AtmosphereParams）、`atmSkyViewLUT`（texture_2d<f32>，绑
   * {@link skyViewTexture}）、`atmLinearClamp`（sampler）、`skyParams`
   * （starIntensity 生效，可选——若省略需删去星星分支）。入口函数
   * `atmSkyReflectionRadiance(dirWorld, up, sunDirW, moonDirW, time)`。
   */
  get skyRadianceWGSL(): string {
    return ATMO_SKY_REFLECTION_WGSL;
  }

  /** @inheritdoc IPass - 由 Scene/EffectManager 在 execute 前注入相机。 */
  updateCamera(camera: Camera): void {
    this._camera = camera;
  }

  /** @inheritdoc IPass - 幂等初始化（LUT 纹理 / 管线 / bindGroup）。 */
  /** @inheritdoc TargetFormatAware - 场景目标格式切换（T5 真 HDR）：置空渲染
   * 管线，_prepare() 的 if(!pipeline) 守卫段（BGL/pipeline/bindGroup）整段重建。 */
  setTargetFormat(format: GPUTextureFormat): void {
    if (this._failed || this._destroyed
      || format === (this._targetFormat ?? this._context.preferredFormat)) return;
    this._targetFormat = format;
    this._renderPipeline = null;
  }

  prepare(): void {
    if (this._failed) return;
    try {
      this._prepare();
    } catch (e) {
      // execute 每帧重入 prepare：初始化抛错后若不设闸会每帧重试刷屏，
      // 且 GPU 资源已部分创建，重试必然同样失败——一次报错并禁用本 pass。
      this._failed = true;
      console.error(`[${this.label}] 初始化失败，效果已禁用:`, e);
    }
  }

  /**
   * 每帧执行：按需 compute（transmittance/multiScat 一次性 + skyView 参数变化时）
   * + 一个 render pass（loadOp load，depthCompare equal 只写背景像素）。
   */
  execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void {
    if (this._destroyed) return;
    this.prepare();
    if (this._failed || !this.show) return;

    // 安全闸：无深度剔除不绘制（见 setDepthView 注释），绝不盖球体 imagery。
    const depthView = typeof this._depthSource === 'function' ? this._depthSource() : this._depthSource;
    if (!depthView || !this._renderPipeline || !this._renderBindGroup) {
      if (!this._warnedNoDepth) {
        this._warnedNoDepth = true;
        console.warn(
          `[${this.label}] 未调用 setDepthView（GBuffer 深度）或管线未就绪，` +
          '跳过绘制：effects 链位于场景合成之后，无深度剔除的天空会覆盖球体 imagery。',
        );
      }
      return;
    }

    // ── CPU 每帧准备：viewHeight 量化 + 太阳本地化 + 夜系数 ────────────────
    const dispatchDirty = this._updatePerFrame(time);

    // ── compute pass：按需 LUT 重建 ───────────────────────────────────────
    if (dispatchDirty) {
      // atmo UBO 只在触发 dispatch 的帧写入（dispatch 与采样必须看到同一份参数）
      this._context.writeBuffer(this._atmoUBO!, 0, this._atmoData);
      const cpass = encoder.beginComputePass({ label: `${this.label}.LUT` });
      if (this._needsStatic && this._transUnit && this._multiScatUnit) {
        this._needsStatic = false;
        cpass.setPipeline(this._transUnit.pipeline);
        cpass.setBindGroup(0, this._transUnit.bindGroup);
        cpass.dispatchWorkgroups(
          ATMO_TRANSMITTANCE_SIZE.width / 8,
          ATMO_TRANSMITTANCE_SIZE.height / 8,
          1,
        );
        cpass.setPipeline(this._multiScatUnit.pipeline);
        cpass.setBindGroup(0, this._multiScatUnit.bindGroup);
        cpass.dispatchWorkgroups(ATMO_MULTISCAT_SIZE.width / 8, ATMO_MULTISCAT_SIZE.height / 8, 1);
      }
      if (this._svDirty && this._skyViewUnit) {
        this._svDirty = false;
        this._svLast.set(this._atmoData);
        cpass.setPipeline(this._skyViewUnit.pipeline);
        cpass.setBindGroup(0, this._skyViewUnit.bindGroup);
        cpass.dispatchWorkgroups(
          ATMO_SKYVIEW_SIZE.width / 8,
          Math.ceil(ATMO_SKYVIEW_SIZE.height / 8),
          1,
        );
      }
      cpass.end();
    }

    // ── render pass：背景像素（depth == 1.0）写天空，深度 equal 剔除前景 ──
    const cam = this._camera;
    const camData = new Float32Array(CAMERA_UBO_SIZE / 4);
    if (cam) {
      writeCameraUniforms(camData, {
        viewMatrix: cam.viewMatrix.values,
        projectionMatrix: cam.projectionMatrix.values,
        viewProjectionMatrix: cam.viewProjectionMatrix.values,
        position: cam.position,
        viewportWidth: this._context.drawingBufferWidth,
        viewportHeight: this._context.drawingBufferHeight,
        near: cam.frustum.near,
        far: cam.frustum.far,
      });
    } else {
      camData.fill(0);
      camData[15] = 1; // view[3][3]，避免退化矩阵
      camData[31] = 1; // proj[3][3]
      camData[47] = 1; // viewProj[3][3]
    }
    this._context.writeBuffer(this._cameraUBO!, 0, camData);
    this._context.writeBuffer(this._skyUBO!, 0, this._skyData);

    const pass = encoder.beginRenderPass({
      label: this.label,
      colorAttachments: [{
        view: target,
        loadOp: 'load', // 叠加在已合成场景之上，只写背景像素
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
      depthStencilAttachment: {
        view: depthView, // GBufferPass depth32float（本帧 gbuffer 阶段已 store）
        depthLoadOp: 'load',
        depthStoreOp: 'store',
      },
    });
    pass.setPipeline(this._renderPipeline);
    pass.setBindGroup(0, this._renderBindGroup);
    pass.draw(3);
    pass.end();
  }

  /** @inheritdoc IPass - 释放全部 GPU 资源。 */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._atmoUBO?.destroy();
    this._skyUBO?.destroy();
    this._cameraUBO?.destroy();
    this._transLUT?.destroy();
    this._multiScatLUT?.destroy();
    this._skyViewLUT?.destroy();
    this._atmoUBO = null;
    this._skyUBO = null;
    this._cameraUBO = null;
    this._transLUT = null;
    this._multiScatLUT = null;
    this._skyViewLUT = null;
    this._sampler = null;
    this._transUnit = null;
    this._multiScatUnit = null;
    this._skyViewUnit = null;
    this._renderPipeline = null;
    this._renderBindGroup = null;
    this._skyViewExternalView = null;
    this._transExternalView = null;
  }

  // ── Internals ─────────────────────────────────────────────────────────────

  private _normalize(v: [number, number, number]): [number, number, number] {
    const len = Math.hypot(v[0], v[1], v[2]);
    if (len < 1e-9) return [0, 1, 0];
    return [v[0] / len, v[1] / len, v[2] / len];
  }

  /**
   * 本地基（与 WGSL fs_main / atmLocalBasis 逐行一致，修改必须同步）：
   * ref = |up.y|>0.9 ? +x : +y；east = normalize(cross(ref, up))；north = cross(up, east)。
   */
  private _localBasis(up: [number, number, number]): { east: [number, number, number]; north: [number, number, number] } {
    const ref: [number, number, number] = Math.abs(up[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
    // east = ref × up
    const ex = ref[1] * up[2] - ref[2] * up[1];
    const ey = ref[2] * up[0] - ref[0] * up[2];
    const ez = ref[0] * up[1] - ref[1] * up[0];
    const el = Math.hypot(ex, ey, ez) || 1;
    const east: [number, number, number] = [ex / el, ey / el, ez / el];
    // north = up × east
    const north: [number, number, number] = [
      up[1] * east[2] - up[2] * east[1],
      up[2] * east[0] - up[0] * east[2],
      up[0] * east[1] - up[1] * east[0],
    ];
    return { east, north };
  }

  /** 把当前世界太阳方向投影成本地系写入 atmoData[12..14]。 */
  private _packSunLocal(up: [number, number, number]): void {
    const { east, north } = this._localBasis(up);
    const s = this._sunDir;
    this._atmoData[12] = s[0] * east[0] + s[1] * east[1] + s[2] * east[2];
    this._atmoData[13] = s[0] * up[0] + s[1] * up[1] + s[2] * up[2];
    this._atmoData[14] = s[0] * north[0] + s[1] * north[1] + s[2] * north[2];
  }

  /**
   * 每帧 CPU 准备（tidewater update() 同款策略）：
   * viewHeight 量化（近地 2 m / 高空 2 %）+ 太阳本地化 + dirty 检测。
   * @returns 是否需要跑 compute dispatch（静态 LUT 或 skyView 变化）。
   */
  private _updatePerFrame(time: number): boolean {
    const cam = this._camera;
    let up: [number, number, number] = [0, 1, 0];
    if (cam) {
      const p = cam.position;
      const r = Math.hypot(p.x, p.y, p.z);
      if (r > 1e-9) up = [p.x / r, p.y / r, p.z / r];
      // 近似：椭球地表半径（归一化）当球壳半径，米制差 × 赤道半径 → km。
      const heightMeters = Math.max(0, (r - Camera.surfaceRadius(p)) * EARTH_RADIUS_KM * 1000);
      const y = Math.max(0.5, heightMeters + 0.5);
      const yq = y < 100
        ? Math.round(y / 2) * 2
        : Math.exp(Math.round(Math.log(y) / LOG_STEP) * LOG_STEP);
      this._atmoData[7] = ATMO_PLANET_RADIUS_KM + Math.max(0.001, yq / 1000);
    }
    this._packSunLocal(up);

    // dirty 检测：参数 UBO 字节相对上次 skyView dispatch 是否变化（tidewater 同款）
    let dirty = false;
    if (this._needsStatic) {
      this._svDirty = true;
      dirty = true;
    } else {
      const cur = this._atmoData;
      for (let i = 0; i < cur.length; i++) {
        if (cur[i] !== this._svLast[i]) { this._svDirty = true; dirty = true; break; }
      }
    }

    const s = this._skyData;
    s[9] = time;
    s[10] = 1 - smoothstep(-0.16, -0.04, this._atmoData[13]); // 夜系数（太阳本地仰角）
    s[11] = this.toneMapped ? 1 : 0;
    if (this._moonDir) {
      s[12] = this._moonDir[0] * up[0] + this._moonDir[1] * up[1] + this._moonDir[2] * up[2];
    } else {
      s[12] = 0;
    }
    return dirty;
  }

  /** 幂等 GPU 资源准备（LUT / UBO / 管线 / bindGroup）。 */
  private _prepare(): void {
    if (this._renderPipeline && this._transUnit && this._multiScatUnit && this._skyViewUnit) return;
    const device = this._context.device;

    // ── LUT 纹理（sample + storage 双用途）────────────────────────────────
    this._transLUT ??= this._makeLUT('AtmosphereSky.Transmittance', ATMO_TRANSMITTANCE_SIZE.width, ATMO_TRANSMITTANCE_SIZE.height);
    this._multiScatLUT ??= this._makeLUT('AtmosphereSky.MultiScat', ATMO_MULTISCAT_SIZE.width, ATMO_MULTISCAT_SIZE.height);
    this._skyViewLUT ??= this._makeLUT('AtmosphereSky.SkyView', ATMO_SKYVIEW_SIZE.width, ATMO_SKYVIEW_SIZE.height);
    this._sampler ??= device.createSampler({
      label: 'AtmosphereSky.Sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    // ── UBO ──────────────────────────────────────────────────────────────
    this._atmoUBO ??= device.createBuffer({
      label: 'AtmosphereSky.AtmoUBO',
      size: ATMO_PARAMS_UBO_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._skyUBO ??= device.createBuffer({
      label: 'AtmosphereSky.SkyUBO',
      size: SKY_PARAMS_UBO_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._cameraUBO ??= device.createBuffer({
      label: 'AtmosphereSky.CameraUBO',
      size: CAMERA_UBO_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ── compute 管线族（幂等）────────────────────────────────────────────
    const transView = this._transLUT.createView();
    const msView = this._multiScatLUT.createView();
    const svView = this._skyViewLUT.createView();

    this._transUnit ??= this._makeComputeUnit(
      'AtmosphereSky.TransmittanceKernel',
      ATMO_TRANSMITTANCE_KERNEL_WGSL,
      [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      ],
      [
        { binding: 0, resource: { buffer: this._atmoUBO! } },
        { binding: 1, resource: transView },
      ],
    );
    this._multiScatUnit ??= this._makeComputeUnit(
      'AtmosphereSky.MultiScatKernel',
      ATMO_MULTISCAT_KERNEL_WGSL,
      [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      ],
      [
        { binding: 0, resource: { buffer: this._atmoUBO! } },
        { binding: 1, resource: transView },
        { binding: 2, resource: this._sampler! },
        { binding: 3, resource: msView },
      ],
    );
    this._skyViewUnit ??= this._makeComputeUnit(
      'AtmosphereSky.SkyViewKernel',
      ATMO_SKYVIEW_KERNEL_WGSL,
      [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      ],
      [
        { binding: 0, resource: { buffer: this._atmoUBO! } },
        { binding: 1, resource: transView },
        { binding: 2, resource: this._sampler! },
        { binding: 3, resource: msView },
        { binding: 4, resource: svView },
      ],
    );

    // ── 渲染管线（全屏三角形，depth equal 只写背景像素）──────────────────
    if (!this._renderPipeline) {
      const bgl = device.createBindGroupLayout({
        label: 'AtmosphereSky.Render.BGL',
        entries: [
          { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
          { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
          { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        ],
      });
      const module = this._context.createShaderModule(ATMO_SKY_RENDER_WGSL, 'AtmosphereSky.Render.Module');
      this._renderPipeline = device.createRenderPipeline({
        label: 'AtmosphereSky.Render',
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        vertex: { module, entryPoint: 'vs_main' },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format: this._targetFormat ?? this._context.preferredFormat }] },
        primitive: { topology: 'triangle-list' },
        // GBufferPass 深度格式（depth32float）；VS 输出 NDC z=1 恰与背景深度相等
        depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'equal' },
      });
      this._renderBindGroup = device.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: this._cameraUBO! } },
          { binding: 1, resource: { buffer: this._atmoUBO! } },
          { binding: 2, resource: { buffer: this._skyUBO! } },
          { binding: 3, resource: svView },
          { binding: 4, resource: transView },
          { binding: 5, resource: this._sampler! },
        ],
      });
    }
  }

  private _makeLUT(label: string, w: number, h: number): GPUTexture {
    return this._context.device.createTexture({
      label,
      size: [w, h, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });
  }

  private _makeComputeUnit(
    label: string,
    code: string,
    entries: GPUBindGroupLayoutEntry[],
    resources: GPUBindGroupEntry[],
  ): ComputeUnit {
    const device = this._context.device;
    const bgl = device.createBindGroupLayout({ label: `${label}.BGL`, entries });
    const pipeline = device.createComputePipeline({
      label,
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module: this._context.createShaderModule(code, `${label}.Module`), entryPoint: 'main' },
    });
    return { pipeline, bindGroup: device.createBindGroup({ layout: bgl, entries: resources }) };
  }
}

/** smoothstep（CPU 侧，与 WGSL 同语义）。 */
function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}
