import type { IPass } from '../scene/passes/IPass';
import type { Context } from '../renderer/Context';
import { Camera } from '../scene/Camera';
import { CAMERA_UBO_SIZE, writeCameraUniforms } from '../renderer/CameraUniforms';
import { SunPosition } from '../scene/SunPosition';
import { JulianDate } from '../math/JulianDate';
import { UNDERWATER_FOG_SHADER } from '../shaders/UnderwaterFogShader';

/**
 * UnderwaterFog - 水下观感全屏叠加 pass（tidewater "split underwater/above-water
 * view" 的水下侧简化版；水面双面渲染归 OceanFFTWater 侧）。
 *
 *   1. 蓝绿深度雾 —— 距离衰减（Beer）：G-Buffer worldPos 重建像素水下
 *      路径长；背景像素（水面/天空，worldPos 清零）用射线-海面球出射距离
 *      （抬头 3m 即透 / 水平 6km 全雾）。未绑定 scenePos 时回退俯仰近似；
 *   2. 浮动颗粒/光斑 —— 两层屏幕空间 hash 点阵缓漂 + 两层正弦波纹相乘的焦散
 *      缓动（呼应 OceanCaustics 观感，独立实现、零纹理采样）；
 *   3. 边缘暗角 + 轻微色偏青 —— 水下通透感；
 *   4. 体积焦散光柱（Track G 联动，tidewater src/post/Underwater.js
 *      _shaftCode 同法）—— sunShaftIntensity > 0 且已 bindCaustics 时，
 *      FS 沿视线 20 步 march 采样 OceanCaustics 真焦散纹理，光柱随
 *      焦散网/波浪摆动（默认 0 关闭零 march 零回归），与屏幕空间
 *      god rays（PostProcessManager.godRays）叠加。
 *
 * 独立性：不依赖 OceanFFT/OceanCaustics 任何符号；bind group 含相机/参数
 * UBO + G-Buffer worldPos 纹理（距离衰减雾，可选绑定）。
 *
 * 混合模式：premultiplied-alpha（one, one-minus-src-alpha）。FS 输出的 rgb
 * 已乘覆盖权重——薄雾区的颗粒/焦散高光自然呈加性提亮，浓雾区近似 alpha
 * 混合染色，且免去逐像素 un-premultiply 除法。intensity = 0 时输出全 0，
 * dst 逐像素原样保留。
 *
 * 零回归保证：`show = false` 或 `_underwater = false` 时 execute 直接 return，
 * 不编码任何 pass。
 *
 * 用法：
 * ```ts
 * const uw = scene.effects.add(new UnderwaterFog(viewer.scene.context, {
 *   intensity: 1.0,
 *   waterColor: [0.05, 0.22, 0.30],
 *   particleDensity: 0.5,
 * }));
 * uw.setUnderwater(true);   // B1 在相机入水判定后调用
 * uw.setIntensity(1.4);
 * uw.show = false;          // 完全无贡献
 * uw.destroy();
 * ```
 */
export interface UnderwaterFogOptions {
  /** 效果强度（乘到雾透明度与高光上）。Default: `1` */
  intensity?: number;
  /** 水体染色（线性 RGB）。Default: `深青蓝 [0.05, 0.22, 0.30]` */
  waterColor?: [number, number, number];
  /** 浮游颗粒密度倍率。Default: `0.5` */
  particleDensity?: number;
  /** 水下太阳光柱强度（Track G 联动；0 = 关闭）。Default: `0`（零回归） */
  sunShaftIntensity?: number;
  /**
   * 参考海床深度（米，海面下）。水下俯视的背景像素（GBuffer 无几何——
   * 海面在相机上方、下半球射线穿不出海面球）按「相机深 + 此深度」算
   * 水下路径长，海床/焦散从青纱下透出；0 = 关闭（俯视背景全雾，旧行为）。
   * Default: `6`
   */
  seabedDepthM?: number;
}

/** 参数 UBO：9 x vec4 = 144 字节（布局见 UnderwaterFogShader 的 UWParams）。 */
const PARAMS_BYTES = 144;

/** 体积焦散光柱的焦散源描述（OceanCaustics 实例的绑定视图 + ENU 基）。 */
export interface UnderwaterCausticsSource {
  /** 焦散纹理 view getter（惰性——纹理在 prepare 后才存在，逐帧取最新身份）。 */
  view: () => GPUTextureView | null;
  /** 焦散覆盖域边长（米；<= 0 视为未绑定，光柱关闭）。 */
  coverMeters: number;
  /** 焦散域中心（归一化 ECEF [x,y,z]，march 局部 ENU 系原点）。 */
  centerEcef: [number, number, number];
  /** ENU 基（归一化 ECEF 向量）。 */
  eastEcef: [number, number, number];
  upEcef: [number, number, number];
  northEcef: [number, number, number];
  /** 浅焦面深度（米，纹理 R 通道）。 */
  shallowDepthM: number;
  /** 深焦面深度（米，纹理 G 通道）。 */
  deepDepthM: number;
}

const DEFAULT_WATER_COLOR: [number, number, number] = [0.05, 0.22, 0.30];

export class UnderwaterFog implements IPass {
  readonly label = 'UnderwaterFog';

  /** 可见性开关（Effect 类型要求）。false 时 execute 零贡献。 */
  show = true;

  private _context: Context;
  private _camera: Camera | null = null;

  private _underwater = false;
  private _intensity: number;
  private _waterColor: [number, number, number];
  private _particleDensity: number;
  private _sunShaftIntensity: number;
  /** 参考海床深度（米；水下俯视背景像素的水路长基准，0 = 关闭全雾）。 */
  private _seabedDepthM: number;

  private _pipeline: GPURenderPipeline | null = null;
  private _bindGroupLayout: GPUBindGroupLayout | null = null;
  private _cameraUBO: GPUBuffer | null = null;
  private _paramsUBO: GPUBuffer | null = null;
  private _bindGroup: GPUBindGroup | null = null;

  /**
   * 场景位置纹理源（G-Buffer worldPos，rgba32float 归一化 ECEF）。距离
   * 衰减雾（近清澈远蓝）的水下路径长来源；null = 回退俯仰近似（零回归）。
   * 用 getter 惰性求值——gbuffer resize 会重建纹理，逐帧取最新身份。
   */
  private _scenePosGetter: (() => GPUTextureView | null) | null = null;
  /** 1×1 零纹理 + 点采样器（scenePos 未绑定时的 BGL 占位，采样值 0 → dist 巨大 → 全雾）。 */
  private _fallbackTex: GPUTexture | null = null;
  private _fallbackView: GPUTextureView | null = null;
  private _posSampler: GPUSampler | null = null;
  /** 当前 BG 里绑定的场景位置 view（身份缓存，防逐帧重建 bind group）。 */
  private _boundScenePos: GPUTextureView | null = null;
  private _failed = false;
  /** 体积焦散光柱的焦散源（null = 未绑定，光柱关闭零回归）。 */
  private _causticsSource: UnderwaterCausticsSource | null = null;
  /** 1×1 rgba16float 零纹理 + 过滤采样器（焦散未绑定的 BGL 占位；march 被 cover>0 uniform 门控不会读）。 */
  private _causticsFallbackTex: GPUTexture | null = null;
  private _causticsFallbackView: GPUTextureView | null = null;
  private _causticsSampler: GPUSampler | null = null;
  /** 当前 BG 里绑定的焦散 view（身份缓存，防逐帧重建 bind group）。 */
  private _boundCaustics: GPUTextureView | null = null;
  private _destroyed = false;
  /** T5 真 HDR：场景目标格式（null = 未下发跟随 preferredFormat 旧行为）。 */
  private _targetFormat: GPUTextureFormat | null = null;

  /** 相机 UBO 暂存（256 B，布局单一来源 src/renderer/CameraUniforms.ts）。 */
  private _camData = new Float32Array(CAMERA_UBO_SIZE / 4);
  /** 参数 UBO 暂存（48 B = 12 floats）。 */
  private _params = new Float32Array(PARAMS_BYTES / 4);

  constructor(context: Context, opts?: UnderwaterFogOptions) {
    this._context = context;
    this._intensity = opts?.intensity ?? 1;
    this._waterColor = opts?.waterColor ?? DEFAULT_WATER_COLOR;
    this._particleDensity = opts?.particleDensity ?? 0.5;
    this._sunShaftIntensity = opts?.sunShaftIntensity ?? 0;
    this._seabedDepthM = opts?.seabedDepthM ?? 6;
  }

  /** 手动开关水下模式（B1 在相机入水判定后调用）。 */
  setUnderwater(on: boolean): void {
    this._underwater = on;
  }

  /**
   * 注入场景位置纹理源（G-Buffer worldPos view 或返回它的 getter）。
   * 绑定后水下雾按水下路径长 Beer 衰减（近清澈/远蓝）；null 回退俯仰近似。
   */
  setScenePosView(getter: (() => GPUTextureView | null) | null): void {
    this._scenePosGetter = getter;
  }

  /** 当前是否处于水下渲染状态。 */
  get underwater(): boolean {
    return this._underwater;
  }

  /** 强度（0 = 完全透明，>1 可加浓）。 */
  setIntensity(v: number): void {
    this._intensity = v;
  }

  /** 水下太阳光柱强度（Track G 联动；0 = 关闭，默认零回归）。 */
  setSunShaftIntensity(v: number): void {
    this._sunShaftIntensity = v;
  }

  /**
   * 绑定/解绑体积焦散光柱的焦散源（OceanCaustics 实例）。null 或
   * coverMeters<=0 → 光柱关闭（shader 端 cCenter.w=0 uniform 门控，零 march）。
   */
  bindCaustics(source: UnderwaterCausticsSource | null): void {
    this._causticsSource = source && source.coverMeters > 0 ? source : null;
  }

  /** 当前水下太阳光柱强度。 */
  get sunShaftIntensity(): number {
    return this._sunShaftIntensity;
  }

  /** @inheritdoc IPass - 由 Scene 在 execute 前注入相机（IPass 可选协议）。 */
  updateCamera(camera: Camera): void {
    this._camera = camera;
  }

  /**
   * T5 真 HDR：目标格式下发（EffectManager 可选协议，Scene.render 逐帧调用
   * 当前渲染目标格式）。变更时销毁旧 UBO 并置空管线/BGL，由 prepare() 惰性
   * 守卫整体重建；同格式幂等早退。未接线的调用路径保持 preferredFormat
   * 旧行为。
   */
  setTargetFormat(format: GPUTextureFormat): void {
    if (this._failed || this._destroyed
      || format === (this._targetFormat ?? this._context.preferredFormat)) return;
    this._targetFormat = format;
    this._cameraUBO?.destroy();
    this._paramsUBO?.destroy();
    this._fallbackTex?.destroy();
    this._causticsFallbackTex?.destroy();
    this._pipeline = null;
    this._bindGroup = null;
    this._bindGroupLayout = null;
    this._cameraUBO = null;
    this._paramsUBO = null;
    this._fallbackTex = null;   // prepare() 会重建全套（含 fallback）
    this._fallbackView = null;
    this._causticsFallbackTex = null;
    this._causticsFallbackView = null;
    this._boundScenePos = null;
    this._boundCaustics = null;
  }

  /** 幂等初始化（管线 / BGL / UBO / bindGroup）。失败置 _failed 闸禁用整效果。 */
  prepare(): void {
    if (this._failed || this._destroyed || this._pipeline) return;
    try {
      this._prepare();
    } catch (e) {
      // execute 每帧重入 prepare：初始化抛错后若不设闸会每帧重试刷屏
      //（同 OceanFFTWater 的 _failed 闸）。
      this._failed = true;
      console.error(`[${this.label}] 初始化失败，效果已禁用:`, e);
    }
  }

  private _prepare(): void {
    const device = this._context.device;
    const shader = this._context.createShaderModule(UNDERWATER_FOG_SHADER, `${this.label}.Module`);

    this._bindGroupLayout = device.createBindGroupLayout({
      label: `${this.label}.BGL`,
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', minBindingSize: CAMERA_UBO_SIZE } },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', minBindingSize: PARAMS_BYTES } },
        // 距离衰减雾：G-Buffer worldPos（rgba32float 不可过滤 → 点采样 +
        // textureSampleLevel，深度值无需滤波）
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } },
        // 体积焦散光柱：真焦散纹理（rgba16float 可过滤）+ 三线性采样器
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });

    // 焦散 1×1 零纹理占位（march 被 cCenter.w>0 uniform 门控，未绑定时不会被读）
    this._causticsFallbackTex = device.createTexture({
      label: `${this.label}.CausticsFallback`,
      size: [1, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this._causticsFallbackView = this._causticsFallbackTex.createView();
    this._causticsSampler = device.createSampler({
      label: `${this.label}.CausticsSampler`,
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    });

    // 1×1 零纹理占位（采样值 0 → FS dist 巨大 → 全雾，语义安全）
    this._fallbackTex = device.createTexture({
      label: `${this.label}.ScenePosFallback`,
      size: [1, 1],
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this._fallbackView = this._fallbackTex.createView();
    this._posSampler = device.createSampler({
      label: `${this.label}.PosSampler`,
      magFilter: 'nearest',
      minFilter: 'nearest',
    });

    this._pipeline = device.createRenderPipeline({
      label: `${this.label}.Pipeline`,
      layout: device.createPipelineLayout({ bindGroupLayouts: [this._bindGroupLayout] }),
      vertex:   { module: shader, entryPoint: 'vs_main' },
      fragment: {
        module: shader,
        entryPoint: 'fs_main',
        targets: [{
          format: this._targetFormat ?? this._context.preferredFormat,
          // premultiplied-alpha 叠加：FS rgb 已乘覆盖权重，薄雾区高光呈加性。
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      // 无深度附件：纯屏幕空间叠加，不参与深度测试。
    });

    this._cameraUBO = device.createBuffer({
      label: `${this.label}.CameraUBO`,
      size:  CAMERA_UBO_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._paramsUBO = device.createBuffer({
      label: `${this.label}.ParamsUBO`,
      size:  PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this._bindGroup = device.createBindGroup({
      label: `${this.label}.BG`,
      layout: this._bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this._cameraUBO } },
        { binding: 1, resource: { buffer: this._paramsUBO } },
        { binding: 2, resource: this._fallbackView! },
        { binding: 3, resource: this._posSampler! },
        { binding: 4, resource: this._causticsFallbackView! },
        { binding: 5, resource: this._causticsSampler! },
      ],
    });
  }

  /**
   * 执行水下观感叠加。
   * @param encoder - 当前帧共享的 command encoder
   * @param target  - 已合成场景的颜色输出（swap-chain view），loadOp: 'load'
   * @param time    - 统一效果时间（秒），驱动颗粒漂移与焦散缓动
   */
  execute(encoder: GPUCommandEncoder, target: GPUTextureView, time: number): void {
    // show = false 或非水下 → 直接 return，不编码任何 pass（零回归）。
    if (this._destroyed || !this.show || !this._underwater) return;
    this.prepare();
    if (this._failed || !this._pipeline || !this._bindGroup || !this._cameraUBO || !this._paramsUBO) return;

    // ── 相机 UBO（布局单一来源：src/renderer/CameraUniforms.ts）──────────
    const camData = this._camData;
    const cam = this._camera;
    if (cam) {
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
    } else {
      // 尚未注入相机：单位化矩阵对角，避免退化矩阵（同 AtmosphereSky 兜底）。
      camData.fill(0);
      camData[15] = 1; // view[3][3]
      camData[31] = 1; // proj[3][3]
      camData[47] = 1; // viewProj[3][3]
    }
    this._context.writeBuffer(this._cameraUBO, 0, camData);

  // ── 纹理源（场景位置 / 焦散）：getter 逐帧取最新身份，任一变更重建 BG ──
  let hasScenePos = 0;
  let hasCaustics = 0;
  if (this._scenePosGetter) {
    const view = this._scenePosGetter();
    if (view) {
      hasScenePos = 1;
      if (view !== this._boundScenePos) this._rebuildBindGroup(view, this._boundCaustics);
    }
  }
  // 体积焦散光柱：焦散源 getter 惰性求值（OceanCaustics 纹理在 prepare 后才有）
  if (this._causticsSource) {
    const cView = this._causticsSource.view();
    if (cView) {
      hasCaustics = 1;
      if (cView !== this._boundCaustics) this._rebuildBindGroup(this._boundScenePos, cView);
    }
  }

  // ── 参数 UBO：[time, intensity, density, hasScenePos] [w, h, seaR, seabed]
  //              [rgb, _] [sunDir.xyz, sunShaftIntensity]
  //              [cCenter.xyz, cover] [cEast.xyz, sunLocal.x]
  //              [cUp.xyz, sunLocal.y] [cNorth.xyz, sunLocal.z]
  //              [D0, D1, _, _]（体积焦散光柱，tidewater _shaftCode 布局）──
  // seaR = 相机星下点海面球半径（归一化，Camera.surfaceRadius 压扁面公式，
  // 与 GBuffer 几何同系）——FS 背景像素用它做射线-海面求交（水下路径长）。
  const p = this._params;
  p[0] = time;
  p[1] = this._intensity;
  p[2] = this._particleDensity;
  p[3] = hasScenePos;
  p[4] = this._context.drawingBufferWidth;
  p[5] = this._context.drawingBufferHeight;
  p[6] = cam ? Camera.surfaceRadius(cam.position) : 1.0;
  p[7] = this._seabedDepthM; // viewport.w：水下俯视参考海床深（米，0=关）
  p[8]  = this._waterColor[0];
  p[9]  = this._waterColor[1];
  p[10] = this._waterColor[2];
  // 太阳方向（地球→太阳，归一化 ECEF；与 PostProcessManager.projectSun 同源约定）
  const [dx, dy, dz] = SunPosition.computeSunDirection(JulianDate.now());
  const sx = -dx, sy = -dy, sz = -dz;
  p[12] = sx;
  p[13] = sy;
  p[14] = sz;
  p[15] = this._sunShaftIntensity;
  // cCenter / cEast / cUp / cNorth / cFocal（未绑定焦散源时全 0 → FS 门控关）
  if (this._causticsSource && hasCaustics) {
    const c = this._causticsSource;
    p[16] = c.centerEcef[0]; p[17] = c.centerEcef[1]; p[18] = c.centerEcef[2];
    p[19] = c.coverMeters;
    p[20] = c.eastEcef[0];  p[21] = c.eastEcef[1];  p[22] = c.eastEcef[2];
    p[23] = sx * c.eastEcef[0] + sy * c.eastEcef[1] + sz * c.eastEcef[2];
    p[24] = c.upEcef[0];    p[25] = c.upEcef[1];    p[26] = c.upEcef[2];
    p[27] = sx * c.upEcef[0] + sy * c.upEcef[1] + sz * c.upEcef[2];
    p[28] = c.northEcef[0]; p[29] = c.northEcef[1]; p[30] = c.northEcef[2];
    p[31] = sx * c.northEcef[0] + sy * c.northEcef[1] + sz * c.northEcef[2];
    p[32] = c.shallowDepthM;
    p[33] = c.deepDepthM;
  } else {
    p[16] = 0; p[17] = 0; p[18] = 0; p[19] = 0;   // cCenter.w = 0 → 光柱关
    p[20] = 0; p[21] = 0; p[22] = 0; p[23] = 0;
    p[24] = 0; p[25] = 0; p[26] = 0; p[27] = 0;
    p[28] = 0; p[29] = 0; p[30] = 0; p[31] = 0;
    p[32] = 0; p[33] = 0;
  }
  p[34] = 0; p[35] = 0; // cFocal.zw 未用（补齐 144 B）
  this._context.writeBuffer(this._paramsUBO, 0, p);

    const pass = encoder.beginRenderPass({
      label: this.label,
      colorAttachments: [{
        view: target, // 叠加在已合成场景之上
        loadOp: 'load',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, this._bindGroup);
    pass.draw(3); // 全屏三角
    pass.end();
  }

  /** 场景位置/焦散 view 任一变更时重建 bind group（身份缓存防逐帧重建）。 */
  private _rebuildBindGroup(scenePosView: GPUTextureView | null, causticsView: GPUTextureView | null): void {
    if (!this._bindGroupLayout) return;
    this._bindGroup = this._context.device.createBindGroup({
      label: `${this.label}.BG`,
      layout: this._bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this._cameraUBO! } },
        { binding: 1, resource: { buffer: this._paramsUBO! } },
        { binding: 2, resource: scenePosView ?? this._fallbackView! },
        { binding: 3, resource: this._posSampler! },
        { binding: 4, resource: causticsView ?? this._causticsFallbackView! },
        { binding: 5, resource: this._causticsSampler! },
      ],
    });
    this._boundScenePos = scenePosView;
    this._boundCaustics = causticsView;
  }

  /** 释放 GPU 资源；调用后本 pass 不可再用。 */
  destroy(): void {
    this._destroyed = true;
    this._cameraUBO?.destroy();
    this._paramsUBO?.destroy();
    this._fallbackTex?.destroy();
    this._causticsFallbackTex?.destroy();
    this._cameraUBO = null;
    this._paramsUBO = null;
    this._fallbackTex = null;
    this._fallbackView = null;
    this._causticsFallbackTex = null;
    this._causticsFallbackView = null;
    this._pipeline = null;
    this._bindGroup = null;
    this._bindGroupLayout = null;
    this._boundScenePos = null;
    this._boundCaustics = null;
}
}
