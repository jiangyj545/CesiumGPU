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
 * 相机处于海面之下时（B1 判定后调 {@link setUnderwater}），在已合成场景之上
 * 叠加一张全屏三角（draw(3)，照 BrightPass/LightingPass 模式）：
 *   1. 蓝绿深度雾 —— 无场景深度纹理，深度感用视线俯仰渐变（低头看海床更浓）
 *      + 屏幕径向暗角近似；
 *   2. 浮动颗粒/光斑 —— 两层屏幕空间 hash 点阵缓漂 + 两层正弦波纹相乘的焦散
 *      缓动（呼应 OceanCaustics 观感，独立实现、零纹理采样）；
 *   3. 边缘暗角 + 轻微色偏青 —— 水下通透感；
 *   4. 水下太阳光柱 —— sunShaftIntensity > 0 时启用（Track G 联动，默认 0
 *      关闭零回归），与屏幕空间 god rays（PostProcessManager.godRays）叠加。
 *
 * 完全独立：不依赖 OceanFFT/OceanCaustics 任何符号，bind group 仅含两个
 * uniform buffer（相机 UBO 256 B + 参数 UBO 48 B），无任何纹理。
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
  /** 水体染色（线性 RGB）。Default: 深青蓝 `[0.05, 0.22, 0.30]` */
  waterColor?: [number, number, number];
  /** 浮游颗粒密度倍率。Default: `0.5` */
  particleDensity?: number;
  /** 水下太阳光柱强度（Track G 联动；0 = 关闭）。Default: `0`（零回归） */
  sunShaftIntensity?: number;
}

/** 参数 UBO：4 x vec4 = 64 字节（布局见 UnderwaterFogShader 的 UWParams）。 */
const PARAMS_BYTES = 64;

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

  private _pipeline: GPURenderPipeline | null = null;
  private _bindGroupLayout: GPUBindGroupLayout | null = null;
  private _cameraUBO: GPUBuffer | null = null;
  private _paramsUBO: GPUBuffer | null = null;
  private _bindGroup: GPUBindGroup | null = null;
  private _failed = false;
  private _destroyed = false;

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
  }

  /** 手动开关水下模式（B1 在相机入水判定后调用）。 */
  setUnderwater(on: boolean): void {
    this._underwater = on;
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

  /** 当前水下太阳光柱强度。 */
  get sunShaftIntensity(): number {
    return this._sunShaftIntensity;
  }

  /** @inheritdoc IPass - 由 Scene 在 execute 前注入相机（IPass 可选协议）。 */
  updateCamera(camera: Camera): void {
    this._camera = camera;
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
      ],
    });

    this._pipeline = device.createRenderPipeline({
      label: `${this.label}.Pipeline`,
      layout: device.createPipelineLayout({ bindGroupLayouts: [this._bindGroupLayout] }),
      vertex:   { module: shader, entryPoint: 'vs_main' },
      fragment: {
        module: shader,
        entryPoint: 'fs_main',
        targets: [{
          format: this._context.preferredFormat,
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

    // ── 参数 UBO：[time, intensity, density, _] [w, h, _, _] [rgb, _]
    //              [sunDir.xyz, sunShaftIntensity]（Track G 水下光柱）──
    const p = this._params;
    p[0] = time;
    p[1] = this._intensity;
    p[2] = this._particleDensity;
    p[4] = this._context.drawingBufferWidth;
    p[5] = this._context.drawingBufferHeight;
    p[8]  = this._waterColor[0];
    p[9]  = this._waterColor[1];
    p[10] = this._waterColor[2];
    // 太阳方向（地球→太阳，归一化 ECEF；与 PostProcessManager.projectSun 同源约定）
    const [dx, dy, dz] = SunPosition.computeSunDirection(JulianDate.now());
    p[12] = -dx;
    p[13] = -dy;
    p[14] = -dz;
    p[15] = this._sunShaftIntensity;
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

  /** 释放 GPU 资源；调用后本 pass 不可再用。 */
  destroy(): void {
    this._destroyed = true;
    this._cameraUBO?.destroy();
    this._paramsUBO?.destroy();
    this._cameraUBO = null;
    this._paramsUBO = null;
    this._pipeline = null;
    this._bindGroup = null;
    this._bindGroupLayout = null;
  }
}
