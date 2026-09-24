import { RenderTexture } from '../renderer/RenderTexture';
import type { Camera } from '../scene/Camera';
import { SunPosition } from '../scene/SunPosition';
import { JulianDate } from '../math/JulianDate';
import {
  defaultPostProcessConfig,
  sanitizeBloomParams,
  sanitizeGhostCount,
  validatePostProcessConfig,
} from './PostProcessConfig';
import type { BloomParams, LensFlareParams, PostProcessConfig, ToneMappingMode } from './PostProcessConfig';
import { BrightPass } from './passes/BrightPass';
import { DownsamplePass } from './passes/DownsamplePass';
import { UpsamplePass } from './passes/UpsamplePass';
import { CompositePass } from './passes/CompositePass';
import type { CompositeFrameParams } from './passes/CompositePass';
import { ToneMappingPass } from './passes/ToneMappingPass';
import { createProceduralDirtTexture } from './DirtTexture';

/** 太阳投影屏外判定的 NDC 边界（任务约定 |ndc| > 1.3 跳过 flare）。 */
const SUN_NDC_MARGIN = 1.3;

/** 中间 RT 格式（Manager 自持管线，格式自洽，不与场景管线冲突）。 */
const INTERNAL_FORMAT: GPUTextureFormat = 'rgba16float';

export type { PostProcessConfig, ToneMappingMode };

/**
 * PostProcessManager - 独立后处理管线（P2：Bloom / LensFlare / HDR ToneMapping）。
 *
 * execute 链（enabled=true 时）：
 *   sceneColor → BrightPass（半分辨率阈值提取）
 *   → DownsamplePass ×(levels-1)（13-tap 金字塔收缩）
 *   → UpsamplePass ×(levels-1)（3×3 tent 回升叠加 → bloomTex）
 *   → CompositePass（scene + bloom + lensFlare ghost/DirtMask）
 *   → ToneMappingPass（none/reinhard/aces + gamma）→ targetView（swapChain）
 *
 * 用法（对外 API，直接改公开字段即可，下一帧生效）：
 *   scene.postProcess.enabled = true;
 *   scene.postProcess.bloom.threshold = 0.9;
 *   scene.postProcess.toneMapping = 'aces';
 *
 * 太阳屏幕位置：SunPosition 方向（归一化 ECEF）经 camera.viewProjectionMatrix
 * 投 NDC；|ndc|>1.3 或 w≤0 时 flareIntensity=0 跳过 flare。
 */
export class PostProcessManager {
  /** 总开关。false = Scene 走旧 swapChain 直渲路径（本类 execute 直接 return）。 */
  enabled = false;

  /** bloom 参数（公开可变对象 —— UI 直改，execute 时夹取生效）。 */
  readonly bloom: BloomParams = { ...defaultPostProcessConfig().bloom };

  /** lensFlare 参数。 */
  readonly lensFlare: LensFlareParams = { ...defaultPostProcessConfig().lensFlare };

  /** 色调映射模式。 */
  toneMapping: ToneMappingMode = 'aces';

  private _device: GPUDevice;
  private _outputFormat: GPUTextureFormat;

  private _width: number;
  private _height: number;

  // ── 懒创建资源（首帧 execute / resize 时）────────────────────────────────
  private _passes: [BrightPass, DownsamplePass, UpsamplePass, CompositePass, ToneMappingPass] | null = null;
  private _sampler: GPUSampler | null = null;
  private _downRTs: RenderTexture[] = [];
  private _upRTs: RenderTexture[] = [];     // levels-1 个，尺寸 = down[0..levels-2]
  private _compositeRT: RenderTexture | null = null;
  private _dirtTexture: GPUTexture | null = null;
  private _blackTex: GPUTexture | null = null;  // bloom 关闭 / dirt 缺失时占位

  // ── bindGroup 缓存（视图/开关变化时重建）──────────────────────────────────
  private _bindGroupsDirty = true;
  private _lastSceneView: GPUTextureView | null = null;
  private _lastBloomEnabled = false;
  private _bgBright: GPUBindGroup | null = null;
  private _bgDown: (GPUBindGroup | null)[] = [];
  private _bgUp: (GPUBindGroup | null)[] = [];
  private _bgComposite: GPUBindGroup | null = null;
  private _bgTone: GPUBindGroup | null = null;

  private _destroyed = false;

  constructor(
    device: GPUDevice,
    outputFormat: GPUTextureFormat,
    width = 1,
    height = 1,
    config?: Partial<PostProcessConfig>,
  ) {
    this._device = device;
    this._outputFormat = outputFormat;
    this._width = Math.max(width, 1);
    this._height = Math.max(height, 1);

    if (config) {
      if (config.enabled !== undefined) this.enabled = config.enabled;
      if (config.bloom) Object.assign(this.bloom, config.bloom);
      if (config.lensFlare) Object.assign(this.lensFlare, config.lensFlare);
      if (config.toneMapping) this.toneMapping = config.toneMapping;
    }
  }

  // ── 公开属性 ─────────────────────────────────────────────────────────────

  /** 当前配置快照（UI / 测试用）。 */
  getConfig(): PostProcessConfig {
    return {
      enabled: this.enabled,
      bloom: { ...this.bloom },
      lensFlare: { ...this.lensFlare },
      toneMapping: this.toneMapping,
    };
  }

  /** 校验当前配置，返回告警列表（空 = 合法）。 */
  validate(): string[] {
    return validatePostProcessConfig(this.getConfig());
  }

  /** 金字塔级联尺寸表（调试 / 测试用）：down 与 up 各级 [w, h]。 */
  getLevelSizes(): { down: [number, number][]; up: [number, number][] } {
    const levels = Math.round(Math.min(Math.max(this.bloom.levels, 4), 6));
    const down: [number, number][] = [];
    const up: [number, number][] = [];
    for (let i = 0; i < levels; i++) {
      const s = this._levelSize(i);
      down.push(s);
      if (i < levels - 1) up.push(s);
    }
    return { down, up };
  }

  // ── 帧执行 ───────────────────────────────────────────────────────────────

  /**
   * 执行后处理链。
   * @param sceneColorView 场景离屏输出（Scene._sceneColorRT.view）
   * @param targetView     最终输出（swapChain view）
   * @param camera         可选 —— 提供才计算太阳屏幕位置（LensFlare）
   */
  execute(
    encoder: GPUCommandEncoder,
    sceneColorView: GPUTextureView,
    targetView: GPUTextureView,
    time: number,
    camera?: Camera,
  ): void {
    // 关闭时不编码任何 pass —— Scene 侧同一帧也不会把场景写进离屏 RT。
    if (!this.enabled || this._destroyed) return;

    const [bright, down, up, composite, tone] = this._ensurePasses();
    const levels = sanitizeBloomParams(this.bloom).levels;

    this._ensureTargets(levels);
    this._ensureBindGroups(sceneColorView, levels);

    // ── 1. Bloom：Bright → Down ×N → Up ×N ────────────────────────────────
    if (this.bloom.enabled) {
      bright.execute(encoder, this._bgBright!, this._downRTs[0].view!,
        this.bloom.threshold, this.bloom.knee);
      for (let i = 1; i < levels; i++) {
        down.execute(encoder, this._bgDown[i - 1]!, this._downRTs[i].view!,
          this._downRTs[i - 1].width, this._downRTs[i - 1].height);
      }
      for (let i = levels - 2; i >= 0; i--) {
        const src = (i === levels - 2) ? this._downRTs[levels - 1] : this._upRTs[i + 1];
        up.execute(encoder, this._bgUp[i]!, this._upRTs[i].view!, src.width, src.height, 0.5);
      }
    }

    // ── 2. Composite：scene + bloom + lensFlare ───────────────────────────
    const flare = this.lensFlare;
    const sun = camera ? PostProcessManager.projectSun(camera) : null;
    const p: CompositeFrameParams = {
      bloomIntensity: this.bloom.enabled ? this.bloom.intensity : 0,
      flareIntensity: (flare.enabled && sun) ? flare.intensity : 0,
      sunUv: sun ? sun.uv : [0.5, 0.5],
      ghosts: sanitizeGhostCount(flare.ghosts),
      time,
      dirtStrength: 1.0,
      aspect: this._width / Math.max(this._height, 1),
    };
    composite.execute(encoder, this._bgComposite!, this._compositeRT!.view!, p);

    // ── 3. ToneMapping → swapChain ────────────────────────────────────────
    const mode = this.toneMapping === 'reinhard' ? 1 : this.toneMapping === 'aces' ? 2 : 0;
    tone.execute(encoder, this._bgTone!, targetView, mode);

    // ── 4. 帧尾：推进延迟退役（3 帧宽限，照 OverlayPass 范式）────────────────
    for (const rt of this._downRTs) rt.onFrameEnd();
    for (const rt of this._upRTs) rt.onFrameEnd();
    this._compositeRT!.onFrameEnd();
  }

  /**
   * 画布 resize：金字塔级联尺寸重算（旧纹理延迟退役，bindGroup 下一帧重建）。
   */
  resize(width: number, height: number): void {
    if (this._destroyed) return;
    width = Math.max(width, 1);
    height = Math.max(height, 1);
    if (width === this._width && height === this._height) return;
    this._width = width;
    this._height = height;

    if (this._downRTs.length > 0) this._ensureTargets(this._downRTs.length);
    else this._bindGroupsDirty = true;
  }

  /** 销毁全部 pass 与 RT（RT 内部含 3 帧宽限队列的立即清空）。 */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    if (this._passes) {
      for (const p of this._passes) p.destroy();
      this._passes = null;
    }
    for (const rt of this._downRTs) rt.destroy();
    for (const rt of this._upRTs) rt.destroy();
    this._downRTs = [];
    this._upRTs = [];
    this._compositeRT?.destroy();
    this._compositeRT = null;
    this._dirtTexture = null; // 自建 dirt 由 GC/设备释放；外部注入的归调用方
    this._blackTex = null;
    this._sampler = null;
    this._bindGroupsDirty = true;
  }

  // ── 私有：资源构建 ────────────────────────────────────────────────────────

  private _ensurePasses() {
    if (!this._passes) {
      const d = this._device;
      this._passes = [
        new BrightPass(d, INTERNAL_FORMAT),
        new DownsamplePass(d, INTERNAL_FORMAT),
        new UpsamplePass(d, INTERNAL_FORMAT),
        new CompositePass(d, INTERNAL_FORMAT),
        new ToneMappingPass(d, this._outputFormat),
      ];
    }
    if (!this._sampler) {
      this._sampler = this._device.createSampler({
        label: 'PostProcess.Sampler',
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
      });
    }
    return this._passes;
  }

  /** 第 i 级尺寸：half 起，每级折半（≥1）。 */
  private _levelSize(i: number): [number, number] {
    const w = Math.max(Math.floor(this._width / (2 << i)), 1);
    const h = Math.max(Math.floor(this._height / (2 << i)), 1);
    return [w, h];
  }

  private _ensureTargets(levels: number): void {
    while (this._downRTs.length < levels) {
      this._downRTs.push(new RenderTexture(
        this._device, INTERNAL_FORMAT, `PostProcess.Down[${this._downRTs.length}]`));
    }
    while (this._upRTs.length < levels - 1) {
      this._upRTs.push(new RenderTexture(
        this._device, INTERNAL_FORMAT, `PostProcess.Up[${this._upRTs.length}]`));
    }
    if (!this._compositeRT) {
      this._compositeRT = new RenderTexture(this._device, INTERNAL_FORMAT, 'PostProcess.Composite');
    }
    let targetsChanged = false;
    for (let i = 0; i < levels; i++) {
      const [w, h] = this._levelSize(i);
      targetsChanged = this._downRTs[i].ensure(w, h) || targetsChanged;
    }
    for (let i = 0; i < levels - 1; i++) {
      const [w, h] = this._levelSize(i);
      targetsChanged = this._upRTs[i].ensure(w, h) || targetsChanged;
    }
    targetsChanged = this._compositeRT.ensure(this._width, this._height) || targetsChanged;
    if (targetsChanged) this._bindGroupsDirty = true;
  }

  private _ensureBindGroups(sceneColorView: GPUTextureView, levels: number): void {
    const [bright, down, up, composite, tone] = this._ensurePasses();
    const sceneChanged = this._lastSceneView !== sceneColorView;
    const bloomChanged = this._lastBloomEnabled !== this.bloom.enabled;
    if (!this._bindGroupsDirty && !sceneChanged && !bloomChanged && this._bgComposite) return;

    const device = this._device;
    const samp = this._sampler!;

    // Bright：sceneColor → down[0]
    this._bgBright = device.createBindGroup({
      label: 'PostProcess.Bright.BG',
      layout: bright.bindGroupLayout,
      entries: [
        { binding: 0, resource: sceneColorView },
        { binding: 1, resource: samp },
        { binding: 2, resource: { buffer: bright.uniformBuffer } },
      ],
    });

    // Down[i]：down[i-1] → down[i]
    this._bgDown = new Array(levels - 1).fill(null);
    for (let i = 1; i < levels; i++) {
      this._bgDown[i - 1] = device.createBindGroup({
        label: `PostProcess.Down[${i}].BG`,
        layout: down.bindGroupLayout,
        entries: [
          { binding: 0, resource: this._downRTs[i - 1].view! },
          { binding: 1, resource: samp },
          { binding: 2, resource: { buffer: down.uniformBuffer } },
        ],
      });
    }

    // Up[i]：src（第一步用 down[levels-1]，否则 up[i+1]）+ down[i] → up[i]
    this._bgUp = new Array(levels - 1).fill(null);
    for (let i = levels - 2; i >= 0; i--) {
      const src = (i === levels - 2) ? this._downRTs[levels - 1] : this._upRTs[i + 1];
      this._bgUp[i] = device.createBindGroup({
        label: `PostProcess.Up[${i}].BG`,
        layout: up.bindGroupLayout,
        entries: [
          { binding: 0, resource: src.view! },
          { binding: 1, resource: this._downRTs[i].view! },
          { binding: 2, resource: samp },
          { binding: 3, resource: { buffer: up.uniformBuffer } },
        ],
      });
    }

    // Composite：scene + bloom（关闭时占位黑纹）+ dirt + sampler + params
    const dirt = this._ensureDirtTexture() ?? this._ensureBlackTex();
    const bloomTex = this.bloom.enabled ? this._upRTs[0].view! : this._ensureBlackTex();
    this._bgComposite = device.createBindGroup({
      label: 'PostProcess.Composite.BG',
      layout: composite.bindGroupLayout,
      entries: [
        { binding: 0, resource: sceneColorView },
        { binding: 1, resource: bloomTex },
        { binding: 2, resource: dirt },
        { binding: 3, resource: samp },
        { binding: 4, resource: { buffer: composite.uniformBuffer } },
      ],
    });

    // ToneMapping：composite → targetView
    this._bgTone = device.createBindGroup({
      label: 'PostProcess.ToneMap.BG',
      layout: tone.bindGroupLayout,
      entries: [
        { binding: 0, resource: this._compositeRT!.view! },
        { binding: 1, resource: samp },
        { binding: 2, resource: { buffer: tone.uniformBuffer } },
      ],
    });

    this._lastSceneView = sceneColorView;
    this._lastBloomEnabled = this.bloom.enabled;
    this._bindGroupsDirty = false;
  }

  /** 默认 dirt（首次使用时生成；node/无 DOM 返回 null 由调用方降级占位）。 */
  private _ensureDirtTexture(): GPUTexture | null {
    if (this.lensFlare.dirtTexture) return this.lensFlare.dirtTexture;
    if (!this._dirtTexture) {
      this._dirtTexture = createProceduralDirtTexture(this._device);
    }
    return this._dirtTexture;
  }

  /** 1×1 黑纹（bloom 关闭 / dirt 缺失时的合法占位绑定）。 */
  private _ensureBlackTex(): GPUTexture {
    if (!this._blackTex) {
      this._blackTex = this._device.createTexture({
        label: 'PostProcess.Black1x1',
        size: { width: 1, height: 1 },
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
    return this._blackTex;
  }

  // ── 太阳屏幕投影 ──────────────────────────────────────────────────────────

  /**
   * 太阳方向（归一化 ECEF 单位球位置）经相机 viewProj 投影到 uv。
   * @returns null = 屏外（|ndc|>1.3）或在相机背后（w≤0）→ 跳过 flare
   */
  static projectSun(camera: Camera): { uv: [number, number] } | null {
    // computeSunDirection 返回"太阳→地球"；取反得到地球→太阳（单位球上的太阳位置）。
    const [dx, dy, dz] = SunPosition.computeSunDirection(JulianDate.now());
    const sx = -dx, sy = -dy, sz = -dz;

    // Matrix4.values 列主序（与作为 WGSL mat4x4 上传的内存布局一致）。
    const m = camera.viewProjectionMatrix.values;
    const cx = m[0] * sx + m[4] * sy + m[8]  * sz + m[12];
    const cy = m[1] * sx + m[5] * sy + m[9]  * sz + m[13];
    const cw = m[3] * sx + m[7] * sy + m[11] * sz + m[15];

    if (cw <= 0) return null; // 相机背后
    const nx = cx / cw;
    const ny = cy / cw;
    if (Math.abs(nx) > SUN_NDC_MARGIN || Math.abs(ny) > SUN_NDC_MARGIN) return null;

    // NDC(y 上) → uv（纹理 y 下，与全屏三角 uv 约定一致）
    return { uv: [nx * 0.5 + 0.5, 0.5 - ny * 0.5] };
  }
}

export default PostProcessManager;
