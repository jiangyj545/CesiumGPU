/**
 * RenderTexture - 渲染目标纹理工厂（P2 后处理管线地基）。
 *
 * 职责（照 GBufferPass._create / OverlayPass 退役范式收拢）：
 *  1. `ensure(w, h)`   创建 / 重建 RT，同尺寸早退（不销毁不重建，先例 GBufferPass.resize:114）。
 *  2. `retire()`       延迟退役：当前纹理进入退役队列而非立即 destroy——WebGPU 中
 *                      上一帧 in-flight submit 可能仍引用它，立即销毁会触发
 *                      uncaptured error 使整帧变黑（先例 OverlayPass RETIRE_GRACE_FRAMES=3）。
 *  3. `onFrameEnd()`   帧计数驱动：退役满 3 帧后才真正 destroy。
 *  4. `destroy()`      立即销毁当前 + 全部退役纹理（仅限终局清理）。
 *
 * 纯 device 级工具，不持有 Context —— node 测试可用记录型桩 device 直接驱动。
 */
export class RenderTexture {
  private _device: GPUDevice;
  private _format: GPUTextureFormat;
  private _label: string;
  private _extraUsage: GPUFlagsConstant;

  private _texture: GPUTexture | null = null;
  private _view: GPUTextureView | null = null;
  private _width  = 0;
  private _height = 0;

  /** 退役中的纹理及其可安全销毁的帧号（见 RETIRE_GRACE_FRAMES）。 */
  private _retired: { texture: GPUTexture; retireAtFrame: number }[] = [];

  /** 单调帧计数（onFrameEnd 递增）。 */
  private _frameIndex = 0;

  /** 退役纹理在销毁前保留的帧数（与 OverlayPass 保持一致）。 */
  static readonly RETIRE_GRACE_FRAMES = 3;

  constructor(
    device: GPUDevice,
    format: GPUTextureFormat,
    label: string,
    extraUsage: GPUFlagsConstant = 0,
  ) {
    this._device     = device;
    this._format     = format;
    this._label      = label;
    this._extraUsage = extraUsage;
  }

  /** 当前纹理（null = 未创建或已 retire）。 */
  get texture(): GPUTexture | null { return this._texture; }

  /** 当前纹理视图（null = 未创建或已 retire）。 */
  get view(): GPUTextureView | null { return this._view; }

  get width(): number  { return this._width; }
  get height(): number { return this._height; }

  /** 退役队列长度（测试 / 调试用）。 */
  get retiredCount(): number { return this._retired.length; }

  /**
   * 创建（或按需重建）RT。同尺寸时早退返回 false —— 不销毁不重建，
   * 视图保持有效（resize 高频抖动零开销，先例 GBufferPass.resize）。
   *
   * @returns 本调用是否发生了纹理创建/重建。
   */
  ensure(width: number, height: number): boolean {
    if (width === this._width && height === this._height && this._texture) {
      return false;
    }

    // 尺寸变化：旧纹理延迟退役（可能有 in-flight submit 引用）。
    this.retire();

    this._width  = width;
    this._height = height;
    this._texture = this._device.createTexture({
      label: this._label,
      size:  { width, height },
      format: this._format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | this._extraUsage,
    });
    this._view = this._texture.createView();
    return true;
  }

  /** 把当前纹理移入延迟退役队列（texture/view 变 null）。 */
  retire(): void {
    if (!this._texture) return;
    this._retired.push({
      texture: this._texture,
      retireAtFrame: this._frameIndex + RenderTexture.RETIRE_GRACE_FRAMES,
    });
    this._texture = null;
    this._view = null;
  }

  /** 帧结束推进：销毁已满宽限期的退役纹理。每帧调用一次。 */
  onFrameEnd(): void {
    this._frameIndex++;
    if (this._retired.length === 0) return;
    this._retired = this._retired.filter((r) => {
      if (r.retireAtFrame <= this._frameIndex) {
        r.texture.destroy();
        return false;
      }
      return true;
    });
  }

  /** 终局清理：立即销毁当前 + 全部退役纹理。调用后本对象不可再用。 */
  destroy(): void {
    if (this._texture) {
      this._texture.destroy();
      this._texture = null;
      this._view = null;
    }
    for (const r of this._retired) r.texture.destroy();
    this._retired = [];
    this._width = 0;
    this._height = 0;
  }
}

export default RenderTexture;
