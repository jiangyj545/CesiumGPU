import { VideoFrameSurface } from './VideoSource';

/**
 * VideoTexture - GPU texture sink for a {@link VideoFrameSurface}: holds an
 * `rgba8unorm` texture and pulls the latest frame from a video surface once
 * per frame via `GPUQueue.copyExternalImageToTexture`.
 *
 * No mipmaps: regenerating the mip chain every frame costs too much for a
 * live video feed. Tilted-view moiré is deferred to phase 2, where the
 * existing MipmapGenerator can be run on a per-frame budget.
 *
 * Texture usage flags mirror ImageryLayer tile textures
 * (TEXTURE_BINDING | COPY_DST | RENDER_ATTACHMENT) so the overlay pass can
 * sample the texture and reuse it as a render target if needed.
 */
export class VideoTexture {
  private _device: GPUDevice;
  private _texture: GPUTexture | null = null;
  private _view: GPUTextureView | null = null;
  private _width: number;
  private _height: number;
  private _warnedSize = '';
  private _destroyed = false;

  constructor(device: GPUDevice, width: number, height: number) {
    this._device = device;
    this._width  = Math.max(1, Math.floor(width));
    this._height = Math.max(1, Math.floor(height));
    this._createTexture();
  }

  /** The uploaded video texture, or null after destroy. */
  get texture(): GPUTexture | null { return this._texture; }

  /** View over {@link texture}, or null after destroy. */
  get view(): GPUTextureView | null { return this._view; }

  get width(): number { return this._width; }
  get height(): number { return this._height; }

  /**
   * Pull a new frame from `surface` into the texture. Returns whether an
   * upload happened this call. When the surface's size no longer matches the
   * texture, the texture is destroyed and recreated at the new size (warned
   * once per distinct size).
   */
  update(surface: VideoFrameSurface): boolean {
    if (this._destroyed || this._texture === null) return false;
    if (!surface.hasNewFrame()) return false;

    const source = surface.externalSource();
    if (source === null) return false;

    const width  = surface.width;
    const height = surface.height;
    if (width > 0 && height > 0 && (width !== this._width || height !== this._height)) {
      const sizeKey = `${width}x${height}`;
      if (this._warnedSize !== sizeKey) {
        this._warnedSize = sizeKey;
        console.warn(
          `VideoTexture: source size changed from ${this._width}x${this._height} to ${sizeKey}; recreating texture`,
        );
      }
      this._width  = width;
      this._height = height;
      this._createTexture();
    }

    // No alpha premultiplication: in the WebGPU spec this knob lives on the
    // destination as `premultipliedAlpha` (default false — stated explicitly
    // here to make the video-overlay contract obvious).
    this._device.queue.copyExternalImageToTexture(
      { source, flipY: false },
      { texture: this._texture, premultipliedAlpha: false },
      { width: this._width, height: this._height },
    );
    return true;
  }

  /** Destroy the GPU texture. Idempotent; a destroyed texture stops updating. */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._texture?.destroy();
    this._texture = null;
    this._view = null;
  }

  // ── Private ────────────────────────────────────────────────────────────────

  /** (Re)create the rgba8unorm texture at the current size. */
  private _createTexture(): void {
    this._texture?.destroy();
    this._texture = this._device.createTexture({
      label:  `VideoTexture.${this._width}x${this._height}`,
      size:   { width: this._width, height: this._height },
      format: 'rgba8unorm',
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this._view = this._texture.createView();
  }
}

/**
 * Shared sampler for sampling a VideoTexture in the overlay pass:
 * bilinear, clamp-to-edge (video frames are edge-clamped, never tiled).
 */
export function createVideoSampler(device: GPUDevice): GPUSampler {
  return device.createSampler({
    minFilter:    'linear',
    magFilter:    'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
}
