/**
 * VideoSource - Unified video frame sources for the gimbal-camera video
 * overlay pipeline.
 *
 * A {@link VideoFrameSurface} is the CPU-side image surface that backs
 * VideoTexture uploads: once per frame the overlay asks `hasNewFrame()` and,
 * when true, uploads `externalSource()` into the GPU texture via
 * `GPUQueue.copyExternalImageToTexture`. Implementations own all media
 * decoding / drawing details; the overlay only consumes pixels.
 *
 * Frame-delivery convention (applies to every implementation): the renderer
 * must call `hasNewFrame()` at most once per frame (VideoTexture.update does
 * exactly this). A second call within the same frame may legitimately return
 * false because the implementation has already latched the "changed" signal
 * of the current frame.
 */

/**
 * An image surface that yields uploadable frames for a video overlay.
 * Implementations: {@link HtmlVideoSource} (streaming `<video>`),
 * {@link CanvasAnimationSource} (procedural 2D canvas),
 * {@link VideoFrameSource} (WebCodecs `VideoFrame` producer).
 */
export interface VideoFrameSurface {
  /** Frame width in pixels (0 until known). */
  readonly width: number;
  /** Frame height in pixels (0 until known). */
  readonly height: number;
  /**
   * Whether a new frame is available since the last call; when true,
   * `externalSource()` returns copyable new content.
   */
  hasNewFrame(): boolean;
  /**
   * The source for `copyExternalImageToTexture` (a CanvasImageSource subset:
   * HTMLVideoElement | HTMLCanvasElement | OffscreenCanvas | VideoFrame),
   * or null while nothing is uploadable. Does not transfer ownership —
   * see {@link VideoFrameSource.consume} for the explicit-release variant.
   */
  externalSource(): GPUCopyExternalImageSourceInfo['source'] | null;
  /** Release media / non-GPU resources. Idempotent. */
  destroy(): void;
}

/** Options for {@link HtmlVideoSource}. */
export interface HtmlVideoSourceOptions {
  /** Default: true (browser autoplay policies require muted). */
  muted?: boolean;
  /** Default: true */
  loop?: boolean;
  /** Default: true */
  autoplay?: boolean;
}

/**
 * {@link VideoFrameSurface} backed by a streaming `<video>` element.
 * A new frame is detected as `readyState >= 2` with a changed
 * `currentTime` between calls.
 */
export class HtmlVideoSource implements VideoFrameSurface {
  private _video: HTMLVideoElement;
  private _lastTime = -1;
  private _destroyed = false;

  constructor(url: string, opts: HtmlVideoSourceOptions = {}) {
    const video = document.createElement('video');
    video.muted       = opts.muted    ?? true;
    video.loop        = opts.loop     ?? true;
    video.autoplay    = opts.autoplay ?? true;
    video.playsInline = true;
    video.crossOrigin = 'anonymous';
    video.src         = url;
    this._video = video;
  }

  get width(): number { return this._video.videoWidth; }
  get height(): number { return this._video.videoHeight; }

  hasNewFrame(): boolean {
    if (this._destroyed) return false;
    const video = this._video;
    if (video.readyState < 2) return false; // HAVE_CURRENT_DATA
    if (video.currentTime === this._lastTime) return false;
    this._lastTime = video.currentTime;
    return true;
  }

  externalSource(): GPUCopyExternalImageSourceInfo['source'] | null {
    if (this._destroyed) return null;
    return this._video;
  }

  /** Pause and detach the media element so the decoder and buffers free up. */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    const video = this._video;
    video.pause();
    video.removeAttribute('src');
    video.load(); // per spec: releases the media resource
  }
}

/** Options for {@link CanvasAnimationSource}. */
export interface CanvasAnimationSourceOptions {
  /** Target draw rate in frames per second. Default: 30 */
  fps?: number;
}

/**
 * {@link VideoFrameSurface} backed by a procedural 2D canvas, redrawn at a
 * fixed fps. Prefers an `OffscreenCanvas` (no DOM coupling, worker-safe) and
 * falls back to a detached `document.createElement('canvas')`.
 *
 * Drawing is lazy: `hasNewFrame()` itself performs the draw when the frame
 * interval has elapsed and then returns true, so one call both polls and
 * produces the frame. This avoids the double-call timing problem a separate
 * `advance()` would have (it would need the exact same timestamp as the
 * subsequent upload read).
 */
export class CanvasAnimationSource implements VideoFrameSurface {
  private _canvas: HTMLCanvasElement | OffscreenCanvas | undefined;
  private _ctx: CanvasRenderingContext2D | null;
  private readonly _drawFn: (ctx: CanvasRenderingContext2D, w: number, h: number, tSec: number) => void;
  private readonly _frameIntervalMs: number;
  private _lastDrawMs = -Infinity;
  private _destroyed = false;

  constructor(
    width: number,
    height: number,
    drawFn: (ctx: CanvasRenderingContext2D, w: number, h: number, tSec: number) => void,
    opts: CanvasAnimationSourceOptions = {},
  ) {
    this._drawFn = drawFn;
    this._frameIntervalMs = 1000 / Math.max(1, opts.fps ?? 30);

    // Node environments (unit tests / SSR) have neither OffscreenCanvas nor a
    // DOM document; construction must fail loudly so the caller knows this
    // source is browser-only.
    let canvas: HTMLCanvasElement | OffscreenCanvas;
    let ctx: CanvasRenderingContext2D | null;
    try {
      if (typeof OffscreenCanvas !== 'undefined') {
        canvas = new OffscreenCanvas(width, height);
        // OffscreenCanvasRenderingContext2D shares the 2D drawing API; the
        // draw callback uses only that common subset, so the view is widened.
        ctx = canvas.getContext('2d') as unknown as CanvasRenderingContext2D | null;
      } else if (typeof document !== 'undefined') {
        canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        ctx = canvas.getContext('2d');
      } else {
        throw new Error('no OffscreenCanvas and no DOM document available');
      }
      if (ctx === null) throw new Error('2D context unavailable');
    } catch (cause) {
      const reason = cause instanceof Error ? cause.message : String(cause);
      throw new Error(
        `CanvasAnimationSource requires a browser environment with canvas 2D support: ${reason}`,
      );
    }
    this._canvas = canvas;
    this._ctx = ctx;
  }

  get width(): number { return this._canvas?.width ?? 0; }
  get height(): number { return this._canvas?.height ?? 0; }

  hasNewFrame(): boolean {
    if (this._destroyed) return false;
    const now = performance.now();
    if (now - this._lastDrawMs < this._frameIntervalMs) return false;
    this._lastDrawMs = now;
    const ctx = this._ctx;
    if (ctx !== null) {
      this._drawFn(ctx, ctx.canvas.width, ctx.canvas.height, now / 1000.0);
    }
    return true;
  }

  externalSource(): GPUCopyExternalImageSourceInfo['source'] | null {
    if (this._destroyed) return null;
    return this._canvas ?? null;
  }

  /** Drop the canvas and draw-callback references. */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._canvas = undefined;
    this._ctx = null;
  }
}

/**
 * {@link VideoFrameSurface} fed by WebCodecs `VideoFrame`s.
 *
 * Lifecycle (simplified, explicit-release design): `pushFrame()` stores the
 * latest frame, closing any still-unconsumed previous one. `externalSource()`
 * hands the pending frame to the uploader **without** clearing it; after the
 * texture upload the consumer MUST call `consume()` to close the frame and
 * clear it. Skipping `consume()` keeps `hasNewFrame()` true (the same frame
 * would be re-uploaded every frame) and leaks the VideoFrame.
 */
export class VideoFrameSource implements VideoFrameSurface {
  private readonly _width: number;
  private readonly _height: number;
  private _pending: VideoFrame | null = null;
  private _destroyed = false;

  constructor(width: number, height: number) {
    this._width = width;
    this._height = height;
  }

  get width(): number { return this._width; }
  get height(): number { return this._height; }

  /** Store a newly decoded frame; a still-unconsumed previous frame is closed. */
  pushFrame(frame: VideoFrame): void {
    if (this._destroyed) {
      frame.close();
      return;
    }
    if (this._pending !== null) this._pending.close();
    this._pending = frame;
  }

  hasNewFrame(): boolean {
    return !this._destroyed && this._pending !== null;
  }

  /** The pending frame (ownership stays with this source until `consume()`). */
  externalSource(): GPUCopyExternalImageSourceInfo['source'] | null {
    if (this._destroyed) return null;
    return this._pending;
  }

  /**
   * Close and clear the pending frame. Call exactly once per uploaded frame,
   * after the VideoTexture upload.
   */
  consume(): void {
    if (this._pending !== null) {
      this._pending.close();
      this._pending = null;
    }
  }

  /** Close any unconsumed pending frame. Idempotent. */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this.consume();
  }
}
