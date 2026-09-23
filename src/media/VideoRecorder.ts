/**
 * VideoRecorder - 4K 视频输出（画布 MediaRecorder 录制）。
 *
 * 职责：
 * - 把渲染画布采集为 MediaStream（canvas.captureStream）
 * - 录制前把画布 resize 到目标分辨率（如 3840x2160），绕过 dpr 直设尺寸，
 *   再走 viewer 的场景 resize 链（scene.resize 同步相机纵横比与各 pass）
 * - 录制结束（onstop）后恢复录制前尺寸并解除 autoResize 暂停
 *
 * 错误不静默吞（项目 Invariant 1）：start/stop 失败一律 reject 带原因；
 * 非 recording 态调用 stopRecording 直接抛错。
 *
 * Usage:
 * ```typescript
 * const viewer = new Viewer('cesiumContainer');
 * await viewer.initialize();
 * await viewer.startRecording({ width: 3840, height: 2160 });
 * // ... 相机巡航 3s ...
 * const blob = await viewer.stopRecording();
 * ```
 */

export interface RecordingOptions {
  /** 目标宽度（像素）。缺省用当前画布宽。 */
  width?: number;
  /** 目标高度（像素）。缺省用当前画布高。 */
  height?: number;
  /** 采集帧率。默认 60。 */
  fps?: number;
  /** 显式 mimeType（直接采用，不走候选链探测）。 */
  mimeType?: string;
  /** 视频比特率（bps）。缺省按 分辨率x帧率x0.15 自适应，clamp 到 8~45 Mbps。 */
  bitrate?: number;
}

/**
 * VideoRecorder 依赖的最小宿主接口（duck-typing）。
 * Viewer 结构性满足；测试可用桩代替，避免引入真引擎类。
 */
export interface VideoRecorderHost {
  /** 渲染画布（录制期间将直设 width/height）。 */
  canvas: HTMLCanvasElement;
  /** 场景（其 resize 已同步相机 aspectRatio 与各 render pass）。 */
  scene: { resize(width: number, height: number): void };
}

/** VideoRecorder 对外状态机。 */
export type VideoRecorderState = 'idle' | 'recording';

/** webm 编码候选链（按质量降序）。 */
const MIME_CANDIDATES = [
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
] as const;

const MIN_BITRATE_BPS = 8_000_000;
const MAX_BITRATE_BPS = 45_000_000;
/** timeslice：每 1s 产出一次 dataavailable 分片，降低长录制内存峰值。 */
const START_TIMESLICE_MS = 1000;

/** 分辨率x帧率x0.15 自适应比特率，clamp 到 [8, 45] Mbps。 */
function autoBitrate(width: number, height: number, fps: number): number {
  const raw = Math.round(width * height * fps * 0.15);
  return Math.min(MAX_BITRATE_BPS, Math.max(MIN_BITRATE_BPS, raw));
}

/** 等待一帧：优先 rAF（node 测试桩为 setTimeout），缺壳时退化为 16ms 定时。 */
function nextFrame(): Promise<void> {
  return new Promise((resolve) => {
    const g = globalThis as unknown as {
      requestAnimationFrame?: (cb: FrameRequestCallback) => number;
    };
    if (typeof g.requestAnimationFrame === 'function') {
      g.requestAnimationFrame(() => resolve());
    } else {
      setTimeout(resolve, 16);
    }
  });
}

export class VideoRecorder {
  private _host: VideoRecorderHost;
  /** 非 null 即处于 recording 态（onstop 中置回 null）。 */
  private _recorder: MediaRecorder | null = null;
  private _chunks: Blob[] = [];
  private _mimeType = '';
  private _savedWidth = 0;
  private _savedHeight = 0;
  private _blob: Blob | null = null;

  /**
   * 录制期间为 true —— 宿主（Viewer）应据此挂起容器自适应 resize，
   * 否则 ResizeObserver 会立刻把 4K 画布改回容器尺寸。
   */
  autoResizePaused = false;

  constructor(host: VideoRecorderHost) {
    this._host = host;
  }

  /** 'idle' | 'recording'。 */
  get state(): VideoRecorderState {
    return this._recorder ? 'recording' : 'idle';
  }

  /**
   * 环境能力探测：需要 MediaRecorder 且画布支持 captureStream。
   */
  isSupported(): boolean {
    const g = globalThis as unknown as { MediaRecorder?: unknown };
    if (typeof g.MediaRecorder !== 'function') return false;
    const canvas = this._host.canvas as unknown as { captureStream?: unknown };
    return typeof canvas.captureStream === 'function';
  }

  /**
   * 采集画布为 MediaStream（按指定帧率）。
   * @throws 环境不支持 captureStream 时抛错。
   */
  captureStream(fps = 60): MediaStream {
    const canvas = this._host.canvas as unknown as {
      captureStream?: (fps: number) => MediaStream;
    };
    if (typeof canvas.captureStream !== 'function') {
      throw new Error('当前环境不支持 canvas.captureStream —— 无法采集视频流');
    }
    return canvas.captureStream(fps);
  }

  /**
   * 开始录制：保存旧尺寸 → resize 到目标 → 等一帧 → recorder.start(timeslice)。
   * 任何一步失败都会恢复现场（尺寸 + autoResize 暂停解除）并 reject 带原因。
   */
  async startRecording(opts: RecordingOptions = {}): Promise<void> {
    if (this.state === 'recording') {
      throw new Error('VideoRecorder 已在录制中 —— 请先调用 stopRecording()');
    }
    if (!this.isSupported()) {
      throw new Error('当前环境缺少 MediaRecorder 或 canvas.captureStream，无法录制');
    }

    const fps = opts.fps ?? 60;
    const canvas = this._host.canvas;
    const width = opts.width ?? canvas.width;
    const height = opts.height ?? canvas.height;
    const mimeType = this._resolveMimeType(opts.mimeType);
    const bitrate = opts.bitrate ?? autoBitrate(width, height, fps);

    this.autoResizePaused = true;
    this._savedWidth = canvas.width;
    this._savedHeight = canvas.height;
    this._chunks = [];
    this._blob = null;
    this._mimeType = mimeType;

    try {
      // 绕过 dpr 直设画布尺寸，再走场景 resize 链（相机纵横比随 scene.resize 同步）
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        this._host.scene.resize(width, height);
      }
      // 等一帧：确保 WebGPU 以新尺寸完成首帧渲染后再开始编码
      await nextFrame();

      const g = globalThis as unknown as { MediaRecorder: typeof MediaRecorder };
      const recorder = new g.MediaRecorder(this.captureStream(fps), {
        mimeType,
        videoBitsPerSecond: bitrate,
      });
      recorder.ondataavailable = (e: BlobEvent) => {
        if (e.data && e.data.size > 0) this._chunks.push(e.data);
      };
      recorder.onerror = (ev) => {
        // 不吞：记录原因（数据分片仍按 ondataavailable 交付）
        console.error('[VideoRecorder] MediaRecorder 录制出错:', ev);
      };
      recorder.start(START_TIMESLICE_MS);
      this._recorder = recorder;
    } catch (err) {
      this._restoreCanvas();
      this.autoResizePaused = false;
      this._recorder = null;
      throw err;
    }
  }

  /**
   * 停止录制：onstop 后恢复录制前尺寸并 resolve Blob。
   * @throws 非 recording 态调用时抛错。
   */
  async stopRecording(): Promise<Blob> {
    if (this.state !== 'recording' || !this._recorder) {
      throw new Error(`stopRecording() 只能在录制中调用（当前 state=${this.state}）`);
    }
    const recorder = this._recorder;
    return new Promise<Blob>((resolve, reject) => {
      recorder.onstop = () => {
        this._restoreCanvas();
        this.autoResizePaused = false;
        this._blob = new Blob(this._chunks, { type: this._mimeType });
        this._recorder = null;
        resolve(this._blob);
      };
      try {
        recorder.stop();
      } catch (err) {
        this._recorder = null;
        reject(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  /**
   * 把最近一次 stopRecording 的结果下载为文件。
   * @throws 尚无录制结果时抛错。
   */
  download(filename?: string): void {
    if (!this._blob) {
      throw new Error('尚无录制结果 —— 请先完成一次 startRecording/stopRecording');
    }
    const url = URL.createObjectURL(this._blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename ?? `cesiumgpu-${new Date().toISOString().replace(/[:.]/g, '-')}.webm`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  // ── Private ────────────────────────────────────────────────────────────────

  /** 恢复录制前画布尺寸并走场景 resize 链。 */
  private _restoreCanvas(): void {
    if (!this._savedWidth || !this._savedHeight) return;
    this._host.canvas.width = this._savedWidth;
    this._host.canvas.height = this._savedHeight;
    this._host.scene.resize(this._savedWidth, this._savedHeight);
  }

  /** mime 解析：显式优先；否则沿候选链用 isTypeSupported 探测，全失败抛错列候选。 */
  private _resolveMimeType(explicit?: string): string {
    if (explicit) return explicit;
    const g = globalThis as unknown as {
      MediaRecorder?: { isTypeSupported?: (type: string) => boolean };
    };
    for (const type of MIME_CANDIDATES) {
      if (g.MediaRecorder?.isTypeSupported?.(type)) return type;
    }
    throw new Error(`浏览器不支持任何视频 mime 候选: ${MIME_CANDIDATES.join(' → ')}`);
  }
}

export default VideoRecorder;
