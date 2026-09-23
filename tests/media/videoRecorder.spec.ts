import { afterEach, describe, expect, it } from 'vitest';
import { VideoRecorder } from '../../src/media/VideoRecorder';
import type { VideoRecorderHost } from '../../src/media/VideoRecorder';

/**
 * VideoRecorder 纯逻辑回归（node 环境，无 WebGPU/DOM）。
 *
 * 桩模式先例：tests/picking/pickViewport.spec.ts（globalThis 补枚举 + 记录型
 * 对象）。本测试把 MediaRecorder / canvas.captureStream / requestAnimationFrame
 * 全部换成记录型桩，断言：
 *  - mime 候选链降级与显式优先
 *  - 录制前后的画布尺寸保存/恢复与 scene.resize 调用序列（autoResize 挂起）
 *  - 状态机 idle → recording → idle 与错误路径不静默吞
 */

const g = globalThis as unknown as Record<string, unknown>;

let rafCallCount = 0;

/** 记录型 MediaRecorder 桩：记录构造参数与 start/stop 调用，stop 异步触发回调。 */
class FakeMediaRecorder {
  static supported = new Set<string>();
  static instances: FakeMediaRecorder[] = [];

  static isTypeSupported(type: string): boolean {
    return FakeMediaRecorder.supported.has(type);
  }

  state: 'inactive' | 'recording' = 'inactive';
  options: MediaRecorderOptions;
  startCalls: Array<number | undefined> = [];
  stopCount = 0;
  stream: unknown;
  ondataavailable: ((e: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;

  constructor(stream: unknown, options: MediaRecorderOptions) {
    this.stream = stream;
    this.options = options;
    FakeMediaRecorder.instances.push(this);
  }

  start(timeslice?: number): void {
    this.state = 'recording';
    this.startCalls.push(timeslice);
  }

  stop(): void {
    this.state = 'inactive';
    this.stopCount += 1;
    // 真实现语义：stop → 最后一次 dataavailable → onstop
    queueMicrotask(() => {
      this.ondataavailable?.({ data: new Blob(['chunk']) });
      this.onstop?.();
    });
  }
}

/** 记录型宿主桩：duck-typing 满足 VideoRecorderHost，不引真 Viewer。 */
function makeHost(w = 1280, h = 720) {
  const canvas = {
    width: w,
    height: h,
    captureStreamCalls: [] as number[],
    captureStream(fps: number) {
      this.captureStreamCalls.push(fps);
      return { __fakeStream: true };
    },
  };
  const scene = {
    resizeCalls: [] as Array<[number, number]>,
    resize(width: number, height: number): void {
      this.resizeCalls.push([width, height]);
    },
  };
  const host = { canvas, scene } as unknown as VideoRecorderHost;
  return { canvas, scene, host };
}

/** 安装全局桩（MediaRecorder + rAF setTimeout 桩），并重置记录状态。 */
function installStubs(supported: string[] = []): void {
  FakeMediaRecorder.supported = new Set(supported);
  FakeMediaRecorder.instances = [];
  g.MediaRecorder = FakeMediaRecorder;
  rafCallCount = 0;
  g.requestAnimationFrame = (cb: () => void) => {
    rafCallCount += 1;
    return setTimeout(cb, 0) as unknown as number;
  };
}

afterEach(() => {
  delete g.MediaRecorder;
  delete g.requestAnimationFrame;
});

describe('VideoRecorder', () => {
  it('① mime 候选链降级：vp9 不支持 → 落到 vp8', async () => {
    installStubs(['video/webm;codecs=vp8', 'video/webm']);
    const { host } = makeHost();
    const rec = new VideoRecorder(host);

    await rec.startRecording({ width: 3840, height: 2160 });

    expect(FakeMediaRecorder.instances).toHaveLength(1);
    expect(FakeMediaRecorder.instances[0].options.mimeType).toBe('video/webm;codecs=vp8');
    await rec.stopRecording();
  });

  it('① mime 候选链全不支持 → startRecording 拒绝并列出候选', async () => {
    installStubs([]);
    const { host } = makeHost();
    const rec = new VideoRecorder(host);

    // 不应静默吞：候选链逐一列出
    await expect(rec.startRecording()).rejects.toThrow(/video\/webm;codecs=vp9.*video\/webm/s);
    expect(FakeMediaRecorder.instances).toHaveLength(0);
  });

  it('② 显式 mimeType 优先（不走候选链探测）', async () => {
    installStubs([]); // supported 为空也不影响显式指定
    const { host } = makeHost();
    const rec = new VideoRecorder(host);

    await rec.startRecording({ mimeType: 'video/mp4;codecs=avc1' });

    expect(FakeMediaRecorder.instances[0].options.mimeType).toBe('video/mp4;codecs=avc1');
    await rec.stopRecording();
  });

  it('③ start 暂停 autoResize + 按序 resize + 等一帧，stop 恢复原尺寸', async () => {
    installStubs(['video/webm']);
    const { host, canvas, scene } = makeHost(1280, 720);
    const rec = new VideoRecorder(host);
    expect(rec.autoResizePaused).toBe(false);

    await rec.startRecording({ width: 3840, height: 2160 });

    // 录制中：挂起标志位打开（宿主 _resizeCanvas 据此直接 return）
    expect(rec.autoResizePaused).toBe(true);
    expect(rec.state).toBe('recording');
    // canvas 直设目标尺寸 → scene.resize 走一次（相机纵横比随之同步）
    expect(canvas.width).toBe(3840);
    expect(canvas.height).toBe(2160);
    expect(scene.resizeCalls).toEqual([[3840, 2160]]);
    // resize 后等了一帧才 recorder.start(timeslice=1000)
    expect(rafCallCount).toBe(1);
    expect(FakeMediaRecorder.instances[0].startCalls).toEqual([1000]);

    const blob = await rec.stopRecording();

    // onstop：恢复录制前尺寸 → 再走一次 scene.resize → 解除挂起
    expect(rec.autoResizePaused).toBe(false);
    expect(canvas.width).toBe(1280);
    expect(canvas.height).toBe(720);
    expect(scene.resizeCalls).toEqual([
      [3840, 2160],
      [1280, 720],
    ]);
    expect(blob).toBeInstanceOf(Blob);
  });

  it('④ 非 recording 态调用 stopRecording → 抛错', async () => {
    installStubs(['video/webm']);
    const { host } = makeHost();
    const rec = new VideoRecorder(host);

    await expect(rec.stopRecording()).rejects.toThrow(/只能在录制中调用/);
  });

  it('⑤ state 机 idle → recording → idle，Blob 带 mime 类型', async () => {
    installStubs(['video/webm']);
    const { host } = makeHost();
    const rec = new VideoRecorder(host);

    expect(rec.state).toBe('idle');
    await rec.startRecording({ fps: 30 });
    expect(rec.state).toBe('recording');

    // 录制中重复 start → 拒绝带原因
    await expect(rec.startRecording()).rejects.toThrow(/已在录制中/);

    const blob = await rec.stopRecording();
    expect(rec.state).toBe('idle');
    expect(blob.type).toBe('video/webm');
    expect(blob.size).toBeGreaterThan(0);
  });

  it('captureStream 默认 60fps；环境不支持时抛错', () => {
    installStubs();
    const { host, canvas } = makeHost();
    const rec = new VideoRecorder(host);

    expect(rec.isSupported()).toBe(true);
    rec.captureStream();
    expect(canvas.captureStreamCalls).toEqual([60]);
    rec.captureStream(30);
    expect(canvas.captureStreamCalls).toEqual([60, 30]);

    const bare = new VideoRecorder(
      { canvas: {} as HTMLCanvasElement, scene: { resize() {} } } as VideoRecorderHost
    );
    expect(bare.isSupported()).toBe(false);
    expect(() => bare.captureStream()).toThrow(/captureStream/);
  });

  it('bitrate 默认自适应：4K@60 clamp 到 45Mbps 上限', async () => {
    installStubs(['video/webm']);
    const { host } = makeHost();
    const rec = new VideoRecorder(host);

    await rec.startRecording({ width: 3840, height: 2160 });
    // 3840*2160*60*0.15 ≈ 74.6Mbps → clamp 45Mbps；小分辨率 720p@60 原样保留
    expect(FakeMediaRecorder.instances[0].options.videoBitsPerSecond).toBe(45_000_000);
    await rec.stopRecording();

    const { host: host2 } = makeHost(1280, 720);
    const rec2 = new VideoRecorder(host2);
    await rec2.startRecording({});
    // 1280*720*60*0.15 = 8.29Mbps（高于 8Mbps 下限，原样保留）
    expect(FakeMediaRecorder.instances[1].options.videoBitsPerSecond).toBe(8_294_400);
    await rec2.stopRecording();
  });
});
