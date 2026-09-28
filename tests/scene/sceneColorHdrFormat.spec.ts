import { describe, expect, it, vi } from 'vitest';
import { Scene } from '../../src/scene/Scene';
import { EffectManager } from '../../src/effect/EffectManager';
import { BowSpray } from '../../src/effect/BowSpray';
import { CausticsFloor } from '../../src/effect/CausticsFloor';
import { defaultPostProcessConfig } from '../../src/postprocessing/PostProcessConfig';
import type { Context } from '../../src/renderer/Context';
import type { Effect } from '../../src/effect/EffectManager';

/**
 * T5 真 HDR 全管线 — 记录型 mock（无真实 WebGPU，causticsFloorWetSand /
 * bowSpray.spec 同款桩模式）。断言：
 *  1. postProcess.enabled 时 SceneColorRT / SceneColorHistory 均以
 *     rgba16float 创建（copyTextureToTexture 源/目标同格式硬约束），且
 *     帧内确实发生了 RT → history 的整帧拷贝；
 *  2. Lighting / Transparent composite / Overlay 管线目标格式跟随
 *     rgba16float；WBOIT accum 管线保持内部目标（rgba16float + r8unorm
 *     revealage）不动；关闭后处理整链切回交换链 preferredFormat（LDR
 *     旧路径零回归——同一 Scene 实例来回切换验证）；
 *  3. EffectManager 向实现 setTargetFormat 协议的特效逐帧下发当前目标
 *     格式（未实现协议的特效不受影响、不传参数不下发）；
 *  4. CausticsFloor / BowSpray 收到变更格式后重建渲染管线（幂等：同格式
 *     不重建），未接线时保持 preferredFormat 旧行为；
 *  5. bloom 默认阈值 ≥ 1.0（HDR 线性域下 LDR 内容不误发光）。
 */

// node 环境无 WebGPU 运行时全局（causticsFloorWetSand / bowSpray.spec 同款兜底）
const g = globalThis as unknown as Record<string, unknown>;
g.GPUTextureUsage ??= {
  TEXTURE_BINDING: 0x08, COPY_DST: 0x04, RENDER_ATTACHMENT: 0x10, COPY_SRC: 0x02,
};
g.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
g.GPUBufferUsage ??= {
  MAP_READ: 0x01, VERTEX: 0x20, INDEX: 0x10, UNIFORM: 0x40,
  COPY_DST: 0x08, COPY_SRC: 0x04, STORAGE: 0x80,
};
g.GPUColorWrite ??= { RED: 0x1, GREEN: 0x2, BLUE: 0x4, ALPHA: 0x8 };

interface CapturedTexture { label: string; format: string }
interface CapturedPipeline { label: string; formats: string[] }
interface CapturedCopy { srcLabel: string; dstLabel: string }

/** RenderPass / ComputePass 通用空实现桩。 */
function makePassStub() {
  return {
    setPipeline: vi.fn(),
    setComputePipeline: vi.fn(),
    setBindGroup: vi.fn(),
    setVertexBuffer: vi.fn(),
    setIndexBuffer: vi.fn(),
    draw: vi.fn(),
    drawIndexed: vi.fn(),
    dispatchWorkgroups: vi.fn(),
    end: vi.fn(),
  };
}

/**
 * 组装一个可完整跑通 Scene.initialize() + Scene.render() 的记录型 harness：
 * device 捕获 createTexture / createRenderPipeline / copyTextureToTexture，
 * 其余一律返回轻量桩对象。
 */
function makeHarness() {
  const textures: CapturedTexture[] = [];
  const pipelines: CapturedPipeline[] = [];
  const copies: CapturedCopy[] = [];
  const passStub = makePassStub();

  const device = {
    createBuffer: (d: { label?: string }) => ({ label: d.label ?? '', destroy: vi.fn() }),
    createTexture: (d: { label?: string; format: string }) => {
      textures.push({ label: d.label ?? '', format: d.format });
      return { label: d.label, destroy: vi.fn(), createView: () => ({}) };
    },
    createShaderModule: vi.fn(() => ({})),
    createBindGroupLayout: vi.fn(() => ({})),
    createPipelineLayout: vi.fn(() => ({})),
    createBindGroup: vi.fn(() => ({})),
    createRenderPipeline: (d: {
      label?: string;
      fragment?: { targets?: { format: string }[] };
    }) => {
      pipelines.push({
        label: d.label ?? '',
        formats: (d.fragment?.targets ?? []).map((t) => t.format),
      });
      return { label: d.label, getBindGroupLayout: () => ({}) };
    },
    createComputePipeline: vi.fn(() => ({ getBindGroupLayout: () => ({}) })),
    createSampler: vi.fn(() => ({})),
    createCommandEncoder: vi.fn(() => ({
      beginRenderPass: () => passStub,
      beginComputePass: () => passStub,
      copyTextureToTexture: (src: { texture: { label?: string } }, dst: { texture: { label?: string } }) => {
        copies.push({ srcLabel: src.texture.label ?? '', dstLabel: dst.texture.label ?? '' });
      },
      finish: () => ({}),
    })),
    queue: {
      writeBuffer: vi.fn(),
      writeTexture: vi.fn(),
      submit: vi.fn(),
      copyExternalImageToTexture: vi.fn(),
    },
  };

  const context = {
    device,
    canvas: { width: 320, height: 240 },
    preferredFormat: 'bgra8unorm',
    initialized: true,
    deviceLost: false,
    get drawingBufferWidth() { return this.canvas.width; },
    get drawingBufferHeight() { return this.canvas.height; },
    createSampler: () => ({}),
    createShaderModule: () => ({}),
    writeBuffer: () => {},
    getCurrentTextureView: () => ({}),
    submit: () => {},
    resize: () => {},
  } as unknown as Context;

  const scene = new Scene(context);
  return { scene, textures, pipelines, copies };
}

/** 取某 label 管线「最后一次创建」的目标格式串（重建语义断言用）。 */
function lastFormats(pipelines: CapturedPipeline[], label: string): string[] | null {
  for (let i = pipelines.length - 1; i >= 0; i--) {
    if (pipelines[i].label === label) return pipelines[i].formats;
  }
  return null;
}

/** 最小可执行特效桩（可选携带 setTargetFormat 协议方法）。 */
function makeEffectSpy(withProtocol: boolean): Effect & { setTargetFormat?: (f: string) => void } {
  return {
    label: withProtocol ? 'SpyWithProtocol' : 'SpyPlain',
    show: true,
    prepare: () => {},
    execute: vi.fn(),
    destroy: () => {},
    ...(withProtocol ? { setTargetFormat: vi.fn() } : {}),
  } as unknown as Effect & { setTargetFormat?: (f: string) => void };
}

describe('T5 真 HDR：Scene 颜色 RT 与全链管线格式', () => {
  it('postProcess.enabled：RT/history 均 rgba16float，pass 管线格式跟随，特效收到 HDR 格式', async () => {
    const h = makeHarness();
    const withProtocol = makeEffectSpy(true);
    const plain = makeEffectSpy(false);
    h.scene.effects.add(withProtocol);
    h.scene.effects.add(plain);

    h.scene.postProcess.enabled = true;
    await h.scene.initialize();
    h.scene.render();

    // ── 离屏 RT 与折射历史：rgba16float 单一来源 ──
    const rt = h.textures.find((t) => t.label === 'Scene.SceneColorRT');
    const hist = h.textures.find((t) => t.label === 'Scene.SceneColorHistory');
    expect(rt?.format).toBe('rgba16float');
    expect(hist?.format).toBe('rgba16float');
    // copyTextureToTexture 同格式硬约束：帧内 RT → history 整帧拷贝发生过
    expect(h.copies.some(
      (c) => c.srcLabel === 'Scene.SceneColorRT' && c.dstLabel === 'Scene.SceneColorHistory',
    )).toBe(true);

    // ── pass 管线目标格式：HDR 路径全链 rgba16float ──
    expect(lastFormats(h.pipelines, 'LightingPipeline')).toEqual(['rgba16float']);
    expect(lastFormats(h.pipelines, 'TransparentCompositePipeline')).toEqual(['rgba16float']);
    expect(lastFormats(h.pipelines, 'OverlayPointPipeline')).toEqual(['rgba16float']);
    expect(lastFormats(h.pipelines, 'OverlayLabelPipeline')).toEqual(['rgba16float']);

    // ── WBOIT 内部目标不动：accum rgba16float + revealage r8unorm ──
    expect(lastFormats(h.pipelines, 'TransparentAccumPipeline')).toEqual(['rgba16float', 'r8unorm']);

    // ── 特效协议：实现者收到 HDR 格式，未实现者不受影响 ──
    expect(withProtocol.setTargetFormat).toHaveBeenLastCalledWith('rgba16float');
    expect(plain.execute).toHaveBeenCalled();
  });

  it('同一 Scene 来回开关：关闭后处理整链切回交换链 preferredFormat（LDR 零回归）', async () => {
    const h = makeHarness();
    const withProtocol = makeEffectSpy(true);
    h.scene.effects.add(withProtocol);
    await h.scene.initialize();

    h.scene.postProcess.enabled = true;
    h.scene.render();
    h.scene.postProcess.enabled = false;
    h.scene.render();

    expect(lastFormats(h.pipelines, 'LightingPipeline')).toEqual(['bgra8unorm']);
    expect(lastFormats(h.pipelines, 'TransparentCompositePipeline')).toEqual(['bgra8unorm']);
    expect(lastFormats(h.pipelines, 'OverlayPointPipeline')).toEqual(['bgra8unorm']);
    expect(lastFormats(h.pipelines, 'OverlayLabelPipeline')).toEqual(['bgra8unorm']);
    // 特效同样收到 LDR 目标格式
    expect(withProtocol.setTargetFormat).toHaveBeenLastCalledWith('bgra8unorm');

    // 再开回来：整链回到 rgba16float（管线重建幂等往复）
    h.scene.postProcess.enabled = true;
    h.scene.render();
    expect(lastFormats(h.pipelines, 'LightingPipeline')).toEqual(['rgba16float']);
    expect(lastFormats(h.pipelines, 'TransparentCompositePipeline')).toEqual(['rgba16float']);
  });

  it('从未开启后处理：不创建离屏 RT，特效走交换链 preferredFormat（旧路径零回归）', async () => {
    const h = makeHarness();
    const withProtocol = makeEffectSpy(true);
    h.scene.effects.add(withProtocol);
    await h.scene.initialize();

    h.scene.render();

    expect(h.textures.find((t) => t.label === 'Scene.SceneColorRT')).toBeUndefined();
    expect(h.textures.find((t) => t.label === 'Scene.SceneColorHistory')).toBeUndefined();
    expect(lastFormats(h.pipelines, 'LightingPipeline')).toEqual(['bgra8unorm']);
    expect(lastFormats(h.pipelines, 'TransparentCompositePipeline')).toEqual(['bgra8unorm']);
    expect(withProtocol.setTargetFormat).toHaveBeenLastCalledWith('bgra8unorm');
  });
});

describe('T5 真 HDR：EffectManager 目标格式协议', () => {
  it('execute 第 4 参下发格式；不传（向后兼容）不下发；show=false 跳过', () => {
    const mgr = new EffectManager();
    const spy = makeEffectSpy(true);
    const hidden = makeEffectSpy(true);
    hidden.show = false;
    mgr.add(spy);
    mgr.add(hidden);

    // 旧三参调用：不下发（向后兼容签名）
    mgr.execute({} as GPUCommandEncoder, {} as GPUTextureView, 1.0);
    expect(spy.setTargetFormat).not.toHaveBeenCalled();

    // 新签名：下发当前目标格式
    mgr.execute({} as GPUCommandEncoder, {} as GPUTextureView, 2.0, 'rgba16float');
    expect(spy.setTargetFormat).toHaveBeenCalledWith('rgba16float');
    // 不可见特效不参与（含格式下发）
    expect(hidden.setTargetFormat).not.toHaveBeenCalled();
  });
});

describe('T5 真 HDR：特效 targetFormat 变更触发管线重建', () => {
  /** 特效/管线级轻量桩 Context（prepare 建管线即可，无需 Scene 全链）。 */
  function makeEffectContext(preferred: string) {
    const pipelines: CapturedPipeline[] = [];
    const device = {
      createBuffer: (d: { label?: string }) => ({ label: d.label ?? '', destroy: vi.fn() }),
      createTexture: () => ({ destroy: vi.fn(), createView: () => ({}) }),
      createShaderModule: vi.fn(() => ({})),
      createBindGroupLayout: vi.fn(() => ({})),
      createPipelineLayout: vi.fn(() => ({})),
      createBindGroup: vi.fn(() => ({})),
      createRenderPipeline: (d: { label?: string; fragment?: { targets?: { format: string }[] } }) => {
        pipelines.push({
          label: d.label ?? '',
          formats: (d.fragment?.targets ?? []).map((t) => t.format),
        });
        return { label: d.label, getBindGroupLayout: () => ({}) };
      },
      createComputePipeline: vi.fn(() => ({ getBindGroupLayout: () => ({}) })),
      createSampler: vi.fn(() => ({})),
      queue: { writeBuffer: vi.fn(), writeTexture: vi.fn() },
    };
    const context = {
      device,
      canvas: { width: 64, height: 64 },
      preferredFormat: preferred,
      initialized: true,
      deviceLost: false,
      get drawingBufferWidth() { return this.canvas.width; },
      get drawingBufferHeight() { return this.canvas.height; },
      createSampler: () => ({}),
      createShaderModule: () => ({}),
      writeBuffer: () => {},
      getCurrentTextureView: () => ({}),
      submit: () => {},
      resize: () => {},
    } as unknown as Context;
    return { context, pipelines };
  }

  it('CausticsFloor：构造期 preferredFormat，setTargetFormat 变更重建、同格式幂等', () => {
    const { context, pipelines } = makeEffectContext('bgra8unorm');
    const floor = new CausticsFloor(context, { center: [121.5, 31.0] });

    floor.prepare();
    expect(lastFormats(pipelines, 'CausticsFloorPipeline')).toEqual(['bgra8unorm']);

    // 变更 → 管线重建为 rgba16float（HDR 场景目标）
    floor.setTargetFormat('rgba16float');
    floor.prepare();
    expect(lastFormats(pipelines, 'CausticsFloorPipeline')).toEqual(['rgba16float']);

    // 同格式幂等：setTargetFormat 早退 + prepare 再入不新增管线
    const before = pipelines.length;
    floor.setTargetFormat('rgba16float');
    floor.prepare();
    expect(pipelines.length).toBe(before);
  });

  it('BowSpray：构造期 preferredFormat，setTargetFormat 仅重建渲染管线', () => {
    const { context, pipelines } = makeEffectContext('bgra8unorm');
    // 构造硬要求：context.initialized + scene.gbufferPass（execute 才碰深度视图）
    const sceneStub = {
      gbufferPass: { getDepthView: () => ({}), textures: { depthView: {} } },
      camera: {},
    } as unknown as ConstructorParameters<typeof BowSpray>[1]['scene'];
    const spray = new BowSpray(context, { scene: sceneStub });

    expect(lastFormats(pipelines, 'BowSpray.RenderPipeline')).toEqual(['bgra8unorm']);
    const renderBefore = pipelines.filter((p) => p.label === 'BowSpray.RenderPipeline').length;

    spray.setTargetFormat('rgba16float');
    expect(lastFormats(pipelines, 'BowSpray.RenderPipeline')).toEqual(['rgba16float']);
    // 仅渲染管线重建（compute 管线与目标格式无关）
    const renderAfter = pipelines.filter((p) => p.label === 'BowSpray.RenderPipeline').length;
    expect(renderAfter).toBe(renderBefore + 1);

    // 同格式幂等 + destroy 后不再重建
    spray.setTargetFormat('rgba16float');
    expect(pipelines.filter((p) => p.label === 'BowSpray.RenderPipeline').length).toBe(renderAfter);
    spray.destroy();
    spray.setTargetFormat('bgra8unorm');
    expect(pipelines.filter((p) => p.label === 'BowSpray.RenderPipeline').length).toBe(renderAfter);
  });
});

describe('T5 真 HDR：bloom 阈值默认值', () => {
  it('默认 threshold ≥ 1.0（HDR 线性域下 LDR 内容不误发光）', () => {
    expect(defaultPostProcessConfig().bloom.threshold).toBeGreaterThanOrEqual(1.0);
  });
});
