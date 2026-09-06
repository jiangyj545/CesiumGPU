import { describe, expect, it } from 'vitest';
import { GBufferPass } from '../../src/scene/passes/GBufferPass';

/**
 * Globe.enableLighting 语义（CesiumJS 对齐）：默认 false（全球均匀亮），
 * GBufferPass 在 false 时把整个球面（base sphere + 瓦片 + 极冠 + 地形）
 * 切到 unlit 管线输出原始 albedo。
 *
 * 背景 bug：默认太阳光下背光半球只剩 ambientIntensity(0.08) × albedo，
 * 深蓝海洋 ≈ 25/255，与太空背景 (≈8/255) 肉眼不可区分 —— 用户看到的
 * “影像破洞”实为背光半球被压黑。
 *
 * node 环境无 WebGPU：mock pass encoder 记录 setPipeline 调用序列，
 * 验证纯逻辑（何时切管线、画完恢复），不验证 GPU 行为。
 */

/** 最小 GPURenderPipeline 替身（用标签区分实例）。 */
function fakePipeline(label: string): GPURenderPipeline {
  return { label } as unknown as GPURenderPipeline;
}

/** 记录 setPipeline / draw 调用序列的 pass encoder 替身。 */
function fakePass() {
  const calls: Array<{ fn: string; arg?: string }> = [];
  return {
    calls,
    pass: {
      setPipeline(p: { label?: string }) {
        calls.push({ fn: 'setPipeline', arg: p.label });
      },
      setBindGroup() { /* unused */ },
      setVertexBuffer() { /* unused */ },
      setIndexBuffer() { /* unused */ },
      drawIndexed() {
        calls.push({ fn: 'draw' });
      },
      draw() {
        calls.push({ fn: 'draw' });
      },
      end() { /* unused */ },
    } as unknown as GPURenderPassEncoder,
  };
}

/** Globe 替身：只暴露 GBufferPass.execute 访问的表面。 */
function fakeGlobe(show: boolean, enableLighting: boolean, withPrimitive = true) {
  return {
    show,
    enableLighting,
    primitive: withPrimitive
      ? ({
          label: 'base',
          drawOpaque(pass: GPURenderPassEncoder) {
            pass.drawIndexed(6);
          },
        } as never)
      : null,
    drawTiles(pass: GPURenderPassEncoder) {
      pass.drawIndexed(3); // 记一次 draw 即可
    },
  } as never;
}

/** GBufferPass.execute 只用到 inputs 的这些字段；primitives.opaque 为空。 */
function makeInputs(globe: never, unlitPipeline: GPURenderPipeline | null) {
  return {
    globe,
    primitives: { opaque: [] },
    cameraBindGroup: {} as GPUBindGroup,
    opaquePipeline: fakePipeline('opaque'),
    unlitPipeline,
    frustumCullingEnabled: () => false,
  };
}

function makePass(inputs: ReturnType<typeof makeInputs>) {
  const ctx = { device: {} } as never;
  const pass = new GBufferPass(ctx, 4, 4, inputs as never);
  const { calls, pass: encoder } = fakePass();
  // 测试替身：拦截私有渲染通道创建（node 下没有真实 GPU 纹理）
  (pass as unknown as Record<string, unknown>)['_beginRenderPass'] = () => encoder;
  return { pass, calls };
}

describe('Globe.enableLighting 管线切换', () => {
  it('enableLighting 默认 false（Globe 字段默认值）', async () => {
    const { Globe } = await import('../../src/scene/Globe');
    // 字段初始化不触 GPU，context 传最小替身即可
    const globe = new Globe({ device: {} } as never);
    expect(globe.enableLighting).toBe(false);
  });

  it('enableLighting=false 时球面走 unlit 管线，画完恢复 opaque', () => {
    const globe = fakeGlobe(true, false);
    const { pass, calls } = makePass(makeInputs(globe, fakePipeline('unlit')));
    pass.execute({} as GPUCommandEncoder);
    const pipelines = calls.filter((c) => c.fn === 'setPipeline').map((c) => c.arg);
    // 序列：初始 opaque → 球体 unlit → 恢复 opaque
    expect(pipelines).toEqual(['opaque', 'unlit', 'opaque']);
  });

  it('enableLighting=true 时球面保持 opaque（lit）管线', () => {
    const globe = fakeGlobe(true, true);
    const { pass, calls } = makePass(makeInputs(globe, fakePipeline('unlit')));
    pass.execute({} as GPUCommandEncoder);
    const pipelines = calls.filter((c) => c.fn === 'setPipeline').map((c) => c.arg);
    // 序列：初始 opaque → 球体 opaque → 恢复 opaque（重复 setPipeline 无害）
    expect(pipelines).toEqual(['opaque', 'opaque', 'opaque']);
  });

  it('unlitPipeline=null 时安全回退 opaque（不崩溃）', () => {
    const globe = fakeGlobe(true, false);
    const { pass, calls } = makePass(makeInputs(globe, null));
    expect(() => pass.execute({} as GPUCommandEncoder)).not.toThrow();
    const pipelines = calls.filter((c) => c.fn === 'setPipeline').map((c) => c.arg);
    expect(pipelines.every((p) => p === 'opaque')).toBe(true);
  });

  it('globe.show=false 时不绘制球体', () => {
    const globe = fakeGlobe(false, false);
    const { pass, calls } = makePass(makeInputs(globe, fakePipeline('unlit')));
    pass.execute({} as GPUCommandEncoder);
    expect(calls.some((c) => c.fn === 'draw')).toBe(false);
    // 只有 execute 开头的初始 setPipeline
    expect(calls.filter((c) => c.fn === 'setPipeline')).toHaveLength(1);
  });
});
