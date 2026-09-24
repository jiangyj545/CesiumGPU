import { describe, expect, it } from 'vitest';
import { PostProcessManager } from '../../src/postprocessing/PostProcessManager';

// node 环境无 WebGPU 全局枚举 —— 桩范式照 tests/picking/pickViewport.spec.ts
const g = globalThis as unknown as Record<string, unknown>;
g.GPUTextureUsage ??= { RENDER_ATTACHMENT: 0x10, TEXTURE_BINDING: 0x08, COPY_DST: 0x04 };
g.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };

/** 记录型桩 encoder：统计 beginRenderPass 调用（= 编码的 pass 数）。 */
function makeStubEncoder() {
  let passes = 0;
  const encoder = {
    beginRenderPass: () => {
      passes++;
      return {
        setPipeline: () => undefined,
        setBindGroup: () => undefined,
        draw: () => undefined,
        end: () => undefined,
      };
    },
  } as unknown as GPUCommandEncoder;
  return { encoder, count: () => passes };
}

describe('PostProcessManager（桩 device）', () => {
  it('enabled=false 时 execute 不编码任何 pass（零回归约束）', () => {
    const manager = new PostProcessManager({} as unknown as GPUDevice, 'bgra8unorm');
    const { encoder, count } = makeStubEncoder();

    manager.execute(encoder, {} as GPUTextureView, {} as GPUTextureView, 0);

    expect(manager.enabled).toBe(false);
    expect(count()).toBe(0);
  });

  it('destroy 后 execute 仍不编码（幂等清理）', () => {
    const manager = new PostProcessManager({} as unknown as GPUDevice, 'bgra8unorm');
    manager.destroy();
    const { encoder, count } = makeStubEncoder();
    manager.execute(encoder, {} as GPUTextureView, {} as GPUTextureView, 0);
    expect(count()).toBe(0);
  });

  it('projectSun：静态投影入口存在（真机相机语义由示例 93 目测验证）', () => {
    expect(typeof PostProcessManager.projectSun).toBe('function');
  });
});
