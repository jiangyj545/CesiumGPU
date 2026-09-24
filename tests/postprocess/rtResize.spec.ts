import { describe, expect, it } from 'vitest';
import { RenderTexture } from '../../src/renderer/RenderTexture';

// node 环境无 WebGPU 全局枚举 —— RenderTexture.ensure 引用它们
// （桩范式照 tests/picking/pickViewport.spec.ts:8-12）
const g = globalThis as unknown as Record<string, unknown>;
g.GPUTextureUsage ??= {
  RENDER_ATTACHMENT: 0x10,
  TEXTURE_BINDING: 0x08,
  COPY_DST: 0x08,
};
g.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };

/** 记录型桩 device：统计 createTexture / destroy 次数。 */
function makeStubDevice() {
  let created = 0;
  let destroyed = 0;
  const device = {
    createTexture: (_desc: GPUTextureDescriptor) => {
      created++;
      return {
        label: 'stub',
        createView: () => ({ __view: created }),
        destroy: () => { destroyed++; },
      } as unknown as GPUTexture;
    },
  } as unknown as GPUDevice;
  return { device, counts: () => ({ created, destroyed }) };
}

describe('RenderTexture 生命周期', () => {
  it('同尺寸 ensure 早退：不销毁不重建，视图保持', () => {
    const { device, counts } = makeStubDevice();
    const rt = new RenderTexture(device, 'rgba16float', 'Test.RT');

    expect(rt.ensure(1920, 1080)).toBe(true);
    const view = rt.view;
    expect(rt.ensure(1920, 1080)).toBe(false); // 早退
    expect(rt.view).toBe(view);                 // 视图未换
    expect(counts()).toEqual({ created: 1, destroyed: 0 });
  });

  it('尺寸变化触发重建，旧纹理进退役队列（延迟 3 帧）', () => {
    const { device, counts } = makeStubDevice();
    const rt = new RenderTexture(device, 'rgba16float', 'Test.RT');
    rt.ensure(1920, 1080);
    const oldTexture = rt.texture;

    rt.ensure(960, 540);
    expect(counts().created).toBe(2);
    expect(rt.texture).not.toBe(oldTexture);
    expect(rt.retiredCount).toBe(1);
    expect(counts().destroyed).toBe(0); // 未满宽限期不销毁

    rt.onFrameEnd(); // 帧 1
    rt.onFrameEnd(); // 帧 2
    expect(counts().destroyed).toBe(0);
    rt.onFrameEnd(); // 帧 3 → retireAtFrame 到期
    expect(counts().destroyed).toBe(1);
    expect(rt.retiredCount).toBe(0);
  });

  it('retire() 后 view 为 null（当前帧不再可绑）', () => {
    const { device } = makeStubDevice();
    const rt = new RenderTexture(device, 'rgba16float', 'Test.RT');
    rt.ensure(100, 100);
    expect(rt.view).not.toBeNull();
    rt.retire();
    expect(rt.view).toBeNull();
    expect(rt.texture).toBeNull();
  });

  it('destroy() 立即清空当前 + 全部退役纹理', () => {
    const { device, counts } = makeStubDevice();
    const rt = new RenderTexture(device, 'rgba16float', 'Test.RT');
    rt.ensure(100, 100);
    rt.ensure(200, 200); // 第一张进退役队列
    rt.destroy();
    expect(counts().destroyed).toBe(2);
    expect(rt.texture).toBeNull();
    expect(rt.retiredCount).toBe(0);
  });
});
