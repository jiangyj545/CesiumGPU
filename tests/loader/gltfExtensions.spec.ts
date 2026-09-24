import { afterEach, describe, expect, it, vi } from 'vitest';
import { GltfLoader } from '../../src/loader/GltfLoader';
import { buildMinimalGlb } from './helpers';

// node 环境直接走 parseGlbBuffer（无需 fetch，全部内嵌于 GLB）

afterEach(() => {
  vi.restoreAllMocks();
});

describe('GltfLoader extensions 门禁', () => {
  it('required 含不支持扩展 → 抛错并逐名列出', async () => {
    const glb = buildMinimalGlb({
      extensionsRequired: ['KHR_draco_mesh_compression', 'KHR_texture_transform'],
    });
    await expect(GltfLoader.parseGlbBuffer(glb)).rejects.toThrow(
      /Unsupported glTF extensions: KHR_draco_mesh_compression, KHR_texture_transform/
    );
  });

  it('required 含 KHR_texture_basisu（白名单内）→ 正常解析不抛', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const glb = buildMinimalGlb({ extensionsRequired: ['KHR_texture_basisu'] });
    const model = await GltfLoader.parseGlbBuffer(glb);
    expect(model.primitives).toHaveLength(1);
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('used 含不支持扩展 → 仅 console.warn 降级，解析成功', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const glb = buildMinimalGlb({ extensionsUsed: ['KHR_materials_variants'] });
    const model = await GltfLoader.parseGlbBuffer(glb);
    expect(model.primitives).toHaveLength(1);
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(String(warnSpy.mock.calls[0][0])).toContain('KHR_materials_variants');
  });

  it('回归：无 extensions 的旧 glb 解析成功且零 warn（行为不变）', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const glb = buildMinimalGlb();
    const model = await GltfLoader.parseGlbBuffer(glb);
    expect(model.primitives).toHaveLength(1);
    expect(warnSpy).not.toHaveBeenCalled();
  });
});
