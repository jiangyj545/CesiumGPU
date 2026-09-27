import { describe, expect, it, vi } from 'vitest';
import { SeabedHeightField } from '../../src/effect/SeabedHeightField';
import type { Context } from '../../src/renderer/Context';

/**
 * SeabedHeightField — 纯 CPU 验证（headless 无 WebGPU adapter，见项目坑记录）：
 * 默认 profile（中心露出/外缘水深/岸线穿越）、确定性 hash 噪声、
 * sampleHeight 场外与单调性、r32float 纹理创建参数（记录型桩 device）。
 */

// headless node 无 WebGPU 全局：与 tests/postprocess/rtResize.spec.ts 同款兜底
const g = globalThis as unknown as Record<string, unknown>;
g.GPUTextureUsage ??= {
  COPY_SRC: 0x01,
  COPY_DST: 0x02,
  TEXTURE_BINDING: 0x04,
  STORAGE_BINDING: 0x08,
  RENDER_ATTACHMENT: 0x10,
};

const N = 64;
const SIZE = 500;
const CENTER = { lon: 120.0, lat: 30.0 };

interface Capture {
  context: Context;
  created: GPUTextureDescriptor[];
  writeTexture: ReturnType<typeof vi.fn>;
  destroyedLabels: string[];
}

/** 记录型桩 Context：捕获 createTexture/writeTexture/destroy。 */
function makeContext(): Capture {
  const created: GPUTextureDescriptor[] = [];
  const writeTexture = vi.fn();
  const destroyedLabels: string[] = [];
  const device = {
    createTexture: (desc: GPUTextureDescriptor) => {
      created.push(desc);
      return {
        label: desc.label,
        createView: () => ({ label: `${desc.label}.View` }),
        destroy: () => destroyedLabels.push(String(desc.label)),
      } as unknown as GPUTexture;
    },
    queue: { writeTexture },
  } as unknown as GPUDevice;
  return { context: { initialized: true, device } as unknown as Context, created, writeTexture, destroyedLabels };
}

function makeField(capture: Capture): SeabedHeightField {
  return new SeabedHeightField(capture.context, {
    center: CENTER,
    sizeM: SIZE,
    gridN: N,
  });
}

describe('SeabedHeightField 默认 profile', () => {
  it('中心高度 > 1.0（沙洲露出水面）', () => {
    const cap = makeContext();
    const field = makeField(cap);
    expect(field.sampleHeight(0, 0)).toBeGreaterThan(1.0);
    expect(field.heights[(N / 2) * N + N / 2]).toBeGreaterThan(1.0);
  });

  it('r > shoreRadius*1.6 处 ≈ edgeDepthM（噪声 ±0.35 容差内）', () => {
    const cap = makeContext();
    const field = makeField(cap);
    const outer = SIZE * 0.22 * 1.8; // > 1.6·shoreR，仍在场内
    const samples: [number, number][] = [
      [outer, 0],
      [0, outer],
      [-outer, -outer * 0.7],
      [outer * 0.5, outer * 0.9],
    ];
    for (const [x, z] of samples) {
      const h = field.sampleHeight(x, z);
      expect(h).toBeLessThan(-6.5); // -7 + 0.35 上界
      expect(h).toBeGreaterThan(-7.4); // -7 - 0.35 下界
    }
  });

  it('存在岸线半径：沿 +x 扫描高度穿越 0', () => {
    const cap = makeContext();
    const field = makeField(cap);
    let prev = field.sampleHeight(0, 0);
    let crossed = false;
    for (let r = 1; r <= SIZE / 2; r += 0.5) {
      const h = field.sampleHeight(r, 0);
      if (prev > 0 && h <= 0) {
        crossed = true;
        break;
      }
      prev = h;
    }
    expect(crossed).toBe(true);
  });
});

describe('SeabedHeightField 确定性', () => {
  it('同参数两次构造高度数组逐位一致（hash 噪声无随机源）', () => {
    const a = makeField(makeContext());
    const b = makeField(makeContext());
    expect(a.heights.length).toBe(b.heights.length);
    for (let i = 0; i < a.heights.length; i++) {
      if (!Object.is(a.heights[i], b.heights[i])) {
        throw new Error(`heights[${i}] 不一致: ${a.heights[i]} vs ${b.heights[i]}`);
      }
    }
    // 确定性之外再确认数组内容本身有效（全有限数，非 NaN/Inf 填充）
    expect(a.heights.every(Number.isFinite)).toBe(true);
    expect(b.heights.every(Number.isFinite)).toBe(true);
  });
});

describe('SeabedHeightField.sampleHeight', () => {
  it('场外返回 edgeDepthM', () => {
    const cap = makeContext();
    const field = makeField(cap);
    expect(field.sampleHeight(SIZE * 0.6, 0)).toBe(-7);
    expect(field.sampleHeight(-SIZE * 0.6, 0)).toBe(-7);
    expect(field.sampleHeight(0, SIZE * 0.6)).toBe(-7);
    expect(field.sampleHeight(SIZE, SIZE)).toBe(-7);
  });

  it('与纹理数据一致：texel 中心采样精确等于数组值，整体单调性大致成立', () => {
    const cap = makeContext();
    const field = makeField(cap);

    // texel 中心（i,j）的平面坐标采样 = heights[j*N+i]（双线性退化为取值）
    for (const [i, j] of [[10, 20], [32, 32], [50, 8], [3, 61]] as const) {
      const x = ((i + 0.5) / N - 0.5) * SIZE;
      const z = ((j + 0.5) / N - 0.5) * SIZE;
      expect(field.sampleHeight(x, z)).toBe(field.heights[j * N + i]);
    }

    // 大致单调：中心 > 岸线内 > 1.6·shoreR 外（噪声容差下仍严格成立）
    const shoreR = SIZE * 0.22;
    const center = field.sampleHeight(0, 0);
    const inner = field.sampleHeight(shoreR, 0);
    const outer = field.sampleHeight(shoreR * 1.6, 0);
    expect(center).toBeGreaterThan(inner);
    expect(inner).toBeGreaterThan(outer);
  });
});

describe('SeabedHeightField 纹理创建', () => {
  it('r32float / 2d / gridN 尺寸 / TEXTURE_BINDING|COPY_DST，writeTexture 一次上传', () => {
    const cap = makeContext();
    const field = makeField(cap);

    expect(cap.created).toHaveLength(1);
    const desc = cap.created[0];
    expect(desc.format).toBe('r32float');
    expect(desc.dimension).toBe('2d');
    expect(desc.size).toEqual([N, N]);
    expect(desc.usage).toBe(GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST);

    expect(cap.writeTexture).toHaveBeenCalledTimes(1);
    const [dest, data, layout, extent] = cap.writeTexture.mock.calls[0];
    expect(dest).toEqual({ texture: expect.objectContaining({ label: 'SeabedHeightField' }) });
    expect(data).toBeInstanceOf(Float32Array);
    expect(data.length).toBe(N * N);
    expect(layout).toEqual({ bytesPerRow: N * 4, rowsPerImage: N });
    expect(extent).toEqual([N, N, 1]);

    // dispose：纹理销毁、view 置 null（CPU 高度数组保留）
    expect(field.seabedTexture).not.toBeNull();
    field.dispose();
    expect(field.seabedTexture).toBeNull();
    expect(cap.destroyedLabels).toEqual(['SeabedHeightField']);
    expect(field.heights.length).toBe(N * N);
  });

  it('centerRadial = ecefFromLonLatHeight 同款归一化位置', () => {
    const cap = makeContext();
    const field = makeField(cap);
    const len = Math.hypot(
      field.centerRadial.x,
      field.centerRadial.y,
      field.centerRadial.z,
    );
    expect(len).toBeGreaterThan(0.99);
    expect(len).toBeLessThan(1.01);
  });

  it('参数校验：Context 未初始化 / gridN 非法抛错', () => {
    const cap = makeContext();
    expect(
      () =>
        new SeabedHeightField({ initialized: false } as unknown as Context, {
          center: CENTER,
        }),
    ).toThrow(/initialize/);
    expect(
      () => new SeabedHeightField(cap.context, { center: CENTER, gridN: 3 }),
    ).toThrow(/gridN/);
  });

  it('bytesPerRow 256B 对齐：gridN=65 抛错（须 64 的倍数），gridN=64 正常构造', () => {
    const cap = makeContext();
    expect(
      () => new SeabedHeightField(cap.context, { center: CENTER, gridN: 65 }),
    ).toThrow(/bytesPerRow|256/);
    expect(
      () => new SeabedHeightField(cap.context, { center: CENTER, gridN: 64 }),
    ).not.toThrow();
  });
});
