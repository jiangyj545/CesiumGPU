import { describe, expect, it } from 'vitest';
import { GltfLoader } from '../../src/loader/GltfLoader';
import { buildAccessorGlb } from './helpers';

/** 读取 POSITION（VEC3 单顶点）反量化后的三个分量。 */
async function readPosition(glb: ArrayBuffer): Promise<number[]> {
  const model = await GltfLoader.parseGlbBuffer(glb);
  expect(model.primitives).toHaveLength(1);
  const v = model.primitives[0].vertices;
  return [v[0], v[1], v[2]];
}

/** 顶点缓冲为 Float32Array：期望值需经 float32 舍入后再比较。 */
function f32(...xs: number[]): number[] {
  return xs.map((x) => Math.fround(x));
}

describe('accessor.normalized 反量化', () => {
  it('ubyte normalized [0,128,255] → [0, 128/255, 1]', async () => {
    const glb = buildAccessorGlb({
      componentType: 5121,
      componentSize: 1,
      write: (dv, off, val) => dv.setUint8(off, val),
      values: [0, 128, 255],
      type: 'VEC3',
      normalized: true,
    });
    expect(await readPosition(glb)).toEqual(f32(0, 128 / 255, 1));
  });

  it('ushort normalized [0,32767,65535] → [0, 32767/65535, 1]', async () => {
    const glb = buildAccessorGlb({
      componentType: 5123,
      componentSize: 2,
      write: (dv, off, val) => dv.setUint16(off, val, true),
      values: [0, 32767, 65535],
      type: 'VEC3',
      normalized: true,
    });
    expect(await readPosition(glb)).toEqual(f32(0, 32767 / 65535, 1));
  });

  it('byte normalized [-64,127,-128] → [-64/127, 1, -1]（-128 钳到 -1）', async () => {
    const glb = buildAccessorGlb({
      componentType: 5120,
      componentSize: 1,
      write: (dv, off, val) => dv.setInt8(off, val),
      values: [-64, 127, -128],
      type: 'VEC3',
      normalized: true,
    });
    expect(await readPosition(glb)).toEqual(f32(-64 / 127, 1, -1));
  });

  it('short normalized [-16384,32767,0] → [-16384/32767, 1, 0]', async () => {
    const glb = buildAccessorGlb({
      componentType: 5122,
      componentSize: 2,
      write: (dv, off, val) => dv.setInt16(off, val, true),
      values: [-16384, 32767, 0],
      type: 'VEC3',
      normalized: true,
    });
    expect(await readPosition(glb)).toEqual(f32(-16384 / 32767, 1, 0));
  });

  it('float accessor 不受 normalized 影响（原值透传）', async () => {
    const glb = buildAccessorGlb({
      componentType: 5126,
      componentSize: 4,
      write: (dv, off, val) => dv.setFloat32(off, val, true),
      values: [0.25, -2, 7],
      type: 'VEC3',
      normalized: true,
    });
    expect(await readPosition(glb)).toEqual([0.25, -2, 7]);
  });

  it('normalized=false 时 ubyte 保持原样不除', async () => {
    const glb = buildAccessorGlb({
      componentType: 5121,
      componentSize: 1,
      write: (dv, off, val) => dv.setUint8(off, val),
      values: [0, 128, 255],
      type: 'VEC3',
      normalized: false,
    });
    expect(await readPosition(glb)).toEqual([0, 128, 255]);
  });
});
