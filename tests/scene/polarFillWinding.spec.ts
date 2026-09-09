import { describe, expect, it } from 'vitest';
import { Globe } from '../../src/scene/Globe';

/**
 * 极冠填充几何 winding 回归测试。
 *
 * 背景 bug（2026-09-09 南极俯视视觉验证发现）：`_buildPolarFillGeometry`
 * 南北两帽曾共用同一份索引模式，但行序方向相反——北极 pole→edge 是纬度
 * 递减，与 createRegion 的 north→south 行序同向（正面朝外）；南极
 * pole→edge 是纬度递增，与 createRegion 反向 → 同一索引模式在南极全部
 * front/back 翻转，被 Primitive 默认管线（frontFace:'ccw' +
 * cullMode:'back'）整体剔除 → 南极帽渲染成洞、极点直接露出太空背景
 * (0,0,13)，而北极帽一切正常。
 *
 * 修复：南极分支翻转索引 winding。本测试用纯几何判据锁定行为：每个
 * 非退化三角形的有向面法向必须与顶点椭球外法向同向（点积 > 0）——
 * 两个极帽、全部 ROOT 列、含裙边行。
 */
describe('Globe polar fill winding', () => {
  const build = (west: number, east: number, isNorth: boolean) => {
    const globe = new Globe({ device: {} } as never);
    const g = globe as unknown as {
      _buildPolarFillGeometry(w: number, e: number, n: boolean): {
        vertices: Float32Array;
        indices: Uint32Array;
      };
    };
    return g._buildPolarFillGeometry(west, east, isNorth);
  };

  /** 所有非退化三角形面法向必须与顶点椭球外法向同向（朝外=可正面渲染）。 */
  const expectOutward = (geometry: { vertices: Float32Array; indices: Uint32Array }, label: string) => {
    const { vertices, indices } = geometry;
    const stride = 8;
    let checked = 0;
    for (let t = 0; t < indices.length; t += 3) {
      const i0 = indices[t], i1 = indices[t + 1], i2 = indices[t + 2];
      const p0 = [vertices[i0*stride], vertices[i0*stride+1], vertices[i0*stride+2]];
      const p1 = [vertices[i1*stride], vertices[i1*stride+1], vertices[i1*stride+2]];
      const p2 = [vertices[i2*stride], vertices[i2*stride+1], vertices[i2*stride+2]];
      const e1 = p1.map((v, i) => v - p0[i]);
      const e2 = p2.map((v, i) => v - p0[i]);
      const fn = [
        e1[1]*e2[2] - e1[2]*e2[1],
        e1[2]*e2[0] - e1[0]*e2[2],
        e1[0]*e2[1] - e1[1]*e2[0],
      ];
      const len = Math.hypot(fn[0], fn[1], fn[2]);
      if (len < 1e-9) continue; // 退化三角形（极点行收缩到一点），不产生片元
      const n0 = [vertices[i0*stride+3], vertices[i0*stride+4], vertices[i0*stride+5]];
      const dot = (fn[0]*n0[0] + fn[1]*n0[1] + fn[2]*n0[2]) / len;
      expect(dot, `${label} triangle #${t/3} faces inward (dot=${dot})`).toBeGreaterThan(0);
      checked++;
    }
    expect(checked, label).toBeGreaterThan(0);
  };

  it('北极填充：全部三角形正面朝外', () => {
    for (let x = 0; x < 4; x++) {
      const west = -Math.PI + (x / 4) * 2 * Math.PI;
      expectOutward(build(west, west + Math.PI / 2, true), `north col ${x}`);
    }
  });

  it('南极填充：全部三角形正面朝外（回归：曾被 backface cull 整体剔除成洞）', () => {
    for (let x = 0; x < 4; x++) {
      const west = -Math.PI + (x / 4) * 2 * Math.PI;
      expectOutward(build(west, west + Math.PI / 2, false), `south col ${x}`);
    }
  });
});
