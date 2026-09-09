import { describe, expect, it } from 'vitest';
import { EllipsoidGeometry } from '../../src/scene/EllipsoidGeometry';
import { Globe } from '../../src/scene/Globe';
import { TileKey } from '../../src/imagery/TileKey';
import type { ImageryLayer } from '../../src/imagery/ImageryLayer';

const MERC_EDGE = Math.atan(Math.sinh(Math.PI)); // ±85.0511°
const SOUTH_OF_TOP_ROW = Math.atan(Math.sinh(Math.PI / 2)); // 66.56°N

/**
 * 极区采样钳制（polarSampleClampNorth）：
 *
 * 背景：天地图/高德的 L2/L3 顶行瓦片在 ~80.7°N 以北是 (7,6,40) 黑填充
 * （L2 顶行 43~104 行、L3 部分整张黑）——从全球视角看 78–85°N 的黑带
 * 被读作"极区接缝黑线"（第九/十次修复后用户仍可见的残余）。
 *
 * 修复：provider 声明 polarSampleClampNorth（度）后，顶行瓦片的行级 v
 * 钳制到该纬度对应的纹理行（北冰洋白浮冰），黑行不再进入画面；85.05°
 * 以北仍是冰白填充。关键实现点：
 * 1. 行级钳制（每行 max(v, clampV)）——端点线性重映射会让 clamp 与瓦片
 *    远端之间所有纬度的内容错位；行级钳制只影响 clamp 以北的行。
 * 2. 顶行（y=0、level>ROOT）强制渲染 ROOT 纹理——细层顶行自身纹理整体
 *    位于黑填充区，只有 ROOT 顶行横跨 clamp 纬度、含真实内容行。
 * 3. clamp 值变化（切换 provider）时几何缓存失效。
 */
describe('polar sample clamp', () => {
  it('行级钳制：clamp 以北各行 v=clampV，以南保持原映射', () => {
    // 78°N 在 L2 顶行纹理中的 v：mercY 线性 → (π − mercY(78°)) / (π/2)
    const clampV = (Math.PI - Math.asinh(Math.tan(78 * Math.PI / 180))) / (Math.PI / 2);

    const g = EllipsoidGeometry.createRegion(-Math.PI / 2, SOUTH_OF_TOP_ROW, 0, MERC_EDGE, 4, {
      mercatorV: true,
      sampleClampVNorth: clampV,
    });
    const slices = 4;
    const rows = 5;
    const vOf = (row: number) => g.vertices[(row * (slices + 1)) * 8 + 7];

    // 北缘行（clamp 以北）→ 恒等于 clampV（smear 78° 行）
    expect(vOf(0)).toBeCloseTo(clampV, 6);
    // 南缘行（66.56°N）→ 原映射 v=1，不受影响
    expect(vOf(rows - 1)).toBeCloseTo(1, 6);
    // 所有行 v ≥ clampV（黑行区 v<clampV 永不被采样；1e-6 容差吸收
    // createRegion 内 mercY 域钳制与测试直算之间的浮点路径差 ~2e-8）
    for (let row = 0; row < rows; row++) {
      expect(vOf(row)).toBeGreaterThanOrEqual(clampV - 1e-6);
    }
  });

  it('无钳制时顶行北缘 v=0（黑行被采样——旧行为）', () => {
    const g = EllipsoidGeometry.createRegion(-Math.PI / 2, SOUTH_OF_TOP_ROW, 0, MERC_EDGE, 4, {
      mercatorV: true,
    });
    expect(g.vertices[7]).toBeCloseTo(0, 6);
  });

  it('Globe：provider 声明 78° 后顶行几何北缘采样 78° 行，且切换时缓存失效', () => {
    const globe = new Globe({ device: {} } as never);
    const g = globe as unknown as {
      _getTileGeometry(tile: TileKey, levels: number, lift?: unknown): { vertices: Float32Array };
      _applyPolarClamp(layer: ImageryLayer): void;
      _geometryCache: Map<string, unknown>;
    };
    const top = new TileKey(2, 0, 0);
    const vOf = (geom: { vertices: Float32Array }) => geom.vertices[7];

    // 无 clamp：北缘采样 v=0
    expect(vOf(g._getTileGeometry(top, 0))).toBeCloseTo(0, 6);
    expect(g._geometryCache.size).toBe(1);

    // 切换到声明 polarSampleClampNorth=78 的 provider：缓存必须失效
    g._applyPolarClamp({ provider: { polarSampleClampNorth: 78 } } as never);
    expect(g._geometryCache.size).toBe(0);

    const clampV = (Math.PI - Math.asinh(Math.tan(78 * Math.PI / 180))) / (Math.PI / 2);
    expect(vOf(g._getTileGeometry(top, 0))).toBeCloseTo(clampV, 6);

    // 细层顶行（L4，经 ROOT 纹理渲染）：uvRect 子矩形 v∈[0,0.25]，行级
    // 钳制后所有行采 clampV（绝对纹理坐标，可越出子矩形——正是 smear 目的）
    const fine = new TileKey(4, 0, 0);
    expect(vOf(g._getTileGeometry(fine, 2))).toBeCloseTo(clampV, 6);
  });
});
