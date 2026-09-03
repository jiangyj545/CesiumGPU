import { describe, it, expect, vi } from 'vitest';
import { Camera } from '../../src/scene/Camera';
import { TerrainLayer } from '../../src/terrain/TerrainLayer';
import type { TerrainProvider, TerrainMeshData } from '../../src/terrain/TerrainProvider';
import {
  geographicTileBounds,
  ecefFromLonLatHeight,
} from '../../src/terrain/TerrainProvider';

/**
 * 跨瓦片边缘缝合（crack stitching）单测。
 *
 * 构造 z11 粗瓦片 + z12 细瓦片相邻的合成场景，验证：
 * - 细瓦片边缘顶点被重采样贴合到实际渲染的粗邻居表面
 * - 同层邻居存在时不触发缝合（签名稳定、结果引用复用）
 * - 粗邻居被细瓦片替换后签名变化并重算
 * - 缝合结果与 LRU 缓存原 mesh 分离（sampleHeight 仍读原始数据）
 */

/** WGS84 polar / equatorial ratio（与引擎约定一致）。 */
const POLAR_RATIO = 6356752.3142 / 6378137.0;
/** WGS84 semi-major axis（metres）。 */
const SEMI_MAJOR = 6378137.0;

/** 按键返回预置合成网格的桩 provider。 */
class ScriptedProvider implements TerrainProvider {
  readonly ready = true;
  readonly maximumLevel = 14;
  readonly tilingScheme = 'geographic' as const;
  readonly meshes = new Map<string, TerrainMeshData>();

  async requestTileGeometry(z: number, x: number, y: number): Promise<TerrainMeshData | null> {
    return this.meshes.get(`${z}/${x}/${y}`) ?? null;
  }

  getTileDataAvailable(z: number, x: number, y: number): boolean {
    return this.meshes.has(`${z}/${x}/${y}`);
  }

  destroy(): void { /* nothing */ }
}

/**
 * n×n 规则网格 mesh（无裙边）：顶点经纬度均匀覆盖瓦片，高度由
 * heightFn(lonDeg, latDeg) 给出，位置按引擎约定（归一化 ECEF，高度折入模长）。
 */
function gridMesh(
  z: number,
  x: number,
  y: number,
  heightFn: (lon: number, lat: number) => number,
  n: number
): TerrainMeshData {
  const b = geographicTileBounds(z, x, y);
  const positions = new Float32Array(n * n * 3);
  const tmp: [number, number, number] = [0, 0, 0];
  let minH = Infinity;
  let maxH = -Infinity;
  for (let iy = 0; iy < n; iy++) {
    for (let ix = 0; ix < n; ix++) {
      const lon = b.west + (ix / (n - 1)) * (b.east - b.west);
      const lat = b.south + (iy / (n - 1)) * (b.north - b.south);
      const h = heightFn(lon, lat);
      ecefFromLonLatHeight(lon, lat, h, tmp);
      const i = (iy * n + ix) * 3;
      positions[i] = tmp[0];
      positions[i + 1] = tmp[1];
      positions[i + 2] = tmp[2];
      minH = Math.min(minH, h);
      maxH = Math.max(maxH, h);
    }
  }
  const indices = new Uint32Array((n - 1) * (n - 1) * 6);
  let o = 0;
  for (let iy = 0; iy < n - 1; iy++) {
    for (let ix = 0; ix < n - 1; ix++) {
      const r0 = iy * n;
      const r1 = (iy + 1) * n;
      indices[o++] = r0 + ix;
      indices[o++] = r1 + ix;
      indices[o++] = r0 + ix + 1;
      indices[o++] = r0 + ix + 1;
      indices[o++] = r1 + ix;
      indices[o++] = r1 + ix + 1;
    }
  }
  return { positions, indices, minHeight: minH, maxHeight: maxH, center: [0, 0, 0] };
}

/** 从归一化 ECEF 位置反推高度（米）——复刻引擎「高度折入模长」约定。 */
function heightOf(positions: Float32Array, vertex: number): number {
  const i = vertex * 3;
  const x = positions[i];
  const y = positions[i + 1] / POLAR_RATIO;
  const z = positions[i + 2] / POLAR_RATIO;
  return (Math.hypot(x, y, z) - 1) * SEMI_MAJOR;
}

// ── 场景 ─────────────────────────────────────────────────────────────────────
// COARSE = (11,2600,1024)，其东邻 (11,2601,1024) 的西南子瓦片 = FINE。
// FINE 的西边与 COARSE 的东边共享边界（干净拼接），东/北边落在
// (11,2601,1024) 粗网格内部（父回退重叠路径）。
const COARSE = { z: 11, x: 2600, y: 1024 };
const FINE   = { z: 12, x: 5202, y: 2048 };
const EAST_SIB = { z: 11, x: 2601, y: 1024 }; // FINE 所在粗瓦片

/** 粗瓦片表面（平滑地形）。 */
const hCoarse = (lon: number, lat: number): number =>
  1000 + 500 * Math.sin(3 * lon) * Math.cos(2 * lat);
/** 细瓦片表面 = 粗表面 + 400 m 细节偏移（边界处天然错开 → T-junction 缝）。 */
const hFine = (lon: number, lat: number): number => hCoarse(lon, lat) + 400;

const COARSE_N = 17;
const FINE_N = 9;

function key(t: { z: number; x: number; y: number }): string {
  return `${t.z}/${t.x}/${t.y}`;
}

function findTile(layer: TerrainLayer, t: { z: number; x: number; y: number }) {
  return layer.renderTiles.find(r => r.z === t.z && r.x === t.x && r.y === t.y);
}

/** 建层 + 相机俯视 COARSE/FINE 共享边界，两阶段「飞入」泵到缝合场景就位。 */
async function makeLayer(withWestSameLevel = false) {
  const provider = new ScriptedProvider();
  provider.meshes.set(key(COARSE), gridMesh(COARSE.z, COARSE.x, COARSE.y, hCoarse, COARSE_N));
  provider.meshes.set(key(EAST_SIB), gridMesh(EAST_SIB.z, EAST_SIB.x, EAST_SIB.y, hCoarse, COARSE_N));
  provider.meshes.set(key(FINE), gridMesh(FINE.z, FINE.x, FINE.y, hFine, FINE_N));
  if (withWestSameLevel) {
    // FINE 的西邻同层瓦片（COARSE 的子瓦片）——命中后西边不再缝合
    provider.meshes.set('12/5201/2048', gridMesh(12, 5201, 2048, hFine, FINE_N));
  }

  const layer = new TerrainLayer({ provider, maximumScreenSpaceError: 2.0 });

  const coarseBounds = geographicTileBounds(COARSE.z, COARSE.x, COARSE.y);
  const fineBounds = geographicTileBounds(FINE.z, FINE.x, FINE.y);
  const lon = (coarseBounds.east + fineBounds.west) / 2; // 共享边界
  const lat = (fineBounds.south + fineBounds.north) / 2;
  const camera = new Camera();

  const update = (): void => layer.update(camera, 1080);

  // 阶段 1（远，归一化高度 0.015）：SSE 在 z11 停止 → 粗瓦片被选中，
  // 其 mesh 进入 LRU（引擎只对选中瓦片发起请求，与真实流式顺序一致）。
  camera.setView({ destination: { lon, lat, height: 0.015 * SEMI_MAJOR } });
  await vi.waitFor(() => {
    update();
    expect(layer.getMesh(COARSE.z, COARSE.x, COARSE.y)).not.toBeNull();
    expect(layer.getMesh(EAST_SIB.z, EAST_SIB.x, EAST_SIB.y)).not.toBeNull();
  });

  // 阶段 2（近，归一化高度 0.008）：SSE 在 z11 细化 / z12 停止 →
  // 粗细瓦片相邻：无 mesh 的 z12 子瓦片回退渲染粗瓦片 → 缝合场景成立。
  camera.setView({ destination: { lon, lat, height: 0.008 * SEMI_MAJOR } });
  await vi.waitFor(() => {
    update();
    if (withWestSameLevel) {
      // 西侧同层瓦片的 mesh 必须已就位，西边才解析为同层
      //（就位前该边会先回退解析到粗瓦片）
      expect(layer.getMesh(12, 5201, 2048)).not.toBeNull();
    }
    const fine = findTile(layer, FINE);
    expect(fine).toBeDefined();
    // 东/北侧始终引用 EAST_SIB 粗回退；西边随变体而定
    expect(fine!.stitchSignature).toContain(key(EAST_SIB));
    expect(fine!.stitchSignature).toMatch(withWestSameLevel ? /^-/ : key(COARSE));
  });

  return { layer, provider, camera, update };
}

describe('terrain 边缘缝合（crack stitching）', () => {
  it('细瓦片边缘顶点贴合到粗邻居表面，内部保持原高度', async () => {
    const { layer } = await makeLayer();

    const fine = findTile(layer, FINE)!;
    expect(fine.stitchedPositions).not.toBeNull();
    expect(fine.stitchSignature).toContain(key(COARSE));

    const stitched = fine.stitchedPositions!;
    const b = geographicTileBounds(FINE.z, FINE.x, FINE.y);
    let edgeChecked = 0;
    let interiorChecked = 0;
    for (let iy = 0; iy < FINE_N; iy++) {
      for (let ix = 0; ix < FINE_N; ix++) {
        const lon = b.west + (ix / (FINE_N - 1)) * (b.east - b.west);
        const lat = b.south + (iy / (FINE_N - 1)) * (b.north - b.south);
        const h = heightOf(stitched, iy * FINE_N + ix);
        if (ix === 0) {
          // 西边缘：被贴到粗瓦片表面（而非细表面的 +400 m）
          expect(Math.abs(h - hCoarse(lon, lat))).toBeLessThan(2.0);
          expect(Math.abs(h - hFine(lon, lat))).toBeGreaterThan(100);
          edgeChecked++;
        } else if (ix === 4 && iy === 4) {
          // 深内部：保持细网格原高度（float32 位置量化误差 < 0.5 m）
          expect(Math.abs(h - hFine(lon, lat))).toBeLessThan(1.0);
          interiorChecked++;
        }
      }
    }
    expect(edgeChecked).toBe(FINE_N);
    expect(interiorChecked).toBe(1);

    // 粗瓦片自身四周均无更粗的渲染邻居 → 不缝合
    const coarse = findTile(layer, COARSE)!;
    expect(coarse.stitchedPositions).toBeNull();
  });

  it('同层邻居存在时西侧不缝合，且签名稳定、缝合结果引用复用（不重算）', async () => {
    const { layer, update } = await makeLayer(true);

    const fine = findTile(layer, FINE)!;
    expect(fine.stitchedPositions).not.toBeNull();
    // 西边解析到同层瓦片 → 签名不含粗瓦片
    expect(fine.stitchSignature).not.toContain(key(COARSE));

    const b = geographicTileBounds(FINE.z, FINE.x, FINE.y);
    for (let iy = 0; iy < FINE_N - 1; iy++) {
      // 排除北缘角点：它同时落在北侧（EAST_SIB 粗回退）的缝合线上，
      // 由北侧贴合是正确行为；此处只验证西侧解析为同层后不被缝合。
      const lon = b.west;
      const lat = b.south + (iy / (FINE_N - 1)) * (b.north - b.south);
      const h = heightOf(fine.stitchedPositions!, iy * FINE_N);
      expect(Math.abs(h - hFine(lon, lat))).toBeLessThan(1.0);
    }

    // 连续两帧：签名相同 + 位置引用相同 → 缝合未重算
    const sig1 = fine.stitchSignature;
    const pos1 = fine.stitchedPositions;
    update();
    const fine2 = findTile(layer, FINE)!;
    expect(fine2.stitchSignature).toBe(sig1);
    expect(fine2.stitchedPositions).toBe(pos1);
  });

  it('渲染邻居被换出（LOD 细化后由细瓦片接管）后签名变化并重算（西边回到细网格高度）', async () => {
    const { layer, provider, camera } = await makeLayer();

    const sig1 = findTile(layer, FINE)!.stitchSignature;
    expect(sig1).toContain(key(COARSE)); // 西侧引用粗瓦片（父回退路径）

    // 预置 COARSE 子树全部 16 片 z13 网格。阶段 A（z12 即为叶）不会探测
    // z13，因此不污染失败缓存；随后提高视口高度（1080 → 1650）使 SSE
    // 细化下探到 z13：西侧由细瓦片接管渲染 → COARSE 不再被任何回退
    // 消费 → 换出渲染集 → FINE 西侧无渲染邻居。
    for (const px of [5200, 5201]) {
      for (const py of [2048, 2049]) {
        for (const dx of [0, 1]) {
          for (const dy of [0, 1]) {
            const x = px * 2 + dx;
            const y = py * 2 + dy;
            provider.meshes.set(`13/${x}/${y}`, gridMesh(13, x, y, hFine, FINE_N));
          }
        }
      }
    }
    await vi.waitFor(() => {
      // 略微抬升相机（缩小视场足迹，保证 z13 叶片数 < 512 上限）并提高
      // 视口高度：sse12 > 2 → z12 细化、sse13 ≤ 2 → z13 为叶
      camera.setView({
        destination: {
          lon: geographicTileBounds(COARSE.z, COARSE.x, COARSE.y).east,
          lat: (geographicTileBounds(FINE.z, FINE.x, FINE.y).south +
            geographicTileBounds(FINE.z, FINE.x, FINE.y).north) / 2,
          height: 0.006 * SEMI_MAJOR,
        },
      });
      layer.update(camera, 2100);
      expect(findTile(layer, FINE)!.stitchSignature).not.toBe(sig1); // 签名变化 → 重算
    });

    const fine2 = findTile(layer, FINE)!; // FINE 仍作为 z13 的回退祖先渲染
    expect(fine2.stitchSignature).not.toContain(key(COARSE));

    const b = geographicTileBounds(FINE.z, FINE.x, FINE.y);
    for (let iy = 0; iy < FINE_N - 1; iy++) {
      // 排除北缘角点（西北角同时落在北侧缝合线上，被北侧贴合是正确行为）
      const lon = b.west;
      const lat = b.south + (iy / (FINE_N - 1)) * (b.north - b.south);
      const h = heightOf(fine2.stitchedPositions!, iy * FINE_N);
      // 西边缘恢复为细网格自身高度（不再贴合已换出的粗表面）
      expect(Math.abs(h - hFine(lon, lat))).toBeLessThan(1.0);
    }
  });

  it('缝合快照与 LRU 原 mesh 分离；sampleHeight 仍读原始数据', async () => {
    const { layer } = await makeLayer();

    const fine = findTile(layer, FINE)!;
    const stitched = fine.stitchedPositions!;
    const original = layer.getMesh(FINE.z, FINE.x, FINE.y)!.positions;
    expect(stitched).not.toBe(original); // 独立副本，非原 positions

    const before = Array.from(original.subarray(0, 3));
    stitched[0] = 9.9; // 改快照不得影响缓存原数据
    expect(Array.from(layer.getMesh(FINE.z, FINE.x, FINE.y)!.positions.subarray(0, 3)))
      .toEqual(before);

    // sampleHeight 取最细覆盖瓦片（FINE 原始 mesh）→ 未被缝合污染
    const b = geographicTileBounds(FINE.z, FINE.x, FINE.y);
    const midLon = (b.west + b.east) / 2;
    const midLat = (b.south + b.north) / 2;
    expect(Math.abs(layer.sampleHeight(midLon, midLat) - hFine(midLon, midLat)))
      .toBeLessThan(1.0);
  });
});
