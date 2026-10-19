import { TerrainLayer, MAX_PIN_LEVEL } from './TerrainLayer';
import { Cartographic } from '../math/Cartographic';

/**
 * Sample terrain heights for the given positions at the provider's most
 * detailed level, mirroring Cesium's `sampleTerrainMostDetailed` (hexapod
 * measurement.js / waypointEditor.js call shape).
 *
 * The bounding rectangle of `positions` is preloaded via
 * `TerrainLayer.ensureRegion` at the provider's `maximumLevel` (clamped
 * internally by the layer), then every position's height is read back with
 * `TerrainLayer.sampleHeight`. Positions without coverage resolve to
 * height `0` (the engine's "clamp to ellipsoid" convention — same
 * short-circuit as waypointEditor).
 *
 * @returns a **new** array of Cartographics (input is never mutated);
 *   `height` is in metres.
 */
export async function sampleTerrainMostDetailed(
  layer: TerrainLayer,
  positions: Cartographic[]
): Promise<Cartographic[]> {
  return sampleTerrain(layer, positions, layer.provider.maximumLevel);
}

/**
 * Level-pinned variant (Cesium `sampleTerrain`): preload the covering tiles
 * at `level` and read back the heights. `level` is clamped to the layer /
 * provider maximum by `ensureRegion`.
 */
export async function sampleTerrain(
  layer: TerrainLayer,
  positions: Cartographic[],
  level: number
): Promise<Cartographic[]> {
  if (positions.length === 0) return [];

  // Cartographic 存弧度；ensureRegion / sampleHeight 以度为入参。
  const RAD2DEG = 180 / Math.PI;
  let west = Infinity, east = -Infinity, south = Infinity, north = -Infinity;
  for (const p of positions) {
    const lon = p.longitude * RAD2DEG;
    const lat = p.latitude * RAD2DEG;
    if (lon < west) west = lon;
    if (lon > east) east = lon;
    if (lat < south) south = lat;
    if (lat > north) north = lat;
  }
  const bounds = { west, south, east, north };

  // Cesium sampleTerrainMostDetailed 语义：结果不受缓存压力影响。pin 必须在
  // 预载 await **之前**生效——预载期间渲染流（更精细瓦片）就会把已到手的
  // 采样瓦片从 LRU 挤出，ensureRegion 结束时它们已不在缓存，回读只得 0。
  // pin 层级与 ensureRegion 的钳制结果对齐，再封顶 MAX_PIN_LEVEL（pin 只允
  // 许粗层）：>L10 时钉住包络的 L10 祖先作兜底——即便更精细采样瓦片被逐出，
  // sampleHeight 也回退到祖先高程而非 0；pin 总数超限时退化为现状并
  // console.info 一次。
  const pinLevel = Math.min(layer.effectiveLevel(level), MAX_PIN_LEVEL);
  let pinned = false;
  try {
    layer.pinRegion(bounds, pinLevel);
    pinned = true;
  } catch {
    if (!pinDegradationNotified) {
      console.info(
        `sampleTerrain: 包络在 L${pinLevel} 反算的 pin 瓦片数超限，退化为不 pin（采样瓦片可能被 LRU 驱逐）`
      );
      pinDegradationNotified = true;
    }
  }

  await layer.ensureRegion(bounds, level);

  // 高度单位：米（Cesium sampleTerrain 语义）。注意与 Cartographic.height
  // 的归一化约定（1 = EARTH_SCALE 米）不同——Cesium API 兼容层刻意保留米，
  // 调用方（示例/测量面板）直接以米显示；如需归一化请自行 ÷EARTH_SCALE。
  try {
    return positions.map((p) => {
      const lon = p.longitude * RAD2DEG;
      const lat = p.latitude * RAD2DEG;
      return new Cartographic(p.longitude, p.latitude, layer.sampleHeight(lon, lat, level));
    });
  } finally {
    if (pinned) layer.unpinRegion(bounds, pinLevel);
  }
}

/** 进程级“pin 退化已提示”标记（只 console.info 一次）。 */
let pinDegradationNotified = false;
