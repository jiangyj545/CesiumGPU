import type { TerrainLayer } from './TerrainLayer';
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

  let west = Infinity, east = -Infinity, south = Infinity, north = -Infinity;
  for (const p of positions) {
    if (p.longitude < west) west = p.longitude;
    if (p.longitude > east) east = p.longitude;
    if (p.latitude < south) south = p.latitude;
    if (p.latitude > north) north = p.latitude;
  }
  await layer.ensureRegion({ west, south, east, north }, level);

  return positions.map(
    (p) => new Cartographic(p.longitude, p.latitude, layer.sampleHeight(p.longitude, p.latitude, level))
  );
}
