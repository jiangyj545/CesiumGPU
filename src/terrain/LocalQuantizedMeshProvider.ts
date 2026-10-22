/**
 * LocalQuantizedMeshProvider - terrain tiles from the local quantized-mesh
 * service (little-endian variant, see QuantizedMeshDecoder).
 *
 * Tiling scheme is geographic EPSG:4326 TMS-style (south-up, z0 = 2×1).
 */

import { HttpTerrainProviderBase, TerrainMeshData } from './TerrainProvider';
import { decompressGzip, decodeQuantizedMesh } from './QuantizedMeshDecoder';

export interface LocalQuantizedMeshProviderOptions {
  /** URL template. Default: local terrain_hubei service. */
  url?: string;
  /** Highest available level. Default: 14 */
  maximumLevel?: number;
  /** Auth headers for every tile request (P1-F，如 Authorization: Bearer）。 */
  headers?: Record<string, string>;
}

const DEFAULT_URL =
  'http://localhost:8888/api/v1/terrain-3d/serve/terrain_hubei/{z}/{x}/{y}.terrain';

export class LocalQuantizedMeshProvider extends HttpTerrainProviderBase {
  constructor(options: LocalQuantizedMeshProviderOptions = {}) {
    super({
      url: options.url ?? DEFAULT_URL,
      tilingScheme: 'geographic',
      maximumLevel: options.maximumLevel ?? 12, // 7777 terrain_hubei 构建范围 max_zoom=12(z13+ 429 限流)
      headers: options.headers,
    });
  }

  protected override async decodeTile(
    response: Response,
    z: number,
    x: number,
    y: number
  ): Promise<TerrainMeshData | null> {
    const compressed = new Uint8Array(await response.arrayBuffer());
    const raw = await decompressGzip(compressed);
    return decodeQuantizedMesh(raw, z, x, y);
  }
}
