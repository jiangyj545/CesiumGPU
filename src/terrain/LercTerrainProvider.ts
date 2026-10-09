import {
  HttpTerrainProviderBase,
  webmercatorTileBounds,
  type TerrainMeshData,
  type TerrainTilingScheme,
} from './TerrainProvider';
import { triangulateHeightGrid } from './TerrariumProvider';
import { decodeLerc1 } from './LercDecoder';

export interface LercTerrainProviderOptions {
  /**
   * URL template of an ArcGIS ImageServer LERC tile endpoint. Default: the
   * ArcGIS World Elevation (Terrain3D) service
   * `…/tile/{z}/{y}/{x}`（注意 y/x 顺序与一般 XYZ 模板相反，对齐
   * mapController.js 的 ArcGISTiledElevationTerrainProvider 用法）。
   */
  url?: string;
  /** Highest available level. Default: 15 */
  maximumLevel?: number;
  /** Auth headers for every tile request (P1-F 模式）。 */
  headers?: Record<string, string>;
}

const DEFAULT_URL =
  'https://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer/tile/{z}/{y}/{x}';


/**
 * LercTerrainProvider – ArcGIS Terrain3D（LERC1，WebMercator 金字塔）地形
 * provider（P2-8，对齐 mapController.js `ArcGISTiledElevationTerrainProvider`）。
 *
 * - 瓦片方案与 TerrariumProvider 同一适配模式：webmercatorTileBounds +
 *   `triangulateHeightGrid`（规则高程网格 → 三角化 → 归一化 ECEF）。
 * - 解码：`decodeLerc1`（"CntZImage"），无效像素（掩码洞）取 0（与海洋
 *   负高程保留策略一致——洞不参与 min/max 统计）。
 * - headers 透传（P1-F）；maximumLevel 默认 15（该服务层级）。
 */
export class LercTerrainProvider extends HttpTerrainProviderBase {
  constructor(options: LercTerrainProviderOptions = {}) {
    super({
      url: options.url ?? DEFAULT_URL,
      tilingScheme: 'webmercator' as TerrainTilingScheme,
      maximumLevel: options.maximumLevel ?? 15,
      headers: options.headers,
    });
  }

  protected override async decodeTile(
    response: Response,
    z: number,
    x: number,
    y: number
  ): Promise<TerrainMeshData | null> {
    const buffer = await response.arrayBuffer();
    const tile = decodeLerc1(buffer);

    // 无效像素 → 0（解码器已置 0 且 mask=0；有效统计见 Lerc1Tile.min/max）。
    const heights = tile.pixels;
    return triangulateHeightGrid(heights, tile.width, z, x, y);
  }
}

/** 某经纬度（度）落在哪一层 webmercator 瓦片（测试 / 文档用）。 */
export function webmercatorTileOfLonLat(
  lonDeg: number,
  latDeg: number,
  level: number
): { z: number; x: number; y: number } {
  const n = 2 ** level;
  const clampedLat = Math.max(-85.05112878, Math.min(85.05112878, latDeg));
  const latRad = (clampedLat * Math.PI) / 180;
  const x = Math.max(0, Math.min(n - 1, Math.floor(((lonDeg + 180) / 360) * n)));
  const y = Math.max(
    0,
    Math.min(
      n - 1,
      Math.floor(((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * n)
    )
  );
  return { z: level, x, y };
}

/** 供测试断言默认 URL 模板（含 y/x 顺序）。 */
export const DEFAULT_LERC_URL = DEFAULT_URL;

/** webmercator 瓦片边界透传（测试用）。 */
export function lercTileBounds(z: number, x: number, y: number) {
  return webmercatorTileBounds(z, x, y);
}
