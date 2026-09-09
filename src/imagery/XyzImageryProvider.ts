import { TileKey } from './TileKey';
import { ImageryProvider, rotateSubdomain } from './ImageryProvider';

export interface XyzImageryProviderOptions {
  /** URL template with `{z}/{x}/{y}` and optional `{s}` placeholders. */
  url: string;
  /** Subdomains for `{s}` rotation. Default: `[]` (no rotation). */
  subdomains?: string[];
  /** Highest available level. Default: 18 */
  maximumLevel?: number;
  /** Native tile size in pixels. Default: 256 */
  tileSize?: number;
  /** See {@link ImageryProvider.polarSampleClampNorth} (degrees). */
  polarSampleClampNorth?: number;
  /** See {@link ImageryProvider.polarSampleClampSouth} (degrees). */
  polarSampleClampSouth?: number;
}

/**
 * XyzImageryProvider - Standard XYZ tile service (OSM-style).
 *
 * Fetches each tile with `fetch` → `blob` → `createImageBitmap`. Any failure
 * resolves to `null` (never rejects), so the caller can skip the tile and
 * keep rendering with a parent tile instead.
 */
export class XyzImageryProvider implements ImageryProvider {
  readonly url: string;
  readonly maximumLevel: number;
  readonly tileSize: number;
  readonly subdomains: readonly string[];
  readonly polarSampleClampNorth?: number;
  readonly polarSampleClampSouth?: number;

  constructor(options: XyzImageryProviderOptions) {
    this.url          = options.url;
    this.subdomains   = options.subdomains ?? [];
    this.maximumLevel = options.maximumLevel ?? 18;
    this.tileSize     = options.tileSize ?? 256;
    this.polarSampleClampNorth = options.polarSampleClampNorth;
    this.polarSampleClampSouth = options.polarSampleClampSouth;
  }

  /** Built-in Open Street Map raster tile preset. */
  static osm(): XyzImageryProvider {
    return new XyzImageryProvider({
      // {s} is a subdomain (DNS-level CNAME), NOT a path segment.
      url:          'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      subdomains:   ['a', 'b', 'c'],
      maximumLevel: 19,
      tileSize:     256,
    });
  }

  /**
   * Built-in 天地图 (TianDiTu) raster tile preset.
   *
   * `layer` picks the map style: 'img' 卫星影像 | 'vec' 矢量底图 | 'cia' 影像注记
   * | 'cva' 矢量注记. Tiles use the Web-Mercator (w) tile matrix set via the
   * DataServer endpoint, which serves `image/*` payloads with CORS enabled.
   */
  static tianditu(tk: string, layer: 'img' | 'vec' | 'cia' | 'cva' = 'img'): XyzImageryProvider {
    return new XyzImageryProvider({
      // DataServer 形式：浏览器端 key（服务端 IP 校验会拒绝，必须浏览器 Referer/UA）
      url:          `https://t{s}.tianditu.gov.cn/DataServer?T=${layer}_w&x={x}&y={y}&l={z}&tk=${tk}`,
      subdomains:   ['0', '1', '2', '3', '4', '5', '6', '7'],
      maximumLevel: 18,
      tileSize:     256,
      // 天地图影像在 ~80.7°N 以北（L2 顶行顶部 43–104 行、L3 部分整张）是
      // (7,6,40) 黑填充——全球视角下 78–85°N 黑带被读作极区接缝环。顶行
      // 瓦片从 78°N 行开始采样（北冰洋浮冰，白），黑行不再进入画面。
      polarSampleClampNorth: 78,
    });
  }

  get ready(): boolean {
    // XYZ services need no metadata negotiation — ready immediately.
    return true;
  }

  /** Substitute `{z}/{x}/{y}/{s}` into the URL template. */
  resolveUrl(key: TileKey): string {
    let url = this.url
      .replace('{z}', String(key.level))
      .replace('{x}', String(key.x))
      .replace('{y}', String(key.y));
    if (this.subdomains.length > 0) {
      url = url.replace('{s}', rotateSubdomain(this.subdomains, key));
    }
    return url;
  }

  /**
   * Fetch one tile. Resolves `null` on network error, non-2xx status or
   * decode failure (errors are logged once per tile, never thrown).
   */
  async requestTile(key: TileKey): Promise<ImageBitmap | null> {
    try {
      const response = await fetch(this.resolveUrl(key), { mode: 'cors' });
      if (!response.ok) {
        console.warn(
          `XyzImageryProvider: tile ${key.key} HTTP ${response.status}`
        );
        return null;
      }
      const blob = await response.blob();
      return await createImageBitmap(blob);
    } catch (error) {
      console.warn(`XyzImageryProvider: tile ${key.key} failed`, error);
      return null;
    }
  }
}

export default XyzImageryProvider;
