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

  constructor(options: XyzImageryProviderOptions) {
    this.url          = options.url;
    this.subdomains   = options.subdomains ?? [];
    this.maximumLevel = options.maximumLevel ?? 18;
    this.tileSize     = options.tileSize ?? 256;
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
