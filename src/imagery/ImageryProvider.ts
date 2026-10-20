import { TileKey } from './TileKey';
import type { CanvasSurface } from './TopRowBlackFill';

/**
 * ImageryProvider - Source of imagery tiles.
 *
 * A provider knows how to turn a {@link TileKey} into an image. The default
 * XYZ scheme is expressed through a URL template with `{z}`, `{x}`, `{y}`
 * placeholders and an optional `{s}` subdomain placeholder that is rotated
 * per request (see {@link ImageryProvider.resolveUrl}).
 */
export interface ImageryProvider {
  /**
   * URL template. Supported placeholders:
   * - `{z}` tile level
   * - `{x}` tile column
   * - `{y}` tile row
   * - `{s}` subdomain, rotated from {@link ImageryProvider.subdomains}
   */
  readonly url: string;

  /** Highest level for which this provider can serve tiles. */
  readonly maximumLevel: number;

  /** Native tile image size in pixels (256 for standard XYZ services). */
  readonly tileSize: number;

  /** Subdomains available for `{s}` rotation (empty when the template has no `{s}`). */
  readonly subdomains: readonly string[];

  /** Whether the provider is ready to serve tiles (metadata loaded). */
  readonly ready: boolean;

  /**
   * Resolve the concrete request URL for a tile, substituting `{z}/{x}/{y}`
   * and rotating `{s}` deterministically over the tile coordinates.
   */
  resolveUrl(key: TileKey): string;

  /**
   * Fetch one tile. Resolves with the decoded bitmap or a 2D canvas surface
   * (both upload directly via copyExternalImageToTexture), or `null` on any
   * failure (network error, non-2xx status, decode error). Never rejects —
   * callers treat `null` as "skip this tile and render the parent instead".
   *
   * Canvas sources matter for composition paths: Chrome's
   * copyExternalImageToTexture silently produces an all-zero texture for
   * ImageBitmaps re-created from an OffscreenCanvas (verified on main:
   * composite tiles uploaded black while the same bitmap reads back healthy
   * through a 2D canvas) — handing the canvas itself up avoids that path.
   */
  requestTile(key: TileKey): Promise<ImageBitmap | CanvasSurface | null>;
}

/**
 * Rotate the `{s}` placeholder deterministically over the tile coordinates
 * so that consecutive tiles spread across subdomains.
 */
export function rotateSubdomain(
  subdomains: readonly string[],
  key: TileKey
): string {
  if (subdomains.length === 0) return '';
  return subdomains[(key.x + key.y + key.level) % subdomains.length];
}
