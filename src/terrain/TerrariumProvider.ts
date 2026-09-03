/**
 * TerrariumProvider - terrain tiles from the Terrarium PNG elevation
 * service (Mapzen/Tilezen style), Web Mercator (slippy XYZ, north-up).
 *
 * Each PNG pixel encodes elevation meters as `(R * 256 + G + B / 256) -
 * 32768` (oceans are negative; there is no NoData value). The image is
 * decoded into a height grid on the CPU and triangulated into
 * {@link TerrainMeshData} (adjacent pixels form triangle pairs).
 *
 * Not usable in Node test environments (no createImageBitmap/OffscreenCanvas);
 * `requestTileGeometry` surfaces a diagnostic error there — tests should
 * exercise the pure decode helpers or mock the bitmap path.
 */

import {
  HttpTerrainProviderBase,
  TerrainMeshData,
  webmercatorTileBounds,
  ecefFromLonLatHeight,
  ecefMetersFromLonLat,
} from './TerrainProvider';

/**
 * Decode one Terrarium pixel into elevation meters.
 * Exported as a pure function for direct unit testing.
 */
export function decodeTerrariumPixel(r: number, g: number, b: number): number {
  return r * 256 + g + b / 256 - 32768;
}

export interface TerrariumProviderOptions {
  /** URL template. Default: AWS elevation-tiles-prod terrarium bucket. */
  url?: string;
  /** Highest available level. Default: 15 */
  maximumLevel?: number;
}

const DEFAULT_URL =
  'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png';

export class TerrariumProvider extends HttpTerrainProviderBase {
  constructor(options: TerrariumProviderOptions = {}) {
    super({
      url: options.url ?? DEFAULT_URL,
      tilingScheme: 'webmercator',
      maximumLevel: options.maximumLevel ?? 15,
    });
  }

  protected override async decodeTile(
    response: Response,
    z: number,
    x: number,
    y: number
  ): Promise<TerrainMeshData | null> {
    if (
      typeof createImageBitmap !== 'function' ||
      typeof OffscreenCanvas === 'undefined'
    ) {
      throw new Error(
        'TerrariumProvider: createImageBitmap/OffscreenCanvas unavailable ' +
          'in this runtime (Node test environments cannot decode PNG tiles)'
      );
    }

    const blob = await response.blob();
    const bitmap = await createImageBitmap(blob);
    try {
      const size = bitmap.width;
      const canvas = new OffscreenCanvas(size, size);
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error('TerrariumProvider: 2d context unavailable');
      }
      ctx.drawImage(bitmap, 0, 0);
      const pixels = ctx.getImageData(0, 0, size, size).data;

      // Height grid sampled at every pixel (row 0 = north, image space).
      const grid = size;
      const heights = new Float32Array(grid * grid);
      for (let r = 0; r < grid; r++) {
        for (let c = 0; c < grid; c++) {
          const p = (r * grid + c) * 4;
          heights[r * grid + c] = decodeTerrariumPixel(
            pixels[p],
            pixels[p + 1],
            pixels[p + 2]
          );
        }
      }
      return triangulateHeightGrid(heights, grid, z, x, y);
    } finally {
      bitmap.close();
    }
  }
}

/**
 * CPU-triangulate a regular height grid into tile geometry. Adjacent grid
 * points form triangle pairs; heights fold into the normalised ECEF radius.
 * No NoData skipping: oceans carry negative heights and are kept.
 */
function triangulateHeightGrid(
  heights: Float32Array,
  grid: number,
  z: number,
  x: number,
  y: number
): TerrainMeshData {
  const bounds = webmercatorTileBounds(z, x, y);

  const vertexCount = grid * grid;
  const positions = new Float32Array(vertexCount * 3);
  const xyz: [number, number, number] = [0, 0, 0];
  let minHeight = Number.POSITIVE_INFINITY;
  let maxHeight = Number.NEGATIVE_INFINITY;

  for (let r = 0; r < grid; r++) {
    // Row 0 is the image's top row = north edge.
    const lat = bounds.north + (bounds.south - bounds.north) * (r / (grid - 1));
    for (let c = 0; c < grid; c++) {
      const lon = bounds.west + (bounds.east - bounds.west) * (c / (grid - 1));
      const h = heights[r * grid + c];
      if (h < minHeight) minHeight = h;
      if (h > maxHeight) maxHeight = h;

      ecefFromLonLatHeight(lon, lat, h, xyz);
      const i = (r * grid + c) * 3;
      positions[i]     = xyz[0];
      positions[i + 1] = xyz[1];
      positions[i + 2] = xyz[2];
    }
  }

  const quadsPerRow = grid - 1;
  const indices = new Uint32Array(quadsPerRow * quadsPerRow * 6);
  let idx = 0;
  for (let r = 0; r < quadsPerRow; r++) {
    for (let c = 0; c < quadsPerRow; c++) {
      const a = r * grid + c;
      const b = a + grid;
      const d = a + 1;
      const e = b + 1;
      indices[idx++] = a;
      indices[idx++] = b;
      indices[idx++] = d;
      indices[idx++] = b;
      indices[idx++] = e;
      indices[idx++] = d;
    }
  }

  const centerLon = (bounds.west + bounds.east) / 2;
  const centerLat = (bounds.north + bounds.south) / 2;

  return {
    positions,
    indices,
    minHeight,
    maxHeight,
    center: ecefMetersFromLonLat(centerLon, centerLat),
  };
}
