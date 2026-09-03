/**
 * QuantizedMeshDecoder - decoder for the little-endian quantized-mesh
 * variant served by the local terrain service (verified byte-by-byte
 * against real tiles, e.g. the Hubei dataset).
 *
 * Binary layout after gzip decompression (ALL little-endian):
 *
 * ```
 * +0   centerX, centerY, centerZ   3 x float64   (tile centre, ECEF meters)
 * +24  horizonOcclusionX/Y         2 x float32   (variant: f32, skipped)
 * +32  centerX/Y/Z repeated        3 x float64   (variant duplicate header,
 *                                                skipped)
 * +56  maxHeight, minHeight        2 x float64   (elevation range;
 *      real tiles carry ~3373.004 / -0.27 here; a float32 fallback pair
 *      at +56/+60 exists for tiles that use the narrower encoding)
 * +64  misc                        3 x float64   (bounding sphere prefix,
 *                                                skipped)
 * +88  vertexCount                 uint32
 * +92  zigzag-delta encoded        vertexCount x uint16 per channel,
 *      u / v / h                   in that order
 *      then indexCount uint32 + watermark uint32 + indexCount x uint16
 *      (if watermark != 65536 every index value must be increased by it;
 *      trailing edge indices / extensions e.g. octvertexnormals are skipped)
 * ```
 *
 * v direction (IMPORTANT): this service uses SOUTH-UP TMS tiles
 * (geographic EPSG:4326, z0 = 2x1, y counted from the south pole).
 * Verified with tile z12/x6553/y2707: its centre must be lon ~108°,
 * lat ~+28.99° (Hubei border) with `lat = (y + 0.5)/2^z * 180 - 90`;
 * the north-up interpretation would yield lat ~-28.99° (wrong hemisphere).
 * Therefore the in-tile fraction maps directly: lat = south + (north -
 * south) * (v / 32767) — v = 0 is the SOUTH edge, v = 32767 the north edge.
 *
 * Heights: h / 32767 * (maxHeight - minHeight) + minHeight.
 *
 * ECEF: unit-ellipsoid direction (same convention as
 * `EllipsoidGeometry.createRegion`, y/z squashed by the WGS84 polar ratio),
 * scaled radially by `1 + height / 6378137`.
 *
 * Skirts: every edge vertex (u or v at 0/32767) is duplicated at
 * `pos * (1 - skirtDepth / 6378137)` with
 * `skirtDepth = max(50, maxHeight - minHeight)` meters; adjacent edge
 * vertices along each of the four tile edges are connected with skirt
 * triangles (corners participate in two edges — harmless overlap).
 */

import {
  TerrainMeshData,
  geographicTileBounds,
  ecefFromLonLatHeight,
  MIN_SKIRT_DEPTH,
} from './TerrainProvider';

/** Header field offsets (bytes) of the little-endian variant. */
const OFFSET = {
  CENTER: 0,          // 3 x f64
  VERTEX_COUNT: 88,   // u32
  VERTICES: 92,       // u16 zigzag-delta: u[vcount] v[vcount] h[vcount]
} as const;

/** Quantized u/v/h full-scale value. */
const QUANT_MAX = 32767;
/** Watermark stored before the index block; 65536 means "no HiLo offset". */
const INDEX_WATERMARK = 65536;
/** WGS84 semi-major axis (meters). */
const WGS84_SEMI_MAJOR = 6378137.0;

/**
 * Decode one zigzag-encoded value: `(enc >>> 1) ^ -(enc & 1)`.
 * Exported for round-trip testing.
 */
export function zigzagDecode(encoded: number): number {
  return (encoded >>> 1) ^ -(encoded & 1);
}

/**
 * Decompress gzip bytes via the native `DecompressionStream` (available in
 * browsers and Node 18+).
 */
export async function decompressGzip(bytes: Uint8Array): Promise<Uint8Array> {
  const Ctor = (
    globalThis as unknown as {
      DecompressionStream?: new (
        format: string
      ) => TransformStream<Uint8Array, Uint8Array>;
    }
  ).DecompressionStream;
  if (!Ctor) {
    throw new Error(
      'decompressGzip: DecompressionStream is not available in this runtime'
    );
  }
  const stream = new Blob([bytes as BlobPart])
    .stream()
    .pipeThrough(new Ctor('gzip'));
  const buffer = await new Response(stream).arrayBuffer();
  return new Uint8Array(buffer);
}

/**
 * Decode a quantized-mesh tile (raw, already gzip-decompressed bytes) into
 * triangulated geometry with skirts.
 *
 * @param data tile bytes without the gzip wrapper
 * @param z    tile level (geographic TMS-style)
 * @param x    tile column
 * @param y    tile row (south-up TMS)
 */
export function decodeQuantizedMesh(
  data: ArrayBuffer | Uint8Array,
  z: number,
  x: number,
  y: number
): TerrainMeshData {
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
  const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);

  const heights = readHeightRange(view);
  const skirtDepth = Math.max(MIN_SKIRT_DEPTH, heights.max - heights.min);

  // --- vertices (zigzag-delta u/v/h) --------------------------------------
  const vertexCount = view.getUint32(OFFSET.VERTEX_COUNT, true);
  let off = OFFSET.VERTICES;

  const uq = new Uint16Array(vertexCount);
  const vq = new Uint16Array(vertexCount);
  const hq = new Uint16Array(vertexCount);
  let uAcc = 0;
  let vAcc = 0;
  let hAcc = 0;
  for (let i = 0; i < vertexCount; i++) {
    uAcc += zigzagDecode(view.getUint16(off, true));
    uq[i] = uAcc;
    off += 2;
  }
  for (let i = 0; i < vertexCount; i++) {
    vAcc += zigzagDecode(view.getUint16(off, true));
    vq[i] = vAcc;
    off += 2;
  }
  for (let i = 0; i < vertexCount; i++) {
    hAcc += zigzagDecode(view.getUint16(off, true));
    hq[i] = hAcc;
    off += 2;
  }

  // --- indices (+ optional HiLo watermark offset) --------------------------
  const indexCount = view.getUint32(off, true);
  const watermark = view.getUint32(off + 4, true);
  off += 8;
  const baseIndices = new Uint32Array(indexCount);
  for (let i = 0; i < indexCount; i++) {
    let index = view.getUint16(off, true);
    off += 2;
    if (watermark !== INDEX_WATERMARK) index += watermark;
    baseIndices[i] = index;
  }
  // 之后的边缘索引 / 扩展块（octvertexnormals 等）全部跳过。

  // --- positions (geographic TMS, south-up) --------------------------------
  const bounds = geographicTileBounds(z, x, y);
  const hRange = heights.max - heights.min;

  let edgeCount = 0;
  for (let i = 0; i < vertexCount; i++) {
    if (
      uq[i] === 0 ||
      uq[i] === QUANT_MAX ||
      vq[i] === 0 ||
      vq[i] === QUANT_MAX
    ) {
      edgeCount++;
    }
  }

  // Main vertices + one skirt copy per edge vertex.
  const totalVertices = vertexCount + edgeCount;
  const positions = new Float32Array(totalVertices * 3);
  const skirtIndex = new Int32Array(vertexCount).fill(-1);
  const isEdge = (i: number): boolean =>
    uq[i] === 0 || uq[i] === QUANT_MAX || vq[i] === 0 || vq[i] === QUANT_MAX;
  const xyz: [number, number, number] = [0, 0, 0];
  let skirtCopy = 0;

  for (let i = 0; i < vertexCount; i++) {
    const fU = uq[i] / QUANT_MAX;
    const fV = vq[i] / QUANT_MAX;
    const lon = bounds.west + (bounds.east - bounds.west) * fU;
    // v=0 → 南边缘（south-up TMS）：v/32767 即纬度分数，无需翻转。
    const lat = bounds.south + (bounds.north - bounds.south) * fV;
    const height = heights.min + (hq[i] / QUANT_MAX) * hRange;

    ecefFromLonLatHeight(lon, lat, height, xyz);
    positions[i * 3]     = xyz[0];
    positions[i * 3 + 1] = xyz[1];
    positions[i * 3 + 2] = xyz[2];

    if (isEdge(i)) {
      const s = vertexCount + skirtCopy++;
      skirtIndex[i] = s;
      // 裙边顶点：原位置沿半径向下压 skirtDepth 米。
      const skirtScale = 1.0 - skirtDepth / WGS84_SEMI_MAJOR;
      positions[s * 3]     = xyz[0] * skirtScale;
      positions[s * 3 + 1] = xyz[1] * skirtScale;
      positions[s * 3 + 2] = xyz[2] * skirtScale;
    }
  }

  // --- triangles: base mesh + skirt ----------------------------------------
  const triangles: number[] = [];
  for (let i = 0; i < indexCount; i++) triangles.push(baseIndices[i]);

  // 四条边分别按另一坐标排序，把相邻边缘顶点连成裙边三角形。
  const west: number[] = [];
  const east: number[] = [];
  const south: number[] = [];
  const north: number[] = [];
  for (let i = 0; i < vertexCount; i++) {
    if (!isEdge(i)) continue;
    if (uq[i] === 0) west.push(i);
    if (uq[i] === QUANT_MAX) east.push(i);
    if (vq[i] === 0) south.push(i);
    if (vq[i] === QUANT_MAX) north.push(i);
  }
  west.sort((a, b) => vq[a] - vq[b]);
  east.sort((a, b) => vq[a] - vq[b]);
  south.sort((a, b) => uq[a] - uq[b]);
  north.sort((a, b) => uq[a] - uq[b]);

  const emitSkirtQuad = (a: number, b: number): void => {
    const sa = skirtIndex[a];
    const sb = skirtIndex[b];
    triangles.push(a, sa, b);
    triangles.push(b, sa, sb);
  };
  for (let i = 0; i + 1 < west.length; i++)  emitSkirtQuad(west[i],  west[i + 1]);
  for (let i = 0; i + 1 < east.length; i++)  emitSkirtQuad(east[i],  east[i + 1]);
  for (let i = 0; i + 1 < south.length; i++) emitSkirtQuad(south[i], south[i + 1]);
  for (let i = 0; i + 1 < north.length; i++) emitSkirtQuad(north[i], north[i + 1]);

  const indices =
    totalVertices <= INDEX_WATERMARK
      ? new Uint16Array(triangles)
      : new Uint32Array(triangles);

  return {
    positions,
    indices,
    minHeight: heights.min,
    maxHeight: heights.max,
    center: [
      view.getFloat64(OFFSET.CENTER, true),
      view.getFloat64(OFFSET.CENTER + 8, true),
      view.getFloat64(OFFSET.CENTER + 16, true),
    ],
  };
}

/**
 * Read the elevation range: primary source is the f32 pair at +56; if the
 * values are implausible, fall back to the f64 pair at +56/+64 (real Hubei
 * tiles carry ~3373.004 / -0.27 there, per byte-level verification). Last
 * resort: 0/0 with a warning.
 */
function readHeightRange(view: DataView): { min: number; max: number } {
  // Primary: float64 pair at +56/+64 (verified against real tiles: the
  // Hubei dataset stores ~3373.004 / -0.27 there). The f32 interpretation
  // of those bytes happens to look plausible (~0 / ~5.3), so it must NOT
  // take precedence.
  const c = view.getFloat64(56, true);
  const d = view.getFloat64(64, true);
  const loF64 = Math.min(c, d);
  const hiF64 = Math.max(c, d);
  if (isPlausible(loF64, hiF64) && hiF64 - loF64 > 1e-3) return { min: loF64, max: hiF64 };

  const a = view.getFloat32(56, true);
  const b = view.getFloat32(60, true);
  const loF32 = Math.min(a, b);
  const hiF32 = Math.max(a, b);
  if (isPlausible(loF32, hiF32)) return { min: loF32, max: hiF32 };

  console.warn('QuantizedMeshDecoder: implausible elevation range, using 0/0');
  return { min: 0, max: 0 };
}

/** Plausibility window covers Earth's elevation extremes with margin. */
function isPlausible(min: number, max: number): boolean {
  return (
    Number.isFinite(min) &&
    Number.isFinite(max) &&
    max > min &&
    max < 10000 &&
    min > -12000
  );
}
