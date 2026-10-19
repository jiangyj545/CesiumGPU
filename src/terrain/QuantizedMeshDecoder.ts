/**
 * QuantizedMeshDecoder - decoder for the little-endian quantized-mesh
 * variant served by the local terrain service (verified byte-by-byte
 * against real tiles, e.g. the Hubei dataset).
 *
 * Binary layout after gzip decompression (ALL little-endian):
 *
 * ```
 * +0   centerX, centerY, centerZ   3 x float64   (tile centre, ECEF meters)
 * +24  minHeight, maxHeight        2 x float32   (official spec position)
 * +32  boundingSphereCenter        3 x float64
 * +56  boundingSphereRadius        float64
 * +64  horizonOcclusionPoint       3 x float64
 * +88  vertexCount                 uint32
 * +92  zigzag-delta encoded        vertexCount x uint16 per channel,
 *      u / v / h                   in that order
 *      then u32 declaredIndexCount followed by a CONTINUOUS uint16
 *      high-water-mark index stream (idx = highest - code; code == 0
 *      pushes highest). The declared count only covers the first segment;
 *      the real stream ends where the edge-index region (four u32 counts
 *      west/south/east/north, each in [1, 200]) begins. Trailing edge
 *      indices / extensions e.g. octvertexnormals are skipped.
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

/** Header field offsets (bytes), matching the CesiumJS 88-byte header. */
const OFFSET = {
  CENTER: 0,          // 3 x f64
  MIN_HEIGHT: 24,     // f32 (official spec position)
  MAX_HEIGHT: 28,     // f32
  VERTEX_COUNT: 88,   // u32
  VERTICES: 92,       // u16 zigzag-delta: u[vcount] v[vcount] h[vcount]
} as const;

/** Quantized u/v/h full-scale value. */
const QUANT_MAX = 32767;
/**
 * Highest elevation any real DEM on Earth reaches (Everest 8849 m) plus
 * margin; header scale values beyond this are per-level quantisation
 * artefacts of the source service, not real elevations.
 */
const PHYSICAL_MAX_ELEVATION = 9000;
/** Maximum vertices addressable with 16-bit indices. */
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

  const heights = readHeightRange(view, z);
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

  // --- indices (high-water-mark, continuous until the edge-index region) ---
  // Layout verified against the live service and the official CesiumJS
  // decoder (CesiumTerrainProvider.createQuantizedMeshTerrainData):
  //   u32 count followed by a CONTINUOUS u16 high-water-mark stream. The
  //   u32 is the **triangle count** in this service's tiles (3× index values);
  //   older readings treated it as a value count, truncating the stream to a
  //   third and leaving every tile ~half untriangulated (sampleHeight covered
  //   only ~50% of each tile — page 63 measured 33% grid coverage). Try the
  //   3×count reading first and validate every decoded index against
  //   vertexCount; on any out-of-range index (or truncated buffer) fall back
  //   to the legacy edge-region scan below.
  const declaredCount = view.getUint32(off, true);
  off += 4;

  const decodeHighWater = (count: number): Uint32Array | null => {
    if (off + 2 * count > u8.length) return null; // truncated buffer
    const out = new Uint32Array(count);
    let highest = 0;
    for (let i = 0; i < count; i++) {
      const code = view.getUint16(off + 2 * i, true);
      const index = highest - code;
      if (code === 0) highest++;
      if (index < 0 || index >= vertexCount) return null; // desynced stream
      out[i] = index;
    }
    return out;
  };

  // 两种读法竞争，取「更长的合法解码」：
  // - A：declared 为三角形数（本服务真实布局，3×declared 个值）；
  // - B：旧读法（declared 为值数）+ 边缘区扫描（合成瓦片 / 旧数据布局）。
  // 合成旧布局瓦片的 declared 恰为值数且带边缘区时，B 通过扫描解出比
  // A 更长的流（A 只读到 min(3×declared, 缓冲) 且常常越界判非法），
  // 旧断言保持成立；真实服务瓦片无 4×u32 边缘区，B 只剩 declared 个
  // 值（旧 bug：三角形只剩 1/3），A 胜出。
  const candidateA = decodeHighWater(3 * declaredCount);
  const findEdgeRegionStart = (): number => {
    const maxScan = Math.floor((u8.length - off - 16) / 2);
    for (let n = declaredCount; n <= maxScan; n++) {
      const p = off + 2 * n;
      if (p % 4 !== 0) continue; // counts are u32-aligned reads
      const w = view.getUint32(p, true);
      const s = view.getUint32(p + 4, true);
      const e = view.getUint32(p + 8, true);
      const no = view.getUint32(p + 12, true);
      if (
        w >= 1 && w <= 200 && s >= 1 && s <= 200 &&
        e >= 1 && e <= 200 && no >= 1 && no <= 200
      ) {
        return n - (n % 3); // whole triangles only
      }
    }
    return declaredCount; // fall back to the declared segment length
  };
  const candidateB = decodeHighWater(findEdgeRegionStart());
  const baseIndices =
    candidateA !== null && (candidateB === null || candidateA.length >= candidateB.length)
      ? candidateA
      : (candidateB ?? new Uint32Array(0));
  off += 2 * baseIndices.length;
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
  for (let i = 0; i < baseIndices.length; i++) triangles.push(baseIndices[i]);

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
    // 裙边顶点紧跟主网格（vertexCount..totalVertices-1），CPU 采样据此
    // 排除裙边三角形（贴边采样不再得到负的裙深高程）。
    skirtVertexStart: vertexCount,
    center: [
      view.getFloat64(OFFSET.CENTER, true),
      view.getFloat64(OFFSET.CENTER + 8, true),
      view.getFloat64(OFFSET.CENTER + 16, true),
    ],
  };
}

/**
 * Read the elevation range. The official spec position is the f32 pair at
 * +24/+28 (verified live: the Hubei service carries e.g. 0/964 or 488/1572
 * there, and coarse-level tiles carry per-level scales ~12886 at z10 that
 * exceed any physical elevation). The legacy f64 pair at +56/+64 (which is
 * actually the bounding-sphere radius in the official layout) is kept as a
 * fallback for tiles whose f32 pair is implausible.
 */
function readHeightRange(view: DataView, z: number): { min: number; max: number } {
  // Primary: official spec f32 pair at +24/+28.
  const a = view.getFloat32(OFFSET.MIN_HEIGHT, true);
  const b = view.getFloat32(OFFSET.MAX_HEIGHT, true);
  const loF32 = Math.min(a, b);
  const hiF32 = Math.max(a, b);
  if (isPlausible(loF32, hiF32) && hiF32 - loF32 > 1e-3) {
    return { min: loF32, max: hiF32 };
  }

  // Fallback: f64 pair at +56/+64 (observed on some tiles). Coarse levels
  // store a per-level scale (~2x per level above z12) that no real
  // elevation reaches — normalise it back.
  const c = view.getFloat64(56, true);
  const d = view.getFloat64(64, true);
  let loF64 = Math.min(c, d);
  let hiF64 = Math.max(c, d);
  if (isPlausible(loF64, hiF64) && hiF64 - loF64 > 1e-3) {
    if (hiF64 > PHYSICAL_MAX_ELEVATION && z < 12) {
      hiF64 = hiF64 / 2 ** (12 - z);
      loF64 = Math.min(loF64, 0);
    }
    return { min: loF64, max: hiF64 };
  }

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
