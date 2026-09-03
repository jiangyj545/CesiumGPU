/**
 * GltfLoader – Parses GLTF 2.0 and GLB 2.0 files.
 *
 * Supports:
 *  - GLB binary container (magic 0x46546C67)
 *  - GLTF JSON with external / data-URI buffers
 *  - All GLTF 2.0 accessor component types (BYTE, UBYTE, SHORT, USHORT, UINT, FLOAT)
 *  - Interleaved and non-interleaved buffer views
 *  - Node hierarchy (TRS and matrix transforms)
 *  - PBR metallic-roughness materials
 *  - Textures: embedded in GLB buffer views, external URLs, or data URIs
 */

// ── GLTF JSON type definitions ────────────────────────────────────────────────

interface GltfJson {
  asset: { version: string };
  scene?: number;
  scenes?: Array<{ nodes?: number[] }>;
  nodes?: GltfNode[];
  meshes?: GltfMesh[];
  materials?: GltfMaterial[];
  textures?: GltfTexture[];
  images?: GltfImage[];
  samplers?: GltfSampler[];
  accessors?: GltfAccessor[];
  bufferViews?: GltfBufferView[];
  buffers?: GltfBuffer[];
}

interface GltfNode {
  name?: string;
  mesh?: number;
  children?: number[];
  /** Column-major 4×4 matrix overrides TRS when present. */
  matrix?: number[];
  translation?: [number, number, number];
  rotation?: [number, number, number, number]; // quaternion [x,y,z,w]
  scale?: [number, number, number];
}

interface GltfMesh {
  name?: string;
  primitives: GltfPrimitive[];
}

interface GltfPrimitive {
  attributes: {
    POSITION?: number;
    NORMAL?: number;
    TEXCOORD_0?: number;
    [key: string]: number | undefined;
  };
  indices?: number;
  material?: number;
  /** 4 = TRIANGLES (default) */
  mode?: number;
}

interface GltfMaterial {
  name?: string;
  pbrMetallicRoughness?: {
    baseColorFactor?: [number, number, number, number];
    baseColorTexture?: { index: number; texCoord?: number };
    metallicFactor?: number;
    roughnessFactor?: number;
  };
  alphaMode?: 'OPAQUE' | 'MASK' | 'BLEND';
  alphaCutoff?: number;
  doubleSided?: boolean;
}

interface GltfTexture {
  source?: number;
  sampler?: number;
}

interface GltfImage {
  uri?: string;
  mimeType?: string;
  bufferView?: number;
}

interface GltfSampler {
  magFilter?: number;
  minFilter?: number;
  wrapS?: number;
  wrapT?: number;
}

interface GltfAccessor {
  bufferView?: number;
  byteOffset?: number;
  componentType: number; // 5120=BYTE,5121=UBYTE,5122=SHORT,5123=USHORT,5125=UINT,5126=FLOAT
  count: number;
  type: string; // 'SCALAR'|'VEC2'|'VEC3'|'VEC4'|'MAT2'|'MAT3'|'MAT4'
  min?: number[];
  max?: number[];
  normalized?: boolean;
}

interface GltfBufferView {
  buffer: number;
  byteOffset?: number;
  byteLength: number;
  byteStride?: number;
  target?: number;
}

interface GltfBuffer {
  byteLength: number;
  uri?: string;
}

// ── Public result types ───────────────────────────────────────────────────────

/** Material properties parsed from a GLTF material definition. */
export interface ParsedMaterial {
  /** RGBA base colour factor [0–1]. */
  baseColorFactor: [number, number, number, number];
  /** Optional decoded image for the base-colour texture. */
  baseColorTexture?: ImageBitmap;
  metallicFactor: number;
  roughnessFactor: number;
  alphaMode: 'OPAQUE' | 'MASK' | 'BLEND';
  doubleSided: boolean;
}

/** One GPU-ready mesh primitive (vertices interleaved, uint32 indices). */
export interface ParsedMeshPrimitive {
  /** Interleaved vertex buffer: [position(3), normal(3), uv(2)] per vertex = 8 floats = 32 bytes. */
  vertices: Float32Array;
  /** Uint32 index buffer. */
  indices: Uint32Array;
  material: ParsedMaterial;
  /** Column-major 4×4 world-space transform for this primitive within the GLTF model. */
  nodeMatrix: Float32Array;
}

/** The full result of parsing a GLTF/GLB file. */
export interface ParsedGltfModel {
  primitives: ParsedMeshPrimitive[];
}

// ── Component-type constants ──────────────────────────────────────────────────

const COMPONENT_TYPE_SIZE: Record<number, number> = {
  5120: 1, // BYTE
  5121: 1, // UNSIGNED_BYTE
  5122: 2, // SHORT
  5123: 2, // UNSIGNED_SHORT
  5125: 4, // UNSIGNED_INT
  5126: 4, // FLOAT
};

const ACCESSOR_TYPE_COUNT: Record<string, number> = {
  SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4,
  MAT2: 4, MAT3: 9, MAT4: 16,
};

// ── GltfLoader ────────────────────────────────────────────────────────────────

/**
 * GltfLoader – Utility class for parsing GLTF 2.0 and GLB 2.0 files.
 */
export class GltfLoader {
  /**
   * Fetch and parse a GLTF or GLB file from the given URL.
   *
   * @param url - Absolute or relative URL pointing to a `.gltf` or `.glb` file.
   * @returns Parsed model data ready to be converted to GPU primitives.
   */
  static async load(url: string): Promise<ParsedGltfModel> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`GltfLoader: failed to fetch "${url}" (${response.status} ${response.statusText})`);
    }

    const buffer = await response.arrayBuffer();

    // Detect GLB by checking the magic number 0x46546C67 ('glTF' in little-endian)
    const magic = new DataView(buffer).getUint32(0, true);
    const isGlb  = magic === 0x46546C67;

    let json: GltfJson;
    let embeddedBinary: ArrayBuffer | null = null;

    if (isGlb) {
      ({ json, embeddedBinary } = GltfLoader._parseGlbHeader(buffer));
    } else {
      json = JSON.parse(new TextDecoder().decode(buffer)) as GltfJson;
    }

    // Resolve all buffers (embedded or external)
    const buffers = await GltfLoader._resolveBuffers(json, url, embeddedBinary);

    // Load all images referenced by materials
    const images = await GltfLoader._resolveImages(json, buffers, url);

    // Walk the scene graph and collect mesh primitives
    return GltfLoader._buildModel(json, buffers, images);
  }

  // ── GLB parsing ─────────────────────────────────────────────────────────────

  private static _parseGlbHeader(
    buffer: ArrayBuffer
  ): { json: GltfJson; embeddedBinary: ArrayBuffer | null } {
    const view   = new DataView(buffer);
    // GLB header: magic(4) + version(4) + length(4) = 12 bytes
    let offset   = 12;
    let json: GltfJson | null = null;
    let embeddedBinary: ArrayBuffer | null = null;

    while (offset < buffer.byteLength) {
      const chunkLength = view.getUint32(offset,     true);
      const chunkType   = view.getUint32(offset + 4, true);
      offset += 8;

      if (chunkType === 0x4E4F534A /* JSON */) {
        const jsonText = new TextDecoder().decode(new Uint8Array(buffer, offset, chunkLength));
        json = JSON.parse(jsonText) as GltfJson;
      } else if (chunkType === 0x004E4942 /* BIN\0 */) {
        embeddedBinary = buffer.slice(offset, offset + chunkLength);
      }

      offset += chunkLength;
    }

    if (!json) throw new Error('GltfLoader: GLB file contains no JSON chunk');
    return { json, embeddedBinary };
  }

  // ── Buffer resolution ────────────────────────────────────────────────────────

  private static async _resolveBuffers(
    json: GltfJson,
    baseUrl: string,
    embeddedBinary: ArrayBuffer | null
  ): Promise<ArrayBuffer[]> {
    const result: ArrayBuffer[] = [];
    if (!json.buffers) return result;

    for (let i = 0; i < json.buffers.length; i++) {
      const buf = json.buffers[i];
      if (i === 0 && embeddedBinary !== null) {
        // First buffer of a GLB is always the embedded binary chunk
        result.push(embeddedBinary);
      } else if (buf.uri) {
        result.push(await GltfLoader._loadUri(buf.uri, baseUrl));
      } else {
        // No URI and no embedded data – should not happen in valid GLTF
        result.push(new ArrayBuffer(buf.byteLength));
      }
    }

    return result;
  }

  private static async _loadUri(uri: string, baseUrl: string): Promise<ArrayBuffer> {
    if (uri.startsWith('data:')) {
      // Data URI: data:[mediatype];base64,<data>
      const commaIdx = uri.indexOf(',');
      const base64   = uri.slice(commaIdx + 1);
      const binary   = atob(base64);
      const bytes    = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return bytes.buffer;
    }
    // Relative or absolute URL
    const resolvedUrl = new URL(uri, baseUrl).href;
    const res = await fetch(resolvedUrl);
    if (!res.ok) throw new Error(`GltfLoader: failed to fetch buffer "${resolvedUrl}"`);
    return res.arrayBuffer();
  }

  // ── Image resolution ─────────────────────────────────────────────────────────

  private static async _resolveImages(
    json: GltfJson,
    buffers: ArrayBuffer[],
    baseUrl: string
  ): Promise<(ImageBitmap | null)[]> {
    if (!json.images) return [];

    const promises = json.images.map(async (img, imageIndex): Promise<ImageBitmap | null> => {
      try {
        if (img.bufferView !== undefined) {
          // Image embedded in a bufferView (typical in GLB)
          const bv = json.bufferViews?.[img.bufferView];
          if (!bv) {
            throw new Error(`GltfLoader: image ${imageIndex} references missing bufferView ${img.bufferView}`);
          }
          const buf = buffers[bv.buffer];
          if (!buf) {
            throw new Error(`GltfLoader: image ${imageIndex} bufferView ${img.bufferView} references missing buffer ${bv.buffer}`);
          }
          const arr = new Uint8Array(buf, bv.byteOffset ?? 0, bv.byteLength);
          const blob = new Blob([arr], { type: img.mimeType ?? 'image/jpeg' });
          return createImageBitmap(blob);
        } else if (img.uri) {
          if (img.uri.startsWith('data:')) {
            const commaIdx = img.uri.indexOf(',');
            const base64   = img.uri.slice(commaIdx + 1);
            const binary   = atob(base64);
            const bytes    = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            const mimeType = img.uri.slice(5, commaIdx).split(';')[0];
            const blob = new Blob([bytes], { type: mimeType });
            return createImageBitmap(blob);
          }
          // External image URL
          const imgUrl = new URL(img.uri, baseUrl).href;
          const res = await fetch(imgUrl);
          if (!res.ok) return null;
          const blob = await res.blob();
          return createImageBitmap(blob);
        }
      } catch (err) {
        console.warn('GltfLoader: failed to load image', err);
      }
      return null;
    });

    return Promise.all(promises);
  }

  // ── Scene graph traversal ────────────────────────────────────────────────────

  private static _buildModel(
    json: GltfJson,
    buffers: ArrayBuffer[],
    images: (ImageBitmap | null)[]
  ): ParsedGltfModel {
    const result: ParsedGltfModel = { primitives: [] };

    const sceneIndex = json.scene ?? 0;
    const scene = json.scenes?.[sceneIndex];
    if (!scene) return result;

    const identityMatrix = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);

    for (const nodeIndex of (scene.nodes ?? [])) {
      GltfLoader._visitNode(json, buffers, images, nodeIndex, identityMatrix, result);
    }

    return result;
  }

  private static _visitNode(
    json: GltfJson,
    buffers: ArrayBuffer[],
    images: (ImageBitmap | null)[],
    nodeIndex: number,
    parentWorldMatrix: Float32Array,
    result: ParsedGltfModel
  ): void {
    const node = json.nodes?.[nodeIndex];
    if (!node) return;

    // Compute local transform
    const localMatrix = node.matrix
      ? new Float32Array(node.matrix)
      : GltfLoader._trsToMatrix(
          node.translation ?? [0, 0, 0],
          node.rotation    ?? [0, 0, 0, 1],
          node.scale       ?? [1, 1, 1]
        );

    const worldMatrix = GltfLoader._multiplyMat4(parentWorldMatrix, localMatrix);

    // Process attached mesh
    if (node.mesh !== undefined) {
      const mesh = json.meshes?.[node.mesh];
      if (mesh) {
        for (const prim of mesh.primitives) {
          const parsed = GltfLoader._parsePrimitive(json, buffers, images, prim);
          if (parsed) {
            result.primitives.push({ ...parsed, nodeMatrix: worldMatrix });
          }
        }
      }
    }

    // Recurse into children
    for (const childIndex of (node.children ?? [])) {
      GltfLoader._visitNode(json, buffers, images, childIndex, worldMatrix, result);
    }
  }

  // ── Mesh primitive parsing ───────────────────────────────────────────────────

  private static _parsePrimitive(
    json: GltfJson,
    buffers: ArrayBuffer[],
    images: (ImageBitmap | null)[],
    prim: GltfPrimitive
  ): Omit<ParsedMeshPrimitive, 'nodeMatrix'> | null {
    // Only triangle-list primitives are supported (mode 4, or absent)
    const mode = prim.mode ?? 4;
    if (mode !== 4) {
      console.warn(`GltfLoader: unsupported primitive mode ${mode}, skipping`);
      return null;
    }

    const posIdx = prim.attributes.POSITION;
    if (posIdx === undefined) return null;

    // Read position accessor
    let positions = GltfLoader._readFloatAccessor(json, buffers, posIdx, 3);
    const vertexCount = positions.length / 3;

    // Read UV0 accessor (optional – default [0,0])
    const uvIdx = prim.attributes.TEXCOORD_0;
    let uvs = uvIdx !== undefined
      ? GltfLoader._readFloatAccessor(json, buffers, uvIdx, 2)
      : new Float32Array(vertexCount * 2);

    // Read indices (resolved first – flat normal generation needs the triangle list)
    let indices: Uint32Array;
    if (prim.indices !== undefined) {
      const raw = GltfLoader._readIndexAccessor(json, buffers, prim.indices);
      indices = raw instanceof Uint32Array ? raw : new Uint32Array(raw);
    } else {
      // Non-indexed: sequential
      indices = new Uint32Array(vertexCount);
      for (let i = 0; i < vertexCount; i++) indices[i] = i;
    }

    // Read normal accessor (optional – generate flat normals if absent)
    const normIdx = prim.attributes.NORMAL;
    let normals: Float32Array;
    if (normIdx !== undefined) {
      normals = GltfLoader._readFloatAccessor(json, buffers, normIdx, 3);
    } else if (prim.indices !== undefined) {
      // Indexed triangles are defined by index triples: un-weld the vertices
      // and assign each triangle its own flat face normal.
      ({ positions, normals, uvs, indices } = GltfLoader._generateFlatNormalsIndexed(positions, uvs, indices));
    } else {
      // Non-indexed: consecutive vertex triples form triangles.
      normals = GltfLoader._generateFlatNormals(positions);
    }

    // Build interleaved buffer: [pos(3), norm(3), uv(2)] × vertexCount
    const outVertexCount = positions.length / 3;
    const vertices = new Float32Array(outVertexCount * 8);
    for (let i = 0; i < outVertexCount; i++) {
      const vi = i * 8;
      vertices[vi + 0] = positions[i * 3 + 0];
      vertices[vi + 1] = positions[i * 3 + 1];
      vertices[vi + 2] = positions[i * 3 + 2];
      vertices[vi + 3] = normals[i * 3 + 0];
      vertices[vi + 4] = normals[i * 3 + 1];
      vertices[vi + 5] = normals[i * 3 + 2];
      vertices[vi + 6] = uvs[i * 2 + 0];
      vertices[vi + 7] = uvs[i * 2 + 1];
    }

    // Material
    const material = GltfLoader._parseMaterial(json, images, prim.material);

    return { vertices, indices, material };
  }

  // ── Accessor reading ─────────────────────────────────────────────────────────

  /**
   * Read an accessor as a flat Float32Array, converting from any component type.
   * @param componentCount - Expected number of components per element (e.g., 3 for VEC3).
   */
  private static _readFloatAccessor(
    json: GltfJson,
    buffers: ArrayBuffer[],
    accessorIndex: number,
    componentCount: number
  ): Float32Array {
    const accessor = json.accessors?.[accessorIndex];
    if (!accessor) {
      throw new Error(`GltfLoader: accessor ${accessorIndex} does not exist in glTF (glTF declares ${json.accessors?.length ?? 0} accessors)`);
    }
    const typeCount  = ACCESSOR_TYPE_COUNT[accessor.type] ?? componentCount;
    const compSize   = COMPONENT_TYPE_SIZE[accessor.componentType] ?? 4;
    const count      = accessor.count;

    if (accessor.bufferView === undefined) {
      // Sparse or zero-filled accessor (sparse not yet supported)
      return new Float32Array(count * typeCount);
    }

    const bv = json.bufferViews?.[accessor.bufferView];
    if (!bv) {
      throw new Error(`GltfLoader: accessor ${accessorIndex} references missing bufferView ${accessor.bufferView}`);
    }
    const buffer = buffers[bv.buffer];
    if (!buffer) {
      throw new Error(`GltfLoader: accessor ${accessorIndex} bufferView ${accessor.bufferView} references missing buffer ${bv.buffer}`);
    }
    const byteOffset = (bv.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
    const stride   = bv.byteStride ?? (typeCount * compSize);
    const isPacked = stride === typeCount * compSize;

    if (accessor.componentType === 5126 /* FLOAT */ && isPacked) {
      // Fast path: already Float32, no stride gaps
      const byteLength = count * typeCount * 4;
      if (byteOffset < 0 || byteOffset + byteLength > buffer.byteLength) {
        throw new Error(`GltfLoader: accessor ${accessorIndex} read out of bounds (byteOffset ${byteOffset} + ${byteLength} bytes exceeds buffer size ${buffer.byteLength})`);
      }
      return new Float32Array(buffer.slice(
        byteOffset,
        byteOffset + byteLength
      ));
    }

    // Slow path: convert each element
    const out   = new Float32Array(count * typeCount);
    const dv    = new DataView(buffer);
    const normalized = accessor.normalized ?? false;

    for (let i = 0; i < count; i++) {
      const base = byteOffset + i * stride;
      for (let j = 0; j < typeCount; j++) {
        const off = base + j * compSize;
        out[i * typeCount + j] = GltfLoader._readComponent(
          dv, off, accessor.componentType, normalized
        );
      }
    }

    return out;
  }

  /**
   * Read an index accessor, returning Uint16Array or Uint32Array depending on component type.
   */
  private static _readIndexAccessor(
    json: GltfJson,
    buffers: ArrayBuffer[],
    accessorIndex: number
  ): Uint16Array | Uint32Array {
    const accessor = json.accessors?.[accessorIndex];
    if (!accessor) {
      throw new Error(`GltfLoader: accessor ${accessorIndex} does not exist in glTF (glTF declares ${json.accessors?.length ?? 0} accessors)`);
    }
    const count      = accessor.count;
    const compType   = accessor.componentType;

    if (accessor.bufferView === undefined) {
      return new Uint32Array(count);
    }

    const bv = json.bufferViews?.[accessor.bufferView];
    if (!bv) {
      throw new Error(`GltfLoader: accessor ${accessorIndex} references missing bufferView ${accessor.bufferView}`);
    }
    const buffer = buffers[bv.buffer];
    if (!buffer) {
      throw new Error(`GltfLoader: accessor ${accessorIndex} bufferView ${accessor.bufferView} references missing buffer ${bv.buffer}`);
    }
    const byteOff  = (bv.byteOffset ?? 0) + (accessor.byteOffset ?? 0);

    if (compType === 5123 /* UNSIGNED_SHORT */) {
      return new Uint16Array(buffer.slice(byteOff, byteOff + count * 2));
    }
    if (compType === 5125 /* UNSIGNED_INT */) {
      return new Uint32Array(buffer.slice(byteOff, byteOff + count * 4));
    }
    if (compType === 5121 /* UNSIGNED_BYTE */) {
      const bytes = new Uint8Array(buffer, byteOff, count);
      const out   = new Uint32Array(count);
      for (let i = 0; i < count; i++) out[i] = bytes[i];
      return out;
    }

    // Fallback
    const out = new Uint32Array(count);
    const dv  = new DataView(buffer);
    for (let i = 0; i < count; i++) {
      out[i] = GltfLoader._readComponent(dv, byteOff + i * 2, compType, false);
    }
    return out;
  }

  private static _readComponent(
    dv: DataView,
    byteOffset: number,
    componentType: number,
    normalized: boolean
  ): number {
    switch (componentType) {
      case 5126: return dv.getFloat32(byteOffset, true);
      case 5125: return dv.getUint32(byteOffset, true);
      case 5123: {
        const v = dv.getUint16(byteOffset, true);
        return normalized ? v / 65535 : v;
      }
      case 5122: {
        const v = dv.getInt16(byteOffset, true);
        return normalized ? Math.max(v / 32767, -1) : v;
      }
      case 5121: {
        const v = dv.getUint8(byteOffset);
        return normalized ? v / 255 : v;
      }
      case 5120: {
        const v = dv.getInt8(byteOffset);
        return normalized ? Math.max(v / 127, -1) : v;
      }
      default: return 0;
    }
  }

  // ── Material parsing ─────────────────────────────────────────────────────────

  private static _parseMaterial(
    json: GltfJson,
    images: (ImageBitmap | null)[],
    materialIndex?: number
  ): ParsedMaterial {
    const defaults: ParsedMaterial = {
      baseColorFactor: [1, 1, 1, 1],
      metallicFactor:  0.0,
      roughnessFactor: 0.5,
      alphaMode:       'OPAQUE',
      doubleSided:     false,
    };

    if (materialIndex === undefined || !json.materials) return defaults;

    const mat = json.materials[materialIndex];
    if (!mat) return defaults;

    const pbr = mat.pbrMetallicRoughness ?? {};
    const baseColorFactor = (pbr.baseColorFactor ?? [1, 1, 1, 1]) as [number, number, number, number];
    const metallicFactor  = pbr.metallicFactor  ?? 0.0;
    const roughnessFactor = pbr.roughnessFactor ?? 0.5;
    const alphaMode       = mat.alphaMode  ?? 'OPAQUE';
    const doubleSided     = mat.doubleSided ?? false;

    // Resolve base-colour texture
    let baseColorTexture: ImageBitmap | undefined;
    if (pbr.baseColorTexture !== undefined && json.textures) {
      const texEntry = json.textures[pbr.baseColorTexture.index];
      if (texEntry?.source !== undefined) {
        const bmp = images[texEntry.source];
        if (bmp) baseColorTexture = bmp;
      }
    }

    return { baseColorFactor, metallicFactor, roughnessFactor, alphaMode, doubleSided, baseColorTexture };
  }

  // ── Normal generation ────────────────────────────────────────────────────────

  /**
   * Generate per-vertex normals from triangle geometry (flat shading).
   */
  private static _generateFlatNormals(positions: Float32Array): Float32Array {
    const normals = new Float32Array(positions.length);
    const triCount = positions.length / 9; // 3 vertices × 3 components
    for (let t = 0; t < triCount; t++) {
      const i = t * 9;
      const ax = positions[i + 0], ay = positions[i + 1], az = positions[i + 2];
      const bx = positions[i + 3], by = positions[i + 4], bz = positions[i + 5];
      const cx = positions[i + 6], cy = positions[i + 7], cz = positions[i + 8];

      // Edge vectors
      const ex = bx - ax, ey = by - ay, ez = bz - az;
      const fx = cx - ax, fy = cy - ay, fz = cz - az;

      // Cross product
      let nx = ey * fz - ez * fy;
      let ny = ez * fx - ex * fz;
      let nz = ex * fy - ey * fx;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (len > 0) { nx /= len; ny /= len; nz /= len; }

      for (let v = 0; v < 3; v++) {
        normals[i + v * 3 + 0] = nx;
        normals[i + v * 3 + 1] = ny;
        normals[i + v * 3 + 2] = nz;
      }
    }
    return normals;
  }

  /**
   * Generate flat per-face normals for an indexed primitive (no NORMAL attribute).
   *
   * Triangles are defined by index triples (not consecutive vertex triples),
   * so shared vertices are un-welded (expanded) and each triangle carries its
   * own face normal – matching the flat-shading semantics of the non-indexed
   * path.
   *
   * @returns Expanded positions/normals/uvs plus sequential indices.
   */
  private static _generateFlatNormalsIndexed(
    positions: Float32Array,
    uvs: Float32Array,
    indices: Uint32Array
  ): { positions: Float32Array; normals: Float32Array; uvs: Float32Array; indices: Uint32Array } {
    const vertexCount = positions.length / 3;
    const triCount = (indices.length / 3) | 0;
    const outPositions = new Float32Array(triCount * 9);
    const outNormals   = new Float32Array(triCount * 9);
    const outUvs       = new Float32Array(triCount * 6);
    const outIndices   = new Uint32Array(triCount * 3);

    for (let t = 0; t < triCount; t++) {
      const cornerIndices = [indices[t * 3 + 0], indices[t * 3 + 1], indices[t * 3 + 2]];
      const ia = cornerIndices[0];
      const ib = cornerIndices[1];
      const ic = cornerIndices[2];

      if (ia >= vertexCount || ib >= vertexCount || ic >= vertexCount) {
        throw new Error(`GltfLoader: primitive index ${Math.max(ia, ib, ic)} out of range (vertex count ${vertexCount})`);
      }

      const ax = positions[ia * 3 + 0], ay = positions[ia * 3 + 1], az = positions[ia * 3 + 2];
      const bx = positions[ib * 3 + 0], by = positions[ib * 3 + 1], bz = positions[ib * 3 + 2];
      const cx = positions[ic * 3 + 0], cy = positions[ic * 3 + 1], cz = positions[ic * 3 + 2];

      // Edge vectors
      const ex = bx - ax, ey = by - ay, ez = bz - az;
      const fx = cx - ax, fy = cy - ay, fz = cz - az;

      // Cross product
      let nx = ey * fz - ez * fy;
      let ny = ez * fx - ex * fz;
      let nz = ex * fy - ey * fx;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (len > 0) { nx /= len; ny /= len; nz /= len; }

      for (let v = 0; v < 3; v++) {
        const src = cornerIndices[v];
        const di  = t * 9 + v * 3;
        outPositions[di + 0] = positions[src * 3 + 0];
        outPositions[di + 1] = positions[src * 3 + 1];
        outPositions[di + 2] = positions[src * 3 + 2];
        outNormals[di + 0] = nx;
        outNormals[di + 1] = ny;
        outNormals[di + 2] = nz;
        outUvs[t * 6 + v * 2 + 0] = uvs[src * 2 + 0];
        outUvs[t * 6 + v * 2 + 1] = uvs[src * 2 + 1];
        outIndices[t * 3 + v] = t * 3 + v;
      }
    }

    return { positions: outPositions, normals: outNormals, uvs: outUvs, indices: outIndices };
  }

  // ── Matrix helpers ───────────────────────────────────────────────────────────

  /**
   * Convert TRS (translation / rotation / scale) to a column-major 4×4 matrix.
   */
  private static _trsToMatrix(
    translation: [number, number, number],
    rotation:    [number, number, number, number],
    scale:       [number, number, number]
  ): Float32Array {
    const [tx, ty, tz] = translation;
    const [qx, qy, qz, qw] = rotation;
    const [sx, sy, sz] = scale;

    // Quaternion → rotation matrix
    const x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
    const xx = qx * x2, xy = qx * y2, xz = qx * z2;
    const yy = qy * y2, yz = qy * z2, zz = qz * z2;
    const wx = qw * x2, wy = qw * y2, wz = qw * z2;

    const m = new Float32Array(16);
    // Column 0
    m[0] = (1 - (yy + zz)) * sx;
    m[1] = (xy + wz) * sx;
    m[2] = (xz - wy) * sx;
    m[3] = 0;
    // Column 1
    m[4] = (xy - wz) * sy;
    m[5] = (1 - (xx + zz)) * sy;
    m[6] = (yz + wx) * sy;
    m[7] = 0;
    // Column 2
    m[8]  = (xz + wy) * sz;
    m[9]  = (yz - wx) * sz;
    m[10] = (1 - (xx + yy)) * sz;
    m[11] = 0;
    // Column 3 (translation)
    m[12] = tx;
    m[13] = ty;
    m[14] = tz;
    m[15] = 1;

    return m;
  }

  /**
   * Multiply two column-major 4×4 matrices: result = a × b.
   */
  private static _multiplyMat4(a: Float32Array, b: Float32Array): Float32Array {
    const r = new Float32Array(16);
    for (let col = 0; col < 4; col++) {
      for (let row = 0; row < 4; row++) {
        let sum = 0;
        for (let k = 0; k < 4; k++) {
          sum += a[k * 4 + row] * b[col * 4 + k];
        }
        r[col * 4 + row] = sum;
      }
    }
    return r;
  }
}

export default GltfLoader;
