/**
 * Ktx2Decoder – KTX2 / Basis Universal GPU 纹理解码器（3DTiles P3 加载层）。
 *
 * - 运行时以 <script> 标签注入 `public/wasm/basis/basis_transcoder.js`（UMD，
 *   经典脚本下 `var BASIS` 即全局；该 UMD 尾部只有 exports/define 分支，
 *   动态 import() 拿不到全局，因此 script 注入是唯一可靠加载方式）。
 *   wasm 由调用方自行 fetch 后以 `BASIS({ wasmBinary })` 注入（同 three.js
 *   KTX2Loader 的生产模式，避免 locateFile 路径歧义），随后调用
 *   `initializeBasis()`。
 * - 转码目标格式按设备特性探测：
 *   texture-compression-bc → BC7（无 alpha 时退 BC1）/ etc2-rgba8unorm → ETC2_RGBA
 *   / 都不支持 → CPU 解码为 RGBA8（rgba8unorm）。
 * - 压缩纹理写入 GPU 走 queue.writeTexture，按块（block）行计算 bytesPerRow，
 *   行字节数向上对齐 256（WebGPU 线性纹理数据校验要求 bytesPerRow 为 256 的倍数），
 *   未对齐的 mip 用填充行组装后整体写入。
 * - transcodeImage 直接写入普通 Uint8Array（three@0.170 同款调用：
 *   transcodeImage(dst, level, layer, face, format, 0, -1, -1)）。
 */

// ── basis_transcoder 的 TranscoderTextureFormat 枚举值（basis_universal 官方一致） ──

/** basis 转码目标格式常量（取值与 basis_universal 枚举一一对应）。 */
export const TranscoderFormat = {
  cTFETC1_RGB: 0,
  cTFETC2_RGBA: 1,
  cTFBC1_RGB: 2,
  cTFBC3_RGBA: 3,
  cTFBC7_RGBA: 6,
  cTFRGBA32: 13,
} as const;

/** 默认 wasm/js 资产基址（vite publicDir 下的静态资源）。 */
const DEFAULT_WASM_BASE_URL = '/wasm/basis/';

/** WebGPU writeTexture 线性数据的 bytesPerRow 必须是 256 的倍数。 */
const BYTES_PER_ROW_ALIGNMENT = 256;

// ── basis_transcoder.js 模块的最小类型面（UMD 全局 BASIS() 工厂） ────────────

/** 新版 KTX2File：转码目标为普通 Uint8Array（three@0.170 生产用法）。 */
interface BasisKtx2File {
  open(): boolean;
  isUASTC(): boolean;
  isETC1S(): boolean;
  getHasAlpha(): boolean;
  getWidth(): number;
  getHeight(): number;
  getLevels(): number;
  getFaces(): number;
  getLayers(): number;
  startTranscoding(): boolean;
  /** (level, layer, face, format)。 */
  getImageTranscodedSizeInBytes(level: number, layer: number, face: number, format: number): number;
  /** (dst, level, layer, face, format, preprocess, fallback?, output?) — three 0.170 同款 8 参。 */
  transcodeImage(
    dst: Uint8Array,
    level: number,
    layer: number,
    face: number,
    format: number,
    preprocess: number,
    channel0: number,
    channel1: number
  ): boolean;
  close(): void;
  delete(): void;
}

interface BasisModule {
  KTX2File: new (data: Uint8Array) => BasisKtx2File;
  initializeBasis?: () => void;
}

type BasisFactory = (config: { wasmBinary?: ArrayBuffer | Uint8Array }) => Promise<BasisModule>;

// ── KTX2 容器头解析（纯函数，node 测试可用） ─────────────────────────────────

/** KTX2 版本 2 的 12 字节 magic：«ABKTX 20»\r\n\x1A\n */
const KTX2_MAGIC = new Uint8Array([
  0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
]);

/** KTX2 容器头为 80 字节固定区，之后是 levelIndex 数组（每级 24 字节）。 */
const KTX2_HEADER_SIZE = 80;
const KTX2_LEVEL_INDEX_ENTRY_SIZE = 24;

/** 解析出的 KTX2 头信息（只含解码 2D 纹理所需的字段）。 */
export interface Ktx2Header {
  /** vkFormat（0 = VK_FORMAT_UNDEFINED，交给 transcoder 转码）。 */
  vkFormat: number;
  width: number;
  height: number;
  levelCount: number;
  /** 0=none, 1=basisLZ(ETC1S), 2=zstd, 3=zlib。 */
  supercompressionScheme: number;
  /** 每级 mip 在文件内的偏移与长度（字节）。 */
  levels: Array<{ offset: number; byteLength: number }>;
}

/** 判断字节流是否以 KTX2 magic 开头（不抛错）。 */
export function isKtx2(data: ArrayBuffer | Uint8Array): boolean {
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  if (bytes.length < KTX2_MAGIC.length) return false;
  for (let i = 0; i < KTX2_MAGIC.length; i++) {
    if (bytes[i] !== KTX2_MAGIC[i]) return false;
  }
  return true;
}

/**
 * 解析 KTX2 容器头。magic 不对或头字段非法时抛错（不静默）。
 * 仅支持 2D 单层单面（3D / 立方体贴图 / 图层数组直接拒绝）。
 */
export function parseKtx2Header(data: ArrayBuffer): Ktx2Header {
  if (!isKtx2(data)) {
    throw new Error('Ktx2Decoder: invalid KTX2 container (magic mismatch, expected «ABKTX 20»\\r\\n\\x1A\\n)');
  }

  const view = new DataView(data);
  const vkFormat = view.getUint32(12, true);
  const width    = view.getUint32(20, true);
  const height   = view.getUint32(24, true);
  const depth    = view.getUint32(28, true);
  const layers   = view.getUint32(32, true);
  const faces    = view.getUint32(36, true);
  const levelCount = view.getUint32(40, true);
  const supercompressionScheme = view.getUint32(44, true);

  if (width < 1 || height < 1) {
    throw new Error(`Ktx2Decoder: invalid KTX2 dimensions ${width}×${height}`);
  }
  if (depth !== 0 || layers !== 0 || faces !== 1) {
    throw new Error(
      `Ktx2Decoder: unsupported KTX2 layout (depth=${depth}, layers=${layers}, faces=${faces}); ` +
      'only 2D single-face textures are supported'
    );
  }
  if (levelCount < 1 || levelCount > 32) {
    throw new Error(`Ktx2Decoder: invalid KTX2 levelCount ${levelCount}`);
  }

  const levels: Array<{ offset: number; byteLength: number }> = [];
  for (let i = 0; i < levelCount; i++) {
    const entry = KTX2_HEADER_SIZE + i * KTX2_LEVEL_INDEX_ENTRY_SIZE;
    // levelIndex 条目：byteOffset(u64) + byteLength(u64) + uncompressedByteLength(u64)
    levels.push({
      offset:    Number(view.getBigUint64(entry, true)),
      byteLength: Number(view.getBigUint64(entry + 8, true)),
    });
  }

  return { vkFormat, width, height, levelCount, supercompressionScheme, levels };
}

// ── 转码目标格式探测（纯函数，node 测试可用） ────────────────────────────────

/** 选定后的转码格式与其 GPU/WebGPU 侧元数据。 */
export interface SelectedTranscodeFormat {
  /** basis_transcoder 的 cTF* 枚举值。 */
  transcoderFormat: number;
  /** WebGPU 纹理格式。 */
  gpuFormat: GPUTextureFormat;
  /** 块尺寸（未压缩格式为 1×1）。 */
  blockWidth: number;
  blockHeight: number;
  /** 每块字节数（未压缩格式 = 每像素 4 字节）。 */
  bytesPerBlock: number;
  /** 是否 GPU 压缩纹理（false = CPU 解码出的 RGBA8）。 */
  compressed: boolean;
}

/**
 * 按设备特性选择转码目标格式（探测结果由调用方缓存）。
 *
 * 三分支：
 * 1. `texture-compression-bc` → BC7（带 alpha）/ BC1（不带 alpha，更省）
 * 2. 否则 `etc2-rgba8unorm`  → ETC2_RGBA
 * 3. 否则                    → CPU 转码 RGBA8（rgba8unorm，全平台可用）
 */
export function selectTranscodeFormat(
  features: Iterable<string>,
  hasAlpha = true
): SelectedTranscodeFormat {
  const set = features instanceof Set ? features : new Set(features);

  if (set.has('texture-compression-bc')) {
    if (hasAlpha) {
      return {
        transcoderFormat: TranscoderFormat.cTFBC7_RGBA,
        gpuFormat: 'bc7-rgba-unorm',
        blockWidth: 4, blockHeight: 4, bytesPerBlock: 16,
        compressed: true,
      };
    }
    return {
      transcoderFormat: TranscoderFormat.cTFBC1_RGB,
      gpuFormat: 'bc1-rgba-unorm',
      blockWidth: 4, blockHeight: 4, bytesPerBlock: 8,
      compressed: true,
    };
  }

  if (set.has('etc2-rgba8unorm')) {
    return {
      transcoderFormat: TranscoderFormat.cTFETC2_RGBA,
      gpuFormat: 'etc2-rgba8unorm',
      blockWidth: 4, blockHeight: 4, bytesPerBlock: 16,
      compressed: true,
    };
  }

  return {
    transcoderFormat: TranscoderFormat.cTFRGBA32,
    gpuFormat: 'rgba8unorm',
    blockWidth: 1, blockHeight: 1, bytesPerBlock: 4,
    compressed: false,
  };
}

// ── 解码结果 ─────────────────────────────────────────────────────────────────

/** 解码完成后的 GPU 纹理（调用方负责 destroy）。 */
export interface DecodedKtx2 {
  texture: GPUTexture;
  format: GPUTextureFormat;
  width: number;
  height: number;
  mipCount: number;
}

// ── Ktx2Decoder ──────────────────────────────────────────────────────────────

export class Ktx2Decoder {
  private static _module: BasisModule | null = null;
  private static _instance: Ktx2Decoder | null = null;
  private static _initPromise: Promise<void> | null = null;

  /**
   * 加载并实例化 basis_transcoder（幂等：重复调用复用同一 Promise）。
   *
   * @param wasmBaseUrl - 资产目录（需以 / 结尾），默认 '/wasm/basis/'，
   *                      目录下须存在 basis_transcoder.js 与 basis_transcoder.wasm。
   */
  static async initialize(wasmBaseUrl: string = DEFAULT_WASM_BASE_URL): Promise<void> {
    if (Ktx2Decoder._module) return;
    if (Ktx2Decoder._initPromise) return Ktx2Decoder._initPromise;

    Ktx2Decoder._initPromise = (async () => {
      const base = wasmBaseUrl.endsWith('/') ? wasmBaseUrl : wasmBaseUrl + '/';
      const jsUrl = base + 'basis_transcoder.js';

      // UMD：经典 <script> 下 var BASIS 即全局；动态 import() 拿不到，必须注入脚本
      if (typeof document === 'undefined') {
        throw new Error('Ktx2Decoder.initialize(): requires a browser document (script-tag injection of basis_transcoder.js)');
      }

      // wasm 由调用方 fetch 后经 wasmBinary 注入（three.js KTX2Loader 同款，
      // 避免 locateFile 在模块化打包/worker 场景下的路径歧义）
      const [factory, wasmBinary] = await Promise.all([
        Ktx2Decoder._loadScriptFactory(jsUrl),
        fetch(base + 'basis_transcoder.wasm').then((res) => {
          if (!res.ok) {
            throw new Error(`Ktx2Decoder.initialize(): failed to fetch basis_transcoder.wasm (${res.status} ${res.statusText})`);
          }
          return res.arrayBuffer();
        }),
      ]);

      const module = await factory({ wasmBinary });
      // basis_universal 要求在 runtime 就绪后调用一次
      if (typeof module.initializeBasis === 'function') module.initializeBasis();

      if (!module || !module.KTX2File) {
        throw new Error('Ktx2Decoder.initialize(): basis_transcoder loaded but BASIS module lacks KTX2File (wrong wasm build?)');
      }
      Ktx2Decoder._module = module;
      Ktx2Decoder._instance = new Ktx2Decoder();
    })();

    try {
      await Ktx2Decoder._initPromise;
    } catch (err) {
      // 失败后清空状态，允许下次重试
      Ktx2Decoder._initPromise = null;
      throw err;
    }
  }

  /** 已初始化的单例；未初始化时返回 null。 */
  static get instance(): Ktx2Decoder | null {
    return Ktx2Decoder._instance;
  }

  /** 注入 <script> 并等待 BASIS 工厂出现在全局。 */
  private static _loadScriptFactory(jsUrl: string): Promise<BasisFactory> {
    const g = globalThis as unknown as Record<string, unknown>;
    if (typeof g.BASIS === 'function') {
      return Promise.resolve(g.BASIS as BasisFactory);
    }

    return new Promise<BasisFactory>((resolve, reject) => {
      const script = document.createElement('script');
      script.src = jsUrl;
      script.async = true;
      script.onload = () => {
        const factory = (globalThis as unknown as Record<string, unknown>).BASIS;
        if (typeof factory !== 'function') {
          reject(new Error(`Ktx2Decoder.initialize(): basis_transcoder.js loaded but global BASIS() is missing (${jsUrl})`));
          return;
        }
        resolve(factory as BasisFactory);
      };
      script.onerror = () => {
        reject(new Error(`Ktx2Decoder.initialize(): failed to load basis_transcoder.js from "${jsUrl}"`));
      };
      document.head.appendChild(script);
    });
  }

  /**
   * 解码一段 KTX2 字节流为 GPUTexture（设备支持压缩时直接上 GPU 压缩格式，
   * 否则 CPU 转码为 RGBA8）。失败抛带 KTX2 头信息的错误，不静默。
   */
  async decodeToGpuTexture(device: GPUDevice, data: ArrayBuffer): Promise<DecodedKtx2> {
    const module = Ktx2Decoder._module;
    if (!module) {
      throw new Error('Ktx2Decoder not initialized: call Ktx2Decoder.initialize() before decoding');
    }

    const header = parseKtx2Header(data);
    const info =
      `(${header.width}×${header.height}, levels=${header.levelCount}, ` +
      `vkFormat=${header.vkFormat}, scheme=${header.supercompressionScheme})`;

    const file = new module.KTX2File(new Uint8Array(data));
    try {
      if (!file.open()) {
        throw new Error(`Ktx2Decoder: KTX2File.open() failed ${info}`);
      }

      const hasAlpha = file.getHasAlpha();
      const fmt = selectTranscodeFormat(device.features, hasAlpha);
      // isUASTC/isETC1S 仅用于错误信息（两种编码的转码调用路径相同）
      const encoding = file.isUASTC() ? 'UASTC' : 'ETC1S';

      if (!file.startTranscoding()) {
        throw new Error(`Ktx2Decoder: startTranscoding() failed ${info} encoding=${encoding}`);
      }

      // 每级 mip 转码到 CPU 暂存（dst 为普通 Uint8Array，转码结果直接写入）
      const mipDatas: Uint8Array<ArrayBuffer>[] = [];
      for (let level = 0; level < header.levelCount; level++) {
        const dst = new Uint8Array(
          file.getImageTranscodedSizeInBytes(level, 0, 0, fmt.transcoderFormat)
        );
        // three@0.170 KTX2Loader 同款 8 参调用
        const ok = file.transcodeImage(dst, level, 0, 0, fmt.transcoderFormat, 0, -1, -1);
        if (!ok) {
          throw new Error(
            `Ktx2Decoder: transcodeImage failed at mip ${level} ${info} ` +
            `encoding=${encoding} target=0x${fmt.transcoderFormat.toString(16)}`
          );
        }
        mipDatas.push(dst);
      }

      const texture = device.createTexture({
        label: 'Ktx2Decoder.Texture',
        size: { width: header.width, height: header.height },
        format: fmt.gpuFormat,
        mipLevelCount: header.levelCount,
        // 压缩纹理不可作为渲染目标，无需 RENDER_ATTACHMENT
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });

      // 逐级 mip 写入（writeTexture 的 bytesPerRow 有 256 对齐要求）
      for (let level = 0; level < header.levelCount; level++) {
        Ktx2Decoder._writeMip(device, texture, header, fmt, level, mipDatas[level]);
      }

      return {
        texture,
        format: fmt.gpuFormat,
        width: header.width,
        height: header.height,
        mipCount: header.levelCount,
      };
    } finally {
      file.close();
      file.delete();
    }
  }

  /**
   * 单级 mip 写入。压缩纹理行字节数 = 块数 × bytesPerBlock，必须对齐 256；
   * 未对齐时组装填充行缓冲后整体写入（避免逐行多次 writeTexture）。
   */
  private static _writeMip(
    device: GPUDevice,
    texture: GPUTexture,
    header: Ktx2Header,
    fmt: SelectedTranscodeFormat,
    level: number,
    data: Uint8Array<ArrayBuffer>
  ): void {
    const width  = Math.max(1, header.width  >> level);
    const height = Math.max(1, header.height >> level);
    const blocksX = Math.ceil(width  / fmt.blockWidth);
    const blocksY = Math.ceil(height / fmt.blockHeight);
    const rowBytes  = blocksX * fmt.bytesPerBlock;
    const alignedRow = Math.ceil(rowBytes / BYTES_PER_ROW_ALIGNMENT) * BYTES_PER_ROW_ALIGNMENT;

    if (alignedRow === rowBytes) {
      device.queue.writeTexture(
        { texture, mipLevel: level },
        data,
        { bytesPerRow: rowBytes, rowsPerImage: blocksY },
        { width, height, depthOrArrayLayers: 1 }
      );
      return;
    }

    // 行填充：把每行拷进 256 对齐的行距
    const padded = new Uint8Array(alignedRow * blocksY);
    for (let row = 0; row < blocksY; row++) {
      const src = data.subarray(row * rowBytes, (row + 1) * rowBytes);
      padded.set(src, row * alignedRow);
    }
    device.queue.writeTexture(
      { texture, mipLevel: level },
      padded,
      { bytesPerRow: alignedRow, rowsPerImage: blocksY },
      { width, height, depthOrArrayLayers: 1 }
    );
  }
}

export default Ktx2Decoder;
