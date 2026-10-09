/**
 * TilesetPrimitive – 3DTiles content → Primitive[] 组装。
 *
 * 流水线（对齐 Model.ts 的 fromGltfAsync / _createPrimitive 模式）：
 *   content fetch → (b3dm 剥壳 | 裸 glb) → GltfLoader.parseGlbBuffer
 *   → 逐 meshPrimitive 组装 `Primitive`（纹理走 ImageBitmap / KTX2 双路）
 *   → 返回未注册的 Primitive[]（场景注册与 show/remove 由 Tileset3D 统一管）。
 *
 * 世界矩阵合成：
 *   world = tileTransform × yUpFix × rtcTranslate × nodeMatrix
 *   - tileTransform：TileNode 的归一化累计矩阵（含 1/R 缩放，右侧可喂米）；
 *   - yUpFix = rotX(+90°)：glTF Y-up → 引擎 Z-up（Model.ts:23-33 同款；
 *     fromRotationX 列布局下 gltf +Y → 引擎 +Z）；
 *   - rtcTranslate：RTC_CENTER（米）平移，作用于 glTF 原始坐标系
 *     （Y-up、yUpFix 之前）——与 CesiumJS 顶点着色器语义一致；
 *   - nodeMatrix：glTF 节点矩阵（米制，parseGlbBuffer 逐 primitive 输出）。
 * 右乘顺序即作用顺序：顶点先经 nodeMatrix，再 rtc，再 Y-up 修正，
 * 最后 tileTransform 归一化上球。
 */

import { Cartesian3 } from '../math/Cartesian3';
import { Color } from '../math/Color';
import { Matrix4 } from '../math/Matrix4';
import { Context } from '../renderer/Context';
import { Primitive } from '../scene/Primitive';
import { GltfLoader } from '../loader/GltfLoader';
import type { ParsedGltfModel, ParsedMeshPrimitive } from '../loader/GltfLoader';
import { Ktx2Decoder } from '../loader/Ktx2Decoder';
import { fetchArrayBuffer } from '../loader/fetchWithGzip';
import {
  extractGlb,
  parseRtcCenter,
  parseFeatureTableJson,
  parseBatchTable,
} from './B3dmDecoder';
import type { ExpandedBatchTable, RtcCenter } from './B3dmDecoder';
import {
  buildTileInstanceIds,
  tilesetPickRegistry,
} from './Cesium3DTileFeature';
import type { TileContent } from './TilesetContentLoader';

/** 一个 tile content 的组装产物。 */
export interface BuiltTileContent {
  primitives: Primitive[];
  /** Primitive 之外独立持有的 GPU 纹理（KTX2 解码产物），销毁时一并释放。 */
  textures: GPUTexture[];
  /**
   * b3dm 批次信息（P1-E feature 挑拣）：instance id 基址与批次表。
   * null = 无 `_BATCHID`（tile 级降级）。Tileset3D 由此向
   * `tilesetPickRegistry` 注册 primitive → feature 映射。
   */
  batch: TileBatchInfo | null;
}

/** b3dm 批次挑拣元数据（BuiltTileContent.batch）。 */
export interface TileBatchInfo {
  /** 全局 instance id 基址（batchId 编码为 base + id + 1）。 */
  base: number;
  /** 批次数（BATCH_LENGTH，或 batchIds 最大值 + 1）。 */
  count: number;
  /** 展开的批次表；无表时 null（getProperty 全 undefined）。 */
  table: ExpandedBatchTable | null;
}

/** content 加载器签名（Tileset3D 集成测试经此注入假装配）。 */
export type TileContentLoader = (
  context: Context,
  content: TileContent,
  tileTransform: Matrix4,
  /** 网络选项（P1-F 认证 headers 等，透传给 fetch）。 */
  fetchOptions?: { init?: RequestInit }
) => Promise<BuiltTileContent>;

/**
 * 默认实现：fetch content → 剥壳/解析 → 组装 Primitive[]。
 * b3dm：extractGlb + RTC_CENTER；glb/gltf：整包直接 parseGlbBuffer。
 * 任何失败向上抛（Tileset3D 计入 failed，单瓦片失败不炸整树）。
 */
export const loadTileContent: TileContentLoader = async (
  context: Context,
  content: TileContent,
  tileTransform: Matrix4,
  fetchOptions?: { init?: RequestInit }
): Promise<BuiltTileContent> => {
  const buffer = await fetchArrayBuffer(content.resolvedUrl, { init: fetchOptions?.init });

  let glbBuffer: ArrayBuffer;
  let rtc: RtcCenter | null = null;
  let batch: TileBatchInfo | null = null;
  if (content.kind === 'b3dm') {
    glbBuffer = extractGlb(buffer);
    rtc = parseRtcCenter(buffer);
    batch = parseTileBatchInfo(buffer);
  } else {
    // .glb 直content（3DTiles 1.1）或未知扩展名按 GLB 试解析
    //（parseGlbBuffer 对非法容器会抛错 → 调用方计入 failed）
    glbBuffer = buffer;
  }

  const parsed = await GltfLoader.parseGlbBuffer(glbBuffer);
  return buildTilePrimitives(context, parsed, tileTransform, rtc, batch);
};

/**
 * 解析 b3dm 的批次元数据：BATCH_LENGTH + batchTable 展开；无批次表或
 * BATCH_LENGTH 缺失时仍返回 base（instance id 空间占位）供注册 tile 级
 * 降级。base 由全局 `tilesetPickRegistry.allocate` 分配（跨瓦片唯一）。
 */
function parseTileBatchInfo(buffer: ArrayBuffer): TileBatchInfo {
  let batchLength = 0;
  try {
    const featureTable = parseFeatureTableJson(buffer);
    batchLength = Number(featureTable.BATCH_LENGTH ?? 0) || 0;
  } catch (error) {
    console.error('TilesetPrimitive: b3dm feature table parse failed', error);
  }
  const table = batchLength > 0 ? parseBatchTable(buffer, batchLength) : null;
  return { base: tilesetPickRegistry.allocate(batchLength), count: batchLength, table };
}

/**
 * 把解析好的 glTF 组装成 Primitive[]（不注册场景——注册/显隐/销毁
 * 由 Tileset3D 渲染集状态机统一管理）。
 */
export async function buildTilePrimitives(
  context: Context,
  parsed: ParsedGltfModel,
  tileTransform: Matrix4,
  rtc: RtcCenter | null,
  batch: TileBatchInfo | null = null
): Promise<BuiltTileContent> {
  const yUpFix = Matrix4.fromRotationX(Math.PI / 2);

  const primitives: Primitive[] = [];
  const textures: GPUTexture[] = [];

  for (const meshPrim of parsed.primitives) {
    const mat  = meshPrim.material;
    const rgba = mat.baseColorFactor;

    const translucent = mat.alphaMode === 'BLEND' || rgba[3] < 1.0;
    const primitive = new Primitive(context, {
      vertices:    meshPrim.vertices,
      indices:     meshPrim.indices,
      color:       new Color(rgba[0], rgba[1], rgba[2], rgba[3]),
      translucent,
      alpha:       rgba[3],
      // 3D Tiles 批次挑拣：_BATCHID → 全局 instance id（base + id + 1）
      instanceIds:
        batch && meshPrim.batchIds
          ? buildTileInstanceIds(meshPrim.batchIds, batch.base)
          : undefined,
    });

    primitive.modelMatrix = computeWorldMatrix(tileTransform, yUpFix, rtc, meshPrim);

    // 纹理：PNG/JPEG（ImageBitmap 直传）或 KHR_texture_basisu（KTX2 转码）
    const texture = await uploadBaseColorTexture(context, primitive, mat);
    if (texture) textures.push(texture);

    primitives.push(primitive);
  }
  return { primitives, textures, batch };
}

/**
 * world = tileTransform × yUpFix × rtcTranslate × nodeMatrix。
 * nodeMatrix 为 Float32Array 列主序，fromArray 拷贝后参与乘法。
 */
function computeWorldMatrix(
  tileTransform: Matrix4,
  yUpFix: Matrix4,
  rtc: RtcCenter | null,
  meshPrim: ParsedMeshPrimitive
): Matrix4 {
  let local = Matrix4.fromArray(meshPrim.nodeMatrix);
  if (rtc) {
    local = Matrix4.multiply(
      Matrix4.fromTranslation(new Cartesian3(rtc.x, rtc.y, rtc.z)),
      local,
      new Matrix4()
    );
  }
  const withUpFix = Matrix4.multiply(yUpFix, local, new Matrix4());
  return Matrix4.multiply(tileTransform, withUpFix, new Matrix4());
}

/** 上传基础色纹理；无纹理返回 null。KTX2 需先 Ktx2Decoder.initialize()。 */
async function uploadBaseColorTexture(
  context: Context,
  primitive: Primitive,
  material: ParsedMeshPrimitive['material']
): Promise<GPUTexture | null> {
  if (material.baseColorTexture) {
    const bitmap = material.baseColorTexture;
    const texture = context.device.createTexture({
      label:  'Tileset3D.BaseColorTex',
      size:   { width: bitmap.width, height: bitmap.height },
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST        |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    context.device.queue.copyExternalImageToTexture(
      { source: bitmap, flipY: false },
      { texture },
      { width: bitmap.width, height: bitmap.height }
    );
    primitive.setTexture(texture);
    return texture;
  }

  if (material.baseColorKtx2) {
    const decoder = Ktx2Decoder.instance;
    if (!decoder) {
      throw new Error('Tileset3D: KTX2 纹理需要先 Ktx2Decoder.initialize()（该 tile 计入 failed）');
    }
    const decoded = await decoder.decodeToGpuTexture(context.device, material.baseColorKtx2);
    primitive.setTexture(decoded.texture);
    return decoded.texture;
  }

  return null;
}
