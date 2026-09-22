import type { Context } from '../renderer/Context';
import { Primitive } from '../scene/Primitive';
import { ExtrudedGeometry } from './ExtrudedGeometry';
import type { ExtrudedGeometryBuffers, LonLatPoint } from './ExtrudedGeometry';
import { GeometryPipeline } from './GeometryPipeline';

export interface CityBlockBuilderOptions {
  /**
   * 单个分块的顶点+索引缓冲预算（字节）。累计到此阈值即 flush 成独立的
   * Primitive（合并 buffer 路径下一块 = 一次 draw）。默认 128MB，留足
   * 余量避开 WebGPU `maxBufferSize` 上限（主流实现 256MB）。
   */
  maxChunkBytes?: number;
}

/** 一个分块的纯数据结果（无 GPU 依赖，可直接测试 / 离线检查）。 */
export interface CityBlockChunk {
  /** 合并后的几何（8 float 交错 + 索引），`Primitive` 直接消费。 */
  geometry: ExtrudedGeometryBuffers;
  /**
   * 每顶点实例 id（u32，F12）——builder 内全局唯一的建筑序号（1 起，
   * < 0x800000）。pick 管线把 `0x800000 | instanceId` 渲染进 id 平面，
   * 消费侧（SensorCam / bbox-from-seg）按 bit23 判别实例像素，实现
   * 逐楼 bbox。语义类别仍走 `geometry.semanticIds`（类别 → 调色板）。
   */
  instanceIds: Uint32Array;
  /** 该分块包含的楼栋数。 */
  buildingCount: number;
}

/**
 * CityBlockBuilder - 城市建筑白模批量合批构建器。
 *
 * ## 为什么是合并 buffer 而不是 GPU instancing
 *
 * 每栋建筑 footprint 顶点数不同（矩形 4 点 / L 形 6 点 / 任意多边形），
 * instanced 要求各实例共享同一份顶点布局，不适用。改为把 N 栋楼的顶点
 * 直接拼进一个大 vertex/index buffer：一个 `Primitive` 一次
 * `drawIndexed` 画完整个街区，CPU 侧只有每楼一次 `ExtrudedGeometry.create`
 * + 一次 typed array 拷贝。
 *
 * ## semanticId（F3 分割 pass 铺路）
 *
 * 每顶点 u32 语义 ID 随几何一起合并（`ExtrudedGeometryBuffers.semanticIds`）。
 * 类别 id 不唯一——F12 起 builder 另分配每建筑唯一实例 id（`instanceIds`，
 * builder 内 1 起递增），经 `Primitive` 的第二顶点缓冲进 pick 管线：
 * 类别给调色板、实例给 bbox（见 `CityBlockChunk.instanceIds`）。
 *
 * ## 分块（WebGPU buffer size 上限）
 *
 * 单 buffer 超 `maxBufferSize`（主流 256MB）会创建失败；累计字节数到
 * `maxChunkBytes`（默认 128MB）自动 flush 成独立 chunk → 独立 Primitive。
 */
export class CityBlockBuilder {
  /** 默认分块阈值：128MB（顶点+索引），低于 WebGPU maxBufferSize 上限。 */
  static readonly DEFAULT_MAX_CHUNK_BYTES = 128 * 1024 * 1024;

  private readonly _maxChunkBytes: number;

  /** 实例 id 分配上限（23-bit）：pick shader 输出 `0x800000 | id`，bit23 为实例标记位。 */
  static readonly MAX_INSTANCE_ID = 0x7FFFFF;

  /** 当前未 flush 分块的累积段（typed array 切片列表，flush 时拼接）。 */
  private _vertSegs: Float32Array[] = [];
  private _idxSegs: number[] = [];        // 已含分块内顶点基址偏移
  private _semanticSegs: Uint32Array[] = [];
  private _instanceSegs: Uint32Array[] = [];
  private _nextInstanceId = 1;            // builder 内全局唯一建筑序号（F12）
  private _chunkVertFloats = 0;           // 当前分块 vertices 总 float 数
  private _chunkBytes = 0;                // 当前分块 顶点+索引 字节数
  private _chunkBuildings = 0;
  private _chunkVertexBase = 0;           // 当前分块首顶点序号（索引偏移用）

  /** 已 flush 的分块。 */
  private _chunks: CityBlockChunk[] = [];

  private _buildingCount = 0;
  private _triangleCount = 0;

  constructor(options: CityBlockBuilderOptions = {}) {
    this._maxChunkBytes = options.maxChunkBytes ?? CityBlockBuilder.DEFAULT_MAX_CHUNK_BYTES;
  }

  /** 累计楼栋总数。 */
  get buildingCount(): number { return this._buildingCount; }

  /** 累计三角形总数（含所有未 flush 分块）。 */
  get triangleCount(): number { return this._triangleCount; }

  /** 已完成 flush 的分块数 + 当前累积中的 1 块（空 builder 为 0）。 */
  get chunkCount(): number { return this._chunks.length + (this._chunkBuildings > 0 ? 1 : 0); }

  /**
   * 添加一栋建筑（footprint 经纬度环 + 高度米 + 语义 ID）。
   * 空/退化 footprint 静默跳过（与 `ExtrudedGeometry.create` 空缓冲约定一致）。
   * @returns this，支持链式调用
   */
  addBuilding(ring: LonLatPoint[], heightM: number, semanticId = 0): this {
    const g = ExtrudedGeometry.create({ ring, heightM, semanticId });
    if (g.indexCount === 0) return this;

    // F12 实例 id：builder 内全局唯一建筑序号（1 起）。semanticId 是类别 id
    // （palette 查表用），不唯一；实例分离靠本 id。
    if (this._nextInstanceId > CityBlockBuilder.MAX_INSTANCE_ID) {
      throw new Error('CityBlockBuilder: 23-bit instance id space exhausted');
    }
    const instanceId = this._nextInstanceId++;

    // 分块预算：顶点(32B/顶点) + 索引(4B/索引)。放不下且当前块非空 → 先 flush
    const bytes = g.vertices.byteLength + g.indices.byteLength;
    if (this._chunkBuildings > 0 && this._chunkBytes + bytes > this._maxChunkBytes) {
      this._flushChunk();
    }

    this._vertSegs.push(g.vertices);
    this._chunkVertFloats += g.vertices.length;
    this._chunkBytes += bytes;
    this._semanticSegs.push(g.semanticIds);

    const buildingInstances = new Uint32Array(g.vertexCount).fill(instanceId);
    this._instanceSegs.push(buildingInstances);

    const base = this._chunkVertexBase;
    for (let i = 0; i < g.indices.length; i++) {
      this._idxSegs.push(base + g.indices[i]);
    }
    this._chunkVertexBase += g.vertexCount;
    this._chunkBuildings++;
    this._buildingCount++;
    this._triangleCount += g.indexCount / 3;
    return this;
  }

  /**
   * 输出所有分块的纯数据（无 GPU 依赖）。
   * 调用后 builder 仍可继续 addBuilding（后续进入新分块）。
   */
  buildChunks(): CityBlockChunk[] {
    if (this._chunkBuildings > 0) this._flushChunk();
    return this._chunks;
  }

  /**
   * 构建 `Primitive` 列表（每分块一个，一次 `drawIndexed` 画一块）。
   * 白模默认：`Color.WHITE` 基色（Primitive 缺省）+ 标准光照管线，
   * 顶点直接烘焙在归一化 ECEF 世界系（modelMatrix 恒等）。
   *
   * 注意：返回的 Primitive 必须经 `scene.addPrimitive(p)` 入场 —— 该方法
   * 会补建 opaque bind group；直接 `scene.primitives.add(p)` 会因缺
   * bind group 被 `Primitive.drawOpaque` 静默跳过（不渲染、不报错）。
   * @param context WebGPU 上下文（`viewer.context`）
   */
  build(context: Context): Primitive[] {
    const chunks = this.buildChunks();
    const out: Primitive[] = [];
    for (const c of chunks) {
      // WebGPU 禁止零尺寸 buffer（空几何块跳过）
      if (c.geometry.indexCount === 0 || c.geometry.vertices.byteLength === 0) continue;
      out.push(new Primitive(context, {
        vertices: c.geometry.vertices,
        indices:  c.geometry.indices,
        instanceIds: c.instanceIds,
      }));
    }
    return out;
  }

  /** 拼接当前累积段为独立分块，重置累积状态。 */
  private _flushChunk(): void {
    const vertices = new Float32Array(this._chunkVertFloats);
    let off = 0;
    for (const seg of this._vertSegs) { vertices.set(seg, off); off += seg.length; }

    const semanticIds = new Uint32Array(vertices.length / GeometryPipeline.VERTEX_STRIDE);
    off = 0;
    for (const seg of this._semanticSegs) { semanticIds.set(seg, off); off += seg.length; }

    const instanceIds = new Uint32Array(vertices.length / GeometryPipeline.VERTEX_STRIDE);
    off = 0;
    for (const seg of this._instanceSegs) { instanceIds.set(seg, off); off += seg.length; }

    const chunk: CityBlockChunk = {
      geometry: {
        vertices,
        indices: Uint32Array.from(this._idxSegs),
        vertexCount: vertices.length / GeometryPipeline.VERTEX_STRIDE,
        indexCount: this._idxSegs.length,
        semanticIds,
      },
      instanceIds,
      buildingCount: this._chunkBuildings,
    };
    this._chunks.push(chunk);

    this._vertSegs = [];
    this._idxSegs = [];
    this._semanticSegs = [];
    this._instanceSegs = [];
    this._chunkVertFloats = 0;
    this._chunkBytes = 0;
    this._chunkBuildings = 0;
    this._chunkVertexBase = 0;
  }
}
