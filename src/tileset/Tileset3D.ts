/**
 * Tileset3D – 3DTiles 树调度器（P3 主类，仿 TerrainLayer 三段式）。
 *
 * 每帧 `update(camera, viewportHeight)` 三段：
 *   1. _selectTiles      —— 树遍历（显式栈）：视锥 6 平面剔除 +
 *                          SSE 细分（geometricError 直接进屏幕误差公式）
 *                          + maxLevel/数量上限（地平线剔除 TODO，见下）；
 *   2. _launchRequests   —— 选中但未驻留的 content 入队：去重（驻留/在途/
 *                          失败窗口）、近者优先、祖先优先、并发上限 8；
 *   3. _buildRenderSet   —— 渲染集状态机（REPLACE/ADD，见下）逐 Primitive
 *                          设 show；上一帧驱逐的墓碑条目此刻真正销毁。
 *
 * ── REPLACE / ADD 渲染集状态机 ─────────────────────────────────────────
 * 记 selected = 本帧 SSE 选中的最深节点集合，loaded = 内容在 LRU 驻留：
 *   - 选中且 loaded                     → 渲染（REPLACE/ADD 同）；
 *   - 选中且未 loaded                   → 自身不画，向上回退：最近的
 *     「有 content 且 loaded」祖先标记为 fallback 渲染（REPLACE 祖先
 *     因此在子未就绪期间保持可见，子全就绪后自动隐藏）；
 *   - 未选中、refine=ADD、是某选中节点的祖先且 loaded → 常驻渲染；
 *   - 未选中、refine=REPLACE、loaded、无未就绪选中后代     → show=false。
 * 祖先 content 加载优先于叶子（_launchRequests 中祖先以
 * `distance - ANCESTOR_PRIORITY` 入队排前）——没有粗层父级，回退链就断了。
 *
 * ── 延迟销毁（两段式）────────────────────────────────────────────────
 * LRU 驱逐先隐藏（show=false，当帧不再提交），下一帧 update() 开头
 * flushDestroyed 后才 scene.primitives.remove（WebGPU 不允许销毁仍被
 * 未提交 command buffer 引用的资源）。
 *
 * ── TODO ──────────────────────────────────────────────────────────────
 * - 地平线剔除未实现（地形/影像层的锥测依赖瓦片贴地假设；3DTiles 内容
 *   可离地，直接套用会误剔高空瓦片）。当前只有视锥 + MAX_SELECTED 兜底。
 * - 3DTiles 1.1 多内容（contents[]）只取第一个。
 * - pnts 点云未支持：console.warn 后计入 failed（不发起网络请求）。
 */

import { Cartesian3 } from '../math/Cartesian3';
import { Context } from '../renderer/Context';
import type { Scene } from '../scene/Scene';
import type { Camera } from '../scene/Camera';
import {
  fetchTilesetJson,
  parseTileset,
  TILESET_EARTH_SCALE,
} from './TilesetContentLoader';
import type { TileNode, TileUrlDecorator } from './TilesetContentLoader';
import { TilesetTileCache } from './TilesetTileCache';
import type { TileCacheEntry } from './TilesetTileCache';
import { loadTileContent } from './TilesetPrimitive';
import type { BuiltTileContent, TileContentLoader } from './TilesetPrimitive';

/** 失败瓦片的重试窗口（限流 429 不是"瓦片不存在"，窗口过后可重试）。 */
const FAILED_RETRY_MS = 30_000;
/** 单帧最多选中的瓦片数（防失控树打爆请求队列）。 */
const MAX_SELECTED_TILES = 256;
/** 祖先 content 请求的排序加成（比叶子先入队）。 */
const ANCESTOR_PRIORITY = 1e6;

export interface Tileset3DOptions {
  /** tileset.json URL（相对页面基址解析）。 */
  url: string;
  /** 宿主场景（Primitive 注册目的地；Context 经私有字段取得）。 */
  scene: Scene;
  /** 细分屏幕误差阈值（像素）。默认 16。 */
  maximumSseError?: number;
  /** 最大树深（根 = 0）。默认 Infinity。 */
  maximumLevel?: number;
  /** LRU 容量（content 条目数）。默认 64。 */
  cacheCapacity?: number;
  /** 并发请求上限。默认 8。 */
  requestConcurrency?: number;
  /** 每请求 URL 加工钩子（Google key 注入）。 */
  urlDecorator?: TileUrlDecorator;
  /**
   * 测试缝：替换 content 加载（集成测试注入假 Primitive，不触网）。
   * 生产代码勿传。
   */
  _contentLoader?: TileContentLoader;
}

export interface Tileset3DStats {
  /** 最近一次 update 选中的瓦片数。 */
  selected: number;
  /** 当前在途请求数。 */
  requested: number;
  /** LRU 驻留 content 数。 */
  cached: number;
  /** 累计失败 content 数（pnts 不支持也计入）。 */
  failed: number;
  /** 最近一次 update 实际可见瓦片数（渲染集）。 */
  drawn: number;
}

// ── SSE 纯函数（tests/tileset/sse.spec.ts 直接覆盖）─────────────────────────

/**
 * 屏幕空间误差（像素）。geometricError 是 3DTiles 原始米制值，先 /R
 * 归一化再进公式（与 ImageryLayer 的 sse = worldSize × vh / (d·2tan(fov/2))
 * 同构，这里 worldSize 即 geometricError/R）。
 */
export function computeTileSse(
  geometricErrorMeters: number,
  distanceNormalized: number,
  viewportHeight: number,
  fovRadians: number,
  earthScale: number = TILESET_EARTH_SCALE
): number {
  const dist = Math.max(distanceNormalized, 1e-9);
  return (
    (geometricErrorMeters / earthScale) *
    (viewportHeight / (dist * 2.0 * Math.tan(fovRadians * 0.5)))
  );
}

/** 细分决策：sse 超阈值 且 有子级 且 未到最大深度。 */
export function shouldRefine(
  sse: number,
  maximumSseError: number,
  level: number,
  maximumLevel: number,
  hasChildren: boolean
): boolean {
  return hasChildren && level < maximumLevel && sse > maximumSseError;
}

// ── 主类 ─────────────────────────────────────────────────────────────────────

interface SelectedTile {
  node: TileNode;
  distance: number;
}

export class Tileset3D {
  private readonly _url: string;
  private readonly _scene: Scene;
  private readonly _maximumSseError: number;
  private readonly _maximumLevel: number;
  private readonly _requestConcurrency: number;
  private readonly _urlDecorator?: TileUrlDecorator;
  private readonly _loadContent: TileContentLoader;

  private readonly _cache: TilesetTileCache;
  /** 在途 content 请求（key = resolvedUrl）。 */
  private readonly _inflight = new Map<string, Promise<void>>();
  /** 失败表：key → 失败时刻（performance.now()），窗口内不重试。 */
  private readonly _failed = new Map<string, number>();
  private _failedTotal = 0;

  private _root: TileNode | null = null;
  private _readyPromise: Promise<void> | null = null;
  private _destroyed = false;

  private _frame = 0;
  private _selected: SelectedTile[] = [];
  private _drawn = 0;
  /** 视锥 6 平面复用缓冲（24 floats）。 */
  private readonly _frustumPlanes = new Float32Array(24);

  constructor(options: Tileset3DOptions) {
    this._url                = options.url;
    this._scene              = options.scene;
    this._maximumSseError    = options.maximumSseError ?? 16;
    this._maximumLevel       = options.maximumLevel ?? Number.POSITIVE_INFINITY;
    this._requestConcurrency = Math.max(1, options.requestConcurrency ?? 8);
    this._urlDecorator       = options.urlDecorator;
    this._loadContent        = options._contentLoader ?? loadTileContent;
    this._cache              = new TilesetTileCache({ capacity: options.cacheCapacity ?? 64 });
  }

  /** 根节点（ready 前为 null）。 */
  get root(): TileNode | null { return this._root; }

  /** 就绪即 tileset.json 已取回并解析成树。幂等（重复调用同 Promise）。 */
  ready(): Promise<void> {
    if (!this._readyPromise) {
      this._readyPromise = this._loadRoot();
    }
    return this._readyPromise;
  }

  /** 最近一次 update 的统计快照。 */
  get stats(): Tileset3DStats {
    return {
      selected: this._selected.length,
      requested: this._inflight.size,
      cached: this._cache.size,
      failed: this._failedTotal,
      drawn: this._drawn,
    };
  }

  /** 每帧调度入口（Viewer.preRender 中调用）。ready 前调用是安全的 no-op。 */
  update(camera: Camera, viewportHeight: number = 1080): void {
    if (this._destroyed || this._root === null) return;

    this._frame++;
    // 两段式销毁第 2 段：上一帧驱逐的条目此刻才真正 remove/destroy
    this._flushDestroyed();

    this._selectTiles(camera, viewportHeight);
    this._launchRequests();
    this._buildRenderSet();
  }

  /** 销毁：卸载全部 Primitive/纹理，清空调度状态；在途请求完成后自弃。 */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;

    const all = this._cache.destroyAll();
    for (const entry of all) this._removeEntry(entry);

    this._inflight.clear();
    this._failed.clear();
    this._selected = [];
    this._drawn = 0;
  }

  // ── 根加载 ────────────────────────────────────────────────────────────────

  private async _loadRoot(): Promise<void> {
    const { json, finalUrl } = await fetchTilesetJson(this._url, {
      urlDecorator: this._urlDecorator,
    });
    // content uri 的相对基准 = 加工后的 tileset.json URL
    const rootNode = parseTileset(json, finalUrl, { urlDecorator: this._urlDecorator });
    if (this._destroyed) return;
    this._root = rootNode;
  }

  // ── 1. 选择 ───────────────────────────────────────────────────────────────

  private _selectTiles(camera: Camera, viewportHeight: number): void {
    const planes = this._extractFrustumPlanes(camera.viewProjectionMatrix.values);
    const camPos = camera.position;
    const fov = camera.frustum.fov;
    // TODO(地平线剔除)：3DTiles 内容可离地，Terrain/Imagery 的锥测
    // （dot(P, camDir) 对地心锥）会误剔高空瓦片；待引入每 tile 离地
    // 高度估计后再启用。当前仅视锥 + MAX_SELECTED 上限兜底。

    const selected: SelectedTile[] = [];
    const stack: TileNode[] = [this._root as TileNode];

    while (stack.length > 0) {
      const node = stack.pop() as TileNode;
      const sphere = node.sphere;

      // ── 视锥剔除（6 平面，Gribb/Hartmann）────────────────────────────
      let outside = false;
      for (let i = 0; i < 6; i++) {
        const o = i * 4;
        const d = planes[o]     * sphere.center.x +
                  planes[o + 1] * sphere.center.y +
                  planes[o + 2] * sphere.center.z +
                  planes[o + 3];
        if (d < -sphere.radius) { outside = true; break; }
      }
      if (outside) continue;

      // ── SSE 细分 ─────────────────────────────────────────────────────
      const distance = Math.max(Cartesian3.distance(camPos, sphere.center), 1e-9);
      const sse = computeTileSse(node.geometricError, distance, viewportHeight, fov);

      if (
        shouldRefine(
          sse,
          this._maximumSseError,
          node.depth,
          this._maximumLevel,
          node.children.length > 0
        ) &&
        selected.length < MAX_SELECTED_TILES
      ) {
        for (const child of node.children) stack.push(child);
      } else {
        selected.push({ node, distance });
      }
    }

    this._selected = selected;
  }

  // ── 2. 请求 ───────────────────────────────────────────────────────────────

  private _launchRequests(): void {
    // 入队：选中的叶子 + 沿途祖先（有 content 才需要加载）
    interface QueueItem { node: TileNode; distance: number; }
    const queue: QueueItem[] = [];
    const seen = new Set<string>();

    const enqueue = (node: TileNode, distance: number): void => {
      const content = node.content;
      if (!content) return;
      const key = content.resolvedUrl;
      if (seen.has(key)) return;

      // pnts 点云本版不支持：warn 一次、计入 failed、不发请求
      if (content.kind === 'pnts') {
        if (!this._failed.has(key)) {
          this._failed.set(key, performance.now());
          this._failedTotal++;
          console.warn(`Tileset3D: point cloud not supported yet（${content.uri} 已跳过）`);
        }
        return;
      }

      if (this._cache.has(key) || this._inflight.has(key) || this._isFailed(key)) return;
      seen.add(key);
      queue.push({ node, distance });
    };

    for (const item of this._selected) {
      enqueue(item.node, item.distance);
      // 祖先链预取（回退保证）：排前（ANCESTOR_PRIORITY 注释）
      let ancestor = item.node.parent;
      while (ancestor !== null) {
        enqueue(ancestor, Math.max(0, item.distance - ANCESTOR_PRIORITY));
        ancestor = ancestor.parent;
      }
    }

    queue.sort((a, b) => a.distance - b.distance);

    for (const item of queue) {
      if (this._inflight.size >= this._requestConcurrency) break;
      this._startRequest(item.node);
    }
  }

  private _startRequest(node: TileNode): void {
    const content = node.content as NonNullable<TileNode['content']>;
    const key = content.resolvedUrl;
    const context = (this._scene as unknown as { _context: Context })._context;

    const task = (async () => {
      const built = await this._loadContent(context, content, node.transform);
      if (this._destroyed) {
        this._disposeBuilt(built); // 在途期间被销毁：直接释放，不上屏
        return;
      }
      // 逐 Primitive 注册（show=false 入场，显隐由渲染集状态机统一决定）
      for (const primitive of built.primitives) {
        primitive.show = false;
        if (primitive.translucent) this._scene.addTransparentPrimitive(primitive);
        else this._scene.addPrimitive(primitive);
      }
      this._cache.insert(key, node, built.primitives, this._frame);
    })();

    this._inflight.set(key, task);
    void task
      .catch((error: unknown) => {
        // 单瓦片失败不炸整树：计数 + console.error + 30s 重试窗口
        this._failed.set(key, performance.now());
        this._failedTotal++;
        console.error(`Tileset3D: content 加载失败 ${content.uri}`, error);
      })
      .finally(() => {
        this._inflight.delete(key);
      });
  }

  /** 失败窗口内的 key 不重试（限流 429 ≠ 瓦片不存在）。 */
  private _isFailed(key: string): boolean {
    const failedAt = this._failed.get(key);
    if (failedAt === undefined) return false;
    if (performance.now() - failedAt > FAILED_RETRY_MS) {
      this._failed.delete(key);
      return false;
    }
    return true;
  }

  // ── 3. 渲染集（REPLACE/ADD 状态机）───────────────────────────────────────

  private _buildRenderSet(): void {
    const selectedSet = new Set<TileNode>();
    // 选中节点的祖先链：ADD 常驻的判定域
    const ancestorsOfSelected = new Set<TileNode>();
    // 需要回退渲染的祖先：从「选中但未就绪」的节点向上，最近 loaded 祖先
    const fallbackVisible = new Set<TileNode>();

    for (const { node } of this._selected) {
      selectedSet.add(node);
      for (let a = node.parent; a !== null; a = a.parent) {
        ancestorsOfSelected.add(a);
      }
      if (!this._isNodeLoaded(node)) {
        for (let a = node.parent; a !== null; a = a.parent) {
          if (a.content !== null && this._isNodeLoaded(a)) {
            fallbackVisible.add(a);
            break;
          }
        }
      }
    }

    let drawn = 0;
    this._cache.forEach((entry) => {
      const node = entry.tile;
      const visible =
        selectedSet.has(node) ||
        (node.refine === 'ADD' && ancestorsOfSelected.has(node)) ||
        fallbackVisible.has(node);

      for (const primitive of entry.primitives) primitive.show = visible;
      if (visible) drawn++;
    });

    this._drawn = drawn;
  }

  private _isNodeLoaded(node: TileNode): boolean {
    return node.content !== null && this._cache.has(node.content.resolvedUrl);
  }

  // ── 延迟销毁 ──────────────────────────────────────────────────────────────

  /** 两段式第 2 段：墓碑条目真正从场景移除（Primitive.destroy）。 */
  private _flushDestroyed(): void {
    for (const entry of this._cache.flushDestroyed(this._frame)) {
      this._removeEntry(entry);
    }
  }

  /** 从场景卸载并销毁一个 content 的全部 GPU 资源。 */
  private _removeEntry(entry: TileCacheEntry): void {
    for (const primitive of entry.primitives) {
      this._scene.primitives.remove(primitive); // 内部调用 primitive.destroy()
    }
  }

  private _disposeBuilt(built: BuiltTileContent): void {
    for (const primitive of built.primitives) primitive.destroy();
    for (const texture of built.textures) texture.destroy();
  }

  // ── 视锥平面（Gribb/Hartmann，WebGPU 列主序）─────────────────────────────

  /**
   * 从 viewProj 矩阵抽取 6 平面（left/right/bottom/top/near/far），
   * 每平面 4 floats [a,b,c,d]，法向已归一化。列主序布局下
   * 行 i = values[i], values[i+4], values[i+8], values[i+12]。
   */
  private _extractFrustumPlanes(values: Float32Array): Float32Array {
    const p = this._frustumPlanes;
    const a0 = values[0], a1 = values[4], a2 = values[8],  a3 = values[12]; // row 0
    const b0 = values[1], b1 = values[5], b2 = values[9],  b3 = values[13]; // row 1
    const c0 = values[2], c1 = values[6], c2 = values[10], c3 = values[14]; // row 2
    const d0 = values[3], d1 = values[7], d2 = values[11], d3 = values[15]; // row 3

    // left = row3 + row0, right = row3 - row0
    setPlane(p, 0,  d0 + a0, d1 + a1, d2 + a2, d3 + a3);
    setPlane(p, 1,  d0 - a0, d1 - a1, d2 - a2, d3 - a3);
    // bottom = row3 + row1, top = row3 - row1
    setPlane(p, 2,  d0 + b0, d1 + b1, d2 + b2, d3 + b3);
    setPlane(p, 3,  d0 - b0, d1 - b1, d2 - b2, d3 - b3);
    // near = row3 + row2, far = row3 - row2
    setPlane(p, 4,  d0 + c0, d1 + c1, d2 + c2, d3 + c3);
    setPlane(p, 5,  d0 - c0, d1 - c1, d2 - c2, d3 - c3);
    return p;
  }
}

/** 归一化后写入平面缓冲。 */
function setPlane(
  p: Float32Array, slot: number,
  a: number, b: number, c: number, d: number
): void {
  const len = Math.hypot(a, b, c) || 1;
  const o = slot * 4;
  p[o] = a / len; p[o + 1] = b / len; p[o + 2] = c / len; p[o + 3] = d / len;
}
