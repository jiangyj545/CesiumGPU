/**
 * TilesetTileCache – tile 内容记录 LRU（3DTiles 调度层）。
 *
 * key = content 的 resolvedUrl（同一 content 被多个 tile 引用时天然去重）；
 * value = { tile, primitives[], lastFrame }。容量压力下驱逐最旧条目。
 *
 * ── 延迟销毁（两段式）─────────────────────────────────────────────────
 * WebGPU 禁止销毁仍被未提交 command buffer 引用的资源。本引擎 Scene
 * 没有 per-frame 回调（frameHooks 是 (encoder, view) 签名，不适用），
 * 故用「标记 + 下一帧 update() 开头销毁」的简单两段式：
 *   1. 驱逐当刻条目移入 graveyard——不立即 destroy；
 *   2. 下一次 `flushDestroyed(frame)`（Tileset3D.update 开头调用，帧号
 *      已 +1）才返回墓碑条目，调度层经 scene.primitives.remove 真正
 *      destroy。驱逐发生在上一帧 submit 之后、下一帧 submit 之前，
 *      中间至少跨过一次完整 submit 边界，满足 WebGPU 约束（语义同
 *      TileTextureCache.onFrameEnd，仅以 update() 为帧界）。
 * 若被驱逐的 key 在墓碑未 flush 前又被请求（相机往返抖动），insert 会
 * 复活该条目——避免「驱逐→立刻重载」抖动下的重复 GPU 上传。
 */

import type { Primitive } from '../scene/Primitive';
import type { TileNode } from './TilesetContentLoader';

/** 一条 tile 内容缓存记录。 */
export interface TileCacheEntry {
  /** content resolvedUrl。 */
  key: string;
  tile: TileNode;
  /** 该 content 组装出的全部 Primitive（调度层负责 add/show/remove）。 */
  primitives: Primitive[];
  /** 最近一次 get/insert 时的帧号（LRU recency）。 */
  lastFrame: number;
}

export interface TilesetTileCacheOptions {
  /** 最大驻留条目数。默认 64（同 TerrainLayer LRU 量级）。 */
  capacity?: number;
}

export interface TilesetTileCacheStats {
  capacity: number;
  size: number;
  /** 累计容量驱逐次数（含已 flush）。 */
  evictions: number;
  /** 当前在墓碑中等待下一帧销毁的条目数。 */
  pendingDestroy: number;
}

interface GraveyardRecord {
  entry: TileCacheEntry;
  /** 驱逐时的帧号；flush 条件 = currentFrame - frame >= DEFER_FRAMES。 */
  frame: number;
}

/** 驱逐后至少经过的 update 次数才真正销毁（=1：下一帧）。 */
const DEFER_FRAMES = 1;

export class TilesetTileCache {
  private readonly _capacity: number;
  /** 插入序 = LRU recency（Map 迭代按插入序；get 时 delete+set 重插）。 */
  private readonly _entries = new Map<string, TileCacheEntry>();
  private readonly _graveyard = new Map<string, GraveyardRecord>();
  private _evictions = 0;

  constructor(options?: TilesetTileCacheOptions) {
    this._capacity = Math.max(1, options?.capacity ?? 64);
  }

  get capacity(): number { return this._capacity; }
  get size(): number { return this._entries.size; }

  get stats(): TilesetTileCacheStats {
    return {
      capacity: this._capacity,
      size: this._entries.size,
      evictions: this._evictions,
      pendingDestroy: this._graveyard.size,
    };
  }

  /** 是否驻留（不刷新 recency）。 */
  has(key: string): boolean {
    return this._entries.has(key);
  }

  /** 取记录并刷新 LRU recency；不存在返回 null。 */
  get(key: string): TileCacheEntry | null {
    const entry = this._entries.get(key);
    if (!entry) return null;
    // 重插 = 提到最新（Map 保持插入序）
    this._entries.delete(key);
    this._entries.set(key, entry);
    return entry;
  }

  /**
   * 插入/刷新记录。容量超出时驱逐最旧条目（墓碑挂起，下一帧销毁）。
   * 若 key 在墓碑中尚未销毁则复活（见文件头「两段式」说明）。
   */
  insert(key: string, tile: TileNode, primitives: Primitive[], frame: number): void {
    // 抖动复活：墓碑里还没销毁的条目直接搬回驻留区
    const buried = this._graveyard.get(key);
    if (buried) {
      this._graveyard.delete(key);
      buried.entry.lastFrame = frame;
      this._entries.delete(key); // 防御：正常不会同时存在于两区
      this._entries.set(key, buried.entry);
      return;
    }

    // 已驻留：刷新 recency 与内容
    const existing = this._entries.get(key);
    if (existing) {
      existing.lastFrame = frame;
      this._entries.delete(key);
      this._entries.set(key, existing);
      return;
    }

    this._entries.set(key, { key, tile, primitives, lastFrame: frame });
    this._evictOverflow(frame);
  }

  /** 遍历驻留条目（不刷新 recency）。 */
  forEach(cb: (entry: TileCacheEntry) => void): void {
    for (const entry of this._entries.values()) cb(entry);
  }

  /**
   * 返回到期待销毁的条目（驱逐后已过 DEFER_FRAMES 个 update）并移出墓碑。
   * 调用方（Tileset3D.update 开头）负责真正 remove/destroy Primitive。
   */
  flushDestroyed(frame: number): TileCacheEntry[] {
    const ready: TileCacheEntry[] = [];
    for (const [key, record] of this._graveyard) {
      if (frame - record.frame >= DEFER_FRAMES) {
        this._graveyard.delete(key);
        ready.push(record.entry);
      }
    }
    return ready;
  }

  /** 驱逐容量溢出的最旧条目（Map 头部 = 最旧）。 */
  private _evictOverflow(frame: number): void {
    while (this._entries.size > this._capacity) {
      const oldestKey = this._entries.keys().next().value as string | undefined;
      if (oldestKey === undefined) break;
      const entry = this._entries.get(oldestKey) as TileCacheEntry;
      this._entries.delete(oldestKey);
      this._graveyard.set(oldestKey, { entry, frame });
      this._evictions++;
    }
  }

  /**
   * 终结：清空墓碑并返回全部条目（驻留 + 墓碑），调用方负责销毁
   * Primitive。此后缓存为空（Tileset3D.destroy 专用）。
   */
  destroyAll(): TileCacheEntry[] {
    const all = [...this._entries.values(), ...[...this._graveyard.values()].map((r) => r.entry)];
    this._entries.clear();
    this._graveyard.clear();
    return all;
  }
}
