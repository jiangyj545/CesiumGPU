/**
 * TileTextureCache - LRU pool of per-tile GPU textures (spike version).
 *
 * Each imagery tile owns an independent GPUTexture (texture-per-tile), so
 * the pool only tracks residency: on eviction the texture is queued for
 * deferred destruction (see {@link TileTextureCache.onFrameEnd}) — a texture
 * may still be referenced by in-flight command buffers, and WebGPU forbids
 * destroying a resource before the submit that uses it has been enqueued.
 *
 * Textures stored here should be created with
 * `TEXTURE_BINDING | COPY_DST | RENDER_ATTACHMENT` usage so they can be
 * mipmapped (see MipmapGenerator).
 */

/** Runtime counters exposed via {@link TileTextureCache.stats}. */
export interface TileTextureCacheStats {
  /** Maximum number of textures kept resident. */
  capacity: number;
  /** Textures currently resident. */
  size: number;
  /** `get()` calls resolved from the pool. */
  hits: number;
  /** `get()` calls that missed. */
  misses: number;
  /** Textures evicted (and destroyed) due to capacity pressure. */
  evictions: number;
  /** hits / (hits + misses), 0 when no lookups yet. */
  hitRate: number;
}

export interface TileTextureCacheOptions {
  /** Maximum resident textures. Default: 200 */
  capacity?: number;
}

/** @internal Texture queued for deferred destruction. */
interface PendingDestroy {
  texture: GPUTexture;
  /** Frame index at enqueue time; destroyed once `current - frame >= DEFER`. */
  frame: number;
}

/**
 * Tiles at or above this zoom level never participate in LRU eviction.
 * Coarse tiles form the upsampling fallback chain (`ImageryLayer.resolveTexture`
 * walks ancestors while a leaf loads); evicting one mid-chain nulls every
 * descendant's fallback at once → primitive destroyed → base-colour hole.
 * The coarse levels are small and bounded (L2-L4 ≈ 336 tiles total), so
 * pinning them costs little and removes the holes.
 */
const PIN_LEVEL = 4;

/**
 * Frames between enqueue and actual `destroy()`. Command buffers referencing
 * an evicted texture complete within 1-2 frames; 2 is conservative.
 */
const DEFERRED_DESTROY_FRAMES = 2;

export class TileTextureCache {
  /** Insertion-ordered map: oldest (LRU) entry first, newest last. */
  private _entries = new Map<string, GPUTexture>();
  private _capacity: number;

  /** Evicted textures waiting out the in-flight command buffer grace window. */
  private _pendingDestroy: PendingDestroy[] = [];
  /**
   * Runtime-pinned keys (imagery textures bound to LIVE terrain primitives).
   * Evicting such a texture would destroy it under the referencing bind
   * group → the terrain tile falls back to a coarse ancestor → re-request →
   * upgrade → destroy/rebuild cycle: the visible clear↔blur FLICKER of
   * terrain draping. Pins are key-level and released by the referrer
   * (see Globe._syncTerrainPrimitives) when its primitive is destroyed.
   */
  private _pins = new Set<string>();
  private _frameIndex = 0;

  private _hits = 0;
  private _misses = 0;
  private _evictions = 0;

  constructor(options: TileTextureCacheOptions = {}) {
    this._capacity = Math.max(1, options.capacity ?? 200);
  }

  get capacity(): number { return this._capacity; }
  get size(): number { return this._entries.size; }

  /** Aggregate counters (hits / misses / evictions / ...). */
  get stats(): TileTextureCacheStats {
    const lookups = this._hits + this._misses;
    return {
      capacity:  this._capacity,
      size:      this._entries.size,
      hits:      this._hits,
      misses:    this._misses,
      evictions: this._evictions,
      hitRate:   lookups === 0 ? 0 : this._hits / lookups,
    };
  }

  /**
   * Fetch the texture for `key`. A hit refreshes its recency.
   * Returns `null` on miss (caller should load and `put()` the tile).
   */
  get(key: string): GPUTexture | null {
    const texture = this._entries.get(key);
    if (texture === undefined) {
      this._misses++;
      return null;
    }
    // Re-insert to move the entry to the end (most recently used)
    this._entries.delete(key);
    this._entries.set(key, texture);
    this._hits++;
    return texture;
  }

  /** Residency check without touching recency or counters. */
  has(key: string): boolean {
    return this._entries.has(key);
  }

  /**
   * Insert (or refresh) the texture for `key`, evicting the oldest entry
   * (queued for deferred destruction) while over capacity.
   */
  put(key: string, texture: GPUTexture): void {
    const existing = this._entries.get(key);
    if (existing !== undefined && existing !== texture) {
      // Replacement: the old texture is no longer referenced
      this._deferDestroy(existing);
      this._entries.delete(key);
    }
    this._entries.set(key, texture);
    this._evictOldest();
  }

  /** Remove the texture for `key` (deferred destruction). Returns false when absent. */
  delete(key: string): boolean {
    const texture = this._entries.get(key);
    if (texture === undefined) return false;
    this._entries.delete(key);
    this._deferDestroy(texture);
    return true;
  }

  /** Destroy and drop every resident texture (immediate; teardown only). */
  clear(): void {
    for (const texture of this._entries.values()) {
      texture.destroy();
    }
    this._entries.clear();
    this._pins.clear();
    for (const item of this._pendingDestroy) item.texture.destroy();
    this._pendingDestroy.length = 0;
  }

  /**
   * Pin a RESIDENT key against LRU eviction (imagery bound to a live terrain
   * primitive — see `_pins`). Returns false when the key is not resident
   * (nothing to pin; the caller treats it as unpinned/unbound).
   */
  pin(key: string): boolean {
    if (!this._entries.has(key)) return false;
    this._pins.add(key);
    return true;
  }

  /** Release a pin when the referencing primitive is destroyed/replaced. */
  unpin(key: string): void {
    this._pins.delete(key);
  }

  /**
   * Advance the frame counter and destroy queued textures whose grace window
   * has elapsed. Call once per frame (after the previous frame's submit);
   * {@link ImageryLayer.update} invokes this at the start of each frame.
   */
  onFrameEnd(): void {
    this._frameIndex++;
    const deadline = this._frameIndex - DEFERRED_DESTROY_FRAMES;
    let write = 0;
    for (let read = 0; read < this._pendingDestroy.length; read++) {
      const item = this._pendingDestroy[read];
      if (item.frame <= deadline) {
        item.texture.destroy();
      } else {
        this._pendingDestroy[write++] = item;
      }
    }
    this._pendingDestroy.length = write;
  }

  /** Evict least-recently-used entries until within capacity. */
  private _evictOldest(): void {
    // Pinned entries are budget-exempt: the capacity bounds evictable tiles,
    // so a fully-pinned coarse set can never starve fine tile insertion.
    let pinned = 0;
    for (const key of this._entries.keys()) if (this._isPinned(key)) pinned++;
    const effectiveCapacity = this._capacity + pinned;
    if (this._entries.size <= effectiveCapacity) return;
    // Only evictable (unpinned) keys may be removed; pinned coarse tiles
    // stay resident regardless of recency (see PIN_LEVEL).
    for (const oldestKey of [...this._entries.keys()]) {
      if (this._entries.size <= effectiveCapacity) break;
      if (this._isPinned(oldestKey)) continue;
      const oldest = this._entries.get(oldestKey);
      this._entries.delete(oldestKey);
      if (oldest !== undefined) this._deferDestroy(oldest);
      this._evictions++;
    }
  }

  /** Coarse (fallback-chain) tiles and runtime pins are exempt from eviction. */
  private _isPinned(key: string): boolean {
    if (this._pins.has(key)) return true;
    // Keys are "{level}/{x}/{y}" — the leading number is the zoom level.
    const slash = key.indexOf('/');
    if (slash <= 0) return false;
    const level = Number(key.slice(0, slash));
    return Number.isFinite(level) && level <= PIN_LEVEL;
  }

  /**
   * Queue `texture` for destruction instead of destroying it inline: the
   * texture may still be referenced by bind groups used in the current or an
   * in-flight command buffer, and destroying it before those submits would
   * invalidate them ("Destroyed texture used in a submit" → black frame).
   */
  private _deferDestroy(texture: GPUTexture): void {
    this._pendingDestroy.push({ texture, frame: this._frameIndex });
  }
}

export default TileTextureCache;
