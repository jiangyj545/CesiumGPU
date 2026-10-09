import { allocatePickInstanceIds } from '../picking/InstanceIdPool';
import type { Primitive } from '../scene/Primitive';

/**
 * Cesium3DTileFeature - per-feature view over a 3D Tiles batch table,
 * mirroring Cesium's `Cesium3DTileFeature` (the subset consumed by
 * hexapod_gcs building-picker: `instanceof` check + `getProperty` /
 * `batchId` / `tileset` duck-typing).
 *
 * Created by the tileset pick registry; exposed through `PickResult.feature`.
 */
export class Cesium3DTileFeature {
  /**
   * @param tileset   owning Tileset3D (duck-typing parity with Cesium)
   * @param batchId   batch index, or `undefined` for tile-level features
   *                  (content without `_BATCHID`)
   * @param table     expanded batch table (`name → value per batch`), or null
   * @param primitive the content primitive the feature belongs to
   */
  constructor(
    readonly tileset: unknown,
    readonly batchId: number | undefined,
    readonly table: Record<string, unknown[]> | null,
    readonly primitive?: Primitive
  ) {}

  /** True when the batch table defines the property. */
  hasProperty(name: string): boolean {
    return this.table !== null && Object.prototype.hasOwnProperty.call(this.table, name);
  }

  /**
   * Batch table property value for this feature's batch, or `undefined`
   * (tile-level features / missing properties / missing batch table).
   */
  getProperty(name: string): unknown {
    if (this.batchId === undefined || this.table === null) return undefined;
    const column = this.table[name];
    if (!Array.isArray(column)) return undefined;
    return column[this.batchId];
  }

  /** All batch table property names (empty for tile-level features). */
  getPropertyNames(): string[] {
    return this.table ? Object.keys(this.table) : [];
  }
}

/** bit23 of the pick id marks a per-instance (batch) pick. */
export const PICK_INSTANCE_FLAG = 0x800000;

/** Mask extracting the instance id payload from a pick id. */
export const PICK_INSTANCE_MASK = 0x7fffff;

/** Per-tile instance range registration (one per content primitive). */
interface InstanceRange {
  base: number;
  count: number;
  feature: Cesium3DTileFeature;
  primitive: Primitive;
}

/** Decode an instance pick id; null when the id is not an instance pick. */
export function decodePickInstanceId(id: number): number | null {
  if ((id & PICK_INSTANCE_FLAG) === 0) return null;
  return id & PICK_INSTANCE_MASK;
}

/**
 * TilesetPickRegistry - maps pick instance ids to tileset features.
 *
 * b3dm `_BATCHID` values are tile-local (0..count-1) but the pick id channel
 * carries a single global number, so every tile's batch range is offset by a
 * globally allocated `base` (`buildTileInstanceIds`). The registry owns the
 * allocator and the instance-id → feature lookup used by PickPass decode.
 * Pure data structure (no GPU) — unit-testable in node.
 */
export class TilesetPickRegistry {
  private _ranges: InstanceRange[] = [];
  private _tileFeatures = new Map<Primitive, Cesium3DTileFeature>();

  /**
   * Reserve the next `count` instance ids and return their base. Batch ids
   * are stored as `base + batchId + 1`, so a range covers `count + 1` slots.
   * Allocation goes through the **process-wide** 23-bit pool
   * (`picking/InstanceIdPool.ts`) shared with `CityBlockBuilder`, keeping
   * the F12 building-instance and tile-batch id spaces disjoint — a decoded
   * instance id can only ever belong to one producer.
   */
  allocate(count: number): number {
    return allocatePickInstanceIds(count + 1);
  }

  /**
   * Register a primitive whose vertices carry instance ids in
   * `[base + 1, base + count]` (see {@link buildTileInstanceIds}).
   */
  registerInstanceRange(
    primitive: Primitive,
    base: number,
    count: number,
    feature: Cesium3DTileFeature
  ): void {
    this._ranges.push({ base, count, feature, primitive });
  }

  /** Register a tile-level feature for a primitive without batch ids. */
  registerTilePrimitive(primitive: Primitive, feature: Cesium3DTileFeature): void {
    this._tileFeatures.set(primitive, feature);
  }

  /** Drop every registration referencing a destroyed feature's primitive. */
  unregister(primitive: Primitive): void {
    this._tileFeatures.delete(primitive);
    this._ranges = this._ranges.filter((r) => r.primitive !== primitive);
  }

  /** Resolve a global instance id to its feature (null = not a tile batch). */
  resolve(instanceId: number): Cesium3DTileFeature | null {
    for (const range of this._ranges) {
      if (instanceId > range.base && instanceId <= range.base + range.count) {
        const batchId = instanceId - range.base - 1;
        return new Cesium3DTileFeature(
          range.feature.tileset,
          batchId,
          range.feature.table,
          range.primitive
        );
      }
    }
    return null;
  }

  /** Tile-level feature for a primitive pick (no batch ids), or null. */
  featureForPrimitive(primitive: Primitive | null): Cesium3DTileFeature | null {
    if (!primitive) return null;
    return this._tileFeatures.get(primitive) ?? null;
  }

}

/**
 * Process-wide default registry consumed by PickPass. Tileset3D registers
 * its content primitives here as tiles load / evict.
 */
export const tilesetPickRegistry = new TilesetPickRegistry();

/**
 * Offset tile-local batch ids into the global instance id space:
 * `result[i] = base + batchId + 1` (0 stays reserved as "no instance").
 */
export function buildTileInstanceIds(batchIds: Uint32Array, base: number): Uint32Array {
  const out = new Uint32Array(batchIds.length);
  for (let i = 0; i < batchIds.length; i++) out[i] = base + batchIds[i] + 1;
  return out;
}
