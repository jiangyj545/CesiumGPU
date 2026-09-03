import { Context } from '../renderer/Context';
import { Primitive, PrimitiveCollection } from './Primitive';
import { PolygonGeometry } from '../geometry/PolygonGeometry';
import { PolylineGeometry } from '../geometry/PolylineGeometry';
import type { InterleavedGeometryBuffers } from '../geometry/GeometryPipeline';
import { Color } from '../math/Color';
import type { Entity } from '../dataSource/Entity';
import { EntityCollection } from '../dataSource/EntityCollection';

/**
 * Host capability the visualizer needs from the Scene.  `Scene` satisfies
 * this structurally; the indirection avoids a circular module dependency.
 */
export interface EntityVisualizerHost {
  readonly primitives: PrimitiveCollection;
  /** Route an opaque primitive through the G-Buffer path. */
  addPrimitive(primitive: Primitive): void;
  /** Route a translucent primitive through the WBOIT transparent path. */
  addTransparentPrimitive(primitive: Primitive): void;
}

/**
 * Polyline pixel width → normalised world width factor.
 *
 * `PolylineGeometry` expects the *full ribbon width in normalised world units*
 * (Earth equatorial radius = 1, so 1 unit ≈ 6378 km and the geometry default
 * 0.001 ≈ 6.4 km).  Entity polylines specify `width` in pixels, and because
 * the ribbon is camera-independent its on-screen width necessarily varies
 * with zoom; this fixed factor is the compromise: 1 px ≈ 64 m of world width
 * (a 2 px line ≈ 128 m on the ground), i.e. the geometry default width at a
 * ~100 px on-screen footprint.  A future iteration may scale this by camera
 * distance each frame to approximate true screen-space width.
 */
const PIXELS_TO_WORLD_WIDTH = 1e-5;

/** Per-entity GPU resources owned by the visualizer. */
interface EntityPrimitives {
  primitives: Primitive[];
  /** Listener registered on the entity's definitionChanged (for removal). */
  definitionListener: (entity: Entity, property: string) => void;
}

/**
 * EntityVisualizer - Bridges `EntityCollection` to the render scene.
 *
 * Rendering routing per visual type:
 * - **polygon** → `PolygonGeometry` → `Primitive`; alpha < 1 goes to the
 *   translucent (WBOIT) queue, otherwise it is opaque and rendered into the
 *   G-Buffer (receives lighting).
 * - **polyline** → `PolylineGeometry` → `Primitive`, always translucent
 *   (ribbons hug the surface and are usually semi-transparent).
 * - **point / label** → NOT rendered here: billboards must keep a constant
 *   pixel size, which world-space meshes cannot express.  They are drawn by
 *   `OverlayPass` in screen space instead.
 *
 * Entity positions and polygon heights are consumed **as-is** in the
 * normalised ECEF convention documented on `Entity` (Earth radius = 1).
 *
 * The visualizer listens to `collectionChanged` (add/remove entities) and to
 * each entity's `definitionChanged` (property edits) and rebuilds only the
 * affected entity's primitives; removed entities have their primitives
 * recycled via `PrimitiveCollection.remove` (which destroys GPU buffers).
 */
export class EntityVisualizer {
  private _context: Context;
  private _entities: EntityCollection;
  private _host: EntityVisualizerHost;
  private _visuals = new Map<Entity, EntityPrimitives>();

  private _collectionListener = (
    added: readonly Entity[],
    removed: readonly Entity[],
    _moved: readonly Entity[]
  ): void => {
    for (const entity of added)   this._track(entity);
    for (const entity of removed) this._untrack(entity);
  };

  constructor(context: Context, entities: EntityCollection, host: EntityVisualizerHost) {
    this._context  = context;
    this._entities = entities;
    this._host     = host;

    entities.collectionChanged.addListener(this._collectionListener);

    // Pick up entities that were added before this visualizer attached.
    for (const entity of entities.values) this._track(entity);
  }

  /** Release all primitives and event listeners. */
  destroy(): void {
    this._entities.collectionChanged.removeCallback(this._collectionListener);
    for (const entity of Array.from(this._visuals.keys())) this._untrack(entity);
  }

  /** Number of entities currently holding GPU primitives (diagnostics). */
  get visualCount(): number {
    return this._visuals.size;
  }

  // ── Private ─────────────────────────────────────────────────────────────────

  /** Subscribe to an entity and build its initial visuals. */
  private _track(entity: Entity): void {
    if (this._visuals.has(entity)) return;

    const primitives: Primitive[] = [];
    const definitionListener = (e: Entity, _property: string): void => {
      this._rebuild(e);
    };
    entity.definitionChanged.addListener(definitionListener);
    this._visuals.set(entity, { primitives, definitionListener });
    this._buildInto(entity, primitives);
  }

  /** Destroy an entity's primitives and detach its listener. */
  private _untrack(entity: Entity): void {
    const entry = this._visuals.get(entity);
    if (!entry) return;
    entity.definitionChanged.removeCallback(entry.definitionListener);
    for (const primitive of entry.primitives) {
      this._host.primitives.remove(primitive); // destroys GPU buffers
    }
    this._visuals.delete(entity);
  }

  /** Rebuild one entity's visuals after a definitionChanged. */
  private _rebuild(entity: Entity): void {
    const entry = this._visuals.get(entity);
    if (!entry) return;
    for (const primitive of entry.primitives) {
      this._host.primitives.remove(primitive);
    }
    entry.primitives.length = 0;
    this._buildInto(entity, entry.primitives);
  }

  /** Build polygon / polyline visuals for an entity into `out`. */
  private _buildInto(entity: Entity, out: Primitive[]): void {
    const polygon = entity.polygon;
    if (polygon?.hierarchy && polygon.hierarchy.positions.length >= 3) {
      const hierarchy = polygon.hierarchy;
      const buffers = PolygonGeometry.fromCartesianPositions(
        hierarchy.positions,
        {
          height: polygon.height ?? 0,
          holes:  hierarchy.holes ?? [],
        }
      );
      this._addPrimitive(buffers, polygon.color ?? Color.CYAN, out);
    }

    const polyline = entity.polyline;
    if (polyline?.positions && polyline.positions.length >= 2) {
      const buffers = PolylineGeometry.fromCartesianPositions(
        polyline.positions,
        { width: (polyline.width ?? 1) * PIXELS_TO_WORLD_WIDTH }
      );
      // Ribbons hug the globe → usually semi-transparent, WBOIT queue.
      this._addPrimitive(buffers, polyline.color ?? Color.WHITE, out, true);
    }
  }

  /** Wrap built buffers in a Primitive and register it with the scene. */
  private _addPrimitive(
    buffers: InterleavedGeometryBuffers,
    color: Color,
    out: Primitive[],
    translucentOverride?: boolean
  ): void {
    // WebGPU forbids zero-sized buffers (degenerate/empty geometry).
    if (buffers.indexCount === 0 || buffers.vertices.byteLength === 0) return;

    const translucent = translucentOverride ?? color.alpha < 1;
    const primitive = new Primitive(this._context, {
      vertices:    buffers.vertices,
      indices:     buffers.indices,
      color,
      translucent,
      alpha:       color.alpha,
    });

    if (translucent) {
      this._host.addTransparentPrimitive(primitive);
    } else {
      this._host.addPrimitive(primitive);
    }
    out.push(primitive);
  }
}

export default EntityVisualizer;
