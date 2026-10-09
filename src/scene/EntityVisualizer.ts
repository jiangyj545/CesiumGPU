import { Context } from '../renderer/Context';
import { Primitive, PrimitiveCollection } from './Primitive';
import { PolygonGeometry } from '../geometry/PolygonGeometry';
import { PolylineGeometry } from '../geometry/PolylineGeometry';
import type { InterleavedGeometryBuffers } from '../geometry/GeometryPipeline';
import { Color } from '../math/Color';
import { Cartesian3 } from '../math/Cartesian3';
import type { Entity } from '../dataSource/Entity';
import { EntityCollection } from '../dataSource/EntityCollection';
import { CylinderGeometry } from '../geometry/CylinderGeometry';
import { composeEntityFrameMatrix } from './EntityModelMatrix';
import { isProperty, resolveProperty } from '../dataSource/CallbackProperty';
import { applyHeightReference, depthTestDisabled } from '../dataSource/BillboardStyle';
import { Model } from './Model';
import { composeEntityModelMatrix, resolveMinimumPixelScale } from './EntityModelMatrix';
import { FlowArrowPolyline } from '../effect/FlowArrowPolyline';
import type { Camera } from './Camera';
import { PathHistory, sampleLeadPositions } from './PathHistory';

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
  /** Camera used for `model.minimumPixelSize` distance scaling. */
  readonly camera: { position: Cartesian3; frustum: { fov: number } };
  /** CSS-pixel viewport height for `minimumPixelSize`. */
  readonly viewportHeightPx: number;
  /** Terrain height sampler for entity HeightReference (P2-4); optional. */
  readonly sampleTerrainHeight?: (lon: number, lat: number) => number;
  /** EffectManager-shaped mount point for FlowArrow polylines (P2-6); optional. */
  readonly effects?: { add(effect: unknown): unknown; remove(effect: unknown): boolean };
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

/** Rebuild threshold for path polyline updates (normalised units, ~6 cm). */
const PATH_EPSILON = 1e-8;

/** Per-entity GPU resources owned by the visualizer. */
interface EntityPrimitives {
  primitives: Primitive[];
  /** Listener registered on the entity's definitionChanged (for removal). */
  definitionListener: (entity: Entity, property: string) => void;
  /** Loaded glTF model (entity.model), async-loaded once. */
  model: Model | null;
  /** Model URI currently loading (guards against duplicate loads). */
  modelLoadingUri: string | null;
  /** Trail sampler for entity.path. */
  history: PathHistory;
  /** Last trail polyline primitive (rebuilt when the buffer changes). */
  pathPrimitive: Primitive | null;
  /** Last sampled polyline positions (dynamic Property change detection). */
  lastPolylinePositions: Cartesian3[] | null;
  /** Polyline primitives only (disableDepthTestDistance per-frame flag). */
  polylinePrimitives: Primitive[];
  /** FlowArrow 特效实例 + 版本（P2-6；material.type==='flow' 时启用）。 */
  flowArrow: FlowArrowPolyline | null;
  flowArrowVersion: number;
  /** Last rebuilt path point count / newest point (rebuild detection). */
  lastPathCount: number;
  lastPathEnd: Cartesian3;
  /** Cylinder primitive + last rebuild signature (dynamic radius/length). */
  cylinderPrimitive: Primitive | null;
  cylinderSignature: string | null;
  cylinderLastPos: Cartesian3 | null;
  cylinderLastOrient: string | null;
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
 * - **model** → `Model.fromGltfAsync` loaded once, then re-transformed every
 *   frame from `entity.position` + `entity.orientation` (see
 *   `composeEntityModelMatrix`).
 * - **path** → past positions kept in a `PathHistory` ring buffer
 *   (`trailTime` window) plus lead samples evaluated from a dynamic position
 *   Property (`leadTime`), rendered as a translucent polyline.
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
 * Non-constant Properties (`CallbackProperty`) are polled once per frame in
 * {@link EntityVisualizer.update}, which `Scene.render` invokes with the
 * frame clock.
 */
export class EntityVisualizer {
  /** True when the last update() mutated GPU state（P2-5 按需渲染脏源）。 */
  private _changed = true;

  /** Consume（读并清零）「上次 update 后场景内容有变」标记（P2-5 脏源）。 */
  consumeChange(): boolean {
    const changed = this._changed;
    this._changed = false;
    return changed;
  }

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

  /**
   * Per-frame dynamic update: poll non-constant Properties (position /
   * orientation / polyline.positions), advance model transforms and path
   * trail buffers. `Scene.render` calls this once per frame with the frame
   * time in seconds.
   */
  update(nowSec: number): void {
    // 脏标记不在此清零：由 consumeChange() 消费时清零，保证 update 窗口外
    //（definitionChanged → _rebuild / collectionChanged）的变更不被吞掉。
    for (const [entity, entry] of this._visuals) {
      const raw = resolveProperty(entity.position);
      // P2-5：带动态 Property 的实体视为持续变化（OverlayPass 的
      // point/label 每帧消费解析值，无 visualizer 侧突变点可捕）。
      if (
        isProperty(entity.position) ||
        isProperty(entity.orientation) ||
        isProperty(entity.label?.text)
      ) {
        this._changed = true;
      }
      // 全部动态消费方（billboard 走 OverlayPass 自己的同款逻辑）。
      const position = raw
        ? applyHeightReference(
            raw,
            entity.heightReference,
            this._host.sampleTerrainHeight
              ? { sampleHeight: this._host.sampleTerrainHeight }
              : null
          )
        : undefined;

      // Dynamic polyline positions → rebuild geometry when the sample changes.
      const positions = resolveProperty(entity.polyline?.positions);
      if (positions && positions !== entry.lastPolylinePositions) {
        if (isProperty(entity.polyline?.positions)) {
          entry.lastPolylinePositions = positions;
          this._rebuild(entity);
          this._changed = true;
        } else {
          entry.lastPolylinePositions = positions;
        }
      }

      this._updateModel(entity, entry, position, nowSec);
      this._updatePath(entity, entry, position, nowSec);
      this._updateCylinder(entity, entry, position);
      this._updateFlowArrow(entity, entry);

      // P2-4：polyline disableDepthTestDistance（穿深）+ distanceDisplayCondition
      //（窗外 prim.show=false）——同款相机距离轮询。
      const polylineDdc = resolveProperty(entity.polyline?.distanceDisplayCondition);
      const disableDepth = entity.polyline?.disableDepthTestDistance;
      if ((polylineDdc || disableDepth !== undefined) && position) {
        const dist = Cartesian3.distance(this._host.camera.position, position) * 6378137.0;
        const disabled = depthTestDisabled(disableDepth, dist);
        for (const prim of entry.polylinePrimitives) {
          prim.alwaysPassDepth = disabled;
          if (polylineDdc) prim.show = polylineDdc.isVisible(dist);
        }
        if (polylineDdc) this._changed = true; // 门控态随距离翻转，需重绘
      }
    }
  }

  // ── Private ─────────────────────────────────────────────────────────────────

  /** Subscribe to an entity and build its initial visuals. */
  private _track(entity: Entity): void {
    if (this._visuals.has(entity)) return;

    const primitives: Primitive[] = [];
    const definitionListener = (e: Entity, _property: string): void => {
      this._rebuild(e);
    };
    const entry: EntityPrimitives = {
      primitives,
      definitionListener,
      model: null,
      modelLoadingUri: null,
      history: new PathHistory(),
      pathPrimitive: null,
      lastPolylinePositions: null,
      lastPathCount: -1,
      lastPathEnd: new Cartesian3(),
      cylinderPrimitive: null,
      cylinderSignature: null,
      cylinderLastPos: null,
      cylinderLastOrient: null,
      polylinePrimitives: [],
      flowArrow: null,
      flowArrowVersion: -1,
    };
    entity.definitionChanged.addListener(definitionListener);
    this._visuals.set(entity, entry);
    this._buildInto(entity, primitives, entry.polylinePrimitives);
    entry.lastPolylinePositions = resolveProperty(entity.polyline?.positions) ?? null;
  }

  /** Destroy an entity's primitives and detach its listener. */
  private _untrack(entity: Entity): void {
    const entry = this._visuals.get(entity);
    if (!entry) return;
    entity.definitionChanged.removeCallback(entry.definitionListener);
    for (const primitive of entry.primitives) {
      this._host.primitives.remove(primitive); // destroys GPU buffers
    }
    entry.model?.destroy();
    if (entry.flowArrow) {
      this._host.effects?.remove(entry.flowArrow);
      entry.flowArrow.destroy();
      entry.flowArrow = null;
    }
    if (entry.pathPrimitive) this._host.primitives.remove(entry.pathPrimitive);
    this._visuals.delete(entity);
    this._changed = true;
  }

  /** Rebuild one entity's visuals after a definitionChanged. */
  private _rebuild(entity: Entity): void {
    const entry = this._visuals.get(entity);
    if (!entry) return;
    for (const primitive of entry.primitives) {
      this._host.primitives.remove(primitive);
    }
    entry.primitives.length = 0;
    entry.polylinePrimitives.length = 0;
    this._buildInto(entity, entry.primitives, entry.polylinePrimitives);
    this._changed = true;
  }

  /** Build polygon / polyline visuals for an entity into `out`. */
  private _buildInto(entity: Entity, out: Primitive[], outPolyline: Primitive[] = []): void {
    const polygon = entity.polygon;
    const hierarchy = resolveProperty(polygon?.hierarchy);
    if (polygon && hierarchy && hierarchy.positions.length >= 3) {
      const buffers = PolygonGeometry.fromCartesianPositions(
        hierarchy.positions,
        {
          height: polygon.height ?? 0,
          holes:  hierarchy.holes ?? [],
        }
      );
      this._addPrimitive(buffers, polygon.color ?? Color.CYAN, out);
    }

    const positions = resolveProperty(entity.polyline?.positions);
    if (positions && positions.length >= 2) {
      const width = resolveProperty(entity.polyline?.width) ?? 1;
      const buffers = PolylineGeometry.fromCartesianPositions(
        positions,
        { width: width * PIXELS_TO_WORLD_WIDTH }
      );
      // Ribbons hug the globe → usually semi-transparent, WBOIT queue.
      const prim = this._addPrimitive(buffers, entity.polyline?.color ?? Color.WHITE, out, true, outPolyline);
      void prim;
    }
  }


  /** Wrap built buffers in a Primitive and register it with the scene. */
  private _addPrimitive(
    buffers: InterleavedGeometryBuffers,
    color: Color,
    out: Primitive[],
    translucentOverride?: boolean,
    outPolyline?: Primitive[]
  ): Primitive | null {
    // WebGPU forbids zero-sized buffers (degenerate/empty geometry).
    if (buffers.indexCount === 0 || buffers.vertices.byteLength === 0) return null;

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
    outPolyline?.push(primitive);
    return primitive;
  }

  /**
   * Load `entity.model` once and refresh its world transform from the
   * resolved position / orientation (minimum-pixel-size aware).
   */
  private _updateModel(
    entity: Entity,
    entry: EntityPrimitives,
    position: Cartesian3 | undefined,
    _nowSec: number
  ): void {
    const model = entity.model;
    if (!model) {
      // Model removed since last frame → release GPU resources.
      if (entry.model) {
        entry.model.destroy();
        entry.model = null;
        entry.modelLoadingUri = null;
      }
      return;
    }
    if (!entry.model && entry.modelLoadingUri !== model.uri) {
      entry.modelLoadingUri = model.uri;
      Model.fromGltfAsync({
        url: model.uri,
        scene: this._host,
        scale: model.scale ?? 1.0,
      })
        .then((loaded) => {
          // Entity may have been untracked / model swapped while loading.
          if (this._visuals.get(entity) !== entry || entity.model?.uri !== model.uri) {
            loaded.destroy();
            return;
          }
          entry.model = loaded;
          entry.modelLoadingUri = null;
          this._changed = true;
        })
        .catch((err: unknown) => {
          console.error('[EntityVisualizer] model load failed:', model.uri, err);
          entry.modelLoadingUri = null;
        });
      return;
    }
    if (!entry.model || !position) return;

    const orientation = resolveProperty(entity.orientation);
    const distanceMetres =
      Cartesian3.distance(this._host.camera.position, position) * 6378137.0;
    const scale = resolveMinimumPixelScale(
      model.scale ?? 1.0,
      model.minimumPixelSize,
      distanceMetres,
      this._host.camera.frustum.fov,
      this._host.viewportHeightPx
    );
    entry.model.modelMatrix = composeEntityModelMatrix(position, orientation, scale);
    entry.model.applyModelMatrix();
    entry.model.show = entity.show;
    this._changed = true;
  }

  /**
   * Sample the entity position into the trail buffer and rebuild the path
   * polyline when the (trail + lead) point set changes.
   */
  private _updatePath(
    entity: Entity,
    entry: EntityPrimitives,
    position: Cartesian3 | undefined,
    nowSec: number
  ): void {
    const path = entity.path;
    if (!path || !position) {
      if (entry.pathPrimitive && !path) {
        this._host.primitives.remove(entry.pathPrimitive);
        entry.pathPrimitive = null;
        entry.history.clear();
      }
      return;
    }

    entry.history.push(nowSec, position);
    const trail = entry.history.trail(path.trailTime ?? Infinity, nowSec);
    const points: Cartesian3[] = trail.map((s) => s.position);
    // Lead segment (future positions) from a dynamic position Property.
    if (path.leadTime && isProperty(entity.position)) {
      points.push(...sampleLeadPositions(entity.position, nowSec, path.leadTime, path.resolution));
    }

    // Rebuild only when the sample count or the newest point moved.
    const newest = points[points.length - 1];
    const changed =
      !entry.pathPrimitive ||
      entry.lastPathCount !== points.length ||
      Cartesian3.distance(newest, entry.lastPathEnd) > PATH_EPSILON;
    if (points.length >= 2 && changed) {
      if (entry.pathPrimitive) this._host.primitives.remove(entry.pathPrimitive);
      const buffers = PolylineGeometry.fromCartesianPositions(points, {
        width: (path.width ?? 2) * PIXELS_TO_WORLD_WIDTH,
      });
      if (buffers.indexCount > 0 && buffers.vertices.byteLength > 0) {
        const color = path.color ?? Color.YELLOW;
        const primitive = new Primitive(this._context, {
          vertices: buffers.vertices,
          indices: buffers.indices,
          color,
          translucent: true,
          alpha: color.alpha,
        });
        this._host.addTransparentPrimitive(primitive);
        entry.pathPrimitive = primitive;
        entry.lastPathCount = points.length;
        entry.lastPathEnd = new Cartesian3(newest.x, newest.y, newest.z);
        this._changed = true;
      }
    }

    // P2-4 打回：path distanceDisplayCondition —— 窗外 prim.show=false。
    // 放在重建之后：新构建的 prim 同帧即被门控（首帧窗外即隐藏）。
    const pathDdc = resolveProperty(entity.path?.distanceDisplayCondition);
    if (entry.pathPrimitive && pathDdc) {
      const dist = Cartesian3.distance(this._host.camera.position, position) * 6378137.0;
      const visible = pathDdc.isVisible(dist);
      if (entry.pathPrimitive.show !== visible) this._changed = true;
      entry.pathPrimitive.show = visible;
    }
  }


  /**
   * FlowArrow 流动箭头线（P2-6）：material.type === 'flow' 时经 EffectManager
   * 挂载 FlowArrowPolyline；每帧 setPositions（动态 Property 引用变化即
   * 重建），重建（version 变化）置 P2-5 脏标记。
   */
  private _updateFlowArrow(entity: Entity, entry: EntityPrimitives): void {
    const material = entity.polyline?.material;
    const isFlow = material?.type === 'flow';

    if (!isFlow || !this._host.effects) {
      if (entry.flowArrow) {
        this._host.effects?.remove(entry.flowArrow);
        entry.flowArrow.destroy();
        entry.flowArrow = null;
        entry.flowArrowVersion = -1;
        this._changed = true;
      }
      return;
    }

    if (!entry.flowArrow) {
      const flow = new FlowArrowPolyline(this._context, {
        material,
        width: resolveProperty(entity.polyline?.width) ?? 6,
      });
      flow.setCamera(this._host.camera as unknown as Camera);
      this._host.effects.add(flow);
      entry.flowArrow = flow;
      entry.flowArrowVersion = flow.version;
      this._changed = true;
    }

    const positions = resolveProperty(entity.polyline?.positions);
    if (positions) entry.flowArrow.setPositions(positions);
    entry.flowArrow.show = entity.show;
    if (entry.flowArrow.version !== entry.flowArrowVersion) {
      entry.flowArrowVersion = entry.flowArrow.version;
      this._changed = true; // 几何重建（动态 positions）→ 按需渲染脏源
    }
  }

  /**
   * Cylinder / cone graphics: rebuild the geometry only when the resolved
   * length / radii signature changes (dynamic Property polling); the world
   * matrix follows position + orientation every frame (no rebuild).
   */
  private _updateCylinder(
    entity: Entity,
    entry: EntityPrimitives,
    position: Cartesian3 | undefined
  ): void {
    const cylinder = entity.cylinder;
    if (!cylinder || !position) {
      if (entry.cylinderPrimitive) {
        this._host.primitives.remove(entry.cylinderPrimitive);
        entry.cylinderPrimitive = null;
        entry.cylinderSignature = null;
      }
      return;
    }

    const length = resolveProperty(cylinder.length) ?? 0;
    const topRadius = resolveProperty(cylinder.topRadius) ?? 0;
    const bottomRadius = resolveProperty(cylinder.bottomRadius) ?? 0;
    if (length <= 0) return;

    const signature = CylinderGeometry.signature({ length, topRadius, bottomRadius });
    if (signature !== entry.cylinderSignature) {
      if (entry.cylinderPrimitive) this._host.primitives.remove(entry.cylinderPrimitive);
      const buffers = CylinderGeometry.createGeometry({ length, topRadius, bottomRadius });
      if (buffers.indexCount === 0 || buffers.vertices.byteLength === 0) return;
      const color = cylinder.color ?? cylinder.material ?? Color.CYAN;
      const primitive = new Primitive(this._context, {
        vertices: buffers.vertices,
        indices: buffers.indices,
        color,
        translucent: true,
        alpha: color.alpha,
      });
      this._host.addTransparentPrimitive(primitive);
      entry.cylinderPrimitive = primitive;
      entry.cylinderSignature = signature;
      this._changed = true;
    }

    // Transform follows position / orientation every frame (cheap, no rebuild).
    const orientation = resolveProperty(entity.orientation);
    const orientKey = orientation
      ? `${orientation.x},${orientation.y},${orientation.z},${orientation.w}`
      : '';
    const posMoved =
      !entry.cylinderLastPos ||
      Cartesian3.distance(position, entry.cylinderLastPos) > 0 ||
      orientKey !== entry.cylinderLastOrient;
    if (entry.cylinderPrimitive && posMoved) {
      entry.cylinderPrimitive.modelMatrix = composeEntityFrameMatrix(position, orientation);
      entry.cylinderLastPos = new Cartesian3(position.x, position.y, position.z);
      entry.cylinderLastOrient = orientKey;
      this._changed = true;
    }

    // P2-4 打回：cylinder distanceDisplayCondition —— 窗外 prim.show=false。
    const cylinderDdc = resolveProperty(entity.cylinder?.distanceDisplayCondition);
    if (entry.cylinderPrimitive && cylinderDdc) {
      const dist = Cartesian3.distance(this._host.camera.position, position) * 6378137.0;
      const visible = cylinderDdc.isVisible(dist);
      if (entry.cylinderPrimitive.show !== visible) this._changed = true;
      entry.cylinderPrimitive.show = visible;
    }
  }
}

export default EntityVisualizer;
