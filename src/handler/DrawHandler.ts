import { ScreenSpaceEventHandler } from '../core/ScreenSpaceEventHandler';
import type { ScreenSpacePosition } from '../core/ScreenSpaceEventHandler';
import { Viewer } from '../core/Viewer';
import { Entity } from '../dataSource/Entity';
import { Event } from '../dataSource/Event';
import { Cartesian3 } from '../math/Cartesian3';
import { Cartographic } from '../math/Cartographic';
import { Color } from '../math/Color';
import { Measure } from '../analysis/Measure';

/**
 * Supported draw types (semantically aligned with EV-Globe's `EV_DrawHandler`).
 */
export type DrawType = 'point' | 'polyline' | 'polygon' | 'rectangle';

/** Payload raised by {@link DrawHandler.drawEnd}. */
export interface DrawEndEvent {
  /** Vertices of the finished shape (normalised ECEF, no duplicated closing vertex). */
  positions: Cartesian3[];
  /** The final entity that was added to `scene.entities`. */
  entity: Entity;
}

/** Preview stroke: translucent green. */
const PREVIEW_LINE_COLOR = new Color(0, 1, 0, 0.6);
/** Preview fill: translucent green. */
const PREVIEW_FILL_COLOR = new Color(0, 1, 0, 0.2);
/** Finished shapes use the engine default blue (Cesium `CORNFLOWERBLUE`). */
const FINAL_LINE_COLOR = Color.CORNFLOWERBLUE;
const FINAL_FILL_COLOR = new Color(0.392, 0.584, 0.929, 0.6);

const PREVIEW_NAME = 'draw-preview';
const PREVIEW_WIDTH = 2;
/** Two vertices closer than this (normalised units, ~6 mm) count as coincident. */
const COINCIDENT_EPSILON = 1e-9;

/**
 * DrawHandler - interactive shape drawing on the globe (T2.5), semantically
 * aligned with EV-Globe's `EV_DrawHandler`.
 *
 * Interaction:
 * - `point`     : LEFT_CLICK picks a ground point → immediately raises `drawEnd`.
 * - `polyline`  : LEFT_CLICK appends vertices; MOUSE_MOVE previews the rubber
 *                 band; RIGHT_CLICK / double-click finishes (>= 2 vertices).
 * - `polygon`   : as polyline but the preview closes the ring and finishing
 *                 requires >= 3 vertices.
 * - `rectangle` : two LEFT_CLICKs (opposite corners) → a 4-corner polygon with
 *                 lat/lon aligned edges.
 *
 * While drawing, `viewer.drawMode` is set so the Viewer suppresses its
 * orbit/tilt/zoom-drag camera handlers; wheel and pinch zoom stay active.
 *
 * ```typescript
 * const draw = new DrawHandler(viewer);
 * draw.drawEnd.addListener(({ positions, entity }) => console.log(entity.id, positions.length));
 * draw.start('polygon');
 * // ... RIGHT_CLICK finishes; draw.start() again to switch type, draw.stop() exits.
 * ```
 */
export class DrawHandler {
  /** Raised when a draw mode is entered, with the new {@link DrawType}. */
  readonly drawStart: Event<[DrawType]> = new Event();
  /** Raised when a shape is completed and added to `scene.entities`. */
  readonly drawEnd: Event<[DrawEndEvent]> = new Event();

  private readonly _viewer: Viewer;
  /**
   * Dedicated input channel sharing the viewer canvas. `ScreenSpaceEventHandler`
   * keeps one action per event type *per instance*, so this handler coexists
   * with the Viewer's camera handler without clobbering it.
   */
  private readonly _input: ScreenSpaceEventHandler;
  private _type: DrawType | null = null;
  private _lastType: DrawType | null = null;
  /** Committed vertices of the shape currently being drawn (normalised ECEF). */
  private _positions: Cartesian3[] = [];
  /** Transient preview entity (removed on completion / stop). */
  private _preview: Entity | null = null;
  /** First corner of the rectangle being drawn. */
  private _firstCorner: Cartographic | null = null;

  constructor(viewer: Viewer) {
    this._viewer = viewer;
    this._input = new ScreenSpaceEventHandler(viewer.canvas);
  }

  /** The draw mode currently active, or `null` when idle. */
  get activeType(): DrawType | null {
    return this._type;
  }

  /**
   * Enter a draw mode. Calling while active first stops the current draft.
   * The handler stays in the mode after each finished shape so consecutive
   * shapes can be drawn without re-calling {@link start}.
   */
  start(type: DrawType): void {
    this.stop();
    this._type = type;
    this._lastType = type;
    this._resetDraft();
    this._viewer.drawMode = true;

    this._input.setInputAction((pos) => { void this._onLeftClick(pos); }, 'LEFT_CLICK');
    this._input.setInputAction((pos) => this._onMouseMove(pos), 'MOUSE_MOVE');
    this._input.setInputAction(() => this._finish(), 'RIGHT_CLICK');
    this._input.setInputAction(() => this._finish(), 'LEFT_DOUBLE_CLICK');

    this.drawStart.raiseEvent(type);
  }

  /** Exit the draw mode and discard the in-progress draft. */
  stop(): void {
    if (this._type === null) return;
    this._input.removeInputAction('LEFT_CLICK');
    this._input.removeInputAction('MOUSE_MOVE');
    this._input.removeInputAction('RIGHT_CLICK');
    this._input.removeInputAction('LEFT_DOUBLE_CLICK');
    this._removePreview();
    this._resetDraft();
    this._viewer.drawMode = false;
    this._type = null;
  }

  /** Alias of {@link stop} (EV_DrawHandler naming). */
  deactivate(): void {
    this.stop();
  }

  /** Re-enter the most recently used draw mode (no-op if never started). */
  activate(): void {
    if (this._lastType !== null) this.start(this._lastType);
  }

  /** Stop drawing and release the DOM listeners of the dedicated input channel. */
  destroy(): void {
    this.stop();
    this._input.destroy();
  }

  // ── Input actions ─────────────────────────────────────────────────────────

  private async _onLeftClick(pos: ScreenSpacePosition): Promise<void> {
    const type = this._type;
    if (type === null) return;

    const picked = await this._pickSurface(pos);
    // The pick is async: the mode may have been stopped or switched meanwhile.
    if (picked === null || this._type !== type) return;

    if (type === 'point') {
      this._complete([picked], this._createPointEntity(picked));
      return;
    }

    if (type === 'rectangle') {
      if (this._firstCorner === null) {
        this._firstCorner = this._toCartographic(picked);
        this._positions = [picked];
      } else {
        const positions = this._rectanglePositions(this._firstCorner, this._toCartographic(picked));
        this._complete(positions, this._createPolygonEntity(positions));
      }
      return;
    }

    // polyline / polygon: accumulate vertices
    this._positions.push(picked);
  }

  private _onMouseMove(pos: ScreenSpacePosition): void {
    const type = this._type;
    if (type === null) return;

    // Synchronous ray/ellipsoid intersection keeps preview cheap (no GPU
    // readback per move). Clicks still use the exact GPU pick result.
    const cursor = this._rayGlobe(pos);
    if (cursor === null) return;

    if (type === 'polyline') {
      if (this._positions.length === 0) return;
      const preview = this._ensurePreview();
      preview.polyline = {
        positions: [...this._positions, cursor],
        width: PREVIEW_WIDTH,
        color: PREVIEW_LINE_COLOR,
      };
      return;
    }

    if (type === 'polygon') {
      if (this._positions.length === 0) return;
      const ring = [...this._positions, cursor];
      this._updateFillPreview(ring);
      return;
    }

    if (type === 'rectangle' && this._firstCorner !== null) {
      this._updateFillPreview(
        this._rectanglePositions(this._firstCorner, this._toCartographic(cursor)),
      );
    }
  }

  /** RIGHT_CLICK / double-click: finish polyline / polygon (or cancel if too few vertices). */
  private _finish(): void {
    const type = this._type;
    if (type !== 'polyline' && type !== 'polygon') return;

    const positions = this._dedupeTrailing(this._positions);
    const required = type === 'polygon' ? 3 : 2;
    if (positions.length < required) {
      // Not enough vertices: cancel the draft, stay in the mode.
      this._removePreview();
      this._resetDraft();
      return;
    }

    const entity = type === 'polyline'
      ? this._createPolylineEntity(positions)
      : this._createPolygonEntity(positions);
    this._complete(positions, entity);
  }

  // ── Completion ────────────────────────────────────────────────────────────

  private _complete(positions: Cartesian3[], entity: Entity): void {
    this._viewer.scene.entities.add(entity);
    this._removePreview();
    this._resetDraft();
    this.drawEnd.raiseEvent({ positions, entity });
  }

  private _resetDraft(): void {
    this._positions = [];
    this._firstCorner = null;
  }

  private _removePreview(): void {
    if (this._preview !== null) {
      this._viewer.scene.entities.remove(this._preview);
      this._preview = null;
    }
  }

  private _ensurePreview(): Entity {
    if (this._preview === null) {
      this._preview = new Entity({ name: PREVIEW_NAME });
      this._viewer.scene.entities.add(this._preview);
    }
    return this._preview;
  }

  /** Polygon-style preview: translucent green fill plus closed green outline. */
  private _updateFillPreview(ring: Cartesian3[]): void {
    const preview = this._ensurePreview();
    preview.polygon = { hierarchy: { positions: ring }, color: PREVIEW_FILL_COLOR };
    preview.polyline = {
      positions: [...ring, ring[0]],
      width: PREVIEW_WIDTH,
      color: PREVIEW_LINE_COLOR,
    };
  }

  // ── Entity factories ──────────────────────────────────────────────────────

  private _createPointEntity(position: Cartesian3): Entity {
    return new Entity({
      name: 'draw-point',
      position,
      point: { color: FINAL_LINE_COLOR, pixelSize: 10 },
    });
  }

  private _createPolylineEntity(positions: Cartesian3[]): Entity {
    return new Entity({
      name: 'draw-polyline',
      polyline: { positions, width: PREVIEW_WIDTH, color: FINAL_LINE_COLOR },
    });
  }

  private _createPolygonEntity(positions: Cartesian3[]): Entity {
    return new Entity({
      name: 'draw-polygon',
      polygon: { hierarchy: { positions }, color: FINAL_FILL_COLOR },
      polyline: { positions: [...positions, positions[0]], width: PREVIEW_WIDTH, color: FINAL_LINE_COLOR },
    });
  }

  // ── Picking helpers ───────────────────────────────────────────────────────

  /**
   * Ground position under `pos`: GPU pick (`scene.pick`) first, falling back to
   * a camera-ray / ellipsoid intersection when nothing is hit.
   */
  private async _pickSurface(pos: ScreenSpacePosition): Promise<Cartesian3 | null> {
    // Handler reports CSS pixels; scene.pick converts them to texels.
    try {
      const result = await this._viewer.scene.pick(pos.x, pos.y);
      if (result !== null && result.worldPosition !== null) {
        return result.worldPosition.clone();
      }
    } catch (err) {
      console.error('[DrawHandler] scene.pick failed:', err);
    }
    return this._rayGlobe(pos);
  }

  /** Synchronous camera-ray / unit-ellipsoid intersection ({@link Measure.rayEllipsoid}). */
  private _rayGlobe(pos: ScreenSpacePosition): Cartesian3 | null {
    const camera = this._viewer.camera;
    const canvas = this._viewer.canvas;
    const ndcX = (pos.x / Math.max(1, canvas.clientWidth)) * 2 - 1;
    const ndcY = 1 - (pos.y / Math.max(1, canvas.clientHeight)) * 2;

    const tanHalfFovY = Math.tan(camera.frustum.fov * 0.5);
    const tanHalfFovX = tanHalfFovY * camera.frustum.aspectRatio;
    const direction = new Cartesian3(
      camera.direction.x + ndcX * tanHalfFovX * camera.right.x + ndcY * tanHalfFovY * camera.up.x,
      camera.direction.y + ndcX * tanHalfFovX * camera.right.y + ndcY * tanHalfFovY * camera.up.y,
      camera.direction.z + ndcX * tanHalfFovX * camera.right.z + ndcY * tanHalfFovY * camera.up.z,
    );
    const hit = Measure.rayEllipsoid(camera.position, direction);
    if (hit === null) return null;
    // Match the GPU pick path: snap the fallback hit onto the terrain.
    return this._terrainClamp(hit);
  }

  /**
   * Snap a picked point onto the terrain surface via `globe.terrainLayer`.
   * Returns the input unchanged when no terrain layer is attached (or no
   * tile covers the location), keeping the ellipsoid behaviour.
   */
  private _terrainClamp(position: Cartesian3): Cartesian3 {
    const terrain = this._viewer.globe.terrainLayer;
    if (!terrain) return position;
    return Measure.clampToTerrain(position, (lon, lat) => terrain.sampleHeight(lon, lat));
  }

  // ── Geometry helpers ──────────────────────────────────────────────────────

  /** Normalised ECEF (Earth radius = 1) → Cartographic (radians). */
  private _toCartographic(position: Cartesian3): Cartographic {
    const s = Measure.EARTH_SCALE;
    return Cartographic.fromCartesian(new Cartesian3(position.x * s, position.y * s, position.z * s));
  }

  /** Lat/lon aligned 4 corners (CCW from south-west) of the rectangle c1–c2. */
  private _rectanglePositions(c1: Cartographic, c2: Cartographic): Cartesian3[] {
    const lonMin = Math.min(c1.longitude, c2.longitude);
    const lonMax = Math.max(c1.longitude, c2.longitude);
    const latMin = Math.min(c1.latitude, c2.latitude);
    const latMax = Math.max(c1.latitude, c2.latitude);
    const corner = (lon: number, lat: number): Cartesian3 =>
      this._terrainClamp(Measure.clampToEllipsoid(Cartesian3.fromRadians(lon, lat)));
    return [
      corner(lonMin, latMin),
      corner(lonMax, latMin),
      corner(lonMax, latMax),
      corner(lonMin, latMax),
    ];
  }

  /** Remove a trailing vertex coincident with its predecessor (double-click finish). */
  private _dedupeTrailing(positions: Cartesian3[]): Cartesian3[] {
    const out = positions.slice();
    while (
      out.length >= 2
      && Cartesian3.distance(out[out.length - 1], out[out.length - 2]) < COINCIDENT_EPSILON
    ) {
      out.pop();
    }
    return out;
  }
}

export default DrawHandler;
