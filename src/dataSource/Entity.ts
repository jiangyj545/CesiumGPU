import { Cartesian3 } from '../math/Cartesian3';
import { Color } from '../math/Color';
import { Event } from './Event';

/**
 * Point graphics. `pixelSize` is expressed in screen pixels; the render
 * layer (T2.2) converts it to world units.
 */
export interface PointGraphics {
  color?: Color;
  pixelSize?: number;
}

/** Polyline graphics. Positions are normalised ECEF (see {@link Entity.position}). */
export interface PolylineGraphics {
  positions?: Cartesian3[];
  width?: number;
  color?: Color;
}

/** Polygon contour: outer `positions` ring plus optional `holes` rings. */
export interface PolygonHierarchy {
  positions: Cartesian3[];
  holes?: Cartesian3[][];
}

/**
 * Polygon graphics. `height` is expressed in normalised units
 * (1.0 = `EARTH_SCALE` metres, matching the position convention) and is used
 * by the render layer for extrusion / draping decisions.
 */
export interface PolygonGraphics {
  hierarchy?: PolygonHierarchy;
  height?: number;
  color?: Color;
}

/** Label graphics rendered by the render layer (T2.2). */
export interface LabelGraphics {
  text?: string;
  font?: string;
  fillColor?: Color;
  show?: boolean;
}

/** Constructor options for {@link Entity}. */
export interface EntityOptions {
  id?: string;
  name?: string;
  show?: boolean;
  showPointOnTop?: boolean;
  position?: Cartesian3;
  point?: PointGraphics;
  polyline?: PolylineGraphics;
  polygon?: PolygonGraphics;
  label?: LabelGraphics;
}

/**
 * Create a RFC4122 v4 uuid without external dependencies.
 */
function createUuid(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  // Fallback for environments without crypto.randomUUID.
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Entity - a unit of geographic data, semantically aligned with Cesium's
 * `Entity` (subset).
 *
 * Unit convention: {@link Entity.position} and all graphics positions are
 * **normalised ECEF** — the engine's world space where Earth's equatorial
 * radius = 1.0 (1 unit = `EARTH_SCALE` = 6378137 m). `Cartesian3.fromDegrees`
 * returns metres, so callers must normalise (divide by `EARTH_SCALE`, then
 * normalise to unit length) before assigning. `GeoJsonDataSource.load` does
 * this conversion for you.
 */
export class Entity {
  readonly id: string;

  /**
   * Raised when a property is reassigned.
   * Listeners receive `(entity, propertyName)`.
   */
  readonly definitionChanged: Event<[Entity, string]> = new Event();

  private _name: string | undefined;
  private _show: boolean;
  private _showPointOnTop: boolean;
  private _position: Cartesian3 | undefined;
  private _point: PointGraphics | undefined;
  private _polyline: PolylineGraphics | undefined;
  private _polygon: PolygonGraphics | undefined;
  private _label: LabelGraphics | undefined;

  constructor(options: EntityOptions = {}) {
    this.id = options.id ?? createUuid();
    this._name = options.name;
    this._show = options.show ?? true;
    this._showPointOnTop = options.showPointOnTop ?? false;
    this._position = options.position;
    this._point = options.point;
    this._polyline = options.polyline;
    this._polygon = options.polygon;
    this._label = options.label;
  }

  /** A human readable name (may be undefined). */
  get name(): string | undefined {
    return this._name;
  }

  set name(value: string | undefined) {
    if (this._name === value) return;
    this._name = value;
    this.definitionChanged.raiseEvent(this, 'name');
  }

  /** Whether the entity is shown. Hidden entities are skipped by the render layer. */
  get show(): boolean {
    return this._show;
  }

  set show(value: boolean) {
    if (this._show === value) return;
    this._show = value;
    this.definitionChanged.raiseEvent(this, 'show');
  }

  /**
   * When true, the point is rendered without depth-testing against the globe
   * surface so it is not occluded by terrain. Interpreted by the render layer.
   */
  get showPointOnTop(): boolean {
    return this._showPointOnTop;
  }

  set showPointOnTop(value: boolean) {
    if (this._showPointOnTop === value) return;
    this._showPointOnTop = value;
    this.definitionChanged.raiseEvent(this, 'showPointOnTop');
  }

  /**
   * Entity position in **normalised ECEF** (Earth equatorial radius = 1.0).
   * Assign `Cartesian3.fromDegrees(...)` results only after dividing by
   * `EARTH_SCALE` and normalising to unit length.
   */
  get position(): Cartesian3 | undefined {
    return this._position;
  }

  set position(value: Cartesian3 | undefined) {
    if (this._position === value) return;
    this._position = value;
    this.definitionChanged.raiseEvent(this, 'position');
  }

  get point(): PointGraphics | undefined {
    return this._point;
  }

  set point(value: PointGraphics | undefined) {
    if (this._point === value) return;
    this._point = value;
    this.definitionChanged.raiseEvent(this, 'point');
  }

  get polyline(): PolylineGraphics | undefined {
    return this._polyline;
  }

  set polyline(value: PolylineGraphics | undefined) {
    if (this._polyline === value) return;
    this._polyline = value;
    this.definitionChanged.raiseEvent(this, 'polyline');
  }

  get polygon(): PolygonGraphics | undefined {
    return this._polygon;
  }

  set polygon(value: PolygonGraphics | undefined) {
    if (this._polygon === value) return;
    this._polygon = value;
    this.definitionChanged.raiseEvent(this, 'polygon');
  }

  get label(): LabelGraphics | undefined {
    return this._label;
  }

  set label(value: LabelGraphics | undefined) {
    if (this._label === value) return;
    this._label = value;
    this.definitionChanged.raiseEvent(this, 'label');
  }
}

export default Entity;
