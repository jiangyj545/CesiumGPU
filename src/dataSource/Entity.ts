import { Cartesian3 } from '../math/Cartesian3';
import { Color } from '../math/Color';
import { Quaternion } from '../math/Quaternion';
import { Event } from './Event';
import type { PropertyOrValue } from './CallbackProperty';
import { Cartesian2 } from '../math/Cartesian2';
import type {
  DistanceDisplayCondition,
  HeightReference,
  HorizontalOrigin,
  LabelStyle,
  VerticalOrigin,
} from './BillboardStyle';
import type { FlowArrowPolylineMaterial } from '../effect/FlowArrowMath';
/**
 * Point graphics. `pixelSize` is expressed in screen pixels; the render
 * layer (T2.2) converts it to world units.
 */
export interface PointGraphics {
  color?: Color;
  pixelSize?: number;
  /** Billboard anchoring against the projected position (default CENTER/CENTER). */
  horizontalOrigin?: HorizontalOrigin;
  verticalOrigin?: VerticalOrigin;
  /** Visibility window in metres (skip rendering outside near..far). */
  distanceDisplayCondition?: DistanceDisplayCondition;
  /** Ignore depth occlusion within this camera distance (metres; Infinity = always). */
  disableDepthTestDistance?: number;
  /** Per-pixel offset added on top of the anchor offset (Cesium pixelOffset). */
  pixelOffset?: PropertyOrValue<Cartesian2>;
}

/** Polyline graphics. Positions are normalised ECEF (see {@link Entity.position}); `positions` may be a dynamic Property. */
export interface PolylineGraphics {
  positions?: PropertyOrValue<Cartesian3[]>;
  /** Ignore depth occlusion within this camera distance (metres; Infinity = always). */
  disableDepthTestDistance?: number;
  /** Ribbon width in CSS pixels (visualizer 换算世界宽)。 */
  width?: number;
  color?: Color;
  /** Visibility window in metres (skip rendering outside near..far; may be a Property). */
  distanceDisplayCondition?: PropertyOrValue<DistanceDisplayCondition>;
  /**
   * FlowArrow 流动箭头材质（P2-6）：type 'flow' 判别；设置后该 polyline
   * 改由 FlowArrowPolyline 特效渲染（不再走普通 ribbon 图元）。
   */
  material?: FlowArrowPolylineMaterial;
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

/** Label graphics rendered by the render layer (T2.2). `text` may be a dynamic Property. */
export interface LabelGraphics {
  text?: PropertyOrValue<string>;
  font?: string;
  fillColor?: Color;
  show?: boolean;
  /** Outline colour / width and FILL vs OUTLINE composition (P2-4). */
  outlineColor?: Color;
  outlineWidth?: number;
  style?: LabelStyle;
  /** Billboard anchoring against the projected position (default legacy: bottom + lift). */
  horizontalOrigin?: HorizontalOrigin;
  verticalOrigin?: VerticalOrigin;
  /** Visibility window in metres (skip rendering outside near..far). */
  distanceDisplayCondition?: DistanceDisplayCondition;
  /** Ignore depth occlusion within this camera distance (metres; Infinity = always). */
  disableDepthTestDistance?: number;
  /** Per-pixel offset added on top of the anchor offset (Cesium pixelOffset). */
  pixelOffset?: PropertyOrValue<Cartesian2>;
}

/**
 * glTF model graphics. `uri` is loaded once by the visualizer and the model
 * primitive is re-positioned every frame from `entity.position` +
 * `entity.orientation`. `scale` is a uniform multiplier on the glTF extent;
 * `minimumPixelSize` (CSS px) clamps the on-screen size from below by
 * scaling up when the camera is far.
 */
export interface ModelGraphics {
  uri: string;
  scale?: number;
  minimumPixelSize?: number;
}

/**
 * Path (track) graphics: samples `entity.position` over time and renders the
 * recent history as a polyline.
 * - `trailTime` — seconds of past positions kept on screen.
 * - `leadTime` — seconds of future positions sampled from a dynamic position
 *   Property (evaluated at `now + t`), rendered ahead of the entity.
 * - `resolution` — sampling step in seconds (default 1).
 */
export interface PathGraphics {
  leadTime?: number;
  trailTime?: number;
  width?: number;
  color?: Color;
  resolution?: number;
  /** Visibility window in metres (skip rendering outside near..far; may be a Property). */
  distanceDisplayCondition?: PropertyOrValue<DistanceDisplayCondition>;
}

/**
 * Cylinder / cone graphics: `length` / `topRadius` / `bottomRadius` in
 * metres along the entity local +Z axis (identity orientation = vertical;
 * a nose cone is `topRadius: 0` with the entity oriented along the flight
 * direction). All numeric fields accept dynamic Properties (P0-A chain).
 */
export interface CylinderGraphics {
  length: PropertyOrValue<number>;
  topRadius: PropertyOrValue<number>;
  bottomRadius: PropertyOrValue<number>;
  color?: Color;
  /** Alias of `color` (Cesium material colour parity). */
  material?: Color;
  /** Visibility window in metres (skip rendering outside near..far; may be a Property). */
  distanceDisplayCondition?: PropertyOrValue<DistanceDisplayCondition>;
}

/** Constructor options for {@link Entity}. */
export interface EntityOptions {
  id?: string;
  name?: string;
  show?: boolean;
  showPointOnTop?: boolean;
  /** Entity-level height semantics against terrain (NONE / CLAMP_TO_GROUND / RELATIVE_TO_GROUND). */
  heightReference?: HeightReference;
  position?: PropertyOrValue<Cartesian3>;
  /** Orientation quaternion in the local ENU frame at `position` (or a Property). */
  orientation?: PropertyOrValue<Quaternion>;
  point?: PointGraphics;
  polyline?: PolylineGraphics;
  polygon?: PolygonGraphics;
  label?: LabelGraphics;
  model?: ModelGraphics;
  path?: PathGraphics;
  cylinder?: CylinderGraphics;
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
  private _heightReference: HeightReference | undefined;
  private _position: PropertyOrValue<Cartesian3> | undefined;
  private _orientation: PropertyOrValue<Quaternion> | undefined;
  private _point: PointGraphics | undefined;
  private _polyline: PolylineGraphics | undefined;
  private _polygon: PolygonGraphics | undefined;
  private _label: LabelGraphics | undefined;
  private _model: ModelGraphics | undefined;
  private _path: PathGraphics | undefined;
  private _cylinder: CylinderGraphics | undefined;
  constructor(options: EntityOptions = {}) {
    this.id = options.id ?? createUuid();
    this._name = options.name;
    this._show = options.show ?? true;
    this._showPointOnTop = options.showPointOnTop ?? false;
    this._heightReference = options.heightReference;
    this._position = options.position;
    this._orientation = options.orientation;
    this._point = options.point;
    this._polyline = options.polyline;
    this._polygon = options.polygon;
    this._label = options.label;
    this._model = options.model;
    this._path = options.path;
    this._cylinder = options.cylinder;
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
   * Entity-level height semantics against the terrain surface (P2-4):
   * CLAMP_TO_GROUND / RELATIVE_TO_GROUND re-scale the entity position against
   * the sampled terrain height before rendering (billboards via OverlayPass,
   * geometry via EntityVisualizer).
   */
  get heightReference(): HeightReference | undefined {
    return this._heightReference;
  }

  set heightReference(value: HeightReference | undefined) {
    if (this._heightReference === value) return;
    this._heightReference = value;
    this.definitionChanged.raiseEvent(this, 'heightReference');
  }

  /**
   * Entity position in **normalised ECEF** (Earth equatorial radius = 1.0),
   * or a Property (`CallbackProperty`, ...) sampled once per frame by the
   * render layer. Assign `Cartesian3.fromDegrees(...)` results only after
   * dividing by `EARTH_SCALE` and normalising to unit length.
   */
  get position(): PropertyOrValue<Cartesian3> | undefined {
    return this._position;
  }

  set position(value: PropertyOrValue<Cartesian3> | undefined) {
    if (this._position === value) return;
    this._position = value;
    this.definitionChanged.raiseEvent(this, 'position');
  }

  /**
   * Orientation in the local ENU frame at `position`, as a quaternion
   * (e.g. `Transforms.headingPitchRollQuaternion(...)`). Dynamic Properties
   * are polled every frame. Consumed by the model graphics.
   */
  get orientation(): PropertyOrValue<Quaternion> | undefined {
    return this._orientation;
  }

  set orientation(value: PropertyOrValue<Quaternion> | undefined) {
    if (this._orientation === value) return;
    this._orientation = value;
    this.definitionChanged.raiseEvent(this, 'orientation');
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

  get model(): ModelGraphics | undefined {
    return this._model;
  }

  set model(value: ModelGraphics | undefined) {
    if (this._model === value) return;
    this._model = value;
    this.definitionChanged.raiseEvent(this, 'model');
  }

  get path(): PathGraphics | undefined {
    return this._path;
  }

  set path(value: PathGraphics | undefined) {
    if (this._path === value) return;
    this._path = value;
    this.definitionChanged.raiseEvent(this, 'path');
  }

  get cylinder(): CylinderGraphics | undefined {
    return this._cylinder;
  }

  set cylinder(value: CylinderGraphics | undefined) {
    if (this._cylinder === value) return;
    this._cylinder = value;
    this.definitionChanged.raiseEvent(this, 'cylinder');
  }
}

export default Entity;
