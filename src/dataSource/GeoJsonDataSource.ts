import { Cartesian3 } from '../math/Cartesian3';
import { Color } from '../math/Color';
import { Entity } from './Entity';
import type { EntityOptions, PolygonHierarchy } from './Entity';
import { EntityCollection } from './EntityCollection';

/** Options for {@link GeoJsonDataSource.load}. */
export interface GeoJsonLoadOptions {
  /**
   * Name of the data source. Defaults to the file name when loading from a
   * URL, otherwise `'GeoJSON'`.
   */
  name?: string;
}

/** Per-feature style overrides read from `properties.style`. */
interface FeatureStyle {
  color?: Color;
  /** Line width in pixels (render layer interprets). */
  width?: number;
  /** Point size in pixels (render layer interprets). */
  pixelSize?: number;
  /** Polygon height in normalised units (input is metres, divided by EARTH_RADIUS). */
  height?: number;
}

const DEFAULT_POINT_COLOR = Color.CORNFLOWERBLUE;
const DEFAULT_POINT_PIXEL_SIZE = 10;
const DEFAULT_STROKE_COLOR = Color.WHITE;
const DEFAULT_STROKE_WIDTH = 2;
const DEFAULT_FILL_COLOR = Color.YELLOW;
const DEFAULT_LABEL_FONT = '16px sans-serif';

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** A GeoJSON position `[lon, lat]` or `[lon, lat, height]`. */
function isPosition(value: unknown): value is number[] {
  return (
    Array.isArray(value) &&
    value.length >= 2 &&
    typeof value[0] === 'number' &&
    typeof value[1] === 'number'
  );
}

function toPosition(value: unknown): number[] {
  if (!isPosition(value)) {
    throw new Error('GeoJsonDataSource: invalid position, expected [lon, lat(, height)]');
  }
  return value;
}

/**
 * Convert a GeoJSON `[lon, lat, (height?)]` position to the engine's
 * **normalised ECEF** convention (Earth equatorial radius = 1.0).
 *
 * `Cartesian3.fromDegrees` returns ECEF in metres, so the result is divided by
 * its ECEF magnitude (≈ `EARTH_SCALE` = 6378137 m) and normalised to unit
 * length: ellipsoid surface points satisfy `|position| = 1` (error < 1e-9).
 * A non-zero coordinate height only perturbs the direction; explicit height
 * is carried separately via `polygon.height`.
 */
function toNormalizedPosition(position: number[]): Cartesian3 {
  const ecef = Cartesian3.fromDegrees(position[0], position[1], position[2] ?? 0.0);
  return Cartesian3.normalize(ecef, new Cartesian3());
}

function toPositions(coordinates: unknown): Cartesian3[] {
  if (!Array.isArray(coordinates)) {
    throw new Error('GeoJsonDataSource: expected an array of positions');
  }
  return coordinates.map((c) => toNormalizedPosition(toPosition(c)));
}

/**
 * Max polygon ring edge length (degrees, ~200 km) before linear subdivision.
 * Simplified boundaries (e.g. a whole-country outline) carry 1000 km+ edges;
 * the chord of such an edge sags ~19 km below the imagery surface and the
 * mid-face gets depth-occluded by tiles (polygon renders as a broken ring
 * with only its edges visible). Subdividing keeps triangulated faces on or
 * above the surface so the fill renders fully.
 */
const MAX_RING_EDGE_DEG = 2.0;

function densifyRing(ring: unknown[]): unknown[] {
  if (!Array.isArray(ring) || ring.length < 2) return ring;
  const out: unknown[] = [ring[0]];
  for (let i = 1; i < ring.length; i++) {
    const a = ring[i - 1] as number[];
    const b = ring[i] as number[];
    if (!Array.isArray(a) || !Array.isArray(b)) {
      out.push(ring[i]);
      continue;
    }
    const cosLat = Math.cos((((a[1] ?? 0) + (b[1] ?? 0)) / 2) * Math.PI / 180);
    const distDeg = Math.hypot((b[0] - a[0]) * cosLat, b[1] - a[1]);
    const n = Math.ceil(distDeg / MAX_RING_EDGE_DEG);
    for (let k = 1; k < n; k++) {
      out.push([
        a[0] + ((b[0] - a[0]) * k) / n,
        a[1] + ((b[1] - a[1]) * k) / n,
        (a[2] ?? 0) + (((b[2] ?? 0) - (a[2] ?? 0)) * k) / n,
      ]);
    }
    out.push(b);
  }
  return out;
}

function subName(base: string, index: number, total: number): string {
  return total > 1 ? `${base} ${index}` : base;
}

/**
 * Parse a style colour: `#rgb`, `#rrggbb`, `#rrggbbaa` or `[r, g, b, (a)]`
 * with byte components in 0-255. Returns undefined for unrecognised input.
 */
function parseColor(value: unknown): Color | undefined {
  if (typeof value === 'string') {
    if (!value.startsWith('#')) return undefined;
    return Color.fromCssColorString(value);
  }
  if (
    Array.isArray(value) &&
    (value.length === 3 || value.length === 4) &&
    value.every((n) => typeof n === 'number')
  ) {
    return Color.fromBytes(value[0], value[1], value[2], value.length === 4 ? value[3] : 255);
  }
  return undefined;
}

/** Extract the style overrides from a feature's `properties.style` member. */
function extractStyle(properties: Record<string, unknown>): FeatureStyle {
  const style: FeatureStyle = {};
  const raw = properties.style;
  if (!isObject(raw)) return style;

  const color = parseColor(raw.color);
  if (color) style.color = color;
  if (typeof raw.width === 'number') style.width = raw.width;
  if (typeof raw.pixelSize === 'number') style.pixelSize = raw.pixelSize;
  if (typeof raw.height === 'number') {
    // metres -> normalised units (1.0 = EARTH_SCALE metres)
    style.height = raw.height / Cartesian3.EARTH_RADIUS;
  }
  return style;
}

/** Height (normalised) from a ring's first position, if it carries one. */
function ringHeight(ring: unknown): number | undefined {
  if (!Array.isArray(ring) || !isPosition(ring[0]) || ring[0].length < 3) return undefined;
  return ring[0][2] / Cartesian3.EARTH_RADIUS;
}

/**
 * GeoJsonDataSource - loads GeoJSON into a collection of {@link Entity}
 * instances, mirroring Cesium's `GeoJsonDataSource` (subset).
 *
 * Supported: `FeatureCollection`, `Feature` and bare geometries of type
 * `Point`, `MultiPoint`, `LineString`, `MultiLineString`, `Polygon`,
 * `MultiPolygon`. All positions are converted to normalised ECEF
 * (Earth equatorial radius = 1.0).
 */
export class GeoJsonDataSource {
  /** Name of this data source. */
  readonly name: string;

  /** Entities created by {@link GeoJsonDataSource.load}. */
  readonly entities: EntityCollection = new EntityCollection();

  constructor(name: string) {
    this.name = name;
  }

  /**
   * Load GeoJSON from a URL or a parsed JSON object.
   *
   * @param input - URL to fetch, or an already-parsed GeoJSON object.
   * @param options - load options (`name` for the data source).
   * @returns a promise resolving to the populated data source.
   */
  static async load(
    input: string | object,
    options: GeoJsonLoadOptions = {}
  ): Promise<GeoJsonDataSource> {
    let json: unknown;
    let defaultName = 'GeoJSON';
    if (typeof input === 'string') {
      const response = await fetch(input);
      if (!response.ok) {
        throw new Error(
          `GeoJsonDataSource.load: failed to fetch '${input}' (HTTP ${response.status})`
        );
      }
      json = await response.json();
      const file = input.split(/[?#]/)[0].split('/');
      defaultName = file[file.length - 1] || 'GeoJSON';
    } else {
      json = input;
    }

    const dataSource = new GeoJsonDataSource(options.name ?? defaultName);
    GeoJsonDataSource.loadObject(json, dataSource.entities);
    return dataSource;
  }

  private static loadObject(json: unknown, entities: EntityCollection): void {
    if (!isObject(json)) {
      throw new Error('GeoJsonDataSource.load: input is not a GeoJSON object');
    }
    const type = json.type;
    if (type === 'FeatureCollection') {
      if (!Array.isArray(json.features)) {
        throw new Error('GeoJsonDataSource.load: FeatureCollection has no features array');
      }
      for (const feature of json.features) {
        GeoJsonDataSource.loadFeature(feature, entities);
      }
      return;
    }
    if (type === 'Feature') {
      GeoJsonDataSource.loadFeature(json, entities);
      return;
    }
    // A bare geometry (Point, LineString, ...) loads as a single entity.
    GeoJsonDataSource.loadGeometry(json, undefined, entities);
  }

  private static loadFeature(feature: unknown, entities: EntityCollection): void {
    if (!isObject(feature)) {
      throw new Error('GeoJsonDataSource.load: invalid feature');
    }
    const properties = isObject(feature.properties) ? feature.properties : {};
    const name = typeof properties.name === 'string' ? properties.name : undefined;
    const style = extractStyle(properties);
    // A feature with `geometry: null` (properties-only) produces no entity.
    if (feature.geometry !== null && feature.geometry !== undefined) {
      GeoJsonDataSource.loadGeometry(feature.geometry, { name, style }, entities);
    }
  }

  private static loadGeometry(
    geometry: unknown,
    context: { name?: string; style?: FeatureStyle } | undefined,
    entities: EntityCollection
  ): void {
    if (!isObject(geometry) || typeof geometry.type !== 'string') {
      throw new Error('GeoJsonDataSource.load: invalid geometry');
    }
    const type = geometry.type;
    const coordinates = geometry.coordinates;
    const style = context?.style ?? {};
    const baseName = context?.name ?? type;

    switch (type) {
      case 'Point': {
        const position = toNormalizedPosition(toPosition(coordinates));
        entities.add(GeoJsonDataSource.createPointEntity(position, style, baseName));
        return;
      }
      case 'MultiPoint': {
        if (!Array.isArray(coordinates)) {
          throw new Error(`GeoJsonDataSource.load: invalid ${type} coordinates`);
        }
        coordinates.forEach((c, i) => {
          const position = toNormalizedPosition(toPosition(c));
          entities.add(
            GeoJsonDataSource.createPointEntity(
              position,
              style,
              subName(baseName, i, coordinates.length)
            )
          );
        });
        return;
      }
      case 'LineString': {
        entities.add(
          new Entity({
            name: baseName,
            polyline: {
              positions: toPositions(coordinates),
              width: style.width ?? DEFAULT_STROKE_WIDTH,
              color: style.color ?? DEFAULT_STROKE_COLOR,
            },
          })
        );
        return;
      }
      case 'MultiLineString': {
        if (!Array.isArray(coordinates)) {
          throw new Error(`GeoJsonDataSource.load: invalid ${type} coordinates`);
        }
        coordinates.forEach((line, i) => {
          entities.add(
            new Entity({
              name: subName(baseName, i, coordinates.length),
              polyline: {
                positions: toPositions(line),
                width: style.width ?? DEFAULT_STROKE_WIDTH,
                color: style.color ?? DEFAULT_STROKE_COLOR,
              },
            })
          );
        });
        return;
      }
      case 'Polygon': {
        entities.add(GeoJsonDataSource.createPolygonEntity(coordinates, style, baseName));
        return;
      }
      case 'MultiPolygon': {
        if (!Array.isArray(coordinates)) {
          throw new Error(`GeoJsonDataSource.load: invalid ${type} coordinates`);
        }
        coordinates.forEach((polygon, i) => {
          entities.add(
            GeoJsonDataSource.createPolygonEntity(
              polygon,
              style,
              subName(baseName, i, coordinates.length)
            )
          );
        });
        return;
      }
      default:
        throw new Error(`GeoJsonDataSource.load: unsupported geometry type '${type}'`);
    }
  }

  private static createPointEntity(
    position: Cartesian3,
    style: FeatureStyle,
    name: string
  ): Entity {
    const options: EntityOptions = {
      name,
      showPointOnTop: true,
      position,
      point: {
        color: style.color ?? DEFAULT_POINT_COLOR,
        pixelSize: style.pixelSize ?? DEFAULT_POINT_PIXEL_SIZE,
      },
    };
    if (name) {
      options.label = { text: name, font: DEFAULT_LABEL_FONT, fillColor: Color.WHITE, show: true };
    }
    return new Entity(options);
  }

  /**
   * Build a polygon entity from GeoJSON polygon coordinates:
   * the outer ring becomes `hierarchy.positions`, remaining rings become
   * `hierarchy.holes`.
   */
  private static createPolygonEntity(
    coordinates: unknown,
    style: FeatureStyle,
    name: string
  ): Entity {
    if (!Array.isArray(coordinates) || coordinates.length === 0) {
      throw new Error('GeoJsonDataSource.load: invalid Polygon coordinates');
    }
    const hierarchy: PolygonHierarchy = {
      positions: toPositions(densifyRing(coordinates[0])),
      holes: coordinates.slice(1).map((ring) => toPositions(densifyRing(ring))),
    };
    if (hierarchy.holes && hierarchy.holes.length === 0) delete hierarchy.holes;

    return new Entity({
      name,
      polygon: {
        hierarchy,
        height: style.height ?? ringHeight(coordinates[0]),
        color: style.color ?? DEFAULT_FILL_COLOR,
      },
    });
  }
}

export default GeoJsonDataSource;
