import { CesiumMath } from './CesiumMath';
import { Cartesian3 } from './Cartesian3';

/**
 * Cartographic - geographic coordinates expressed in radians.
 *
 * - `longitude` / `latitude`: radians
 * - `height`: normalised units (1 unit = {@link Cartographic.EARTH_SCALE} metres
 *   above the WGS84 ellipsoid), consistent with the engine's internal unit
 *   convention where Earth's equatorial radius = 1.0.
 *
 * `toCartesian()` converts to ECEF metres (same convention as
 * `Cartesian3.fromDegrees`); `fromCartesian()` converts back.
 */
export class Cartographic {
  constructor(
    public longitude: number = 0.0,
    public latitude: number = 0.0,
    public height: number = 0.0
  ) {}

  /**
   * Scale factor: 1 normalised unit = EARTH_SCALE metres.
   * Same constant as `Camera.EARTH_SCALE`; duplicated here because importing
   * from `scene/Camera` would create a circular dependency.
   */
  static readonly EARTH_SCALE = 6378137.0;

  /** WGS84 semi-major axis (m) */
  private static readonly A = 6378137.0;
  /** WGS84 semi-minor axis (m) */
  private static readonly B = 6356752.3142;

  clone(result?: Cartographic): Cartographic {
    if (!result) return new Cartographic(this.longitude, this.latitude, this.height);
    result.longitude = this.longitude;
    result.latitude = this.latitude;
    result.height = this.height;
    return result;
  }

  equals(right?: Cartographic): boolean {
    return (
      this === right ||
      (!!right &&
        this.longitude === right.longitude &&
        this.latitude === right.latitude &&
        this.height === right.height)
    );
  }

  equalsEpsilon(
    right: Cartographic,
    relativeEpsilon: number = 0,
    absoluteEpsilon: number = relativeEpsilon
  ): boolean {
    return (
      this === right ||
      (CesiumMath.equalsEpsilon(this.longitude, right.longitude, relativeEpsilon, absoluteEpsilon) &&
        CesiumMath.equalsEpsilon(this.latitude, right.latitude, relativeEpsilon, absoluteEpsilon) &&
        CesiumMath.equalsEpsilon(this.height, right.height, relativeEpsilon, absoluteEpsilon))
    );
  }

  toString(): string {
    return `(${this.longitude}, ${this.latitude}, ${this.height})`;
  }

  /**
   * Convert to an ECEF position in metres (same convention as
   * `Cartesian3.fromDegrees`), using a WGS84 ellipsoid.
   * The normalised height is scaled up to metres first.
   */
  toCartesian(result?: Cartesian3): Cartesian3 {
    return Cartesian3.fromRadians(
      this.longitude,
      this.latitude,
      this.height * Cartographic.EARTH_SCALE,
      result
    );
  }

  /**
   * Create a Cartographic from degrees.
   * @param longitude - degrees
   * @param latitude - degrees
   * @param height - height in normalised units (1 unit = EARTH_SCALE metres)
   */
  static fromDegrees(
    longitude: number,
    latitude: number,
    height: number = 0.0,
    result?: Cartographic
  ): Cartographic {
    return Cartographic.fromRadians(
      CesiumMath.toRadians(longitude),
      CesiumMath.toRadians(latitude),
      height,
      result
    );
  }

  /**
   * Create a Cartographic from radians.
   * @param height - height in normalised units (1 unit = EARTH_SCALE metres)
   */
  static fromRadians(
    longitude: number,
    latitude: number,
    height: number = 0.0,
    result?: Cartographic
  ): Cartographic {
    if (!result) return new Cartographic(longitude, latitude, height);
    result.longitude = longitude;
    result.latitude = latitude;
    result.height = height;
    return result;
  }

  /**
   * Inverse geodetic conversion: ECEF position (metres) to Cartographic.
   * The resulting height is divided by {@link Cartographic.EARTH_SCALE} to be
   * stored in normalised units. Uses Newton iteration on the geodetic
   * latitude, which converges to full double precision within a few steps.
   */
  static fromCartesian(cartesian: Cartesian3, result?: Cartographic): Cartographic {
    const a = Cartographic.A;
    const b = Cartographic.B;
    const a2 = a * a;
    const b2 = b * b;
    const e2 = 1.0 - b2 / a2; // first eccentricity squared
    const ep2 = (a2 - b2) / b2; // second eccentricity squared

    const x = cartesian.x;
    const y = cartesian.y;
    const z = cartesian.z;
    const p = Math.sqrt(x * x + y * y);

    const longitude = Math.atan2(y, x);
    let latitude: number;
    let heightMeters: number;

    // Polar axis: p/cos(lat) is numerically degenerate, use the closed form.
    if (p < 1e-6) {
      latitude = z >= 0.0 ? CesiumMath.PI_OVER_TWO : -CesiumMath.PI_OVER_TWO;
      heightMeters = Math.abs(z) - b;
    } else {
      // Initial estimate via the reduced latitude (Bowring's formula).
      const theta = Math.atan2(z * a, p * b);
      latitude = Math.atan2(
        z + ep2 * b * Math.sin(theta) ** 3,
        p - e2 * a * Math.cos(theta) ** 3
      );

      // Newton iteration: lat(i+1) = atan2(z + e2*N*sin(lat), p)
      for (let i = 0; i < 8; i++) {
        const sinLat = Math.sin(latitude);
        const N = a / Math.sqrt(1.0 - e2 * sinLat * sinLat);
        const next = Math.atan2(z + e2 * N * sinLat, p);
        if (Math.abs(next - latitude) < 1e-15) {
          latitude = next;
          break;
        }
        latitude = next;
      }

      // Height from the converged latitude: p = (N + h) * cos(lat)
      const cosLat = Math.cos(latitude);
      const sinLat = Math.sin(latitude);
      const N = a / Math.sqrt(1.0 - e2 * sinLat * sinLat);
      heightMeters = p / cosLat - N;
    }

    if (!result) {
      return new Cartographic(longitude, latitude, heightMeters / Cartographic.EARTH_SCALE);
    }
    result.longitude = longitude;
    result.latitude = latitude;
    result.height = heightMeters / Cartographic.EARTH_SCALE;
    return result;
  }
}

export default Cartographic;
