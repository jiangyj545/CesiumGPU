import { Cartesian3 } from '../math/Cartesian3';
import { Matrix4 } from '../math/Matrix4';
import { Transforms } from '../math/Transforms';
import { Measure } from '../analysis/Measure';
import { GeometryPipeline } from './GeometryPipeline';
import type { InterleavedGeometryBuffers } from './GeometryPipeline';

export interface PolygonGeometryOptions {
  /**
   * Height above the surface, in normalized units (Earth radius = 1).
   * Vertices are placed at radius `1 + height` along their radial direction;
   * with the default 0 the polygon lies on the sphere surface.
   * Ignored when {@link heightReference} is `'clampToGround'`.
   */
  height?: number;
  /**
   * Inner rings (holes), each in normalized ECEF, inside the outer ring.
   * Default: none.
   */
  holes?: Cartesian3[][];
  /**
   * Vertex height mode.
   * - `'none'` (default): vertices sit at the flat `1 + height` shell.
   * - `'clampToGround'`: every ring vertex (outer ring and holes) is snapped
   *   onto the terrain surface before triangulation. Note only the vertices
   *   follow the terrain — the triangle interior stays a coarse shell.
   *
   * Requires {@link sampleHeight}; without it this behaves as `'none'`.
   */
  heightReference?: 'none' | 'clampToGround';
  /**
   * Terrain height sampler `(lonDeg, latDeg) => heightMetres`, typically
   * bound to `globe.terrainLayer.sampleHeight` by the caller. The geometry
   * module never imports the terrain stack — the callback keeps layering
   * clean. Required by `heightReference: 'clampToGround'`.
   */
  sampleHeight?: (lon: number, lat: number) => number;
}

// ── Earcut (self-contained 2D triangulation, classic algorithm) ──────────────

/** Doubly-linked polygon vertex used by the ear-clipping triangulation. */
class EarNode {
  readonly i: number;   // Vertex index in the flattened coordinate array.
  readonly x: number;
  readonly y: number;
  steiner = false;
  prev: EarNode;
  next: EarNode;

  constructor(i: number, x: number, y: number) {
    this.i = i;
    this.x = x;
    this.y = y;
    this.prev = this;
    this.next = this;
  }
}

/** Twice the signed area of triangle (p, q, r) (earcut convention). */
function area(p: EarNode, q: EarNode, r: EarNode): number {
  return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
}

function sign(num: number): number {
  return num > 0 ? 1 : num < 0 ? -1 : 0;
}

function equals(p1: EarNode, p2: EarNode): boolean {
  return p1.x === p2.x && p1.y === p2.y;
}

function onSegment(p: EarNode, q: EarNode, r: EarNode): boolean {
  return (
    q.x <= Math.max(p.x, r.x) && q.x >= Math.min(p.x, r.x) &&
    q.y <= Math.max(p.y, r.y) && q.y >= Math.min(p.y, r.y)
  );
}

/** Whether segments (p1,q1) and (p2,q2) intersect. */
function intersects(p1: EarNode, q1: EarNode, p2: EarNode, q2: EarNode): boolean {
  const o1 = sign(area(p1, q1, p2));
  const o2 = sign(area(p1, q1, q2));
  const o3 = sign(area(p2, q2, p1));
  const o4 = sign(area(p2, q2, q1));

  if (o1 !== o2 && o3 !== o4) return true;

  if (o1 === 0 && onSegment(p1, p2, q1)) return true;
  if (o2 === 0 && onSegment(p1, q2, q1)) return true;
  if (o3 === 0 && onSegment(p2, p1, q2)) return true;
  if (o4 === 0 && onSegment(p2, q1, q2)) return true;
  return false;
}

/** Whether any polygon edge intersects segment (a,b). */
function intersectsPolygon(a: EarNode, b: EarNode): boolean {
  let p = a;
  do {
    if (
      p.i !== a.i && p.next.i !== a.i && p.i !== b.i && p.next.i !== b.i &&
      intersects(p, p.next, a, b)
    ) {
      return true;
    }
    p = p.next;
  } while (p !== a);
  return false;
}

/** Whether point b is locally inside the polygon at vertex a. */
function locallyInside(a: EarNode, b: EarNode): boolean {
  return area(a.prev, a, a.next) < 0
    ? area(a, b, a.next) >= 0 && area(a, a.prev, b) >= 0
    : area(a, b, a.prev) < 0 || area(a, a.next, b) < 0;
}

/** Whether the midpoint of segment (a,b) is inside the polygon (ray casting). */
function middleInside(a: EarNode, b: EarNode): boolean {
  let p = a;
  let inside = false;
  const px = (a.x + b.x) / 2;
  const py = (a.y + b.y) / 2;

  do {
    if (
      (p.y > py) !== (p.next.y > py) && p.next.y !== p.y &&
      px < ((p.next.x - p.x) * (py - p.y)) / (p.next.y - p.y) + p.x
    ) {
      inside = !inside;
    }
    p = p.next;
  } while (p !== a);

  return inside;
}

/** Whether (a,b) is a valid internal diagonal for splitting the polygon. */
function isValidDiagonal(a: EarNode, b: EarNode): boolean {
  return (
    a.next.i !== b.i && a.prev.i !== b.i && !intersectsPolygon(a, b) &&
    ((locallyInside(a, b) && locallyInside(b, a) && middleInside(a, b) &&
      (area(a.prev, a, b.prev) !== 0 || area(a, b.prev, b) !== 0)) ||
      (equals(a, b) && area(a.prev, a, a.next) > 0 && area(b.prev, b, b.next) > 0))
  );
}

/** Whether point p lies inside triangle (a,b,c). */
function pointInTriangle(
  ax: number, ay: number,
  bx: number, by: number,
  cx: number, cy: number,
  px: number, py: number
): boolean {
  return (
    (cx - px) * (ay - py) >= (ax - px) * (cy - py) &&
    (ax - px) * (by - py) >= (bx - px) * (ay - py) &&
    (bx - px) * (cy - py) >= (cx - px) * (by - py)
  );
}

function insertNode(i: number, x: number, y: number, last: EarNode | undefined): EarNode {
  const p = new EarNode(i, x, y);
  if (!last) {
    p.prev = p;
    p.next = p;
  } else {
    p.next = last.next;
    p.prev = last;
    last.next.prev = p;
    last.next = p;
  }
  return p;
}

function removeNode(p: EarNode): void {
  p.next.prev = p.prev;
  p.prev.next = p.next;
}

/** Shoelace signed area of ring [start, end) (positive => CCW). */
function signedArea(data: number[], start: number, end: number, dim: number): number {
  let sum = 0;
  let j = end - dim;
  for (let i = start; i < end; i += dim) {
    sum += (data[j] - data[i]) * (data[i + 1] + data[j + 1]);
    j = i;
  }
  return sum;
}

/** Build a circular linked list for ring [start, end), forcing outer CCW / hole CW winding. */
function linkedList(data: number[], start: number, end: number, dim: number, clockwise: boolean): EarNode | undefined {
  let last: EarNode | undefined;

  if (clockwise === signedArea(data, start, end, dim) > 0) {
    for (let i = start; i < end; i += dim) last = insertNode(i, data[i], data[i + 1], last);
  } else {
    for (let i = end - dim; i >= start; i -= dim) last = insertNode(i, data[i], data[i + 1], last);
  }

  if (last && equals(last, last.next)) {
    removeNode(last);
    last = last.next;
  }
  return last;
}

/** Remove collinear/duplicate points; returns the new list tail. */
function filterPoints(start: EarNode | null, end?: EarNode | null): EarNode | null {
  if (!start) return start;
  let endNode: EarNode = end ?? start;

  let p: EarNode = start;
  let again: boolean;
  do {
    again = false;
    if (!p.steiner && (equals(p, p.next) || area(p.prev, p, p.next) === 0)) {
      removeNode(p);
      p = endNode = p.prev;
      if (p === p.next) break;
      again = true;
    } else {
      p = p.next;
    }
  } while (again || p !== endNode);

  return endNode;
}

/** Whether the triangle (prev, ear, next) contains no other polygon vertex. */
function isEar(ear: EarNode): boolean {
  const a = ear.prev;
  const b = ear;
  const c = ear.next;

  if (area(a, b, c) >= 0) return false; // Reflex, can't be an ear.

  let p = ear.next.next;
  while (p !== ear.prev) {
    if (
      pointInTriangle(a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y) &&
      area(p.prev, p, p.next) >= 0
    ) {
      return false;
    }
    p = p.next;
  }
  return true;
}

function cureLocalIntersections(start: EarNode | null, triangles: number[], dim: number): EarNode | null {
  if (!start) return start;
  let p: EarNode = start;
  do {
    const a = p.prev;
    const b = p.next.next;

    if (!equals(a, b) && intersects(a, p, p.next, b) && locallyInside(a, b) && locallyInside(b, a)) {
      triangles.push(a.i / dim);
      triangles.push(p.i / dim);
      triangles.push(b.i / dim);

      removeNode(p);
      removeNode(p.next);

      p = start = b;
    }
    p = p.next;
  } while (p !== start);

  return filterPoints(p);
}

function splitEarcut(start: EarNode, triangles: number[], dim: number): void {
  let a: EarNode = start;
  do {
    let b: EarNode = a.next.next;
    while (b !== a.prev) {
      if (a.i !== b.i && isValidDiagonal(a, b)) {
        const c = splitPolygon(a, b);

        const aFiltered = filterPoints(a, a.next);
        const cFiltered = filterPoints(c, c.next);

        earcutLinked(aFiltered, triangles, dim);
        earcutLinked(cFiltered, triangles, dim);
        return;
      }
      b = b.next;
    }
    a = a.next;
  } while (a !== start);
}

function earcutLinked(ear: EarNode | null, triangles: number[], dim: number, pass: number = 0): void {
  if (!ear) return;

  let stop: EarNode = ear;
  let current: EarNode | null = ear;

  while (current && current.prev !== current.next) {
    const prev: EarNode = current.prev;
    const next: EarNode = current.next;

    if (isEar(current)) {
      triangles.push(prev.i / dim);
      triangles.push(current.i / dim);
      triangles.push(next.i / dim);

      removeNode(current);

      // Skipping the next vertex leads to fewer sliver triangles.
      current = next.next;
      stop = next.next;
      continue;
    }

    current = current.next;

    if (current === stop) {
      if (pass === 0) {
        earcutLinked(filterPoints(current), triangles, dim, 1);
      } else if (pass === 1) {
        const cured = cureLocalIntersections(filterPoints(current), triangles, dim);
        earcutLinked(cured, triangles, dim, 2);
      } else if (pass === 2) {
        splitEarcut(current, triangles, dim);
      }
      break;
    }
  }
}

function getLeftmost(start: EarNode): EarNode {
  let p = start;
  let leftmost = start;
  do {
    if (p.x < leftmost.x || (p.x === leftmost.x && p.y < leftmost.y)) leftmost = p;
    p = p.next;
  } while (p !== start);
  return leftmost;
}

function compareX(a: EarNode, b: EarNode): number {
  return a.x - b.x;
}

/** Bridge a hole into the outer ring via its leftmost vertex. */
function eliminateHole(hole: EarNode, outerNode: EarNode): EarNode {
  const bridge = findHoleBridge(hole, outerNode);
  if (!bridge) return outerNode;

  const bridgeReverse = splitPolygon(bridge, hole);
  const filteredBridge = filterPoints(bridge, bridgeReverse.next);
  return filterPoints(bridgeReverse, filteredBridge ? filteredBridge.next : undefined) ?? outerNode;
}

/** Insert a bridge between a and b, linking the two polygon halves; returns the new half's head. */
function splitPolygon(a: EarNode, b: EarNode): EarNode {
  const a2 = new EarNode(a.i, a.x, a.y);
  const b2 = new EarNode(b.i, b.x, b.y);
  const an = a.next;
  const bp = b.prev;

  a.next = b;
  b.prev = a;

  a2.next = an;
  an.prev = a2;

  b2.next = a2;
  a2.prev = b2;

  bp.next = b2;
  b2.prev = bp;

  return b2;
}

function findHoleBridge(hole: EarNode, outerNode: EarNode): EarNode | null {
  let p = outerNode;
  const hx = hole.x;
  const hy = hole.y;
  let qx = -Infinity;
  let m: EarNode | undefined;

  do {
    if (hy <= p.y && hy >= p.next.y && p.next.y !== p.y) {
      const x = p.x + ((hy - p.y) * (p.next.x - p.x)) / (p.next.y - p.y);
      if (x <= hx && x > qx) {
        qx = x;
        if (x === hx) {
          if (hy === p.y) return p;
          if (hy === p.next.y) return p.next;
        }
        m = p.x < p.next.x ? p : p.next;
      }
    }
    p = p.next;
  } while (p !== outerNode);

  if (!m) return null;

  if (hx === qx) return m.prev;

  const stop = m;
  const mx = m.x;
  const my = m.y;
  let tanMin = Infinity;

  p = m.next;
  while (p !== stop) {
    if (
      hx >= p.x && p.x >= mx && hx !== p.x &&
      pointInTriangle(hy < my ? hx : qx, hy, mx, my, hy < my ? qx : hx, hy, p.x, p.y)
    ) {
      const tan = Math.abs(hy - p.y) / (hx - p.x);
      if ((tan < tanMin || (tan === tanMin && p.x > m.x)) && locallyInside(p, hole)) {
        m = p;
        tanMin = tan;
      }
    }
    p = p.next;
  }
  return m;
}

function eliminateHoles(data: number[], holeIndices: number[], outerNode: EarNode, dim: number): EarNode {
  const queue: EarNode[] = [];
  const len = holeIndices.length;

  for (let i = 0; i < len; i++) {
    const start = holeIndices[i] * dim;
    const end = i < len - 1 ? holeIndices[i + 1] * dim : data.length;
    const list = linkedList(data, start, end, dim, false);
    if (list) {
      if (list === list.next) list.steiner = true;
      queue.push(getLeftmost(list));
    }
  }

  queue.sort(compareX);

  let outer = outerNode;
  for (let i = 0; i < queue.length; i++) {
    outer = eliminateHole(queue[i], outer);
  }
  return outer;
}

/**
 * Triangulate a 2D polygon (with optional holes) using the classic earcut
 * algorithm. Returns a flat triangle index list into the flattened vertex
 * array. Output triangles are CCW in the input coordinate system.
 *
 * @param coords      - flattened vertex coordinates (x, y at stride `dim`)
 * @param holeIndices - vertex index where each hole ring starts
 * @param dim         - floats per vertex in `coords` (default 2)
 */
export function earcut2D(coords: number[], holeIndices?: number[], dim: number = 2): Uint32Array {
  const hasHoles = !!holeIndices && holeIndices.length > 0;
  const outerLen = hasHoles ? (holeIndices as number[])[0] * dim : coords.length;

  const outerNode = linkedList(coords, 0, outerLen, dim, true);
  const triangles: number[] = [];

  if (!outerNode || outerNode.next === outerNode.prev) {
    return new Uint32Array(0);
  }

  let ear: EarNode = outerNode;
  if (hasHoles) {
    ear = eliminateHoles(coords, holeIndices as number[], ear, dim);
  }

  earcutLinked(ear, triangles, dim);

  return new Uint32Array(triangles);
}

// ── PolygonGeometry ──────────────────────────────────────────────────────────

/**
 * PolygonGeometry - Triangulates a spherical polygon and returns interleaved
 * buffers, in normalized ECEF space (Earth equatorial radius = 1.0).
 *
 * The polygon is projected onto the local east-north plane built at the first
 * vertex (inverse of `Transforms.eastNorthUpToFixedFrame`), triangulated in
 * 2D with a self-implemented earcut, then the vertices are restored on the
 * sphere (radius `1 + height`, `height` in normalized units). Output normals
 * face outward from the globe; triangles wind CCW when seen from outside.
 */
export class PolygonGeometry {
  /**
   * Build the polygon geometry.
   *
   * @param positions - outer ring in normalized ECEF (Earth radius = 1),
   *                    either winding; output is winding-normalized.
   * @param options   - height / holes
   */
  static fromCartesianPositions(
    positions: Cartesian3[],
    options: PolygonGeometryOptions = {}
  ): InterleavedGeometryBuffers {
    if (positions.length < 3) {
      return GeometryPipeline.emptyBuffers();
    }

    const height = options.height ?? 0.0;
    const holes = options.holes ?? [];

    // Terrain clamping pass: snap every ring vertex onto the terrain surface
    // before triangulation. No-op unless heightReference === 'clampToGround'
    // and a sampler was provided; the input arrays are never modified.
    const sample = options.sampleHeight;
    const terrainClamped =
      options.heightReference === 'clampToGround' && typeof sample === 'function';
    const rings: Cartesian3[][] = [positions, ...holes].map((ring) =>
      terrainClamped && sample
        ? ring.map((p) => Measure.clampToTerrain(p, sample))
        : ring,
    );

    // Local ENU frame at the first (possibly terrain-clamped) vertex; project
    // onto its east-north plane.
    const origin = rings[0][0];
    const enu = Transforms.eastNorthUpToFixedFrame(origin);
    const inverse = Matrix4.inverse(enu, new Matrix4());
    const inv = inverse.values;
    const flat: number[] = [];             // [east, north] pairs
    const sourcePoints: Cartesian3[] = []; // same order as `flat`
    const holeIndices: number[] = [];

    let eastMin = Infinity, eastMax = -Infinity;
    let northMin = Infinity, northMax = -Infinity;

    for (let r = 0; r < rings.length; r++) {
      if (r > 0) holeIndices.push(flat.length / 2);
      const ring = rings[r];
      if (ring.length < 3) {
        return GeometryPipeline.emptyBuffers();
      }
      for (const p of ring) {
        const east = inv[0] * p.x + inv[4] * p.y + inv[8] * p.z + inv[12];
        const north = inv[1] * p.x + inv[5] * p.y + inv[9] * p.z + inv[13];
        flat.push(east, north);
        sourcePoints.push(p);
        eastMin = Math.min(eastMin, east);
        eastMax = Math.max(eastMax, east);
        northMin = Math.min(northMin, north);
        northMax = Math.max(northMax, north);
      }
    }

    const triangles = earcut2D(flat, holeIndices.length > 0 ? holeIndices : undefined, 2);
    if (triangles.length === 0) {
      return GeometryPipeline.emptyBuffers();
    }

    const vertexCount = sourcePoints.length;
    const eastRange = eastMax - eastMin > 1e-15 ? eastMax - eastMin : 1.0;
    const northRange = northMax - northMin > 1e-15 ? northMax - northMin : 1.0;

    const positionsOut = new Float32Array(vertexCount * 3);
    const normalsOut = new Float32Array(vertexCount * 3);
    const uvsOut = new Float32Array(vertexCount * 2);

    for (let i = 0; i < vertexCount; i++) {
      const src = sourcePoints[i];
      const len = Cartesian3.magnitude(src);
      const dir = len > 1e-15
        ? new Cartesian3(src.x / len, src.y / len, src.z / len)
        : new Cartesian3(0, 0, 1); // Degenerate position guard.

      // Terrain-clamped vertices carry their own radius (magnitude of the
      // clamped point, polar squash included); dir * len reproduces src.
      const radius = terrainClamped ? len : 1.0 + height;

      positionsOut[i * 3] = dir.x * radius;
      positionsOut[i * 3 + 1] = dir.y * radius;
      positionsOut[i * 3 + 2] = dir.z * radius;
      normalsOut[i * 3] = dir.x;
      normalsOut[i * 3 + 1] = dir.y;
      normalsOut[i * 3 + 2] = dir.z;

      uvsOut[i * 2] = (flat[i * 2] - eastMin) / eastRange;
      uvsOut[i * 2 + 1] = (flat[i * 2 + 1] - northMin) / northRange;
    }

    return {
      vertices: GeometryPipeline.interleave(positionsOut, normalsOut, uvsOut),
      indices: triangles,
      vertexCount,
      indexCount: triangles.length,
    };
  }
}

export default PolygonGeometry;
