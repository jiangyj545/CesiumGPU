/**
 * TilesetContentLoader – tileset.json 取数 + 树解析（3DTiles 调度层 P3）。
 *
 * 职责：
 * ① fetch tileset.json（复用 `fetchArrayBuffer`，支持 .gz / urlDecorator 注入）；
 * ② 把 tileset.json 树解析成 `TileNode` 树：逐级累计 transform、统一
 *    boundingVolume → 世界（归一化 ECEF）包围球、refine 继承、content
 *    URI 相对 tileset.json 解析（可经 urlDecorator 再加工，Google key 注入）。
 *
 * ── transform 归一化推导（为什么左乘 S、子级为何只右乘）────────────────
 *
 * 记 R = 6378137（WGS84 赤道半径，米），S = uniform(1/R)。
 * 3DTiles 的 tile transform T 把 tile 局部点 p（米）映射到 ECEF 米：
 *   p_ecef = T·p = M·p + t          （M 单位旋转基，t 米制平移）
 * 引擎内部世界 = 归一化 ECEF（赤道半径 = 1），故目标变换 N 满足
 *   q = p_ecef / R = (M·p + t)/R = (M/R)·p + t/R。
 * 矩阵语言即 **N = S·T（S 左乘）**：
 *   - 旋转列被 1/R 缩放 → 右侧喂原始「米」数值的顶点即得归一化世界；
 *   - 平移列被 1/R 缩放 → 米制 ECEF 平移落到归一化球面。
 * 验证已知点：T = translate([6378137,0,0])（赤道 0°E）时
 *   N·(0,0,0) = t/R = [1,0,0] —— 正好落在归一化赤道与本初子午线交点。✓
 * （注意不能右乘：T·S 会保留米制平移 t，映射点跑出地球 6 千万倍。）
 *
 * 子级累计：米制累计矩阵 T_acc(子) = T_acc(父)·T(子)，而
 *   N(子) = S·T_acc(子) = S·T_acc(父)·T(子) = N(父)·T(子)。
 * 因此递推时父级只在根处左乘一次 S，之后每级 **右乘** 原始米制矩阵，
 * 与「逐级整体 S×T_acc」按结合律逐点等价，省去每级一次全矩阵缩放。
 *
 * boundingVolume 的 box/sphere 位于 tile 局部坐标（米）：其中心 c 经
 * N 映射即世界归一化中心；半径按 N 旋转列最大模（单位旋转基下 = 1/R）
 * 缩放，即 r_世界 = r_米 / R。region 是绝对 WGS84 弧度（不随 transform），
 * 直接按米制 ECEF 造球后整体 /R。
 */

import { Cartesian3 } from '../math/Cartesian3';
import { Matrix4 } from '../math/Matrix4';
import { fetchArrayBuffer } from '../loader/fetchWithGzip';

/** WGS84 赤道半径（米）——1 引擎单位 = 6378137 米（同 Camera.EARTH_SCALE）。 */
export const TILESET_EARTH_SCALE = 6378137.0;

/** 细化策略：ADD 父子同显；REPLACE 子全就绪后隐藏父。 */
export type TileRefine = 'ADD' | 'REPLACE';

/** 内容类型（按扩展名判定）。pnts 本版不支持（计入 failed + warn）。 */
export type TileContentKind = 'b3dm' | 'glb' | 'pnts' | 'unknown';

export interface TileContent {
  /** tileset.json 里的原始 uri（可能相对）。 */
  uri: string;
  /** 相对 tileset.json 解析、再经 urlDecorator 加工后的最终请求 URL。 */
  resolvedUrl: string;
  kind: TileContentKind;
}

/** 解析后的瓦片节点（树）。 */
export interface TileNode {
  /** 累计归一化世界矩阵（见文件头推导：根左乘 S，子级右乘米制矩阵）。 */
  transform: Matrix4;
  /** 米制几何误差（3DTiles 原始值；SSE 计算时再 /R 归一化）。 */
  geometricError: number;
  /** 细化策略（子级未声明时继承父级；根缺省 REPLACE）。 */
  refine: TileRefine;
  /** 世界（归一化 ECEF）包围球——视锥剔除与 SSE 距离用。 */
  sphere: { center: Cartesian3; radius: number };
  content: TileContent | null;
  children: TileNode[];
  parent: TileNode | null;
  /** 根 = 0。 */
  depth: number;
}

/** 每请求 URL 加工钩子（Google Tiles 的 key 注入等）。返回最终 URL。 */
export type TileUrlDecorator = (url: string) => string;

export interface LoadTilesetOptions {
  /** 所有请求 URL（含 tileset.json 本身）在解析后统一经过此钩子。 */
  urlDecorator?: TileUrlDecorator;
  /** 附加到 tileset.json（及后续 content）请求的 HTTP 头（P1-F 认证）。 */
  headers?: Record<string, string>;
}

// ── tileset.json 原始结构（只声明调度层消费的字段）─────────────────────────

interface RawBoundingVolume {
  box?: number[];
  sphere?: number[];
  region?: number[];
}

interface RawContent {
  uri?: string;
}

interface RawTile {
  transform?: number[];
  geometricError?: number;
  refine?: string;
  boundingVolume?: RawBoundingVolume;
  content?: RawContent;
  contents?: RawContent[];
  children?: RawTile[];
}

interface RawTileset {
  root?: RawTile;
  geometricError?: number;
}

// ── 取数 ─────────────────────────────────────────────────────────────────────

/**
 * fetch 并解析 tileset.json。URL 先经 urlDecorator（Google key 注入），
 * 走 fetchArrayBuffer（显式 .gz 自动解压）→ JSON.parse。
 * 网络/JSON 错误向上抛（调用方决定整树失败语义）。
 */
export async function fetchTilesetJson(
  url: string,
  options?: LoadTilesetOptions
): Promise<{ json: RawTileset; finalUrl: string }> {
  const finalUrl = options?.urlDecorator ? options.urlDecorator(url) : url;
  const buffer = await fetchArrayBuffer(finalUrl, options?.headers ? { init: { headers: options.headers } } : undefined);
  const text = new TextDecoder().decode(buffer);
  const json = JSON.parse(text) as RawTileset;
  if (!json || typeof json !== 'object' || !json.root) {
    throw new Error(`TilesetContentLoader: "${finalUrl}" 不是合法 tileset.json（缺 root）`);
  }
  return { json, finalUrl };
}

// ── 树解析 ───────────────────────────────────────────────────────────────────

/**
 * 解析 tileset 树。`baseUrl` 为 tileset.json 的最终 URL（content uri 的
 * 相对基准）；`baseUrl` 传入前应已完成 urlDecorator 加工。
 * root 缺失时抛错（fetchTilesetJson 已挡，双保险）。
 */
export function parseTileset(
  json: RawTileset,
  baseUrl: string,
  options?: LoadTilesetOptions
): TileNode {
  const root = json.root;
  if (!root) throw new Error('TilesetContentLoader: tileset.json 缺少 root 节点');

  const rootTransform = readTransform(root.transform);
  return buildNode(root, rootTransform, null, 0, 'REPLACE', baseUrl, options);
}

/** 16 元列主序 → Matrix4；缺省单位阵。 */
function readTransform(values?: number[]): Matrix4 {
  if (!values || values.length !== 16) return Matrix4.identity();
  return Matrix4.fromArray(values);
}

/** 递归构造节点。`parentNormalized` 为父级归一化累计矩阵。 */
function buildNode(
  raw: RawTile,
  localMetreTransform: Matrix4,
  parentNormalized: Matrix4 | null,
  depth: number,
  inheritedRefine: TileRefine,
  baseUrl: string,
  options?: LoadTilesetOptions
): TileNode {
  // 归一化累计（见文件头推导）：根 = S×T；子 = N父×T子（等价于整体 S×T_acc）
  const transform = parentNormalized
    ? Matrix4.multiply(parentNormalized, localMetreTransform, new Matrix4())
    : Matrix4.multiply(
        Matrix4.fromUniformScale(1.0 / TILESET_EARTH_SCALE),
        localMetreTransform,
        new Matrix4()
      );

  const refine: TileRefine =
    raw.refine === 'ADD' ? 'ADD' : raw.refine === 'REPLACE' ? 'REPLACE' : inheritedRefine;

  const geometricError = typeof raw.geometricError === 'number' ? raw.geometricError : 0;

  const content = resolveContent(raw.content ?? pickFirstContent(raw.contents), baseUrl, options);

  const node: TileNode = {
    transform,
    geometricError,
    refine,
    sphere: boundingVolumeToSphere(raw.boundingVolume, transform),
    content,
    children: [],
    parent: null,
    depth,
  };

  for (const child of raw.children ?? []) {
    const childNode = buildNode(
      child,
      readTransform(child.transform),
      transform,
      depth + 1,
      refine,
      baseUrl,
      options
    );
    childNode.parent = node;
    node.children.push(childNode);
  }
  return node;
}

/** contents[]（3DTiles 1.1 多内容）取第一个；本调度器暂不展开多内容。 */
function pickFirstContent(contents?: RawContent[]): RawContent | undefined {
  return contents && contents.length > 0 ? contents[0] : undefined;
}

function resolveContent(
  raw: RawContent | undefined,
  baseUrl: string,
  options?: LoadTilesetOptions
): TileContent | null {
  if (!raw || typeof raw.uri !== 'string' || raw.uri.length === 0) return null;
  const uri = raw.uri;
  // 相对 tileset.json URL 解析为绝对 URL，再过 urlDecorator（key 注入等）。
  // baseUrl 可能本身是根相对路径（'/data/.../tileset.json'，无 origin）——
  // 直接 new URL(uri, baseUrl) 会抛 TypeError 落入保底分支，导致相对 uri
  // 被相对「页面」解析而 404。先对 baseUrl 补全页面 origin 再解析。
  let resolvedUrl: string;
  try {
    const base = new URL(baseUrl, typeof location !== 'undefined' ? location.href : 'http://localhost/').href;
    resolvedUrl = new URL(uri, base).href;
  } catch {
    resolvedUrl = uri; // 非法 URL 时保底原样（请求时再报错，不静默）
  }
  if (options?.urlDecorator) resolvedUrl = options.urlDecorator(resolvedUrl);

  const lower = uri.split('?')[0].toLowerCase();
  let kind: TileContentKind = 'unknown';
  if (lower.endsWith('.b3dm')) kind = 'b3dm';
  else if (lower.endsWith('.glb') || lower.endsWith('.gltf')) kind = 'glb';
  else if (lower.endsWith('.pnts')) kind = 'pnts';

  return { uri, resolvedUrl, kind };
}

// ── boundingVolume → 世界包围球 ─────────────────────────────────────────────

/**
 * 统一转世界（归一化 ECEF）包围球。
 * - sphere [cx,cy,cz,r]：局部米制 → transform 映射中心、r/R 缩放半径；
 * - box [cx,cy,cz, 3×3 半轴（列主序）]：半轴列最大模为保守半径（轴对齐
 *   半轴退化为 AABB 外接球，调度剔除用足够）；中心同 sphere；
 * - region [w,s,e,n,minH,maxH]（弧度）：绝对 WGS84，与 transform 无关。
 *   8 角点（经纬×高度组合）米制 ECEF 造保守球后整体 /R。
 * 缺失/非法 boundingVolume → 单位球兜底（warn，不炸整树）。
 */
function boundingVolumeToSphere(
  bv: RawBoundingVolume | undefined,
  transform: Matrix4
): { center: Cartesian3; radius: number } {
  const fallback = { center: new Cartesian3(0, 0, 0), radius: 1 };
  if (!bv) {
    console.warn('TilesetContentLoader: tile 缺少 boundingVolume，用单位球兜底');
    return fallback;
  }

  if (bv.sphere && bv.sphere.length >= 4) {
    const [cx, cy, cz, r] = bv.sphere;
    return transformSphere(new Cartesian3(cx, cy, cz), r, transform);
  }

  if (bv.box && bv.box.length >= 12) {
    const [cx, cy, cz, x0, x1, x2, y0, y1, y2, z0, z1, z2] = bv.box;
    // 保守半径 = 三个半轴列的最大模
    const colLen = (a: number, b: number, c: number) => Math.hypot(a, b, c);
    const r = Math.max(colLen(x0, x1, x2), colLen(y0, y1, y2), colLen(z0, z1, z2));
    return transformSphere(new Cartesian3(cx, cy, cz), r, transform);
  }

  if (bv.region && bv.region.length >= 6) {
    const [west, south, east, north, minH, maxH] = bv.region;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (const lon of [west, east]) {
      for (const lat of [south, north]) {
        for (const h of [minH, maxH]) {
          const p = Cartesian3.fromRadians(lon, lat, h);
          minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
          minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
          minZ = Math.min(minZ, p.z); maxZ = Math.max(maxZ, p.z);
        }
      }
    }
    const centreMetre = new Cartesian3((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);
    const radiusMetre = Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) / 2;
    const inv = 1.0 / TILESET_EARTH_SCALE;
    return {
      center: new Cartesian3(centreMetre.x * inv, centreMetre.y * inv, centreMetre.z * inv),
      radius: Math.max(radiusMetre * inv, 1e-9),
    };
  }

  console.warn('TilesetContentLoader: boundingVolume 字段无法识别，用单位球兜底');
  return fallback;
}

/** 局部米制球（中心 c、半径 r）经归一化累计矩阵映射为世界球。 */
function transformSphere(
  c: Cartesian3,
  r: number,
  transform: Matrix4
): { center: Cartesian3; radius: number } {
  const m = transform.values;
  // 列主序：世界中心 = M·c（上 3×3 + 平移列）；c 是米制数值
  const center = new Cartesian3(
    m[0] * c.x + m[4] * c.y + m[8]  * c.z + m[12],
    m[1] * c.x + m[5] * c.y + m[9]  * c.z + m[13],
    m[2] * c.x + m[6] * c.y + m[10] * c.z + m[14]
  );
  // 半径按旋转列最大模缩放（单位旋转基下 = 1/R，即米 → 归一化）
  const s0 = Math.hypot(m[0], m[1], m[2]);
  const s1 = Math.hypot(m[4], m[5], m[6]);
  const s2 = Math.hypot(m[8], m[9], m[10]);
  return { center, radius: Math.max(r * Math.max(s0, s1, s2), 1e-9) };
}
