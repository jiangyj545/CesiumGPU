#!/usr/bin/env node
/**
 * buildTileset.mjs – 从本地 glb 生成自制离线演示 3DTiles（node ESM，零依赖）。
 *
 * 用法：node scripts/buildTileset.mjs        （幂等：重复运行覆盖输出）
 *
 * 产出 public/data/tilesets/demo/：
 *   tileset.json                       两级 LOD 演示树（refine = ADD）
 *   {Model}_l0.glb                     L0 = 原模型字节原样拷贝
 *   {Model}_l1.glb                     L1 = 「顶点抽稀 50%」简版：
 *                                      隔行丢弃索引（保留 floor(count/6)*3
 *                                      个索引，即整三角对），不做真实网格
 *                                      简化——L1 是 L0 的严格几何子集，
 *                                      顶点数据逐位相同，同屏叠加时深度值
 *                                      相等，不会 z-fight（refine=ADD 下
 *                                      父子同显即建立在该前提上）。
 *
 * 树结构：
 *   root（ENU@(114.3°E,30.5°N,0)，GE 3000，refine ADD）
 *   ├─ CesiumAir        tile（局部东移 -200m，GE 250）content l0
 *   │   └─ 同盒 child（GE 60）content l1
 *   ├─ CesiumMilkTruck  tile（0m，250）content l0 + l1 child
 *   └─ CesiumMan        tile（+200m，250）content l0 + l1 child
 *
 * 坐标约定：tileset.json 的 transform 为 **米制 ECEF**（列主序 4×4，
 * 旋转列 = ENU 单位基、平移列 = 米制 ECEF）——与真实 3DTiles 数据一致，
 * 加载层（TilesetContentLoader）负责 ×1/6378137 归一化。
 * glb 保持 glTF Y-up 原样，由 TilesetPrimitive 的 yUpFix（rotX(+90°)）
 * 修正，本脚本不做预变换。
 *
 * ENU 基与引擎 Transforms.eastNorthUpToFixedFrame 同构（geocentric 纬度
 * atan2(z, hypot(x,y))），平移用引擎 Cartesian3.fromRadians 同款 WGS84
 * 大地公式（N = a/√(1-e²sin²φ)），保证与引擎渲染对齐。
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = join(REPO_ROOT, 'public/data/tilesets/demo');

const LON_DEG = 114.3;   // 武汉附近（与离线地形/影像演示区域同区）
const LAT_DEG = 30.5;
const HEIGHT_M = 0;

/** 摆放清单：东西向错开 ~200m。 */
const MODELS = [
  { name: 'CesiumAir',       src: 'public/data/model/CesiumAir/Cesium_Air.glb',            offsetEast: -200, offsetNorth: 0 },
  { name: 'CesiumMilkTruck', src: 'public/data/model/CesiumMilkTruck/CesiumMilkTruck.glb', offsetEast:    0, offsetNorth: 0 },
  { name: 'CesiumMan',       src: 'public/data/model/CesiumMan/Cesium_Man.glb',            offsetEast:  200, offsetNorth: 0 },
];

const GE_ROOT = 3000;
const GE_L0 = 250;
const GE_L1 = 60;

// ── WGS84 / ENU（与引擎数学同构的米制版本）────────────────────────────────

const A = 6378137.0;          // WGS84 长半轴（米）
const B = 6356752.3142;       // WGS84 短半轴（米）
const E2 = 1.0 - (B * B) / (A * A);

/** 引擎 Cartesian3.fromRadians 同款大地坐标 → 米制 ECEF。 */
function geodeticToEcef(lonRad, latRad, heightM) {
  const sinLat = Math.sin(latRad);
  const cosLat = Math.cos(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);
  const N = A / Math.sqrt(1.0 - E2 * sinLat * sinLat);
  return {
    x: (N + heightM) * cosLat * cosLon,
    y: (N + heightM) * cosLat * sinLon,
    z: (N * (1.0 - E2) + heightM) * sinLat,
  };
}

/**
 * ENU → 米制 ECEF 的 4×4（列主序 16 元，tileset.json transform 格式）。
 * 基向量用 geocentric 纬度（atan2(z, hypot(x,y))），与引擎
 * Transforms.eastNorthUpToFixedFrame 推导一致。
 */
function enuToEcefMatrix(lonDeg, latDeg, heightM) {
  const p = geodeticToEcef(lonDeg * Math.PI / 180, latDeg * Math.PI / 180, heightM);
  const lon = Math.atan2(p.y, p.x);
  const latG = Math.atan2(p.z, Math.hypot(p.x, p.y));
  const sinLon = Math.sin(lon), cosLon = Math.cos(lon);
  const sinLat = Math.sin(latG), cosLat = Math.cos(latG);

  // 列主序：col0=east, col1=north, col2=up, col3=平移（米制 ECEF）
  return [
    -sinLon,          cosLon,           0,      0,
    -sinLat * cosLon, -sinLat * sinLon, cosLat, 0,
    cosLat * cosLon,  cosLat * sinLon,  sinLat, 0,
    p.x,              p.y,              p.z,    1,
  ];
}

/** 根局部（ENU）平移矩阵。 */
function translationMatrix(tx, ty, tz) {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, tx, ty, tz, 1];
}

// ── GLB 解析 / 重序列化 / L1 抽稀 ─────────────────────────────────────────

const GLB_MAGIC = 0x46546C67;
const CHUNK_JSON = 0x4E4F534A;
const CHUNK_BIN = 0x004E4942;

function parseGlb(buffer) {
  const dv = new DataView(buffer);
  if (dv.getUint32(0, true) !== GLB_MAGIC) {
    throw new Error(`buildTileset: 输入不是 GLB（magic 0x${dv.getUint32(0, true).toString(16)}）`);
  }
  const total = dv.getUint32(8, true);
  let json = null;
  let bin = null;
  let off = 12;
  while (off < total) {
    const len = dv.getUint32(off, true);
    const type = dv.getUint32(off + 4, true);
    const data = new Uint8Array(buffer, off + 8, len);
    if (type === CHUNK_JSON) json = JSON.parse(new TextDecoder().decode(data));
    else if (type === CHUNK_BIN) bin = data.slice(); // 拷贝，脱离原 buffer 生命周期
    off += 8 + len;
  }
  if (!json) throw new Error('buildTileset: GLB 缺少 JSON chunk');
  return { json, bin };
}

function serializeGlb(json, bin) {
  const jsonBytes = new TextEncoder().encode(JSON.stringify(json));
  const jsonPad = (4 - (jsonBytes.length % 4)) % 4;
  const binLen = bin ? bin.length : 0;
  const binPad = bin ? (4 - (binLen % 4)) % 4 : 0;
  const total = 12 + 8 + jsonBytes.length + jsonPad + (bin ? 8 + binLen + binPad : 0);
  const out = new ArrayBuffer(total);
  const dv = new DataView(out);
  const bytes = new Uint8Array(out);

  dv.setUint32(0, GLB_MAGIC, true);
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);

  let off = 12;
  dv.setUint32(off, jsonBytes.length + jsonPad, true);
  dv.setUint32(off + 4, CHUNK_JSON, true);
  bytes.set(jsonBytes, off + 8);
  bytes.fill(0x20, off + 8 + jsonBytes.length, off + 8 + jsonBytes.length + jsonPad);
  off += 8 + jsonBytes.length + jsonPad;

  if (bin) {
    dv.setUint32(off, binLen + binPad, true);
    dv.setUint32(off + 4, CHUNK_BIN, true);
    bytes.set(bin, off + 8);
  }
  return Buffer.from(out);
}

const INDEX_COMPONENT_SIZE = { 5121: 1, 5123: 2, 5125: 4 };

/**
 * L1「顶点抽稀 50%」简版：对每个带索引的 primitive 隔三角丢弃——保留
 * floor(count/6)*3 个索引（整三角对，奇数个三角时舍最后一个）。只改
 * accessor.count / bufferView.byteLength，顶点与 BIN 原样保留
 * （TODO：真正的网格简化需要拓扑重整，超出本脚本定位）。
 */
function thinIndices(json) {
  let trisBefore = 0;
  let trisAfter = 0;
  for (const mesh of json.meshes ?? []) {
    for (const prim of mesh.primitives ?? []) {
      if (prim.indices === undefined) continue; // 非索引 primitive 无法隔行丢
      const accessor = json.accessors[prim.indices];
      const compSize = INDEX_COMPONENT_SIZE[accessor.componentType];
      if (!compSize) continue;

      trisBefore += accessor.count / 3;
      const keep = Math.floor(accessor.count / 6) * 3;
      accessor.count = keep;
      delete accessor.min;
      delete accessor.max;
      trisAfter += keep / 3;

      // bufferView 仅被本 accessor 引用时才收短 byteLength（瘦身）
      const bv = json.bufferViews?.[accessor.bufferView];
      if (bv) {
        const shared = (json.accessors ?? []).some(
          (a, i) => i !== prim.indices && a.bufferView === accessor.bufferView
        );
        if (!shared) {
          const rel = accessor.byteOffset ?? 0;
          bv.byteLength = Math.min(bv.byteLength ?? Infinity, rel + keep * compSize);
        }
      }
    }
  }
  return { trisBefore, trisAfter };
}

/** glTF POSITION accessor min/max（Y-up 米）合并 → ENU 局部 box（12 元）。 */
function enuBoxOfModel(json) {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  let found = false;
  for (const mesh of json.meshes ?? []) {
    for (const prim of mesh.primitives ?? []) {
      const acc = json.accessors?.[prim.attributes?.POSITION];
      if (!acc || !acc.min || !acc.max) continue;
      found = true;
      minX = Math.min(minX, acc.min[0]); maxX = Math.max(maxX, acc.max[0]);
      minY = Math.min(minY, acc.min[1]); maxY = Math.max(maxY, acc.max[1]);
      minZ = Math.min(minZ, acc.min[2]); maxZ = Math.max(maxZ, acc.max[2]);
    }
  }
  if (!found) throw new Error('buildTileset: 所有 primitive 均缺 POSITION min/max，无法造 box');

  // glTF Y-up → ENU（yUpFix rotX(+90°) 的作用）：(x,y,z)_gltf → (x,−z,y)_ENU
  const corners = [
    [minX, minY, minZ], [maxX, minY, minZ], [minX, maxY, minZ], [maxX, maxY, minZ],
    [minX, minY, maxZ], [maxX, minY, maxZ], [minX, maxY, maxZ], [maxX, maxY, maxZ],
  ];
  let e0 = Infinity, n0 = Infinity, u0 = Infinity;
  let e1 = -Infinity, n1 = -Infinity, u1 = -Infinity;
  for (const [x, y, z] of corners) {
    const e = x, n = -z, u = y;
    e0 = Math.min(e0, e); e1 = Math.max(e1, e);
    n0 = Math.min(n0, n); n1 = Math.max(n1, n);
    u0 = Math.min(u0, u); u1 = Math.max(u1, u);
  }
  // 1.5× 外扩余量（LOD 选择 / 视锥剔除的保守性）
  const cx = (e0 + e1) / 2, cy = (n0 + n1) / 2, cz = (u0 + u1) / 2;
  const hx = Math.max((e1 - e0) / 2, 1) * 1.5;
  const hy = Math.max((n1 - n0) / 2, 1) * 1.5;
  const hz = Math.max((u1 - u0) / 2, 1) * 1.5;
  // 轴对齐 box：[center, hx·x̂, hy·ŷ, hz·ẑ]
  return [cx, cy, cz, hx, 0, 0, 0, hy, 0, 0, 0, hz];
}

/** 两个 box 的并集 AABB（同局部坐标系）。 */
function unionBox(a, b) {
  const half = (v, i) => Math.hypot(v[3 + i * 3], v[4 + i * 3], v[5 + i * 3]);
  const out = new Array(12).fill(0);
  for (let axis = 0; axis < 3; axis++) {
    const aC = a[axis], aH = half(a, axis);
    const bC = b[axis], bH = half(b, axis);
    const lo = Math.min(aC - aH, bC - bH);
    const hi = Math.max(aC + aH, bC + bH);
    out[axis] = (lo + hi) / 2;
    out[3 + axis * 3] = (hi - lo) / 2;
  }
  return out;
}

// ── 主流程 ───────────────────────────────────────────────────────────────────

function main() {
  mkdirSync(OUT_DIR, { recursive: true });

  const children = [];
  for (const model of MODELS) {
    const srcPath = join(REPO_ROOT, model.src);
    const raw = readFileSync(srcPath);
    // node Buffer → ArrayBuffer 拷贝（parseGlb 按 DataView 读）
    const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
    const { json, bin } = parseGlb(buffer);

    // L0：原模型字节原样（零改写）
    const l0Name = `${model.name}_l0.glb`;
    writeFileSync(join(OUT_DIR, l0Name), raw);

    // L1：隔行丢索引（未引用的 BIN 尾部原样保留——合法且省事）
    let l1Name = null;
    if (bin) {
      const thinned = JSON.parse(JSON.stringify(json)); // 深拷贝，别污染 L0 的 json
      const { trisBefore, trisAfter } = thinIndices(thinned);
      l1Name = `${model.name}_l1.glb`;
      writeFileSync(join(OUT_DIR, l1Name), serializeGlb(thinned, bin));
      console.log(
        `[buildTileset] ${model.name}: L0 ${(raw.length / 1024).toFixed(0)}KB, ` +
        `L1 三角 ${Math.round(trisBefore)} → ${Math.round(trisAfter)}（隔行丢弃，非真实简化）`
      );
    } else {
      console.warn(`[buildTileset] ${model.name}: 无 BIN chunk，跳过 L1`);
    }

    const box = enuBoxOfModel(json);
    const tile = {
      transform: translationMatrix(model.offsetEast, model.offsetNorth, 0),
      boundingVolume: { box },
      geometricError: GE_L0,
      content: { uri: l0Name },
      children: [],
    };
    if (l1Name) {
      tile.children.push({
        boundingVolume: { box },
        geometricError: GE_L1,
        content: { uri: l1Name },
      });
    }
    children.push(tile);
  }

  // 根 box = 子 box 并集（根局部 = ENU 原点系）
  let rootBox = children[0].boundingVolume.box;
  for (let i = 1; i < children.length; i++) rootBox = unionBox(rootBox, children[i].boundingVolume.box);

  const tileset = {
    asset: { version: '1.1' },
    geometricError: GE_ROOT,
    root: {
      transform: enuToEcefMatrix(LON_DEG, LAT_DEG, HEIGHT_M),
      boundingVolume: { box: rootBox },
      refine: 'ADD',
      geometricError: GE_ROOT,
      children,
    },
  };

  const outPath = join(OUT_DIR, 'tileset.json');
  writeFileSync(outPath, JSON.stringify(tileset, null, 2));
  console.log(`[buildTileset] 已生成 ${outPath}（root refine=ADD，两级 LOD，米制 ECEF transform）`);
}

main();
