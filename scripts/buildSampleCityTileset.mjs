#!/usr/bin/env node
/**
 * buildSampleCityTileset.mjs – 程序化样板城市 → 合法 3DTiles（node ESM，零依赖）。
 *
 * Phase 1「倾斜摄影样板城市入库」生产端（读取器验收数据）。与
 * buildTileset.mjs（3 演示模型）互补：本脚本产出真实规模的城市白模——
 * 程序化合成 ≥1000 栋建筑、4×5 分块、REPLACE 两级 LOD（L0 体块 →
 * L1 坡屋顶+屋顶设备细节），b3dm 容器 + BATCH_LENGTH + BatchTable
 * （name/height/floors，支撑 P1-E feature 挑拣）。
 *
 * 用法：node scripts/buildSampleCityTileset.mjs   （幂等：固定种子、覆盖输出）
 *
 * 产出 public/data/tilesets/sample-city/：
 *   tileset.json           root(REPLACE, GE 6000) → 20×L0(GE 600) → 20×L1(GE 100)
 *   tiles/l0_{r}_{c}.b3dm  远档：每栋 10 三角体块（4 墙 + 平顶，无底面）
 *   tiles/l1_{r}_{c}.b3dm  近档：墙(8) + 坡屋顶(4) + 屋顶设备盒(10) = 22 三角/栋
 *
 * 树结构（refine REPLACE：子就绪后隐藏父 → 近高远低）：
 *   root（ENU@(114.4057°E,30.5070°N) 光谷，米制 ECEF transform，无 content）
 *   ├─ L0 tile (r,c)  … 20 个，content = l0 b3dm
 *   │   └─ L1 tile    … 同盒 child，content = l1 b3dm
 *
 * 几何坐标约定：GLB 顶点在 glTF Y-up 轴系（x=东、y=上、z=−北），加载层
 * yUpFix(rotX+90°) 映射回 ENU (x,−z,y)；boundingVolume.box 直接以 ENU
 * 轴系写（与 buildTileset.mjs 的 enuBoxOfModel 映射一致）。glb 不带
 * NORMAL——GltfLoader 对缺 NORMAL 的索引 primitive 走平直法线生成。
 *
 * LOD 判定（SSE = 467.7×GE/d_m，vh=1080/fov60°，maximumSseError=16）：
 *   d<17.5km → L0 请求；d<2.9km → L1 细分（近高远低成立）。
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = join(REPO_ROOT, 'public/data/tilesets/sample-city');

// ── 城市参数（固定种子 → 幂等可回归）────────────────────────────────────────
const LON_DEG = 114.4057;   // 光谷广场附近
const LAT_DEG = 30.5070;

const GRID_COLS = 32;        // 东西向 cell 数
const GRID_ROWS = 40;        // 南北向 cell 数（合计 1280 栋 ≥ 1000）
const CELL = 100;            // cell 边长（米）＝ 街区 60 + 街道 40
const BLOCK = 60;            // 建筑footprint最大边（米，每 cell 居中一栋）

const TILE_COLS = 8;         // 每 L0 瓦片 8×8 cell
const TILE_ROWS = 8;
const TILES_X = GRID_COLS / TILE_COLS;   // 4
const TILES_Y = GRID_ROWS / TILE_ROWS;   // 5

const GE_ROOT = 6000;
const GE_L0 = 600;
const GE_L1 = 100;
const BOX_MARGIN = 2;        // tile 包盒外扩（米）

/** mulberry32 固定种子 PRNG（避免每次生成城市形态漂移）。 */
function makeRng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── 建筑清单生成 ─────────────────────────────────────────────────────────────

/**
 * 程序化建筑：每 cell 一栋，footprint 40~60m 居中，高度双峰分布
 * （85% 多层 12~40m / 15% 高层 50~90m）。
 * @returns {{e0,n0,e1,n1,u1,name,height,floors}[]}
 */
function generateBuildings() {
  const rng = makeRng(20260801);
  const out = [];
  for (let row = 0; row < GRID_ROWS; row++) {
    for (let col = 0; col < GRID_COLS; col++) {
      const cx = (col + 0.5) * CELL;   // 东向（米，相对城市中心）
      const cy = (row + 0.5) * CELL;   // 北向
      const w = 40 + rng() * (BLOCK - 40);
      const d = 40 + rng() * (BLOCK - 40);
      const tower = rng() < 0.15;
      const h = tower ? 50 + rng() * 40 : 12 + rng() * 28;
      out.push({
        e0: cx - w / 2, e1: cx + w / 2,
        n0: cy - d / 2, n1: cy + d / 2,
        u1: h,
        name: `B${String(out.length).padStart(4, '0')}`,
        height: Math.round(h * 10) / 10,
        floors: Math.max(1, Math.round(h / 3)),
      });
    }
  }
  return out;
}

// ── 几何装配（ENU → glTF Y-up：x=东、y=上、z=−北）───────────────────────────

class MeshBuilder {
  constructor() {
    this.pos = [];        // glTF 轴系顶点
    this.idx = [];        // 三角索引
    this.batchId = [];    // 每顶点 batchId（glTF float32）
    this.batchCount = 0;
  }

  /** 分配新批次 → 返回 batchId。 */
  nextBatch() { return this.batchCount++; }

  /** ENU 三点 → glTF 三角（外向序由调用方保证）。 */
  tri(a, b, c, batch) {
    const base = this.pos.length / 3;
    // ENU(x,y,z)=(东,北,上) → glTF(x,y,z)=(东,上,−北)
    for (const [e, n, u] of [a, b, c]) this.pos.push(e, u, -n);
    this.idx.push(base, base + 1, base + 2);
    for (let i = 0; i < 3; i++) this.batchId.push(batch);
  }

  /** 竖直面（ENU）：底边 a→b（u0→u1）；外向由 (a,b) 顺序决定。 */
  wall(a, b, u0, u1, batch) {
    this.tri([a[0], a[1], u0], [b[0], b[1], u0], [b[0], b[1], u1], batch);
    this.tri([a[0], a[1], u0], [b[0], b[1], u1], [a[0], a[1], u1], batch);
  }

  /** L0 体块：4 墙 + 平顶 = 10 三角。 */
  addBlock(b) {
    const id = this.nextBatch();
    const { e0, n0, e1, n1, u1 } = b;
    // 墙（ENU 俯视逆时针 → 外向）
    this.wall([e0, n0], [e1, n0], 0, u1, id);  // 南
    this.wall([e1, n0], [e1, n1], 0, u1, id);  // 东
    this.wall([e1, n1], [e0, n1], 0, u1, id);  // 北
    this.wall([e0, n1], [e0, n0], 0, u1, id);  // 西
    // 平顶（俯视逆时针朝上）
    this.tri([e0, n0, u1], [e1, n0, u1], [e1, n1, u1], id);
    this.tri([e0, n0, u1], [e1, n1, u1], [e0, n1, u1], id);
  }

  /**
   * L1 细节：墙体(8) + 双坡屋顶(4，屋脊沿东西长轴) + 屋顶设备盒(10)
   * = 22 三角/栋（L0 的 2.2 倍面数，近档可见差异）。
   */
  addBlockDetailed(b) {
    const id = this.nextBatch();
    const { e0, n0, e1, n1, u1 } = b;
    // 墙（无顶面——顶由坡屋顶承担）
    this.wall([e0, n0], [e1, n0], 0, u1, id);
    this.wall([e1, n0], [e1, n1], 0, u1, id);
    this.wall([e1, n1], [e0, n1], 0, u1, id);
    this.wall([e0, n1], [e0, n0], 0, u1, id);
    // 双坡屋顶：屋脊沿东西向
    const rise = Math.min(4, (n1 - n0) / 5);   // 坡高 ≤4m
    const ridgeMid = (n0 + n1) / 2;
    const ridge0 = [e0, ridgeMid, u1 + rise];
    const ridge1 = [e1, ridgeMid, u1 + rise];
    this.tri([e0, n0, u1], [e1, n0, u1], ridge1, id);   // 南坡
    this.tri([e0, n0, u1], ridge1, ridge0, id);
    this.tri([e1, n1, u1], [e0, n1, u1], ridge0, id);   // 北坡
    this.tri([e1, n1, u1], ridge0, ridge1, id);
    // 屋顶设备盒（电梯机房）
    const bx0 = e0 + (e1 - e0) * 0.25, bx1 = bx0 + (e1 - e0) * 0.25;
    const by0 = n0 + (n1 - n0) * 0.35, by1 = by0 + (n1 - n0) * 0.3;
    const bz = u1 + rise, bh = bz + 3;
    this.wall([bx0, by0], [bx1, by0], bz, bh, id);
    this.wall([bx1, by0], [bx1, by1], bz, bh, id);
    this.wall([bx1, by1], [bx0, by1], bz, bh, id);
    this.wall([bx0, by1], [bx0, by0], bz, bh, id);
    this.tri([bx0, by0, bh], [bx1, by0, bh], [bx1, by1, bh], id);
    this.tri([bx0, by0, bh], [bx1, by1, bh], [bx0, by1, bh], id);
  }
}

// ── GLB 序列化（最小面：POSITION + indices + _BATCHID，无纹理无 NORMAL）────

const GLB_MAGIC = 0x46546C67;

function buildGlb(mesh) {
  const positions = new Float32Array(mesh.pos);
  const indices = new Uint32Array(mesh.idx);
  const batchIds = new Float32Array(mesh.batchId);

  const binParts = [
    new Uint8Array(positions.buffer),
    new Uint8Array(indices.buffer),
    new Uint8Array(batchIds.buffer),
  ];
  const views = [];
  let offset = 0;
  for (const p of binParts) {
    offset = Math.ceil(offset / 4) * 4;   // 全部 4 字节对齐
    views.push({ buffer: 0, byteOffset: offset, byteLength: p.length });
    offset += p.length;
  }
  const bin = new Uint8Array(offset);
  binParts.forEach((p, i) => bin.set(p, views[i].byteOffset));

  // POSITION min/max（glTF 规范要求）
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < positions.length; i += 3) {
    minX = Math.min(minX, positions[i]); maxX = Math.max(maxX, positions[i]);
    minY = Math.min(minY, positions[i + 1]); maxY = Math.max(maxY, positions[i + 1]);
    minZ = Math.min(minZ, positions[i + 2]); maxZ = Math.max(maxZ, positions[i + 2]);
  }

  const json = {
    asset: { version: '2.0', generator: 'buildSampleCityTileset' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, name: 'tile' }],
    meshes: [{
      primitives: [{
        attributes: { POSITION: 0, _BATCHID: 2 },
        indices: 1,
        mode: 4,
      }],
    }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: positions.length / 3, type: 'VEC3', min: [minX, minY, minZ], max: [maxX, maxY, maxZ] },
      { bufferView: 1, componentType: 5125, count: indices.length, type: 'SCALAR' },
      { bufferView: 2, componentType: 5126, count: batchIds.length, type: 'SCALAR' },
    ],
    bufferViews: views,
    buffers: [{ byteLength: bin.length }],
  };
  return { json, bin };
}

// ── b3dm 容器（28B 头 + 表 + GLB）────────────────────────────────────────────

function pad4Json(payload) {
  const raw = new TextEncoder().encode(JSON.stringify(payload));
  const pad = (4 - (raw.length % 4)) % 4;
  const out = new Uint8Array(raw.length + pad);
  out.set(raw, 0);
  out.fill(0x20, raw.length);   // b3dm 表 JSON 用空格填充对齐 4
  return out;
}

function packB3dm(glbJson, bin, batchCount, batchTable) {
  // 先打 GLB（JSON chunk + BIN chunk），再裹 b3dm 头
  const jsonBytes = new TextEncoder().encode(JSON.stringify(glbJson));
  const jsonPad = (4 - (jsonBytes.length % 4)) % 4;
  const binPad = (4 - (bin.length % 4)) % 4;
  const glbLen = 12 + 8 + jsonBytes.length + jsonPad + 8 + bin.length + binPad;
  const glb = new ArrayBuffer(glbLen);
  const gdv = new DataView(glb);
  const gbytes = new Uint8Array(glb);
  gdv.setUint32(0, GLB_MAGIC, true);
  gdv.setUint32(4, 2, true);
  gdv.setUint32(8, glbLen, true);
  let off = 12;
  gdv.setUint32(off, jsonBytes.length + jsonPad, true);
  gdv.setUint32(off + 4, 0x4E4F534A, true); // 'JSON'
  gbytes.set(jsonBytes, off + 8);
  gbytes.fill(0x20, off + 8 + jsonBytes.length, off + 8 + jsonBytes.length + jsonPad); // 规范：空格填充
  off += 8 + jsonBytes.length + jsonPad;
  gdv.setUint32(off, bin.length + binPad, true);
  gdv.setUint32(off + 4, 0x004E4942, true); // 'BIN\0'
  gbytes.set(bin, off + 8);

  const ftJson = pad4Json({ BATCH_LENGTH: batchCount });
  const btJson = pad4Json(batchTable);
  const total = 28 + ftJson.length + btJson.length + glbLen;
  const out = new ArrayBuffer(total);
  const dv = new DataView(out);
  const bytes = new Uint8Array(out);
  dv.setUint32(0, 0x6d643362, true);  // 'b3dm'
  dv.setUint32(4, 1, true);           // version
  dv.setUint32(8, ftJson.length, true);
  dv.setUint32(12, 0, true);
  dv.setUint32(16, btJson.length, true);
  dv.setUint32(20, 0, true);
  dv.setUint32(24, 0, true);          // gltfFormat = 0（内嵌 GLB）
  bytes.set(ftJson, 28);
  bytes.set(btJson, 28 + ftJson.length);
  bytes.set(gbytes, 28 + ftJson.length + btJson.length);
  return Buffer.from(out);
}

// ── WGS84 / ENU（与 buildTileset.mjs / 引擎数学同构）────────────────────────

const A = 6378137.0;
const B = 6356752.3142;
const E2 = 1.0 - (B * B) / (A * A);

function geodeticToEcef(lonRad, latRad, heightM) {
  const sinLat = Math.sin(latRad), cosLat = Math.cos(latRad);
  const cosLon = Math.cos(lonRad), sinLon = Math.sin(lonRad);
  const N = A / Math.sqrt(1.0 - E2 * sinLat * sinLat);
  return {
    x: (N + heightM) * cosLat * cosLon,
    y: (N + heightM) * cosLat * sinLon,
    z: (N * (1.0 - E2) + heightM) * sinLat,
  };
}

/** ENU → 米制 ECEF 4×4（列主序，tileset.json transform 格式）。 */
function enuToEcefMatrix(lonDeg, latDeg, heightM) {
  const p = geodeticToEcef(lonDeg * Math.PI / 180, latDeg * Math.PI / 180, heightM);
  const lon = Math.atan2(p.y, p.x);
  const latG = Math.atan2(p.z, Math.hypot(p.x, p.y));
  const sinLon = Math.sin(lon), cosLon = Math.cos(lon);
  const sinLat = Math.sin(latG), cosLat = Math.cos(latG);
  return [
    -sinLon,          cosLon,           0,      0,
    -sinLat * cosLon, -sinLat * sinLon, cosLat, 0,
    cosLat * cosLon,  cosLat * sinLon,  sinLat, 0,
    p.x,              p.y,              p.z,    1,
  ];
}

/** ENU 局部 box（12 元轴对齐）+ 外扩余量。 */
function enuBox(e0, n0, u0, e1, n1, u1) {
  const cx = (e0 + e1) / 2, cy = (n0 + n1) / 2, cz = (u0 + u1) / 2;
  const hx = (e1 - e0) / 2 + BOX_MARGIN, hy = (n1 - n0) / 2 + BOX_MARGIN, hz = (u1 - u0) / 2 + BOX_MARGIN;
  return [cx, cy, cz, hx, 0, 0, 0, hy, 0, 0, 0, hz];
}

// ── 主流程 ───────────────────────────────────────────────────────────────────

function main() {
  const t0 = Date.now();
  const buildings = generateBuildings();
  const tilesDir = join(OUT_DIR, 'tiles');
  mkdirSync(tilesDir, { recursive: true });

  const children = [];
  let triL0 = 0, triL1 = 0, bytesL0 = 0, bytesL1 = 0;

  for (let ty = 0; ty < TILES_Y; ty++) {
    for (let tx = 0; tx < TILES_X; tx++) {
      const meshL0 = new MeshBuilder();
      const meshL1 = new MeshBuilder();
      const slice = buildings.filter((b) => {
        const col = Math.floor(b.e0 / CELL);   // 建筑西南角所在 cell
        const row = Math.floor(b.n0 / CELL);
        return Math.floor(col / TILE_COLS) === tx && Math.floor(row / TILE_ROWS) === ty;
      });
      if (slice.length === 0) throw new Error(`tile (${ty},${tx}) 无建筑——分块参数错误`);

      for (const b of slice) {
        meshL0.addBlock(b);
        meshL1.addBlockDetailed(b);
      }

      // 该瓦片 ENU 包盒（含坡顶/设备盒高程余量）
      const e0 = tx * TILE_COLS * CELL, e1 = (tx + 1) * TILE_COLS * CELL;
      const n0 = ty * TILE_ROWS * CELL, n1 = (ty + 1) * TILE_ROWS * CELL;
      const uTop = Math.max(...slice.map((b) => b.u1)) + 8;
      const box = enuBox(e0, n0, 0, e1, n1, uTop);

      const table = {
        name: slice.map((b) => b.name),
        height: slice.map((b) => b.height),
        floors: slice.map((b) => b.floors),
      };

      const g0 = buildGlb(meshL0);
      const l0 = packB3dm(g0.json, g0.bin, slice.length, table);
      const l0Name = `l0_${ty}_${tx}.b3dm`;
      writeFileSync(join(tilesDir, l0Name), l0);

      const g1 = buildGlb(meshL1);
      const l1 = packB3dm(g1.json, g1.bin, slice.length, table);
      const l1Name = `l1_${ty}_${tx}.b3dm`;
      writeFileSync(join(tilesDir, l1Name), l1);

      triL0 += meshL0.idx.length / 3;
      triL1 += meshL1.idx.length / 3;
      bytesL0 += l0.length; bytesL1 += l1.length;

      children.push({
        boundingVolume: { box },
        geometricError: GE_L0,
        content: { uri: `tiles/${l0Name}` },
        children: [{
          boundingVolume: { box },
          geometricError: GE_L1,
          content: { uri: `tiles/${l1Name}` },
        }],
      });
    }
  }

  // 根 box = 全城 ENU AABB（child 无 transform → 与根同局部系）
  const city = enuBox(0, 0, 0, GRID_COLS * CELL, GRID_ROWS * CELL, 100);
  const tileset = {
    asset: { version: '1.1' },
    geometricError: GE_ROOT * 2,
    root: {
      transform: enuToEcefMatrix(LON_DEG, LAT_DEG, 0),
      boundingVolume: { box: city },
      refine: 'REPLACE',
      geometricError: GE_ROOT,
      children,
    },
  };
  writeFileSync(join(OUT_DIR, 'tileset.json'), JSON.stringify(tileset, null, 1));

  console.log(`[buildSampleCity] 建筑 ${buildings.length} 栋（${GRID_COLS}×${GRID_ROWS} cell × ${CELL}m）`);
  console.log(`[buildSampleCity] 瓦片 ${children.length}×2（L0 ${TILES_X}×${TILES_Y} + L1 同构），refine=REPLACE`);
  console.log(`[buildSampleCity] 三角 L0 ${triL0}（${(triL0 / buildings.length).toFixed(0)}/栋）` +
    ` / L1 ${triL1}（${(triL1 / buildings.length).toFixed(0)}/栋，${(triL1 / triL0).toFixed(1)}×）`);
  console.log(`[buildSampleCity] 字节 L0 ${(bytesL0 / 1024).toFixed(0)}KB / L1 ${(bytesL1 / 1024).toFixed(0)}KB` +
    ` / ${Date.now() - t0}ms → ${OUT_DIR}`);
}

main();
