import { GeometryPipeline } from './GeometryPipeline';
import type { InterleavedGeometryBuffers } from './GeometryPipeline';

export interface CreateArrowOptions {
  /** 箭头总长（米），沿局部 +Y 轴（基座在原点、尖端在 +lengthM）。> 0。 */
  lengthM: number;
  /** 锥杆半径（米）。缺省 = 4% 总长。 */
  shaftRadiusM?: number;
  /** 锥头底半径（米）。缺省 = 12% 总长。 */
  headRadiusM?: number;
  /** 锥头长度（米）。缺省 = 30% 总长（自动 clamp 到 ≤ 60% 总长）。 */
  headLenM?: number;
}

/**
 * 每根箭头的固定顶点预算（几何规模只由 opts 决定、与 lengthM 无关，
 * 调用方据此预分配/校验合并缓冲）：
 * - 锥杆 4 面棱柱：每面独立 4 顶点（平直着色不共边）= 16 顶点 / 24 索引；
 * - 锥头 4 面尖锥：每面独立 3 顶点 = 12 顶点 / 12 索引；
 * - 基座封口（朝 −Y，看尾翼不穿帮）：4 顶点 / 6 索引。
 * 合计 **32 顶点 / 42 索引 / 14 三角形**。
 */
export const WIND_ARROW_VERTEX_COUNT = 32;
export const WIND_ARROW_INDEX_COUNT = 42;

/**
 * WindArrowGeometry - 单根三维风场箭头：4 面锥杆 + 4 面锥头，沿局部
 * **+Y 轴**（基座 y=0，尖端 y=+lengthM），单位为米。
 *
 * 输出 8-float 交错顶点（pos3 + normal3 + uv2，`Primitive` 契约，
 * 见 `GeometryPipeline.VERTEX_STRIDE`），供 `WindFieldBuilder` 按格点
 * 旋转平移合并进分桶大 buffer（一次 drawIndexed 画一档）。
 *
 * ## Winding 约定（对齐 Primitive frontFace:'ccw' + cullMode:'back'）
 *
 * 截面正方形顶点取 θ = 45° − 90°·k（k=0..3，θ 递减序），此时环边
 * e = cₖ₊₁ − cₖ 满足 `e × +Y = 面外向法向`（与 `ExtrudedGeometry` 侧面
 * `e × up` 同推导），故：
 * - 锥杆侧面四边形 (bₖ, bₖ₊₁, tₖ₊₁, tₖ) 从外侧看 CCW ✓
 * - 锥头三角 (bₖ, bₖ₊₁, tip) 几何法向朝外 ✓
 * - 基座封口用 θ 递增序（环反向）三角形化，几何法向 = −Y ✓
 *
 * 全部法线为**几何法向**（叉积归一化），平直着色 —— 4 面箭头的棱线
 * 明暗对比正是风场的可读性来源。
 */
export class WindArrowGeometry {
  /**
   * 生成单根箭头几何。
   * @returns 交错缓冲；lengthM 非有限或 ≤ 0 时返回空缓冲
   *          （`GeometryPipeline.emptyBuffers` 同约定，调用方跳过）。
   */
  static create(options: CreateArrowOptions): InterleavedGeometryBuffers {
    const L = options.lengthM;
    if (!Number.isFinite(L) || L <= 0) {
      return GeometryPipeline.emptyBuffers();
    }
    // 比例参数：缺省按总长比例；headLen 上限 60% 保证锥杆非负长度。
    const headLen = Math.min(options.headLenM ?? L * 0.30, L * 0.60);
    const shaftR = options.shaftRadiusM ?? L * 0.04;
    const headR  = options.headRadiusM  ?? L * 0.12;
    const shaftTop = L - headLen;

    // ── 截面正方形（单位半径，θ 递减序 → 环边 e × +Y = 外向）────────
    // [cosθ, sinθ]，θ = 45°, -45°, -135°, -225°（x→cos, z→sin）
    const R2 = Math.SQRT1_2;
    const corner = [
      [ R2,  R2],
      [ R2, -R2],
      [-R2, -R2],
      [-R2,  R2],
    ];

    const positions = new Float32Array(WIND_ARROW_VERTEX_COUNT * 3);
    const normals   = new Float32Array(WIND_ARROW_VERTEX_COUNT * 3);
    const uvs       = new Float32Array(WIND_ARROW_VERTEX_COUNT * 2);
    const indices   = new Uint32Array(WIND_ARROW_INDEX_COUNT);

    const writeVert = (
      v: number, x: number, y: number, z: number,
      nx: number, ny: number, nz: number,
      u: number, w: number,
    ): void => {
      positions[v * 3] = x; positions[v * 3 + 1] = y; positions[v * 3 + 2] = z;
      normals[v * 3] = nx; normals[v * 3 + 1] = ny; normals[v * 3 + 2] = nz;
      uvs[v * 2] = u; uvs[v * 2 + 1] = w;
    };

    let vert = 0;
    let idx = 0;

    // ── 锥杆：4 侧面，法向 = normalize(e × +Y)（(ex,0,ez)×(0,1,0) = (-ez,0,ex)）──
    for (let k = 0; k < 4; k++) {
      const c0 = corner[k]!;
      const c1 = corner[(k + 1) % 4]!;
      const ex = (c1[0] - c0[0]) * shaftR;
      const ez = (c1[1] - c0[1]) * shaftR;
      const el = Math.hypot(ex, ez);
      const nx = -ez / el, nz = ex / el;
      const b = vert;                    // 本面 4 顶点基址
      writeVert(b,     c0[0] * shaftR, 0,        c0[1] * shaftR, nx, 0, nz, k / 4,       0);
      writeVert(b + 1, c1[0] * shaftR, 0,        c1[1] * shaftR, nx, 0, nz, (k + 1) / 4, 0);
      writeVert(b + 2, c1[0] * shaftR, shaftTop, c1[1] * shaftR, nx, 0, nz, (k + 1) / 4, 0.75);
      writeVert(b + 3, c0[0] * shaftR, shaftTop, c0[1] * shaftR, nx, 0, nz, k / 4,       0.75);
      indices[idx] = b; indices[idx + 1] = b + 1; indices[idx + 2] = b + 2;
      indices[idx + 3] = b; indices[idx + 4] = b + 2; indices[idx + 5] = b + 3;
      vert += 4; idx += 6;
    }

    // ── 锥头：4 三角面 (bₖ, bₖ₊₁, tip)，法向 = normalize(e1 × e2) ──────
    for (let k = 0; k < 4; k++) {
      const c0 = corner[k]!;
      const c1 = corner[(k + 1) % 4]!;
      const b0x = c0[0] * headR, b0z = c0[1] * headR;
      const b1x = c1[0] * headR, b1z = c1[1] * headR;
      const e1x = b1x - b0x, e1z = b1z - b0z;        // 环边（e1y = 0）
      const e2x = -b0x, e2y = L - shaftTop, e2z = -b0z; // → 尖端
      // n = e1 × e2（展开：e1y = 0）
      const nx = -e1z * e2y;
      const ny = e1z * e2x - e1x * e2z;
      const nz = e1x * e2y;
      const nl = Math.hypot(nx, ny, nz);
      const b = vert;
      writeVert(b,     b0x, shaftTop, b0z, nx / nl, ny / nl, nz / nl, k / 4,       0.75);
      writeVert(b + 1, b1x, shaftTop, b1z, nx / nl, ny / nl, nz / nl, (k + 1) / 4, 0.75);
      writeVert(b + 2, 0,   L,        0,   nx / nl, ny / nl, nz / nl, (k + 0.5) / 4, 1);
      indices[idx] = b; indices[idx + 1] = b + 1; indices[idx + 2] = b + 2;
      vert += 3; idx += 3;
    }

    // ── 基座封口：θ 递增序（环反向）→ 扇形三角形法向 −Y，看尾翼不穿帮 ─
    const cap = vert; // 4 顶点（顶点预算 16+12+4 = 32 的最后一段）
    for (let k = 0; k < 4; k++) {
      // 反向环 = corner[(4 - k) % 4]（θ 递增序：45°, 135°, 225°, 315°）
      const c = corner[(4 - k) % 4]!;
      writeVert(cap + k, c[0] * shaftR, 0, c[1] * shaftR, 0, -1, 0,
        (c[0] + 1) * 0.5, (c[1] + 1) * 0.5);
    }
    indices[idx] = cap;     indices[idx + 1] = cap + 1; indices[idx + 2] = cap + 2;
    indices[idx + 3] = cap; indices[idx + 4] = cap + 2; indices[idx + 5] = cap + 3;

    return {
      vertices: GeometryPipeline.interleave(positions, normals, uvs),
      indices,
      vertexCount: WIND_ARROW_VERTEX_COUNT,
      indexCount: WIND_ARROW_INDEX_COUNT,
    };
  }
}
