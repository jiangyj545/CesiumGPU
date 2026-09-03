import { Cartesian3 } from '../math/Cartesian3';
import { Matrix4 } from '../math/Matrix4';

/**
 * GimbalMath - 云台（无人机吊舱相机）视频流叠加到三维球的纯数学核心。
 *
 * - **纯数学模块**：无 WebGPU / DOM 依赖，可直接在 node 环境单元测试
 *   （见 `tests/media/gimbalMath.spec.ts`）。
 * - **归一化 ECEF 约定**（与 `Camera` 一致）：地球赤道半径 = 1，
 *   极轴 Z = b/a ≈ 0.9966472；米 → 归一化坐标乘 `1 / EARTH_SCALE`。
 * - 旋转链与引擎主相机逐一对齐，保证叠加结果与渲染一致：
 *   - ENU 基底      → 镜像 `Camera._eastNorthUp`（east = normalize(cross(UNIT_Z, up))）
 *   - Rodrigues 旋转 → 镜像 `Camera._rotateByAxis`（公开为 `rotateAroundAxis`）
 *   - HPR 旋转链    → 镜像 `Camera._applyOrientation`
 *   - 角射线生成    → 镜像 `Camera.pickGlobe` 的 NDC → 射线公式
 *   - 射线求交      → 椭球（仅 Z 压缩）替代 pickGlobe 的单位球
 *
 * 所有导出函数均为纯函数；批量路径（足迹网格 / 纹理矩阵 / 视锥线框）
 * 通过 result 出参 + 模块级 scratch 复用对象，避免每帧分配。
 */

/** 云台位姿与内参（角度全弧度）。`position` 为归一化 ECEF。 */
export interface GimbalPose {
  /** 归一化 ECEF 位置（赤道半径 = 1，米制坐标需先除以 EARTH_SCALE）。 */
  position: Cartesian3;
  /** 0 = 北，正 = 顺时针向东（与 `Camera._applyOrientation` / DJI gimbal_yaw 一致）。 */
  heading: number;
  /** 0 = 地平线，负 = 向下，-PI/2 = nadir。 */
  pitch: number;
  /** 绕视线（roll）。 */
  roll: number;
  /** 垂直 FOV（弧度）。 */
  fovY: number;
  /** 宽 / 高。 */
  aspect: number;
}

/**
 * 视频足迹：4 角地面点。
 * 角点顺序 TL, TR, BR, BL，与 NDC (-1,1), (1,1), (1,-1), (-1,-1) 对应。
 */
export interface Footprint {
  corners: [Cartesian3, Cartesian3, Cartesian3, Cartesian3];
  /** 足迹中心近似（视线中心射线落点）。 */
  center: Cartesian3;
  /** 任一角射线（或中心射线）未命中椭球则 false。 */
  valid: boolean;
  /** 中心射线斜距（米）。 */
  centerRangeMeters: number;
}

/** WGS84 极半轴 / 赤道半轴 = 6356752.3142 / 6378137 ≈ 0.9966472。 */
export const WGS84_POLAR_RATIO = 6356752.3142 / 6378137;

/** 1 归一化单位对应的米数（WGS84 赤道半径）。 */
export const EARTH_SCALE = 6378137;

/** 地球遮挡判定半径的平方（略小于 1 的容差值，供遮挡模块使用）。 */
export const GLOBE_OCCLUDE_R2 = 0.998 * 0.998;

/** NDC y（向上为正）到纹理 v（向下为正）的翻转符号。 */
export const NDC_TO_UV_Y_SIGN = -1;

/** 归一化椭球极半径（= WGS84_POLAR_RATIO 的语义别名）。 */
const _RHO = WGS84_POLAR_RATIO;

// ── 模块级 scratch（热路径零分配；单线程约定，导出函数不重入共享 scratch） ──
const _tmpA = new Cartesian3();
const _rayDir = new Cartesian3();
const _surf = new Cartesian3();
const _target = new Cartesian3();
const _nTL = new Cartesian3();
const _nTR = new Cartesian3();
const _nBR = new Cartesian3();
const _nBL = new Cartesian3();
const _halfScale = new Cartesian3(0.5, NDC_TO_UV_Y_SIGN * 0.5, 0.5);
const _uvOffset = new Cartesian3(0.5, 0.5, 0.5);
const _mView = new Matrix4();
const _mProj = new Matrix4();
const _mTemp = new Matrix4();
const _mScale = new Matrix4();
const _mTrans = new Matrix4();

/**
 * 某点的局部东 / 北 / 天坐标系（geocentric up）。
 * up = normalize(position)；east = normalize(cross(UNIT_Z, up)) = normalize(-up.y, up.x, 0)，
 * 极点退化时 east 取固定 (1,0,0) —— 与 `Camera._eastNorthUp` 完全一致。
 */
export function enuBasisAt(
  positionNorm: Cartesian3,
): { east: Cartesian3; north: Cartesian3; up: Cartesian3 } {
  const up = Cartesian3.normalize(positionNorm, new Cartesian3());
  const east = new Cartesian3();
  const ex = -up.y;
  const ey = up.x;
  const eMag = Math.hypot(ex, ey);
  if (eMag < 1e-6) {
    east.x = 1;
    east.y = 0;
    east.z = 0;
  } else {
    east.x = ex / eMag;
    east.y = ey / eMag;
    east.z = 0;
  }
  const north = Cartesian3.normalize(Cartesian3.cross(up, east, new Cartesian3()), new Cartesian3());
  return { east, north, up };
}

/**
 * Rodrigues 公式：把 v 绕单位轴 axis 旋转 angle（弧度），结果写入 result。
 * result 可与 v 别名（镜像 `Camera._rotateByAxis`）。
 */
export function rotateAroundAxis(
  v: Cartesian3,
  axis: Cartesian3,
  angle: number,
  result: Cartesian3,
): Cartesian3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const dot = Cartesian3.dot(axis, v);
  Cartesian3.cross(axis, v, _tmpA);
  // 先算进临时量再写入，保证 result 可安全别名 v
  const rx = v.x * c + _tmpA.x * s + axis.x * dot * (1 - c);
  const ry = v.y * c + _tmpA.y * s + axis.y * dot * (1 - c);
  const rz = v.z * c + _tmpA.z * s + axis.z * dot * (1 - c);
  result.x = rx;
  result.y = ry;
  result.z = rz;
  return result;
}

/**
 * 云台视线坐标系（精确镜像 `Camera._applyOrientation` 的旋转链）：
 * 1. heading：north / east 绕 up 轴旋转 -heading；
 * 2. pitch：lookH / up 绕 rightH 轴旋转 pitch（负 = 向下）；
 * 3. right = normalize(cross(dir, camUp))，finalUp = cross(right, dir)；
 * 4. roll 非 0 时 right / up 绕 dir 旋转 roll。
 */
export function gimbalBasis(pose: GimbalPose): { dir: Cartesian3; up: Cartesian3; right: Cartesian3 } {
  const enu = enuBasisAt(pose.position);
  const up = enu.up;
  const east = enu.east;
  const north = enu.north;

  const lookH = rotateAroundAxis(north, up, -pose.heading, new Cartesian3());
  const rightH = rotateAroundAxis(east, up, -pose.heading, new Cartesian3());

  const dir = rotateAroundAxis(lookH, rightH, pose.pitch, new Cartesian3());
  const camUp = rotateAroundAxis(up, rightH, pose.pitch, new Cartesian3());

  const right = Cartesian3.normalize(Cartesian3.cross(dir, camUp, new Cartesian3()), new Cartesian3());
  let finalUp: Cartesian3;
  if (pose.roll !== 0.0) {
    rotateAroundAxis(right, dir, pose.roll, right);
    finalUp = rotateAroundAxis(camUp, dir, pose.roll, new Cartesian3());
  } else {
    finalUp = camUp;
  }
  return {
    dir: Cartesian3.normalize(dir, dir),
    up: Cartesian3.normalize(finalUp, finalUp),
    right: Cartesian3.normalize(right, right),
  };
}

/**
 * 由 NDC 坐标生成穿过该像素的归一化世界射线方向（镜像 `Camera.pickGlobe`）：
 * dir_c = normalize(dir + ndcX * tan(fovY/2) * aspect * right + ndcY * tan(fovY/2) * up)。
 */
export function gimbalCornerRay(
  pose: GimbalPose,
  ndcX: number,
  ndcY: number,
  resultDir: Cartesian3,
): void {
  const basis = gimbalBasis(pose);
  _cornerRayFromBasis(pose, basis, ndcX, ndcY, resultDir);
}

/** `gimbalCornerRay` 的内部版本：复用已算好的云台基底。写入归一化方向。 */
function _cornerRayFromBasis(
  pose: GimbalPose,
  basis: { dir: Cartesian3; up: Cartesian3; right: Cartesian3 },
  ndcX: number,
  ndcY: number,
  resultDir: Cartesian3,
): void {
  const tanHalfFovY = Math.tan(pose.fovY * 0.5);
  const tanHalfFovX = tanHalfFovY * pose.aspect;
  resultDir.x = basis.dir.x + ndcX * tanHalfFovX * basis.right.x + ndcY * tanHalfFovY * basis.up.x;
  resultDir.y = basis.dir.y + ndcX * tanHalfFovX * basis.right.y + ndcY * tanHalfFovY * basis.up.y;
  resultDir.z = basis.dir.z + ndcX * tanHalfFovX * basis.right.z + ndcY * tanHalfFovY * basis.up.z;
  Cartesian3.normalize(resultDir, resultDir);
}

/**
 * 射线与归一化椭球的最近正交点参数 t（沿 `dir`，`dir` 需先归一化）。
 *
 * 归一化椭球空间仅 Z 压缩：把原点 / 方向的 z 乘 1/rho 映射到单位球空间，
 * 解二次方程取最小非负根；返回的 t 已通过二次项系数 A 归算回真实
 * 归一化空间（即命中点 = origin + t * dir，其 z 分量天然落在椭球上）。
 * 判别式 < 0（未命中）或无非负根（椭球内背向情形）返回 null。
 */
export function intersectRayEllipsoid(origin: Cartesian3, dir: Cartesian3): number | null {
  const oz = origin.z / _RHO;
  const dx = dir.x;
  const dy = dir.y;
  const dz = dir.z / _RHO;

  const a = dx * dx + dy * dy + dz * dz;
  if (a === 0.0) return null;
  const b = 2.0 * (origin.x * dx + origin.y * dy + oz * dz);
  const c = origin.x * origin.x + origin.y * origin.y + oz * oz - 1.0;

  const discriminant = b * b - 4.0 * a * c;
  if (discriminant < 0.0) return null;

  const sq = Math.sqrt(discriminant);
  const t0 = (-b - sq) / (2.0 * a);
  if (t0 >= 0.0) return t0;
  const t1 = (-b + sq) / (2.0 * a);
  if (t1 >= 0.0) return t1;
  return null;
}

/** 单条角射线求交并写入 corner；未命中时写入 position + dir 并返回 false。 */
function _cornerInto(
  pose: GimbalPose,
  basis: { dir: Cartesian3; up: Cartesian3; right: Cartesian3 },
  ndcX: number,
  ndcY: number,
  corner: Cartesian3,
): boolean {
  _cornerRayFromBasis(pose, basis, ndcX, ndcY, _rayDir);
  const t = intersectRayEllipsoid(pose.position, _rayDir);
  if (t === null) {
    corner.x = pose.position.x + _rayDir.x;
    corner.y = pose.position.y + _rayDir.y;
    corner.z = pose.position.z + _rayDir.z;
    return false;
  }
  corner.x = pose.position.x + t * _rayDir.x;
  corner.y = pose.position.y + t * _rayDir.y;
  corner.z = pose.position.z + t * _rayDir.z;
  return true;
}

/**
 * 计算视频足迹：4 角 NDC (-1,1), (1,1), (1,-1), (-1,-1) 射线求交椭球；
 * 中心射线 (0,0) 求交得 center / centerRangeMeters。
 * 任一射线未命中 → valid = false（未命中角点写入 position + dir 的有限占位点）。
 * `result` 出参复用已有 Footprint（含其 corners / center 对象）避免分配。
 */
export function computeFootprint(pose: GimbalPose, result?: Footprint): Footprint {
  const fp: Footprint =
    result ?? {
      corners: [new Cartesian3(), new Cartesian3(), new Cartesian3(), new Cartesian3()],
      center: new Cartesian3(),
      valid: false,
      centerRangeMeters: 0.0,
    };

  const basis = gimbalBasis(pose);
  let valid = true;
  valid = _cornerInto(pose, basis, -1.0, 1.0, fp.corners[0]) && valid;  // TL
  valid = _cornerInto(pose, basis, 1.0, 1.0, fp.corners[1]) && valid;   // TR
  valid = _cornerInto(pose, basis, 1.0, -1.0, fp.corners[2]) && valid;  // BR
  valid = _cornerInto(pose, basis, -1.0, -1.0, fp.corners[3]) && valid; // BL

  // 中心射线（视线中心）
  _cornerRayFromBasis(pose, basis, 0.0, 0.0, _rayDir);
  const tc = intersectRayEllipsoid(pose.position, _rayDir);
  if (tc === null) {
    valid = false;
    fp.center.x = pose.position.x + _rayDir.x;
    fp.center.y = pose.position.y + _rayDir.y;
    fp.center.z = pose.position.z + _rayDir.z;
    fp.centerRangeMeters = 0.0;
  } else {
    Cartesian3.multiplyByScalar(_rayDir, tc, _tmpA);
    Cartesian3.add(pose.position, _tmpA, fp.center);
    fp.centerRangeMeters = Cartesian3.distance(pose.position, fp.center) * EARTH_SCALE;
  }

  fp.valid = valid;
  return fp;
}

/**
 * 把点径向回投到归一化椭球面并沿径向抬升 heightOffsetMeters：
 * s = 1 / sqrt(x² + y² + (z/rho)²)，result = p · s · (1 + heightOffsetMeters / EARTH_SCALE)。
 * 这是将来接入 DEM 高程时的可替换点（替换本函数即可换采样源）。
 */
export function projectToSurface(
  p: Cartesian3,
  heightOffsetMeters: number,
  result: Cartesian3,
): Cartesian3 {
  const zr = p.z / _RHO;
  const s = 1.0 / Math.sqrt(p.x * p.x + p.y * p.y + zr * zr);
  const k = s * (1.0 + heightOffsetMeters / EARTH_SCALE);
  result.x = p.x * k;
  result.y = p.y * k;
  result.z = p.z * k;
  return result;
}

/**
 * 由足迹 4 角双线性插值生成 (subdivisions+1)² 顶点的贴地网格。
 *
 * - 每个插值点经 `projectToSurface` 径向回投椭球面并抬升 heightOffsetMeters；
 * - positionsOut 布局 xyz 紧凑（行主序：j 外层沿 TL→BL、i 内层沿 TL→TR，TL 为原点）；
 * - 索引（quad 双三角形）写入 indicesOut：每 quad 6 个索引，
 *   基址 base = j*(n+1)+i，三角形 (base, base+n+1, base+1) / (base+1, base+n+1, base+n+2)；
 * - fp.valid = false 时直接返回 { vertexCount: 0, indexCount: 0 } 且不写出。
 * 调用方需保证 positionsOut ≥ 3*(n+1)² 、indicesOut ≥ 6n²。
 */
export function buildFootprintGrid(
  fp: Footprint,
  subdivisions: number,
  heightOffsetMeters: number,
  positionsOut: Float32Array,
  indicesOut: Uint32Array,
): { vertexCount: number; indexCount: number } {
  if (!fp.valid) return { vertexCount: 0, indexCount: 0 };

  const n = subdivisions;
  const tl = fp.corners[0];
  const tr = fp.corners[1];
  const br = fp.corners[2];
  const bl = fp.corners[3];

  let vi = 0;
  for (let j = 0; j <= n; j++) {
    const fv = j / n;
    for (let i = 0; i <= n; i++) {
      const fu = i / n;
      const wTL = (1.0 - fu) * (1.0 - fv);
      const wTR = fu * (1.0 - fv);
      const wBR = fu * fv;
      const wBL = (1.0 - fu) * fv;
      _surf.x = tl.x * wTL + tr.x * wTR + br.x * wBR + bl.x * wBL;
      _surf.y = tl.y * wTL + tr.y * wTR + br.y * wBR + bl.y * wBL;
      _surf.z = tl.z * wTL + tr.z * wTR + br.z * wBR + bl.z * wBL;
      projectToSurface(_surf, heightOffsetMeters, _surf);
      positionsOut[vi++] = _surf.x;
      positionsOut[vi++] = _surf.y;
      positionsOut[vi++] = _surf.z;
    }
  }

  let ii = 0;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const v00 = j * (n + 1) + i;
      const v10 = v00 + 1;
      const v01 = v00 + (n + 1);
      const v11 = v01 + 1;
      indicesOut[ii++] = v00;
      indicesOut[ii++] = v01;
      indicesOut[ii++] = v10;
      indicesOut[ii++] = v10;
      indicesOut[ii++] = v01;
      indicesOut[ii++] = v11;
    }
  }

  return { vertexCount: (n + 1) * (n + 1), indexCount: 6 * n * n };
}

/**
 * 构建把世界坐标投影到视频纹理 [0,1]² 的纹理矩阵：
 *
 *   texMatrix = translate(0.5, 0.5, 0.5) · scale(0.5, NDC_TO_UV_Y_SIGN·0.5, 0.5) · proj · view
 *
 * - view：Matrix4.lookAt(position, position + dir（单位长度前瞻点）, up)
 * - proj：Matrix4.perspective(fovY, aspect, near / EARTH_SCALE, far / EARTH_SCALE)
 *   （注意归一化空间尺度小，near/far 米制入参在内部归一化）
 * `result` 出参复用。
 */
export function buildTextureMatrix(
  pose: GimbalPose,
  near: number,
  far: number,
  result?: Matrix4,
): Matrix4 {
  const out = result ?? new Matrix4();
  const basis = gimbalBasis(pose);

  Cartesian3.add(pose.position, basis.dir, _target);
  Matrix4.lookAt(pose.position, _target, basis.up, _mView);
  Matrix4.perspective(pose.fovY, pose.aspect, near / EARTH_SCALE, far / EARTH_SCALE, _mProj);

  Matrix4.multiply(_mProj, _mView, _mTemp);   // proj · view
  Matrix4.fromScale(_halfScale, _mScale);     // scale(0.5, -0.5, 0.5)
  Matrix4.fromTranslation(_uvOffset, _mTrans); // translate(0.5, 0.5, 0.5)
  Matrix4.multiply(_mScale, _mTemp, out);     // scale · proj · view
  Matrix4.multiply(_mTrans, out, out);        // translate · scale · proj · view
  return out;
}

/**
 * 生成视锥线框（line-list）顶点对，写入 out（xyz 紧凑），返回写入的 float 数（顶点数*3）：
 * 1. 近平面矩形 4 条边（TL→TR→BR→BL→TL，近平面角 = 沿 dir 的 nearMeters 处矩形四角）；
 * 2. fp.valid 时追加：无人机位置 → 4 地面角 连线（4 条，共 24 floats）。
 * fp.valid = false 时只写近平面部分（24 floats）。
 * 调用方需保证 out ≥ 48 floats。
 */
export function buildFrustumLineVertices(
  pose: GimbalPose,
  fp: Footprint,
  nearMeters: number,
  out: Float32Array,
): number {
  const basis = gimbalBasis(pose);
  const tanHalfFovY = Math.tan(pose.fovY * 0.5);
  const nearN = nearMeters / EARTH_SCALE;
  const halfW = nearN * tanHalfFovY * pose.aspect;
  const halfH = nearN * tanHalfFovY;

  // 近平面 4 角（顺序与 NDC TL/TR/BR/BL 一致）
  _nearCorner(pose.position, basis, nearN, -halfW, halfH, _nTL);
  _nearCorner(pose.position, basis, nearN, halfW, halfH, _nTR);
  _nearCorner(pose.position, basis, nearN, halfW, -halfH, _nBR);
  _nearCorner(pose.position, basis, nearN, -halfW, -halfH, _nBL);

  let f = 0;
  f = _writeSegment(out, f, _nTL, _nTR);
  f = _writeSegment(out, f, _nTR, _nBR);
  f = _writeSegment(out, f, _nBR, _nBL);
  f = _writeSegment(out, f, _nBL, _nTL);

  if (fp.valid) {
    for (let i = 0; i < 4; i++) {
      f = _writeSegment(out, f, pose.position, fp.corners[i]);
    }
  }
  return f;
}

/** 近平面角点：pos + dir*dist + right*ox + up*oy（dist = nearN，ox/oy 为相机平面内偏移）。 */
function _nearCorner(
  pos: Cartesian3,
  basis: { dir: Cartesian3; up: Cartesian3; right: Cartesian3 },
  dist: number,
  ox: number,
  oy: number,
  result: Cartesian3,
): void {
  result.x = pos.x + basis.dir.x * dist + basis.right.x * ox + basis.up.x * oy;
  result.y = pos.y + basis.dir.y * dist + basis.right.y * ox + basis.up.y * oy;
  result.z = pos.z + basis.dir.z * dist + basis.right.z * ox + basis.up.z * oy;
}

/** 写入一条线段的两个端点（6 floats），返回新的游标。 */
function _writeSegment(out: Float32Array, cursor: number, a: Cartesian3, b: Cartesian3): number {
  out[cursor] = a.x;
  out[cursor + 1] = a.y;
  out[cursor + 2] = a.z;
  out[cursor + 3] = b.x;
  out[cursor + 4] = b.y;
  out[cursor + 5] = b.z;
  return cursor + 6;
}
