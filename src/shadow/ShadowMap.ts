import { Context } from '../renderer/Context';
import { Camera } from '../scene/Camera';
import { DirectionalLight } from '../scene/DirectionalLight';
import { Primitive } from '../scene/Primitive';
import { Cartesian3 } from '../math/Cartesian3';
import { Matrix4 } from '../math/Matrix4';
import { SHADOW_DEPTH_SHADER } from '../shaders/ShadowShader';

/**
 * ShadowMap — A1 CSSM 级联阴影管理器（引擎首个阴影系统）。
 *
 * 职责：
 *  1. 级联分割（视锥自适应，米制约定 → 归一化世界换算 `/EARTH_SCALE`）。
 *  2. 太阳方向正交矩阵拟合（光源视锥角点拟合 + 纹素对齐 + 近平面沿太阳
 *     方向拉宽一个 span，防止高于地表的 caster 被近平面裁掉）。
 *  3. 深度渲染：3 层 depth32float 纹理数组，每级联一个 depth-only render
 *     pass（无颜色目标、无 fragment 阶段）。
 *  4. receiver 绑定：uniform（3×view-proj + 分割 + 纹素 + 参数）、
 *     2d-array 视图、comparison 采样器，供 LightingPass 组 2 采样。
 *
 * 挂载（Scene.render，Lighting 之前）：
 *   shadow.updateFrame(camera, lights) → shadow.execute(encoder, getOpaquePipeline)
 *   → lightingPass.setShadowSource(shadow.receiverBinding())
 *
 * prepare 幂等 + `_failed` 闸（OceanFFTWater 模式）：任何 GPU 构建/校验
 * 失败仅记录并强制 `enabled = false`，整帧零污染（无阴影 = 零回归）。
 */

/** Receiver 组 2 UBO 大小（3×mat4 = 192 + 3×vec4 = 48 → 240，对齐到 256）。 */
export const SHADOW_RECEIVER_UBO_SIZE = 256;

/** 每级联光源 view-proj 在共享 UBO 中的步长（minUniformBufferOffsetAlignment 安全值）。 */
const CAM_UBO_STRIDE = 256;

export const SHADOW_CASCADE_COUNT = 3;

/**
 * LightingPass 组 2 的绑定载荷（解耦：LightingPass 只 import 本类型，
 * 不 import ShadowMap 类，避免 scene→shadow→scene 环）。
 */
export interface ShadowReceiverBinding {
  /** 级联矩阵/分割/参数 uniform buffer。 */
  ubo: GPUBuffer;
  /** 3 层 depth32float 纹理的 2d-array 视图。 */
  textureView: GPUTextureView;
  /** PCF comparison 采样器（compare: 'less'）。 */
  comparisonSampler: GPUSampler;
}

/** 纯几何 — 视锥自适应级联分割（米制）。默认远端 10 km 覆盖：250m/1.5km/10km。 */
export function computeCascadeSplitEnds(
  frustumNearM: number,
  frustumFarM: number,
  maxCoverM: number = 10000,
): [number, number, number] {
  const fractions = [0.025, 0.15, 1.0];
  const far = Math.max(Math.min(frustumFarM, maxCoverM), 1);
  let e0 = far * fractions[0];
  let e1 = far * fractions[1];
  let e2 = far * fractions[2];
  // 近平面抬升：级联起点不得落在相机近平面之前（否则整层深度图恒空）。
  const base = Math.max(frustumNearM, 0.5);
  e0 = Math.max(e0, base * 1.2);
  e1 = Math.max(e1, e0 * 3);
  e2 = Math.max(e2, e1 * 3);
  return [e0, e1, e2];
}

/** 级联拟合输出。 */
export interface LightOrthoFit {
  /** 正交光源 view-proj（列主序，WebGPU NDC z ∈ [0,1]）。 */
  viewProj: Matrix4;
  /** 该级联一个纹素对应的世界尺寸（归一化单位，normal-offset 用）。 */
  texelWorld: number;
}

/** 纯几何 — 太阳方向正交基。z = 光传播方向（太阳→场景，同 DirectionalLight.direction）。 */
export function computeLightBasis(lightDir: Cartesian3): {
  x: Cartesian3; y: Cartesian3; z: Cartesian3;
} {
  const z = Cartesian3.normalize(lightDir, new Cartesian3());
  // 光轴近竖直时辅助向上向量退化 → 换 X 轴。
  const helper = Math.abs(z.z) > 0.99 ? Cartesian3.UNIT_X : Cartesian3.UNIT_Z;
  const x = Cartesian3.normalize(Cartesian3.cross(helper, z, new Cartesian3()), new Cartesian3());
  const y = Cartesian3.cross(z, x, new Cartesian3());
  return { x, y, z };
}

const _csmDir   = new Cartesian3();
const _csmUp    = new Cartesian3();
const _csmRight = new Cartesian3();

/** 纯几何 — 相机视锥子块 [nearNorm, farNorm] 的 8 个世界角点（归一化 ECEF）。 */
export function computeCascadeCorners(
  camera: Camera,
  nearNorm: number,
  farNorm: number,
  out: Cartesian3[],
): void {
  const pos   = camera.position;
  const dir   = Cartesian3.normalize(camera.direction, _csmDir);
  const up    = Cartesian3.normalize(camera.up, _csmUp);
  const right = Cartesian3.normalize(
    Cartesian3.cross(dir, up, _csmRight), _csmRight);

  const tanF    = Math.tan(camera.frustum.fov / 2);
  const aspect  = camera.frustum.aspectRatio;
  const dists   = [nearNorm, farNorm];
  let k = 0;
  for (const d of dists) {
    const hw = d * tanF * aspect;
    const hh = d * tanF;
    for (const sy of [-1, 1]) {
      for (const sx of [-1, 1]) {
        const c = out[k++];
        c.x = pos.x + dir.x * d + right.x * hw * sx + up.x * hh * sy;
        c.y = pos.y + dir.y * d + right.y * hw * sx + up.y * hh * sy;
        c.z = pos.z + dir.z * d + right.z * hw * sx + up.z * hh * sy;
      }
    }
  }
}

/** 纯几何 — 把 8 角点拟合到光源正交盒：10% 外扩 + 纹素对齐 + 近平面拉宽。 */
export function fitLightOrtho(
  corners: Cartesian3[],
  basis: { x: Cartesian3; y: Cartesian3; z: Cartesian3 },
  mapSize: number,
): LightOrthoFit {
  const n = corners.length;
  let cx = 0, cy = 0, cz = 0;
  for (const c of corners) { cx += c.x; cy += c.y; cz += c.z; }
  cx /= n; cy /= n; cz /= n;

  let lxMin = Infinity, lxMax = -Infinity;
  let lyMin = Infinity, lyMax = -Infinity;
  let lzMin = Infinity, lzMax = -Infinity;
  for (const c of corners) {
    const dx = c.x - cx, dy = c.y - cy, dz = c.z - cz;
    const lx = basis.x.x * dx + basis.x.y * dy + basis.x.z * dz;
    const ly = basis.y.x * dx + basis.y.y * dy + basis.y.z * dz;
    const lz = basis.z.x * dx + basis.z.y * dy + basis.z.z * dz;
    if (lx < lxMin) lxMin = lx; if (lx > lxMax) lxMax = lx;
    if (ly < lyMin) lyMin = ly; if (ly > lyMax) lyMax = ly;
    if (lz < lzMin) lzMin = lz; if (lz > lzMax) lzMax = lz;
  }

  // 10% 外扩（每侧 5%）吸收 caster 半径与球面凸出。
  const padX = (lxMax - lxMin) * 0.05;
  const padY = (lyMax - lyMin) * 0.05;
  lxMin -= padX; lxMax += padX;
  lyMin -= padY; lyMax += padY;

  // 纹素对齐：抑制相机移动时阴影边缘闪烁。
  let texelWorld = (lxMax - lxMin) / mapSize;
  lxMin = Math.floor(lxMin / texelWorld) * texelWorld;
  lxMax = Math.ceil(lxMax / texelWorld) * texelWorld;
  const texelY = (lyMax - lyMin) / mapSize;
  lyMin = Math.floor(lyMin / texelY) * texelY;
  lyMax = Math.ceil(lyMax / texelY) * texelY;
  texelWorld = (lxMax - lxMin) / mapSize;

  // 近平面沿太阳方向拉宽一整个 span（caster 头顶空间），远平面再放宽半 span。
  const spanZ = Math.max(lzMax - lzMin, 1e-9);
  const zNear = lzMin - spanZ;
  const zFar  = lzMax + spanZ * 0.5;

  // 光源视图（列主序）：基向量作行 → m[col*4+row]。
  const v = new Float32Array(16);
  v[0] = basis.x.x; v[1] = basis.y.x; v[2]  = basis.z.x; v[3]  = 0;
  v[4] = basis.x.y; v[5] = basis.y.y; v[6]  = basis.z.y; v[7]  = 0;
  v[8] = basis.x.z; v[9] = basis.y.z; v[10] = basis.z.z; v[11] = 0;
  v[12] = -(basis.x.x * cx + basis.x.y * cy + basis.x.z * cz);
  v[13] = -(basis.y.x * cx + basis.y.y * cy + basis.y.z * cz);
  v[14] = -(basis.z.x * cx + basis.z.y * cy + basis.z.z * cz);
  v[15] = 1;

  // 正交投影 → WebGPU NDC z ∈ [0,1]：z' = (lz - zNear)/(zFar - zNear)。
  const wx = lxMax - lxMin;
  const wy = lyMax - lyMin;
  const wz = zFar - zNear;
  const p = new Float32Array(16);
  p[0]  = 2 / wx;
  p[5]  = 2 / wy;
  p[10] = 1 / wz;
  p[12] = -(lxMax + lxMin) / wx;
  p[13] = -(lyMax + lyMin) / wy;
  p[14] = -zNear / wz;
  p[15] = 1;

  const viewProj = Matrix4.multiply(
    Matrix4.fromArray(p), Matrix4.fromArray(v), new Matrix4());
  return { viewProj, texelWorld };
}

export class ShadowMap {
  /** 总开关（默认关 —— 关闭时全链路零开销、零回归）。 */
  enabled = false;

  /** 每级联阴影图边长（像素）。改动后下一次 prepare 重建纹理。 */
  mapSize = 2048;

  /** 深度偏移（归一化深度单位，占级联 z 跨度比例）。 */
  depthBias = 0.0006;

  /** Normal-offset 强度（纹素倍数）。 */
  normalOffsetScale = 2.0;

  private readonly _context: Context;
  private readonly _casters = new Set<Primitive>();

  private _texture: GPUTexture | null = null;
  private _layerViews: (GPUTextureView | null)[] = [null, null, null];
  private _arrayView: GPUTextureView | null = null;
  private _pipeline: GPURenderPipeline | null = null;
  private _preparedFor: GPURenderPipeline | null = null;
  private _camUBO: GPUBuffer | null = null;
  private _camBGs: (GPUBindGroup | null)[] = [null, null, null];
  private _receiverUBO: GPUBuffer | null = null;
  private _comparisonSampler: GPUSampler | null = null;
  private _binding: ShadowReceiverBinding | null = null;

  private _needsRebuild = false;
  private _prepared = false;
  private _failed = false;

  /** 每帧矩阵与参数（updateFrame 更新；execute/receiver UBO 消费）。 */
  private readonly _viewProjs: Matrix4[] = [
    new Matrix4(), new Matrix4(), new Matrix4(),
  ];
  private _splitEndsM: [number, number, number] = [250, 1500, 10000];
  private readonly _texelSizes = new Float32Array(3);
  private readonly _rxData = new Float32Array(SHADOW_RECEIVER_UBO_SIZE / 4);
  private readonly _camData = new Float32Array((CAM_UBO_STRIDE / 4) * SHADOW_CASCADE_COUNT);
  private readonly _corners: Cartesian3[] = Array.from(
    { length: 8 }, () => new Cartesian3());

  constructor(context: Context) {
    this._context = context;
  }

  // ── Caster 注册 API ─────────────────────────────────────────────────────────

  /** 注册阴影投射体（Primitive 系：Primitive / Model 子图元 / Tileset3D 图元 / 地形图元）。 */
  addCaster(primitive: Primitive): void {
    this._casters.add(primitive);
  }

  /** 取消注册。返回是否确实移除。 */
  removeCaster(primitive: Primitive): boolean {
    return this._casters.delete(primitive);
  }

  /** 当前 caster 数（验证 / 调试）。 */
  get casterCount(): number { return this._casters.size; }

  /** 本帧级联远边界（米制，验证 / 调试）。 */
  get splitEndsMetres(): readonly [number, number, number] { return this._splitEndsM; }

  /** 调试：深度 pass 累计执行次数与累计 draw 调用数（验证探针用）。 */
  debugDepthPassCount = 0;
  debugDrawCallCount = 0;

  /** 改变阴影图分辨率（256–8192）。立即退役旧纹理，下一次 execute 前重建。 */
  setMapSize(px: number): void {
    const s = Math.min(Math.max(Math.round(px), 256), 8192);
    if (s === this.mapSize && !this._needsRebuild) return;
    this.mapSize = s;
    this._needsRebuild = true;
  }

  // ── 帧驱动 ──────────────────────────────────────────────────────────────────

  /**
   * 每帧 CPU 侧：级联分割 + 光源正交矩阵拟合 + 两块 UBO 数据填充。
   * 纯数学，不触碰 device（GPU 资源由 execute 内 prepare 惰性创建）。
   */
  updateFrame(camera: Camera, lights: DirectionalLight[]): void {
    if (lights.length === 0) return;
    const ld = lights[0].direction;
    const dir = new Cartesian3(ld[0], ld[1], ld[2]);

    // 米制 ↔ 归一化换算（约定：米制距离 / EARTH_SCALE）。
    const nearM = Math.max(camera.frustum.near * Camera.EARTH_SCALE, 0.5);
    const farM  = camera.frustum.far * Camera.EARTH_SCALE;
    this._splitEndsM = computeCascadeSplitEnds(nearM, farM);

    const basis = computeLightBasis(dir);

    for (let i = 0; i < SHADOW_CASCADE_COUNT; i++) {
      const nN = (i === 0 ? 0 : this._splitEndsM[i - 1]) / Camera.EARTH_SCALE;
      const fN = this._splitEndsM[i] / Camera.EARTH_SCALE;
      computeCascadeCorners(camera, nN, fN, this._corners);
      const fit = fitLightOrtho(this._corners, basis, this.mapSize);
      this._viewProjs[i] = fit.viewProj;
      this._texelSizes[i] = fit.texelWorld;
      // 写入步长必须与 bind group 的 256B 偏移一致（= 64 float），否则
      // 级联 1/2 的 bind group 读到零矩阵，深度图恒空。
      this._camData.set(fit.viewProj.values, i * (CAM_UBO_STRIDE / 4));
    }

    // Receiver UBO 载荷（布局见 SHADOW_RECEIVER_MODULE 文档）。
    const rx = this._rxData;
    rx.fill(0);
    rx.set(this._viewProjs[0].values, 0);
    rx.set(this._viewProjs[1].values, 16);
    rx.set(this._viewProjs[2].values, 32);
    const es = Camera.EARTH_SCALE;
    rx[48] = this._splitEndsM[0] / es;
    rx[49] = this._splitEndsM[1] / es;
    rx[50] = this._splitEndsM[2] / es;
    rx[52] = this._texelSizes[0];
    rx[53] = this._texelSizes[1];
    rx[54] = this._texelSizes[2];
    rx[56] = 1.0; // params.x enabled（receiverBinding 已含开关闸，双保险）
    rx[57] = this.depthBias;
    rx[58] = this.normalOffsetScale;
    rx[59] = this.mapSize;
  }

  /**
   * 级联深度渲染：每级联一个 depth-only render pass（同 encoder）。
   * `getOpaquePipeline` 惰性取共享 G-Buffer 管线 —— 阴影管线组 1 必须
   * 按身份复用其 BGL1，Primitive 的模型 bind group 才能跨管线共用。
   */
  execute(
    encoder: GPUCommandEncoder,
    getOpaquePipeline: () => GPURenderPipeline | null,
  ): void {
    if (!this.enabled || this._failed || this._casters.size === 0) return;
    const opaque = getOpaquePipeline();
    if (!opaque) return;

    this.prepare(opaque);
    if (!this._prepared || !this._pipeline || !this._camUBO || !this._receiverUBO) return;

    this._context.writeBuffer(this._camUBO, 0, this._camData);
    this._context.writeBuffer(this._receiverUBO, 0, this._rxData);

    for (let i = 0; i < SHADOW_CASCADE_COUNT; i++) {
      const view = this._layerViews[i];
      const camBG = this._camBGs[i];
      if (!view || !camBG) continue;

      const pass = encoder.beginRenderPass({
        label: `Shadow.CascadeDepth[${i}]`,
        colorAttachments: [],
        depthStencilAttachment: {
          view,
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      });
      pass.setPipeline(this._pipeline);
      pass.setBindGroup(0, camBG);
      for (const caster of this._casters) {
        // 逐 caster 绑定组 1（模型 UBO/纹理）+ 顶点索引缓冲并绘制。
        caster.drawShadow(pass, camBG);
        this.debugDrawCallCount++;
      }
      pass.end();
      this.debugDepthPassCount++;
    }
  }

  /**
   * LightingPass 组 2 绑定载荷。null = 阴影未激活（LightingPass 走无阴影
   * 管线变体 —— 关灯零回归的根源）。开关关闭 / 未就绪 / 失败 / 无 caster
   * 均返回 null。
   */
  receiverBinding(): ShadowReceiverBinding | null {
    if (!this.enabled || this._failed || !this._prepared) return null;
    if (this._casters.size === 0) return null;
    return this._binding;
  }

  // ── GPU 资源 ────────────────────────────────────────────────────────────────

  /** 构建 / 重建 GPU 资源。幂等；失败置 `_failed` 并强制关阴影（OceanFFTWater 闸模式）。 */
  private prepare(opaquePipeline: GPURenderPipeline): void {
    if (this._prepared && !this._needsRebuild && this._preparedFor === opaquePipeline) return;
    try {
      this._teardownGpu();
      const device = this._context.device;
      const s = this.mapSize;

      // 3 层 depth32float 纹理数组。
      this._texture = device.createTexture({
        label: 'Shadow.CascadeAtlas',
        size: { width: s, height: s, depthOrArrayLayers: SHADOW_CASCADE_COUNT },
        format: 'depth32float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      for (let i = 0; i < SHADOW_CASCADE_COUNT; i++) {
        this._layerViews[i] = this._texture.createView({
          label: `Shadow.CascadeAtlas.layer${i}`,
          dimension: '2d',
          baseArrayLayer: i,
          arrayLayerCount: 1,
        });
      }
      this._arrayView = this._texture.createView({
        label: 'Shadow.CascadeAtlas.array',
        dimension: '2d-array',
        baseArrayLayer: 0,
        arrayLayerCount: SHADOW_CASCADE_COUNT,
      });

      this._camUBO = device.createBuffer({
        label: 'Shadow.CascadeViewProjUBO',
        size: CAM_UBO_STRIDE * SHADOW_CASCADE_COUNT,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this._receiverUBO = device.createBuffer({
        label: 'Shadow.ReceiverUBO',
        size: SHADOW_RECEIVER_UBO_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this._comparisonSampler = this._context.createSampler({ compare: 'less' });

      const shader = device.createShaderModule({
        code: SHADOW_DEPTH_SHADER,
        label: 'ShadowDepthShader',
      });
      const camBGL = device.createBindGroupLayout({
        label: 'Shadow.BGL0',
        entries: [{
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'uniform' },
        }],
      });
      // 组 1 按身份复用 G-Buffer 管线 BGL1 —— Primitive._opaqueModelBindGroup
      // 以该布局对象创建，跨管线共用必须同一身份（先例 buildUnlitPipeline）。
      this._pipeline = device.createRenderPipeline({
        label: 'Shadow.CascadeDepthPipeline',
        layout: device.createPipelineLayout({
          bindGroupLayouts: [camBGL, opaquePipeline.getBindGroupLayout(1)],
        }),
        vertex: {
          module: shader,
          entryPoint: 'vs_main',
          buffers: [{
            arrayStride: 32, // 与 G-Buffer 相同的 8 float 交错布局
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' }, // position
            ],
          }],
        },
        // 无颜色目标 → fragment 阶段可省略（WebGPU 规范）。
        depthStencil: {
          format: 'depth32float',
          depthWriteEnabled: true,
          depthCompare: 'less',
        },
      });

      // 每级联一个相机 bind group：同一 UBO 的 256B 对齐偏移。
      const layout = this._pipeline.getBindGroupLayout(0);
      for (let i = 0; i < SHADOW_CASCADE_COUNT; i++) {
        this._camBGs[i] = device.createBindGroup({
          label: `Shadow.CascadeBG${i}`,
          layout,
          entries: [{
            binding: 0,
            resource: { buffer: this._camUBO, offset: i * CAM_UBO_STRIDE, size: 64 },
          }],
        });
      }

      this._binding = {
        ubo: this._receiverUBO,
        textureView: this._arrayView,
        comparisonSampler: this._comparisonSampler,
      };

      this._prepared = true;
      this._preparedFor = opaquePipeline;
      this._needsRebuild = false;
      this._failed = false;
    } catch (e) {
      // 失败闸：整帧零污染（无阴影 = 无阴影管线 = 零回归）。
      this._failed = true;
      this.enabled = false;
      this._teardownGpu();
      console.error('ShadowMap.prepare failed, shadows disabled:', e);
    }
  }

  /** 释放纹理 / 缓冲 / 视图（bind group、pipeline 随之失效一并置空）。 */
  private _teardownGpu(): void {
    this._texture?.destroy();
    this._texture = null;
    this._layerViews = [null, null, null];
    this._arrayView = null;
    this._camUBO?.destroy();
    this._camUBO = null;
    this._receiverUBO?.destroy();
    this._receiverUBO = null;
    this._pipeline = null;
    this._preparedFor = null;
    this._camBGs = [null, null, null];
    this._comparisonSampler = null;
    this._binding = null;
    this._prepared = false;
  }

  /** 终局清理（调用后本对象不可再用；caster 所有权归 Scene，不在此销毁）。 */
  destroy(): void {
    this._teardownGpu();
    this._casters.clear();
    this._failed = false;
  }
}

export default ShadowMap;
