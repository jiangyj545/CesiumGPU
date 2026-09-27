import type { Context } from '../renderer/Context';
import { Cartesian3 } from '../math/Cartesian3';
import { ecefFromLonLatHeight } from '../terrain/TerrainProvider';

/**
 * SeabedHeightField — 程序化沙洲高度场（海洋 L4 地基，一次性 CPU 生成上传，非 IPass）。
 *
 * 局部平面坐标约定：以 {@link SeabedHeightOptions.center} 为原点的 ENU 平面
 * （x=东、z=北、y=上），单位米，与 OceanFFTWater/WakeField 的 planeXZ 同域；
 * 高度 = 海拔（海平面 = 0，正值露出水面）。
 *
 * 默认 profile（heightFn 未传时）：径向 r=√(x²+z²) 上
 * `smoothstep(shoreRadius*1.6, shoreRadius*0.25, r)` 从 edgeDepthM 平滑升至
 * peakM 的浅滩，叠加 3 层确定性 value-noise（尺度 sizeM/8、/16、/32，总幅度
 * ±0.35m）模拟沙波起伏。噪声基于整数格 hash（无 Math.random），同参数两次
 * 构造高度数组逐位一致——岸线泡沫与湿沙线的回归测试依赖该确定性。
 *
 * 数据：gridN² Float32 高度数组 → 一次性上传 r32float 纹理（R 通道 = 海拔米，
 * usage TEXTURE_BINDING | COPY_DST），供海面/海床 shader 采样；CPU 侧
 * {@link sampleHeight} 双线性采样供 CausticsFloor 烘焙网格顶点。
 *
 * 生命周期：构造即生成并上传；`dispose()` 释放 GPU 纹理（此后
 * {@link seabedTexture} 返回 null），CPU 高度数组保留可供离线烘焙。
 */
export interface SeabedHeightOptions {
  /** 沙洲中心经纬度（度）。 */
  center: { lon: number; lat: number };
  /** 场边长（米）。默认 500。 */
  sizeM?: number;
  /** 纹理分辨率（gridN²）。默认 256。gridN 须为 64 的倍数（bytesPerRow = gridN*4 须满足 WebGPU 256B 行对齐）。 */
  gridN?: number;
  /** 沙洲峰值海拔（米，正值露出水面）。默认 1.8。 */
  peakM?: number;
  /** 岸线（海拔 0）基准半径（米）。默认 sizeM*0.22。 */
  shoreRadiusM?: number;
  /** 外缘水深（负值）。默认 -7。 */
  edgeDepthM?: number;
  /** 自定义高度（局部平面米坐标）；传入后覆盖默认 profile（含噪声）。 */
  heightFn?: (x: number, z: number) => number;
}

/** 整数格坐标 hash → [0,1)，确定性（同输入恒同输出）。 */
function hash2(ix: number, iy: number): number {
  let h = (Math.imul(ix, 374761393) + Math.imul(iy, 668265263)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

/** 双线性 value-noise（格点 hash 插值，[-1,1]），cellM 为格距（米）。 */
function valueNoise(x: number, z: number, cellM: number): number {
  const fx = x / cellM;
  const fz = z / cellM;
  const ix = Math.floor(fx);
  const iz = Math.floor(fz);
  const tx = fx - ix;
  const tz = fz - iz;
  const u = tx * tx * (3 - 2 * tx);
  const v = tz * tz * (3 - 2 * tz);
  const h00 = hash2(ix, iz) * 2 - 1;
  const h10 = hash2(ix + 1, iz) * 2 - 1;
  const h01 = hash2(ix, iz + 1) * 2 - 1;
  const h11 = hash2(ix + 1, iz + 1) * 2 - 1;
  return (h00 * (1 - u) + h10 * u) * (1 - v) + (h01 * (1 - u) + h11 * u) * v;
}

/** clamp 平滑阶跃；边序反转（e0>e1）时自然给出 1→0 的衰减曲线。 */
function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

export class SeabedHeightField {
  /** 场边长（米）。 */
  readonly sizeMeters: number;
  /** 纹理/高度数组边长（格）。 */
  readonly gridN: number;
  /** center 的归一化 ECEF 位置（海拔 0，与 ecefFromLonLatHeight 同约定），供消费方建局部基。 */
  readonly centerRadial: Cartesian3;

  private readonly _heights: Float32Array;
  private readonly _edgeDepthM: number;
  private _texture: GPUTexture | null;
  private _view: GPUTextureView | null;

  constructor(context: Context, options: SeabedHeightOptions) {
    if (!context.initialized) {
      throw new Error('SeabedHeightField: Context.initialize() must complete first.');
    }

    const sizeM = options.sizeM ?? 500;
    const gridN = options.gridN ?? 256;
    if (!(sizeM > 0) || !Number.isFinite(sizeM)) {
      throw new Error(`SeabedHeightField: sizeM must be a positive finite number, got ${sizeM}.`);
    }
    if (!Number.isInteger(gridN) || gridN < 4) {
      throw new Error(`SeabedHeightField: gridN must be an integer >= 4, got ${gridN}.`);
    }
    // WebGPU writeTexture 硬约束：bytesPerRow 必须 256B 对齐，即 gridN*4 % 256 == 0
    // → gridN 须为 64 的倍数（64/128/256/512/1024…），否则真机上上传即校验错误。
    if ((gridN * 4) % 256 !== 0) {
      throw new Error(
        `SeabedHeightField: bytesPerRow = gridN*4 must be a multiple of 256 (WebGPU ` +
        `texel row alignment); gridN must therefore be a multiple of 64 ` +
        `(e.g. 64/128/256/512/1024), got ${gridN}.`,
      );
    }

    this.sizeMeters = sizeM;
    this.gridN = gridN;
    this._edgeDepthM = options.edgeDepthM ?? -7;

    const radial: [number, number, number] = [0, 0, 0];
    ecefFromLonLatHeight(options.center.lon, options.center.lat, 0, radial);
    this.centerRadial = new Cartesian3(radial[0], radial[1], radial[2]);

    // 高度场：行主序 j=z、i=x（与 writeTexture 行序一致）
    const peakM = options.peakM ?? 1.8;
    const shoreR = options.shoreRadiusM ?? sizeM * 0.22;
    const noiseAmp = 0.35;
    const layers = [
      { cell: sizeM / 8, amp: noiseAmp * 0.5 },
      { cell: sizeM / 16, amp: noiseAmp * 0.3 },
      { cell: sizeM / 32, amp: noiseAmp * 0.2 },
    ];
    const heightFn = options.heightFn;
    const n = gridN;
    const heights = new Float32Array(n * n);
    for (let j = 0; j < n; j++) {
      const z = ((j + 0.5) / n - 0.5) * sizeM;
      for (let i = 0; i < n; i++) {
        const x = ((i + 0.5) / n - 0.5) * sizeM;
        if (heightFn) {
          heights[j * n + i] = heightFn(x, z);
          continue;
        }
        const r = Math.sqrt(x * x + z * z);
        // r ≤ 0.25·shoreR → 1（峰值），r ≥ 1.6·shoreR → 0（外缘水深）
        const t = smoothstep(shoreR * 1.6, shoreR * 0.25, r);
        let h = this._edgeDepthM + (peakM - this._edgeDepthM) * t;
        for (const layer of layers) {
          h += valueNoise(x, z, layer.cell) * layer.amp;
        }
        heights[j * n + i] = h;
      }
    }
    this._heights = heights;

    // 一次性上传 r32float（R 通道 = 海拔米）；不可过滤格式，消费方用 nearest
    // 或 textureLoad 采样（岸线阈值判断无需线性过滤）。
    const device = context.device;
    const texture = device.createTexture({
      label: 'SeabedHeightField',
      size: [n, n],
      format: 'r32float',
      dimension: '2d',
      mipLevelCount: 1,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture },
      heights,
      { bytesPerRow: n * 4, rowsPerImage: n },
      [n, n, 1],
    );
    this._texture = texture;
    this._view = texture.createView({ label: 'SeabedHeightField.View', dimension: '2d' });
  }

  /** r32float 高度纹理视图（R 通道 = 海拔米）。`dispose()` 后为 null。 */
  get seabedTexture(): GPUTextureView | null {
    return this._view;
  }

  /**
   * CPU 侧高度数组（行主序 j=z、i=x，海拔米）——烘焙网格/回归测试直接读。
   * 只读视图（不拷贝）：Readonly<Float32Array> 在类型层拒绝索引写入
   * （readonly 修饰符不能直接用于 Float32Array，取其映射等价形式）；
   * 方法级修改（fill/set）靠约定禁止。
   */
  get heights(): Readonly<Float32Array> {
    return this._heights;
  }

  /**
   * CPU 双线性采样（局部平面米坐标，与纹理 uv 同域）。场外返回 edgeDepthM。
   * 仅供 CausticsFloor 烘焙网格顶点等 CPU 消费。
   */
  sampleHeight(x: number, z: number): number {
    const n = this.gridN;
    const u = x / this.sizeMeters + 0.5;
    const v = z / this.sizeMeters + 0.5;
    if (u < 0 || u >= 1 || v < 0 || v >= 1) return this._edgeDepthM;

    const fx = u * n - 0.5;
    const fz = v * n - 0.5;
    const i0 = Math.floor(fx);
    const j0 = Math.floor(fz);
    const tx = fx - i0;
    const tz = fz - j0;
    // 边缘 texel 钳位（纹理内仍双线性，不出场）
    const i1 = Math.min(n - 1, i0 + 1);
    const j1 = Math.min(n - 1, j0 + 1);
    const c0 = Math.max(0, i0);
    const c1 = Math.max(0, j0);
    const h = this._heights;
    const h00 = h[c1 * n + c0];
    const h10 = h[c1 * n + i1];
    const h01 = h[j1 * n + c0];
    const h11 = h[j1 * n + i1];
    return (h00 * (1 - tx) + h10 * tx) * (1 - tz) + (h01 * (1 - tx) + h11 * tx) * tz;
  }

  /** 释放 GPU 纹理。CPU 高度数组保留（离线烘焙仍可用 sampleHeight/heights）。 */
  dispose(): void {
    this._texture?.destroy();
    this._texture = null;
    this._view = null;
  }
}
