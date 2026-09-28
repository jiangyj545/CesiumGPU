import { beforeAll, describe, expect, it } from 'vitest';
import { OceanFFTWater } from '../../src/effect/OceanFFTWater';
import { SeabedHeightField } from '../../src/effect/SeabedHeightField';
import {
  OCEAN_FFT_RENDER_SHADER,
  OCEAN_RENDER_PARAMS_UBO_BYTES,
  OCEAN_BREAK_PARAMS_UBO_BYTES,
} from '../../src/shaders/OceanFFTShader';
import type { Camera } from '../../src/scene/Camera';
import type { Context } from '../../src/renderer/Context';

/**
 * L4-A 岸线泡沫注入验证（无 GPU adapter，全 mock device）：
 *   1. RP flag 置位/清位 + binding 13 占位/场视图切换
 *   2. seabedUv 原点差投影仿射系数（CPU 手算对照）
 *   3. shader 源串结构断言（破碎带 smoothstep / edgeFade / 128B UBO）
 * flag=0 零回归由初始值自证：jacW=1 / shoreCov=0 / edgeFade=1 → 下游公式
 * 逐项退化回原路径（covTotal = foam*1 + 0 + wakeFoam、alpha=1）。
 */

/** 每纬度度数对应米数（与 OceanFFTWater 内部换算一致）。 */
const METERS_PER_DEG = 111320;

/** RenderParams UBO 内 f32 索引（128B 布局：shoreParams@96=f24、seabedUv@112=f28）。 */
const RP_SHORE_FLAG = 24;
const RP_SHORE_STRENGTH = 25;
const RP_UV_OFF_X = 28;
const RP_UV_OFF_Z = 29;
const RP_UV_SCL_X = 30;
const RP_UV_SCL_Z = 31;
/** L5 featureFlags（underwater.w）：bit0 天空反射 / bit1 SSS / bit2 破碎波 / bit3 染色。 */
const RP_FEATURE_FLAGS = 23;

/** GPUShaderStage.FRAGMENT（mock 环境常量值，与 beforeAll 注入一致）。 */
const STAGE_FRAGMENT = 2;
const STAGE_VERTEX = 1;

interface UboWrite {
  label: string;
  f32: Float32Array;
}

interface BglEntry {
  binding: number;
  visibility?: number;
  texture?: { sampleType?: string };
  buffer?: { type?: string; minBindingSize?: number };
  sampler?: { type?: string };
}

interface MockHarness {
  ctx: Context;
  uboWrites: UboWrite[];
  texWrites: { label: string; data: Float32Array }[];
  bindGroups: { entries: { binding: number; resource: unknown }[] }[];
  layouts: { label: string; entries: BglEntry[] }[];
}

/** 记录型 mock Context：UBO 写入 / 纹理上传 / bind group / BGL 全量捕获。 */
function makeHarness(): MockHarness {
  const uboWrites: UboWrite[] = [];
  const texWrites: { label: string; data: Float32Array }[] = [];
  const bindGroups: { entries: { binding: number; resource: unknown }[] }[] = [];
  const layouts: { label: string; entries: BglEntry[] }[] = [];

  const device = {
    createBuffer: (d: { label: string; size: number }) => ({ label: d.label, size: d.size, destroy() {} }),
    createTexture: (d: { label: string }) => ({
      label: d.label,
      createView: (opts?: { label?: string }) => ({
        label: opts?.label ?? `${d.label}.View`,
        textureLabel: d.label,
      }),
      destroy() {},
    }),
    createSampler: () => ({}),
    createShaderModule: (code: string, label?: string) => ({ label, codeChars: code.length }),
    createBindGroupLayout: (d: { label: string; entries: BglEntry[] }) => {
      const layout = { label: d.label, entries: d.entries };
      layouts.push(layout);
      return layout;
    },
    createPipelineLayout: (d: { bindGroupLayouts: unknown[] }) => ({ bindGroupLayouts: d.bindGroupLayouts }),
    createComputePipeline: (d: { label: string }) => ({ label: d.label }),
    createRenderPipeline: (d: { label: string }) => ({ label: d.label }),
    createBindGroup: (d: { entries: { binding: number; resource: unknown }[] }) => {
      const bg = { entries: d.entries };
      bindGroups.push(bg);
      return bg;
    },
    queue: {
      writeTexture: (dest: { texture: { label: string } }, data: Float32Array) => {
        texWrites.push({ label: dest.texture.label, data: new Float32Array(data) });
      },
    },
  };

  const ctx = {
    initialized: true,
    device,
    preferredFormat: 'bgra8unorm',
    drawingBufferWidth: 800,
    drawingBufferHeight: 600,
    createSampler: () => ({}),
    createShaderModule: (code: string, label?: string) => ({ label, codeChars: code.length }),
    writeBuffer: (buffer: { label: string }, _offset: number, data: ArrayBufferView) => {
      // 快照拷贝（RP Float32Array 每帧复用，会被下一帧覆盖）
      const src = new Float32Array(
        data.buffer as ArrayBuffer, data.byteOffset, data.byteLength / 4
      );
      uboWrites.push({ label: buffer.label, f32: new Float32Array(src) });
    },
  } as unknown as Context;

  return { ctx, uboWrites, texWrites, bindGroups, layouts };
}

/** writeCameraUniforms 只读 enumerated 字段（viewProj 矩阵 / position / frustum）——全零即可。 */
function makeCamera(): Camera {
  return {
    position: { x: 0, y: 0, z: 2 },
    viewMatrix: { values: new Float32Array(16) },
    projectionMatrix: { values: new Float32Array(16) },
    viewProjectionMatrix: { values: new Float32Array(16) },
    frustum: { near: 0.1, far: 1e7 },
  } as unknown as Camera;
}

const computePass = {
  setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {},
};
const renderPass = {
  setPipeline() {}, setBindGroup() {}, setVertexBuffer() {}, setIndexBuffer() {},
  drawIndexed() {}, end() {},
};
const encoder = {
  beginComputePass: () => computePass,
  beginRenderPass: () => renderPass,
} as unknown as GPUCommandEncoder;

function makeOcean(ctx: Context): OceanFFTWater {
  return new OceanFFTWater(ctx, { center: [121.0, 31.0], extentKm: 0.5, segments: 4 });
}

function makeSeabed(ctx: Context, lon: number, lat: number, sizeM = 500): SeabedHeightField {
  // gridN 须为 64 的倍数（bytesPerRow = gridN*4 须 256B 对齐，构造函数校验）
  return new SeabedHeightField(ctx, { center: { lon, lat }, sizeM, gridN: 64 });
}

/** 第 n 帧（负数 = 从尾部倒数）写入的 RenderParams UBO 内容。 */
function rpAt(h: MockHarness, n: number): Float32Array {
  const writes = h.uboWrites.filter((w) => w.label === 'OceanFFTRenderParamsUBO');
  if (n < 0) n = writes.length + n;
  return writes[n].f32;
}

function binding13(h: MockHarness): { binding: number; resource: unknown } {
  const bg = h.bindGroups[h.bindGroups.length - 1];
  const e = bg.entries.find((x) => x.binding === 13);
  if (!e) throw new Error('bind group 缺 binding 13');
  return e;
}

beforeAll(() => {
  const g = globalThis as unknown as Record<string, unknown>;
  g.GPUBufferUsage ??= {
    MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32,
    UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512,
  };
  g.GPUTextureUsage ??= {
    COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16,
  };
  g.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
});

describe('OceanFFTWater 岸线泡沫注入（L4-A）', () => {
  it('未绑定 seabed：RP flag=0，binding 13 挂 1×1 全 -1000 占位，BGL 为 unfilterable-float', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());

    ocean.execute(encoder, {} as GPUTextureView, 0.016);

    expect(rpAt(h, -1)[RP_SHORE_FLAG]).toBe(0);

    const e13 = binding13(h);
    expect((e13.resource as { textureLabel: string }).textureLabel).toBe('OceanFFTSeabedFallback');

    const seed = h.texWrites.find((w) => w.label === 'OceanFFTSeabedFallback');
    if (!seed) throw new Error('占位纹理未上传');
    expect(seed.data.length).toBe(1);
    expect(seed.data[0]).toBe(-1000);

    const bgl = h.layouts.find((l) => l.label === 'OceanFFTRenderBGL');
    if (!bgl) throw new Error('渲染 BGL 未创建');
    expect(bgl.entries[13].binding).toBe(13);
    expect(bgl.entries[13].texture?.sampleType).toBe('unfilterable-float');
    // T4 起顶点阶段也 textureLoad（破碎波波形变形）→ 双阶段
    expect(bgl.entries[13].visibility).toBe(STAGE_VERTEX | STAGE_FRAGMENT);
  });

  it('bindSeabed(field) 置位 flag 且 binding 13 换场视图；bindSeabed(null) 清位回占位', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());
    const field = makeSeabed(h.ctx, 121.0, 31.0);

    ocean.bindSeabed(field);
    ocean.execute(encoder, {} as GPUTextureView, 0.016);
    expect(rpAt(h, -1)[RP_SHORE_FLAG]).toBe(1);
    expect(binding13(h).resource).toBe(field.seabedTexture);

    ocean.bindSeabed(null);
    ocean.execute(encoder, {} as GPUTextureView, 0.032);
    expect(rpAt(h, -1)[RP_SHORE_FLAG]).toBe(0);
    expect((binding13(h).resource as { textureLabel: string }).textureLabel)
      .toBe('OceanFFTSeabedFallback');
  });

  it('setShoreStrength 走 UBO RP[25]，默认 1.0', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());
    ocean.bindSeabed(makeSeabed(h.ctx, 121.0, 31.0));

    ocean.execute(encoder, {} as GPUTextureView, 0.016);
    expect(rpAt(h, -1)[RP_SHORE_STRENGTH]).toBe(1);

    ocean.setShoreStrength(2.5);
    ocean.execute(encoder, {} as GPUTextureView, 0.032);
    expect(rpAt(h, -1)[RP_SHORE_STRENGTH]).toBe(2.5);
  });

  it('setBreakWave 置 RP[23].bit2 且 binding 19 UBO gain.x=1；null 复位归零（T4）', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());
    ocean.bindSeabed(makeSeabed(h.ctx, 121.0, 31.0));

    // 未启用：bit2=0，breakParams UBO 全零
    ocean.execute(encoder, {} as GPUTextureView, 0.016);
    expect(rpAt(h, -1)[RP_FEATURE_FLAGS] & 4).toBe(0);
    let bk = h.uboWrites.find((w) => w.label === 'OceanFFTBreakParamsUBO');
    expect(bk).toBeDefined();
    expect(bk!.f32[0]).toBe(0);

    ocean.setBreakWave({ ampScale: 1.3, lean: 0.6, bandNearM: 3, bandFarM: 8 });
    ocean.execute(encoder, {} as GPUTextureView, 0.032);
    expect(rpAt(h, -1)[RP_FEATURE_FLAGS] & 4).toBe(4);
    const bks = h.uboWrites.filter((w) => w.label === 'OceanFFTBreakParamsUBO');
    bk = bks[bks.length - 1];
    expect(bk.f32[0]).toBe(1);        // gain.x = enable
    expect(bk.f32[1]).toBeCloseTo(1.3, 6);
    expect(bk.f32[2]).toBeCloseTo(0.6, 6);
    expect(bk.f32[4]).toBe(3);        // bandNear
    expect(bk.f32[5]).toBe(8);        // bandFar

    ocean.setBreakWave(null);
    ocean.execute(encoder, {} as GPUTextureView, 0.048);
    expect(rpAt(h, -1)[RP_FEATURE_FLAGS] & 4).toBe(0);
  });

  it('setBreakWave bandFar <= bandNear 抛错（防反序 smoothstep 边界重叠 NaN）', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    expect(() => ocean.setBreakWave({ bandNearM: 8, bandFarM: 3 })).toThrow(/bandFarM/);
    expect(() => ocean.setBreakWave({ bandNearM: 3, bandFarM: 3 })).toThrow(/bandFarM/);
  });

  it('seabedUv 映射：同 center → offset(0.5,0.5)；海床偏东 200m → x=0.1、偏北 100m → y=0.3', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());
    const dLon = 200 / (METERS_PER_DEG * Math.cos((31 * Math.PI) / 180));
    const dLat = 100 / METERS_PER_DEG;

    // 同中心：原点差为 0 → uv = planeXZ/size + 0.5
    ocean.bindSeabed(makeSeabed(h.ctx, 121.0, 31.0));
    ocean.execute(encoder, {} as GPUTextureView, 0.016);
    let rp = rpAt(h, -1);
    expect(rp[RP_UV_OFF_X]).toBeCloseTo(0.5, 6);
    expect(rp[RP_UV_OFF_Z]).toBeCloseTo(0.5, 6);
    expect(rp[RP_UV_SCL_X]).toBeCloseTo(1 / 500, 8);
    expect(rp[RP_UV_SCL_Z]).toBeCloseTo(1 / 500, 8);

    // 海床中心偏东 200m：oceanCenter - seabedCenter 投影 east = -200m
    // → 海面原点（planeXZ=0）映到海床场内 uv.x = 0.5 - 200/500 = 0.1（西侧）。
    // 交叉轴有 ~1.5e-3 相对泄漏：项目归一化 ECEF 只压扁 Y/Z（X 不缩放），
    // 纬线为椭圆 → 东向位移曲率分量按 sinφ·(1-k²)/2 量级漏入北基（31°N
    // 实测 1.524e-3，随距离线性，主轴精确到 1e-8）——约定固有，非公式错。
    ocean.bindSeabed(makeSeabed(h.ctx, 121.0 + dLon, 31.0));
    ocean.execute(encoder, {} as GPUTextureView, 0.032);
    rp = rpAt(h, -1);
    expect(rp[RP_UV_OFF_X]).toBeCloseTo(0.5 - 200 / 500, 4);
    expect(Math.abs(rp[RP_UV_OFF_Z] - 0.5)).toBeLessThan(2e-3); // ≤1m@500m 场

    // 海床中心偏北 100m → uv.y = 0.3，东西不动（同上交叉轴容差）
    ocean.bindSeabed(makeSeabed(h.ctx, 121.0, 31.0 + dLat));
    ocean.execute(encoder, {} as GPUTextureView, 0.048);
    rp = rpAt(h, -1);
    expect(Math.abs(rp[RP_UV_OFF_X] - 0.5)).toBeLessThan(2e-3);
    expect(rp[RP_UV_OFF_Z]).toBeCloseTo(0.5 - 100 / 500, 4);
  });
});

describe('OCEAN_FFT_RENDER_SHADER 岸线泡沫源串', () => {
  const s = OCEAN_FFT_RENDER_SHADER.replace(/\s+/g, '');

  it('binding 13 声明 + struct 新字段', () => {
    expect(s).toContain('@binding(13)varseabedTex:texture_2d<f32>');
    expect(s).toContain('shoreParams:vec4<f32>');
    expect(s).toContain('seabedUv:vec4<f32>');
    expect(OCEAN_RENDER_PARAMS_UBO_BYTES).toBe(128);
  });

  it('破碎带 smoothstep + Jacobian 打 5 折 + 浅水 alpha 收边', () => {
    expect(s).toContain('smoothstep(7.0,3.0,depth)');
    expect(s).toContain('1.0-0.5*surf');
    expect(s).toContain('surf*renderParams.shoreParams.y');
    expect(s).toContain('smoothstep(0.0,0.05,depth)');
    expect(s).toContain('vec4<f32>(col,edgeFade)');
  });

  it('flag=0 零回归自证：初始值退化 + 总量公式还原原语义', () => {
    expect(s).toContain('varjacW=1.0;');
    expect(s).toContain('varshoreCov=0.0;');
    expect(s).toContain('varedgeFade=1.0;');
    expect(s).toContain('letcovTotal=foam*jacW+shoreCov+wakeFoam;');
    expect(s).toContain('letcoverage=clamp(covTotal,0.0,1.0);');
  });

  it('T4 破碎波：binding 19 / BreakParams / VS 变形 / FS 花边与内卷水幕源串', () => {
    expect(s).toContain('@binding(19)var<uniform>breakParams:BreakParams');
    expect(s).toContain('gain:vec4<f32>');
    expect(s).toContain('band:vec4<f32>');
    expect(OCEAN_BREAK_PARAMS_UBO_BYTES).toBe(64);
    // VS：海床 textureLoad → 水深 → 破碎权重；振幅调制 + 波峰前倾
    expect(s).toContain('@location(4)breakW:f32');
    expect(s).toContain('breakW=smoothstep(breakParams.band.y,breakParams.band.x,disp.y-sbH)');
    expect(s).toContain('disp.y*=mix(1.0,breakParams.gain.y,breakW)');
    expect(s).toContain('disp.xz-=g0.xy*max(disp.y,0.0)*breakParams.gain.z*breakW');
    // FS：foam lace 独立 smoothstep + 内卷水幕（nLit·viewDir<0 背面门）
    expect(s).toContain('letlaceW=smoothstep(breakParams.band.w,breakParams.band.w*0.35,shoreDepth)');
    expect(s).toContain('letbackFace=select(0.0,1.0,dot(nLit,viewDir)<0.0)');
    expect(s).toContain('foamMask=clamp(foamMask+foamW*0.5,0.0,1.0)');
  });

  it('T4 binding 19 BGL 双阶段 + minBindingSize 64B', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());
    ocean.execute(encoder, {} as GPUTextureView, 0.016); // BGL 在 prepare 惰性创建
    const bgl = h.layouts.find((l) => l.label === 'OceanFFTRenderBGL');
    if (!bgl) throw new Error('渲染 BGL 未创建');
    expect(bgl.entries[19].binding).toBe(19);
    expect(bgl.entries[19].buffer?.type).toBe('uniform');
    expect(bgl.entries[19].buffer?.minBindingSize).toBe(OCEAN_BREAK_PARAMS_UBO_BYTES);
    expect(bgl.entries[19].visibility).toBe(STAGE_VERTEX | STAGE_FRAGMENT);
  });
});
