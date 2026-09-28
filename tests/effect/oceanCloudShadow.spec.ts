import { beforeAll, describe, expect, it } from 'vitest';
import { OceanFFTWater } from '../../src/effect/OceanFFTWater';
import { OCEAN_FFT_RENDER_SHADER } from '../../src/shaders/OceanFFTShader';
import { CLOUD_SHADOW_PARAMS_UBO_BYTES } from '../../src/shaders/CloudShader';
import type { CloudShadowInfo } from '../../src/effect/CloudLayer';
import type { Camera } from '../../src/scene/Camera';
import type { Context } from '../../src/renderer/Context';

/**
 * Track C 海面云影接线验证（无 GPU adapter，全 mock device，
 * oceanSkyReflection.spec.ts 同款记录型 harness）：
 *   1. T1 未绑定 bindCloudShadow（零回归）：binding 20 挂 1×1 黑 rgba8unorm
 *      占位、binding 21 挂 96B UBO 且每帧写全零（intensity 0 短路恒 1）
 *   2. T2 绑定后：binding 20 换真实 view、UBO 写入 planeCenter/ENU 基/
 *      extentScale(EARTH_SCALE/extentM)/intensity 0.65/sunDir
 *   3. T3 缓存键：view 身份变更才重建 bind group；解绑回退占位
 *   4. T4 shader 源串：binding 20-22 声明 + cloudShadowFactor 海面调用 +
 *      天光反射/太阳高光项同乘 cloudF
 */

/** CloudShadowParams UBO f32 索引（布局单一来源 CloudShader.ts）。 */
const CS_PLANE_CENTER = 0;
const CS_CLOUD_RADIUS = 3;
const CS_EAST = 4;
const CS_NORTH = 8;
const CS_EXTENT_SCALE = 12;
const CS_INTENSITY = 13;
const CS_SUN = 16;

const EARTH_SCALE = 6378137;

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
  bindGroups: { entries: { binding: number; resource: unknown }[] }[];
  layouts: { label: string; entries: BglEntry[] }[];
}

/** 记录型 mock Context（oceanSkyReflection.spec.ts 同款）。 */
function makeHarness(): MockHarness {
  const uboWrites: UboWrite[] = [];
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
    queue: { writeTexture: () => {} },
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
      // 快照拷贝（载荷 Float32Array 每帧复用，会被下一帧覆盖）
      const src = new Float32Array(
        data.buffer as ArrayBuffer, data.byteOffset, data.byteLength / 4
      );
      uboWrites.push({ label: buffer.label, f32: new Float32Array(src) });
    },
  } as unknown as Context;

  return { ctx, uboWrites, bindGroups, layouts };
}

const computePass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
const renderPass = {
  setPipeline() {}, setBindGroup() {}, setVertexBuffer() {}, setIndexBuffer() {},
  drawIndexed() {}, end() {},
};
const encoder = {
  beginComputePass: () => computePass,
  beginRenderPass: () => renderPass,
} as unknown as GPUCommandEncoder;

function makeOcean(ctx: Context): OceanFFTWater {
  return new OceanFFTWater(ctx, { center: [121.65, 31.15], segments: 8 });
}

function makeCamera(): Camera {
  return {
    viewMatrix: { values: new Float32Array(16) },
    projectionMatrix: { values: new Float32Array(16) },
    viewProjectionMatrix: { values: new Float32Array(16) },
    position: { x: 0, y: 0, z: 1.01 },
    frustum: { near: 1, far: 100 },
  } as unknown as Camera;
}

/** CloudLayer.cloudShadow 同构契约快照。 */
function makeCloudShadow(tag: string): CloudShadowInfo {
  return {
    view: { label: `${tag}.view` } as unknown as GPUTextureView,
    centerEcef: [0.21, 0.22, 0.95],
    eastDir: [0.98, 0.0, -0.2],
    northDir: [0.0, 0.98, 0.2],
    cloudRadiusNorm: 1.0004,
    extentM: 8192,
    sunDirection: [0.4, 0.5, 0.77],
  };
}

function csWrites(h: MockHarness): Float32Array[] {
  return h.uboWrites.filter((w) => w.label === 'OceanFFTCloudShadowUBO').map((w) => w.f32);
}

function bgEntry(h: MockHarness, binding: number): unknown {
  const bg = h.bindGroups[h.bindGroups.length - 1];
  const e = bg.entries.find((x) => x.binding === binding);
  if (!e) throw new Error(`bind group 缺 binding ${binding}`);
  return e.resource;
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

describe('T1 未绑定 bindCloudShadow（零回归路径）', () => {
  it('binding 20 挂 1×1 黑占位、binding 21 挂 96B UBO 且写全零', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());

    ocean.execute(encoder, {} as GPUTextureView, 0.016);

    const view = bgEntry(h, 20) as { textureLabel: string };
    expect(view.textureLabel).toBe('OceanFFTCloudFallback');
    const ubo = (bgEntry(h, 21) as { buffer: { label: string } }).buffer;
    expect(ubo.label).toBe('OceanFFTCloudShadowUBO');
    const writes = csWrites(h);
    expect(writes.length).toBe(1);
    expect(writes[0].length).toBe(CLOUD_SHADOW_PARAMS_UBO_BYTES / 4);
    expect(writes[0].every((v) => v === 0)).toBe(true); // intensity 0 → 短路恒 1
  });

  it('BGL binding 20-22 声明：float 纹理 / 96B uniform / filtering 采样器（FRAGMENT）', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());
    ocean.execute(encoder, {} as GPUTextureView, 0.016);

    const bgl = h.layouts.find((l) => l.label === 'OceanFFTRenderBGL');
    if (!bgl) throw new Error('渲染 BGL 未创建');
    expect(bgl.entries[20]).toMatchObject({ binding: 20, texture: { sampleType: 'float' } });
    expect(bgl.entries[21]).toMatchObject({
      binding: 21, buffer: { type: 'uniform', minBindingSize: CLOUD_SHADOW_PARAMS_UBO_BYTES },
    });
    expect(bgl.entries[22]).toMatchObject({ binding: 22, sampler: { type: 'filtering' } });
    expect(bgl.entries[20].visibility).toBe(2); // FRAGMENT 专用
  });
});

describe('T2 绑定 bindCloudShadow（云影激活）', () => {
  it('binding 20 换真实 view；UBO 写入几何/extentScale/intensity/sunDir', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());
    const src = makeCloudShadow('clouds');
    ocean.bindCloudShadow(() => src);

    ocean.execute(encoder, {} as GPUTextureView, 0.016);

    expect(bgEntry(h, 20)).toBe(src.view);
    const w = csWrites(h)[0];
    // UBO 走 f32 载荷：以 Math.fround 对齐 f32 舍入后逐位比较
    expect(w[CS_PLANE_CENTER + 0]).toBe(Math.fround(src.centerEcef[0]));
    expect(w[CS_PLANE_CENTER + 1]).toBe(Math.fround(src.centerEcef[1]));
    expect(w[CS_PLANE_CENTER + 2]).toBe(Math.fround(src.centerEcef[2]));
    expect(w[CS_CLOUD_RADIUS]).toBe(Math.fround(src.cloudRadiusNorm));
    expect(w[CS_EAST]).toBe(Math.fround(src.eastDir[0]));
    expect(w[CS_NORTH]).toBe(Math.fround(src.northDir[0]));
    expect(w[CS_EXTENT_SCALE]).toBe(Math.fround(EARTH_SCALE / src.extentM));
    expect(w[CS_INTENSITY]).toBe(Math.fround(0.65));
    expect(w[CS_SUN + 0]).toBe(Math.fround(src.sunDirection[0]));
    expect(w[CS_SUN + 2]).toBe(Math.fround(src.sunDirection[2]));
  });

  it('源返回 null（prepare 前 / destroy 后）→ 占位 + 全零 UBO', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());
    ocean.bindCloudShadow(() => makeCloudShadow('ready'));
    ocean.execute(encoder, {} as GPUTextureView, 0.016);

    ocean.bindCloudShadow(() => null);
    ocean.execute(encoder, {} as GPUTextureView, 0.032);

    expect((bgEntry(h, 20) as { textureLabel: string }).textureLabel).toBe('OceanFFTCloudFallback');
    expect(csWrites(h)[1].every((v) => v === 0)).toBe(true);
  });
});

describe('T3 bind group 缓存键（view 身份）', () => {
  it('view 身份变更才重建；同 view 连帧不重建', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());
    const src = makeCloudShadow('a');
    ocean.bindCloudShadow(() => src);

    ocean.execute(encoder, {} as GPUTextureView, 0.016);
    ocean.execute(encoder, {} as GPUTextureView, 0.032);
    const afterTwo = h.bindGroups.length;

    ocean.bindCloudShadow(() => makeCloudShadow('b')); // view 换新身份
    ocean.execute(encoder, {} as GPUTextureView, 0.048);
    expect(h.bindGroups.length).toBe(afterTwo + 1);
    expect(bgEntry(h, 20)).not.toBe(src.view);
  });
});

describe('T4 OCEAN_FFT_RENDER_SHADER 云影源串', () => {
  const s = OCEAN_FFT_RENDER_SHADER.replace(/\s+/g, '');

  it('binding 20-22 组 0 声明（CloudShadowReceiverModule 海面变体）', () => {
    expect(s).toContain('@group(0)@binding(20)varcloudShadowTex:texture_2d<f32>');
    expect(s).toContain('@group(0)@binding(21)var<uniform>cloudShadowParams:CloudShadowParams');
    expect(s).toContain('@group(0)@binding(22)varcloudShadowSamp:sampler');
    expect(s).toContain('fncloudShadowFactor(worldPos:vec3<f32>)->f32');
  });

  it('fs_main：云影因子水下恒 1；天光反射与太阳高光项同乘 cloudF', () => {
    expect(s).toContain('letcloudF=select(cloudShadowFactor(input.worldPos),1.0,uw)');
    // L5 源串断言的前缀子串仍须保留（回归保护）
    expect(s).toContain('+skyRad*fresnel*0.9*shadowF*cloudF');
    expect(s).toContain('vec3<f32>(spec)*mix(vec3<f32>(1.0),sunTint,flagSpecTint)*shadowF*cloudF');
  });

  it('intensity<=0 短路恒 1（模块单一来源）', () => {
    expect(s).toContain('if(intensity<=0.0){return1.0;}');
  });
});
