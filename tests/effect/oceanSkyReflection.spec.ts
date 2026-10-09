import { beforeAll, describe, expect, it } from 'vitest';
import { OceanFFTWater } from '../../src/effect/OceanFFTWater';
import type { OceanAtmosphereBinding } from '../../src/effect/OceanFFTWater';
import { AtmosphereSky } from '../../src/effect/AtmosphereSky';
import {
  OCEAN_FFT_RENDER_SHADER,
} from '../../src/shaders/OceanFFTShader';
import {
  ATMO_PARAMS_UBO_BYTES,
  SKY_PARAMS_UBO_BYTES,
} from '../../src/shaders/AtmosphereSkyShader';
import type { Camera } from '../../src/scene/Camera';
import type { Context } from '../../src/renderer/Context';

/**
 * L5 物大气海面集成验证（无 GPU adapter，全 mock device）：
 *   1. T1 未绑定 bindAtmosphere：featureFlags bit0=0（旧平色零回归）+ binding
 *      14/16 挂 1×1 黑 fallback、17/18 挂全零 UBO
 *   2. T1 绑定后：bit0/bit3 置位、真实资源进 bind group、身份变更重建
 *   3. T2/T3 染色与 SSS 的 flag 短路（bit3/bit1）+ RP[22] 强度通道
 *   4. T0 AtmosphereSky getter：prepare 前 null / execute 后稳定身份 / destroy 后 null
 *   5. shader 源串：binding 14-19 声明 + atmSkyReflectionRadiance 拼接 +
 *      sunTint / SSS 公式子串
 */

/** RenderParams f32 索引：RP[21] time / RP[22] SSS 强度 / RP[23] featureFlags。 */
const RP_TIME = 21;
const RP_SSS_STRENGTH = 22;
const RP_FEATURE_FLAGS = 23;

const STAGE_VERTEX = 1;
const STAGE_FRAGMENT = 2;

interface UboWrite {
  label: string;
  f32: Float32Array;
}

interface BglEntry {
  binding: number;
  visibility?: number;
  texture?: { sampleType?: string; viewDimension?: string };
  buffer?: { type?: string; minBindingSize?: number };
  sampler?: { type?: string };
}

interface MockHarness {
  ctx: Context;
  uboWrites: UboWrite[];
  bindGroups: { entries: { binding: number; resource: unknown }[] }[];
  layouts: { label: string; entries: BglEntry[] }[];
  buffers: { label: string; size: number }[];
}

/** 记录型 mock Context（oceanShoreFoam.spec.ts 同款）。 */
function makeHarness(): MockHarness {
  const uboWrites: UboWrite[] = [];
  const bindGroups: { entries: { binding: number; resource: unknown }[] }[] = [];
  const layouts: { label: string; entries: BglEntry[] }[] = [];
  const buffers: { label: string; size: number }[] = [];

  const device = {
    createBuffer: (d: { label: string; size: number }) => {
      const b = { label: d.label, size: d.size, destroy() {} };
      buffers.push(b);
      return b;
    },
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
      const src = new Float32Array(
        data.buffer as ArrayBuffer, data.byteOffset, data.byteLength / 4
      );
      uboWrites.push({ label: buffer.label, f32: new Float32Array(src) });
    },
  } as unknown as Context;

  return { ctx, uboWrites, bindGroups, layouts, buffers };
}

function makeCamera(): Camera {
  return {
    position: { x: 0, y: 0, z: 2 },
    viewMatrix: { values: new Float32Array(16) },
    projectionMatrix: { values: new Float32Array(16) },
    viewProjectionMatrix: { values: new Float32Array(16) },
    frustum: { near: 0.1, far: 1e7 },
  } as unknown as Camera;
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
  return new OceanFFTWater(ctx, { center: [121.0, 31.0], extentKm: 0.5, segments: 4 });
}

/** 合成大气绑定源（mock 视图/UBO 带可断言的身份标记）。 */
function makeAtmoSource(tag = 'atmo'): OceanAtmosphereBinding {
  return {
    skyView: { textureLabel: `${tag}.skyView` } as unknown as GPUTextureView,
    transmittance: { textureLabel: `${tag}.trans` } as unknown as GPUTextureView,
    atmosphereParams: { label: `${tag}.atmoUBO` } as unknown as GPUBuffer,
    skyParams: { label: `${tag}.skyUBO` } as unknown as GPUBuffer,
  };
}

function rpAt(h: MockHarness, n: number): Float32Array {
  const writes = h.uboWrites.filter((w) => w.label === 'OceanFFTRenderParamsUBO');
  if (n < 0) n = writes.length + n;
  return writes[n].f32;
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

describe('T1 未绑定 bindAtmosphere（零回归路径）', () => {
  it('featureFlags bit0/bit1/bit3 = 0，RP[21] 时间秒推进', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());

    ocean.execute(encoder, {} as GPUTextureView, 0.016);
    ocean.execute(encoder, {} as GPUTextureView, 0.048);

    const rp = rpAt(h, -1);
    expect(rp[RP_FEATURE_FLAGS] & 1).toBe(0); // bit0 skyReflect
    expect(rp[RP_FEATURE_FLAGS] & 2).toBe(0); // bit1 sss（默认强度 0）
    expect(rp[RP_FEATURE_FLAGS] & 8).toBe(0); // bit3 specTint（未绑定短路）
    expect(rpAt(h, -2)[RP_TIME]).toBe(0);
    expect(rp[RP_TIME]).toBeCloseTo(0.032, 6);
  });

  it('binding 14/16 挂 1×1 黑 rgba16float 占位、17/18 挂全零 UBO', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());
    ocean.execute(encoder, {} as GPUTextureView, 0.016);

    expect((bgEntry(h, 14) as { textureLabel: string }).textureLabel).toBe('OceanFFTFallback1x1');
    expect((bgEntry(h, 16) as { textureLabel: string }).textureLabel).toBe('OceanFFTFallback1x1');
    expect((bgEntry(h, 17) as { buffer: { label: string } }).buffer.label).toBe('OceanFFTAtmoFallbackUBO');
    expect((bgEntry(h, 18) as { buffer: { label: string } }).buffer.label).toBe('OceanFFTAtmoFallbackUBO');
    const fallbackUbo = h.buffers.find((b) => b.label === 'OceanFFTAtmoFallbackUBO');
    expect(fallbackUbo?.size).toBe(64);
  });

  it('BGL 14-19 声明：LUT float / filtering sampler / 64B uniform ×3', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());
    ocean.execute(encoder, {} as GPUTextureView, 0.016);
    const bgl = h.layouts.find((l) => l.label === 'OceanFFTRenderBGL');
    if (!bgl) throw new Error('渲染 BGL 未创建');
    expect(bgl.entries[14]).toMatchObject({ binding: 14, texture: { sampleType: 'float' } });
    expect(bgl.entries[15]).toMatchObject({ binding: 15, sampler: { type: 'filtering' } });
    expect(bgl.entries[16]).toMatchObject({ binding: 16, texture: { sampleType: 'float' } });
    expect(bgl.entries[17]).toMatchObject({ binding: 17, buffer: { type: 'uniform', minBindingSize: 64 } });
    expect(bgl.entries[18]).toMatchObject({ binding: 18, buffer: { type: 'uniform', minBindingSize: 64 } });
    expect(bgl.entries[14].visibility).toBe(STAGE_FRAGMENT);
    expect(bgl.entries[19].visibility).toBe(STAGE_VERTEX | STAGE_FRAGMENT);
  });
});

describe('T1 绑定 AtmosphereSky（反射 / 染色激活）', () => {
  it('绑定后 bit0/bit3 置位，binding 14/16/17/18 换真实资源', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());
    const src = makeAtmoSource();
    ocean.bindAtmosphere(() => src);

    ocean.execute(encoder, {} as GPUTextureView, 0.016);
    const rp = rpAt(h, -1);
    expect(rp[RP_FEATURE_FLAGS] & 1).toBe(1);
    expect(rp[RP_FEATURE_FLAGS] & 8).toBe(8);
    expect(bgEntry(h, 14)).toBe(src.skyView);
    expect(bgEntry(h, 16)).toBe(src.transmittance);
    // buffer 型 entry 的 resource 形如 { buffer }（GPUBufferBinding）
    expect((bgEntry(h, 17) as { buffer: unknown }).buffer).toBe(src.atmosphereParams);
    expect((bgEntry(h, 18) as { buffer: unknown }).buffer).toBe(src.skyParams);
  });

  it('部分字段未就绪（skyView=null）→ fallback + bit0=0（T0 契约：prepare 前安全）', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());
    ocean.bindAtmosphere(() => ({ ...makeAtmoSource(), skyView: null }));

    ocean.execute(encoder, {} as GPUTextureView, 0.016);
    expect(rpAt(h, -1)[RP_FEATURE_FLAGS] & 1).toBe(0);
    expect((bgEntry(h, 14) as { textureLabel: string }).textureLabel).toBe('OceanFFTFallback1x1');
  });

  it('源身份变更重建 bind group；bindAtmosphere(null) 回退占位', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());
    ocean.bindAtmosphere(() => makeAtmoSource('a'));
    ocean.execute(encoder, {} as GPUTextureView, 0.016);
    const bgCountA = h.bindGroups.length;

    // 新源：身份变更 → 重建且用新资源
    const srcB = makeAtmoSource('b');
    ocean.bindAtmosphere(() => srcB);
    ocean.execute(encoder, {} as GPUTextureView, 0.048);
    expect(h.bindGroups.length).toBe(bgCountA + 1);
    expect(bgEntry(h, 14)).toBe(srcB.skyView);
    expect((bgEntry(h, 17) as { buffer: unknown }).buffer).toBe(srcB.atmosphereParams);

    // 解绑：回占位 + bit0=0
    ocean.bindAtmosphere(null);
    ocean.execute(encoder, {} as GPUTextureView, 0.064);
    expect(rpAt(h, -1)[RP_FEATURE_FLAGS] & 1).toBe(0);
    expect((bgEntry(h, 14) as { textureLabel: string }).textureLabel).toBe('OceanFFTFallback1x1');
  });
});

describe('T2/T3 染色与 SSS 开关（featureFlags 短路）', () => {
  it('setSunTint(false) 清 bit3；setSkyReflection(false) 清 bit0（仍绑定时）', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());
    const src = makeAtmoSource();
    ocean.bindAtmosphere(() => src);

    ocean.setSunTint(false);
    ocean.execute(encoder, {} as GPUTextureView, 0.016);
    expect(rpAt(h, -1)[RP_FEATURE_FLAGS] & 1).toBe(1);
    expect(rpAt(h, -1)[RP_FEATURE_FLAGS] & 8).toBe(0);

    ocean.setSkyReflection(false);
    ocean.setSunTint(true);
    ocean.execute(encoder, {} as GPUTextureView, 0.032);
    expect(rpAt(h, -1)[RP_FEATURE_FLAGS] & 1).toBe(0);
    expect(rpAt(h, -1)[RP_FEATURE_FLAGS] & 8).toBe(8);
  });

  it('setSubsurfaceStrength 走 RP[22] 并置 bit1；0 复位；非法值抛错', () => {
    const h = makeHarness();
    const ocean = makeOcean(h.ctx);
    ocean.updateCamera(makeCamera());

    ocean.setSubsurfaceStrength(1.5);
    ocean.execute(encoder, {} as GPUTextureView, 0.016);
    expect(rpAt(h, -1)[RP_SSS_STRENGTH]).toBe(1.5);
    expect(rpAt(h, -1)[RP_FEATURE_FLAGS] & 2).toBe(2);

    ocean.setSubsurfaceStrength(0);
    ocean.execute(encoder, {} as GPUTextureView, 0.032);
    expect(rpAt(h, -1)[RP_SSS_STRENGTH]).toBe(0);
    expect(rpAt(h, -1)[RP_FEATURE_FLAGS] & 2).toBe(0);

    expect(() => ocean.setSubsurfaceStrength(-1)).toThrow(/非法/);
    expect(() => ocean.setSubsurfaceStrength(NaN)).toThrow(/非法/);
  });
});

describe('T0 AtmosphereSky 绑定资源 getter', () => {
  /** AtmosphereSky 专用 mock（prepare 创建 LUT/UBO/管线族）。 */
  function makeSkyHarness(): { ctx: Context; buffers: { label: string }[] } {
    const buffers: { label: string }[] = [];
    const device = {
      createBuffer: (d: { label: string; size: number }) => {
        const b = { label: d.label, size: d.size, destroy() {} };
        buffers.push(b);
        return b;
      },
      createTexture: (d: { label: string }) => ({
        label: d.label,
        createView: (opts?: { label?: string }) => ({ label: opts?.label ?? `${d.label}.View` }),
        destroy() {},
      }),
      createSampler: () => ({}),
      createShaderModule: () => ({}),
      createBindGroupLayout: (d: { label: string; entries: unknown[] }) => ({ label: d.label, entries: d.entries }),
      createPipelineLayout: (d: { bindGroupLayouts: unknown[] }) => ({ bindGroupLayouts: d.bindGroupLayouts }),
      createComputePipeline: (d: { label: string }) => ({ label: d.label }),
      createRenderPipeline: (d: { label: string }) => ({ label: d.label }),
      createBindGroup: (d: { entries: unknown[] }) => ({ entries: d.entries }),
      queue: { writeTexture: () => {} },
    };
    const ctx = {
      initialized: true,
      device,
      preferredFormat: 'bgra8unorm',
      drawingBufferWidth: 800,
      drawingBufferHeight: 600,
      createSampler: () => ({}),
      createShaderModule: () => ({}),
      writeBuffer: () => {},
    } as unknown as Context;
    return { ctx, buffers };
  }

  it('prepare 前 null；execute（prepare）后 buffer/view 稳定身份；UBO 64B', () => {
    const { ctx, buffers } = makeSkyHarness();
    const sky = new AtmosphereSky(ctx, {});
    expect(sky.atmosphereParamsBuffer).toBeNull();
    expect(sky.skyParamsBuffer).toBeNull();
    expect(sky.transmittanceTexture).toBeNull();

    // execute 内部先 prepare；无 depthView 安全闸提前返回但资源已创建
    sky.execute(encoder, {} as GPUTextureView, 0.016);

    const atmoUbo = sky.atmosphereParamsBuffer;
    const skyUbo = sky.skyParamsBuffer;
    expect(atmoUbo).not.toBeNull();
    expect(skyUbo).not.toBeNull();
    expect((atmoUbo as unknown as { label: string }).label).toBe('AtmosphereSky.AtmoUBO');
    expect((skyUbo as unknown as { label: string }).label).toBe('AtmosphereSky.SkyUBO');
    expect(buffers.find((b) => b.label === 'AtmosphereSky.AtmoUBO')?.size).toBe(ATMO_PARAMS_UBO_BYTES);
    expect(buffers.find((b) => b.label === 'AtmosphereSky.SkyUBO')?.size).toBe(SKY_PARAMS_UBO_BYTES);

    // 重复调用返回同一身份（bind group 缓存键的前提）
    expect(sky.atmosphereParamsBuffer).toBe(atmoUbo);
    expect(sky.skyParamsBuffer).toBe(skyUbo);
    const tr1 = sky.transmittanceTexture;
    const tr2 = sky.transmittanceTexture;
    expect(tr1).not.toBeNull();
    expect(tr2).toBe(tr1);
    const sv1 = sky.skyViewTexture;
    expect(sky.skyViewTexture).toBe(sv1);
  });

  it('destroy 后全部 getter 返回 null', () => {
    const { ctx } = makeSkyHarness();
    const sky = new AtmosphereSky(ctx, {});
    sky.execute(encoder, {} as GPUTextureView, 0.016);
    expect(sky.atmosphereParamsBuffer).not.toBeNull();
    sky.destroy();
    expect(sky.atmosphereParamsBuffer).toBeNull();
    expect(sky.skyParamsBuffer).toBeNull();
    expect(sky.transmittanceTexture).toBeNull();
    expect(sky.skyViewTexture).toBeNull();
  });
});

describe('OCEAN_FFT_RENDER_SHADER L5 源串', () => {
  const s = OCEAN_FFT_RENDER_SHADER.replace(/\s+/g, '');

  it('binding 14-19 声明 + 大气反射/透射函数拼接', () => {
    expect(s).toContain('@binding(14)varatmSkyViewLUT:texture_2d<f32>');
    expect(s).toContain('@binding(15)varatmLinearClamp:sampler');
    expect(s).toContain('@binding(16)varatmTransmittanceLUT:texture_2d<f32>');
    expect(s).toContain('@binding(17)var<uniform>atmosphereParams:AtmosphereParams');
    expect(s).toContain('@binding(18)var<uniform>skyParams:SkyParams');
    expect(s).toContain('fnatmSkyReflectionRadiance(dirWorld:vec3f,up:vec3f');
    expect(s).toContain('fnatmosphereSampleTransmittance(r:f32,mu:f32)->vec3f');
  });

  it('T1 反射基同源（相机天顶）+ 短路回平色', () => {
    expect(s).toContain('letatmUp=normalize(camPos)');
    expect(s).toContain('letflagSky=select(0.0,1.0,(ffb&1)!=0)');
    expect(s).toContain('letrefl=reflect(-viewDir,nLit)');
    expect(s).toContain('varskyRad=renderParams.skyColor.rgb');
    // 2026-10-06 视觉迭代 Round6 V2：反射项乘入 bandCalm（近场群谷压镜像反射）
    expect(s).toContain('+skyRad*fresnel*0.9*bandCalm*shadowF');
    expect(s).toContain('renderParams.underwater.y)');
    // 同轮折射权重改 (1-f)²·0.75 + shoreDepth 门控（掠射视线采屏幕空间历史帧
    // 的天空亮色不再以 0.30 混入前景）
    expect(s).toContain(
      'letrefrW=clamp((1.0-fresnel)*(1.0-fresnel)*0.75,0.0,0.60)*(1.0-smoothstep(4.5,8.0,shoreDepth));'
    );
  });

  it('T2 太阳染色源串 + bit3 短路 mix', () => {
    expect(s).toContain('sunTint=atmosphereSampleTransmittance(atmosphereParams.viewHeight,clamp(dot(sunDir,atmUp),-1.0,1.0))');
    // 2026-10-06 视觉迭代 Round6 V2：bandCalm 近场谷 0.78→0.92（太阳耀光门控
    // ——iso2 全源关断实锤残留亮源为耀光柱，非泡沫）
    expect(s).toContain('letbandCalm=1.0-0.92*nearW*(1.0-swellBand);');
    expect(s).toContain('vec3<f32>(spec)*mix(vec3<f32>(1.0),sunTint,flagSpecTint)*bandCalm*shadowF');
  });

  it('T3 SSS 青绿透射公式源串', () => {
    // 2026-10-06 视觉迭代：透射色绿斑 [0.06,0.42,0.34] → 偏青蓝 [0.05,0.30,0.38]
    expect(s).toContain('col+=vec3<f32>(0.05,0.30,0.38)*sssW*sunTint');
    expect(s).toContain('letH=clamp(input.waveH*0.28,0.0,1.0)');
    expect(s).toContain('*upY*renderParams.underwater.z*shadowF');
    expect(s).toContain('col=mix(col,col*(1.0-0.25*H),0.5)');
  });
});
