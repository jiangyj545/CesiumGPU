import { beforeAll, describe, expect, it, vi } from 'vitest';
import { CloudLayer } from '../../src/effect/CloudLayer';
import { LightingPass } from '../../src/scene/passes/LightingPass';
import type { CloudShadowReceiverBinding } from '../../src/scene/passes/LightingPass';
import {
  LIGHTING_SHADER,
  LIGHTING_SHADER_HDR,
  LIGHTING_SHADER_SHADOW,
  LIGHTING_SHADER_SHADOW_HDR,
  LIGHTING_SHADER_CLOUD,
  LIGHTING_SHADER_CLOUD_HDR,
  LIGHTING_SHADER_SHADOW_CLOUD,
  LIGHTING_SHADER_SHADOW_CLOUD_HDR,
  injectCloudShadowCore,
} from '../../src/shaders/LightingShader';
import {
  CLOUD_RENDER_WGSL,
  CLOUD_SHADOW_BAKE_WGSL,
  CLOUD_SHADOW_RECEIVER_MODULE,
  CLOUD_CAMERA_UBO_BYTES,
  CLOUD_PARAMS_UBO_BYTES,
  CIRRUS_PARAMS_UBO_BYTES,
  CLOUD_SHADOW_PARAMS_UBO_BYTES,
} from '../../src/shaders/CloudShader';
import type { GBufferPass } from '../../src/scene/passes/GBufferPass';
import { Camera } from '../../src/scene/Camera';
import { Cartesian3 } from '../../src/math/Cartesian3';
import type { Context } from '../../src/renderer/Context';

/**
 * Track C 体积云层 + 云影 —— 纯 CPU / 字符串验证（记录型桩 device，
 * 无真实 WebGPU，causticsFloorWetSand.spec 同款模式）：
 *   1. CloudLayer：prepare 幂等 / execute compute+render 全链路 / 深度安全闸 /
 *      destroy 后 getter null / setTargetFormat 管线重建幂等 / 云影 getter
 *      结构与身份稳定 / setter 写 UBO（coverage/extent 布局断言）/ 风漂移推进
 *   2. LightingShader：8 变体正交组合 + 既有 4 导出零回归 + 锚点漂移抛错
 *   3. LightingPass：setCloudShadowSource(null) 零回归（2 组 + 原变体）、
 *      接入后 3 组 + 云影变体 + execute setBindGroup(3)、摘除还原
 */

const ES = Camera.EARTH_SCALE;

/** GPU 常量（mock 环境注入，与 WebGPU 枚举无对应要求）。 */
const STAGE_VERTEX = 1;
const STAGE_FRAGMENT = 2;
const STAGE_COMPUTE = 4;

interface WriteRec {
  label: string;
  f32: Float32Array;
}

interface MockHarness {
  ctx: Context;
  writes: WriteRec[];
  buffers: { label: string; size: number; destroyed: boolean; destroy: ReturnType<typeof vi.fn> }[];
  textures: GPUTextureDescriptor[];
  samplers: GPUSamplerDescriptor[];
  modules: { code: string; label?: string }[];
  renderPipelines: GPURenderPipelineDescriptor[];
  computePipelines: GPUComputePipelineDescriptor[];
  pipelineLayouts: GPUPipelineLayoutDescriptor[];
  bindGroups: GPUBindGroupDescriptor[];
}

/** 记录型桩 Context：捕获 createBuffer / writeBuffer / 管线与模块创建。 */
function makeHarness(): MockHarness {
  const h: MockHarness = {
    writes: [],
    buffers: [],
    textures: [],
    samplers: [],
    modules: [],
    renderPipelines: [],
    computePipelines: [],
    pipelineLayouts: [],
    bindGroups: [],
    ctx: null as unknown as Context,
  };

  const device = {
    createBuffer: (desc: GPUBufferDescriptor) => {
      const b = { label: desc.label ?? '', size: desc.size, destroyed: false, destroy: vi.fn(() => { b.destroyed = true; }) };
      h.buffers.push(b);
      return b;
    },
    createTexture: (desc: GPUTextureDescriptor) => {
      h.textures.push(desc);
      return {
        label: desc.label,
        createView: () => ({ label: `${desc.label}.View` }),
        destroy: vi.fn(),
      };
    },
    createSampler: (desc: GPUSamplerDescriptor) => {
      h.samplers.push(desc);
      return { label: desc.label };
    },
    createBindGroupLayout: (d: GPUBindGroupLayoutDescriptor) => ({ label: d.label }),
    createPipelineLayout: (d: GPUPipelineLayoutDescriptor) => {
      h.pipelineLayouts.push(d);
      return { label: 'pl', bindGroupLayouts: d.bindGroupLayouts };
    },
    createShaderModule: (d: GPUShaderModuleDescriptor) => {
      h.modules.push({ code: d.code as string, label: d.label });
      return { label: d.label };
    },
    createRenderPipeline: (d: GPURenderPipelineDescriptor) => {
      h.renderPipelines.push(d);
      return { label: d.label };
    },
    createComputePipeline: (d: GPUComputePipelineDescriptor) => {
      h.computePipelines.push(d);
      return { label: d.label };
    },
    createBindGroup: (d: GPUBindGroupDescriptor) => {
      h.bindGroups.push(d);
      return { label: d.label };
    },
  };

  const writeBuffer = vi.fn((buffer: { label?: string }, _offset: number, data: BufferSource) => {
    h.writes.push({ label: buffer?.label ?? '', f32: new Float32Array(data as Float32Array) });
  });

  h.ctx = {
    initialized: true,
    device,
    preferredFormat: 'bgra8unorm',
    drawingBufferWidth: 64,
    drawingBufferHeight: 64,
    writeBuffer,
    createShaderModule: (code: string, label?: string) => {
      h.modules.push({ code, label });
      return { label };
    },
  } as unknown as Context;

  return h;
}

function makeLayer(harness?: MockHarness): { layer: CloudLayer; h: MockHarness } {
  const h = harness ?? makeHarness();
  const layer = new CloudLayer(h.ctx, { center: [121.5, 31.0] });
  return { layer, h };
}

function makeEncoder(): {
  encoder: GPUCommandEncoder;
  renderPasses: Record<string, ReturnType<typeof vi.fn>>[];
  computePasses: Record<string, ReturnType<typeof vi.fn>>[];
} {
  const renderPasses: Record<string, ReturnType<typeof vi.fn>>[] = [];
  const computePasses: Record<string, ReturnType<typeof vi.fn>>[] = [];
  const encoder = {
    beginRenderPass: vi.fn((desc: GPURenderPassDescriptor) => {
      const p = {
        desc,
        setPipeline: vi.fn(),
        setBindGroup: vi.fn(),
        draw: vi.fn(),
        end: vi.fn(),
      };
      renderPasses.push(p);
      return p;
    }),
    beginComputePass: vi.fn((_desc: GPUComputePassDescriptor) => {
      const p = {
        setPipeline: vi.fn(),
        setBindGroup: vi.fn(),
        dispatchWorkgroups: vi.fn(),
        end: vi.fn(),
      };
      computePasses.push(p);
      return p;
    }),
  } as unknown as GPUCommandEncoder;
  return { encoder, renderPasses, computePasses };
}

function makeCamera(): Camera {
  const cam = new Camera();
  cam.position = new Cartesian3(0, 0, 2);
  cam.direction = new Cartesian3(0, 0, -1);
  cam.up = new Cartesian3(0, 1, 0);
  return cam;
}

function lastWrite(h: MockHarness, label: string): WriteRec | undefined {
  const list = h.writes.filter((w) => w.label === label);
  return list[list.length - 1];
}

function makeGBufferFake(): GBufferPass {
  return {
    textures: { albedoView: {}, normalView: {}, worldPosView: {} },
    sampler: {},
  } as unknown as GBufferPass;
}

beforeAll(() => {
  const g = globalThis as Record<string, unknown>;
  g.GPUTextureUsage ??= { TEXTURE_BINDING: 0x100, STORAGE_BINDING: 0x80, COPY_DST: 0x08, RENDER_ATTACHMENT: 0x10 };
  g.GPUShaderStage ??= { VERTEX: STAGE_VERTEX, FRAGMENT: STAGE_FRAGMENT, COMPUTE: STAGE_COMPUTE };
  g.GPUBufferUsage ??= { MAP_READ: 0x01, VERTEX: 0x20, INDEX: 0x10, UNIFORM: 0x40, COPY_DST: 0x08 };
});

describe('CloudLayer - IPass 生命周期', () => {
  it('prepare 幂等：重复调用不重复建管线/纹理/采样器', () => {
    const { layer, h } = makeLayer();
    layer.prepare();
    layer.prepare();
    expect(h.renderPipelines).toHaveLength(1);
    expect(h.computePipelines).toHaveLength(1);
    expect(h.textures).toHaveLength(1);
    expect(h.samplers).toHaveLength(1);
    expect(h.buffers.map((b) => b.size).sort((a, b) => a - b))
      .toEqual(
        [CLOUD_CAMERA_UBO_BYTES, CLOUD_PARAMS_UBO_BYTES, CIRRUS_PARAMS_UBO_BYTES, CLOUD_SHADOW_PARAMS_UBO_BYTES]
          .sort((a, b) => a - b),
      );
  });

  it('execute 全链路：compute 烘焙 + loadOp load 渲染 + 深度透传', () => {
    const { layer, h } = makeLayer();
    const depthView = { label: 'gbufferDepth' };
    layer.setDepthView(() => depthView as unknown as GPUTextureView);
    layer.updateCamera(makeCamera());
    layer.prepare();
    const { encoder, renderPasses, computePasses } = makeEncoder();
    const target = { label: 'swapchain' } as unknown as GPUTextureView;

    layer.execute(encoder, target, 1.0);

    expect(computePasses).toHaveLength(1);
    expect(computePasses[0].dispatchWorkgroups).toHaveBeenCalledWith(64, 64); // 512/8
    expect(renderPasses).toHaveLength(1);
    const desc = renderPasses[0].desc as GPURenderPassDescriptor;
    expect(desc.colorAttachments![0].loadOp).toBe('load');
    expect(desc.depthStencilAttachment!.view).toBe(depthView);
    expect(renderPasses[0].draw).toHaveBeenCalledWith(3);
    expect(lastWrite(h, 'CloudLayer.CameraUBO')).toBeDefined();
    expect(lastWrite(h, 'CloudLayer.ParamsUBO')).toBeDefined();
    expect(lastWrite(h, 'CloudLayer.ShadowUBO')).toBeDefined();

    // 风漂移推进：dt = 1s × 9 m/s → 东向 +0.009 km（默认风向 0 = 东）。
    const before = lastWrite(h, 'CloudLayer.ShadowUBO')!.f32[20];
    layer.execute(encoder, target, 2.0);
    const after = lastWrite(h, 'CloudLayer.ShadowUBO')!.f32[20];
    expect(after - before).toBeCloseTo(0.009, 6);
  });

  it('深度安全闸：未 setDepthView 不绘制（同 AtmosphereSky 协议）', () => {
    const { layer } = makeLayer();
    layer.updateCamera(makeCamera());
    layer.prepare();
    const { encoder, renderPasses, computePasses } = makeEncoder();
    layer.execute(encoder, {} as GPUTextureView, 1.0);
    expect(renderPasses).toHaveLength(0);
    expect(computePasses).toHaveLength(0);
  });

  it('destroy 后 cloudShadow / cloudShadowSource 均 null，资源全部释放', () => {
    const { layer, h } = makeLayer();
    layer.prepare();
    expect(layer.cloudShadow).not.toBeNull();
    layer.destroy();
    expect(layer.cloudShadow).toBeNull();
    expect(layer.cloudShadowSource).toBeNull();
    expect(h.buffers.every((b) => b.destroyed)).toBe(true);
    layer.execute(makeEncoder().encoder, {} as GPUTextureView, 1.0); // 幂等无害
  });

  it('setTargetFormat：格式变更重建渲染管线，同格式幂等早退', () => {
    const { layer, h } = makeLayer();
    layer.prepare();
    expect(h.renderPipelines[0].fragment?.targets?.[0]?.format).toBe('bgra8unorm');

    layer.setTargetFormat('rgba16float');
    layer.prepare();
    expect(h.renderPipelines).toHaveLength(2);
    expect(h.renderPipelines[1].fragment?.targets?.[0]?.format).toBe('rgba16float');
    // 烘焙 compute 管线与格式无关，不重建。
    expect(h.computePipelines).toHaveLength(1);

    layer.setTargetFormat('rgba16float');
    layer.prepare();
    expect(h.renderPipelines).toHaveLength(2); // 幂等
  });
});

describe('CloudLayer - 云影契约', () => {
  it('prepare 前 cloudShadow 为 null', () => {
    const { layer } = makeLayer();
    expect(layer.cloudShadow).toBeNull();
    expect(layer.cloudShadowSource).toBeNull();
  });

  it('cloudShadow 结构：中心/基/半径/范围齐全，source 身份稳定', () => {
    const { layer, h } = makeLayer();
    layer.prepare();
    const info = layer.cloudShadow!;
    expect(info.extentM).toBe(8192);
    expect(info.centerEcef).toHaveLength(3);
    expect(info.eastDir).toHaveLength(3);
    expect(info.northDir).toHaveLength(3);
    expect(info.sunDirection).toHaveLength(3);
    // 归一化 ECEF 椭球 |p|≈0.9933~1（引擎 y/z 压扁约定），云高在其上加 2200m/R。
    expect(info.cloudRadiusNorm).toBeGreaterThan(0.99);
    expect(info.cloudRadiusNorm).toBeLessThan(1.01);
    // 中心 = 星下点方向 × 云球半径（模长 ≈ Rc）。
    const norm = Math.hypot(info.centerEcef[0], info.centerEcef[1], info.centerEcef[2]);
    expect(norm).toBeCloseTo(info.cloudRadiusNorm, 6);

    const source = layer.cloudShadowSource!;
    expect(source.view).toBe(info.view);
    expect(h.bindGroups.length).toBeGreaterThan(0);
    expect(layer.cloudShadowSource).toBe(layer.cloudShadowSource); // 身份稳定（bind group 缓存键）
  });

  it('coverage / extent / intensity setter 即时写 UBO（布局断言）', () => {
    const { layer, h } = makeLayer();
    layer.prepare();
    layer.setCoverage(0.8);
    expect(lastWrite(h, 'CloudLayer.ParamsUBO')!.f32[16]).toBeCloseTo(0.8, 6); // params0.x @64
    expect(lastWrite(h, 'CloudLayer.ShadowUBO')!.f32[14]).toBeCloseTo(0.8, 6); // params.z @48+8
    layer.setShadowExtentM(4096);
    expect(lastWrite(h, 'CloudLayer.ShadowUBO')!.f32[12]).toBeCloseTo(ES / 4096, 3); // extentScale @48
    layer.setShadowIntensity(0.4);
    expect(lastWrite(h, 'CloudLayer.ShadowUBO')!.f32[13]).toBeCloseTo(0.4, 6); // intensity @48+4
  });

  it('非法 setter 值抛错（OceanFFTWater 风格校验）', () => {
    const { layer } = makeLayer();
    expect(() => layer.setCoverage(NaN)).toThrow();
    expect(() => layer.setSunDirection([0, 0, 0])).toThrow();
    expect(() => layer.setCloudHeightM(1)).toThrow();
  });
});

describe('CloudLayer - 卷云第二层', () => {
  it('默认参数零回归：enable=true 常态存在，48B UBO 布局写入正确', () => {
    const { layer, h } = makeLayer();
    layer.setDepthView(() => ({}) as unknown as GPUTextureView);
    layer.updateCamera(makeCamera());
    layer.prepare();
    const { encoder } = makeEncoder();
    layer.execute(encoder, {} as GPUTextureView, 1.0);

    const c = lastWrite(h, 'CloudLayer.CirrusUBO')!;
    expect(c.f32.length).toBe(CIRRUS_PARAMS_UBO_BYTES / 4);
    expect(c.f32[4]).toBe(1); // params0.x = enable（默认 true，卷云常态存在）
    expect(c.f32[5]).toBeCloseTo(0.42, 6); // params0.y = coverage
    expect(c.f32[6]).toBeCloseTo(0.55, 6); // params0.z = density
    expect(c.f32[8]).toBeCloseTo(32 * Math.PI / 180, 6); // params1.x = stretchRad
    expect(c.f32[9]).toBeCloseTo(0.35, 6); // params1.y = streak
  });

  it('setCirrus 部分字段更新 + heightM 独立云高球面（高于主云）', () => {
    const { layer, h } = makeLayer();
    layer.prepare();
    layer.setCirrus({ coverage: 0.7, enable: false });
    let c = lastWrite(h, 'CloudLayer.CirrusUBO')!;
    expect(c.f32[4]).toBe(0); // 关闭整层
    expect(c.f32[5]).toBeCloseTo(0.7, 6);
    expect(c.f32[6]).toBeCloseTo(0.55, 6); // 未传字段不动

    layer.setCirrus({ enable: true, stretchDirDeg: 120, streak: 0.8, heightM: 9000 });
    c = lastWrite(h, 'CloudLayer.CirrusUBO')!;
    expect(c.f32[4]).toBe(1);
    expect(c.f32[8]).toBeCloseTo(120 * Math.PI / 180, 6);
    expect(c.f32[9]).toBeCloseTo(0.8, 6);
    // 卷云半径 @12B 高于主云半径（CloudParams.planeCenter.w @16B）。
    const p = lastWrite(h, 'CloudLayer.ParamsUBO')!;
    expect(c.f32[3]).toBeGreaterThan(p.f32[7]);
    expect(c.f32[3] - p.f32[7]).toBeCloseTo((9000 - 2200) / ES, 6);
  });

  it('非法 setCirrus 值抛错（部分字段校验，合法字段不受牵连）', () => {
    const { layer } = makeLayer();
    expect(() => layer.setCirrus({ coverage: NaN })).toThrow();
    expect(() => layer.setCirrus({ heightM: 10 })).toThrow();
    expect(() => layer.setCirrus({ density: Infinity })).toThrow();
    expect(() => layer.setCirrus({ stretchDirDeg: NaN })).toThrow();
    expect(() => layer.setCirrus({ streak: NaN })).toThrow();
    expect(() => layer.setCirrus({ enable: 'yes' as unknown as boolean })).toThrow();
  });

  it('渲染与烘焙 bind group 均挂 CirrusUBO（binding 2 同一 buffer）', () => {
    const { layer, h } = makeLayer();
    layer.prepare();
    const renderBG = h.bindGroups.find((b) => b.label === 'CloudLayer.Render.BG')!;
    const bakeBG = h.bindGroups.find((b) => b.label === 'CloudLayer.Bake.BG')!;
    expect(renderBG.entries).toHaveLength(3);
    expect(bakeBG.entries).toHaveLength(3);
    const cirrusBuf = h.buffers.find((b) => b.label === 'CloudLayer.CirrusUBO');
    // 两处 entry 资源是同一底层 buffer（mock 记录对象同引用）。
    expect(renderBG.entries[2].resource).toEqual({ buffer: cirrusBuf });
    expect(bakeBG.entries[2].resource).toEqual({ buffer: cirrusBuf });
  });

  it('构造器 cirrus 选项与 setCirrus 等价', () => {
    const { layer, h } = makeLayer();
    const layer2 = new CloudLayer(h.ctx, {
      center: [121.5, 31.0],
      cirrus: { enable: false, coverage: 0.9, heightM: 10500 },
    });
    layer2.setDepthView(() => ({}) as unknown as GPUTextureView);
    layer2.updateCamera(makeCamera());
    layer2.prepare();
    layer2.execute(makeEncoder().encoder, {} as GPUTextureView, 1.0);
    const c = lastWrite(h, 'CloudLayer.CirrusUBO')!;
    expect(c.f32[4]).toBe(0);
    expect(c.f32[5]).toBeCloseTo(0.9, 6);
  });
});

describe('CloudShader 源串结构', () => {
  it('渲染 / 烘焙入口与 UBO 字节常量自洽', () => {
    expect(CLOUD_CAMERA_UBO_BYTES).toBe(96);
    expect(CLOUD_PARAMS_UBO_BYTES).toBe(128);
    expect(CIRRUS_PARAMS_UBO_BYTES).toBe(48);
    expect(CLOUD_SHADOW_PARAMS_UBO_BYTES).toBe(96);
    // 渲染：云高球面求交 + 预乘 alpha；烘焙：rgba8unorm storage（r8 不可 storage）。
    expect(CLOUD_RENDER_WGSL).toContain('cloudRaySphereNearest');
    expect(CLOUD_RENDER_WGSL).toContain('return vec4<f32>(premul, alpha)');
    expect(CLOUD_RENDER_WGSL).toContain('CLOUD_EARTH_SCALE');
    expect(CLOUD_RENDER_WGSL).toContain('cloudDensity');
    expect(CLOUD_SHADOW_BAKE_WGSL).toContain('texture_storage_2d<rgba8unorm, write>');
    expect(CLOUD_SHADOW_BAKE_WGSL).toContain('@workgroup_size(8, 8)');
    expect(CLOUD_SHADOW_BAKE_WGSL).toContain('cloudDensity');
  });

  it('主云散射升级：Beer-Powder + 银边 + 半球环境光 + 3 方向能量传播', () => {
    // Beer-Powder：穿透项 × 粉末项（峰形能量，云心亮边缘透光）。
    expect(CLOUD_RENDER_WGSL).toContain('exp(-cont * CLOUD_EXTINCTION)');
    expect(CLOUD_RENDER_WGSL).toContain('1.0 - exp(-cont * CLOUD_EXTINCTION * 2.0)');
    // 银边：视线朝太阳 × 薄云 (1-dens) 的发丝光。
    expect(CLOUD_RENDER_WGSL).toContain(
      'pow(clamp(dot(rd, cloud.sunDirection.xyz), 0.0, 1.0), 4.0) * (1.0 - dens) * 0.85',
    );
    // 云底环境光：半球近似 × 光深增亮地板（厚云芯多重散射不塌黑）。
    expect(CLOUD_RENDER_WGSL).toContain(
      'mix(cloud.tintLow.rgb, cloud.tintHigh.rgb, hemi)',
    );
    expect(CLOUD_RENDER_WGSL).toContain('(0.35 + 0.45 * cont)'); // 2026-09-30 连续云内深度驱动（cloudFieldN）
    // 多方向能量传播：太阳向 + ±60° 正交抖动，0.5/0.25/0.25 加权。
    expect(CLOUD_RENDER_WGSL).toContain('let dSun = cloudDensity');
    expect(CLOUD_RENDER_WGSL).toContain('let dSideA = cloudDensity');
    expect(CLOUD_RENDER_WGSL).toContain('let dSideB = cloudDensity');
    expect(CLOUD_RENDER_WGSL).toContain(
      '(dSun - dens) * 0.5 + (dSideA - dens) * 0.25 + (dSideB - dens) * 0.25',
    );
    // 黄昏染色保留且与 Beer 能量乘法耦合（染色在 Beer 项之后整乘）。
    expect(CLOUD_RENDER_WGSL).toContain(
      'mix(mCol, mCol * vec3<f32>(1.3, 0.55, 0.32), dusk * 0.85)',
    );
  });

  it('卷云第二层：纤维噪声 / alpha 压顶 / 烘焙条纹影叠加', () => {
    // 纤维状：拉伸方向 uv 压缩 ×0.15 + 高频横向扰动。
    expect(CLOUD_RENDER_WGSL).toContain('fn cirrusDensity');
    expect(CLOUD_RENDER_WGSL).toContain('q.x * 0.15');
    // 同一 pass 内先卷云（远）后主云（近）over 叠加。
    expect(CLOUD_RENDER_WGSL).toContain('cirrus.params0.x > 0.5');
    expect(CLOUD_RENDER_WGSL).toContain('cirrus.planeCenter.w');
    // alpha 压 ≤ 0.5。
    expect(CLOUD_RENDER_WGSL).toContain('min(clamp(cDens * cirrus.params0.z, 0.0, 1.0), 0.42)');
    // 烘焙：卷云密度 ×0.3 低权重叠加（enable 闸）。
    expect(CLOUD_SHADOW_BAKE_WGSL).toContain('cirrusDensity');
    expect(CLOUD_SHADOW_BAKE_WGSL).toContain('clamp(dens + cD * 0.3, 0.0, 1.0)');
    // 烘焙与渲染共用同一 CirrusParams UBO（binding 2），96B 影 UBO 契约不动。
    expect(CLOUD_SHADOW_BAKE_WGSL).toContain('@group(0) @binding(2) var<uniform> cirrus : CirrusParams');
    expect(CLOUD_SHADOW_BAKE_WGSL).toContain('struct CloudShadowParams');
    // 消费侧生成器契约零回归：receiver module 无卷云痕迹（逐字节稳定）。
    expect(CLOUD_SHADOW_RECEIVER_MODULE).not.toContain('cirrus');
  });
});

describe('LightingShader 云影变体', () => {
  it('既有 4 导出零回归：不含云影痕迹', () => {
    for (const s of [LIGHTING_SHADER, LIGHTING_SHADER_HDR, LIGHTING_SHADER_SHADOW, LIGHTING_SHADER_SHADOW_HDR]) {
      expect(s).not.toContain('cloudShadowFactor');
      expect(s).not.toContain('@group(3)');
    }
    expect(LIGHTING_SHADER).not.toContain('cssmShadowFactor');
    expect(LIGHTING_SHADER_SHADOW).toContain('cssmShadowFactor');
  });

  it('云影 4 导出：锚点注入正确，与 CSSM 正交组合', () => {
    expect(LIGHTING_SHADER_CLOUD).toContain('cloudShadowFactor');
    expect(LIGHTING_SHADER_CLOUD).not.toContain('cssmShadowFactor');
    expect(LIGHTING_SHADER_CLOUD_HDR).toContain('cloudShadowFactor');
    expect(LIGHTING_SHADER_SHADOW_CLOUD).toContain('cloudShadowFactor');
    expect(LIGHTING_SHADER_SHADOW_CLOUD).toContain('cssmShadowFactor');
    expect(LIGHTING_SHADER_SHADOW_CLOUD).toContain('viewDist'); // CSSM 循环锚点已替换
    expect(LIGHTING_SHADER_SHADOW_CLOUD_HDR).toContain('cloudShadowFactor');
    // 8 变体两两不同。
    expect(new Set([
      LIGHTING_SHADER, LIGHTING_SHADER_HDR,
      LIGHTING_SHADER_SHADOW, LIGHTING_SHADER_SHADOW_HDR,
      LIGHTING_SHADER_CLOUD, LIGHTING_SHADER_CLOUD_HDR,
      LIGHTING_SHADER_SHADOW_CLOUD, LIGHTING_SHADER_SHADOW_CLOUD_HDR,
    ]).size).toBe(8);
    // 云影循环替换正确：仅第 0 盏光乘因子。
    expect(LIGHTING_SHADER_CLOUD).toContain('shadowFactor = cloudShadowFactor(worldPos);');
    expect(LIGHTING_SHADER_SHADOW_CLOUD).toContain('cssmShadowFactor(worldPos, normal, viewDist)\n        * cloudShadowFactor(worldPos);');
  });

  it('锚点漂移抛错（禁止静默产出错误 shader）', () => {
    expect(() => injectCloudShadowCore('fn fs_main() {}')).toThrow(/anchors drifted/);
  });
});

describe('LightingPass 云影接线', () => {
  const CLOUD_SRC = {
    view: { label: 'cloudShadowView' },
    ubo: { label: 'cloudShadowUbo' },
    sampler: { label: 'cloudShadowSampler' },
  } as unknown as CloudShadowReceiverBinding;

  it('setCloudShadowSource(null)：零回归（原变体 + 2 组 + execute 只绑组 0/1）', () => {
    const h = makeHarness();
    const pass = new LightingPass(h.ctx, makeGBufferFake(), 'bgra8unorm');
    pass.setCloudShadowSource(null); // 未接时早退（同一引用），零开销
    pass.prepare();
    expect(h.modules[h.modules.length - 1].code).toBe(LIGHTING_SHADER);
    expect(h.pipelineLayouts[h.pipelineLayouts.length - 1].bindGroupLayouts).toHaveLength(2);

    const { encoder, renderPasses } = makeEncoder();
    pass.execute(encoder, {} as GPUTextureView, 1.0);
    expect(renderPasses[0].setBindGroup).toHaveBeenCalledTimes(2);
    expect(renderPasses[0].setBindGroup).toHaveBeenCalledWith(0, expect.anything());
    expect(renderPasses[0].setBindGroup).toHaveBeenCalledWith(1, expect.anything());
    expect(renderPasses[0].draw).toHaveBeenCalledWith(3);
  });

  it('接入云影：3 组布局 + 云影变体 + execute 补绑组 3', () => {
    const h = makeHarness();
    const pass = new LightingPass(h.ctx, makeGBufferFake(), 'bgra8unorm');
    pass.setCloudShadowSource(CLOUD_SRC);
    expect(h.modules[h.modules.length - 1].code).toBe(LIGHTING_SHADER_CLOUD);
    expect(h.pipelineLayouts[h.pipelineLayouts.length - 1].bindGroupLayouts).toHaveLength(3);
    // 组 3 bind group：uniform + texture + sampler 三资源。
    const bg3 = h.bindGroups[h.bindGroups.length - 1];
    expect(bg3.entries).toHaveLength(3);
    expect(bg3.entries[0].resource).toEqual({ buffer: CLOUD_SRC.ubo });
    expect(bg3.entries[1].resource).toBe(CLOUD_SRC.view);
    expect(bg3.entries[2].resource).toBe(CLOUD_SRC.sampler);

    const { encoder, renderPasses } = makeEncoder();
    pass.execute(encoder, {} as GPUTextureView, 1.0);
    expect(renderPasses[0].setBindGroup).toHaveBeenCalledWith(3, expect.anything());

    // 同一引用重复传入零开销早退（管线不重建）。
    const pipelinesBefore = h.renderPipelines.length;
    pass.setCloudShadowSource(CLOUD_SRC);
    expect(h.renderPipelines.length).toBe(pipelinesBefore);
  });

  it('摘除云影：回到原双组管线与原变体', () => {
    const h = makeHarness();
    const pass = new LightingPass(h.ctx, makeGBufferFake(), 'bgra8unorm');
    pass.setCloudShadowSource(CLOUD_SRC);
    pass.setCloudShadowSource(null);
    expect(h.modules[h.modules.length - 1].code).toBe(LIGHTING_SHADER);
    expect(h.pipelineLayouts[h.pipelineLayouts.length - 1].bindGroupLayouts).toHaveLength(2);

    const { encoder, renderPasses } = makeEncoder();
    pass.execute(encoder, {} as GPUTextureView, 1.0);
    const calls = renderPasses[0].setBindGroup.mock.calls.map((c) => c[0]);
    expect(calls).not.toContain(3);
  });
});
