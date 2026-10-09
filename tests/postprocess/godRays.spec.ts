import { describe, expect, it, vi } from 'vitest';

// 真实 SunPosition 随时刻漂移 → 太阳 NDC 不定，pass 计数会闪；
// 打桩成确定性方向：identity viewProj 下投到 ndc=(0, -0.5)，
// 落在 flare 1.3 与 godRays 2.0 两个 margin 内。
vi.mock('../../src/scene/SunPosition', () => ({
  SunPosition: { computeSunDirection: () => [0, 0.5, -1] },
}));

import { PostProcessManager } from '../../src/postprocessing/PostProcessManager';
import { GodRaysPass } from '../../src/postprocessing/passes/GodRaysPass';
import {
  defaultPostProcessConfig,
  sanitizeGodRaysParams,
  validatePostProcessConfig,
} from '../../src/postprocessing/PostProcessConfig';
import { GOD_RAYS_SHADER } from '../../src/shaders/postprocess/GodRaysShader';
import { COMPOSITE_SHADER } from '../../src/shaders/postprocess/CompositeShader';
import { UNDERWATER_FOG_SHADER } from '../../src/shaders/UnderwaterFogShader';
import { UnderwaterFog } from '../../src/effect/UnderwaterFog';
import type { Context } from '../../src/renderer/Context';
import type { Camera } from '../../src/scene/Camera';

// node 环境无 WebGPU 全局枚举 —— 桩范式照 tests/postprocess/postProcessManager.spec.ts
const g = globalThis as unknown as Record<string, unknown>;
g.GPUTextureUsage ??= { RENDER_ATTACHMENT: 0x10, TEXTURE_BINDING: 0x08, COPY_DST: 0x04 };
g.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
g.GPUBufferUsage ??= { UNIFORM: 0x40, COPY_DST: 0x08 };

/** 记录型桩 device：统计创建调用（管线/绑定/UBO/纹理）与 writeBuffer。 */
function makeStubDevice() {
  const calls = {
    shaders: 0, bgls: 0, pipelines: 0, buffers: 0,
    bindGroups: 0, textures: 0, writeBuffer: 0,
  };
  const device = {
    createShaderModule: () => { calls.shaders++; return {}; },
    createBindGroupLayout: () => { calls.bgls++; return {}; },
    createPipelineLayout: () => ({}),
    createRenderPipeline: () => { calls.pipelines++; return {}; },
    createBuffer: () => { calls.buffers++; return { destroy: () => undefined }; },
    createBindGroup: () => { calls.bindGroups++; return {}; },
    createTexture: () => { calls.textures++; return { createView: () => ({}) }; },
    createSampler: () => ({}),
    queue: { writeBuffer: () => { calls.writeBuffer++; } },
  } as unknown as GPUDevice;
  return { device, calls };
}

/** 记录型桩 encoder：收集 beginRenderPass 的 label（顺序 = 编码的 pass 链）。 */
function makeStubEncoder() {
  const labels: string[] = [];
  const encoder = {
    beginRenderPass: (desc: { label?: string }) => {
      labels.push(desc.label ?? '?');
      return {
        setPipeline: () => undefined,
        setBindGroup: () => undefined,
        draw: () => undefined,
        end: () => undefined,
      };
    },
  } as unknown as GPUCommandEncoder;
  return { encoder, labels };
}

/** identity viewProj 桩相机（列主序 values；配 mock 太阳 → ndc=(0,-0.5)）。 */
function makeStubCamera(): Camera {
  return {
    viewProjectionMatrix: {
      values: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    },
  } as unknown as Camera;
}

/** bloom(默认 5 级: 1+4+4) + composite + tone = 11 个基础 pass。 */
const BASE_PASS_COUNT = 11;

describe('PostProcessConfig godRays 夹取与默认值', () => {
  it('默认配置：godRays.enabled=false（90 页零回归），validate 无告警', () => {
    const cfg = defaultPostProcessConfig();
    expect(cfg.godRays.enabled).toBe(false);
    expect(cfg.godRays.intensity).toBeCloseTo(0.6);
    expect(cfg.godRays.tint).toEqual([1.0, 0.93, 0.85]);
    expect(validatePostProcessConfig(cfg)).toEqual([]);
  });

  it('sanitizeGodRaysParams：数值域夹取', () => {
    const mk = (over: Partial<ReturnType<typeof defaultPostProcessConfig>['godRays']>) =>
      sanitizeGodRaysParams({ ...defaultPostProcessConfig().godRays, ...over });
    expect(mk({ intensity: -1 }).intensity).toBe(0);
    expect(mk({ intensity: 99 }).intensity).toBe(4);
    expect(mk({ decay: 0.2 }).decay).toBe(0.5);
    expect(mk({ decay: 1.5 }).decay).toBeCloseTo(0.999);
    expect(mk({ density: 0.01 }).density).toBeCloseTo(0.1);
    expect(mk({ density: 9 }).density).toBe(3);
    expect(mk({ weight: -1 }).weight).toBe(0);
    expect(mk({ weight: 5 }).weight).toBe(2);
    expect(mk({ threshold: -1 }).threshold).toBe(0);
    expect(mk({ threshold: 9 }).threshold).toBe(4);
  });

  it('sanitizeGodRaysParams：NaN 归到域下界（不抛异常打崩渲染循环）', () => {
    const gr = sanitizeGodRaysParams({
      ...defaultPostProcessConfig().godRays,
      intensity: NaN, decay: NaN, density: NaN, weight: NaN, threshold: NaN,
      tint: [NaN, 99, -1],
    });
    expect(gr.intensity).toBe(0);
    expect(gr.decay).toBe(0.5);
    expect(gr.density).toBeCloseTo(0.1);
    expect(gr.weight).toBe(0);
    expect(gr.threshold).toBe(0);
    expect(gr.tint).toEqual([0, 4, 0]);
  });
});

describe('GodRaysPass（桩 device）', () => {
  it('prepare 懒创建管线/BGL/UBO 且幂等；execute 编码单个 render pass', () => {
    const { device, calls } = makeStubDevice();
    const pass = new GodRaysPass(device, 'rgba16float');

    pass.prepare();
    expect(calls.pipelines).toBe(1);
    expect(calls.bgls).toBe(1);
    expect(calls.buffers).toBe(1);

    pass.prepare(); // 幂等：不重复创建
    expect(calls.pipelines).toBe(1);

    const { encoder, labels } = makeStubEncoder();
    pass.execute(encoder, {} as GPUBindGroup, {} as GPUTextureView, {
      sunUv: [0.5, 0.75],
      intensity: 0.6,
      decay: 0.96,
      density: 0.92,
      weight: 0.45,
      threshold: 1.0,
      aspect: 16 / 9,
      tint: [1.0, 0.93, 0.85],
    });
    expect(labels).toEqual(['PostProcess.GodRays']);
    expect(calls.writeBuffer).toBe(1); // 每帧参数 UBO 下发

    pass.destroy();
  });

  it('shader 源串含径向 march 关键量（decay/STEPS/亮度 mask）', () => {
    expect(GOD_RAYS_SHADER).toContain('illum *= params.decay');
    expect(GOD_RAYS_SHADER).toContain('const STEPS');
    expect(GOD_RAYS_SHADER).toContain('lum - params.threshold');
  });
});

describe('PostProcessManager godRays 链插入（桩 device + 桩太阳）', () => {
  it('默认参数零回归：manager.enabled=false 时 execute 不编码任何 pass', () => {
    const { device } = makeStubDevice();
    const manager = new PostProcessManager(device, 'bgra8unorm');
    const { encoder, labels } = makeStubEncoder();
    manager.execute(encoder, {} as GPUTextureView, {} as GPUTextureView, 0, makeStubCamera());
    expect(labels.length).toBe(0);
  });

  it('godRays.enabled=false（后处理开）：execute 链不含 GodRays pass', () => {
    const { device } = makeStubDevice();
    const manager = new PostProcessManager(device, 'bgra8unorm');
    manager.enabled = true;
    expect(manager.godRays.enabled).toBe(false); // 默认关闭

    const { encoder, labels } = makeStubEncoder();
    manager.execute(encoder, {} as GPUTextureView, {} as GPUTextureView, 0, makeStubCamera());
    expect(labels.length).toBe(BASE_PASS_COUNT);
    expect(labels).not.toContain('PostProcess.GodRays');
  });

  it('godRays.enabled=true：Composite 前插入恰好一个 GodRays pass（顺序断言）', () => {
    const { device } = makeStubDevice();
    const manager = new PostProcessManager(device, 'bgra8unorm');
    manager.enabled = true;
    manager.godRays.enabled = true;

    const { encoder, labels } = makeStubEncoder();
    manager.execute(encoder, {} as GPUTextureView, {} as GPUTextureView, 0, makeStubCamera());

    expect(labels.filter((l) => l === 'PostProcess.GodRays').length).toBe(1);
    expect(labels.length).toBe(BASE_PASS_COUNT + 1);
    // 顺序：GodRays 在 Composite 之前（Composite 采样其输出）
    const gi = labels.indexOf('PostProcess.GodRays');
    const ci = labels.indexOf('PostProcess.Composite');
    expect(gi).toBeGreaterThanOrEqual(0);
    expect(ci).toBeGreaterThan(gi);
  });

  it('运行时开关往返：off→on→off pass 数随之增减（bindGroup 重建路径）', () => {
    const { device } = makeStubDevice();
    const manager = new PostProcessManager(device, 'bgra8unorm');
    manager.enabled = true;

    const run = () => {
      const { encoder, labels } = makeStubEncoder();
      manager.execute(encoder, {} as GPUTextureView, {} as GPUTextureView, 0, makeStubCamera());
      return labels;
    };
    expect(run().length).toBe(BASE_PASS_COUNT);          // off
    manager.godRays.enabled = true;
    expect(run().length).toBe(BASE_PASS_COUNT + 1);      // on
    expect(run().length).toBe(BASE_PASS_COUNT + 1);      // on 稳定帧
    manager.godRays.enabled = false;
    expect(run().length).toBe(BASE_PASS_COUNT);          // off 回落
  });

  it('getConfig 快照含 godRays 且 validate 通过', () => {
    const { device } = makeStubDevice();
    const manager = new PostProcessManager(device, 'bgra8unorm');
    manager.godRays.enabled = true;
    const cfg = manager.getConfig();
    expect(cfg.godRays.enabled).toBe(true);
    expect(manager.validate()).toEqual([]);
    manager.godRays.decay = 3; // 越界 → validate 告警
    expect(manager.validate().some((p) => p.includes('godRays.decay'))).toBe(true);
  });
});

describe('projectSun margin 语义（桩太阳 + 桩相机）', () => {
  it('返回 uv + ndc；margin 收紧可屏蔽（god rays 余晖放宽的对照面）', () => {
    const cam = makeStubCamera();
    const sun = PostProcessManager.projectSun(cam);
    expect(sun).not.toBeNull();
    expect(sun!.ndc).toEqual([0, -0.5]);
    expect(sun!.uv).toEqual([0.5, 0.75]); // 0.5 - (-0.5)*0.5

    // |ndc|=0.5：margin 0.3 内不可见，2.0 内可见
    expect(PostProcessManager.projectSun(cam, 0.3)).toBeNull();
    expect(PostProcessManager.projectSun(cam, 2.0)).not.toBeNull();
  });
});

describe('UnderwaterFog sunShaft（Track G 水下光柱联动）', () => {
  it('sunShaftIntensity 默认 0（零回归），可经 opts / setter 设置', () => {
    const uw = new UnderwaterFog({} as unknown as Context);
    expect(uw.sunShaftIntensity).toBe(0);

    const uw2 = new UnderwaterFog({} as unknown as Context, { sunShaftIntensity: 1.5 });
    expect(uw2.sunShaftIntensity).toBe(1.5);
    uw2.setSunShaftIntensity(0.8);
    expect(uw2.sunShaftIntensity).toBe(0.8);
  });

  it('shader 源串：体积焦散 march（causticsTex + 静态 IGN + HG g=0.85）', () => {
    // UBO 第 4 个 vec4（默认强度 0 = 乘 0 零贡献）
    expect(UNDERWATER_FOG_SHADER).toContain('sunShaft');
    expect(UNDERWATER_FOG_SHADER).toContain('uw.sunShaft.w');
    // march 采真焦散纹理（tidewater _shaftCode 同法）
    expect(UNDERWATER_FOG_SHADER).toContain('causticsTex');
    expect(UNDERWATER_FOG_SHADER).toContain('cNet');
    expect(UNDERWATER_FOG_SHADER).toContain('shaftAcc');
    // 绑定门控：未绑焦散源（cover=0）不 march
    expect(UNDERWATER_FOG_SHADER).toContain('cCenter.w > 0.5');
    // 静态 IGN jitter（无 TAA 防闪烁）
    expect(UNDERWATER_FOG_SHADER).toContain('52.9829189');
    // HG(g=0.85) 前向散射（tidewater 相函数）
    expect(UNDERWATER_FOG_SHADER).toContain('0.85');
    // Round 3 LDR 域标定：K=12 增益替代 tidewater HDR 域 sunE·sigS·2.5
    expect(UNDERWATER_FOG_SHADER).toContain('sunTint');
    expect(UNDERWATER_FOG_SHADER).toContain('12.0');
    // Round 3：窄锥前向峰（峰值 ×2.1）——仰视 Snell 窗独立束状漏光
    expect(UNDERWATER_FOG_SHADER).toContain('fwdBoost');
    // Round 3：俯视减益——沙面真焦散网不被上方辉光摊平
    expect(UNDERWATER_FOG_SHADER).toContain('downDamp');
    // Round 4 宽束重标定：LOD 1.4 宽级联 + 重映射 + 仰视加权
    expect(UNDERWATER_FOG_SHADER).toContain('(cNet - 0.72) * 2.6');
    expect(UNDERWATER_FOG_SHADER).toContain(', 1.4)');
    expect(UNDERWATER_FOG_SHADER).toContain('clamp((cNet - 0.72) * 2.6, -1.2, 2.2)');
    expect(UNDERWATER_FOG_SHADER).toContain('mix(1.0, 0.45');
    expect(UNDERWATER_FOG_SHADER).toContain('mix(1.0, 1.2');
    // Round 5：束软限幅膝（仰视防 Snell 窗过曝）+ 保对比合成
    expect(UNDERWATER_FOG_SHADER).toContain('beamPeak');
    expect(UNDERWATER_FOG_SHADER).toContain('0.55 * max(beamPeak - 0.55');
    expect(UNDERWATER_FOG_SHADER).toContain('dot(sunBeam, vec3<f32>(0.3333)) * 0.75');
  });

  it('bindCaustics：默认未绑不影响 sunShaftIntensity；null / 旧屏幕空间辐条已删', () => {
    const uw = new UnderwaterFog({} as unknown as Context);
    expect(uw.sunShaftIntensity).toBe(0);
    uw.setSunShaftIntensity(1.2);   // 未绑焦散源也可设强度（shader uniform 门控）
    expect(uw.sunShaftIntensity).toBe(1.2);
    expect(() => uw.bindCaustics(null)).not.toThrow();
    expect(() => uw.bindCaustics({
      view: () => null,
      coverMeters: 500,
      centerEcef: [1, 0, 0],
      eastEcef: [0, 1, 0],
      upEcef: [0, 0, 1],
      northEcef: [0, 1, 0],
      shallowDepthM: 1,
      deepDepthM: 4,
    })).not.toThrow();
    // 旧屏幕空间假光柱（放射星芒）已移除
    expect(UNDERWATER_FOG_SHADER).not.toContain('streak');
    expect(UNDERWATER_FOG_SHADER).not.toContain('hg * streak * shimmer');
  });

  it('Composite 源串含 godRays 采样输入与强度参数', () => {
    expect(COMPOSITE_SHADER).toContain('godTex');
    expect(COMPOSITE_SHADER).toContain('godRayIntensity');
    expect(COMPOSITE_SHADER).toContain('godRays * params.godRayIntensity');
  });
});
