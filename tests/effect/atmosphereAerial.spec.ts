import { describe, expect, it } from 'vitest';
import {
  LIGHTING_SHADER,
  LIGHTING_SHADER_HDR,
  LIGHTING_SHADER_SHADOW,
  LIGHTING_SHADER_SHADOW_HDR,
  LIGHTING_SHADER_CLOUD,
  LIGHTING_SHADER_CLOUD_HDR,
  LIGHTING_SHADER_SHADOW_CLOUD,
  LIGHTING_SHADER_SHADOW_CLOUD_HDR,
  LIGHTING_SHADER_AP,
  LIGHTING_SHADER_AP_HDR,
  LIGHTING_SHADER_AP_SHADOW,
  LIGHTING_SHADER_AP_SHADOW_HDR,
  LIGHTING_SHADER_AP_CLOUD,
  LIGHTING_SHADER_AP_CLOUD_HDR,
  LIGHTING_SHADER_AP_SHADOW_CLOUD,
  LIGHTING_SHADER_AP_SHADOW_CLOUD_HDR,
  injectAPCore,
  selectLightingShader,
} from '../../src/shaders/LightingShader';
import {
  ATMO_AERIAL_WGSL,
  ATMO_SKY_REFLECTION_WGSL,
  AERIAL_PARAMS_STRUCT_WGSL,
  AERIAL_PARAMS_UBO_BYTES,
  ATMO_PARAMS_STRUCT_WGSL,
} from '../../src/shaders/AtmosphereSkyShader';

/**
 * Track A 大气透视（aerial perspective）——地形雾随太阳/视角物理染色。
 * headless 纯源串断言（无 WebGPU adapter，同 cloudLayer.spec 约定）：
 *  - AP 模块（ATMO_AERIAL_WGSL）导出与绑定契约；
 *  - LightingShader 8 个 AP 变体锚点注入（组 4，与既有 8 变体正交，共 16）；
 *  - 既有 8 变体零回归（逐字节不受 AP 影响）；
 *  - apStrength = 0 恒走旧 limb haze（零回归闸）。
 */

const ORIGINAL_8 = [
  LIGHTING_SHADER, LIGHTING_SHADER_HDR,
  LIGHTING_SHADER_SHADOW, LIGHTING_SHADER_SHADOW_HDR,
  LIGHTING_SHADER_CLOUD, LIGHTING_SHADER_CLOUD_HDR,
  LIGHTING_SHADER_SHADOW_CLOUD, LIGHTING_SHADER_SHADOW_CLOUD_HDR,
];

const AP_8 = [
  LIGHTING_SHADER_AP, LIGHTING_SHADER_AP_HDR,
  LIGHTING_SHADER_AP_SHADOW, LIGHTING_SHADER_AP_SHADOW_HDR,
  LIGHTING_SHADER_AP_CLOUD, LIGHTING_SHADER_AP_CLOUD_HDR,
  LIGHTING_SHADER_AP_SHADOW_CLOUD, LIGHTING_SHADER_AP_SHADOW_CLOUD_HDR,
];

describe('AtmosphereSkyShader AP 模块', () => {
  it('atmAerialColor 源串存在：本地基 + atmSkyLuminance 同源查询，无日月盘/星空', () => {
    expect(ATMO_AERIAL_WGSL).toContain('fn atmAerialColor(dirWorld: vec3f, up: vec3f, sunDirWorld: vec3f) -> vec3f');
    expect(ATMO_AERIAL_WGSL).toContain('fn atmAerialLocalBasis(up: vec3f) -> mat3x3<f32>');
    expect(ATMO_AERIAL_WGSL).toContain('fn atmSkyLuminance(dirL: vec3f, sunL: vec3f) -> vec3f');
    expect(ATMO_AERIAL_WGSL).toContain('textureSampleLevel(atmSkyViewLUT, atmLinearClamp');
    // 雾不发光点：不含日盘/月盘/星空/多重散射采样。
    expect(ATMO_AERIAL_WGSL).not.toContain('skySunDisk');
    expect(ATMO_AERIAL_WGSL).not.toContain('skyMoon');
    expect(ATMO_AERIAL_WGSL).not.toContain('skyStars');
    expect(ATMO_AERIAL_WGSL).not.toContain('atmosphereSampleMultiScat');
  });

  it('与 ATMO_SKY_REFLECTION_WGSL 无符号冲突（OceanFFT 消费不被破坏）', () => {
    // 反射块导出签名逐字不动（OceanFFTShader 拼接依赖）。
    expect(ATMO_SKY_REFLECTION_WGSL).toContain('fn atmSkyReflectionRadiance(');
    expect(ATMO_SKY_REFLECTION_WGSL).toContain('fn atmLocalBasis(up: vec3f) -> mat3x3<f32>');
    // AP 块用独立命名的 atmAerialLocalBasis，两块即使同仓库并存也不撞符号。
    expect(ATMO_AERIAL_WGSL).not.toContain('fn atmLocalBasis');
    expect(ATMO_AERIAL_WGSL).not.toContain('atmSkyReflectionRadiance');
  });

  it('AerialParams UBO 32B 布局契约（T6 写入依据）', () => {
    expect(AERIAL_PARAMS_UBO_BYTES).toBe(32);
    // 偏移注释齐全：strength @0 / distM @4 / sunDir @16（vec3f 16B 对齐）。
    expect(AERIAL_PARAMS_STRUCT_WGSL).toContain('strength : f32,      // @0');
    expect(AERIAL_PARAMS_STRUCT_WGSL).toContain('distM    : f32,      // @4');
    expect(AERIAL_PARAMS_STRUCT_WGSL).toContain('sunDir   : vec3f,    // @16');
    expect(AERIAL_PARAMS_STRUCT_WGSL).toContain('struct AerialParams');
  });
});

describe('LightingShader AP 变体', () => {
  it('组 4 声明契约逐字存在（T6 BGL 依据：uniform/texture/sampler/uniform）', () => {
    for (const code of AP_8) {
      expect(code).toContain('@group(4) @binding(0) var<uniform> aerialParams     : AerialParams;');
      expect(code).toContain('@group(4) @binding(1) var atmSkyViewLUT             : texture_2d<f32>;');
      expect(code).toContain('@group(4) @binding(2) var atmLinearClamp            : sampler;');
      expect(code).toContain('@group(4) @binding(3) var<uniform> atmosphereParams : AtmosphereParams;');
      expect(code).toContain(ATMO_PARAMS_STRUCT_WGSL);
      expect(code).toContain(AERIAL_PARAMS_STRUCT_WGSL);
    }
  });

  it('AP 雾数学：fogAmount = 1-exp(-dist/distM)，雾色 = atmAerialColor(-viewDir, up, sunDir)', () => {
    for (const code of AP_8) {
      expect(code).toContain('let fogAmount = 1.0 - exp(-apDist / max(aerialParams.distM, 1.0));');
      expect(code).toContain('atmAerialColor(-viewDir, normalize(worldPos), normalize(aerialParams.sunDir))');
      expect(code).toContain('color = mix(color, fogCol, clamp(fogAmount * aerialParams.strength, 0.0, 1.0));');
      // 仅注入一处 @fragment 入口（声明前插不复制 fs_main）。
      expect(code.split('@fragment').length - 1).toBe(1);
    }
  });

  it('apStrength = 0 零回归：旧 limb haze 表达式保留为回退分支 + 显式开关守卫', () => {
    for (const code of AP_8) {
      expect(code).toContain('if (aerialParams.strength > 0.0) {');
      expect(code).toContain('let haze = vec3<f32>(0.1, 0.2, 0.4) * pow(limbFactor, 3.0) * 0.8;');
      expect(code).toContain('color = color + haze;');
    }
  });

  it('AP 与 CSSM/云影正交组合：8 变体各含对应阴影签名', () => {
    expect(LIGHTING_SHADER_AP).not.toContain('cssmShadowFactor');
    expect(LIGHTING_SHADER_AP).not.toContain('cloudShadowFactor');
    expect(LIGHTING_SHADER_AP_SHADOW).toContain('cssmShadowFactor');
    expect(LIGHTING_SHADER_AP_SHADOW).toContain('texture_depth_2d_array');
    expect(LIGHTING_SHADER_AP_SHADOW).toContain('shadowFactor * computeDirectionalLight');
    expect(LIGHTING_SHADER_AP_CLOUD).toContain('cloudShadowFactor');
    expect(LIGHTING_SHADER_AP_CLOUD).not.toContain('cssmShadowFactor');
    // 云影 + CSSM 双因子乘积串逐字保留（cloudLayer.spec 同款断言）。
    expect(LIGHTING_SHADER_AP_SHADOW_CLOUD).toContain('cssmShadowFactor(worldPos, normal, viewDist)\n        * cloudShadowFactor(worldPos);');
    // LDR/HDR 尾段分叉保留。
    for (const ldr of [LIGHTING_SHADER_AP, LIGHTING_SHADER_AP_SHADOW, LIGHTING_SHADER_AP_CLOUD, LIGHTING_SHADER_AP_SHADOW_CLOUD]) {
      expect(ldr).toContain('Reinhard');
    }
    for (const hdr of [LIGHTING_SHADER_AP_HDR, LIGHTING_SHADER_AP_SHADOW_HDR, LIGHTING_SHADER_AP_CLOUD_HDR, LIGHTING_SHADER_AP_SHADOW_CLOUD_HDR]) {
      expect(hdr).not.toContain('Reinhard');
    }
  });

  it('16 变体两两不同（8 旧 + 8 AP）', () => {
    expect(new Set([...ORIGINAL_8, ...AP_8]).size).toBe(16);
  });

  it('锚点漂移抛错（禁止静默产出错误 shader）', () => {
    expect(() => injectAPCore('fn fs_main() {}')).toThrow(/anchors drifted/);
    // 声明锚点在但雾尾段已改（锚点漂移的另一半）也须抛错。
    expect(() => injectAPCore('@fragment\nfn fs_main(in: VSOut) {}\n  let haze = vec3<f32>(9.9);\n  color = color + haze;'))
      .toThrow(/anchors drifted/);
  });
});

describe('既有 8 变体零回归（AP 不触碰）', () => {
  it('原 8 变体不含任何 AP 痕迹（组 4 / aerialParams / atmAerialColor）', () => {
    for (const code of ORIGINAL_8) {
      expect(code).not.toContain('@group(4)');
      expect(code).not.toContain('aerialParams');
      expect(code).not.toContain('atmAerialColor');
      expect(code).not.toContain('AerialParams');
    }
    // 旧雾路径逐字节保留。
    expect(LIGHTING_SHADER).toContain('let haze = vec3<f32>(0.1, 0.2, 0.4) * pow(limbFactor, 3.0) * 0.8;\n\n  color = color + haze;');
  });

  it('原 8 变体两两不同（cloudLayer.spec 同款守卫）', () => {
    expect(new Set(ORIGINAL_8).size).toBe(8);
  });
});

describe('LightingPass 16 变体选择（selectLightingShader，T6 接线）', () => {
  /** 期望映射：未接 AP → 原 8 变体；接 AP → 对应 LIGHTING_SHADER_AP_*。 */
  const EXPECT = (hdr: boolean, shadow: boolean, cloud: boolean, aerial: boolean): string => {
    if (!aerial) {
      if (!shadow) return cloud ? (hdr ? LIGHTING_SHADER_CLOUD_HDR : LIGHTING_SHADER_CLOUD)
                                : (hdr ? LIGHTING_SHADER_HDR : LIGHTING_SHADER);
      return cloud ? (hdr ? LIGHTING_SHADER_SHADOW_CLOUD_HDR : LIGHTING_SHADER_SHADOW_CLOUD)
                   : (hdr ? LIGHTING_SHADER_SHADOW_HDR : LIGHTING_SHADER_SHADOW);
    }
    if (!shadow) return cloud ? (hdr ? LIGHTING_SHADER_AP_CLOUD_HDR : LIGHTING_SHADER_AP_CLOUD)
                              : (hdr ? LIGHTING_SHADER_AP_HDR : LIGHTING_SHADER_AP);
    return cloud ? (hdr ? LIGHTING_SHADER_AP_SHADOW_CLOUD_HDR : LIGHTING_SHADER_AP_SHADOW_CLOUD)
                 : (hdr ? LIGHTING_SHADER_AP_SHADOW_HDR : LIGHTING_SHADER_AP_SHADOW);
  };

  it('16 组合穷举：hdr × CSSM × 云影 × AP 正交映射逐一对准', () => {
    for (const hdr of [false, true]) {
      for (const shadow of [false, true]) {
        for (const cloud of [false, true]) {
          for (const aerial of [false, true]) {
            expect(selectLightingShader(hdr, shadow, cloud, aerial))
              .toBe(EXPECT(hdr, shadow, cloud, aerial));
          }
        }
      }
    }
  });

  it('未接 AP（aerial=false）8 组合恒落原 8 变体（零回归闸）', () => {
    for (const hdr of [false, true]) {
      for (const shadow of [false, true]) {
        for (const cloud of [false, true]) {
          expect(EXPECT(hdr, shadow, cloud, false)).not.toContain('@group(4)');
        }
      }
    }
  });

  it('接 AP 8 组合含组 4 声明（aerialParams 绑定落地）', () => {
    for (const hdr of [false, true]) {
      for (const shadow of [false, true]) {
        for (const cloud of [false, true]) {
          expect(selectLightingShader(hdr, shadow, cloud, true))
            .toContain('@group(4) @binding(0) var<uniform> aerialParams     : AerialParams;');
        }
      }
    }
  });
});
