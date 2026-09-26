import { describe, expect, it } from 'vitest';
import { Camera } from '../../src/scene/Camera';
import { Cartesian3 } from '../../src/math/Cartesian3';
import {
  LIGHTING_SHADER,
  LIGHTING_SHADER_HDR,
  LIGHTING_SHADER_SHADOW,
  LIGHTING_SHADER_SHADOW_HDR,
} from '../../src/shaders/LightingShader';
import {
  SHADOW_CASCADE_COUNT,
  ShadowMap,
  computeCascadeCorners,
  computeCascadeSplitEnds,
  computeLightBasis,
  fitLightOrtho,
} from '../../src/shadow/ShadowMap';
import type { LightOrthoFit } from '../../src/shadow/ShadowMap';
import type { Context } from '../../src/renderer/Context';

/**
 * A1 CSSM 级联阴影 — 纯数学 / 纯字符串验证（无 GPU 依赖）：
 * 级联分割、光源正交基、正交拟合（NDC 包含性 + 深度方向）、
 * Lighting shader 阴影变体注入、updateFrame 冒烟。
 */

const ES = Camera.EARTH_SCALE;

/** 合成相机：原点上方 2（归一化）俯视 -z，60° 视角。 */
function makeCamera(): Camera {
  const cam = new Camera();
  cam.position = new Cartesian3(0, 0, 2);
  cam.direction = new Cartesian3(0, 0, -1);
  cam.up = new Cartesian3(0, 1, 0);
  cam.frustum.fov = Math.PI / 3;
  cam.frustum.aspectRatio = 1.0;
  cam.frustum.near = 0.001;
  cam.frustum.far = 2.0;
  return cam;
}

/** 列主序 mat4 × 点（w=1），返回 NDC xyz。 */
function project(m: Float32Array, p: Cartesian3): [number, number, number] {
  const x = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12];
  const y = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13];
  const z = m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14];
  return [x, y, z];
}

describe('computeCascadeSplitEnds', () => {
  it('远端 10km 时给出默认 250m / 1.5km / 10km 覆盖', () => {
    const ends = computeCascadeSplitEnds(1, 10000);
    expect(ends[0]).toBeCloseTo(250, 5);
    expect(ends[1]).toBeCloseTo(1500, 5);
    expect(ends[2]).toBeCloseTo(10000, 5);
  });

  it('相机远平面小于 10km 时按比例收缩', () => {
    const ends = computeCascadeSplitEnds(1, 5000);
    expect(ends[0]).toBeCloseTo(125, 5);
    expect(ends[2]).toBeCloseTo(5000, 5);
  });

  it('分割严格递增且不落在相机近平面之前', () => {
    const ends = computeCascadeSplitEnds(3000, 50000);
    expect(ends[0]).toBeGreaterThan(3000);
    expect(ends[1]).toBeGreaterThan(ends[0] * 2);
    expect(ends[2]).toBeGreaterThan(ends[1] * 2);
  });
});

describe('computeLightBasis', () => {
  it('正交归一，z 轴 = 光传播方向', () => {
    const dir = Cartesian3.normalize(new Cartesian3(0.5, -0.3, -0.8), new Cartesian3());
    const b = computeLightBasis(dir);
    expect(Cartesian3.dot(b.x, b.y)).toBeCloseTo(0, 6);
    expect(Cartesian3.dot(b.y, b.z)).toBeCloseTo(0, 6);
    expect(Cartesian3.dot(b.x, b.z)).toBeCloseTo(0, 6);
    expect(Cartesian3.magnitude(b.z)).toBeCloseTo(1, 6);
    expect(Cartesian3.dot(b.z, dir)).toBeCloseTo(1, 5);
  });

  it('光轴近竖直时不退化（无 NaN）', () => {
    const b = computeLightBasis(new Cartesian3(0.001, 0, -1));
    expect(Number.isFinite(b.x.x + b.x.y + b.x.z)).toBe(true);
    expect(Number.isFinite(b.y.x + b.y.y + b.y.z)).toBe(true);
  });
});

describe('fitLightOrtho', () => {
  const cam = makeCamera();
  const ends = computeCascadeSplitEnds(1, 10000);
  const corners: Cartesian3[] = Array.from({ length: 8 }, () => new Cartesian3());

  function fitCascade(i: number): LightOrthoFit {
    computeCascadeCorners(
      cam,
      (i === 0 ? 0 : ends[i - 1]) / ES,
      ends[i] / ES,
      corners,
    );
    return fitLightOrtho(corners, computeLightBasis(new Cartesian3(1, 0, 0)), 2048);
  }

  it('全部级联角点落在光源 NDC 盒内（xy∈[-1,1]，z∈[0,1]）', () => {
    for (let i = 0; i < SHADOW_CASCADE_COUNT; i++) {
      const fit = fitCascade(i);
      const m = fit.viewProj.values;
      for (const c of corners) {
        const [nx, ny, nz] = project(m, c);
        expect(nx).toBeGreaterThanOrEqual(-1 - 1e-4);
        expect(nx).toBeLessThanOrEqual(1 + 1e-4);
        expect(ny).toBeGreaterThanOrEqual(-1 - 1e-4);
        expect(ny).toBeLessThanOrEqual(1 + 1e-4);
        expect(nz).toBeGreaterThan(0);
        expect(nz).toBeLessThan(1);
      }
    }
  });

  it('深度沿光传播方向单调递增（面向太阳的表面深度更小）', () => {
    const fit = fitCascade(1);
    const b = computeLightBasis(new Cartesian3(1, 0, 0));
    const m = fit.viewProj.values;
    const center = new Cartesian3(0, 0, 2 - (ends[1] / ES) * 0.5);
    const near = Cartesian3.add(
      center,
      Cartesian3.multiplyByScalar(b.z, -0.001, new Cartesian3()),
      new Cartesian3());
    const far = Cartesian3.add(
      center,
      Cartesian3.multiplyByScalar(b.z, 0.001, new Cartesian3()),
      new Cartesian3());
    expect(project(m, near)[2]).toBeLessThan(project(m, far)[2]);
  });

  it('纹素世界尺寸为正且随级联跨度增大', () => {
    const t0 = fitCascade(0).texelWorld;
    const t2 = fitCascade(2).texelWorld;
    expect(t0).toBeGreaterThan(0);
    expect(t2).toBeGreaterThan(t0);
  });
});

describe('Lighting shader 阴影变体注入', () => {
  it('无阴影变体保持纯净（无 cssm / 组 2）', () => {
    expect(LIGHTING_SHADER).not.toContain('cssmShadowFactor');
    expect(LIGHTING_SHADER).not.toContain('@group(2)');
    expect(LIGHTING_SHADER_HDR).not.toContain('cssmShadowFactor');
  });

  it('阴影变体含组 2 绑定、PCF 采样与级联选择', () => {
    for (const code of [LIGHTING_SHADER_SHADOW, LIGHTING_SHADER_SHADOW_HDR]) {
      expect(code).toContain('@group(2) @binding(0) var<uniform> shadow');
      expect(code).toContain('texture_depth_2d_array');
      expect(code).toContain('sampler_comparison');
      expect(code).toContain('textureSampleCompareLevel');
      expect(code).toContain('cssmSelectCascade');
      expect(code).toContain('shadowFactor * computeDirectionalLight');
    }
  });

  it('LDR/HDR 尾段分叉保持不变', () => {
    expect(LIGHTING_SHADER_SHADOW).toContain('Reinhard');
    expect(LIGHTING_SHADER_SHADOW_HDR).not.toContain('Reinhard');
  });
});

describe('ShadowMap.updateFrame（CPU 侧冒烟，无 device）', () => {
  it('随相机远平面自适应分割且不抛错', () => {
    // 仅 updateFrame 使用 context（且不触碰 device）——桩对象足够。
    const shadow = new ShadowMap({} as unknown as Context);
    shadow.enabled = true;
    const cam = makeCamera();
    cam.frustum.near = 100 / ES;   // 100 m
    cam.frustum.far = 8000 / ES;   // 8 km → 覆盖收缩到 8km
    shadow.updateFrame(cam, [
      { direction: [0.5, -0.3, -0.8] } as never,
    ]);
    const ends = shadow.splitEndsMetres;
    expect(ends[0]).toBeGreaterThan(100);
    expect(ends[2]).toBeCloseTo(8000, 0);
    expect(shadow.casterCount).toBe(0);

    const caster = {} as never;
    shadow.addCaster(caster);
    expect(shadow.casterCount).toBe(1);
    expect(shadow.removeCaster(caster)).toBe(true);
    expect(shadow.removeCaster(caster)).toBe(false);
  });

  it('无灯光时不更新不抛错', () => {
    const shadow = new ShadowMap({} as unknown as Context);
    expect(() => shadow.updateFrame(makeCamera(), [])).not.toThrow();
  });
});
