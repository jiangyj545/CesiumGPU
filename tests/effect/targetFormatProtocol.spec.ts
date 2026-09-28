import { describe, expect, it } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

/**
 * TargetFormatAware 协议守卫（2026-09-28 BreakSpray attachment state 校验错
 * 黑屏的防回归）：凡 src/effect/**.ts 中创建 GPURenderPipeline 的特效类，
 * 必须实现 `setTargetFormat(format)`——否则 postProcess.enabled=true（真 HDR，
 * 场景 RT 为 rgba16float）页面上特效管线仍按 preferredFormat（bgra8unorm）
 * 创建，SetPipeline 触发 attachment state 不兼容、整帧级联作废。
 * 静态字符串级检查（duck-typing 协议没有 TS 接口约束，运行时才发现——与
 * EffectManager 的 `(effect as ...).setTargetFormat?.(...)` 调用点约定一致）。
 */

const EFFECT_ROOT = join(__dirname, '../../src/effect');

/**
 * 豁免清单：渲染目标是**自持离屏纹理**（非 Scene 下发的场景 RT）的特效——
 * 管线格式与 targetFormat 无关（execute 忽略场景 target），不接协议是正确行为。
 * 加新豁免必须在对应源文件注释里写明"自持离屏目标"理由（OceanCaustics L70 同款）。
 */
const SELF_OWNED_TARGET_EXEMPT = new Set(['OceanCaustics.ts']);

function collectTs(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) collectTs(p, out);
    else if (name.endsWith('.ts') && !name.endsWith('.spec.ts')) out.push(p);
  }
  return out;
}

describe('TargetFormatAware 协议守卫', () => {
  it('所有创建 GPURenderPipeline 的特效都实现 setTargetFormat', () => {
    const violations: string[] = [];
    const files = collectTs(EFFECT_ROOT);
    expect(files.length).toBeGreaterThan(15); // 健全性
    for (const f of files) {
      const src = readFileSync(f, 'utf8');
      const hasPipeline = /createRenderPipeline\s*\(/.test(src);
      const hasSetter = /setTargetFormat\s*\(\s*\w+\s*:\s*GPUTextureFormat\s*\)/.test(src);
      if (hasPipeline && !hasSetter && !SELF_OWNED_TARGET_EXEMPT.has(f.split(/[\\/]/).pop()!)) {
        violations.push(`${f.split('src')[1]}：有 createRenderPipeline 但未实现 setTargetFormat`);
      }
    }
    expect(violations, `漏接 TargetFormatAware 协议的特效：\n${violations.join('\n')}`).toHaveLength(0);
  });

  it('守卫自检：能区分有/无 setTargetFormat', () => {
    const ok = 'createRenderPipeline({}); setTargetFormat(format: GPUTextureFormat) {}';
    expect(/createRenderPipeline\s*\(/.test(ok) && /setTargetFormat\s*\(\s*\w+\s*:\s*GPUTextureFormat\s*\)/.test(ok)).toBe(true);
    const bad = 'createRenderPipeline({});';
    expect(/setTargetFormat\s*\(\s*\w+\s*:\s*GPUTextureFormat\s*\)/.test(bad)).toBe(false);
  });
});
