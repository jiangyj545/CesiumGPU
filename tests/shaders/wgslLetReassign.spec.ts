import { describe, expect, it } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { OCEAN_FFT_RENDER_SHADER } from '../../src/shaders/OceanFFTShader';
import { ATMO_SKY_RENDER_WGSL } from '../../src/shaders/AtmosphereSkyShader';

/**
 * WGSL `let` 不可变静态守卫（2026-09-28 CloudShader "cannot assign to 'let q'"
 * 真机级联作废黑屏的防回归）：
 *  - mock device 单测永远抓不到 WGSL 解析错误（GPU 对象只在真设备创建，且新版
 *    Chrome 的 createShaderModule 解析错为同步抛——单测桩直接跳过编译）；
 *  - 本测试提取 src/shaders/**.ts 中全部 "wgsl" 标记反引号段，静态扫描段内
 *    `let X = ...` 之后 30 行内出现行首 `X <op>= ...` 重赋值的模式（== 比较
 *    与 TS 模板段天然排除——只扫 WGSL 字符串内部）。
 * 有意保守：宁可漏报（跨作用域 shadow 等复杂情况），不可误报阻塞 CI。
 */

const SHADER_ROOT = join(__dirname, '../../src/shaders');

/** 递归收集 .ts 文件。 */
function collectTs(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) collectTs(p, out);
    else if (name.endsWith('.ts')) out.push(p);
  }
  return out;
}

/** 提取源文件中全部 "wgsl" 标记反引号模板的段内容（跨行）。 */
function extractWgslSegments(source: string): string[] {
  const segs: string[] = [];
  const re = /\/\*\s*wgsl\s*\*\/\s*`([\s\S]*?)`/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(source)) !== null) segs.push(m[1]);
  return segs;
}

/** 剥 WGSL 注释（行注释与块注释；段内无字符串字面量风险，安全近似）。 */
function stripWgslComments(wgsl: string): string {
  return wgsl
    .split('\n')
    .map((l) => l.replace(/\/\/.*$/, ''))
    .join('\n')
    .replace(/\/\*[\s\S]*?\*\//g, '');
}

/** 单段扫描：先剥注释并按顶层函数边界分块，块内 let 被重赋值才算（跨函数
 *  作用域的同名变量是合法 shadow/独立声明，不误报）。 */
function scanSegment(seg: string, segIdx: number, file: string): string[] {
  const hits: string[] = [];
  const cleaned = stripWgslComments(seg);
  // 按顶层 fn / 入口属性分块（@vertex/@fragment/@compute 后跟 fn 同块）
  const blocks = cleaned.split(/(?=^(?:@\w+\s*\n)?fn\s)/m);
  for (const block of blocks) {
    const lines = block.split('\n');
    const letNames = new Map<string, number>();
    lines.forEach((line, i) => {
      const decl = line.match(/^\s*let\s+([a-zA-Z_]\w*)\s*(?::[^=]+)?=/);
      if (decl) letNames.set(decl[1], i);
    });
    letNames.forEach((declIdx, name) => {
      for (let j = declIdx + 1; j < Math.min(declIdx + 30, lines.length); j++) {
        const reassign = new RegExp(
          `^\\s*${name}\\s*(?:\\+=|-=|\\*=|/=|%=)(?!=)|^\\s*${name}\\s*=(?!=)`,
        );
        if (reassign.test(lines[j])) {
          hits.push(
            `${file} [wgsl#${segIdx}] let "${name}"（声明行 ${declIdx + 1}）在行 ${j + 1} 被重赋值：${lines[j].trim().slice(0, 60)}`,
          );
          break;
        }
      }
    });
  }
  return hits;
}

describe('WGSL let 不可变静态守卫', () => {
  it('src/shaders 全部 WGSL 段无 let 重赋值', () => {
    const files = collectTs(SHADER_ROOT);
    expect(files.length).toBeGreaterThan(20); // 采样健全性：shader 文件齐全
    const violations: string[] = [];
    for (const f of files) {
      const segs = extractWgslSegments(readFileSync(f, 'utf8'));
      segs.forEach((seg, i) => violations.push(...scanSegment(seg, i, f.split('src')[1])));
    }
    expect(violations, `WGSL let 重赋值（改为 var）：\n${violations.join('\n')}`).toHaveLength(0);
  });

  it('守卫自检：能捕获 CloudShader 历史坏模式', () => {
    const bad = ['let q = fract(p);', 'q += dot(q, q + 45.32);', 'return fract(q.x * q.y);'].join('\n');
    expect(scanSegment(bad, 0, '(selftest)')).toHaveLength(1);
    const good = ['var q = fract(p);', 'q += dot(q, q + 45.32);'].join('\n');
    expect(scanSegment(good, 0, '(selftest)')).toHaveLength(0);
    const compare = ['let ci = 1;', 'let x = ci == 2;'].join('\n');
    expect(scanSegment(compare, 0, '(selftest)')).toHaveLength(0); // == 比较不误报
  });
});

/**
 * 拼接型导出模块的 fn 调用解析守卫（2026-09-28 ATMO_SKY_REFLECTION_WGSL 漏拼
 * SKY_COMPOSITE 致 skyHash13 unresolved call target 级联作废黑屏的防回归）。
 * 只检查最终拼接产物（消费方真实拿到的 WGSL 串），builtin 白名单外未解析即报。
 */
const WGSL_BUILTINS = new Set([
  // 类型构造
  'vec2', 'vec3', 'vec4', 'vec2f', 'vec3f', 'vec2u', 'vec3u', 'vec2i', 'vec3i', 'vec4f',
  'mat2x2', 'mat3x3', 'mat4x4', 'mat3x3f', 'f32', 'u32', 'i32', 'bool', 'array',
  // 数学
  'dot', 'cross', 'distance', 'length', 'normalize', 'reflect', 'refract', 'faceForward',
  'abs', 'sign', 'clamp', 'saturate', 'min', 'max', 'mix', 'step', 'smoothstep', 'fma',
  'floor', 'ceil', 'round', 'trunc', 'fract', 'modf', 'lerp', 'pow', 'exp', 'exp2', 'log', 'log2',
  'sqrt', 'inverseSqrt', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh',
  'select', 'unpack4x8snorm',
  // 纹理/采样
  'textureSample', 'textureSampleLevel', 'textureSampleCompare', 'textureSampleCompareLevel',
  'textureLoad', 'textureStore', 'textureDimensions', 'textureNumLevels', 'textureGather',
  // 原子/位/派生
  'atomicLoad', 'atomicStore', 'atomicAdd', 'atomicMax', 'atomicMin', 'atomicExchange',
  'dpdx', 'dpdy', 'fwidth', 'dpdxCoarse', 'dpdyCoarse',
  // 构造/转换辅助
  'bitcast', 'countOneBits', 'reverseBits', 'all', 'any', 'transpose', 'determinant',
]);

/** 扫描最终 WGSL 模块：调用了但未定义且非 builtin 的标识符（先剥注释——
 *  注释文本 "high (cells..." 会被误当调用）。 */
function scanUnresolvedCalls(wgsl: string, label: string): string[] {
  const cleaned = stripWgslComments(wgsl);
  const defined = new Set<string>();
  for (const m of cleaned.matchAll(/\bfn\s+([a-zA-Z_]\w*)\s*[<(]/g)) defined.add(m[1]);
  const called = new Set<string>();
  for (const m of cleaned.matchAll(/(?<![.\w@$])([a-zA-Z_]\w*)\s*\(/g)) {
    const name = m[1];
    if (!['if', 'for', 'while', 'switch', 'return', 'discard', 'let', 'var', 'const', 'fn', 'struct', 'loop', 'break', 'continue', 'else', 'defer'].includes(name)) {
      called.add(name);
    }
  }
  const unresolved: string[] = [];
  called.forEach((name) => {
    if (!defined.has(name) && !WGSL_BUILTINS.has(name)) {
      unresolved.push(`${label}: 调用 "${name}(" 无 fn 定义且非 builtin`);
    }
  });
  return unresolved;
}

describe('WGSL 拼接模块 fn 调用解析守卫', () => {
  it('OCEAN_FFT_RENDER_SHADER（含 ATMO 反射块拼接产物）无未解析调用', () => {
    const hits = scanUnresolvedCalls(OCEAN_FFT_RENDER_SHADER, 'OCEAN_FFT_RENDER_SHADER');
    expect(hits, hits.join('\n')).toHaveLength(0);
  });

  it('ATMO_SKY_RENDER_WGSL（天空背景渲染）无未解析调用', () => {
    const hits = scanUnresolvedCalls(ATMO_SKY_RENDER_WGSL, 'ATMO_SKY_RENDER_WGSL');
    expect(hits, hits.join('\n')).toHaveLength(0);
  });

  it('守卫自检：能捕获漏拼 skyHash13 的历史坏模式', () => {
    const bad = 'fn skyStars() { let h = skyHash13(vec3f(1.0)); }';
    expect(scanUnresolvedCalls(bad, '(selftest)')).toHaveLength(1);
    const good = 'fn skyHash13(p: vec3f) -> f32 { return fract(p.x); } fn skyStars() { let h = skyHash13(vec3f(1.0)); }';
    expect(scanUnresolvedCalls(good, '(selftest)')).toHaveLength(0);
  });
});
