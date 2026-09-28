import { describe, expect, it } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

/**
 * WGSL `let` 不可变静态守卫（2026-09-28 CloudShader "cannot assign to 'let q'"
 * 真机级联作废黑屏的防回归）：
 *  - mock device 单测永远抓不到 WGSL 解析错误（GPU 对象只在真设备创建，且新版
 *    Chrome 的 createShaderModule 解析错为同步抛——单测桩直接跳过编译）；
 *  - 本测试提取 src/shaders/**.ts 中全部 /* wgsl */ 反引号段，静态扫描段内
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

/** 提取源文件中全部 /* wgsl */ 反引号模板的段内容（跨行）。 */
function extractWgslSegments(source: string): string[] {
  const segs: string[] = [];
  const re = /\/\*\s*wgsl\s*\*\/\s*`([\s\S]*?)`/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(source)) !== null) segs.push(m[1]);
  return segs;
}

/** 单段扫描：let 声明后 30 行内的行首重赋值（= += -= *= /= %=，排除 ==）。 */
function scanSegment(seg: string, segIdx: number, file: string): string[] {
  const hits: string[] = [];
  const lines = seg.split('\n');
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
