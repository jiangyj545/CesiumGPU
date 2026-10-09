/**
 * libSmoke.spec.ts – P1-G 库产物冒烟：
 * dist-lib/ 存在时（npm run build:lib 之后）验证
 * ① ESM 产物可 import 且关键符号齐全（无顶层副作用炸裂）；
 * ② UMD 产物评估后挂载 globalThis.CesiumGPU 且同名符号可用。
 * 未构建时整体 skip（本文件不触发构建，保持 vitest 快速）。
 */
import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import vm from 'node:vm';

const distLib = resolve(dirname(fileURLToPath(import.meta.url)), '../../dist-lib');
const built = existsSync(resolve(distLib, 'cesiumgpu.es.js'));

const SYMBOLS = [
  'Viewer',
  'Cartesian3',
  'SceneTransforms',
  'CallbackProperty',
  'HeadingPitchRange',
  'HeadingPitchRoll',
  'Cesium3DTileFeature',
  'CylinderGeometry',
  'PathHistory',
  'EntityTracker',
  'Transforms',
] as const;

describe.skipIf(!built)('dist-lib 库产物冒烟', () => {
  it('ESM import 拿到全部关键符号', async () => {
    const mod = (await import(resolve(distLib, 'cesiumgpu.es.js'))) as Record<string, unknown>;
    for (const name of SYMBOLS) {
      expect(mod[name], `ESM 缺符号 ${name}`).toBeDefined();
    }
    expect(mod['Viewer']).toBeTypeOf('function');
  });

  it('UMD 评估后挂载 CesiumGPU global 且符号可用', () => {
    const code = readFileSync(resolve(distLib, 'cesiumgpu.umd.js'), 'utf8');
    const sandbox: Record<string, unknown> = {};
    vm.createContext(sandbox);
    vm.runInContext(code, sandbox, { timeout: 5000 });
    const G = sandbox['CesiumGPU'] as Record<string, unknown> | undefined;
    expect(G).toBeDefined();
    for (const name of SYMBOLS) {
      expect(G![name], `UMD 缺符号 ${name}`).toBeDefined();
    }
  });
});
