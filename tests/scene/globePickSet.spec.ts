import { describe, expect, it } from 'vitest';
import { Globe } from '../../src/scene/Globe';
import type { Primitive } from '../../src/scene/Primitive';

/**
 * Globe.tilePrimitives getter 必须与 drawTiles 的绘制集合同源：
 *
 * 背景 bug：`_terrainPrimitives.size > 0` 时 drawTiles 只画地形 primitive 并
 * 提前 return（影像瓦片不画），但 getter 只返回 `_tilePrimitives`、不含地形。
 * PickPass 用该 getter 做 GPU 拾取重绘 → 地形激活时 G-Buffer 渲染的是地形
 * 网格，拾取 pass 画的却是平面影像瓦片，拾取命中位置系统性偏差。
 *
 * node 环境无 WebGPU：mock Primitive 最小表面（只暴露 drawOpaque），
 * mock pass encoder 记录每个 primitive 的 draw 调用，验证纯集合逻辑。
 */

/** 最小 Primitive 替身：drawOpaque 往 pass 上记一条自己的标签。 */
function mockPrimitive(label: string): Primitive {
  return {
    label,
    drawOpaque(pass: GPURenderPassEncoder) {
      (pass as unknown as { draw(l: string): void }).draw(label);
    },
  } as unknown as Primitive;
}

/** 记录 draw 调用标签序列的 pass encoder 替身。 */
function makePassRecorder() {
  const drawn: string[] = [];
  const pass = {
    draw(label: string) {
      drawn.push(label);
    },
  } as unknown as GPURenderPassEncoder;
  return { drawn, pass };
}

/** 注入 mock 条目到 private Map（仓库先例：globeLighting.spec.ts 同手法）。 */
function inject(globe: Globe, parts: { imagery?: string[]; terrain?: string[] }) {
  const g = globe as unknown as Record<string, any>;
  for (const label of parts.imagery ?? []) {
    g._tilePrimitives.set(label, { primitive: mockPrimitive(label), texture: {} });
  }
  for (const label of parts.terrain ?? []) {
    g._terrainPrimitives.set(label, { primitive: mockPrimitive(label), texture: {} });
  }
}

/** getter 返回集的标签视图（mock primitive 自带 label 便于比对）。 */
function labelsOf(primitives: Primitive[]): string[] {
  return primitives.map((p) => (p as unknown as { label: string }).label);
}

describe('Globe.tilePrimitives 与 drawTiles 绘制集合同源', () => {
  it('地形 Map 空 + 影像 Map 有条目 → getter 返回影像条目', () => {
    // 字段初始化不触 GPU，context 传最小替身即可
    const globe = new Globe({ device: {} } as never);
    inject(globe, { imagery: ['1/0/0', '1/1/0'] });

    expect(labelsOf(globe.tilePrimitives)).toEqual(['1/0/0', '1/1/0']);
  });

  it('地形 Map 有条目 → getter 只返回地形条目（不含影像）', () => {
    const globe = new Globe({ device: {} } as never);
    inject(globe, { imagery: ['1/0/0'], terrain: ['3/4/2', '3/5/2'] });

    expect(labelsOf(globe.tilePrimitives)).toEqual(['3/4/2', '3/5/2']);
  });

  it('drawTiles 的绘制集合与 getter 集合一致（地形激活）', () => {
    const globe = new Globe({ device: {} } as never);
    inject(globe, { imagery: ['1/0/0'], terrain: ['3/4/2'] });

    const { drawn, pass } = makePassRecorder();
    globe.drawTiles(pass, {} as GPUBindGroup);

    // 互斥分支：只画地形，影像瓦片不画（与 getter 同源）
    expect(drawn).toEqual(['3/4/2']);
    expect(drawn).toEqual(labelsOf(globe.tilePrimitives));
  });

  it('drawTiles 的绘制集合与 getter 集合一致（地形为空）', () => {
    const globe = new Globe({ device: {} } as never);
    inject(globe, { imagery: ['1/0/0', '2/1/1'] });

    const { drawn, pass } = makePassRecorder();
    globe.drawTiles(pass, {} as GPUBindGroup);

    expect(drawn).toEqual(['1/0/0', '2/1/1']);
    expect(drawn).toEqual(labelsOf(globe.tilePrimitives));
  });
});
