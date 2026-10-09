import { describe, expect, it } from 'vitest';
import { RenderScheduler } from '../../src/core/RenderScheduler';
import { EntityVisualizer, type EntityVisualizerHost } from '../../src/scene/EntityVisualizer';
import { EntityCollection } from '../../src/dataSource/EntityCollection';
import { Entity } from '../../src/dataSource/Entity';
import { CallbackProperty } from '../../src/dataSource/CallbackProperty';
import { Context } from '../../src/renderer/Context';
import { Cartesian3 } from '../../src/math/Cartesian3';

describe('RenderScheduler（P2-5 按需渲染）', () => {
  it('无 dirty 连续 tick 只渲染第一帧', () => {
    const s = new RenderScheduler(Number.POSITIVE_INFINITY, 0);
    expect(s.shouldRender(0.0)).toBe(true);  // 初始脏
    expect(s.shouldRender(0.016)).toBe(false);
    expect(s.shouldRender(0.032)).toBe(false);
    expect(s.shouldRender(1.0)).toBe(false);
  });

  it('requestRender() 显式触发一帧，消费后回静', () => {
    const s = new RenderScheduler();
    s.shouldRender(0);
    s.requestRender();
    expect(s.shouldRender(0.016)).toBe(true);
    expect(s.shouldRender(0.032)).toBe(false);
  });

  it('markDirty()（指纹变化/实体增删）恢复渲染一帧', () => {
    const s = new RenderScheduler();
    s.shouldRender(0);
    s.markDirty();
    expect(s.shouldRender(0.016)).toBe(true);
    expect(s.shouldRender(0.032)).toBe(false);
  });

  it('maximumRenderTimeChange：静止超过阈值自动补渲一帧', () => {
    const s = new RenderScheduler(2.0, 0);
    expect(s.shouldRender(0)).toBe(true);
    expect(s.shouldRender(1.0)).toBe(false);
    expect(s.shouldRender(1.99)).toBe(false);
    expect(s.shouldRender(2.0)).toBe(true);  // 达到阈值
    expect(s.shouldRender(2.5)).toBe(false); // 补渲后重新计时
    expect(s.shouldRender(3.99)).toBe(false);
    expect(s.shouldRender(4.0)).toBe(true);
  });

  it('默认 Infinity：永不自动补渲（hexapod 语义）', () => {
    const s = new RenderScheduler();
    s.shouldRender(0);
    expect(s.shouldRender(1e6)).toBe(false);
  });
});

describe('EntityVisualizer 脏标记契约（P2-5 脏源）', () => {
  const makeVisualizer = (collection: EntityCollection): EntityVisualizer => {
    const host = {
      primitives: { remove: () => undefined },
      addPrimitive: () => undefined,
      addTransparentPrimitive: () => undefined,
      camera: { position: new Cartesian3(0, 0, 2), frustum: { fov: 1 } },
      viewportHeightPx: 800,
    } as unknown as EntityVisualizerHost;
    return new EntityVisualizer(new Context({} as HTMLCanvasElement), collection, host);
  };

  it('实体加入（collectionChanged/_track）→ 脏；静态无图形实体 update 后回静', () => {
    const collection = new EntityCollection();
    collection.add(new Entity({ position: new Cartesian3(1, 0, 0) }));
    const v = makeVisualizer(collection);
    expect(v.consumeChange()).toBe(true); // 初始构建即变
    v.update(0.1);
    expect(v.consumeChange()).toBe(false); // 静态实体无变化
  });

  it('CallbackProperty 动态实体：每次 update 持续置脏（不误判静止）', () => {
    const collection = new EntityCollection();
    collection.add(
      new Entity({
        position: new CallbackProperty(() => new Cartesian3(1, 0, 0)),
        point: { pixelSize: 6 }, // OverlayPass 每帧消费（visualizer 无突变点）
      })
    );
    const context = new Context({} as HTMLCanvasElement);
    const host = {
      primitives: { remove: () => undefined },
      addPrimitive: () => undefined,
      addTransparentPrimitive: () => undefined,
      camera: { position: new Cartesian3(0, 0, 2), frustum: { fov: 1 } },
      viewportHeightPx: 800,
    } as unknown as EntityVisualizerHost;
    const v = new EntityVisualizer(context, collection, host);
    v.update(0.1);
    expect(v.consumeChange()).toBe(true);
    v.update(0.2);
    expect(v.consumeChange()).toBe(true); // 逐帧
    v.update(0.3);
    expect(v.consumeChange()).toBe(true);
  });

  it('definitionChanged → _rebuild：update 窗口外的变更不被吞掉', () => {
    const collection = new EntityCollection();
    const e = new Entity({ position: new Cartesian3(1, 0, 0) });
    collection.add(e);
    const context = new Context({} as HTMLCanvasElement);
    const host = {
      primitives: { remove: () => undefined },
      addPrimitive: () => undefined,
      addTransparentPrimitive: () => undefined,
      camera: { position: new Cartesian3(0, 0, 2), frustum: { fov: 1 } },
      viewportHeightPx: 800,
    } as unknown as EntityVisualizerHost;
    const v = new EntityVisualizer(context, collection, host);
    v.update(0.1);
    v.consumeChange(); // 清零
    e.show = false; // definitionChanged → _rebuild（无图形，不触 GPU）
    expect(v.consumeChange()).toBe(true);
  });
});
