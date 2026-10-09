/**
 * entityDistanceGates.spec.ts – P2-4 打回回归：
 * cylinder / path / polyline 的 distanceDisplayCondition 逐帧距离门控
 * （窗外 prim.show=false / 窗内恢复），以及 pixelOffset 与锚点的组合偏移。
 *
 * GPU-free：Context 用 stub（Proxy device 满足 Primitive 构造的
 * createBuffer/createTexture 调用，不触真实 WebGPU）；图元引用经 host 的
 * addTransparentPrimitive 记录捕获。
 */
import { describe, expect, it, vi } from 'vitest';
import { EntityVisualizer, type EntityVisualizerHost } from '../../src/scene/EntityVisualizer';
import { EntityCollection } from '../../src/dataSource/EntityCollection';
import { Entity } from '../../src/dataSource/Entity';
import { CallbackProperty } from '../../src/dataSource/CallbackProperty';
import {
  DistanceDisplayCondition,
  billboardScreenOffset,
  HorizontalOrigin,
  VerticalOrigin,
} from '../../src/dataSource/BillboardStyle';
import { Context } from '../../src/renderer/Context';
import { Cartesian2 } from '../../src/math/Cartesian2';
import { Cartesian3 } from '../../src/math/Cartesian3';
import type { Primitive } from '../../src/scene/Primitive';

const EARTH = 6378137.0;
const entityAt = new Cartesian3(1, 0, 0);

// node 无 WebGPU 全局：Primitive 构造只读 usage 位标志常量，stub 即可。
vi.stubGlobal('GPUBufferUsage', {
  VERTEX: 0x20, COPY_DST: 0x8, INDEX: 0x10, UNIFORM: 0x40, COPY_SRC: 0x4,
});
vi.stubGlobal('GPUTextureUsage', {
  TEXTURE_BINDING: 0x4, COPY_DST: 0x8, RENDER_ATTACHMENT: 0x10, COPY_SRC: 0x4,
});

/** Proxy device：任意方法调用返回空对象（Primitive 构造足够，无真实 GPU）。 */
function stubContext(): Context {
  const device = new Proxy(
    {},
    {
      get: (_target, prop) => {
        if (prop === 'queue') {
          return {
            writeBuffer: () => undefined,
            copyExternalImageToTexture: () => undefined,
            writeTexture: () => undefined,
          };
        }
        return () => ({ label: String(prop), destroy: () => undefined });
      },
    }
  ) as unknown as GPUDevice;
  return {
    device,
    writeBuffer: () => undefined,
    createSampler: () => ({ label: 'stub-sampler' }),
  } as unknown as Context;
}

/** 可变相机 + 记录 add*Primitive 的 host。 */
function makeVisualizer(collection: EntityCollection, cameraDistanceMetres: number) {
  const added: Primitive[] = [];
  const camera = { position: cameraAt(cameraDistanceMetres), frustum: { fov: 1 } };
  const host: EntityVisualizerHost = {
    primitives: { remove: () => undefined } as unknown as EntityVisualizerHost['primitives'],
    addPrimitive: (p) => added.push(p),
    addTransparentPrimitive: (p) => added.push(p),
    camera,
    viewportHeightPx: 800,
  };
  const visualizer = new EntityVisualizer(stubContext(), collection, host);
  return { visualizer, added, camera };
}

/** 相机与实体位置相距 distanceMetres（同向径向，纯数值可控）。 */
function cameraAt(distanceMetres: number): Cartesian3 {
  return new Cartesian3(1 + distanceMetres / EARTH, 0, 0);
}

describe('distanceDisplayCondition 图元级门控（打回回归）', () => {
  it('polyline：窗外 prim.show=false，相机拉近窗内恢复 true', () => {
    const collection = new EntityCollection();
    collection.add(
      new Entity({
        position: entityAt,
        polyline: {
          positions: [new Cartesian3(1, 0, 0), new Cartesian3(0, 1, 0)],
          distanceDisplayCondition: new DistanceDisplayCondition(1000, 5000),
        },
      })
    );
    const { visualizer, added, camera } = makeVisualizer(collection, 6000);
    visualizer.update(0.1);
    expect(added.length).toBe(1);
    expect(added[0].show).toBe(false); // 6 km > far → 窗外

    visualizer.update(0.2); // 距离未变 → 仍 false
    expect(added[0].show).toBe(false);

    camera.position = cameraAt(2000); // 拉近到窗内
    visualizer.update(0.3);
    expect(added[0].show).toBe(true);
  });

  it('polyline DDC 可为 CallbackProperty（逐帧解析）', () => {
    const collection = new EntityCollection();
    let far = 5000;
    collection.add(
      new Entity({
        position: entityAt,
        polyline: {
          positions: [new Cartesian3(1, 0, 0), new Cartesian3(0, 1, 0)],
          distanceDisplayCondition: new CallbackProperty(
            () => new DistanceDisplayCondition(0, far)
          ),
        },
      })
    );
    const { visualizer, added } = makeVisualizer(collection, 2000);
    visualizer.update(0.1);
    expect(added[0].show).toBe(true);
    far = 1000; // 动态收紧窗口 → 窗外
    visualizer.update(0.2);
    expect(added[0].show).toBe(false);
  });

  it('cylinder：窗外 prim.show=false / 窗内恢复（机头锥场景）', () => {
    const collection = new EntityCollection();
    collection.add(
      new Entity({
        position: entityAt,
        cylinder: {
          length: 100,
          topRadius: 0,
          bottomRadius: 20,
          distanceDisplayCondition: new DistanceDisplayCondition(500, 3000),
        },
      })
    );
    const { visualizer, added, camera } = makeVisualizer(collection, 4000);
    visualizer.update(0.1);
    expect(added.length).toBe(1);
    expect(added[0].show).toBe(false);

    camera.position = cameraAt(1000);
    visualizer.update(0.2);
    expect(added[0].show).toBe(true);
  });

  it('path：窗外 pathPrimitive.show=false / 窗内恢复', () => {
    const collection = new EntityCollection();
    collection.add(
      new Entity({
        // 每次采样移动 ~64 m 的位置：非退化 path 几何 + 相机距离可控。
        position: (() => {
          let step = 0;
          return new CallbackProperty(
            () => new Cartesian3(1 + 1e-5 * ++step, 0, 0)
          );
        })(),
        path: {
          trailTime: 10,
          distanceDisplayCondition: new DistanceDisplayCondition(0, 1000),
        },
      })
    );
    const { visualizer, added, camera } = makeVisualizer(collection, 2000);
    visualizer.update(0.1); // 1 个采样点：仅入史
    visualizer.update(0.2); // 2 个点 → path 几何构建（首帧尚无门控对象）
    visualizer.update(0.3); // 门控对已存在 prim 生效：相机 ~2 km > far → false
    const pathPrim = added[added.length - 1];
    expect(pathPrim).toBeDefined();
    expect(pathPrim.show).toBe(false);

    camera.position = cameraAt(200); // 拉近到窗内（实体 ~1.00003）
    visualizer.update(0.4);
    expect(added[added.length - 1].show).toBe(true);
  });

  it('无 DDC 时 prim.show 不被门控改写（行为不变）', () => {
    const collection = new EntityCollection();
    collection.add(
      new Entity({
        position: entityAt,
        polyline: { positions: [new Cartesian3(1, 0, 0), new Cartesian3(0, 1, 0)] },
      })
    );
    const { visualizer, added } = makeVisualizer(collection, 999_999);
    visualizer.update(0.1);
    expect(added[0].show).toBe(true);
  });
});

describe('pixelOffset 组合偏移（P2-4 附注）', () => {
  it('billboardScreenOffset = 九宫锚点 + 逐像素偏移', () => {
    // hexapod droneController 形态：LEFT/TOP 锚点 + (0, -40) 上移。
    const off = billboardScreenOffset(
      HorizontalOrigin.LEFT,
      VerticalOrigin.TOP,
      100,
      20,
      new Cartesian2(0, -40)
    );
    expect(off.dx).toBe(-50);               // LEFT → -w/2 + 0
    expect(off.dy).toBeCloseTo(10 - 40, 9); // TOP → +h/2，再 -40px 上移

    const c = billboardScreenOffset(
      HorizontalOrigin.CENTER,
      VerticalOrigin.CENTER,
      100,
      20,
      new Cartesian2(6, -8)
    );
    expect(c.dx).toBe(6);
    expect(c.dy).toBe(-8);
  });

  it('pixelOffset 缺省时等价纯锚点偏移', () => {
    const a = billboardScreenOffset(HorizontalOrigin.RIGHT, VerticalOrigin.BOTTOM, 40, 10);
    expect(a.dx).toBe(20);
    expect(a.dy).toBe(-5);
  });

  it('Entity 字段接受 CallbackProperty<Cartesian2> 并可解析', () => {
    const e = new Entity({
      point: {
        pixelSize: 6,
        pixelOffset: new CallbackProperty(() => new Cartesian2(3, -12)),
      },
      label: { text: 'x', pixelOffset: new Callback2Proxy() },
    });
    expect(typeof e.point?.pixelOffset?.getValue).toBe('function');
    expect(typeof e.label?.pixelOffset?.getValue).toBe('function');
    expect(e.label.pixelOffset?.getValue()).toEqual(new Cartesian2(0, -40));
  });
});

/** 固定 (0,-40) 的最小 Property 形状（duck-typed）。 */
class Callback2Proxy {
  getValue(): Cartesian2 {
    return new Cartesian2(0, -40);
  }
}
