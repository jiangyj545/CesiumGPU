/**
 * polylineScreenSpaceWidth.spec.ts – 引擎级修复回归：
 * Entity polyline/path 屏幕空间恒定像素宽（CesiumJS 语义）。
 *
 * 修复前：width(像素) × PIXELS_TO_WORLD_WIDTH 固定换算世界宽 ribbon，
 * 远视距（320 km ~ 9.5 Mm）屏上厚度 0.05–0.7 px 不可见。
 * 修复后：PolylineGeometry `screenSpaceWidth` 模式输出中心线 + 侧向向量，
 * TransparentShader 顶点着色器投影后按 halfWidthDevicePx 展开 —— 任意
 * 相机距离下线宽恒为 width CSS 像素。
 *
 * GPU-free：Context 用记录型桩（捕获 MaterialUBO 写入 / 管线描述符），
 * 不触真实 WebGPU。
 */
import { describe, expect, it, vi } from 'vitest';
import { EntityVisualizer, type EntityVisualizerHost } from '../../src/scene/EntityVisualizer';
import { EntityCollection } from '../../src/dataSource/EntityCollection';
import { Entity } from '../../src/dataSource/Entity';
import { Context } from '../../src/renderer/Context';
import { Cartesian3 } from '../../src/math/Cartesian3';
import { Color } from '../../src/math/Color';
import type { InterleavedGeometryBuffers } from '../../src/geometry/GeometryPipeline';
import { TransparentPass } from '../../src/scene/passes/TransparentPass';
import { TRANSPARENT_ACCUMULATION_SHADER } from '../../src/shaders/TransparentShader';
import type { Primitive } from '../../src/scene/Primitive';
import { PolylineGeometry } from '../../src/geometry/PolylineGeometry';

const EARTH = 6378137.0;

// node 无 WebGPU 全局：构造只读位标志常量，stub 即可。
vi.stubGlobal('GPUBufferUsage', {
  VERTEX: 0x20, COPY_DST: 0x8, INDEX: 0x10, UNIFORM: 0x40, COPY_SRC: 0x4,
});
vi.stubGlobal('GPUTextureUsage', {
  TEXTURE_BINDING: 0x4, COPY_DST: 0x8, RENDER_ATTACHMENT: 0x10, COPY_SRC: 0x4,
});
vi.stubGlobal('GPUColorWrite', { RED: 4 });
vi.stubGlobal('GPUShaderStage', { VERTEX: 1, FRAGMENT: 2 });

interface UboWrite { label: string; data: Float32Array }

/**
 * 记录型桩 Context：按 label 捕获 writeBuffer 快照（Primitive 经
 * context.writeBuffer 下发 MaterialUBO），createBuffer 返回带 label 对象。
 */
function recordingContext(drawingBufferHeight = 1600): { context: Context; writes: UboWrite[] } {
  const writes: UboWrite[] = [];
  const device = {
    createBuffer: (d: GPUBufferDescriptor) => ({ label: d.label ?? '' }),
    createSampler: () => ({}),
    createTexture: () => ({ createView: () => ({}) }),
    createShaderModule: () => ({}),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createBindGroup: () => ({}),
    createRenderPipeline: (d: { label?: string }) => ({
      label: d.label,
      getBindGroupLayout: () => ({}),
    }),
    queue: { writeTexture: () => undefined },
  };
  const context = {
    device,
    drawingBufferWidth: drawingBufferHeight,
    drawingBufferHeight,
    writeBuffer: (buffer: { label?: string }, _offset: number, data: Float32Array) => {
      const copy = new Float32Array(data.length);
      copy.set(data);
      writes.push({ label: buffer?.label ?? '', data: copy });
    },
    createSampler: () => ({}),
  } as unknown as Context;
  return { context, writes };
}

/** 记录 add*Primitive 的 host（viewportHeightPx=800 CSS px → 与 1600 设备 px 组成 dpr=2）。 */
function makeVisualizer(context: Context, collection: EntityCollection) {
  const added: Primitive[] = [];
  const host: EntityVisualizerHost = {
    primitives: { remove: () => undefined } as unknown as EntityVisualizerHost['primitives'],
    addPrimitive: (p) => added.push(p),
    addTransparentPrimitive: (p) => added.push(p),
    camera: { position: new Cartesian3(2, 0, 0), frustum: { fov: 1 } },
    viewportHeightPx: 800,
  };
  return { visualizer: new EntityVisualizer(context, collection, host), added };
}

const materialWrites = (writes: UboWrite[]) =>
  writes.filter((w) => w.label === 'Primitive.MaterialUBO');

const fromDegrees = (lon: number, lat: number): Cartesian3 => {
  const rad = Math.PI / 180;
  const lonR = lon * rad, latR = lat * rad;
  const cos = Math.cos(latR);
  return new Cartesian3(cos * Math.cos(lonR), cos * Math.sin(lonR), Math.sin(latR));
};

// ── 1. PolylineGeometry screenSpaceWidth 模式 ───────────────────────────────

describe('PolylineGeometry screenSpaceWidth 模式（中心线 + 侧向向量）', () => {
  const line = [fromDegrees(100, 30), fromDegrees(104, 34), fromDegrees(108, 31)];

  const deinterleave = (buffers: InterleavedGeometryBuffers) => {
    const v = buffers.vertices;
    const pos: Cartesian3[] = [];
    const nrm: Cartesian3[] = [];
    const uv: [number, number][] = [];
    for (let i = 0; i < buffers.vertexCount; i++) {
      const o = i * 8;
      pos.push(new Cartesian3(v[o], v[o + 1], v[o + 2]));
      nrm.push(new Cartesian3(v[o + 3], v[o + 4], v[o + 5]));
      uv.push([v[o + 6], v[o + 7]]);
    }
    return { pos, nrm, uv };
  };

  it('position 落在中心线上（L/R 对顶点重合，= 输入点），v ∈ {0,1}', () => {
    const { pos, uv } = deinterleave(
      PolylineGeometry.fromCartesianPositions(line, { screenSpaceWidth: true })
    );
    // 每段 4 顶点 [L0,R0,L1,R1]：L 与 R 同一中心线点。
    for (let seg = 0; seg < line.length - 1; seg++) {
      const [l0, r0, l1, r1] = [pos[seg * 4], pos[seg * 4 + 1], pos[seg * 4 + 2], pos[seg * 4 + 3]];
      expect(Cartesian3.distance(l0, r0)).toBeLessThan(1e-6);
      expect(Cartesian3.distance(l1, r1)).toBeLessThan(1e-6);
      // 中心线 = 输入折线点（float32 顶点缓冲 ~1e-7 舍入）。
      expect(Cartesian3.distance(l0, line[seg])).toBeLessThan(1e-6);
      expect(Cartesian3.distance(l1, line[seg + 1])).toBeLessThan(1e-6);
    }
    expect(uv.every(([ , v]) => v === 0 || v === 1)).toBe(true);
  });

  it('normal 槽 = 单位侧向向量（⊥ 径向、⊥ 段切向），与世界模式（径向）不同', () => {
    const screen = deinterleave(
      PolylineGeometry.fromCartesianPositions(line, { screenSpaceWidth: true })
    );
    const world = deinterleave(
      PolylineGeometry.fromCartesianPositions(line, { width: 0.01 })
    );
    for (let seg = 0; seg < line.length - 1; seg++) {
      const tangent = new Cartesian3(
        line[seg + 1].x - line[seg].x,
        line[seg + 1].y - line[seg].y,
        line[seg + 1].z - line[seg].z
      );
      for (const vi of [seg * 4, seg * 4 + 2]) {
        const n = screen.nrm[vi];
        expect(Math.hypot(n.x, n.y, n.z)).toBeCloseTo(1, 5);
        // ⊥ 径向（radial = normalize(position)）。
        const p = screen.pos[vi];
        const radialDot = n.x * p.x + n.y * p.y + n.z * p.z;
        expect(Math.abs(radialDot)).toBeLessThan(1e-6);
        // ⊥ 段切向 —— 仅折线端点（端点切向=单段弦向；内部顶点切向
        // 取邻段平均（miter 语义），与单段弦向有夹角，不作弦向断言）。
        const isPolylineEnd = vi === 0 || vi === (line.length - 2) * 4 + 2;
        if (isPolylineEnd) {
          const tDot = n.x * tangent.x + n.y * tangent.y + n.z * tangent.z;
          expect(Math.abs(tDot)).toBeLessThan(1e-6);
        }
        // 世界模式 normal 是径向（回归防护：两模式语义确实分离）。
        const wn = world.nrm[vi];
        expect(Math.hypot(wn.x - p.x, wn.y - p.y, wn.z - p.z)).toBeLessThan(1e-6);
      }
    }
  });

  it('宽度不再烘进几何：不同 width 的 screenSpaceWidth buffer 逐字节相同（相机距离无关的前提）', () => {
    const a = PolylineGeometry.fromCartesianPositions(line, { screenSpaceWidth: true, width: 1 });
    const b = PolylineGeometry.fromCartesianPositions(line, { screenSpaceWidth: true, width: 10 });
    expect(Array.from(a.vertices)).toEqual(Array.from(b.vertices));

    // 对照：世界模式宽度确实烘进几何（证明语义分离不是巧合）。
    const w1 = PolylineGeometry.fromCartesianPositions(line, { width: 1e-5 });
    const w2 = PolylineGeometry.fromCartesianPositions(line, { width: 2e-5 });
    expect(Array.from(w1.vertices)).not.toEqual(Array.from(w2.vertices));
  });
});

// ── 2. Primitive / EntityVisualizer uniform 链路 ─────────────────────────────

describe('EntityVisualizer → Primitive MaterialUBO（halfWidthDevicePx 单位）', () => {
  it('polyline width=5、dpr=2 → MaterialUBO[6] = 5/2×2 = 5 设备像素半宽', () => {
    const collection = new EntityCollection();
    collection.add(new Entity({
      position: new Cartesian3(1, 0, 0),
      polyline: {
        positions: [fromDegrees(100, 30), fromDegrees(104, 34)],
        width: 5,
      },
    }));
    const { context, writes } = recordingContext(1600); // 设备 1600 / CSS 800 → dpr 2
    const { visualizer } = makeVisualizer(context, collection);
    visualizer.update(0.1);

    const mw = materialWrites(writes);
    expect(mw.length).toBeGreaterThan(0);
    const last = mw[mw.length - 1].data;
    expect(last[6]).toBe(5);          // (5 CSS px / 2) × dpr 2
    expect(last[0]).toBeCloseTo(1, 6); // baseColor 不受影响（默认 WHITE）
  });

  it('path width=3、dpr=2 → 轨迹 prim MaterialUBO[6] = 3（3/2×2）', () => {
    const collection = new EntityCollection();
    const entity = new Entity({
      position: new Cartesian3(1, 0, 0),
      path: { width: 3 },
    });
    collection.add(entity);
    const { context, writes } = recordingContext(1600);
    const { visualizer } = makeVisualizer(context, collection);

    visualizer.update(0.1); // 首帧：轨迹仅 1 点，无 prim
    entity.position = new Cartesian3(0.99, 0.1, 0);
    visualizer.update(20.0); // 第二点 → 重建轨迹 ribbon

    const mw = materialWrites(writes);
    expect(mw.length).toBeGreaterThan(0);
    expect(mw[mw.length - 1].data[6]).toBe(3);
  });

  it('普通多边形图元不受影响：MaterialUBO[6] = 0（世界空间网格路径）', () => {
    const collection = new EntityCollection();
    collection.add(new Entity({
      position: new Cartesian3(1, 0, 0),
      polygon: {
        hierarchy: {
          positions: [fromDegrees(100, 30), fromDegrees(101, 30), fromDegrees(100, 31)],
        },
        color: new Color(1, 0, 0, 0.5),
      },
    }));
    const { context, writes } = recordingContext(1600);
    const { visualizer } = makeVisualizer(context, collection);
    visualizer.update(0.1);

    const mw = materialWrites(writes);
    expect(mw.length).toBeGreaterThan(0);
    expect(mw[mw.length - 1].data[6]).toBe(0);
  });
});

// ── 3. 着色器 / 管线契约 ────────────────────────────────────────────────────

describe('TransparentShader 像素展开契约（uniform 存在 + 单位正确）', () => {
  it('MaterialUniforms 声明 screenHalfWidthPx，顶点展开读取 camera.viewportSize 并除以 w（透视正确）', () => {
    expect(TRANSPARENT_ACCUMULATION_SHADER).toContain('screenHalfWidthPx : f32');
    // 展开 branch 门控在该 uniform 上。
    expect(TRANSPARENT_ACCUMULATION_SHADER).toContain('material.screenHalfWidthPx > 0.0');
    // 设备像素 viewport（与 camera UBO 的 viewportSize = drawingBuffer 尺寸同单位）。
    expect(TRANSPARENT_ACCUMULATION_SHADER).toContain('camera.viewportSize * 0.5');
    // 透视正确：侧向投影除以本顶点 w 后再乘 viewport。
    expect(TRANSPARENT_ACCUMULATION_SHADER).toContain('sideClip / w * halfVp');
    // 展开回 clip：offPx / halfVp * w。
    expect(TRANSPARENT_ACCUMULATION_SHADER).toContain('offPx / halfVp * w');
  });

  it('展开公式在 2e5 m 与 5e6 m 视距下屏上半宽恒等于 halfWidthDevicePx（远视距回归）', () => {
    // WGSL 公式的 JS 转录：deviceOffset = (offPx / halfVp × w) / w × halfVp。
    const halfVp: [number, number] = [960, 540]; // 1920×1080 设备像素
    const halfWidthPx = 5;                        // width 5 CSS px × dpr 2
    const deviceHalfWidth = (sideClip: [number, number], w: number): number => {
      const dirPx: [number, number] = [
        (sideClip[0] / w) * halfVp[0],
        (sideClip[1] / w) * halfVp[1],
      ];
      const dl = Math.max(Math.hypot(dirPx[0], dirPx[1]), 1e-6);
      const offPx: [number, number] = [
        (dirPx[0] / dl) * halfWidthPx,
        (dirPx[1] / dl) * halfWidthPx,
      ];
      const offClip: [number, number] = [
        (offPx[0] / halfVp[0]) * w,
        (offPx[1] / halfVp[1]) * w,
      ];
      return Math.hypot((offClip[0] / w) * halfVp[0], (offClip[1] / w) * halfVp[1]);
    };

    for (const distanceMetres of [2e5, 5e6]) {
      const w = distanceMetres / EARTH; // 归一化视距 ≈ 该顶点的 clip.w
      expect(deviceHalfWidth([0.3, 0.9], w)).toBeCloseTo(halfWidthPx, 6);
      expect(deviceHalfWidth([-1.2, 0.4], w)).toBeCloseTo(halfWidthPx, 6);
    }

    // 对照（修复前的根因）：固定世界宽 ribbon（width×1e-5 ≈ width×63.78 m）
    // 的屏上厚度随距离缩小 25×（5e6/2e5）——远视距亚像素不可见。
    const fovY = 1.0;
    const bakedPx = (px: number, distanceMetres: number): number => {
      const worldMetres = px * 1e-5 * EARTH;
      return (worldMetres / (2 * distanceMetres * Math.tan(fovY / 2))) * 800;
    };
    expect(bakedPx(5, 2e5) / bakedPx(5, 5e6)).toBeCloseTo(5e6 / 2e5, 6);
    expect(bakedPx(5, 5e6)).toBeLessThan(0.5); // 远视距下旧方案 < 0.5 px
  });
});

describe('TransparentPass 累积管线（ribbon 单面几何不剔除）', () => {
  it('accum 管线 cullMode none（掠射/反向视角整条不消失）', () => {
    const pipelines: { label?: string; primitive?: { cullMode?: string } }[] = [];
    const device = {
      createBuffer: (d: GPUBufferDescriptor) => ({ label: d.label ?? '' }),
      createSampler: () => ({}),
      createTexture: () => ({ createView: () => ({}) }),
      createShaderModule: () => ({}),
      createBindGroupLayout: () => ({}),
      createPipelineLayout: () => ({}),
      createBindGroup: () => ({}),
      createRenderPipeline: (d: { label?: string; primitive?: { cullMode?: string } }) => {
        pipelines.push({ label: d.label, primitive: d.primitive });
        return { label: d.label, getBindGroupLayout: () => ({}) };
      },
    };
    const context = {
      device,
      writeBuffer: () => undefined,
      createSampler: () => ({}),
    } as unknown as Context;

    const pass = new TransparentPass(
      context, {} as never, { transparent: [] } as never, 8, 8, 'rgba16float'
    );
    pass.prepare();

    const accum = pipelines.filter((p) => p.label?.startsWith('TransparentAccumPipeline'));
    expect(accum.length).toBe(2); // 常规 + always（穿深）
    for (const p of accum) expect(p.primitive?.cullMode).toBe('none');
  });
});
