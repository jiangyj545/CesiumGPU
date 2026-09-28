import { describe, expect, it, vi } from 'vitest';
import { SwashField } from '../../src/effect/SwashField';
import { SeabedHeightField } from '../../src/effect/SeabedHeightField';
import { CausticsFloor } from '../../src/effect/CausticsFloor';
import { FLOOR_PARAMS_UBO_BYTES } from '../../src/shaders/OceanCausticsShader';
import { buildSwashShader, SWASH_PARAMS_BYTES } from '../../src/shaders/SwashComputeShader';
import { Camera } from '../../src/scene/Camera';
import { Cartesian3 } from '../../src/math/Cartesian3';
import type { Context } from '../../src/renderer/Context';
import type { FFTDataSource, OceanCaustics } from '../../src/effect/OceanCaustics';

/**
 * SwashField 1D Saint-Venant 浅水场 + CausticsFloor.bindSwash 集成 —
 * 纯 CPU / 字符串验证（记录型桩 device，无真实 WebGPU，
 * causticsFloorWetSand.spec 同款模式）：
 * ping-pong 交替、CFL 子步拆分、reset 首帧、参数夹取、bindSeabed 幂等、
 * 源串物理公式、bindSwash 后 UBO P[32..35]/binding 6、未绑定零回归。
 */

const CENTER: [number, number] = [121.5, 31.0];

// node 环境无 WebGPU 运行时全局（causticsFloorWetSand.spec 同款兜底 + compute 扩展）
const g = globalThis as unknown as Record<string, unknown>;
g.GPUTextureUsage ??= { TEXTURE_BINDING: 0x08, COPY_DST: 0x04, STORAGE_BINDING: 0x10, RENDER_ATTACHMENT: 0x20 };
g.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
g.GPUBufferUsage ??= { MAP_READ: 0x01, VERTEX: 0x20, INDEX: 0x10, UNIFORM: 0x40, COPY_DST: 0x08 };

/** 桩纹理视图（反查所属纹理，bind group 绑定断言用）。 */
interface StubView {
  label: string;
  tex: StubTexture;
}
/** 桩纹理。 */
interface StubTexture {
  label: string;
  format: string;
  usage: number;
  destroyed: boolean;
  destroy: ReturnType<typeof vi.fn>;
  createView: (desc?: { label?: string }) => StubView;
}
/** 桩 buffer。 */
interface StubBuffer {
  label: string;
  size: number;
  destroyed: boolean;
  destroy: ReturnType<typeof vi.fn>;
}
/** 桩 bind group（捕获 entries 供绑定断言）。 */
interface StubBindGroup {
  label: string;
  entries: { binding: number; resource: unknown }[];
}
/** 桩 compute/render pass（捕获 dispatch 尺寸序列）。 */
interface StubPass {
  setPipeline: ReturnType<typeof vi.fn>;
  setBindGroup: ReturnType<typeof vi.fn>;
  dispatches: [number, number][];
  drawIndexed: ReturnType<typeof vi.fn>;
  end: ReturnType<typeof vi.fn>;
}

/**
 * 记录型桩 Context：捕获 createTexture / createBuffer / createBindGroup /
 * writeBuffer / compute dispatch，其余返回轻量桩对象。
 */
function makeContext() {
  const textures: StubTexture[] = [];
  const buffers: StubBuffer[] = [];
  const bindGroups: StubBindGroup[] = [];
  const passes: StubPass[] = [];
  const writeTexture = vi.fn();
  /**
   * writeBuffer 捕获（数据即时快照）：SwashField 打包复用同一 Float32Array，
   * 引用在后续子步会被改写——此处按调用时刻拷贝，测试读到历史状态。
   */
  const written: { buf: unknown; off: number; data: Float32Array }[] = [];
  const writeBuffer = vi.fn((buf: unknown, off: number, data: ArrayBufferView) => {
    const src = data as Float32Array;
    const copy = new Float32Array(src.length);
    copy.set(src);
    written.push({ buf, off, data: copy });
  });
  const clearWrites = () => {
    writeBuffer.mockClear();
    written.length = 0;
  };
  const device = {
    createTexture: (desc: GPUTextureDescriptor) => {
      let view: StubView | null = null; // 记忆化：同纹理恒同视图对象（绑定身份断言）
      const tex: StubTexture = {
        label: desc.label ?? '',
        format: desc.format,
        usage: desc.usage,
        destroyed: false,
        destroy: vi.fn(() => { tex.destroyed = true; }),
        createView: (d?: { label?: string }) => {
          view ??= { label: d?.label ?? `${desc.label}.View`, tex };
          return view;
        },
      };
      textures.push(tex);
      return tex;
    },
    createBuffer: (desc: GPUBufferDescriptor) => {
      const b: StubBuffer = {
        label: desc.label ?? '',
        size: desc.size,
        destroyed: false,
        destroy: vi.fn(() => { b.destroyed = true; }),
      };
      buffers.push(b);
      return b;
    },
    createBindGroupLayout: (d: GPUBindGroupLayoutDescriptor) => ({ label: d.label, entries: d.entries }),
    createShaderModule: () => ({}),
    createPipelineLayout: (d: GPUPipelineLayoutDescriptor) => ({ label: 'pl' }),
    createComputePipeline: (d: GPUComputePipelineDescriptor) => ({ label: d.label }),
    createRenderPipeline: (d: GPURenderPipelineDescriptor) => ({ label: d.label }),
    createSampler: () => ({}),
    createBindGroup: (d: GPUBindGroupDescriptor) => {
      const bg: StubBindGroup = { label: d.label ?? '', entries: d.entries as { binding: number; resource: unknown }[] };
      bindGroups.push(bg);
      return bg;
    },
    queue: { writeTexture },
  };
  const makePass = (): StubPass => {
    const pass: StubPass = {
      setPipeline: vi.fn(),
      setBindGroup: vi.fn(),
      dispatches: [],
      drawIndexed: vi.fn(),
      end: vi.fn(),
    };
    passes.push(pass);
    return pass;
  };
  const encoder = {
    beginComputePass: vi.fn(() => {
      const pass = makePass();
      return {
        setPipeline: pass.setPipeline,
        setBindGroup: pass.setBindGroup,
        dispatchWorkgroups: vi.fn((x: number, y: number) => pass.dispatches.push([x, y])),
        end: pass.end,
      };
    }),
    beginRenderPass: vi.fn(() => {
      const pass = makePass();
      return {
        setPipeline: pass.setPipeline,
        setBindGroup: pass.setBindGroup,
        setVertexBuffer: vi.fn(),
        setIndexBuffer: vi.fn(),
        drawIndexed: pass.drawIndexed,
        end: pass.end,
      };
    }),
  };
  const context = {
    initialized: true,
    device,
    preferredFormat: 'bgra8unorm',
    drawingBufferWidth: 64,
    drawingBufferHeight: 64,
    writeBuffer,
    createShaderModule: () => ({}),
  } as unknown as Context;
  return { context, writeBuffer, writeTexture, written, clearWrites, textures, buffers, bindGroups, passes, encoder };
}

/** 平缓沙丘海床（中心 +1.5m → 边缘 ~0，确定性；sizeM 可调仿射断言用）。 */
function makeField(context: Context, sizeM = 500): SeabedHeightField {
  return new SeabedHeightField(context, {
    center: { lon: CENTER[0], lat: CENTER[1] },
    sizeM,
    gridN: 64,
    heightFn: (x, z) =>
      1.5 * Math.exp(-(x * x + z * z) / (2 * 120 * 120))
      + 0.01 * Math.sin(x * 0.1) * Math.cos(z * 0.1),
  });
}

const FFT_STUB = {
  derivativeTexture: { label: 'deriv' },
  cascadeSizes: [7.1, 33.3],
} as unknown as FFTDataSource;

const CAUSTICS_STUB = { causticsTexture: { label: 'cau' } } as unknown as OceanCaustics;

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

/** 触发一次 floor.execute 并返回 FloorParams UBO 快照（40 floats）。 */
function floorParams(
  floor: CausticsFloor,
  ctx: { clearWrites: () => void; written: { buf: unknown; off: number; data: Float32Array }[] },
  encoder: object,
): Float32Array {
  floor.prepare();
  floor.updateCamera(makeCamera());
  floor.bindFFT(FFT_STUB);
  floor.bindCaustics(CAUSTICS_STUB);
  ctx.clearWrites();
  floor.execute(encoder as GPUCommandEncoder, {} as GPUTextureView, 12.5);
  const call = ctx.written.find(
    (w) => (w.buf as { label?: string }).label === 'CausticsFloorParamsUBO'
  );
  expect(call).toBeDefined();
  return call!.data;
}

/** SwashField.Params UBO 写入快照列表（每子步一次）。 */
function swashWrites(
  written: { buf: unknown; off: number; data: Float32Array }[],
): Float32Array[] {
  return written
    .filter((w) => (w.buf as { label?: string }).label === 'SwashField.Params')
    .map((w) => w.data);
}

describe('SwashField 构造与参数（1D Saint-Venant Track W）', () => {
  it('构造校验：未初始化 Context 抛错；gridN 上取 8 倍数并夹 64..512；非法窗口回退 500', () => {
    const { context } = makeContext();
    expect(() => new SwashField({ initialized: false } as unknown as Context, {})).toThrow();

    expect(new SwashField(context, { gridN: 100 }).gridN).toBe(104);   // ceil(100/8)*8
    expect(new SwashField(context, { gridN: 2000 }).gridN).toBe(512);  // 上限
    expect(new SwashField(context, { gridN: 3 }).gridN).toBe(64);      // 下限
    expect(new SwashField(context, { gridN: 128 }).gridN).toBe(128);   // 默认 128

    const bad = new SwashField(context, { windowSizeM: -5 });
    expect(bad.windowSizeMeters).toBe(500);
    expect(bad.cellM).toBeCloseTo(500 / 128, 12);
  });

  it('参数 setter 夹取：NaN 保持现值、越界钳边界，uniform 同步（amp idx8 / omega idx9）', () => {
    const { context, written, encoder } = makeContext();
    const swash = new SwashField(context, { gridN: 128 });
    expect(swash.wavePeriod).toBe(7);
    expect(swash.waveAmplitude).toBe(0.35);
    expect(swash.friction).toBe(0.06);
    expect(swash.dryDepth).toBe(0.02);

    swash.setWavePeriod(Number.NaN);
    expect(swash.wavePeriod).toBe(7);       // NaN 保持现值
    swash.setWavePeriod(0);
    expect(swash.wavePeriod).toBe(2);       // 下限
    swash.setWavePeriod(100);
    expect(swash.wavePeriod).toBe(20);      // 上限

    swash.setWaveAmplitude(Number.NaN);
    expect(swash.waveAmplitude).toBe(0.35);
    swash.setWaveAmplitude(5);
    expect(swash.waveAmplitude).toBe(2);
    swash.setWaveAmplitude(0.001);
    expect(swash.waveAmplitude).toBe(0.01);

    swash.setFriction(-1);
    expect(swash.friction).toBe(0);
    swash.setFriction(9);
    expect(swash.friction).toBe(1);

    swash.setDryDepth(10);
    expect(swash.dryDepth).toBe(0.5);
    swash.setDryDepth(0);
    expect(swash.dryDepth).toBe(0.001);

    // uniform 反映（SWASH_P_WAVE=8：x=amp idx8, y=omega idx9）
    swash.prepare();
    swash.execute(encoder as GPUCommandEncoder, {} as GPUTextureView, 0);
    swash.setWaveAmplitude(0.5);
    swash.setWavePeriod(5);
    swash.execute(encoder as GPUCommandEncoder, {} as GPUTextureView, 1 / 60);
    const writes = swashWrites(written);
    expect(writes.length).toBe(2);
    expect(writes[1][8]).toBe(Math.fround(0.5));
    expect(writes[1][9]).toBeCloseTo((2 * Math.PI) / 5, 6);
  });
});

describe('SwashField ping-pong / dispatch / reset', () => {
  it('prepare 幂等：双 rgba16float 纹理（STORAGE|TEXTURE）、UBO 64B、bg[k] 读 k 写 k^1', () => {
    const { context, textures, buffers, bindGroups, writeTexture } = makeContext();
    const swash = new SwashField(context, { gridN: 128 });
    swash.prepare();
    swash.prepare(); // 幂等
    const fields = textures.filter((t) => t.label === 'SwashField.FieldA' || t.label === 'SwashField.FieldB');
    expect(fields).toHaveLength(2);
    expect(fields[0].format).toBe('rgba16float');
    expect(fields[0].usage).toBe(0x10 | 0x08); // STORAGE_BINDING | TEXTURE_BINDING（桩 fake 值）

    const ubo = buffers.find((b) => b.label === 'SwashField.Params');
    expect(ubo?.size).toBe(SWASH_PARAMS_BYTES);
    expect(SWASH_PARAMS_BYTES).toBe(64);

    // 未绑海床 → binding 3 走 r32float -1000 极深 fallback
    expect(writeTexture).toHaveBeenCalled();
    const fbCall = writeTexture.mock.calls[0] as unknown[];
    expect(new Float32Array(fbCall[1] as ArrayBufferLike)[0]).toBe(-1000);

    expect(bindGroups).toHaveLength(2);
    const viewA = fields[0].createView();
    const viewB = fields[1].createView();
    expect(bindGroups[0].entries[1].resource).toBe(viewA); // 读 A
    expect(bindGroups[0].entries[2].resource).toBe(viewB); // 写 B
    expect(bindGroups[1].entries[1].resource).toBe(viewB); // 读 B
    expect(bindGroups[1].entries[2].resource).toBe(viewA); // 写 A
    expect(bindGroups[0].entries[3].resource.label).toBe('SwashField.SeabedFallback.View');
  });

  it('execute：swashView 先 null 后非 null 且逐次交替；dispatch 16×16；首帧 reset=1', () => {
    const { context, written, passes, encoder } = makeContext();
    const swash = new SwashField(context, { gridN: 128 });
    expect(swash.swashView).toBeNull(); // prepare 前

    swash.execute(encoder as GPUCommandEncoder, {} as GPUTextureView, 0);
    expect(passes).toHaveLength(1);
    expect(passes[0].dispatches).toEqual([[16, 16]]); // 128/8 = 16²
    const v0 = swash.swashView;
    expect(v0).not.toBeNull();

    swash.execute(encoder as GPUCommandEncoder, {} as GPUTextureView, 1 / 60);
    const v1 = swash.swashView;
    expect(v1).not.toBe(v0); // ping-pong 交替
    swash.execute(encoder as GPUCommandEncoder, {} as GPUTextureView, 2 / 60);
    expect(swash.swashView).toBe(v0); // 回到 A

    // reset 只在首个子步消费一次（uniform phys.w idx7）
    const writes = swashWrites(written);
    expect(writes.length).toBe(3);
    expect(writes[0][7]).toBe(1);
    expect(writes[1][7]).toBe(0);
    expect(writes[2][7]).toBe(0);
  });

  it('CFL 子步：windowSizeM=40（cell≈0.31m）dt=1/30 拆 3 子步，dtSub 合计=dt', () => {
    const { context, written, passes, encoder } = makeContext();
    const swash = new SwashField(context, { gridN: 128, windowSizeM: 40 });
    swash.execute(encoder as GPUCommandEncoder, {} as GPUTextureView, 0);       // 首帧 dt=0 → 1 子步
    swash.execute(encoder as GPUCommandEncoder, {} as GPUTextureView, 1 / 30);  // → 3 子步

    expect(passes[0].dispatches).toHaveLength(1);
    expect(passes[1].dispatches).toEqual([[16, 16], [16, 16], [16, 16]]);

    // 每子步一次 uniform 写；dt(idx2) 合计 = dt（1/30）
    const writes = swashWrites(written);
    expect(writes).toHaveLength(4); // 1 + 3
    const dtSum = writes[1][2] + writes[2][2] + writes[3][2];
    expect(dtSum).toBeCloseTo(1 / 30, 6);
    // 时间单调推进（grid.w idx3）
    expect(writes[2][3]).toBeGreaterThan(writes[1][3]);
    expect(writes[3][3]).toBeGreaterThan(writes[2][3]);
  });

  it('默认 500m 窗（cell≈3.9m）dt=1/30 单子步（CFL 富余）', () => {
    const { context, passes, encoder } = makeContext();
    const swash = new SwashField(context, { gridN: 128 });
    swash.execute(encoder as GPUCommandEncoder, {} as GPUTextureView, 0);
    swash.execute(encoder as GPUCommandEncoder, {} as GPUTextureView, 1 / 30);
    expect(passes[1].dispatches).toHaveLength(1);
  });

  it('bindSeabed：窗口/中心跟随 field，binding 3 切 seabedTexture，幂等 + null 恢复', () => {
    const { context, bindGroups, encoder } = makeContext();
    const swash = new SwashField(context, { gridN: 128 });
    const field = makeField(context, 420);
    swash.bindSeabed(field);
    expect(swash.windowSizeMeters).toBe(420);
    expect(swash.cellM).toBeCloseTo(420 / 128, 12);
    expect(swash.centerRadial).toBe(field.centerRadial); // 同引用透传

    swash.execute(encoder as GPUCommandEncoder, {} as GPUTextureView, 0);
    const bgBound = bindGroups.at(-1)!;
    expect(bgBound.entries[3].resource.label).toBe('SeabedHeightField.View');

    const count = bindGroups.length;
    swash.bindSeabed(field); // 同引用 → 无操作
    swash.execute(encoder as GPUCommandEncoder, {} as GPUTextureView, 1 / 60);
    expect(bindGroups.length).toBe(count); // 不重建

    swash.bindSeabed(null);
    expect(swash.windowSizeMeters).toBe(500);
    expect(swash.centerRadial).not.toBe(field.centerRadial);
    swash.execute(encoder as GPUCommandEncoder, {} as GPUTextureView, 2 / 60);
    expect(bindGroups.at(-1)!.entries[3].resource.label).toBe('SwashField.SeabedFallback.View');
  });

  it('destroy：纹理/UBO 全部销毁，execute 不再 dispatch，swashView 归 null', () => {
    const { context, textures, buffers, passes, encoder } = makeContext();
    const swash = new SwashField(context, { gridN: 128 });
    swash.execute(encoder as GPUCommandEncoder, {} as GPUTextureView, 0);
    swash.destroy();
    expect(textures.filter((t) => t.label.startsWith('SwashField.Field')).every((t) => t.destroyed)).toBe(true);
    expect(buffers.find((b) => b.label === 'SwashField.Params')?.destroyed).toBe(true);
    expect(swash.swashView).toBeNull();

    const count = passes.length;
    swash.execute(encoder as GPUCommandEncoder, {} as GPUTextureView, 1 / 30);
    expect(passes.length).toBe(count); // 销毁后 no-op
  });
});

describe('SwashComputeShader 源串（物理公式钉死）', () => {
  const src = buildSwashShader(128);
  it('gridN 编译期注入 + workgroup 尺寸', () => {
    expect(src).toContain('const SW_N : u32 = 128u;');
    expect(src).toContain('@workgroup_size(8, 8, 1)');
  });
  it('Rusanov 数值通量 + 干面通量置零（湿润化阈值）', () => {
    expect(src).toContain('0.5 * ( dC * sC.g + dR * sR.g ) - 0.5 * aR * ( sR.r - sC.r )');
    expect(src).toContain('select( fR, 0.0, max( dC, dR ) < dry )');
    expect(src).toContain('select( fL, 0.0, max( dC, dL ) < dry )');
  });
  it('动量：坡度中心差分 + 半隐式 Manning 摩擦（分母 ≥ dry 不除零）', () => {
    expect(src).toContain('u1 / ( 1.0 + dt * sp.phys.y * abs( u1 )');
    expect(src).toContain('- dt * g * ( sR.r - sL.r ) / ( 2.0 * e )');
  });
  it('入射波 forcing：深水带松弛 + 主波正弦 + 切向相位', () => {
    expect(src).toContain('sp.wave.x * sin( sp.wave.y * sp.grid.w + phi )');
    expect(src).toContain('eta1 = mix( eta1, target, k )');
    expect(src).toContain('if ( inRange && b < sp.misc.y ) {');
  });
  it('干湿前沿 + 海床 textureLoad（r32float unfilterable）+ ping-pong 输出', () => {
    expect(src).toContain('if ( eta1 - b < dry ) {');
    expect(src).toContain('textureLoad( seabedTex, tc, 0 ).r');
    expect(src).toContain('texture_storage_2d<rgba16float, write>');
    expect(src).toContain('u1 = 0.0;');
    expect(src).toContain('eta1 = max( eta1, b );');
  });
});

describe('CausticsFloor.bindSwash 集成（Track W）', () => {
  it('bindSwash：P[32..35] 仿射（同源海床 → 0.5 偏移 + 1/sizeM 缩放）、binding 6 = swashView', () => {
    const { context, bindGroups, clearWrites, written, encoder } = makeContext();
    const field = makeField(context, 420);
    const swash = new SwashField(context, { gridN: 128 });
    swash.bindSeabed(field);

    const floor = new CausticsFloor(context, { context, center: CENTER, sizeM: 400, depthM: 6 });
    floor.bindSeabed(field);
    floor.bindSwash(swash);
    floor.setSwash(0.8);

    swash.execute(encoder as GPUCommandEncoder, {} as GPUTextureView, 0); // swashView 就绪
    const P = floorParams(floor, { clearWrites, written }, encoder);
    expect(FLOOR_PARAMS_UBO_BYTES).toBe(160);
    expect(P.length).toBe(40);
    // 同源海床：中心重合 → offset (0.5, 0.5)，scale 1/420
    expect([P[32], P[33]]).toEqual([0.5, 0.5]);
    expect([P[34], P[35]]).toEqual([Math.fround(1 / 420), Math.fround(1 / 420)]);
    expect(P[27]).toBe(Math.fround(0.8)); // wet.w 门控照常

    const bg = bindGroups.at(-1)!;
    expect(bg.entries[6].binding).toBe(6);
    expect(bg.entries[6].resource).toBe(swash.swashView); // 模拟场视图
  });

  it('bindSwash(null)：P[32..35] 复位 + P[0..31] 与从未绑定时逐位一致（零回归）', () => {
    const { context, bindGroups, clearWrites, written, encoder } = makeContext();
    const field = makeField(context, 420);

    // floorA：从不绑 swash（基线）
    const floorA = new CausticsFloor(context, { context, center: CENTER, sizeM: 400, depthM: 6, shoreStrength: 0.6 });
    floorA.bindSeabed(field);
    floorA.setSwash(0.8);
    const pa = floorParams(floorA, { clearWrites, written }, encoder);

    // floorB：绑 → 解绑
    const swash = new SwashField(context, { gridN: 128 });
    swash.bindSeabed(field);
    swash.execute(encoder as GPUCommandEncoder, {} as GPUTextureView, 0);
    const floorB = new CausticsFloor(context, { context, center: CENTER, sizeM: 400, depthM: 6, shoreStrength: 0.6 });
    floorB.bindSeabed(field);
    floorB.bindSwash(swash);
    floorB.setSwash(0.8);
    expect(floorParams(floorB, { clearWrites, written }, encoder)[34]).not.toBe(0); // 绑定态生效
    floorB.bindSwash(null);
    const pb = floorParams(floorB, { clearWrites, written }, encoder);

    expect([pb[32], pb[33], pb[34], pb[35]]).toEqual([0.5, 0.5, 0, 0]);
    expect(Array.from(pb.slice(0, 32))).toEqual(Array.from(pa.slice(0, 32))); // 零回归

    // 绑定态曾有真实模拟视图进 binding 6；解绑后回落占位纹理（绑定合法性，
    // FS 侧 swash_uv.z==0 门控根本不采样）
    const boundBg = bindGroups.find(
      (b) => b.entries.length === 7
        && (b.entries[6].resource as { label?: string }).label === 'SwashField.FieldB.View'
    );
    expect(boundBg).toBeDefined();
  });
});
