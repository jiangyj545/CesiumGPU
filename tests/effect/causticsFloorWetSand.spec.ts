import { describe, expect, it, vi } from 'vitest';
import { CausticsFloor } from '../../src/effect/CausticsFloor';
import { SeabedHeightField } from '../../src/effect/SeabedHeightField';
import { CAUSTICS_FLOOR_SHADER, FLOOR_PARAMS_UBO_BYTES } from '../../src/shaders/OceanCausticsShader';
import { ecefFromLonLatHeight } from '../../src/terrain/TerrainProvider';
import { Camera } from '../../src/scene/Camera';
import { Cartesian3 } from '../../src/math/Cartesian3';
import type { Context } from '../../src/renderer/Context';
import type { FFTDataSource, OceanCaustics } from '../../src/effect/OceanCaustics';

/**
 * 海洋 L4-B 海床起伏几何 + 湿沙线 — 纯 CPU / 字符串验证（记录型桩 device，
 * 无真实 WebGPU，cssm.spec 同款模式）：
 * bindSeabed(null) 零回归、bindSeabed(field) 顶点海拔烘焙、UBO P[24..27]、
 * 湿沙 shader 源串、bindSeabed 幂等。
 */

const ES = Camera.EARTH_SCALE;
const CENTER: [number, number] = [121.5, 31.0];
const SIZE_M = 400;
const SEGMENTS = 128;
const STRIDE = 6; // [ecefX, ecefY, ecefZ, planeX, planeZ, seabedH]

// node 环境无 WebGPU 运行时全局（bowSpray/seabedField.spec 同款兜底）
const g = globalThis as unknown as Record<string, unknown>;
g.GPUTextureUsage ??= { TEXTURE_BINDING: 0x08, COPY_DST: 0x04, RENDER_ATTACHMENT: 0x10 };
g.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
g.GPUBufferUsage ??= { MAP_READ: 0x01, VERTEX: 0x20, INDEX: 0x10, UNIFORM: 0x40, COPY_DST: 0x08 };

/** 记录型桩 Context：捕获 createBuffer / writeBuffer，其余返回轻量桩对象。 */
function makeContext() {
  const writeBuffer = vi.fn();
  const buffers: { label: string; size: number }[] = [];
  const device = {
    createBuffer: (desc: GPUBufferDescriptor) => {
      const b = { label: desc.label ?? '', size: desc.size, destroyed: false, destroy: vi.fn() };
      buffers.push(b);
      return b;
    },
    createBindGroupLayout: (d: GPUBindGroupLayoutDescriptor) => ({ label: d.label }),
    createShaderModule: () => ({}),
    createPipelineLayout: (d: GPUPipelineLayoutDescriptor) => ({ label: 'pl' }),
    createRenderPipeline: (d: GPURenderPipelineDescriptor) => ({ label: d.label }),
    createSampler: () => ({}),
    createBindGroup: () => ({ label: 'bg' }),
    // binding 5 fallback 纹理（_seabedFallbackView 惰性创建需要，makeFieldContext 同款）
    createTexture: (desc: GPUTextureDescriptor) => ({
      label: desc.label,
      createView: () => ({ label: `${desc.label}.View` }),
      destroy: vi.fn(),
    }),
    queue: { writeTexture: vi.fn() },
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
  return { context, writeBuffer, buffers };
}

/** SeabedHeightField 专用桩 Context（seabedField.spec 同款，只需 createTexture/queue）。 */
function makeFieldContext(): Context {
  const device = {
    createTexture: (desc: GPUTextureDescriptor) => ({
      label: desc.label,
      createView: () => ({ label: `${desc.label}.View` }),
      destroy: vi.fn(),
    }),
    queue: { writeTexture: vi.fn() },
  };
  return { initialized: true, device } as unknown as Context;
}

/** 峰值露出水面的平缓沙丘（中心 +1.5m → 边缘 ~0，确定性）。 */
function makeField(sizeM = 500): SeabedHeightField {
  return new SeabedHeightField(makeFieldContext(), {
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

function makeEncoder(): GPUCommandEncoder {
  const pass = {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    setVertexBuffer: vi.fn(),
    setIndexBuffer: vi.fn(),
    drawIndexed: vi.fn(),
    end: vi.fn(),
  };
  return { beginRenderPass: vi.fn(() => pass) } as unknown as GPUCommandEncoder;
}

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

/** floor 顶点几何（内部字段白盒访问，cssm.spec 无 GPU 断言同思路）。 */
function geoOf(floor: CausticsFloor): { vertices: Float32Array; indexCount: number } {
  return (floor as unknown as { _geometry: { vertices: Float32Array; indexCount: number } })._geometry;
}

function vertexBufferOf(floor: CausticsFloor): object {
  return (floor as unknown as { _vertexBuffer: object })._vertexBuffer as object;
}

/** 触发一次 execute 并返回写入的 FloorParams UBO 快照（Float32Array，32 floats）。 */
function paramsOf(floor: CausticsFloor, writeBuffer: ReturnType<typeof vi.fn>): Float32Array {
  floor.prepare();
  floor.updateCamera(makeCamera());
  floor.bindFFT(FFT_STUB);
  floor.bindCaustics(CAUSTICS_STUB);
  writeBuffer.mockClear();
  floor.execute(makeEncoder(), {} as GPUTextureView, 12.5);
  const call = writeBuffer.mock.calls.find(
    (c: unknown[]) => (c[0] as { label?: string }).label === 'CausticsFloorParamsUBO'
  );
  expect(call).toBeDefined();
  return call![2] as Float32Array;
}

/** 平面米 → 经纬度（与 _buildGeometry 同款小范围线性换算）。 */
function lonLatOf(planeX: number, planeZ: number): { lon: number; lat: number } {
  const metersPerDegLon = 111320 * Math.cos((CENTER[1] * Math.PI) / 180);
  return {
    lon: CENTER[0] + planeX / metersPerDegLon,
    lat: CENTER[1] + planeZ / 111320,
  };
}

describe('CausticsFloor 湿沙 UBO / 几何（海洋 L4-B）', () => {
  it('bindSeabed(null)：几何逐字节恢复平底现状，UBO P[24..27] 全 0', () => {
    const { context, writeBuffer } = makeContext();
    const floor = new CausticsFloor(context, { context, center: CENTER, sizeM: SIZE_M, depthM: 6 });
    const initial = Float32Array.from(geoOf(floor).vertices);

    // field → null 往返：重建出的平底几何与构造时逐字节一致（零回归）
    floor.bindSeabed(makeField());
    expect(geoOf(floor).vertices).not.toEqual(initial);
    floor.bindSeabed(null);
    expect(geoOf(floor).vertices).toEqual(initial);

    // 处女对象上 bindSeabed(null)：幂等早退，无重建
    const snap = geoOf(floor).vertices;
    floor.bindSeabed(null);
    expect(geoOf(floor).vertices).toBe(snap);

    // 顶点前 5 分量 = 引擎 ecefFromLonLatHeight(lon, lat, 0)（与旧版逐位一致）
    expect(geoOf(floor).indexCount).toBe(SEGMENTS * SEGMENTS * 6);
    for (const [ix, iy] of [[0, 0], [128, 0], [64, 64], [128, 128], [0, 128]] as const) {
      const v = (iy * (SEGMENTS + 1) + ix) * STRIDE;
      const planeX = (ix / SEGMENTS - 0.5) * SIZE_M;
      const planeZ = (iy / SEGMENTS - 0.5) * SIZE_M;
      expect(geoOf(floor).vertices[v + 3]).toBe(Math.fround(planeX));
      expect(geoOf(floor).vertices[v + 4]).toBe(Math.fround(planeZ));
      const { lon, lat } = lonLatOf(planeX, planeZ);
      const expected: [number, number, number] = [0, 0, 0];
      ecefFromLonLatHeight(lon, lat, 0, expected);
      expect(geoOf(floor).vertices[v + 0]).toBe(Math.fround(expected[0]));
      expect(geoOf(floor).vertices[v + 1]).toBe(Math.fround(expected[1]));
      expect(geoOf(floor).vertices[v + 2]).toBe(Math.fround(expected[2]));
      expect(geoOf(floor).vertices[v + 5]).toBe(0); // 平底模式海拔通道写 0
    }

    // UBO：P[0..23] 与旧版一致 + P[24..27] 全 0（湿沙全关）
    const P = paramsOf(floor, writeBuffer);
    expect(FLOOR_PARAMS_UBO_BYTES).toBe(160); // 128B + swash_uv 仿射（Track W）
    expect(P.length).toBe(FLOOR_PARAMS_UBO_BYTES / 4);
    expect(P[8]).toBe(400);   // spacing.x = sizeM（无 caustics 回退）
    expect(P[9]).toBe(400);   // spacing.y
    expect(P[10]).toBe(6);    // spacing.z depthM
    expect(P[11]).toBe(Math.fround(7.1));   // spacing.w 级联0（UBO 为 f32）
    expect(P[12]).toBe(2.5);  // depths.x
    expect(P[13]).toBe(6);    // depths.y
    expect(P[14]).toBe(Math.fround(33.3));  // depths.z
    expect(P[15]).toBe(Math.fround(0.92));  // depths.w
    expect([P[16], P[17], P[18]]).toEqual([Math.fround(0.90), Math.fround(0.85), Math.fround(0.72)]); // sandCol（2026-09-30 白沙提亮，tidewater 观感）
    expect([P[20], P[21], P[22]]).toEqual([Math.fround(0.55), Math.fround(0.72), Math.fround(0.88)]); // skyCol
    expect(P[24]).toBe(0);    // shoreStrength 关
    expect(P[25]).toBe(12.5); // time 透传
    expect(P[26]).toBe(0);    // 平底模式
    expect(P[27]).toBe(0);    // swash 默认关（零回归）
    expect([P[28], P[29], P[30], P[31]]).toEqual([0.5, 0.5, 0, 0]); // seabedUv 未绑定
    expect([P[32], P[33], P[34], P[35]]).toEqual([0.5, 0.5, 0, 0]); // swashUv 未绑定（解析路径）
  });

  it('bindSeabed(field)：顶点海拔 = sampleHeight，ECEF 经引擎单源投影（抽 5 顶点反算）', () => {
    const { context } = makeContext();
    const floor = new CausticsFloor(context, { context, center: CENTER, sizeM: SIZE_M, depthM: 6 });
    const field = makeField();
    floor.bindSeabed(field);

    const vertices = geoOf(floor).vertices;
    const samples = [[0, 0], [128, 128], [64, 64], [30, 100], [110, 20]] as const;
    const base: [number, number, number] = [0, 0, 0];
    for (const [ix, iy] of samples) {
      const v = (iy * (SEGMENTS + 1) + ix) * STRIDE;
      const planeX = vertices[v + 3];
      const planeZ = vertices[v + 4];
      const h = field.sampleHeight(planeX, planeZ);
      const { lon, lat } = lonLatOf(planeX, planeZ);

      // 顶点海拔通道 = f64 sampleHeight 的 f32 量化（|误差| < 1e-3）
      expect(Math.abs(vertices[v + 5] - h)).toBeLessThan(1e-3);

      // ECEF 逐位 = 引擎 ecefFromLonLatHeight(lon, lat, h)（单一来源换算，零误差）
      ecefFromLonLatHeight(lon, lat, h, base);
      expect(vertices[v + 0]).toBe(Math.fround(base[0]));
      expect(vertices[v + 1]).toBe(Math.fround(base[1]));
      expect(vertices[v + 2]).toBe(Math.fround(base[2]));

      // 反算海拔（语义复核）：ecef(h) = ecef(0)·(1+h/ES) 同向缩放
      // → h = (|v_h|/|v_0| - 1)·ES。顶点为 f32（|v|~1，1 ulp ≈ 0.38m 海拔
      // 量化极限），故反算界取 0.5m；逐位断言已给出零误差证明。
      ecefFromLonLatHeight(lon, lat, 0, base);
      const m0 = Math.hypot(base[0], base[1], base[2]);
      const mh = Math.hypot(vertices[v + 0], vertices[v + 1], vertices[v + 2]);
      const recovered = (mh / m0 - 1) * ES;
      expect(Math.abs(recovered - h)).toBeLessThan(0.5);
    }
  });

  it('UBO P[24] 写 shoreStrength，P[26] 随 bindSeabed 切高度模式', () => {
    const { context, writeBuffer } = makeContext();
    const floor = new CausticsFloor(context, {
      context, center: CENTER, sizeM: SIZE_M, depthM: 6, shoreStrength: 0.8,
    });

    expect(paramsOf(floor, writeBuffer)[24]).toBe(Math.fround(0.8)); // UBO 为 f32
    expect(paramsOf(floor, writeBuffer)[26]).toBe(0);

    floor.bindSeabed(makeField());
    const P = paramsOf(floor, writeBuffer);
    expect(P[24]).toBe(Math.fround(0.8));
    expect(P[26]).toBe(1); // 高度场模式：VS 不再下移 depthM
    // seabedUv 仿射：中心重合 → offset=(0.5,0.5)，scale=1/sizeMeters（默认 500）
    expect([P[28], P[29]]).toEqual([0.5, 0.5]);
    expect([P[30], P[31]]).toEqual([Math.fround(1 / 500), Math.fround(1 / 500)]);

    // 非法值钳 0（NaN/负数不进 UBO）
    floor.shoreStrength = Number.NaN;
    expect(paramsOf(floor, writeBuffer)[24]).toBe(0);
    floor.shoreStrength = -2;
    expect(paramsOf(floor, writeBuffer)[24]).toBe(0);
  });

  it('shader 源串：wetK 公式 / 0.58 湿色 / runup 正弦 / shoreStrength 门', () => {
    expect(CAUSTICS_FLOOR_SHADER).toContain(
      'smoothstep( 1.7, 0.5, seabedH + nz * 0.3 ) * shoreStrength * mottle * shoreFade'
    );
    expect(CAUSTICS_FLOOR_SHADER).toContain('0.7 + 0.3 * nz2');                           // mottle
    expect(CAUSTICS_FLOOR_SHADER).toContain('col * select( 0.58, 0.85, uwCam ) * vec3<f32>( 0.97, 0.98, 1.0 )'); // 湿色（水下相机放宽 0.85，uwFog 统一水柱）
    expect(CAUSTICS_FLOOR_SHADER).toContain('0.25 + 0.45 * sin( wt * 0.7 + nz * 3.0 )');  // runup
    expect(CAUSTICS_FLOOR_SHADER).toContain(
      'smoothstep( 0.22, 0.0, abs( seabedH - runup ) ) * shoreStrength * shoreFade'
    );
    expect(CAUSTICS_FLOOR_SHADER).toContain('0.5 * swashLine');  // 混白
    expect(CAUSTICS_FLOOR_SHADER).toContain('wt * 0.15');        // time 噪声漂移
    expect(CAUSTICS_FLOOR_SHADER).toContain('let shoreStrength = max( floorParams.wet.x, 0.0 )'); // 门
    // 高度场模式 VS 门（wet.z）：烘焙模式不再下移 depthM
    expect(CAUSTICS_FLOOR_SHADER).toContain('select( flatDrop, vec3<f32>( 0.0 ), floorParams.wet.z > 0.5 )');
    // 掠射条纹修复：FS textureLoad 高度场 + 湿沙细节距离淡出
    expect(CAUSTICS_FLOOR_SHADER).toContain('textureLoad( seabedTex, tc, 0 ).r');
    expect(CAUSTICS_FLOOR_SHADER).toContain('let shoreFade = 1.0 - smoothstep( 80.0, 180.0, distM )');
    expect(CAUSTICS_FLOOR_SHADER).toContain('seabed_uv: vec4<f32>');
  });

  it('bindSeabed 幂等：同 field 不重建不重写，顶点缓冲对象恒一', () => {
    const { context, writeBuffer, buffers } = makeContext();
    const floor = new CausticsFloor(context, { context, center: CENTER, sizeM: SIZE_M, depthM: 6 });
    const field = makeField();

    // prepare 前 bind：只重建 CPU 几何，不触碰 device
    floor.bindSeabed(field);
    floor.bindSeabed(field); // 同引用 → 无操作
    expect(buffers.filter((b) => b.label === 'CausticsFloorVertices')).toHaveLength(0);

    floor.prepare();
    const vb1 = vertexBufferOf(floor);
    expect(buffers.filter((b) => b.label === 'CausticsFloorVertices')).toHaveLength(1);

    writeBuffer.mockClear();
    floor.bindSeabed(field); // prepare 后同引用：原位不重写（幂等早退）
    expect(vertexBufferOf(floor)).toBe(vb1);
    expect(writeBuffer).not.toHaveBeenCalled();

    expect(geoOf(floor).indexCount).toBe(SEGMENTS * SEGMENTS * 6);
    expect(buffers.filter((b) => b.label === 'CausticsFloorVertices')).toHaveLength(1);

    // 换 field：原位重写（同字节数），不新建缓冲
    floor.bindSeabed(makeField(420));
    expect(vertexBufferOf(floor)).toBe(vb1);
    expect(buffers.filter((b) => b.label === 'CausticsFloorVertices')).toHaveLength(1);
    expect(writeBuffer).toHaveBeenCalledTimes(1);
    const [buf, offset, data] = writeBuffer.mock.calls[0] as [object, number, Float32Array];
    expect(buf).toBe(vb1);
    expect(offset).toBe(0);
    expect(data.byteLength).toBe((SEGMENTS + 1) * (SEGMENTS + 1) * STRIDE * 4);
  });

  it('swash：setSwash 写 P[27]（默认 0 零回归），shader 门控 / 相位场 / 白沫源串', () => {
    const { context, writeBuffer } = makeContext();
    const floor = new CausticsFloor(context, { context, center: CENTER, sizeM: SIZE_M, depthM: 6 });
    expect(paramsOf(floor, writeBuffer)[27]).toBe(0); // 默认关

    floor.setSwash(0.8);
    floor.bindSeabed(makeField());
    const P = paramsOf(floor, writeBuffer);
    expect(P[27]).toBe(Math.fround(0.8)); // UBO 为 f32
    expect(P[24]).toBe(0);                // 与 shoreStrength 独立（未开湿沙仍 swash 生效）

    // 非法值钳 [0,1]
    floor.setSwash(Number.NaN);
    expect(paramsOf(floor, writeBuffer)[27]).toBe(0);
    floor.setSwash(-0.5);
    expect(paramsOf(floor, writeBuffer)[27]).toBe(0);
    floor.setSwash(2.5);
    expect(paramsOf(floor, writeBuffer)[27]).toBe(1);

    // 构造选项同款语义
    const { context: ctx2, writeBuffer: wb2 } = makeContext();
    const f2 = new CausticsFloor(ctx2, { context: ctx2, center: CENTER, sizeM: SIZE_M, depthM: 6, swash: 0.5 });
    expect(paramsOf(f2, wb2)[27]).toBe(Math.fround(0.5));

    // shader 源串：wet.w>0.001 分支门控（0 = 整块跳过逐字节零回归）
    expect(CAUSTICS_FLOOR_SHADER).toContain('let swashAmt = max( floorParams.wet.w, 0.0 )');
    expect(CAUSTICS_FLOOR_SHADER).toContain('if ( swashAmt > 0.001 ) {');
    // 岸线相位场：海床梯度 ±2 texel 差分 + 平底 vnoise 回退相位
    expect(CAUSTICS_FLOOR_SHADER).toContain('input.planeXZ.x * 0.42 + nz * 3.0');
    expect(CAUSTICS_FLOOR_SHADER).toContain(
      'textureLoad( seabedTex, clamp( tcS + vec2<i32>( 2, 0 ), vec2<i32>( 0 ), hiS ), 0 ).r'
    );
    // 水位涨落：run-up 幅度 0.6m × T≈7s（2π/7≈0.898）正弦 + 平滑 smoothstep 带
    expect(CAUSTICS_FLOOR_SHADER).toContain('0.6 * swashAmt * sin( phase )');
    expect(CAUSTICS_FLOOR_SHADER).toContain('wt * 0.898');
    expect(CAUSTICS_FLOOR_SHADER).toContain('0.25 + max( swashLevel, 0.0 )');
    // 回退白沫线：退潮门（cos<0）+ 白沫色与海面泡沫一致
    expect(CAUSTICS_FLOOR_SHADER).toContain('clamp( - cos( phase ), 0.0, 1.0 )');
    expect(CAUSTICS_FLOOR_SHADER).toContain('vec3<f32>( 0.90, 0.94, 0.97 )');
  });
});
