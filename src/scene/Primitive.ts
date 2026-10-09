import { Context } from '../renderer/Context';
import { Matrix4 } from '../math/Matrix4';
import { Cartesian3 } from '../math/Cartesian3';
import { Color } from '../math/Color';
import { GBUFFER_SHADER, GBUFFER_UNLIT_ENTRY_POINT } from '../shaders/GBufferShader';

export interface PrimitiveOptions {
  /** Interleaved vertex buffer: [position(3), normal(3), uv(2)] per vertex */
  vertices: Float32Array;
  indices:  Uint32Array;
  /** World-space position (normalised ECEF, Earth radius = 1) */
  position?: Cartesian3;
  /** Uniform scale in world units */
  scale?: number;
  /** Base colour if no texture */
  color?: Color;
  /** Whether this primitive should be rendered as transparent */
  translucent?: boolean;
  /** Alpha for transparent primitives */
  alpha?: number;
  /**
   * Unlit (emissive) rendering: draw with the unlit G-Buffer pipeline variant.
   * The G-Buffer fragment writes a zero-length normal marker which the
   * lighting pass detects to output raw albedo (no shading).  pickId and
   * bind-group structure are identical to lit primitives.
   */
  unlit?: boolean;
  /**
   * Per-vertex pick id override (u32, one entry per vertex) — F12 instance-level
   * picking. Stored in a SECOND vertex buffer (stride 4, format 'uint32') that
   * only the pick pipeline binds (see PickPass); the G-Buffer path is untouched.
   * A value of 0 means "no instance" and falls back to the primitive pickId.
   * Ids must stay below 0x800000: the pick shader emits `0x800000 | instanceId`
   * so the instance id space never collides with primitive pickIds.
   */
  instanceIds?: Uint32Array;
}

/**
 * BoundingSphere - Simple centre + radius sphere in world space, used for
 * frustum culling (see {@link Primitive.boundingSphere} / renderer/Frustum).
 */
export interface BoundingSphere {
  /** World-space centre. */
  center: Cartesian3;
  /** World-space radius (local radius × model scale, inflated 5%). */
  radius: number;
}

/**
 * Primitive - A renderable mesh in the scene.
 * Matches Cesium's Primitive API (simplified).
 */
export class Primitive {
  /** Global 24-bit pick-id allocator (id 0 is reserved for background). */
  private static _nextPickId = 1;

  show: boolean = true;
  /**
   * Draw with depth testing disabled (P2-4 `disableDepthTestDistance`):
   * translucent primitives render through geometry. Read each frame by the
   * TransparentPass pipeline switch (`depthCompare: 'always'`).
   */
  alwaysPassDepth = false;
  modelMatrix: Matrix4;

  private readonly _pickId: number;

  /** 24-bit GPU pick id (readonly, assigned once at construction). */
  get pickId(): number { return this._pickId; }

  private _context: Context;
  private _vertexBuffer: GPUBuffer;
  /** Optional per-vertex instance id buffer (pick pipeline slot 1); null otherwise. */
  private _instanceIdBuffer: GPUBuffer | null = null;
  private _indexBuffer: GPUBuffer;
  private _indexCount: number;
  private _translucent: boolean;
  private readonly _unlit: boolean;

  private _modelUniformBuffer: GPUBuffer;
  private _materialUniformBuffer: GPUBuffer;
  private _opaquePipeline: GPURenderPipeline | null = null;
  private _opaqueModelBindGroup: GPUBindGroup | null = null;

  private _color: Color;
  private _alpha: number;

  /** Local-space bounding sphere from the vertex min/max corners. */
  private readonly _localBoundingSphere: BoundingSphere;
  /** Reused world-space sphere (mutated in place by the getter — no per-frame allocation). */
  private readonly _worldBoundingSphere: BoundingSphere = {
    center: new Cartesian3(0, 0, 0),
    radius: 0,
  };

  // Base-colour texture (1×1 white if none supplied)
  private _texture:  GPUTexture;
  private _sampler:  GPUSampler;

  constructor(context: Context, options: PrimitiveOptions) {
    this._context    = context;
    this._translucent = options.translucent ?? false;
    this._unlit      = options.unlit ?? false;
    this._color      = options.color ?? Color.WHITE;
    this._alpha      = options.alpha ?? 1.0;

    // Allocate the 24-bit GPU pick id (0 = background / no hit)
    if (Primitive._nextPickId > 0xFFFFFF) {
      throw new Error('Primitive: 24-bit pick id space exhausted');
    }
    this._pickId = Primitive._nextPickId++;

    const device = context.device;

    // Vertex buffer
    this._vertexBuffer = device.createBuffer({
      label: 'Primitive.VBO',
      size:  options.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    // Copy through the view (not .buffer) so subarray() views upload exactly
    // their own range instead of the whole underlying ArrayBuffer.
    context.writeBuffer(this._vertexBuffer, 0, new Float32Array(options.vertices));

    // Optional per-vertex instance id buffer (F12) — bound only by the pick
    // pipeline's second vertex slot; never touched by the G-Buffer path.
    if (options.instanceIds && options.instanceIds.length > 0) {
      this._instanceIdBuffer = device.createBuffer({
        label: 'Primitive.InstanceIdVBO',
        size:  options.instanceIds.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      context.writeBuffer(this._instanceIdBuffer, 0, new Uint32Array(options.instanceIds));
    }

    // Index buffer
    this._indexBuffer = device.createBuffer({
      label: 'Primitive.IBO',
      size:  options.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    context.writeBuffer(this._indexBuffer, 0, new Uint32Array(options.indices));
    this._indexCount = options.indices.length;

    // Model uniform buffer: modelMatrix (64 bytes) + normalMatrix (64 bytes) = 128 bytes
    this._modelUniformBuffer = device.createBuffer({
      label: 'Primitive.ModelUBO',
      size:  128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Material uniform buffer: baseColor(16) + roughness(4) + metallic(4) + pad(8) = 32 bytes
    this._materialUniformBuffer = device.createBuffer({
      label: 'Primitive.MaterialUBO',
      size:  32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Model matrix
    this.modelMatrix = Matrix4.identity();
    if (options.position) {
      Matrix4.fromTranslation(options.position, this.modelMatrix);
    }
    if (options.scale !== undefined) {
      const s = options.scale;
      this.modelMatrix.values[0]  *= s;
      this.modelMatrix.values[5]  *= s;
      this.modelMatrix.values[10] *= s;
    }

    // 1×1 white texture (placeholder)
    this._texture = device.createTexture({
      label:  'Primitive.BaseColor',
      size:   { width: 1, height: 1 },
      format: 'rgba8unorm',
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const white = new Uint8Array([
      Math.round(this._color.red   * 255),
      Math.round(this._color.green * 255),
      Math.round(this._color.blue  * 255),
      255,
    ]);
    device.queue.writeTexture(
      { texture: this._texture },
      white,
      { bytesPerRow: 4 },
      { width: 1, height: 1 }
    );

    this._sampler = context.createSampler({
      minFilter: 'linear',
      magFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    });

    this._updateModelUBO();
    this._updateMaterialUBO();

    this._localBoundingSphere = this._computeBoundingSphere(options.vertices);
  }

  get translucent(): boolean { return this._translucent; }

  /** True when this primitive carries a per-vertex instance id buffer (F12). */
  get hasInstanceIds(): boolean { return this._instanceIdBuffer !== null; }

  /** Per-vertex instance id buffer (pick pipeline vertex slot 1), or null. */
  get instanceIdBuffer(): GPUBuffer | null { return this._instanceIdBuffer; }

  /** True when this primitive draws through the unlit (emissive) pipeline. */
  get unlit(): boolean { return this._unlit; }

  /**
   * World-space bounding sphere (local vertex sphere transformed by the
   * current modelMatrix), used for frustum culling. The returned object is
   * reused between calls — treat it as read-only and do not store it.
   */
  get boundingSphere(): BoundingSphere {
    const m  = this.modelMatrix.values;
    const lc = this._localBoundingSphere.center;
    const wc = this._worldBoundingSphere.center;

    // Transform the centre (upper-left 3×3 + translation column).
    wc.x = m[0] * lc.x + m[4] * lc.y + m[8]  * lc.z + m[12];
    wc.y = m[1] * lc.x + m[5] * lc.y + m[9]  * lc.z + m[13];
    wc.z = m[2] * lc.x + m[6] * lc.y + m[10] * lc.z + m[14];

    // Radius scales with the largest basis-column magnitude (uniform scale
    // in this engine, but the max keeps the sphere safe under any scaling).
    const s0 = Math.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
    const s1 = Math.sqrt(m[4] * m[4] + m[5] * m[5] + m[6] * m[6]);
    const s2 = Math.sqrt(m[8] * m[8] + m[9] * m[9] + m[10] * m[10]);
    this._worldBoundingSphere.radius =
      this._localBoundingSphere.radius * Math.max(s0, s1, s2);

    return this._worldBoundingSphere;
  }

  set color(c: Color) {
    this._color = c;
    this._updateMaterialUBO();
  }

  set alpha(a: number) {
    this._alpha = a;
    this._updateMaterialUBO();
  }

  /**
   * Upload an earth texture image.
   */
  setTexture(texture: GPUTexture, sampler?: GPUSampler): void {
    this._texture = texture;
    if (sampler) this._sampler = sampler;
    // Rebuild the bind group immediately so the new texture takes effect
    if (this._opaquePipeline) {
      this.buildOpaqueBindGroup(this._opaquePipeline);
    }
  }

  // ── G-Buffer (opaque) rendering ─────────────────────────────────────────────

  /**
   * Build the per-model bind group for the G-Buffer pass.
   */
  buildOpaqueBindGroup(pipeline: GPURenderPipeline): void {
    this._opaquePipeline = pipeline;
    const device = this._context.device;
    this._opaqueModelBindGroup = device.createBindGroup({
      label:  'Primitive.OpaqueBG',
      layout: pipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: this._modelUniformBuffer } },
        { binding: 1, resource: this._texture.createView() },
        { binding: 2, resource: this._sampler },
      ],
    });
  }

  /**
   * Bind and draw this primitive in the G-Buffer pass.
   * @param pass         - active render pass encoder
   * @param cameraGroup  - bind group 0 (camera uniforms)
   */
  drawOpaque(pass: GPURenderPassEncoder, cameraGroup: GPUBindGroup): void {
    if (!this.show || this._translucent) return;
    if (!this._opaqueModelBindGroup) return;

    this._updateModelUBO();

    pass.setBindGroup(0, cameraGroup);
    pass.setBindGroup(1, this._opaqueModelBindGroup);
    pass.setVertexBuffer(0, this._vertexBuffer);
    pass.setIndexBuffer(this._indexBuffer, 'uint32');
    pass.drawIndexed(this._indexCount);
  }

  /**
   * A1 CSSM：把本图元绘制进级联深度 pass（depth-only，无颜色目标）。
   * 组 1 复用 `_opaqueModelBindGroup` —— 阴影管线必须以
   * `opaquePipeline.getBindGroupLayout(1)`（同一身份）构建管线，本方法
   * 才能跨管线共用该 bind group（先例 buildUnlitPipeline）。
   *
   * @param pass              - 激活中的级联深度 render pass encoder
   * @param shadowCameraGroup - 该级联的光源 view-proj bind group（组 0）
   */
  drawShadow(pass: GPURenderPassEncoder, shadowCameraGroup: GPUBindGroup): void {
    if (!this.show || this._translucent) return;
    if (!this._opaqueModelBindGroup) return;

    this._updateModelUBO();

    pass.setBindGroup(0, shadowCameraGroup);
    pass.setBindGroup(1, this._opaqueModelBindGroup);
    pass.setVertexBuffer(0, this._vertexBuffer);
    pass.setIndexBuffer(this._indexBuffer, 'uint32');
    pass.drawIndexed(this._indexCount);
  }

  // ── Transparent rendering ──────────────────────────────────────────────────

  private _transparentModelBindGroup: GPUBindGroup | null = null;
  private _cameraBindGroup: GPUBindGroup | null = null;

  buildTransparentBindGroups(
    pipeline: GPURenderPipeline,
    cameraUBO: GPUBuffer,
    cameraBGL: GPUBindGroupLayout
  ): void {
    const device = this._context.device;

    this._cameraBindGroup = device.createBindGroup({
      label:  'Primitive.TranspCameraBG',
      layout: cameraBGL,
      entries: [{ binding: 0, resource: { buffer: cameraUBO } }],
    });

    this._transparentModelBindGroup = device.createBindGroup({
      label:  'Primitive.TranspModelBG',
      layout: pipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: this._modelUniformBuffer } },
        { binding: 1, resource: { buffer: this._materialUniformBuffer } },
      ],
    });
  }

  drawTransparent(pass: GPURenderPassEncoder): void {
    if (!this.show || !this._translucent) return;
    if (!this._transparentModelBindGroup || !this._cameraBindGroup) return;

    this._updateModelUBO();

    pass.setBindGroup(0, this._cameraBindGroup);
    pass.setBindGroup(1, this._transparentModelBindGroup);
    pass.setVertexBuffer(0, this._vertexBuffer);
    pass.setIndexBuffer(this._indexBuffer, 'uint32');
    pass.drawIndexed(this._indexCount);
  }

  destroy(): void {
    this._vertexBuffer.destroy();
    this._instanceIdBuffer?.destroy();
    this._indexBuffer.destroy();
    this._modelUniformBuffer.destroy();
    this._materialUniformBuffer.destroy();
    this._texture.destroy();
  }

  // ── Private ────────────────────────────────────────────────────────────────

  /**
   * Compute the local-space bounding sphere from vertex positions (interleaved
   * layout: 8 floats per vertex, position at offset 0). Min/max corners give
   * centre + half-diagonal radius; the radius is inflated by 5% to absorb the
   * sphere-vs-AABB corner slack so culling stays conservative.
   */
  private _computeBoundingSphere(vertices: Float32Array): BoundingSphere {
    const stride = 8; // position(3) + normal(3) + uv(2)

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < vertices.length; i += stride) {
      const x = vertices[i];
      const y = vertices[i + 1];
      const z = vertices[i + 2];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
    }

    if (!Number.isFinite(minX)) {
      // Empty vertex data: degenerate sphere at the origin.
      return { center: new Cartesian3(0, 0, 0), radius: 0 };
    }

    const center = new Cartesian3(
      (minX + maxX) * 0.5,
      (minY + maxY) * 0.5,
      (minZ + maxZ) * 0.5
    );
    const dx = maxX - minX;
    const dy = maxY - minY;
    const dz = maxZ - minZ;
    // 5% margin over the exact half-diagonal (sphere fitting an AABB).
    const radius = 0.5 * Math.sqrt(dx * dx + dy * dy + dz * dz) * 1.05;
    return { center, radius };
  }

  private _updateModelUBO(): void {
    const m = this.modelMatrix.values;

    // Compute the normal matrix = transpose(inverse(upper-left 3x3 of model))
    // For uniform-scaled objects, the normal matrix = model matrix (upper-left 3×3).
    const nm = new Float32Array(16);
    nm[0]  = m[0]; nm[1]  = m[1]; nm[2]  = m[2];  nm[3]  = 0;
    nm[4]  = m[4]; nm[5]  = m[5]; nm[6]  = m[6];  nm[7]  = 0;
    nm[8]  = m[8]; nm[9]  = m[9]; nm[10] = m[10]; nm[11] = 0;
    nm[12] = 0;    nm[13] = 0;    nm[14] = 0;      nm[15] = 1;

    const data = new Float32Array(32); // 128 bytes / 4
    data.set(m, 0);
    data.set(nm, 16);
    this._context.writeBuffer(this._modelUniformBuffer, 0, data);
  }

  private _updateMaterialUBO(): void {
    const data = new Float32Array(8); // 32 bytes / 4
    data[0] = this._color.red;
    data[1] = this._color.green;
    data[2] = this._color.blue;
    data[3] = this._alpha;
    data[4] = 0.5; // roughness
    data[5] = 0.0; // metallic
    this._context.writeBuffer(this._materialUniformBuffer, 0, data);
  }
}

/**
 * PrimitiveCollection - A collection of Primitive objects.
 * Matches Cesium's PrimitiveCollection API.
 */
export class PrimitiveCollection {
  private _primitives: Primitive[] = [];
  show: boolean = true;

  add(primitive: Primitive): Primitive {
    this._primitives.push(primitive);
    return primitive;
  }

  remove(primitive: Primitive): boolean {
    const idx = this._primitives.indexOf(primitive);
    if (idx === -1) return false;
    primitive.destroy();
    this._primitives.splice(idx, 1);
    return true;
  }

  contains(primitive: Primitive): boolean {
    return this._primitives.includes(primitive);
  }

  removeAll(): void {
    for (const p of this._primitives) p.destroy();
    this._primitives = [];
  }

  get length(): number { return this._primitives.length; }

  get(index: number): Primitive { return this._primitives[index]; }

  [Symbol.iterator](): Iterator<Primitive> {
    return this._primitives[Symbol.iterator]();
  }

  get opaque(): Primitive[] {
    return this._primitives.filter(p => p.show && !p.translucent);
  }

  get transparent(): Primitive[] {
    return this._primitives.filter(p => p.show && p.translucent);
  }
}

/**
 * Build a render pipeline for the G-Buffer pass.
 * Shared by all opaque Primitive objects.
 */
export function buildOpaquePipeline(context: Context, depthFormat: GPUTextureFormat): GPURenderPipeline {
  const device = context.device;
  const shader = device.createShaderModule({ code: GBUFFER_SHADER, label: 'GBufferShader' });

  const cameraBGL = device.createBindGroupLayout({
    label: 'GBuffer.BGL0',
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
  });

  const modelBGL = device.createBindGroupLayout({
    label: 'GBuffer.BGL1',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
    ],
  });

  return device.createRenderPipeline({
    label:  'OpaquePipeline',
    layout: device.createPipelineLayout({ bindGroupLayouts: [cameraBGL, modelBGL] }),
    vertex: {
      module:     shader,
      entryPoint: 'vs_main',
      buffers: [{
        arrayStride: 32, // 8 floats × 4 bytes
        attributes: [
          { shaderLocation: 0, offset: 0,  format: 'float32x3' }, // position
          { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
          { shaderLocation: 2, offset: 24, format: 'float32x2' }, // uv
        ],
      }],
    },
    fragment: {
      module:     shader,
      entryPoint: 'fs_main',
      targets: [
        { format: 'rgba8unorm'  }, // albedo
        { format: 'rgba16float' }, // normal
        { format: 'rgba32float' }, // worldPos
      ],
    },
    primitive:    { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { format: depthFormat, depthWriteEnabled: true, depthCompare: 'less' },
  });
}

/**
 * Build the unlit G-Buffer pipeline variant (T3.1) used by emissive effect
 * geometry.  Same vertex layout, same G-Buffer targets and — crucially — the
 * SAME bind group layouts as {@link buildOpaquePipeline}: the layouts are
 * extracted from the opaque pipeline, so per-primitive bind groups built via
 * `Primitive.buildOpaqueBindGroup` stay valid when Scene switches pipelines
 * mid-pass for unlit primitives (WebGPU compares bind group layouts by
 * object identity).
 *
 * @param context        - renderer context
 * @param depthFormat    - depth format of the G-Buffer ('depth32float')
 * @param opaquePipeline - the pipeline built by {@link buildOpaquePipeline}
 */
export function buildUnlitPipeline(
  context: Context,
  depthFormat: GPUTextureFormat,
  opaquePipeline: GPURenderPipeline
): GPURenderPipeline {
  const device = context.device;
  // Same shader module as the lit path; only the fragment entry differs
  // (WGSL has no preprocessor, so the variant is a second entry point).
  const shader = device.createShaderModule({ code: GBUFFER_SHADER, label: 'GBufferShader' });

  // Identity-preserving layout reuse (see doc comment above).
  const cameraBGL = opaquePipeline.getBindGroupLayout(0);
  const modelBGL  = opaquePipeline.getBindGroupLayout(1);

  return device.createRenderPipeline({
    label:  'UnlitPipeline',
    layout: device.createPipelineLayout({ bindGroupLayouts: [cameraBGL, modelBGL] }),
    vertex: {
      module:     shader,
      entryPoint: 'vs_main',
      buffers: [{
        arrayStride: 32, // 8 floats × 4 bytes
        attributes: [
          { shaderLocation: 0, offset: 0,  format: 'float32x3' }, // position
          { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
          { shaderLocation: 2, offset: 24, format: 'float32x2' }, // uv
        ],
      }],
    },
    fragment: {
      module:     shader,
      entryPoint: GBUFFER_UNLIT_ENTRY_POINT,
      targets: [
        { format: 'rgba8unorm'  }, // albedo   — base colour written as-is
        { format: 'rgba16float' }, // normal   — zero vector (unlit marker)
        { format: 'rgba32float' }, // worldPos — kept for picking / depth
      ],
    },
    primitive:    { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { format: depthFormat, depthWriteEnabled: true, depthCompare: 'less' },
  });
}

export default Primitive;
