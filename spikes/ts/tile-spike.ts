/**
 * Spike: tile texture pipeline verification.
 *
 * Validates:
 * 1. Real XYZ tile (OpenStreetMap) upload as a mipmapped rgba8unorm texture
 * 2. Render-based mipmap generation (MipmapGenerator)
 * 3. Upload frame spikes: 200 tiles in one frame vs streamed 10/frame
 * 4. LRU eviction calls texture.destroy() (TileTextureCache)
 *
 * Run: http://localhost:3001/spikes/tile-spike.html
 * Not part of the tsc project (tsconfig includes "src" only) — spike code.
 */
import { MipmapGenerator } from '../../src/renderer/MipmapGenerator';
import { TileTextureCache } from '../../src/imagery/TileTextureCache';

const TILE_SIZE = 256;
const TILE_ZXY  = { z: 10, x: 843, y: 388 }; // Beijing
const TILE_URL  = `https://tile.openstreetmap.org/${TILE_ZXY.z}/${TILE_ZXY.x}/${TILE_ZXY.y}.png`;
const BURST_COUNT     = 200;
const STREAM_FRAMES   = 20;
const STREAM_PER_FRAME = 10;

type TileSource = ImageBitmap | HTMLCanvasElement;

const mipGen = new MipmapGenerator();
let device: GPUDevice;
let displayCtx: GPUCanvasContext;
let displayPipeline: GPURenderPipeline;
let displaySampler: GPUSampler;

/** Display blit shader (fullscreen triangle, samples an explicit view). */
const DISPLAY_SHADER = /* wgsl */ `
struct VSOutput {
  @builtin(position) position : vec4f,
  @location(0) uv             : vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VSOutput {
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  let p = pos[vertexIndex];
  var out : VSOutput;
  out.position = vec4f(p, 0.0, 1.0);
  out.uv = vec2f((p.x + 1.0) * 0.5, 1.0 - (p.y + 1.0) * 0.5);
  return out;
}

@group(0) @binding(0) var srcTexture : texture_2d<f32>;
@group(0) @binding(1) var srcSampler : sampler;

@fragment
fn fs_main(in : VSOutput) -> @location(0) vec4f {
  return textureSample(srcTexture, srcSampler, in.uv);
}
`;

// ── DOM helpers ──────────────────────────────────────────────────────────────

function setText(id: string, text: string): void {
  document.getElementById(id)!.textContent = text;
}

function setVerdict(html: string): void {
  document.getElementById('verdict')!.innerHTML = html;
}

function mark(id: string, ok: boolean): void {
  const el = document.getElementById(id)!;
  el.textContent = ok ? 'PASS' : 'FAIL';
  el.className   = ok ? 'pass' : 'fail';
}

/** Next rAF timestamp. */
function raf(): Promise<number> {
  return new Promise(resolve => requestAnimationFrame(resolve));
}

/** Run `work` inside one rAF callback; resolve with that frame's interval. */
async function measureFrameInterval(work: () => void): Promise<number> {
  await raf(); // let the compositor settle
  const start = await raf();
  return new Promise<number>(resolve => {
    requestAnimationFrame(() => {
      work();
      requestAnimationFrame(end => resolve(end - start));
    });
  });
}

// ── Tile sources ─────────────────────────────────────────────────────────────

async function loadOsmTile(): Promise<TileSource | null> {
  try {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    await new Promise<void>((resolve, reject) => {
      img.onload  = () => resolve();
      img.onerror = () => reject(new Error(`tile fetch failed: ${TILE_URL}`));
      img.src     = TILE_URL;
    });
    return await createImageBitmap(img);
  } catch (err) {
    console.warn('[tile-spike] OSM tile unavailable, using procedural fallback', err);
    return null;
  }
}

/** Deterministic pseudo-random generator (mulberry32). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** High-frequency fake "street map" tile so mip levels are visually obvious. */
function makeProceduralTile(seed: number): HTMLCanvasElement {
  const rand   = mulberry32(seed);
  const canvas = document.createElement('canvas');
  canvas.width  = TILE_SIZE;
  canvas.height = TILE_SIZE;
  const ctx = canvas.getContext('2d')!;

  const hue = Math.floor(rand() * 360);
  ctx.fillStyle = `hsl(${hue}, 30%, ${20 + rand() * 30}%)`;
  ctx.fillRect(0, 0, TILE_SIZE, TILE_SIZE);

  // City blocks
  for (let i = 0; i < 120; i++) {
    const w = 8 + rand() * 40;
    const h = 8 + rand() * 40;
    ctx.fillStyle = `hsl(${(hue + rand() * 60) % 360}, 45%, ${25 + rand() * 45}%)`;
    ctx.fillRect(rand() * TILE_SIZE, rand() * TILE_SIZE, w, h);
  }
  // Roads
  ctx.strokeStyle = 'rgba(255,255,255,0.75)';
  for (let i = 0; i < 10; i++) {
    ctx.lineWidth = 1 + rand() * 2;
    ctx.beginPath();
    if (rand() > 0.5) {
      const y = rand() * TILE_SIZE;
      ctx.moveTo(0, y);
      ctx.lineTo(TILE_SIZE, y + (rand() - 0.5) * 40);
    } else {
      const x = rand() * TILE_SIZE;
      ctx.moveTo(x, 0);
      ctx.lineTo(x + (rand() - 0.5) * 40, TILE_SIZE);
    }
    ctx.stroke();
  }
  return canvas;
}

// ── Upload ───────────────────────────────────────────────────────────────────

/** create texture → copy level 0 → render-based mipmap chain. */
function uploadTile(source: TileSource): GPUTexture {
  const mipLevelCount = MipmapGenerator.requiredMipLevelCount(source.width, source.height);
  const texture = device.createTexture({
    label: 'Tile',
    size:  { width: source.width, height: source.height },
    format: 'rgba8unorm',
    // RENDER_ATTACHMENT | TEXTURE_BINDING required by MipmapGenerator,
    // COPY_DST for copyExternalImageToTexture.
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT,
    mipLevelCount,
  });
  device.queue.copyExternalImageToTexture(
    { source },
    { texture },
    { width: source.width, height: source.height }
  );
  mipGen.generateMips(device, texture, 'rgba8unorm');
  return texture;
}

// ── Display ──────────────────────────────────────────────────────────────────

function initDisplay(canvas: HTMLCanvasElement): void {
  const ctx = canvas.getContext('webgpu') as GPUCanvasContext | null;
  if (!ctx) throw new Error('canvas webgpu context unavailable');
  displayCtx = ctx;
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });

  const module = device.createShaderModule({ code: DISPLAY_SHADER, label: 'TileSpike.Display' });
  displayPipeline = device.createRenderPipeline({
    label: 'TileSpike.DisplayPipeline',
    layout: 'auto',
    vertex:   { module, entryPoint: 'vs_main' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  displaySampler = device.createSampler({
    minFilter:    'linear',
    magFilter:    'linear',
    mipmapFilter: 'linear',
  });
}

/** Draw the full mip chain (level 0..N) left-to-right on the canvas. */
function drawMipChain(texture: GPUTexture): void {
  const levels = texture.mipLevelCount;
  const encoder = device.createCommandEncoder({ label: 'TileSpike.Encoder' });
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view:       displayCtx.getCurrentTexture().createView(),
      loadOp:     'clear',
      clearValue: { r: 0.02, g: 0.04, b: 0.08, a: 1.0 },
      storeOp:    'store',
    }],
  });
  pass.setPipeline(displayPipeline);
  let x = 24;
  const y = 24;
  for (let level = 0; level < levels; level++) {
    const size = Math.max(1, TILE_SIZE >> level);
    const view = texture.createView({
      label:         `TileSpike.ViewL${level}`,
      dimension:     '2d',
      baseMipLevel:  level,
      mipLevelCount: 1,
    });
    pass.setViewport(x, y, size, size, 0, 1);
    pass.setBindGroup(0, device.createBindGroup({
      layout: displayPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: view },
        { binding: 1, resource: displaySampler },
      ],
    }));
    pass.draw(3);
    x += size + 12;
  }
  pass.end();
  device.queue.submit([encoder.finish()]);
}

// ── LRU verification ─────────────────────────────────────────────────────────

interface LruCheck {
  pass: boolean;
  detail: string;
}

/**
 * Capacity 8, insert 20 textures with destroy() spies.
 * Expect: tiles 0..11 evicted + destroyed, size 8, get(tile-0) miss.
 */
function verifyLru(device_: GPUDevice): LruCheck {
  const cache = new TileTextureCache({ capacity: 8 });
  const destroyed = new Set<string>();
  let lruMisses = 0;
  let lruHits = 0;

  for (let i = 0; i < 20; i++) {
    const key = `tile-${i}`;
    const tex = device_.createTexture({
      label: key,
      size:  { width: 4, height: 4 },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });
    // Spy on destroy() to prove the cache destroys evicted textures.
    const realDestroy = tex.destroy.bind(tex);
    (tex as unknown as { destroy: () => void }).destroy = () => {
      destroyed.add(key);
      realDestroy();
    };
    cache.put(key, tex);
  }

  const gotOld = cache.get('tile-0');   // evicted → null
  const gotNew = cache.get('tile-19');  // resident → texture
  if (gotOld === null) lruMisses++;
  if (gotNew !== null) lruHits++;

  const stats = cache.stats;
  const expectedEvicted = Array.from({ length: 12 }, (_, i) => `tile-${i}`);
  const allEvictedDestroyed = expectedEvicted.every(k => destroyed.has(k));
  const noResidentDestroyed = [12, 13, 19].every(i => !destroyed.has(`tile-${i}`));

  const pass =
    stats.evictions === 12 &&
    stats.size === 8 &&
    lruMisses === 1 &&
    lruHits === 1 &&
    gotOld === null &&
    gotNew !== null &&
    allEvictedDestroyed &&
    noResidentDestroyed;

  const detail =
    `evictions=${stats.evictions}/12, size=${stats.size}/8, ` +
    `get(tile-0)=${gotOld === null ? 'miss(null)' : 'ERROR'}, ` +
    `get(tile-19)=${gotNew !== null ? 'hit' : 'ERROR'}, ` +
    `destroyed(0..11)=${allEvictedDestroyed}, 未误毁常驻纹理=${noResidentDestroyed}`;

  cache.clear();
  return { pass, detail };
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function run(): Promise<void> {
  setVerdict('初始化 WebGPU…');

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    setVerdict('<span class="fail">FAIL</span>：requestAdapter 返回 null（无 WebGPU 支持）');
    return;
  }
  device = await adapter.requestDevice();
  device.addEventListener('uncapturederror', (e: Event) => {
    console.error('[tile-spike] uncaptured GPU error', (e as GPUUncapturedErrorEvent).error);
  });

  initDisplay(document.getElementById('tileCanvas') as HTMLCanvasElement);

  // 1. Real OSM tile (fallback: procedural)
  const osmTile = await loadOsmTile();
  const usedFallback = osmTile === null;
  setText('r-osm', usedFallback ? '回退（程序生成）' : '已加载');
  if (usedFallback) {
    setText('d-osm', 'CORS/网络失败 → 程序生成 256×256 模拟瓦片');
  }

  const displaySource: TileSource = osmTile ?? makeProceduralTile(1);
  const displayTexture = uploadTile(displaySource);
  const mipLevels = displayTexture.mipLevelCount;
  setText('r-mips', String(mipLevels));
  mark('r-mips', mipLevels === 9);
  drawMipChain(displayTexture);

  // 2. Single-tile upload cost: cold (first, pipeline compile) + warm average
  let cold = 0;
  let warmSum = 0;
  const WARM_RUNS = 20;
  for (let i = 0; i < WARM_RUNS + 1; i++) {
    const t0 = performance.now();
    const tex = uploadTile(displaySource);
    const dt = performance.now() - t0;
    tex.destroy();
    if (i === 0) cold = dt;
    else warmSum += dt;
  }
  const warmAvg = warmSum / WARM_RUNS;
  setText('r-cold', `${cold.toFixed(2)} ms`);
  setText('r-single', `${warmAvg.toFixed(3)} ms`);
  mark('r-single', warmAvg < 2);

  // 3a. Burst: all 200 tiles inside one frame
  let burstSync = 0;
  const burstInterval = await measureFrameInterval(() => {
    const t0 = performance.now();
    for (let i = 0; i < BURST_COUNT; i++) {
      const tex = uploadTile(makeProceduralTile(1000 + i));
      tex.destroy();
    }
    burstSync = performance.now() - t0;
  });
  setText('r-burst-sync', `${burstSync.toFixed(1)} ms`);
  setText('r-burst-frame', `${burstInterval.toFixed(1)} ms`);
  mark('r-burst-frame', burstInterval < 100);

  // 3b. Streaming: 20 frames × 10 tiles, track max rAF interval
  const bitmaps: TileSource[] = [];
  for (let i = 0; i < STREAM_PER_FRAME; i++) bitmaps.push(makeProceduralTile(2000 + i));
  let last = await raf();
  let maxInterval = 0;
  let tileSum = 0;
  let tileCount = 0;
  for (let f = 0; f < STREAM_FRAMES; f++) {
    const now = await raf();
    maxInterval = Math.max(maxInterval, now - last);
    last = now;
    for (let i = 0; i < STREAM_PER_FRAME; i++) {
      const t0 = performance.now();
      const tex = uploadTile(bitmaps[i]);
      const dt = performance.now() - t0;
      tileSum += dt;
      tileCount++;
      tex.destroy();
    }
  }
  await raf();
  const perTile = tileSum / tileCount;
  setText('r-stream-frame', `${maxInterval.toFixed(1)} ms`);
  setText('r-stream-tile', `${perTile.toFixed(3)} ms`);
  mark('r-stream-frame', maxInterval < 34);

  // 4. LRU eviction + destroy verification
  const lru = verifyLru(device);
  setText('r-lru', lru.pass ? 'PASS' : 'FAIL');
  mark('r-lru', lru.pass);
  setText('d-lru', lru.detail);

  // Verdict
  setVerdict(
    `<b>结论：</b>texture-per-tile 方案可行。单张 256×256 create+upload+mip 链 ≈ ` +
    `<b>${warmAvg.toFixed(3)} ms</b>（冷启动 ${cold.toFixed(2)} ms，含 blit pipeline 编译）。` +
    `200 张单帧连发 rAF 间隔 <b>${burstInterval.toFixed(1)} ms</b>（同步耗时 ${burstSync.toFixed(1)} ms）→ 不可单帧全量上传。` +
    `按 10 张/帧流送时最大 rAF 间隔 <b>${maxInterval.toFixed(1)} ms</b>。` +
    `LRU evict destroy 校验：${lru.pass ? '<span class="pass">正确</span>' : '<span class="fail">异常</span>'}。`
  );
}

run().catch(err => {
  console.error(err);
  setVerdict(`<span class="fail">FAIL</span>：${String(err)}`);
});
