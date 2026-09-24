/**
 * DirtTexture - 程序化镜头污渍纹理（P2 LensFlare DirtMask 默认资源）。
 *
 * CPU canvas 生成 256×256 rgba8unorm 上传 GPU（node 测试环境友好：
 * `typeof document === 'undefined'` 时返回 null，Manager 降级为
 * flare 不乘 dirt 并 console.error 提示 —— 错误不静默）。
 * 斑点使用固定种子 LCG —— 同一引擎版本产物确定，便于回归比对。
 */

/** 固定种子线性同余发生器（确定性噪声）。 */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/**
 * 生成默认 dirt 纹理。
 * @returns GPU 纹理；无 DOM 环境（node 测试）返回 null。
 */
export function createProceduralDirtTexture(device: GPUDevice): GPUTexture | null {
  if (typeof document === 'undefined') return null;

  const SIZE = 256;
  const canvas = document.createElement('canvas');
  canvas.width  = SIZE;
  canvas.height = SIZE;
  const ctx2d = canvas.getContext('2d');
  if (!ctx2d) return null;

  // 1. 径向渐变底：中心透明（dirt 在强光中心不遮挡），边缘微亮
  const grad = ctx2d.createRadialGradient(
    SIZE / 2, SIZE / 2, SIZE * 0.1,
    SIZE / 2, SIZE / 2, SIZE * 0.7,
  );
  grad.addColorStop(0.0, 'rgba(0,0,0,0)');
  grad.addColorStop(0.7, 'rgba(28,28,32,0.25)');
  grad.addColorStop(1.0, 'rgba(52,52,58,0.45)');
  ctx2d.fillStyle = grad;
  ctx2d.fillRect(0, 0, SIZE, SIZE);

  // 2. 固定种子斑点（污渍/灰尘颗粒，径向柔边）
  const rand = lcg(0xC351C0DE);
  for (let i = 0; i < 90; i++) {
    const x = rand() * SIZE;
    const y = rand() * SIZE;
    const r = 1 + rand() * 6;
    const a = 0.05 + rand() * 0.30;
    const g2 = ctx2d.createRadialGradient(x, y, 0, x, y, r);
    g2.addColorStop(0, `rgba(255,250,240,${a.toFixed(3)})`);
    g2.addColorStop(1, 'rgba(255,250,240,0)');
    ctx2d.fillStyle = g2;
    ctx2d.beginPath();
    ctx2d.arc(x, y, r, 0, Math.PI * 2);
    ctx2d.fill();
  }

  // 3. 上传 GPU
  const texture = device.createTexture({
    label: 'PostProcess.DirtDefault',
    size:  { width: SIZE, height: SIZE },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture(
    { source: canvas },
    { texture },
    { width: SIZE, height: SIZE },
  );
  return texture;
}
