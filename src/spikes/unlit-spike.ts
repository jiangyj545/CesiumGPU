import { Viewer } from '../core/Viewer';
import { Cartesian3 } from '../math/Cartesian3';
import { Color } from '../math/Color';
import { Camera } from '../scene/Camera';
import { Primitive } from '../scene/Primitive';
import { EllipsoidGeometry } from '../scene/EllipsoidGeometry';
import { Context } from '../renderer/Context';
import type { IPass } from '../scene/passes/IPass';

/**
 * T3.1 验证页: unlit G-Buffer 管线变体 + IPass / EffectManager 地基。
 *
 * 场景 = 地球 + 左侧 unlit 红色球 + 右侧普通 lit 白色球。
 * 预期: unlit 球纯红无明暗（fs_unlit 写零法线 → LightingPass dot(N,N)<0.5
 * 直通 albedo），lit 球有 Blinn-Phong 明暗对比。
 * 同时挂载一个 FrameProbeEffect 验证 Scene.effects (EffectManager) 的
 * IPass 调度链路（composite 之后逐帧 execute，time 为秒）。
 * 结果镜像到 window.__unlitSpike 供 QA 探针读取。
 */

/** console 错误收集（验收条件：无 console 错误）。 */
const consoleErrors: string[] = [];
window.addEventListener('error', (e) => consoleErrors.push(`error: ${e.message}`));
window.addEventListener('unhandledrejection', (e) =>
  consoleErrors.push(`unhandledrejection: ${String(e.reason)}`)
);

/**
 * 最小 IPass 实现验证 EffectManager 调度：不产生渲染 pass，仅计数。
 * 验收点: Scene.render() 在 Transparent composite 之后逐帧调用 execute,
 * 并传入统一时间（秒）。
 */
class FrameProbeEffect implements IPass {
  readonly label = 'FrameProbeEffect';
  show = true;
  frames = 0;
  lastTime = 0;

  /** 幂等：本 effect 无 GPU 资源，重复调用无副作用。 */
  prepare(): void {}

  execute(_encoder: GPUCommandEncoder, _target: GPUTextureView, time: number): void {
    this.frames++;
    this.lastTime = time;
  }

  destroy(): void {}
}

/** 在 (lon, lat, altM) 放一个半径 radiusM 的球（unit-sphere 空间）。 */
function makeSphere(
  context: Context,
  lon: number,
  lat: number,
  altM: number,
  radiusM: number,
  color: Color,
  unlit: boolean
): Primitive {
  const S   = Camera.EARTH_SCALE;
  const p   = Cartesian3.fromDegrees(lon, lat, altM);
  const pos = new Cartesian3(p.x / S, p.y / S, p.z / S);

  const geom = new EllipsoidGeometry({
    radii:           new Cartesian3(6378137.0, 6378137.0, 6356752.3142),
    stackPartitions: 32,
    slicePartitions: 16,
  });
  const { vertices, indices } = geom.createInterleavedBuffer();

  return new Primitive(context, {
    vertices,
    indices,
    position: pos,
    scale:    radiusM / S,
    color,
    translucent: false,
    unlit,
  });
}

function waitFrames(n: number): Promise<void> {
  return new Promise((resolve) => {
    let count = 0;
    const step = (): void => {
      if (++count >= n) resolve();
      else requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}

async function run(): Promise<void> {
  const container = document.getElementById('cesiumContainer');
  const statsEl   = document.getElementById('stats');
  const errorEl   = document.getElementById('error');
  if (!container || !statsEl || !errorEl) throw new Error('unlit-spike: DOM 元素缺失');

  const viewer = new Viewer(container, { powerPreference: 'high-performance' });
  try {
    await viewer.initialize();
  } catch (err) {
    errorEl.style.display = 'block';
    errorEl.textContent = `WebGPU 初始化失败: ${String(err)}`;
    console.error(err);
    return;
  }
  (window as unknown as { __unlitSpike: object }).__unlitSpike = {
    viewer,
    consoleErrors,
  };

  viewer.camera.setView({
    destination: Cartesian3.fromDegrees(116.4, 34.0, 9_000_000),
  });

  viewer.globe.loadImageryFromUrl('/images/world.jpg').catch(() => {
    viewer.globe.generateProceduralTexture();
  });

  // ── 左: unlit 发光红球 / 右: 普通 lit 白球 ─────────────────────────────────
  const unlitSphere = makeSphere(viewer.context, 103, 31, 1_000_000, 380_000, new Color(1.0, 0.05, 0.05, 1.0), true);
  const litSphere   = makeSphere(viewer.context, 129, 31, 1_000_000, 380_000, new Color(0.95, 0.95, 0.95, 1.0), false);
  viewer.scene.addPrimitive(unlitSphere);
  viewer.scene.addPrimitive(litSphere);

  // ── EffectManager 链路探测（IPass 调度 + 统一时间） ─────────────────────────
  const probe = new FrameProbeEffect();
  viewer.scene.effects.add(probe);

  await waitFrames(10);

  // ── HUD 循环 ────────────────────────────────────────────────────────────────
  setInterval(() => {
    statsEl.textContent =
      `unlit 球: ${unlitSphere.unlit ? 'unlit 管线' : 'lit 管线'} | ` +
      `EffectManager: ${viewer.scene.effects.list.length} 个 effect, probe ${probe.frames} 帧 (t=${probe.lastTime.toFixed(2)}s)\n` +
      `console 错误: ${consoleErrors.length}${consoleErrors.length ? ' ← ' + consoleErrors[consoleErrors.length - 1] : ''}`;
  }, 250);
}

run().catch((err) => {
  console.error(err);
  const errorEl = document.getElementById('error');
  if (errorEl) {
    errorEl.style.display = 'block';
    errorEl.textContent = `spike 失败: ${String(err)}`;
  }
});
