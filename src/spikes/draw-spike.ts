import { Viewer } from '../core/Viewer';
import { DrawHandler } from '../handler/DrawHandler';
import type { DrawType, DrawEndEvent } from '../handler/DrawHandler';
import { XyzImageryProvider } from '../imagery/XyzImageryProvider';
import { Cartesian3 } from '../math/Cartesian3';
import { Cartographic } from '../math/Cartographic';
import { Measure } from '../analysis/Measure';

/**
 * Spike: 绘制交互 DrawHandler（T2.5）。
 * Open at: /spikes/draw-spike.html (需 WebGPU)
 *
 * 验证点：
 * 1. 四种类型（点/线/面/矩形）均可完成绘制并生成正式 Entity；
 * 2. polyline/polygon 移动鼠标时实时预览（半透明绿），完成后变默认蓝；
 * 3. RIGHT_CLICK / 双击结束；多边形至少 3 点；
 * 4. 绘制模式下左键拖拽不旋转相机，滚轮缩放保留；
 * 5. 同一模式下可连续绘制（连画 3 个多边形无报错）。
 */

const MODE_LABELS: Record<DrawType, string> = {
  point: '画点',
  polyline: '画线',
  polygon: '画面',
  rectangle: '画矩形',
};

function toLonLat(p: Cartesian3): string {
  const s = Measure.EARTH_SCALE;
  const c = Cartographic.fromCartesian(new Cartesian3(p.x * s, p.y * s, p.z * s));
  const lon = (c.longitude * 180) / Math.PI;
  const lat = (c.latitude * 180) / Math.PI;
  return `${lon.toFixed(4)}, ${lat.toFixed(4)}`;
}

declare global {
  interface Window {
    /** QA hook (same pattern as tile-lod-spike's `__tileLodSpike`). */
    __drawSpike: { viewer: Viewer; draw: DrawHandler };
  }
}

async function main(): Promise<void> {
  const container = document.getElementById('cesiumContainer');
  const statsEl = document.getElementById('stats');
  const infoEl = document.getElementById('drawInfo');
  if (!container || !statsEl || !infoEl) {
    throw new Error('draw-spike: #cesiumContainer / #stats / #drawInfo not found');
  }
  statsEl.textContent = 'CesiumGPU draw-spike';

  const viewer = new Viewer(container);
  await viewer.initialize();

  // ── Base imagery: AMap (高德) satellite XYZ tiles ──────────────────────────
  viewer.globe.layers.addImageryProvider(new XyzImageryProvider({
    url:          'https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
    subdomains:   ['1', '2', '3', '4'],
    maximumLevel: 18,
    tileSize:     256,
  }));

  viewer.camera.setView({
    destination: Cartesian3.fromDegrees(116.39, 39.9, 1_500_000),
  });

  // ── DrawHandler ─────────────────────────────────────────────────────────────
  const draw = new DrawHandler(viewer);
  let drawnCount = 0;

  const updatePanel = (mode: string, detail: string): void => {
    infoEl.textContent
      = `模式: ${mode}\n实体数: ${viewer.scene.entities.length}\n${detail}`;
  };

  draw.drawStart.addListener((type) => {
    updatePanel(`${MODE_LABELS[type]}（绘制中…）`, '左键取点，右键/双击结束');
    for (const btn of document.querySelectorAll<HTMLButtonElement>('#toolbar button[data-type]')) {
      btn.classList.toggle('active', btn.dataset.type === type);
    }
  });

  draw.drawEnd.addListener(({ positions, entity }: DrawEndEvent) => {
    drawnCount++;
    const lines = positions.map((p, i) => `  #${i + 1}: ${toLonLat(p)}`).join('\n');
    updatePanel(`${entity.name}（可继续绘制）`,
      `第 ${drawnCount} 个实体\n顶点 ${positions.length} 个（经度, 纬度）:\n${lines}`);
    console.log(`[draw-spike] drawEnd #${drawnCount}: ${entity.name}, ${positions.length} vertices`);
  });

  // ── Toolbar ─────────────────────────────────────────────────────────────────
  const stopAll = (): void => {
    draw.stop();
    for (const btn of document.querySelectorAll<HTMLButtonElement>('#toolbar button[data-type]')) {
      btn.classList.remove('active');
    }
    updatePanel('空闲', '点击顶部按钮开始绘制');
  };

  document.querySelectorAll<HTMLButtonElement>('#toolbar button[data-type]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const type = btn.dataset.type as DrawType;
      if (draw.activeType === type) {
        stopAll();
      } else {
        draw.start(type);
      }
    });
  });

  document.getElementById('btn-clear')?.addEventListener('click', () => {
    stopAll();
    viewer.scene.entities.removeAll();
    drawnCount = 0;
    updatePanel('空闲', '已清除全部实体');
  });

  updatePanel('空闲', '点击顶部按钮开始绘制');

  // QA hook (same pattern as tile-lod-spike's `__tileLodSpike`)
  window.__drawSpike = { viewer, draw };
}

main().catch((err: unknown) => {
  console.error('draw-spike failed:', err);
  const el = document.getElementById('stats');
  if (el) el.textContent = `draw-spike failed: ${err instanceof Error ? err.message : String(err)}`;
});
