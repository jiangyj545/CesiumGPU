import { ScreenSpaceEventHandler } from '../core/ScreenSpaceEventHandler';
import type { ScreenSpaceEventType, ScreenSpacePosition } from '../core/ScreenSpaceEventHandler';

const canvas = document.getElementById('target') as HTMLCanvasElement;
const logEl = document.getElementById('log') as HTMLElement;

const lines: string[] = [];
function log(msg: string): void {
  const t = (performance.now() / 1000).toFixed(2);
  lines.unshift(`[${t}s] ${msg}`);
  if (lines.length > 40) lines.pop();
  logEl.textContent = lines.join('\n');
}

const handler = new ScreenSpaceEventHandler(canvas);

// MOUSE_MOVE fires per mouse step — throttle its log to keep the page readable.
let lastMoveLog = 0;

function describe(pos: ScreenSpacePosition): string {
  let detail = `@ (${pos.x.toFixed(0)}, ${pos.y.toFixed(0)})`;
  const wheel = pos as ScreenSpacePosition & { deltaY?: number };
  const pinch = pos as ScreenSpacePosition & { dist?: number; lastDist?: number };
  if (wheel.deltaY !== undefined) detail += ` deltaY=${wheel.deltaY}`;
  if (pinch.dist !== undefined && pinch.lastDist !== undefined) {
    detail += ` dist=${pinch.dist.toFixed(0)} last=${pinch.lastDist.toFixed(0)}`;
  }
  return detail;
}

const types: ScreenSpaceEventType[] = [
  'LEFT_DOWN', 'LEFT_UP', 'LEFT_CLICK', 'LEFT_DOUBLE_CLICK',
  'RIGHT_DOWN', 'RIGHT_UP', 'RIGHT_CLICK',
  'MIDDLE_DOWN', 'MIDDLE_UP',
  'WHEEL', 'PINCH', 'MOUSE_MOVE',
];

for (const type of types) {
  handler.setInputAction((pos) => {
    const detail = describe(pos);
    if (type === 'MOUSE_MOVE') {
      const now = performance.now();
      if (now - lastMoveLog < 150) return;
      lastMoveLog = now;
      log(`MOUSE_MOVE ${detail}`);
    } else {
      log(`${type} ${detail}`);
    }
  }, type);
}

log('ScreenSpaceEventHandler ready — 在画布上操作（左键拖/点、双击、右键拖/点、中键、滚轮、触摸）');
